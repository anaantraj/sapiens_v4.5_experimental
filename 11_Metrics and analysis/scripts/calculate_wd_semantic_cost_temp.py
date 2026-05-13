#!/usr/bin/env python3
"""
Temporary Script: Calculate WD with Semantic Cost Matrix
========================================================

This script calculates Wasserstein Distance using semantic cost matrix (embeddings-based)
for comparison purposes. Results are saved to a temporary folder.

Usage:
    python Metrics and analysis/scripts/calculate_wd_semantic_cost_temp.py
"""

import json
import numpy as np
import sys
import logging
import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict
from tqdm import tqdm
import yaml
import matplotlib.pyplot as plt
import seaborn as sns

try:
    import ot  # Python Optimal Transport library
    HAS_OT = True
except ImportError:
    ot = None
    HAS_OT = False
    logging.error("POT (Python Optimal Transport) library not found. Install with: pip install POT")
    sys.exit(1)

# Add project root to path
BASE_DIR = Path(__file__).parent.parent.parent
sys.path.insert(0, str(BASE_DIR))

try:
    from dotenv import load_dotenv
    load_dotenv(BASE_DIR / ".env", override=True)
except ImportError:
    pass

from utils.openai_client import create_openai_client
from utils.wandb_utils import get_openai_config

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global epsilon for numerical stability
EPSILON = 1e-10

# Temporary output directory
TEMP_OUTPUT_DIR = BASE_DIR / "Metrics and analysis/artifacts/temp_semantic_wd"
TEMP_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Embedding cache to avoid redundant API calls
EMBEDDING_CACHE = {}

# Set style for plots
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
sns.set_context("paper", font_scale=1.2)


def get_topic_embedding(topic_name: str, client: Any, embedding_model: str) -> List[float]:
    """Get embedding for a topic with caching."""
    if topic_name in EMBEDDING_CACHE:
        return EMBEDDING_CACHE[topic_name]
    
    try:
        response = client.embeddings.create(
            input=[str(topic_name)],
            model=embedding_model
        )
        embedding = response.data[0].embedding
        EMBEDDING_CACHE[topic_name] = embedding
        return embedding
    except Exception as e:
        logger.warning(f"Error getting embedding for '{topic_name}': {e}")
        # Return zero vector as fallback
        default_dim = 1536  # text-embedding-3-small dimension
        embedding = [0.0] * default_dim
        EMBEDDING_CACHE[topic_name] = embedding
        return embedding


def build_semantic_cost_matrix(theme_list: List[str], client: Any, embedding_model: str) -> np.ndarray:
    """
    Build semantic cost matrix using embeddings.
    Cost = 1 - cosine_similarity(embedding_i, embedding_j)
    
    Args:
        theme_list: List of theme names
        client: OpenAI-compatible client
        embedding_model: Embedding model to use
        
    Returns:
        Cost matrix of shape (n, n) where n = len(theme_list)
    """
    logger.info(f"Building semantic cost matrix for {len(theme_list)} themes...")
    
    # Get embeddings for all themes
    embeddings = []
    for theme in tqdm(theme_list, desc="Getting embeddings"):
        emb = get_topic_embedding(theme, client, embedding_model)
        embeddings.append(emb)
    
    embeddings = np.array(embeddings)
    
    # Normalize embeddings
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1
    embeddings_norm = embeddings / norms
    
    # Compute cosine similarity
    cosine_sim = np.dot(embeddings_norm, embeddings_norm.T)
    
    # Convert to cost: cost = 1 - similarity (higher similarity = lower cost)
    cost_matrix = 1 - cosine_sim
    cost_matrix = np.maximum(cost_matrix, 0)  # Ensure non-negative
    
    logger.info(f"Cost matrix built. Shape: {cost_matrix.shape}, Min: {cost_matrix.min():.4f}, Max: {cost_matrix.max():.4f}")
    
    return cost_matrix


def normalize_distribution(theme_dict: Dict[str, float]) -> Dict[str, float]:
    """Normalize a theme probability distribution to sum to 1.0."""
    if not theme_dict:
        return {}
    
    total = sum(theme_dict.values())
    if total == 0:
        return {}
    
    return {theme: prob / total for theme, prob in theme_dict.items()}


def align_distributions(
    actual_themes: Dict[str, float],
    predicted_themes: Dict[str, float]
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Align two theme distributions to the same theme set.
    
    Returns:
        actual_array, predicted_array, theme_list
    """
    # Get all unique themes from both distributions
    all_themes = sorted(set(actual_themes.keys()) | set(predicted_themes.keys()))
    
    if not all_themes:
        return np.array([]), np.array([]), []
    
    # Create aligned arrays
    actual_array = np.array([actual_themes.get(theme, 0.0) for theme in all_themes])
    predicted_array = np.array([predicted_themes.get(theme, 0.0) for theme in all_themes])
    
    # Normalize to ensure they sum to 1.0
    actual_sum = actual_array.sum()
    predicted_sum = predicted_array.sum()
    
    if actual_sum > EPSILON:
        actual_array = actual_array / actual_sum
    else:
        actual_array = np.zeros_like(actual_array)
    
    if predicted_sum > EPSILON:
        predicted_array = predicted_array / predicted_sum
    else:
        predicted_array = np.zeros_like(predicted_array)
    
    return actual_array, predicted_array, all_themes


def calculate_wd_semantic_cost(
    actual_array: np.ndarray,
    predicted_array: np.ndarray,
    theme_list: List[str],
    cost_matrix_cache: Dict[tuple, np.ndarray],
    client: Any
) -> float:
    """
    Calculate Wasserstein-1 distance using semantic cost matrix.
    
    Args:
        actual_array: Q - Ground truth probability distribution
        predicted_array: P - Predicted probability distribution
        theme_list: List of theme names (for cost matrix)
        cost_matrix_cache: Cache for cost matrices (keyed by tuple of sorted themes)
        client: OpenAI-compatible client
        
    Returns:
        Wasserstein-1 distance (float)
    """
    if len(actual_array) == 0 or len(predicted_array) == 0:
        return float('nan')
    
    if len(actual_array) != len(predicted_array):
        return float('nan')
    
    # Ensure they sum to 1.0
    actual_sum = actual_array.sum()
    predicted_sum = predicted_array.sum()
    
    if actual_sum < EPSILON or predicted_sum < EPSILON:
        return float('nan')
    
    # Normalize
    P = predicted_array / predicted_sum
    Q = actual_array / actual_sum
    
    n = len(P)
    
    # Get or build cost matrix for this theme set
    theme_key = tuple(sorted(theme_list))
    if theme_key not in cost_matrix_cache:
        cost_matrix_cache[theme_key] = build_semantic_cost_matrix(theme_list, client)
    
    cost_matrix = cost_matrix_cache[theme_key]
    
    # Ensure cost matrix matches current theme set size
    if cost_matrix.shape[0] != n:
        # Rebuild if size mismatch (shouldn't happen, but safety check)
        cost_matrix = build_semantic_cost_matrix(theme_list, client)
        cost_matrix_cache[theme_key] = cost_matrix
    
    # Use optimal transport to compute Wasserstein-1 distance
    try:
        wd = ot.emd2(P, Q, cost_matrix, numItermax=1000000)
        return float(wd)
    except Exception as e:
        logger.warning(f"Error in optimal transport calculation: {e}")
        return float('nan')


def process_review(
    review: Dict[str, Any],
    cost_matrix_cache: Dict[tuple, np.ndarray],
    client: Any
) -> Optional[Dict[str, Any]]:
    """Process a single review and calculate WD with semantic cost."""
    try:
        prediction = review.get('prediction', {})
        actual = review.get('actual', {})
        
        if not prediction or not actual:
            return None
        
        # Get theme distributions
        actual_themes = actual.get('predicted_themes', {})
        predicted_themes = prediction.get('predicted_themes', {})
        
        # Handle different formats (list vs dict)
        if isinstance(actual_themes, list):
            if not actual_themes:
                return None
            actual_themes = {theme: 1.0 / len(actual_themes) for theme in actual_themes}
        
        if isinstance(predicted_themes, list):
            if not predicted_themes:
                return None
            predicted_themes = {theme: 1.0 / len(predicted_themes) for theme in predicted_themes}
        
        # Validate
        if not actual_themes or not predicted_themes:
            return None
        
        # Normalize
        actual_themes = normalize_distribution(actual_themes)
        predicted_themes = normalize_distribution(predicted_themes)
        
        if not actual_themes or not predicted_themes:
            return None
        
        # Align distributions
        actual_array, predicted_array, theme_list = align_distributions(
            actual_themes, predicted_themes
        )
        
        if len(actual_array) == 0:
            return None
        
        # Calculate WD with semantic cost
        wd = calculate_wd_semantic_cost(
            actual_array, predicted_array, theme_list, cost_matrix_cache, client
        )
        
        if np.isnan(wd) or not np.isfinite(wd):
            return None
        
        return {
            'wd': wd,
            'theme_list': theme_list,
            'num_themes': len(theme_list)
        }
        
    except Exception as e:
        logger.warning(f"Error processing review: {e}")
        return None


def process_json_file(filepath: Path, cost_matrix_cache: Dict[tuple, np.ndarray], client: Any) -> List[Dict[str, Any]]:
    """Process a single JSON file."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        results = []
        user_predictions = data.get('user_predictions', {})
        
        # Extract persona name from metadata
        persona_name = data.get('metadata', {}).get('persona_name', 'Unknown')
        if not persona_name or persona_name == 'Unknown':
            persona_name = data.get('persona_name', 'Unknown')
        
        micro_cluster_id = filepath.stem
        
        for user_id, reviews in user_predictions.items():
            if not isinstance(reviews, list):
                continue
            
            for review in reviews:
                if 'user_id' not in review:
                    review['user_id'] = user_id
                
                result = process_review(review, cost_matrix_cache, client)
                if result:
                    result['user_id'] = user_id
                    result['filepath'] = str(filepath)
                    result['cluster_name'] = filepath.parent.name
                    result['micro_cluster_id'] = micro_cluster_id
                    result['persona_name'] = persona_name
                    results.append(result)
        
        return results
        
    except Exception as e:
        logger.error(f"Error processing file {filepath}: {e}")
        return []


def find_json_files(base_dir: Path) -> List[Path]:
    """Find all JSON files in artifact directories."""
    json_files = []
    
    # Pattern: *micro_*_summary_enhanced_persona_micro_cluster_accuracy.json
    pattern = "*micro_*_summary_enhanced_persona_micro_cluster_accuracy.json"
    
    for json_file in base_dir.rglob(pattern):
        json_files.append(json_file)
    
    return sorted(json_files)


def create_visualizations(
    all_results: List[Dict[str, Any]],
    output_dir: Path,
    context_name: str
):
    """
    Create visualizations for semantic cost WD results.
    
    Args:
        all_results: List of all review results with WD calculations
        output_dir: Output directory for saving plots
        context_name: Name of the context for title
    """
    logger.info("Creating visualizations...")
    
    # Extract WD values
    wd_values = [r['wd'] for r in all_results if not np.isnan(r['wd']) and np.isfinite(r['wd'])]
    
    if not wd_values:
        logger.warning("No valid WD values to plot")
        return
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Calculate overall statistics
    overall_mean = np.mean(wd_values)
    overall_median = np.median(wd_values)
    overall_std = np.std(wd_values)
    
    # ========================================================================
    # 1. OVERALL WD DISTRIBUTION
    # ========================================================================
    fig, ax = plt.subplots(figsize=(14, 8))
    fig.suptitle(f'Overall Wasserstein-1 Distance Distribution (Semantic Cost)\n{context_name}', 
                fontsize=20, fontweight='bold', y=0.98)
    
    # Histogram with KDE
    n, bins, patches = ax.hist(wd_values, bins=50, density=True, alpha=0.6, 
                               color='steelblue', edgecolor='black', linewidth=0.8, 
                               label='Distribution', zorder=1)
    
    # Add KDE
    try:
        from scipy.stats import gaussian_kde
        kde = gaussian_kde(wd_values)
        x_range = np.linspace(min(wd_values), max(wd_values), 200)
        ax.plot(x_range, kde(x_range), 'r-', linewidth=3, label='KDE (Smoothed)', zorder=3)
    except:
        pass
    
    # Add statistics lines
    ax.axvline(overall_mean, color='green', linestyle='--', linewidth=3, 
               label=f'Mean: {overall_mean:.4f}', zorder=4, alpha=0.8)
    ax.axvline(overall_median, color='orange', linestyle='--', linewidth=3, 
               label=f'Median: {overall_median:.4f}', zorder=4, alpha=0.8)
    
    # Add statistics text box
    stats_text = f'Statistics Summary:\n\nMean: {overall_mean:.4f}\nMedian: {overall_median:.4f}\nStd Dev: {overall_std:.4f}\nMin: {np.min(wd_values):.4f}\nMax: {np.max(wd_values):.4f}\nCount: {len(wd_values):,}'
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
            fontsize=11, verticalalignment='top', horizontalalignment='left',
            bbox=dict(boxstyle='round', facecolor='white', edgecolor='black', 
                     linewidth=1.5, alpha=0.95), zorder=5, family='monospace')
    
    ax.set_xlabel('Wasserstein-1 Distance (Semantic Cost)', fontsize=15, fontweight='bold', labelpad=10)
    ax.set_ylabel('Density', fontsize=15, fontweight='bold', labelpad=10)
    ax.legend(loc='upper right', fontsize=11, framealpha=0.95, 
             bbox_to_anchor=(0.98, 0.98), frameon=True, fancybox=True, shadow=True)
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(output_dir / '01_overall_wd_distribution_semantic.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    # ========================================================================
    # 2. CLUSTER-LEVEL WD ANALYSIS
    # ========================================================================
    cluster_wd = defaultdict(list)
    cluster_stats = {}
    
    for r in all_results:
        if not np.isnan(r['wd']) and np.isfinite(r['wd']):
            cluster = r.get('cluster_name', 'unknown')
            cluster_wd[cluster].append(r['wd'])
    
    # Calculate cluster statistics
    for cluster, wds in cluster_wd.items():
        cluster_stats[cluster] = {
            'mean': np.mean(wds),
            'median': np.median(wds),
            'std': np.std(wds),
            'count': len(wds)
        }
    
    clusters = sorted(cluster_stats.keys())
    cluster_means = [cluster_stats[c]['mean'] for c in clusters]
    cluster_stds = [cluster_stats[c]['std'] for c in clusters]
    cluster_counts = [cluster_stats[c]['count'] for c in clusters]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
    fig.suptitle(f'Cluster-Level Wasserstein-1 Distance Analysis (Semantic Cost)\n{context_name}', 
                fontsize=20, fontweight='bold', y=0.98)
    
    # Plot 1: Mean WD by cluster with error bars
    x_pos = np.arange(len(clusters))
    bars = ax1.bar(x_pos, cluster_means, yerr=cluster_stds, capsize=10, alpha=0.8, 
                   color='steelblue', edgecolor='black', linewidth=2, width=0.6)
    
    # Add value labels on bars
    max_val = max([m + s for m, s in zip(cluster_means, cluster_stds)]) if cluster_means else 1.0
    label_offset = max_val * 0.03
    
    for i, (mean, std, count) in enumerate(zip(cluster_means, cluster_stds, cluster_counts)):
        ax1.text(i, mean + std + label_offset, f'{mean:.3f}', ha='center', va='bottom', 
                fontsize=12, fontweight='bold', color='black')
        ax1.text(i, -max_val * 0.08, f'n={count}', ha='center', va='top', 
                fontsize=10, style='italic', color='gray')
    
    ax1.set_xlabel('Cluster', fontsize=14, fontweight='bold', labelpad=10)
    ax1.set_ylabel('Average Wasserstein-1 Distance', fontsize=14, fontweight='bold', labelpad=10)
    ax1.set_title('Average WD by Cluster (with std dev)', fontsize=15, fontweight='bold', pad=15)
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(clusters, fontsize=12)
    ax1.grid(True, alpha=0.3, axis='y', linestyle='--', linewidth=0.8)
    ax1.set_axisbelow(True)
    
    # Overall mean line
    ax1.axhline(overall_mean, color='red', linestyle='--', linewidth=2.5, 
               label=f'Overall Mean: {overall_mean:.4f}', alpha=0.8, zorder=0)
    ax1.legend(fontsize=11, loc='upper left', framealpha=0.95)
    y_max = max([m + s for m, s in zip(cluster_means, cluster_stds)]) * 1.15 if cluster_means else 1.0
    ax1.set_ylim([-max_val * 0.12, y_max])
    
    # Plot 2: Box plot by cluster
    cluster_data = [cluster_wd[c] for c in clusters]
    bp = ax2.boxplot(cluster_data, labels=clusters, patch_artist=True, 
                    showmeans=True, meanline=True, widths=0.6)
    
    for patch in bp['boxes']:
        patch.set_facecolor('lightblue')
        patch.set_alpha(0.7)
        patch.set_edgecolor('black')
        patch.set_linewidth(2)
    
    for element in ['whiskers', 'fliers', 'means', 'medians', 'caps']:
        plt.setp(bp[element], color='black', linewidth=1.5)
    
    ax2.set_xlabel('Cluster', fontsize=14, fontweight='bold', labelpad=10)
    ax2.set_ylabel('Wasserstein-1 Distance', fontsize=14, fontweight='bold', labelpad=10)
    ax2.set_title('WD Distribution by Cluster', fontsize=15, fontweight='bold', pad=15)
    ax2.tick_params(axis='x', labelsize=12)
    ax2.tick_params(axis='y', labelsize=11)
    ax2.grid(True, alpha=0.3, axis='y', linestyle='--', linewidth=0.8)
    ax2.set_axisbelow(True)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(output_dir / '02_cluster_level_wd_semantic.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    # ========================================================================
    # 3. TRIBE (MICRO-CLUSTER) LEVEL WD BY CLUSTER (Using Persona Names)
    # ========================================================================
    cluster_micro_wd = defaultdict(lambda: defaultdict(list))
    cluster_micro_stats = defaultdict(dict)
    cluster_micro_persona = defaultdict(dict)
    
    for r in all_results:
        if not np.isnan(r['wd']) and np.isfinite(r['wd']):
            cluster = r.get('cluster_name', 'unknown')
            micro = r.get('micro_cluster_id', 'unknown')
            persona = r.get('persona_name', 'Unknown')
            cluster_micro_wd[cluster][micro].append(r['wd'])
            cluster_micro_persona[cluster][micro] = persona
    
    # Calculate statistics for each micro-cluster
    for cluster in cluster_micro_wd:
        for micro, wds in cluster_micro_wd[cluster].items():
            cluster_micro_stats[cluster][micro] = {
                'mean': np.mean(wds),
                'median': np.median(wds),
                'std': np.std(wds),
                'count': len(wds),
                'persona_name': cluster_micro_persona[cluster].get(micro, 'Unknown')
            }
    
    clusters_sorted = sorted(cluster_micro_stats.keys())
    
    # Create separate graph for each cluster
    for cluster in clusters_sorted:
        micro_clusters = sorted(cluster_micro_stats[cluster].keys())
        micro_means = [cluster_micro_stats[cluster][m]['mean'] for m in micro_clusters]
        micro_stds = [cluster_micro_stats[cluster][m]['std'] for m in micro_clusters]
        micro_counts = [cluster_micro_stats[cluster][m]['count'] for m in micro_clusters]
        persona_names = [cluster_micro_stats[cluster][m]['persona_name'] for m in micro_clusters]
        
        # Truncate long persona names for display
        display_names = []
        for pname in persona_names:
            if len(pname) > 40:
                display_names.append(pname[:37] + '...')
            else:
                display_names.append(pname)
        
        num_tribes = len(micro_clusters)
        fig_width = max(16, num_tribes * 0.8)
        fig_height = 8
        
        fig, ax = plt.subplots(figsize=(fig_width, fig_height))
        fig.suptitle(f'Average WD by Tribe (Persona) - {cluster.upper()} (Semantic Cost)', 
                    fontsize=18, fontweight='bold', y=0.98)
        
        x_pos = np.arange(len(micro_clusters))
        bars = ax.bar(x_pos, micro_means, yerr=micro_stds, capsize=8, alpha=0.8,
                     color='coral', edgecolor='black', linewidth=2, width=0.6)
        
        max_val = max([m + s for m, s in zip(micro_means, micro_stds)]) if micro_means else 1.0
        label_offset = max_val * 0.04
        
        for i, (mean, std, count) in enumerate(zip(micro_means, micro_stds, micro_counts)):
            ax.text(i, mean + std + label_offset, f'{mean:.3f}', ha='center', va='bottom',
                   fontsize=11, fontweight='bold', color='black')
            ax.text(i, -max_val * 0.08, f'n={count}', ha='center', va='top',
                   fontsize=10, style='italic', color='gray')
        
        cluster_mean = cluster_stats[cluster]['mean']
        ax.axhline(cluster_mean, color='blue', linestyle='--', linewidth=3,
                  label=f'Cluster Avg: {cluster_mean:.4f}', alpha=0.8, zorder=0)
        ax.axhline(overall_mean, color='red', linestyle=':', linewidth=2,
                  label=f'Overall Mean: {overall_mean:.4f}', alpha=0.6, zorder=0)
        
        ax.set_xlabel('Tribe (Persona Name)', fontsize=14, fontweight='bold', labelpad=12)
        ax.set_ylabel('Average Wasserstein-1 Distance', fontsize=14, fontweight='bold', labelpad=12)
        ax.set_title(f'Cluster Average WD: {cluster_mean:.4f} | Number of Tribes: {len(micro_clusters)}', 
                    fontsize=13, fontweight='bold', pad=15)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(display_names, rotation=45, ha='right', fontsize=10)
        ax.legend(fontsize=11, loc='upper right', framealpha=0.95, frameon=True)
        ax.grid(True, alpha=0.3, axis='y', linestyle='--', linewidth=0.8)
        ax.set_axisbelow(True)
        
        y_max = max([m + s for m, s in zip(micro_means, micro_stds)]) * 1.25 if micro_means else 1.0
        ax.set_ylim([-max_val * 0.12, y_max])
        ax.tick_params(axis='x', pad=15)
        
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        filename = f'03_tribe_level_wd_{cluster}_semantic.png'
        plt.savefig(output_dir / filename, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
    
    logger.info(f"Visualizations saved to: {output_dir}")


def main():
    """Main execution."""
    logger.info("=" * 80)
    logger.info("CALCULATING WD WITH SEMANTIC COST MATRIX (TEMPORARY)")
    logger.info("=" * 80)
    
    # Load config
    config_path = BASE_DIR / 'Metrics and analysis/config.yaml'
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    wd_config = config.get('wasserstein_distance', {})
    directories_config = wd_config.get('directories', {})
    artifacts_dir_str = directories_config.get('artifacts_dir', '07_sgo_training/artifacts')
    artifacts_base_dir = BASE_DIR / artifacts_dir_str
    
    # Get context name
    contexts_config = wd_config.get('contexts', {})
    context_name = contexts_config.get('context_name', 'pre_sgo_context_all_topics_v6_gpt_5_mini_v4')
    
    artifact_path = artifacts_base_dir / context_name
    
    if not artifact_path.exists():
        logger.error(f"Artifact directory not found: {artifact_path}")
        return
    
    logger.info(f"Processing artifacts from: {artifact_path}")
    
    # Initialize OpenAI-compatible client
    openai_cfg = get_openai_config()
    openai_api_key = os.environ.get('OPENAI_API_KEY') or openai_cfg.get('api_key')
    if not openai_api_key:
        logger.error("OpenAI API key not found")
        return

    embedding_model = config.get('openai', {}).get('embedding_model')
    if not embedding_model:
        logger.error("openai.embedding_model not found in Metrics and analysis/config.yaml")
        return

    client = create_openai_client(openai_config=openai_cfg, timeout=120.0)
    logger.info("OpenAI-compatible client initialized")
    
    # Find all JSON files
    json_files = find_json_files(artifact_path)
    logger.info(f"Found {len(json_files)} JSON files to process")
    
    # Process files
    all_results = []
    cost_matrix_cache = {}  # Cache cost matrices by theme set
    
    for filepath in tqdm(json_files, desc="Processing files"):
        results = process_json_file(filepath, cost_matrix_cache, client)
        all_results.extend(results)
    
    logger.info(f"Processed {len(all_results)} reviews")
    
    # Calculate statistics
    wd_values = [r['wd'] for r in all_results if not np.isnan(r['wd']) and np.isfinite(r['wd'])]
    
    if not wd_values:
        logger.error("No valid WD values calculated")
        return
    
    mean_wd = np.mean(wd_values)
    median_wd = np.median(wd_values)
    std_wd = np.std(wd_values)
    min_wd = np.min(wd_values)
    max_wd = np.max(wd_values)
    
    logger.info("=" * 80)
    logger.info("RESULTS (SEMANTIC COST MATRIX)")
    logger.info("=" * 80)
    logger.info(f"Total reviews processed: {len(all_results)}")
    logger.info(f"Valid WD values: {len(wd_values)}")
    logger.info(f"Mean WD: {mean_wd:.6f}")
    logger.info(f"Median WD: {median_wd:.6f}")
    logger.info(f"Std Dev WD: {std_wd:.6f}")
    logger.info(f"Min WD: {min_wd:.6f}")
    logger.info(f"Max WD: {max_wd:.6f}")
    logger.info("=" * 80)
    
    # Save results
    output_file = TEMP_OUTPUT_DIR / f"semantic_wd_results_{context_name}.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump({
            'statistics': {
                'total_reviews': len(all_results),
                'valid_wd_count': len(wd_values),
                'mean_wd': float(mean_wd),
                'median_wd': float(median_wd),
                'std_wd': float(std_wd),
                'min_wd': float(min_wd),
                'max_wd': float(max_wd)
            },
            'results': all_results
        }, f, indent=2)
    
    logger.info(f"Results saved to: {output_file}")
    
    # Save summary
    summary_file = TEMP_OUTPUT_DIR / f"semantic_wd_summary_{context_name}.txt"
    with open(summary_file, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("WD WITH SEMANTIC COST MATRIX - SUMMARY\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Context: {context_name}\n")
        f.write(f"Total reviews processed: {len(all_results)}\n")
        f.write(f"Valid WD values: {len(wd_values)}\n\n")
        f.write("Statistics:\n")
        f.write(f"  Mean WD:   {mean_wd:.6f}\n")
        f.write(f"  Median WD: {median_wd:.6f}\n")
        f.write(f"  Std Dev:   {std_wd:.6f}\n")
        f.write(f"  Min WD:    {min_wd:.6f}\n")
        f.write(f"  Max WD:    {max_wd:.6f}\n")
        f.write("\n" + "=" * 80 + "\n")
    
    logger.info(f"Summary saved to: {summary_file}")
    
    # Create visualizations
    create_visualizations(all_results, TEMP_OUTPUT_DIR, context_name)
    
    logger.info(f"\nAll results saved to temporary folder: {TEMP_OUTPUT_DIR}")


if __name__ == "__main__":
    main()

