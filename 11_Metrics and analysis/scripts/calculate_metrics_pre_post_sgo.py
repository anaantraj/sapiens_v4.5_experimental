#!/usr/bin/env python3
"""
Calculate JSD and WD Metrics for Pre-SGO and Post-SGO Predictions
==================================================================

This script:
1. Loads configuration from wandb (config.yaml)
2. Downloads pre_sgo_context and post_sgo_context artifacts from W&B
3. For each review, calculates JSD and/or WD between:
   - actual.predicted_themes (ground truth probability distribution)
   - prediction.predicted_themes (predicted probability distribution)
4. Creates visualization graphs (individual or comparison based on mode)
5. Saves results to output directory

Usage:
    python Metrics and analysis/scripts/calculate_metrics_pre_post_sgo.py
"""

import json
import numpy as np
import sys
import logging
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import matplotlib.patches as mpatches
import seaborn as sns
from scipy.stats import entropy

# Add project root to path
BASE_DIR = Path(__file__).parent.parent.parent
sys.path.insert(0, str(BASE_DIR))

try:
    import ot  # Python Optimal Transport library
    HAS_OT = True
except ImportError:
    ot = None
    HAS_OT = False
    logging.warning("POT (Python Optimal Transport) library not found. Install with: pip install POT")

from utils.wandb_utils import (
    init_wandb_run,
    get_stage_config,
    use_artifact,
    finish_run,
    load_config,
    log_artifact,
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global epsilon for numerical stability
EPSILON = 1e-10


# ============================================================================
# JSD Calculation Functions
# ============================================================================

def compute_jsd(P: np.ndarray, Q: np.ndarray, epsilon: float = EPSILON) -> float:
    """
    Compute Jensen-Shannon Divergence between two probability distributions.
    
    JSD(P||Q) = 0.5 * KL(P||M) + 0.5 * KL(Q||M)
    where M = 0.5 * (P + Q)
    """
    # Add epsilon to avoid zeros
    P = P + epsilon
    Q = Q + epsilon
    
    # Re-normalize after adding epsilon
    P = P / P.sum()
    Q = Q / Q.sum()
    
    # Compute mixture distribution
    M = 0.5 * (P + Q)
    
    # Compute KL divergences
    kl_pm = entropy(P, M, base=2)
    kl_qm = entropy(Q, M, base=2)
    
    # JSD is the average of the two KL divergences
    jsd = 0.5 * kl_pm + 0.5 * kl_qm
    
    return float(jsd)


# ============================================================================
# WD Calculation Functions
# ============================================================================

def calculate_wasserstein_1_distance(
    actual_array: np.ndarray,
    predicted_array: np.ndarray
) -> float:
    """
    Calculate Wasserstein-1 distance (p=1) using optimal transport formulation.
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
    
    # Normalize again as safety check
    P = predicted_array / predicted_sum  # Predicted distribution (source)
    Q = actual_array / actual_sum  # Actual distribution (target)
    
    n = len(P)
    
    # Build cost matrix: C[i,j] = 1 if i != j, else 0
    cost_matrix = np.ones((n, n)) - np.eye(n)
    
    # Use optimal transport to compute Wasserstein-1 distance
    if HAS_OT:
        try:
            wd = ot.emd2(P, Q, cost_matrix, numItermax=1000000)
            return float(wd)
        except Exception as e:
            logger.warning(f"Error in optimal transport calculation: {e}")
            # Fallback to L1 distance
            wd = np.sum(np.abs(P - Q))
            return float(wd)
    else:
        # Fallback: Use L1 distance if POT library is not available
        wd = np.sum(np.abs(P - Q))
        return float(wd)


# ============================================================================
# Common Functions
# ============================================================================

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
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Align two theme distributions to the same set of themes and normalize.
    Uses union method: all unique themes from both distributions.
    """
    # Get all unique themes from both distributions (union method)
    all_themes = sorted(set(actual_themes.keys()) | set(predicted_themes.keys()))
    
    if not all_themes:
        return np.array([]), np.array([])
    
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
    
    return actual_array, predicted_array


def process_review(review: Dict[str, Any], calculate_jsd: bool = True, calculate_wd: bool = True) -> Optional[Dict[str, Any]]:
    """
    Process a single review and calculate JSD and/or WD.
    
    Args:
        review: Review dictionary with 'prediction' and 'actual' keys
        calculate_jsd: If True, calculate JSD
        calculate_wd: If True, calculate WD
        
    Returns:
        Dictionary with metrics and metadata, or None if processing failed
    """
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
        
        # Validate that we have distributions
        if not actual_themes or not predicted_themes:
            return None
        
        # Normalize distributions to ensure they sum to 1.0
        actual_themes = normalize_distribution(actual_themes)
        predicted_themes = normalize_distribution(predicted_themes)
        
        if not actual_themes or not predicted_themes:
            return None
        
        # Align distributions
        actual_array, predicted_array = align_distributions(actual_themes, predicted_themes)
        
        if len(actual_array) == 0 or len(predicted_array) == 0:
            return None
        
        # Calculate metrics
        result = {
            'category': review.get('category', 'Unknown'),
            'asin': review.get('asin', ''),
            'user_id': review.get('user_id', ''),
            'num_actual_themes': len(actual_themes),
            'num_predicted_themes': len(predicted_themes),
            'num_common_themes': len(set(actual_themes.keys()) & set(predicted_themes.keys())),
            'tribe_id': None,  # Will be set during processing
            'cluster_id': None,  # Will be set during processing
            'micro_id': None,  # Will be set during processing
            'actual_themes': actual_themes,  # Store for visualization
            'predicted_themes': predicted_themes  # Store for visualization
        }
        
        if calculate_jsd:
            jsd = compute_jsd(actual_array, predicted_array)
            result['jsd'] = jsd
        
        if calculate_wd:
            wd = calculate_wasserstein_1_distance(actual_array, predicted_array)
            if np.isnan(wd) or not np.isfinite(wd):
                if calculate_jsd:
                    # If JSD is the only metric, we can still return it
                    pass
                else:
                    return None
            result['wd'] = wd
        
        return result
        
    except Exception as e:
        logger.warning(f"Error processing review: {e}")
        return None


def process_artifact_directory(
    artifact_path: Path, 
    max_reviews: Optional[int] = None, 
    add_metrics_to_files: bool = True, 
    filter_cluster: Optional[str] = None,
    calculate_jsd: bool = True,
    calculate_wd: bool = True
) -> Tuple[List[Dict[str, Any]], List[Path]]:
    """
    Process all JSON files in an artifact directory.
    
    Args:
        artifact_path: Path to artifact directory
        max_reviews: Maximum number of reviews to process (None for all)
        add_metrics_to_files: If True, add metrics to each review in the original files
        filter_cluster: Optional cluster ID to filter (e.g., "cluster_1"). If None, processes all clusters.
        calculate_jsd: If True, calculate JSD
        calculate_wd: If True, calculate WD
        
    Returns:
        Tuple of (list of per-review results, list of updated file paths)
    """
    all_results = []
    updated_files = []
    
    # Find all JSON files recursively
    json_files = list(artifact_path.rglob("*.json"))
    
    logger.info(f"Found {len(json_files)} JSON files in artifact")
    
    if filter_cluster:
        json_files = [f for f in json_files if filter_cluster in str(f)]
        logger.info(f"Filtered to {len(json_files)} files for {filter_cluster}")
    
    for json_file in tqdm(json_files, desc="Processing files"):
        # Extract tribe information from file path
        file_path_str = str(json_file)
        cluster_id = None
        micro_id = None
        tribe_id = None
        
        cluster_match = re.search(r'cluster_(\d+)', file_path_str)
        micro_match = re.search(r'micro_(\d+)', file_path_str)
        
        if cluster_match:
            cluster_id = f"cluster_{cluster_match.group(1)}"
        if micro_match:
            micro_id = f"micro_{micro_match.group(1)}"
        
        if filter_cluster and cluster_id != filter_cluster:
            continue
        
        if cluster_id and micro_id:
            tribe_id = f"{cluster_id}/{micro_id}"
        
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except Exception as e:
            logger.warning(f"Error reading file {json_file}: {e}")
            continue
        
        if not isinstance(data, dict):
            continue
        
        if 'user_predictions' not in data:
            continue
        
        persona_name = data.get('persona_name', None)
        if not persona_name:
            metadata = data.get('metadata', {})
            persona_name = metadata.get('persona_name', None)
        
        user_predictions = data.get('user_predictions', {})
        file_updated = False
        
        for user_id, reviews in user_predictions.items():
            if not isinstance(reviews, list):
                continue
            
            for review in reviews:
                prediction = review.get('prediction', {})
                actual = review.get('actual', {})
                
                if not prediction or not actual:
                    continue
                
                actual_themes = actual.get('predicted_themes', {})
                predicted_themes = prediction.get('predicted_themes', {})
                
                if isinstance(actual_themes, list):
                    if not actual_themes:
                        continue
                    actual_themes = {theme: 1.0 / len(actual_themes) for theme in actual_themes}
                
                if isinstance(predicted_themes, list):
                    if not predicted_themes:
                        continue
                    predicted_themes = {theme: 1.0 / len(predicted_themes) for theme in predicted_themes}
                
                if not actual_themes or not predicted_themes:
                    continue
                
                actual_themes_normalized = normalize_distribution(actual_themes)
                predicted_themes_normalized = normalize_distribution(predicted_themes)
                
                if not actual_themes_normalized or not predicted_themes_normalized:
                    continue
                
                if 'user_id' not in review:
                    review['user_id'] = user_id
                
                review['tribe_id'] = tribe_id
                review['cluster_id'] = cluster_id
                review['micro_id'] = micro_id
                review['tribe_name'] = persona_name
                
                review_result = process_review(review, calculate_jsd=calculate_jsd, calculate_wd=calculate_wd)
                if review_result:
                    if add_metrics_to_files:
                        if 'metrics' not in review:
                            review['metrics'] = {}
                        if calculate_jsd and 'jsd' in review_result:
                            review['metrics']['jsd'] = float(review_result['jsd'])
                        if calculate_wd and 'wd' in review_result:
                            review['metrics']['wd'] = float(review_result['wd'])
                        file_updated = True
                    
                    review_result['tribe_id'] = tribe_id
                    review_result['cluster_id'] = cluster_id
                    review_result['micro_id'] = micro_id
                    review_result['tribe_name'] = persona_name
                    all_results.append(review_result)
        
        if add_metrics_to_files and file_updated:
            try:
                with open(json_file, 'w', encoding='utf-8') as f:
                    json.dump(data, f, indent=2, ensure_ascii=False)
                updated_files.append(json_file)
            except Exception as e:
                logger.warning(f"Failed to update file {json_file}: {e}")
        
        if max_reviews and len(all_results) >= max_reviews:
            all_results = all_results[:max_reviews]
            break
    
    logger.info(f"Processed {len(all_results)} reviews")
    if add_metrics_to_files:
        logger.info(f"Updated {len(updated_files)} files with metrics")
    return all_results, updated_files


# ============================================================================
# Visualization Functions
# ============================================================================

def group_results_by_cluster_and_tribe(
    results: List[Dict[str, Any]], 
    metric_name: str = 'jsd',
    min_reviews: int = 3
) -> Tuple[Dict[str, Dict[str, Dict]], Dict[str, list], Dict[str, str]]:
    """
    Group results by cluster and tribe for a given metric.
    
    Returns:
        Tuple of (cluster_tribe_data, cluster_all_values, tribe_names_map)
        - cluster_tribe_data: {cluster_id: {tribe_id: {'mean': float, 'count': int, 'values': list}}}
        - cluster_all_values: {cluster_id: [all_values]} for direct averaging
        - tribe_names_map: {tribe_id: tribe_name}
    """
    cluster_tribe_data = defaultdict(lambda: defaultdict(lambda: {'values': [], 'count': 0}))
    cluster_all_values = defaultdict(list)
    tribe_names_map = {}
    
    for r in results:
        cluster_id = r.get('cluster_id')
        tribe_id = r.get('tribe_id')
        tribe_name = r.get('tribe_name')
        metric_val = r.get(metric_name)
        
        if cluster_id and tribe_id and metric_val is not None:
            cluster_tribe_data[cluster_id][tribe_id]['values'].append(metric_val)
            cluster_tribe_data[cluster_id][tribe_id]['count'] += 1
            cluster_all_values[cluster_id].append(metric_val)
            if tribe_name:
                tribe_names_map[tribe_id] = tribe_name
    
    # Calculate means and filter by min_reviews
    result = {}
    for cluster_id, tribes in cluster_tribe_data.items():
        result[cluster_id] = {}
        for tribe_id, data in tribes.items():
            if len(data['values']) >= min_reviews:
                result[cluster_id][tribe_id] = {
                    'mean': float(np.mean(data['values'])),
                    'count': data['count'],
                    'values': data['values']
                }
    
    return result, cluster_all_values, tribe_names_map


def create_overall_comparison_graph(
    pre_mean: float,
    post_mean: float,
    metric_name: str,
    metric_label: str,
    output_path: Path
):
    """Create overall comparison graph (Pre-SGO vs Post-SGO)."""
    fig, ax = plt.subplots(figsize=(8, 6))
    
    categories = ['Pre-SGO\n(Initial Persona)', 'Post-SGO\n(Optimized Persona)']
    values = [pre_mean, post_mean]
    colors = ['#E74C3C', '#3498DB']  # Red for Pre-SGO, Blue for Post-SGO
    
    bars = ax.bar(categories, values, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    
    # Add value labels on bars
    for bar, val in zip(bars, values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.4f}',
                ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    # Calculate and display percentage change
    if pre_mean > 0:
        percent_change = ((post_mean - pre_mean) / pre_mean) * 100
        change_color = 'green' if percent_change < 0 else 'red'
        change_symbol = '↓' if percent_change < 0 else '↑'
        
        # Add arrow annotation
        ax.annotate('', xy=(1, post_mean), xytext=(0, pre_mean),
                   arrowprops=dict(arrowstyle='->', lw=2.5, color=change_color))
        
        # Add percentage change text
        mid_y = (pre_mean + post_mean) / 2
        ax.text(0.5, mid_y, f'{abs(percent_change):.1f}% {change_symbol}',
               ha='center', va='center', fontsize=11, fontweight='bold',
               color=change_color, bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8))
    
    ax.set_ylabel(metric_label, fontsize=12, fontweight='bold')
    ax.set_title(f'(a) {metric_name.upper()}: Real vs. Synthetic', fontsize=14, fontweight='bold', pad=20)
    ax.set_ylim(0, max(values) * 1.2)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Add legend
    pre_patch = mpatches.Patch(color='#E74C3C', label='Pre-SGO (Initial Persona)')
    post_patch = mpatches.Patch(color='#3498DB', label='Post-SGO (Optimized Persona)')
    ax.legend(handles=[pre_patch, post_patch], loc='upper right', framealpha=0.9)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved overall comparison graph: {output_path}")


def create_cluster_tribe_comparison_graph(
    cluster_id: str,
    pre_tribes: Dict[str, Dict],
    post_tribes: Dict[str, Dict],
    metric_name: str,
    metric_label: str,
    output_path: Path
):
    """Create per-cluster tribe comparison graph."""
    # Get all unique tribes that appear in either pre or post
    all_tribe_ids = set(pre_tribes.keys()) | set(post_tribes.keys())
    
    if not all_tribe_ids:
        logger.warning(f"No tribes found for {cluster_id}")
        return
    
    # Prepare data for plotting
    tribe_names = []
    pre_vals = []
    post_vals = []
    improvements = []
    similar = []
    
    for tribe_id in sorted(all_tribe_ids):
        pre_data = pre_tribes.get(tribe_id, {})
        post_data = post_tribes.get(tribe_id, {})
        
        pre_mean = pre_data.get('mean', None)
        post_mean = post_data.get('mean', None)
        
        # Only include tribes that have both pre and post data
        if pre_mean is not None and post_mean is not None:
            tribe_name = post_data.get('tribe_name') or pre_data.get('tribe_name') or tribe_id
            if len(tribe_name) > 40:
                tribe_name = tribe_name[:37] + '...'
            
            tribe_names.append(tribe_name)
            pre_vals.append(pre_mean)
            post_vals.append(post_mean)
            
            # Check if improved or similar
            if pre_mean > 0 and post_mean > 0:
                if abs(post_mean - pre_mean) < 0.01:
                    similar.append(len(tribe_names) - 1)
                elif post_mean < pre_mean:
                    improvements.append(len(tribe_names) - 1)
    
    if not tribe_names:
        logger.warning(f"No matching tribes with both pre and post data for {cluster_id}")
        return
    
    # Create figure
    fig, ax = plt.subplots(figsize=(14, max(8, len(tribe_names) * 0.4)))
    
    y_pos = np.arange(len(tribe_names))
    bar_height = 0.35
    
    # Create horizontal bars (Pre-SGO=Blue, Post-SGO=Red)
    bars_pre = ax.barh(y_pos - bar_height/2, pre_vals, bar_height, 
                       label='Pre-SGO', color='#3498DB', alpha=0.8, edgecolor='black', linewidth=0.5)
    bars_post = ax.barh(y_pos + bar_height/2, post_vals, bar_height,
                        label='Post-SGO', color='#E74C3C', alpha=0.8, edgecolor='black', linewidth=0.5)
    
    # Add value labels on bars
    for i, (pre_val, post_val) in enumerate(zip(pre_vals, post_vals)):
        ax.text(pre_val, y_pos[i] - bar_height/2, f' {pre_val:.3f}',
               va='center', ha='left', fontsize=8)
        ax.text(post_val, y_pos[i] + bar_height/2, f' {post_val:.3f}',
               va='center', ha='left', fontsize=8)
    
    # Set y-axis labels
    ax.set_yticks(y_pos)
    ax.set_yticklabels(tribe_names, fontsize=9)
    ax.invert_yaxis()
    
    # Set x-axis
    max_val = max(max(pre_vals), max(post_vals))
    ax.set_xlabel(metric_label, fontsize=11, fontweight='bold')
    ax.set_xlim(0, max_val * 1.15)
    ax.grid(axis='x', alpha=0.3, linestyle='--')
    
    # Title with statistics
    num_improved = len(improvements)
    num_similar = len(similar)
    num_total = len(tribe_names)
    title = f"{cluster_id.replace('_', ' ').title()} - Tribes with improved Topic Alignment ({num_improved} decreased + {num_similar} similar, out of {num_total} total)"
    ax.set_title(title, fontsize=12, fontweight='bold', pad=15)
    
    # Add legend
    ax.legend(loc='lower right', framealpha=0.9, fontsize=10)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved cluster comparison graph: {output_path} ({num_total} tribes)")


def create_individual_visualizations(
    results: List[Dict[str, Any]],
    output_dir: Path,
    prefix: str,
    metric_name: str = 'jsd',
    metric_label: str = 'Jensen-Shannon Divergence'
):
    """
    Create individual visualization graphs for a single artifact.
    Similar to create_visualizations in calculate_jsd_pre_sgo.py but supports both JSD and WD.
    """
    artifact_output_dir = output_dir / prefix
    artifact_output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Saving graphs to: {artifact_output_dir}")
    
    sns.set_style("whitegrid")
    plt.rcParams['figure.dpi'] = 300
    plt.rcParams['savefig.dpi'] = 300
    
    if not results:
        return
    
    metric_values = np.array([r.get(metric_name) for r in results if r.get(metric_name) is not None])
    if len(metric_values) == 0:
        logger.warning(f"No {metric_name} values found in results")
        return
    
    mean_metric = np.mean(metric_values)
    median_metric = np.median(metric_values)
    
    # Group by cluster and tribe
    cluster_tribe_data, cluster_all_values, tribe_names_map = group_results_by_cluster_and_tribe(
        results, metric_name=metric_name, min_reviews=3
    )
    
    # Graph 1: Distribution Histogram
    metric_min, metric_max = metric_values.min(), metric_values.max()
    bins = np.linspace(metric_min, metric_max, 30)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    hist, _ = np.histogram(metric_values, bins=bins)
    hist_percent = (hist / len(metric_values)) * 100
    
    fig, ax = plt.subplots(figsize=(12, 7))
    bars = ax.bar(bin_centers, hist_percent, width=(bins[1]-bins[0])*0.8, 
                 alpha=0.8, color='#4A7FB5', edgecolor='#2B5A82', linewidth=1.2)
    ax.axvline(mean_metric, color='#FF6B6B', linestyle='--', linewidth=2.5, 
              label=f'{metric_name.upper()}(Mean): {mean_metric:.4f}', zorder=5)
    ax.axvline(median_metric, color='#51CF66', linestyle='--', linewidth=2.5, 
              label=f'Median: {median_metric:.4f}', zorder=5)
    ax.set_xlabel(metric_label, fontsize=13, fontweight='bold')
    ax.set_ylabel('Percentage (%)', fontsize=13, fontweight='bold')
    ax.set_title(f'{metric_name.upper()} Distribution: Real vs Synthetic\n{prefix.replace("_", " ").title()} (n={len(metric_values):,} reviews)', 
                 fontsize=14, fontweight='bold', pad=15)
    ax.legend(loc='upper right', fontsize=11, framealpha=0.9)
    ax.grid(True, alpha=0.3, axis='y', linestyle=':', linewidth=0.8)
    plt.tight_layout()
    filename = f"{metric_name}_distribution.png"
    plt.savefig(artifact_output_dir / filename, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    logger.info(f"Saved: {filename}")
    
    # Graph 2: Per-Tribe by Cluster (simplified version)
    if cluster_tribe_data and len(cluster_tribe_data) > 0:
        clusters = sorted(cluster_tribe_data.keys())
        num_clusters = len(clusters)
        
        if num_clusters == 1:
            rows, cols = 1, 1
        else:
            cols = int(np.ceil(np.sqrt(num_clusters)))
            rows = int(np.ceil(num_clusters / cols))
        
        fig = plt.figure(figsize=(cols * 12, rows * 8), facecolor='white')
        gs = GridSpec(rows, cols, figure=fig, hspace=0.8, wspace=0.6)
        colors = plt.cm.viridis(np.linspace(0.2, 0.8, 20))
        
        # Calculate cluster averages from all individual review values
        cluster_avg = {}
        for cluster_id in clusters:
            if cluster_id in cluster_all_values and len(cluster_all_values[cluster_id]) > 0:
                cluster_avg[cluster_id] = np.mean(cluster_all_values[cluster_id])
            else:
                cluster_avg[cluster_id] = 0.0
        
        for idx, cluster_id in enumerate(clusters):
            row = idx // cols
            col = idx % cols
            ax = fig.add_subplot(gs[row, col])
            
            tribes_data = cluster_tribe_data[cluster_id]
            if not tribes_data:
                continue
            
            sorted_tribes = sorted(tribes_data.items(), key=lambda x: x[1]['mean'])
            tribe_ids = [t for t, _ in sorted_tribes]
            tribe_labels = []
            for tribe_id in tribe_ids:
                tribe_name = tribe_names_map.get(tribe_id)
                if tribe_name:
                    if len(tribe_name) > 30:
                        tribe_labels.append(tribe_name[:27] + '...')
                    else:
                        tribe_labels.append(tribe_name)
                else:
                    micro_match = re.search(r'micro_(\d+)', tribe_id)
                    if micro_match:
                        tribe_labels.append(f'Tribe {micro_match.group(1)}')
                    else:
                        tribe_labels.append(f'Tribe {tribe_id.split("/")[-1]}')
            
            metric_vals = [v['mean'] for _, v in sorted_tribes]
            num_tribes = len(tribe_labels)
            bar_colors = [colors[i % len(colors)] for i in range(num_tribes)]
            bar_width = 0.5 if num_tribes > 20 else 0.6 if num_tribes > 15 else 0.7
            
            bars = ax.bar(range(num_tribes), metric_vals, alpha=0.9, 
                         color=bar_colors, edgecolor='white', linewidth=2, width=bar_width)
            
            if metric_vals:
                max_metric = max(metric_vals)
                y_padding = max_metric * 0.05
                for i, (bar, val) in enumerate(zip(bars, metric_vals)):
                    height = bar.get_height()
                    x_pos = bar.get_x() + bar.get_width()/2
                    ax.text(x_pos, height + y_padding, f'{val:.3f}', 
                           ha='center', va='bottom', fontsize=9, fontweight='bold')
            
            ax.set_xticks(range(num_tribes))
            font_size = 8 if num_tribes > 20 else 9 if num_tribes > 15 else 10
            rotation = 85 if num_tribes > 20 else 80 if num_tribes > 15 else 70
            ax.set_xticklabels(tribe_labels, fontsize=font_size, rotation=rotation, ha='right')
            ax.set_xlabel('Tribe', fontsize=14, fontweight='bold', labelpad=25)
            ax.set_ylabel(f'Mean {metric_name.upper()}', fontsize=14, fontweight='bold', labelpad=20)
            if metric_vals:
                ax.set_ylim(0, max(metric_vals) * 1.25)
            
            cluster_num = cluster_id.replace('cluster_', '') if cluster_id else '?'
            avg_metric = cluster_avg.get(cluster_id, 0.0)
            ax.set_title(f'Cluster {cluster_num} ({len(tribe_labels)} tribes, Avg {metric_name.upper()}: {avg_metric:.3f})', 
                        fontsize=15, fontweight='bold', pad=20)
            ax.grid(True, alpha=0.2, axis='x')
            ax.set_facecolor('#fafafa')
        
        plt.suptitle(f'Per-Tribe {metric_name.upper()} by Cluster: Real vs Synthetic\n{prefix.replace("_", " ").title()}', 
                    fontsize=20, fontweight='bold', y=0.98)
        plt.tight_layout(rect=[0, 0, 1, 0.95], pad=4.0)
        filename = f"per_tribe_{metric_name}.png"
        plt.savefig(artifact_output_dir / filename, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        logger.info(f"Saved: {filename}")
    
    # Save statistics
    stats = {
        'approach': 'per_review_then_aggregate',
        'mean': float(mean_metric),
        'median': float(median_metric),
        'std': float(np.std(metric_values)),
        'min': float(np.min(metric_values)),
        'max': float(np.max(metric_values)),
        'count': len(metric_values)
    }
    stats_file = artifact_output_dir / f"{metric_name}_statistics.json"
    with open(stats_file, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2)
    logger.info(f"Saved: {metric_name}_statistics.json")
    
    logger.info(f"{metric_name.upper()} Results Summary - {prefix.replace('_', ' ').title()}")
    logger.info(f"Per-Review {metric_name.upper()}: Mean = {mean_metric:.6f}, Median = {median_metric:.6f}")



def main():
    """Main execution function."""
    logger.info("=" * 80)
    logger.info("METRICS CALCULATION FOR PRE-SGO AND POST-SGO PREDICTIONS")
    logger.info("=" * 80)
    
    # Initialize W&B run
    run = init_wandb_run(
        run_name="calculate_metrics_pre_post_sgo",
        stage="Metrics and analysis",
        config={"description": "Calculate JSD and WD for pre-SGO and post-SGO predictions"}
    )
    
    if run is None:
        logger.warning("W&B run initialization failed - running in local mode")
    
    try:
        # Load configuration
        config = get_stage_config("Metrics and analysis")
        if not config:
            config_path = BASE_DIR / "Metrics and analysis" / "config.yaml"
            if config_path.exists():
                import yaml
                with open(config_path, 'r', encoding='utf-8') as f:
                    config = yaml.safe_load(f) or {}
            else:
                logger.error("Failed to load configuration")
                return
        
        # Get configuration
        input_artifacts = config.get('input_artifacts', {})
        artifact_type = config.get('artifact_type', 'dataset')
        output_config = config.get('output', {})
        jsd_config = config.get('jsd', {})
        wd_config = config.get('wd', {})
        metrics_config = config.get('metrics_calculation', {})
        processing_config = config.get('processing', {})
        
        # Get mode and metrics to calculate
        mode = metrics_config.get('mode', 'both')  # 'single' or 'both'
        calculate_jsd = metrics_config.get('calculate_jsd', True)
        calculate_wd = metrics_config.get('calculate_wd', True)
        
        # Set global epsilon
        global EPSILON
        EPSILON = float(jsd_config.get('epsilon', 1e-10))
        
        # Create output directory
        output_dir = BASE_DIR / output_config.get('directory', 'Metrics and analysis/artifacts')
        output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Output directory: {output_dir.absolute()}")
        
        max_reviews = processing_config.get('max_reviews')
        if max_reviews:
            max_reviews = int(max_reviews)
        
        # Process based on mode
        if mode == 'both':
            # Comparison mode: process both pre-SGO and post-SGO
            logger.info(f"\n{'='*80}")
            logger.info(f"MODE: COMPARISON (Pre-SGO vs Post-SGO)")
            logger.info(f"{'='*80}")
            
            pre_sgo_artifact_name = input_artifacts.get('pre_sgo_context')
            post_sgo_artifact_name = input_artifacts.get('post_sgo_context')
            
            pre_results = None
            post_results = None
            
            # Process Pre-SGO
            if pre_sgo_artifact_name:
                logger.info(f"\nProcessing PRE-SGO CONTEXT: {pre_sgo_artifact_name}")
                if run:
                    artifact_path = use_artifact(run, pre_sgo_artifact_name, artifact_type=artifact_type)
                else:
                    artifact_path = BASE_DIR / "07_sgo_training" / "artifacts" / pre_sgo_artifact_name.split(':')[0]
                    if not artifact_path.exists():
                        artifact_path = None
                
                if artifact_path and artifact_path.exists():
                    logger.info(f"Artifact downloaded to: {artifact_path}")
                    pre_results, _ = process_artifact_directory(
                        artifact_path, max_reviews, add_metrics_to_files=False,
                        filter_cluster=None, calculate_jsd=calculate_jsd, calculate_wd=calculate_wd
                    )
                    logger.info(f"Processed {len(pre_results)} reviews from Pre-SGO")
            
            # Process Post-SGO
            if post_sgo_artifact_name:
                logger.info(f"\nProcessing POST-SGO CONTEXT: {post_sgo_artifact_name}")
                if run:
                    artifact_path = use_artifact(run, post_sgo_artifact_name, artifact_type=artifact_type)
                else:
                    artifact_path = BASE_DIR / "07_post_sgo_predictions" / "artifacts" / post_sgo_artifact_name.split(':')[0]
                    if not artifact_path.exists():
                        artifact_path = BASE_DIR / "artifacts" / post_sgo_artifact_name.split(':')[0].split('/')[-1]
                        if not artifact_path.exists():
                            artifact_path = None
                
                if artifact_path and artifact_path.exists():
                    logger.info(f"Artifact downloaded to: {artifact_path}")
                    post_results, _ = process_artifact_directory(
                        artifact_path, max_reviews, add_metrics_to_files=False,
                        filter_cluster=None, calculate_jsd=calculate_jsd, calculate_wd=calculate_wd
                    )
                    logger.info(f"Processed {len(post_results)} reviews from Post-SGO")
            
            # Generate comparison graphs
            if pre_results and post_results:
                comparison_dir = output_dir / "metrics_comparison"
                comparison_dir.mkdir(parents=True, exist_ok=True)
                
                # JSD Comparison
                if calculate_jsd:
                    pre_jsd_values = [r.get('jsd') for r in pre_results if r.get('jsd') is not None]
                    post_jsd_values = [r.get('jsd') for r in post_results if r.get('jsd') is not None]
                    
                    if pre_jsd_values and post_jsd_values:
                        pre_mean_jsd = np.mean(pre_jsd_values)
                        post_mean_jsd = np.mean(post_jsd_values)
                        
                        # Overall comparison
                        create_overall_comparison_graph(
                            pre_mean_jsd, post_mean_jsd, 'jsd', 'Jensen-Shannon Divergence',
                            comparison_dir / "overall_jsd_comparison.png"
                        )
                        
                        # Per-cluster tribe comparison
                        pre_grouped, _, pre_tribe_names = group_results_by_cluster_and_tribe(pre_results, 'jsd', min_reviews=1)
                        post_grouped, _, post_tribe_names = group_results_by_cluster_and_tribe(post_results, 'jsd', min_reviews=1)
                        all_tribe_names = {**pre_tribe_names, **post_tribe_names}
                        
                        all_clusters = sorted(set(pre_grouped.keys()) | set(post_grouped.keys()))
                        for cluster_id in all_clusters:
                            pre_tribes = pre_grouped.get(cluster_id, {})
                            post_tribes = post_grouped.get(cluster_id, {})
                            
                            # Add tribe names to data
                            for tribe_id, data in pre_tribes.items():
                                data['tribe_name'] = all_tribe_names.get(tribe_id)
                            for tribe_id, data in post_tribes.items():
                                data['tribe_name'] = all_tribe_names.get(tribe_id)
                            
                            create_cluster_tribe_comparison_graph(
                                cluster_id, pre_tribes, post_tribes, 'jsd', 'Jensen-Shannon Divergence',
                                comparison_dir / f"{cluster_id}_tribe_jsd_comparison.png"
                            )
                
                # WD Comparison
                if calculate_wd:
                    pre_wd_values = [r.get('wd') for r in pre_results if r.get('wd') is not None]
                    post_wd_values = [r.get('wd') for r in post_results if r.get('wd') is not None]
                    
                    if pre_wd_values and post_wd_values:
                        pre_mean_wd = np.mean(pre_wd_values)
                        post_mean_wd = np.mean(post_wd_values)
                        
                        # Overall comparison
                        create_overall_comparison_graph(
                            pre_mean_wd, post_mean_wd, 'wd', 'Wasserstein Distance',
                            comparison_dir / "overall_wd_comparison.png"
                        )
                        
                        # Per-cluster tribe comparison
                        pre_grouped, _, pre_tribe_names = group_results_by_cluster_and_tribe(pre_results, 'wd', min_reviews=1)
                        post_grouped, _, post_tribe_names = group_results_by_cluster_and_tribe(post_results, 'wd', min_reviews=1)
                        all_tribe_names = {**pre_tribe_names, **post_tribe_names}
                        
                        all_clusters = sorted(set(pre_grouped.keys()) | set(post_grouped.keys()))
                        for cluster_id in all_clusters:
                            pre_tribes = pre_grouped.get(cluster_id, {})
                            post_tribes = post_grouped.get(cluster_id, {})
                            
                            # Add tribe names to data
                            for tribe_id, data in pre_tribes.items():
                                data['tribe_name'] = all_tribe_names.get(tribe_id)
                            for tribe_id, data in post_tribes.items():
                                data['tribe_name'] = all_tribe_names.get(tribe_id)
                            
                            create_cluster_tribe_comparison_graph(
                                cluster_id, pre_tribes, post_tribes, 'wd', 'Wasserstein Distance',
                                comparison_dir / f"{cluster_id}_tribe_wd_comparison.png"
                            )
            
        else:
            # Single mode: process one artifact and generate individual graphs
            logger.info(f"\n{'='*80}")
            logger.info(f"MODE: SINGLE")
            logger.info(f"{'='*80}")
            
            # Process pre_sgo_context
            context_artifact_name = input_artifacts.get('pre_sgo_context')
            if context_artifact_name:
                logger.info(f"\nProcessing PRE-SGO CONTEXT: {context_artifact_name}")
                if run:
                    artifact_path = use_artifact(run, context_artifact_name, artifact_type=artifact_type)
                else:
                    artifact_path = BASE_DIR / "07_sgo_training" / "artifacts" / context_artifact_name.split(':')[0]
                    if not artifact_path.exists():
                        artifact_path = None
                
                if artifact_path and artifact_path.exists():
                    logger.info(f"Artifact downloaded to: {artifact_path}")
                    results, updated_files = process_artifact_directory(
                        artifact_path, max_reviews, add_metrics_to_files=True,
                        filter_cluster=None, calculate_jsd=calculate_jsd, calculate_wd=calculate_wd
                    )
                    
                    # Save per-review results
                    output_file = output_dir / "approach1_per_review_pre_sgo_context.json"
                    with open(output_file, 'w', encoding='utf-8') as f:
                        json.dump(results, f, indent=2)
                    logger.info(f"Saved per-review results to: {output_file}")
                    
                    # Generate individual visualizations
                    if calculate_jsd:
                        create_individual_visualizations(
                            results, output_dir, "pre_sgo_context", 
                            metric_name='jsd', metric_label='Jensen-Shannon Divergence'
                        )
                    if calculate_wd:
                        create_individual_visualizations(
                            results, output_dir, "pre_sgo_context",
                            metric_name='wd', metric_label='Wasserstein Distance'
                        )
            
            # Process post_sgo_context
            post_sgo_artifact_name = input_artifacts.get('post_sgo_context')
            if post_sgo_artifact_name:
                logger.info(f"\nProcessing POST-SGO CONTEXT: {post_sgo_artifact_name}")
                if run:
                    artifact_path = use_artifact(run, post_sgo_artifact_name, artifact_type=artifact_type)
                else:
                    artifact_path = BASE_DIR / "07_post_sgo_predictions" / "artifacts" / post_sgo_artifact_name.split(':')[0]
                    if not artifact_path.exists():
                        artifact_path = BASE_DIR / "artifacts" / post_sgo_artifact_name.split(':')[0].split('/')[-1]
                        if not artifact_path.exists():
                            artifact_path = None
                
                if artifact_path and artifact_path.exists():
                    logger.info(f"Artifact downloaded to: {artifact_path}")
                    results, updated_files = process_artifact_directory(
                        artifact_path, max_reviews, add_metrics_to_files=True,
                        filter_cluster=None, calculate_jsd=calculate_jsd, calculate_wd=calculate_wd
                    )
                    
                    # Save per-review results
                    output_file = output_dir / "approach1_per_review_post_sgo_context.json"
                    with open(output_file, 'w', encoding='utf-8') as f:
                        json.dump(results, f, indent=2)
                    logger.info(f"Saved per-review results to: {output_file}")
                    
                    # Generate individual visualizations
                    if calculate_jsd:
                        create_individual_visualizations(
                            results, output_dir, "post_sgo_context",
                            metric_name='jsd', metric_label='Jensen-Shannon Divergence'
                        )
                    if calculate_wd:
                        create_individual_visualizations(
                            results, output_dir, "post_sgo_context",
                            metric_name='wd', metric_label='Wasserstein Distance'
                        )
        
        # Upload all metrics graphs and results to WandB
        if run and output_dir.exists():
            logger.info("Uploading metrics and graphs to WandB...")
            artifact_type = config.get("artifact_type", "result")
            try:
                log_artifact(
                    run=run,
                    artifact_name=config.get("output_artifacts", {}).get("metrics_pre_post_sgo", "metrics_pre_post_sgo_graphs"),
                    artifact_type=artifact_type,
                    artifact_path=str(output_dir),
                    metadata={"description": "Pre/post SGO JSD and WD metrics, per-review results, and visualizations"},
                    aliases=["latest"],
                )
                logger.info("✓ Uploaded metrics and graphs to WandB")
            except Exception as e:
                logger.warning(f"Failed to upload metrics artifact to WandB: {e}")
        
        logger.info("\n" + "=" * 80)
        logger.info("METRICS CALCULATION COMPLETE")
        logger.info("=" * 80)
        
    except Exception as e:
        logger.error(f"Error in main execution: {e}", exc_info=True)
    finally:
        if run:
            finish_run(run)


if __name__ == "__main__":
    main()















