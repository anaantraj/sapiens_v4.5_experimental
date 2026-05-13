#!/usr/bin/env python3
"""
Generate comparison graphs for Pre-SGO vs Post-SGO JSD metrics.

This script creates:
1. Overall comparison graph (Pre-SGO vs Post-SGO mean JSD)
2. Per-cluster tribe comparison graphs (one graph per cluster showing all tribes)

Note: This script uses the same JSD calculation functions as calculate_jsd_pre_sgo.py
to ensure consistency in how JSD is computed.
"""

import json
import logging
import sys
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from tqdm import tqdm
from scipy.stats import entropy

# Add project root to path to import JSD calculation functions and WandB utilities
BASE_DIR = Path(__file__).parent.parent.parent
sys.path.insert(0, str(BASE_DIR))

from utils.wandb_utils import (
    init_wandb_run,
    get_stage_config,
    use_artifact,
    finish_run,
    load_config,
    log_artifact,
)

# Import JSD calculation functions from calculate_jsd_pre_sgo.py
# We'll define them here to ensure exact same calculation method

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global epsilon for numerical stability (same as calculate_jsd_pre_sgo.py)
EPSILON = 1e-10

# Set style
sns.set_style("whitegrid")
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9
plt.rcParams['legend.fontsize'] = 9


# ============================================================================
# JSD Calculation Functions (same as calculate_jsd_pre_sgo.py)
# ============================================================================

def compute_jsd(P: np.ndarray, Q: np.ndarray, epsilon: float = EPSILON) -> float:
    """
    Compute Jensen-Shannon Divergence between two probability distributions.
    
    JSD(P||Q) = 0.5 * KL(P||M) + 0.5 * KL(Q||M)
    where M = 0.5 * (P + Q)
    
    Args:
        P: First probability distribution (numpy array) - already normalized
        Q: Second probability distribution (numpy array) - already normalized
        epsilon: Small value to avoid log(0)
        
    Returns:
        JSD value (float)
    """
    # Add epsilon to avoid zeros (but keep original distribution shape)
    P = P + epsilon
    Q = Q + epsilon
    
    # Re-normalize after adding epsilon to maintain probability distribution
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


def normalize_distribution(theme_dict: Dict[str, float]) -> Dict[str, float]:
    """
    Normalize a theme probability distribution to sum to 1.0.
    
    Args:
        theme_dict: Dictionary mapping theme names to probabilities
        
    Returns:
        Normalized dictionary
    """
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
    
    Args:
        actual_themes: Ground truth theme distribution
        predicted_themes: Predicted theme distribution
        
    Returns:
        Tuple of (actual_array, predicted_array) with aligned themes, normalized to sum to 1.0
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


def calculate_jsd_for_review(review: Dict[str, Any]) -> Optional[float]:
    """
    Calculate JSD for a single review using the same method as calculate_jsd_pre_sgo.py.
    
    Args:
        review: Review dictionary with 'prediction' and 'actual' keys
        
    Returns:
        JSD value or None if calculation fails
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
            # Convert list to uniform distribution
            if not actual_themes:
                return None
            actual_themes = {theme: 1.0 / len(actual_themes) for theme in actual_themes}
        
        if isinstance(predicted_themes, list):
            # Convert list to uniform distribution
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
        
        # Align distributions (will also normalize again as safety check)
        actual_array, predicted_array = align_distributions(actual_themes, predicted_themes)
        
        if len(actual_array) == 0 or len(predicted_array) == 0:
            return None
        
        # Calculate JSD using the same method
        jsd = compute_jsd(actual_array, predicted_array)
        
        return jsd
        
    except Exception as e:
        logger.warning(f"Error calculating JSD for review: {e}")
        return None


def load_per_review_results(file_path: Path, recalculate_jsd: bool = False) -> List[Dict]:
    """
    Load per-review JSD results from JSON file.
    
    Args:
        file_path: Path to JSON file with per-review results
        recalculate_jsd: If True, recalculate JSD using the same method as calculate_jsd_pre_sgo.py
                        If False, use the pre-calculated JSD values from the file
    
    Returns:
        List of review dictionaries with JSD values
    """
    if not file_path.exists():
        logger.error(f"File not found: {file_path}")
        return []
    
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Handle both list and dict formats
    if isinstance(data, list):
        results = data
    elif isinstance(data, dict):
        results = list(data.values())
    else:
        logger.error(f"Unexpected data format in {file_path}")
        return []
    
    # If recalculating JSD, we need to load from artifact files instead
    # For now, we'll use the pre-calculated values but verify they were calculated correctly
    if recalculate_jsd:
        logger.warning("Recalculation from artifact files not yet implemented. Using pre-calculated JSD values.")
        # TODO: If needed, implement loading from artifact files and recalculating
    
    logger.info(f"Loaded {len(results)} reviews from {file_path.name}")
    
    # Verify all results have JSD values
    results_with_jsd = [r for r in results if r.get('jsd') is not None]
    if len(results_with_jsd) < len(results):
        logger.warning(f"Some reviews missing JSD values: {len(results) - len(results_with_jsd)} out of {len(results)}")
    
    return results_with_jsd


def calculate_overall_mean_jsd(results: List[Dict]) -> float:
    """Calculate overall mean JSD from all individual reviews."""
    jsd_values = [r.get('jsd') for r in results if r.get('jsd') is not None]
    if not jsd_values:
        return 0.0
    return float(np.mean(jsd_values))


def group_by_cluster_and_tribe(results: List[Dict], min_reviews: int = 1) -> Dict[str, Dict[str, Dict]]:
    """
    Group results by cluster and tribe.
    Returns: {cluster_id: {tribe_id: {'jsds': [jsd_values], 'tribe_name': name, 'count': N}}}
    
    Args:
        min_reviews: Minimum number of reviews required to include a tribe (default: 1 for comparison graphs)
    """
    grouped = defaultdict(lambda: defaultdict(lambda: {'jsds': [], 'tribe_name': None, 'count': 0}))
    
    for r in results:
        cluster_id = r.get('cluster_id')
        tribe_id = r.get('tribe_id')
        tribe_name = r.get('tribe_name')
        jsd_val = r.get('jsd')
        
        if cluster_id and tribe_id and jsd_val is not None:
            grouped[cluster_id][tribe_id]['jsds'].append(jsd_val)
            if tribe_name:
                grouped[cluster_id][tribe_id]['tribe_name'] = tribe_name
            grouped[cluster_id][tribe_id]['count'] += 1
    
    # Convert to regular dict and calculate means
    result = {}
    for cluster_id, tribes in grouped.items():
        result[cluster_id] = {}
        for tribe_id, data in tribes.items():
            if len(data['jsds']) >= min_reviews:  # Include tribes with at least min_reviews
                result[cluster_id][tribe_id] = {
                    'mean_jsd': float(np.mean(data['jsds'])),
                    'tribe_name': data['tribe_name'] or tribe_id,
                    'count': data['count'],
                    'jsds': data['jsds']  # Keep individual JSDs for cluster-level averaging
                }
    
    return result


def create_overall_comparison_graph(
    pre_sgo_mean: float,
    post_sgo_mean: float,
    output_path: Path
):
    """Create overall comparison graph (Pre-SGO vs Post-SGO)."""
    fig, ax = plt.subplots(figsize=(8, 6))
    
    categories = ['Pre-SGO\n(Initial Persona)', 'Post-SGO\n(Optimized Persona)']
    values = [pre_sgo_mean, post_sgo_mean]
    colors = ['#E74C3C', '#3498DB']  # Red for Pre-SGO, Blue for Post-SGO (matching user's image)
    
    bars = ax.bar(categories, values, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    
    # Add value labels on bars
    for bar, val in zip(bars, values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.4f}',
                ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    # Calculate and display percentage change
    if pre_sgo_mean > 0:
        percent_change = ((post_sgo_mean - pre_sgo_mean) / pre_sgo_mean) * 100
        change_color = 'green' if percent_change < 0 else 'red'
        change_symbol = '↓' if percent_change < 0 else '↑'
        
        # Add arrow annotation
        ax.annotate('', xy=(1, post_sgo_mean), xytext=(0, pre_sgo_mean),
                   arrowprops=dict(arrowstyle='->', lw=2.5, color=change_color))
        
        # Add percentage change text
        mid_y = (pre_sgo_mean + post_sgo_mean) / 2
        ax.text(0.5, mid_y, f'{abs(percent_change):.1f}% {change_symbol}',
               ha='center', va='center', fontsize=11, fontweight='bold',
               color=change_color, bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8))
    
    ax.set_ylabel('Jensen-Shannon Divergence', fontsize=12, fontweight='bold')
    ax.set_title('(a) JSD: Real vs. Synthetic', fontsize=14, fontweight='bold', pad=20)
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
    pre_sgo_tribes: Dict[str, Dict],
    post_sgo_tribes: Dict[str, Dict],
    output_path: Path
):
    """Create per-cluster tribe comparison graph."""
    # Get all unique tribes that appear in either pre or post
    all_tribe_ids = set(pre_sgo_tribes.keys()) | set(post_sgo_tribes.keys())
    
    if not all_tribe_ids:
        logger.warning(f"No tribes found for {cluster_id}")
        return
    
    # Prepare data for plotting
    tribe_names = []
    pre_jsds = []
    post_jsds = []
    improvements = []  # Track which tribes improved (Post-SGO < Pre-SGO)
    similar = []  # Track tribes with similar values (within 0.01)
    
    for tribe_id in sorted(all_tribe_ids):
        pre_data = pre_sgo_tribes.get(tribe_id, {})
        post_data = post_sgo_tribes.get(tribe_id, {})
        
        pre_mean = pre_data.get('mean_jsd', None)
        post_mean = post_data.get('mean_jsd', None)
        
        # Only include tribes that have both pre and post data for meaningful comparison
        if pre_mean is not None and post_mean is not None:
            tribe_name = post_data.get('tribe_name') or pre_data.get('tribe_name') or tribe_id
            # Truncate long names
            if len(tribe_name) > 40:
                tribe_name = tribe_name[:37] + '...'
            
            tribe_names.append(tribe_name)
            pre_jsds.append(pre_mean)
            post_jsds.append(post_mean)
            
            # Check if improved or similar (only if both values exist)
            if pre_mean > 0 and post_mean > 0:
                if abs(post_mean - pre_mean) < 0.01:
                    similar.append(len(tribe_names) - 1)
                elif post_mean < pre_mean:
                    improvements.append(len(tribe_names) - 1)
    
    if not tribe_names:
        logger.warning(f"No tribes found for {cluster_id}")
        return
    
    # Create figure with two subplots (JSD only, but keeping structure for potential WD addition)
    fig, ax = plt.subplots(figsize=(14, max(8, len(tribe_names) * 0.4)))
    
    y_pos = np.arange(len(tribe_names))
    bar_height = 0.35
    
    # Create horizontal bars (matching user's image: Pre-SGO=Blue, Post-SGO=Red)
    bars_pre = ax.barh(y_pos - bar_height/2, pre_jsds, bar_height, 
                       label='Pre-SGO', color='#3498DB', alpha=0.8, edgecolor='black', linewidth=0.5)
    bars_post = ax.barh(y_pos + bar_height/2, post_jsds, bar_height,
                        label='Post-SGO', color='#E74C3C', alpha=0.8, edgecolor='black', linewidth=0.5)
    
    # Add value labels on bars
    for i, (pre_val, post_val) in enumerate(zip(pre_jsds, post_jsds)):
        # Pre-SGO value
        ax.text(pre_val, y_pos[i] - bar_height/2, f' {pre_val:.3f}',
               va='center', ha='left', fontsize=8)
        # Post-SGO value
        ax.text(post_val, y_pos[i] + bar_height/2, f' {post_val:.3f}',
               va='center', ha='left', fontsize=8)
    
    # Set y-axis labels
    ax.set_yticks(y_pos)
    ax.set_yticklabels(tribe_names, fontsize=9)
    ax.invert_yaxis()  # Top to bottom
    
    # Set x-axis
    max_val = max(max(pre_jsds), max(post_jsds))
    ax.set_xlabel('Jensen-Shannon Divergence', fontsize=11, fontweight='bold')
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


def extract_jsd_from_artifact_files(artifact_path: Path) -> List[Dict]:
    """
    Extract JSD values from artifact files (similar to process_artifact_directory in calculate_jsd_pre_sgo.py).
    
    Args:
        artifact_path: Path to artifact directory
        
    Returns:
        List of review dictionaries with JSD values
    """
    import re
    results = []
    
    # Find all JSON files recursively
    json_files = list(artifact_path.rglob("*.json"))
    logger.info(f"Found {len(json_files)} JSON files in artifact")
    
    for json_file in tqdm(json_files, desc=f"Extracting JSD from {artifact_path.name}"):
        # Extract cluster and micro from path
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
        
        if cluster_id and micro_id:
            tribe_id = f"{cluster_id}/{micro_id}"
        
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except Exception as e:
            logger.debug(f"Error reading file {json_file}: {e}")
            continue
        
        # Skip if data is not a dict or doesn't have user_predictions
        if not isinstance(data, dict) or 'user_predictions' not in data:
            continue
        
        persona_name = data.get('persona_name', None)
        if not persona_name:
            metadata = data.get('metadata', {})
            persona_name = metadata.get('persona_name', None)
        
        user_predictions = data.get('user_predictions', {})
        
        # Extract JSD from each review
        for user_id, reviews in user_predictions.items():
            if not isinstance(reviews, list):
                continue
            
            for review in reviews:
                # Get JSD from metrics if available
                metrics = review.get('metrics', {})
                jsd_value = metrics.get('jsd')
                
                if jsd_value is not None:
                    result = {
                        'jsd': float(jsd_value),
                        'tribe_id': tribe_id,
                        'cluster_id': cluster_id,
                        'micro_id': micro_id,
                        'tribe_name': persona_name,
                        'user_id': user_id,
                        'asin': review.get('asin', ''),
                        'category': review.get('category', 'Unknown')
                    }
                    results.append(result)
    
    logger.info(f"Extracted JSD from {len(results)} reviews")
    return results


def load_results_from_wandb_artifact(run, artifact_name: str, artifact_type: str = "dataset") -> Optional[List[Dict]]:
    """
    Load per-review JSD results from WandB artifact.
    
    Args:
        run: WandB run object
        artifact_name: Name of the artifact (e.g., "pre_sgo_context_all_topics_v6_gpt_5_mini_v4_with_jsd:latest")
        artifact_type: Type of artifact (default: "dataset")
        
    Returns:
        List of review dictionaries with JSD values, or None if loading fails
    """
    try:
        artifact_path = use_artifact(run, artifact_name, artifact_type=artifact_type)
        if not artifact_path or not artifact_path.exists():
            logger.warning(f"Artifact not found: {artifact_name}")
            return None
        
        logger.info(f"Artifact downloaded to: {artifact_path}")
        
        # First, try to find per-review JSON file in the artifact
        json_files = list(artifact_path.rglob("approach1_per_review_*.json"))
        if not json_files:
            json_files = list(artifact_path.rglob("*per_review*.json"))
        
        if json_files:
            # Use the per-review JSON file if found
            json_file = json_files[0]
            logger.info(f"Found per-review results file in artifact: {json_file}")
            return load_per_review_results(json_file)
        else:
            # Extract JSD values from individual review files in the artifact
            logger.info(f"No per-review JSON file found, extracting JSD from artifact files...")
            return extract_jsd_from_artifact_files(artifact_path)
            
    except Exception as e:
        logger.error(f"Error loading artifact {artifact_name}: {e}")
        return None


def main():
    """Main execution function."""
    logger.info("=" * 80)
    logger.info("GENERATING JSD COMPARISON GRAPHS: Pre-SGO vs Post-SGO")
    logger.info("=" * 80)
    
    # Initialize W&B run
    run = init_wandb_run(
        run_name="generate_jsd_comparison_graphs",
        stage="Metrics and analysis",
        config={"description": "Generate Pre-SGO vs Post-SGO JSD comparison graphs"}
    )
    
    if run is None:
        logger.warning("W&B run initialization failed - will try to load from local files")
    
    # Load configuration
    try:
        config = get_stage_config("Metrics and analysis")
        if not config:
            # Fallback: load config file directly
            config_path = BASE_DIR / "Metrics and analysis" / "config.yaml"
            if config_path.exists():
                import yaml
                with open(config_path, 'r', encoding='utf-8') as f:
                    config = yaml.safe_load(f) or {}
            else:
                logger.error("Failed to load configuration")
                config = {}
    except Exception as e:
        logger.warning(f"Error loading config: {e}, using defaults")
        config = {}
    
    input_artifacts = config.get('input_artifacts', {})
    artifact_type = config.get('artifact_type', 'dataset')
    
    # Define paths
    base_dir = Path(__file__).parent.parent.parent
    artifacts_dir = base_dir / "Metrics and analysis" / "artifacts"
    output_dir = artifacts_dir / "jsd_comparison"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Try to load from WandB artifacts first, fallback to local files
    pre_sgo_results = None
    post_sgo_results = None
    
    # Load Pre-SGO results
    logger.info("Loading Pre-SGO results...")
    if run:
        # Try to load from WandB artifact (the _with_jsd version)
        pre_sgo_artifact_name = input_artifacts.get('pre_sgo_context', '')
        if pre_sgo_artifact_name:
            # Try the _with_jsd version first
            artifact_name_with_jsd = f"{pre_sgo_artifact_name.split(':')[0]}_with_jsd:latest"
            pre_sgo_results = load_results_from_wandb_artifact(run, artifact_name_with_jsd, artifact_type)
    
    # Fallback to local file if WandB loading failed
    if not pre_sgo_results:
        pre_sgo_file = artifacts_dir / "approach1_per_review_pre_sgo_context.json"
        if pre_sgo_file.exists():
            logger.info(f"Loading from local file: {pre_sgo_file}")
            pre_sgo_results = load_per_review_results(pre_sgo_file)
        else:
            logger.warning(f"Local file not found: {pre_sgo_file}")
    
    # Load Post-SGO results
    logger.info("Loading Post-SGO results...")
    if run:
        # Try to load from WandB artifact (the _with_jsd version)
        post_sgo_artifact_name = input_artifacts.get('post_sgo_context', '')
        if post_sgo_artifact_name:
            # Try the _with_jsd version first
            artifact_name_with_jsd = f"{post_sgo_artifact_name.split(':')[0]}_with_jsd:latest"
            post_sgo_results = load_results_from_wandb_artifact(run, artifact_name_with_jsd, artifact_type)
    
    # Fallback to local file if WandB loading failed
    if not post_sgo_results:
        post_sgo_file = artifacts_dir / "approach1_per_review_post_sgo_context.json"
        if post_sgo_file.exists():
            logger.info(f"Loading from local file: {post_sgo_file}")
            post_sgo_results = load_per_review_results(post_sgo_file)
        else:
            logger.warning(f"Local file not found: {post_sgo_file}")
    
    if not pre_sgo_results or not post_sgo_results:
        logger.error("Missing required data files. Please run calculate_jsd_pre_sgo.py first or ensure artifacts are available in WandB.")
        if run:
            finish_run(run)
        return
    
    # Calculate overall means (from all individual reviews)
    pre_sgo_mean = calculate_overall_mean_jsd(pre_sgo_results)
    post_sgo_mean = calculate_overall_mean_jsd(post_sgo_results)
    
    logger.info(f"Pre-SGO Overall Mean JSD: {pre_sgo_mean:.6f}")
    logger.info(f"Post-SGO Overall Mean JSD: {post_sgo_mean:.6f}")
    
    # Create overall comparison graph
    overall_graph_path = output_dir / "overall_jsd_comparison.png"
    create_overall_comparison_graph(pre_sgo_mean, post_sgo_mean, overall_graph_path)
    
    # Group by cluster and tribe (use min_reviews=1 for comparison to show all available tribes)
    logger.info("Grouping results by cluster and tribe...")
    pre_sgo_grouped = group_by_cluster_and_tribe(pre_sgo_results, min_reviews=1)
    post_sgo_grouped = group_by_cluster_and_tribe(post_sgo_results, min_reviews=1)
    
    # Get all clusters (from both pre and post)
    all_clusters = sorted(set(pre_sgo_grouped.keys()) | set(post_sgo_grouped.keys()))
    
    logger.info(f"Found {len(all_clusters)} cluster(s): {all_clusters}")
    
    # Create per-cluster comparison graphs
    for cluster_id in all_clusters:
        pre_tribes = pre_sgo_grouped.get(cluster_id, {})
        post_tribes = post_sgo_grouped.get(cluster_id, {})
        
        cluster_output_path = output_dir / f"{cluster_id}_tribe_jsd_comparison.png"
        create_cluster_tribe_comparison_graph(
            cluster_id, pre_tribes, post_tribes, cluster_output_path
        )
    
    logger.info("=" * 80)
    logger.info("COMPARISON GRAPHS GENERATION COMPLETE")
    logger.info(f"Output directory: {output_dir}")
    logger.info("=" * 80)
    
    # Upload outputs to W&B (separate artifact name for this script)
    if run and output_dir.exists():
        artifact_type = config.get("artifact_type", "result")
        artifact_name = config.get("output_artifacts", {}).get("jsd_comparison_graphs", "jsd_comparison_graphs")
        try:
            log_artifact(
                run=run,
                artifact_name=artifact_name,
                artifact_type=artifact_type,
                artifact_path=str(output_dir),
                metadata={"description": "Pre-SGO vs Post-SGO JSD comparison graphs (overall and per-cluster)"},
                aliases=["latest"],
            )
            logger.info("✓ Uploaded JSD comparison graphs to W&B")
        except Exception as e:
            logger.warning(f"Failed to upload artifact to W&B: {e}")
    
    # Finish WandB run
    if run:
        finish_run(run)


if __name__ == "__main__":
    main()

