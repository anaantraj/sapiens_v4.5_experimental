#!/usr/bin/env python3
"""
Calculate WD (Wasserstein Distance) for Pre-SGO and Post-SGO Predictions
========================================================================

This script:
1. Loads configuration from wandb (config.yaml)
2. Downloads pre_sgo_context and post_sgo_context artifacts from W&B
3. For each review, calculates WD between:
   - actual.predicted_themes (ground truth probability distribution)
   - prediction.predicted_themes (predicted probability distribution)
4. Creates visualization graphs
5. Saves results to output directory

Usage:
    python Metrics and analysis/scripts/calculate_wd_pre_post_sgo.py
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
import seaborn as sns

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
    logging.warning("Falling back to L1 distance approximation")

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


def calculate_wasserstein_1_distance(
    actual_array: np.ndarray,
    predicted_array: np.ndarray
) -> float:
    """
    Calculate Wasserstein-1 distance (p=1) using optimal transport formulation.
    
    Formula: W_1(P, Q) = min_γ sum_i sum_j γ_ij * C_ij
    where:
    - P = predicted distribution (source)
    - Q = actual distribution (target)
    - C_ij = cost matrix: C[i,j] = 1 if i != j, else 0 (unit cost)
    - γ = optimal transport plan
    
    Args:
        actual_array: Q - Ground truth probability distribution (normalized, sums to 1.0)
        predicted_array: P - Predicted probability distribution (normalized, sums to 1.0)
        
    Returns:
        Wasserstein-1 distance (float) using optimal transport
    """
    if len(actual_array) == 0 or len(predicted_array) == 0:
        return float('nan')
    
    if len(actual_array) != len(predicted_array):
        return float('nan')
    
    # Ensure they sum to 1.0 (safety check)
    actual_sum = actual_array.sum()
    predicted_sum = predicted_array.sum()
    
    if actual_sum < EPSILON or predicted_sum < EPSILON:
        return float('nan')
    
    # Normalize again as safety check
    P = predicted_array / predicted_sum  # Predicted distribution (source)
    Q = actual_array / actual_sum  # Actual distribution (target)
    
    n = len(P)
    
    # Build cost matrix: C[i,j] = 1 if i != j, else 0
    cost_matrix = np.ones((n, n)) - np.eye(n)  # 1 everywhere except diagonal (0)
    
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


def process_review(review: Dict[str, Any], is_baseline: bool = False) -> Optional[Dict[str, Any]]:
    """
    Process a single review and calculate WD.
    
    Args:
        review: Review dictionary with 'prediction' and 'actual' keys
        is_baseline: If True, uses 'topic_probabilities' instead of 'predicted_themes' for actual
        
    Returns:
        Dictionary with WD and metadata, or None if processing failed
    """
    try:
        prediction = review.get('prediction', {})
        actual = review.get('actual', {})
        
        if not prediction or not actual:
            return None
        
        # Get theme distributions
        # Baseline files use 'topic_probabilities' for actual, others use 'predicted_themes'
        # For pre-SGO, prefer topic_probabilities if available (for fair comparison with baseline)
        if is_baseline:
            # Baseline: Try topic_probabilities first, then themes (list), then predicted_themes
            actual_themes = actual.get('topic_probabilities', {})
            if not actual_themes:
                # Fallback to themes list (use directly, will be handled below)
                actual_themes = actual.get('themes', [])
                if not actual_themes:
                    # Last fallback: predicted_themes
                    actual_themes = actual.get('predicted_themes', {})
        else:
            # Pre-SGO: Check for topic_probabilities first (if files were processed)
            # Otherwise use predicted_themes (might be list or dict)
            actual_themes = actual.get('topic_probabilities', actual.get('predicted_themes', {}))
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
        
        # Calculate WD
        wd = calculate_wasserstein_1_distance(actual_array, predicted_array)
        
        if np.isnan(wd) or not np.isfinite(wd):
            return None
        
        # Extract metadata
        result = {
            'wd': wd,
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
        
        return result
        
    except Exception as e:
        logger.warning(f"Error processing review: {e}")
        return None


def process_artifact_directory(artifact_path: Path, max_reviews: Optional[int] = None, add_wd_to_files: bool = True, filter_cluster: Optional[str] = None, calculate_jsd: bool = False, calculate_wd: bool = True, is_baseline: bool = False) -> Tuple[List[Dict[str, Any]], List[Path]]:
    """
    Process all JSON files in an artifact directory.
    
    Args:
        artifact_path: Path to artifact directory
        max_reviews: Maximum number of reviews to process (None for all)
        add_wd_to_files: If True, add WD metric to each review in the original files
        filter_cluster: Optional cluster ID to filter (e.g., "cluster_1"). If None, processes all clusters.
        
    Returns:
        Tuple of (list of per-review WD results, list of updated file paths)
    """
    all_results = []
    updated_files = []
    
    # Track processed reviews to avoid duplicates (same review in multiple files)
    processed_reviews = set()  # (review_text_hash, asin, user_id)
    
    # Find all JSON files recursively, excluding cache files and _with_probs files
    json_files = [f for f in artifact_path.rglob("*.json") 
                  if "_cache" not in str(f) and "/cache/" not in str(f).lower()
                  and "_with_probs" not in str(f)]
    
    logger.info(f"Found {len(json_files)} JSON files in artifact (excluding cache files and _with_probs files)")
    
    # For baseline files, process the main JSON file directly
    if is_baseline:
        # Baseline files are typically a single large JSON file
        baseline_files = [f for f in json_files if 'baseline_predictions' in str(f).lower()]
        if baseline_files:
            json_files = baseline_files[:1]  # Process first baseline file found
            logger.info(f"Processing baseline file: {json_files[0]}")
        elif json_files:
            # If no file with 'baseline' in name, use the first JSON file
            json_files = json_files[:1]
            logger.info(f"Processing first JSON file as baseline: {json_files[0]}")
    
    if filter_cluster and not is_baseline:
        # Filter files to only include the specified cluster (not applicable for baseline)
        json_files = [f for f in json_files if filter_cluster in str(f)]
        logger.info(f"Filtered to {len(json_files)} files for {filter_cluster}")
    
    for json_file in tqdm(json_files, desc="Processing files"):
        # Extract tribe information from file path
        file_path_str = str(json_file)
        cluster_id = None
        micro_id = None
        tribe_id = None
        
        # Extract cluster and micro from path
        cluster_match = re.search(r'cluster_(\d+)', file_path_str)
        micro_match = re.search(r'micro_(\d+)', file_path_str)
        
        if cluster_match:
            cluster_id = f"cluster_{cluster_match.group(1)}"
        if micro_match:
            micro_id = f"micro_{micro_match.group(1)}"
        
        # Skip if cluster filter is set and doesn't match
        if filter_cluster and cluster_id != filter_cluster:
            continue
        
        if cluster_id and micro_id:
            tribe_id = f"{cluster_id}/{micro_id}"
        
        # Read file once
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except Exception as e:
            logger.warning(f"Error reading file {json_file}: {e}")
            continue
        
        # Skip if data is not a dict (e.g., if it's a list)
        if not isinstance(data, dict):
            logger.debug(f"Skipping file {json_file}: data is not a dict (type: {type(data)})")
            continue
        
        # For baseline files, process directly (flat structure)
        if is_baseline:
            # Process baseline format: flat dict with review keys
            for review_key, review in data.items():
                if not isinstance(review, dict):
                    continue
                
                # Extract user_id from key if not present
                if 'user_id' not in review:
                    parts = review_key.rsplit('_review_', 1)
                    if len(parts) == 2:
                        review['user_id'] = parts[0]
                
                # Create unique identifier for this review to avoid duplicates (for baseline)
                review_text = review.get('review_text') or review.get('actual', {}).get('review_text', '')
                asin_val = review.get('asin') or review.get('actual', {}).get('asin', '')
                user_id_val = review.get('user_id', '')
                review_key_uniq = (hash(review_text[:200]) if review_text else 0, asin_val, user_id_val)
                
                # Skip if we've already processed this review
                if review_key_uniq in processed_reviews:
                    continue
                processed_reviews.add(review_key_uniq)
                
                # Calculate per-review WD
                review_result = process_review(review, is_baseline=True)
                if review_result and 'wd' in review_result:
                    wd_value = review_result['wd']
                    
                    # Add WD to the review in the original data structure
                    if add_wd_to_files:
                        if 'metrics' not in review:
                            review['metrics'] = {}
                        review['metrics']['wd'] = float(wd_value)
                    
                    all_results.append(review_result)
            
            # Save updated file if WD was added
            if add_wd_to_files:
                try:
                    with open(json_file, 'w', encoding='utf-8') as f:
                        json.dump(data, f, indent=2, ensure_ascii=False)
                    updated_files.append(json_file)
                    logger.debug(f"Updated baseline file with WD: {json_file}")
                except Exception as e:
                    logger.warning(f"Failed to update file {json_file}: {e}")
            
            if max_reviews and len(all_results) >= max_reviews:
                all_results = all_results[:max_reviews]
                break
            continue
        
        # Skip files that don't have user_predictions (e.g., grand_summary files)
        if 'user_predictions' not in data:
            logger.debug(f"Skipping file {json_file}: no user_predictions field")
            continue
        
        # Extract persona_name (tribe name) from data
        persona_name = data.get('persona_name', None)
        if not persona_name:
            # Try alternative locations
            metadata = data.get('metadata', {})
            persona_name = metadata.get('persona_name', None)
        
        user_predictions = data.get('user_predictions', {})
        file_updated = False
        
        # Process each review
        for user_id, reviews in user_predictions.items():
            if not isinstance(reviews, list):
                continue
            
            for review in reviews:
                prediction = review.get('prediction', {})
                actual = review.get('actual', {})
                
                if not prediction or not actual:
                    continue
                
                # For pre-SGO, prefer topic_probabilities if available (for fair comparison with baseline)
                actual_themes = actual.get('topic_probabilities', actual.get('predicted_themes', {}))
                predicted_themes = prediction.get('predicted_themes', {})
                
                # Handle list format
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
                
                # Normalize distributions before processing
                actual_themes_normalized = normalize_distribution(actual_themes)
                predicted_themes_normalized = normalize_distribution(predicted_themes)
                
                if not actual_themes_normalized or not predicted_themes_normalized:
                    continue
                
                # Add user_id to review if not present
                if 'user_id' not in review:
                    review['user_id'] = user_id
                
                # Create unique identifier for this review to avoid duplicates
                review_text = review.get('review_text') or actual.get('review_text', '')
                asin_val = review.get('asin') or actual.get('asin', '')
                user_id_val = review.get('user_id', user_id)
                review_key = (hash(review_text[:200]) if review_text else 0, asin_val, user_id_val)
                
                # Skip if we've already processed this review
                if review_key in processed_reviews:
                    continue
                processed_reviews.add(review_key)
                
                # Add tribe information to review
                review['tribe_id'] = tribe_id
                review['cluster_id'] = cluster_id
                review['micro_id'] = micro_id
                review['tribe_name'] = persona_name  # Store tribe name
                
                # Calculate per-review WD
                review_result = process_review(review, is_baseline=False)
                if review_result and 'wd' in review_result:
                    wd_value = review_result['wd']
                    
                    # Add WD to the review in the original data structure
                    if add_wd_to_files:
                        if 'metrics' not in review:
                            review['metrics'] = {}
                        review['metrics']['wd'] = float(wd_value)
                        file_updated = True
                    
                    # Add tribe info to result
                    review_result['tribe_id'] = tribe_id
                    review_result['cluster_id'] = cluster_id
                    review_result['micro_id'] = micro_id
                    review_result['tribe_name'] = persona_name  # Store tribe name
                    all_results.append(review_result)
        
        # Save updated file if WD was added
        if add_wd_to_files and file_updated:
            try:
                with open(json_file, 'w', encoding='utf-8') as f:
                    json.dump(data, f, indent=2, ensure_ascii=False)
                updated_files.append(json_file)
                logger.debug(f"Updated file with WD: {json_file}")
            except Exception as e:
                logger.warning(f"Failed to update file {json_file}: {e}")
        
        if max_reviews and len(all_results) >= max_reviews:
            all_results = all_results[:max_reviews]
            break
    
    logger.info(f"Processed {len(all_results)} reviews")
    if add_wd_to_files:
        logger.info(f"Updated {len(updated_files)} files with WD metrics")
    return all_results, updated_files



def create_visualizations(
    results: List[Dict[str, Any]],
    output_dir: Path,
    prefix: str = "pre_sgo_context"
):
    """
    Create visualization graphs for per-review WD analysis.
    
    Args:
        results: List of per-review WD results
        output_dir: Base directory to save graphs (artifacts folder)
        prefix: Prefix for output filenames (e.g., "pre_sgo_context")
    """
    # Create subdirectory for this artifact type
    artifact_output_dir = output_dir / prefix
    artifact_output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Saving graphs to: {artifact_output_dir}")
    
    # Set style
    sns.set_style("whitegrid")
    plt.rcParams['figure.dpi'] = 300
    plt.rcParams['savefig.dpi'] = 300
    
    # ============================================================================
    # APPROACH 1: Per-Review WD (then aggregate statistics)
    # ============================================================================
    if results:
        wd_values = np.array([r['wd'] for r in results])
        mean_wd = np.mean(wd_values)
        median_wd = np.median(wd_values)
        std_wd = np.std(wd_values)
        
        # Calculate per-tribe WD statistics
        tribe_wd = defaultdict(list)
        cluster_tribes = defaultdict(dict)  # cluster_id -> {tribe_id: wd_list}
        cluster_all_wds = defaultdict(list)  # cluster_id -> [all_wds] for direct averaging
        tribe_names_map = {}  # tribe_id -> tribe_name
        
        for r in results:
            tribe_id = r.get('tribe_id')
            cluster_id = r.get('cluster_id')
            tribe_name = r.get('tribe_name')
            wd_val = r.get('wd')
            if tribe_id and cluster_id and wd_val is not None:
                tribe_wd[tribe_id].append(wd_val)
                if tribe_name:
                    tribe_names_map[tribe_id] = tribe_name
                if cluster_id not in cluster_tribes:
                    cluster_tribes[cluster_id] = {}
                if tribe_id not in cluster_tribes[cluster_id]:
                    cluster_tribes[cluster_id][tribe_id] = []
                cluster_tribes[cluster_id][tribe_id].append(wd_val)
                # Also collect all WDs for cluster-level averaging (from individual reviews)
                cluster_all_wds[cluster_id].append(wd_val)
        
        # Calculate mean WD per tribe (average of individual review WDs)
        tribe_mean_wd = {tribe: np.mean(wds) for tribe, wds in tribe_wd.items() if len(wds) >= 3}
        tribe_review_counts = {tribe: len(wds) for tribe, wds in tribe_wd.items()}
        
        # Calculate mean WD per tribe per cluster (average of individual review WDs)
        cluster_tribe_mean_wd = {}
        cluster_tribe_counts = {}
        for cluster_id, tribes_dict in cluster_tribes.items():
            cluster_tribe_mean_wd[cluster_id] = {}
            cluster_tribe_counts[cluster_id] = {}
            for tribe_id, wds in tribes_dict.items():
                if len(wds) >= 3:
                    # Average of individual review WDs in this tribe
                    cluster_tribe_mean_wd[cluster_id][tribe_id] = np.mean(wds)
                    cluster_tribe_counts[cluster_id][tribe_id] = len(wds)
        
        # Aggregate actual and predicted distributions separately for visualization
        # Note: This mixes categories, but shows overall distribution patterns
        aggregated_actual_all = defaultdict(float)
        aggregated_predicted_all = defaultdict(float)
        total_reviews_for_dist = 0
        
        for r in results:
            actual_themes = r.get('actual_themes', {})
            predicted_themes = r.get('predicted_themes', {})
            
            if actual_themes and predicted_themes:
                total_reviews_for_dist += 1
                for theme, prob in actual_themes.items():
                    aggregated_actual_all[theme] += prob
                for theme, prob in predicted_themes.items():
                    aggregated_predicted_all[theme] += prob
        
        # Normalize aggregated distributions
        actual_sum = sum(aggregated_actual_all.values())
        predicted_sum = sum(aggregated_predicted_all.values())
        
        if actual_sum > 0 and predicted_sum > 0:
            actual_dist_all = {theme: prob / actual_sum for theme, prob in aggregated_actual_all.items()}
            predicted_dist_all = {theme: prob / predicted_sum for theme, prob in aggregated_predicted_all.items()}
            
            # Sort by probability for better visualization
            actual_sorted = sorted(actual_dist_all.items(), key=lambda x: x[1], reverse=True)
            predicted_sorted = sorted(predicted_dist_all.items(), key=lambda x: x[1], reverse=True)
            
            # Get top themes for visualization (top 20)
            top_actual = dict(actual_sorted[:20])
            top_predicted = dict(predicted_sorted[:20])
            
            # Get all unique themes from both
            all_themes_vis = sorted(set(top_actual.keys()) | set(top_predicted.keys()))
        
        # ========================================================================
        # Graph 1: WD Distribution Histogram (with percentage)
        # ========================================================================
        # Create bins for WD values
        wd_min, wd_max = wd_values.min(), wd_values.max()
        bins = np.linspace(wd_min, wd_max, 30)
        bin_centers = (bins[:-1] + bins[1:]) / 2
        
        # Calculate histogram
        hist, _ = np.histogram(wd_values, bins=bins)
        hist_percent = (hist / len(wd_values)) * 100  # Convert to percentage
        
        fig, ax = plt.subplots(figsize=(12, 7))
        
        # Create bar chart with percentage
        bars = ax.bar(bin_centers, hist_percent, width=(bins[1]-bins[0])*0.8, 
                     alpha=0.8, color='#4A7FB5', edgecolor='#2B5A82', linewidth=1.2, 
                     label='Real vs Synthetic WD')
        
        # Add mean and median lines
        mean_percent = np.interp(mean_wd, bin_centers, hist_percent)
        ax.axvline(mean_wd, color='#FF6B6B', linestyle='--', linewidth=2.5, 
                  label=f'WD(Mean): {mean_wd:.4f}', zorder=5)
        ax.axvline(median_wd, color='#51CF66', linestyle='--', linewidth=2.5, 
                  label=f'Median: {median_wd:.4f}', zorder=5)
        
        # Add explanation for Y-axis range
        max_percent = np.max(hist_percent)
        y_axis_note = f'Y-axis shows % of reviews per bin\n(Max: {max_percent:.1f}% in any single bin)'
        ax.text(0.02, 0.98, y_axis_note, transform=ax.transAxes,
                fontsize=10, color='dimgray', style='italic',
                verticalalignment='top', horizontalalignment='left',
                bbox=dict(boxstyle='round', facecolor='white', edgecolor='gray', 
                         linewidth=1, alpha=0.7), zorder=10)
        
        ax.set_xlabel('Wasserstein Distance', fontsize=13, fontweight='bold')
        ax.set_ylabel('Percentage (%)', fontsize=13, fontweight='bold')
        ax.set_title(f'WD Distribution: Real vs Synthetic\n{prefix.replace("_", " ").title()} (n={len(wd_values):,} reviews)', 
                     fontsize=14, fontweight='bold', pad=15)
        ax.legend(loc='upper right', fontsize=11, framealpha=0.9)
        ax.grid(True, alpha=0.3, axis='y', linestyle=':', linewidth=0.8)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        plt.tight_layout()
        filename = f"wd_distribution.png"
        plt.savefig(artifact_output_dir / filename, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        logger.info(f"Saved: {filename}")
        
        # ========================================================================
        # Graph 2: Per-Tribe WD Comparison (Grouped by Cluster)
        # ========================================================================
        if cluster_tribe_mean_wd and len(cluster_tribe_mean_wd) > 0:
            # Get all clusters and keep them in numerical order (1, 2, 3, 4, 5...)
            clusters = sorted(cluster_tribe_mean_wd.keys())
            num_clusters = len(clusters)
            
            # Calculate number of tribes per cluster to determine sizing
            tribes_per_cluster = {cid: len(cluster_tribe_mean_wd[cid]) for cid in clusters}
            max_tribes = max(tribes_per_cluster.values()) if tribes_per_cluster else 1
            
            # Use GridSpec for flexible sizing
            # Layout: Cluster 2 gets more space (span 2 columns), others get 1 column each
            if num_clusters == 1:
                rows, cols = 1, 1
            elif 'cluster_2' in clusters:
                # Cluster 2 spans 2 columns, others get 1 column each
                # Layout: Row 1: Cluster 1, Cluster 2 (2 cols), Cluster 3
                #         Row 2: Cluster 4, Cluster 5, ...
                cols = max(4, num_clusters)  # Enough columns for all clusters
                rows = 2
            else:
                cols = int(np.ceil(np.sqrt(num_clusters)))
                rows = int(np.ceil(num_clusters / cols))
            
            # Large figure size with generous spacing to prevent overlapping
            fig = plt.figure(figsize=(cols * 12, rows * 8), facecolor='white')
            gs = GridSpec(rows, cols, figure=fig, hspace=0.8, wspace=0.6)
            
            # Color palette for better visual appeal
            colors = plt.cm.viridis(np.linspace(0.2, 0.8, 20))  # Use viridis colormap
            
            # Calculate average WD per cluster
            cluster_avg_wd = {}
            for cluster_id in clusters:
                # Calculate cluster average directly from all individual review WDs in that cluster
                if cluster_id in cluster_all_wds and len(cluster_all_wds[cluster_id]) > 0:
                    cluster_avg_wd[cluster_id] = np.mean(cluster_all_wds[cluster_id])
                else:
                    # Fallback: if no direct data, use tribe means (shouldn't happen)
                    cluster_tribes_data = cluster_tribe_mean_wd[cluster_id]
                    if cluster_tribes_data:
                        cluster_avg_wd[cluster_id] = np.mean(list(cluster_tribes_data.values()))
                    else:
                        cluster_avg_wd[cluster_id] = 0.0
            
            # Plot each cluster in order
            for idx, cluster_id in enumerate(clusters):
                cluster_tribes_data = cluster_tribe_mean_wd[cluster_id]
                cluster_counts = cluster_tribe_counts[cluster_id]
                
                if not cluster_tribes_data:
                    continue
                
                # Determine subplot position - keep clusters in numerical order
                if 'cluster_2' in clusters and cluster_id == 'cluster_2':
                    # Cluster 2 spans 2 columns in first row for more space
                    ax = fig.add_subplot(gs[0, 1:3])  # Columns 1-2 (after cluster_1)
                elif 'cluster_2' in clusters:
                    # Position other clusters in order
                    cluster_num = int(cluster_id.replace('cluster_', ''))
                    if cluster_num == 1:
                        # Cluster 1 in first row, first column
                        ax = fig.add_subplot(gs[0, 0])
                    elif cluster_num == 3:
                        # Cluster 3 in first row, after cluster_2
                        ax = fig.add_subplot(gs[0, 3])
                    else:
                        # Cluster 4, 5, etc. in second row
                        row = 1
                        col = cluster_num - 4  # cluster_4 -> 0, cluster_5 -> 1, etc.
                        if col < 0:
                            col = 0
                        if col >= cols:
                            col = cols - 1
                        ax = fig.add_subplot(gs[row, col])
                else:
                    # No cluster_2, use regular grid in order
                    row = idx // cols
                    col = idx % cols
                    ax = fig.add_subplot(gs[row, col])
                
                # Sort tribes by WD value (ascending - lowest first)
                sorted_tribes = sorted(cluster_tribes_data.items(), key=lambda x: x[1])
                tribe_ids = [t for t, _ in sorted_tribes]
                
                # Get tribe names (use persona_name if available, otherwise fallback)
                tribe_labels = []
                for tribe_id in tribe_ids:
                    tribe_name = tribe_names_map.get(tribe_id)
                    if tribe_name:
                        # Truncate long names if needed
                        if len(tribe_name) > 30:
                            tribe_labels.append(tribe_name[:27] + '...')
                        else:
                            tribe_labels.append(tribe_name)
                    else:
                        # Fallback to micro number
                        micro_match = re.search(r'micro_(\d+)', tribe_id)
                        if micro_match:
                            tribe_labels.append(f'Tribe {micro_match.group(1)}')
                        else:
                            tribe_labels.append(f'Tribe {tribe_id.split("/")[-1]}')
                
                tribe_wd_vals = [v for _, v in sorted_tribes]
                tribe_counts = [cluster_counts.get(t, 0) for t in tribe_ids]
                
                # Create vertical bar chart with gradient colors
                num_tribes = len(tribe_labels)
                bar_colors = [colors[i % len(colors)] for i in range(num_tribes)]
                
                # Adjust bar width based on number of tribes to prevent overlapping
                if num_tribes > 20:
                    bar_width = 0.5
                elif num_tribes > 15:
                    bar_width = 0.6
                else:
                    bar_width = 0.7
                
                bars = ax.bar(range(num_tribes), tribe_wd_vals, alpha=0.9, 
                              color=bar_colors, edgecolor='white', linewidth=2, width=bar_width)
                
                # Add WD value and review count labels - adjust font size and spacing
                if tribe_wd_vals:
                    max_wd = max(tribe_wd_vals)
                    y_padding = max_wd * 0.05  # More padding to prevent overlap
                    
                    # Adjust font sizes based on number of tribes
                    if num_tribes > 20:
                        wd_fontsize = 9
                        count_fontsize = 8
                    elif num_tribes > 15:
                        wd_fontsize = 10
                        count_fontsize = 9
                    else:
                        wd_fontsize = 11
                        count_fontsize = 10
                    
                    for i, (bar, wd_val, count) in enumerate(zip(bars, tribe_wd_vals, tribe_counts)):
                        height = bar.get_height()
                        x_pos = bar.get_x() + bar.get_width()/2
                        
                        # Add WD value on top of bar with more spacing
                        ax.text(x_pos, height + y_padding, 
                               f'{wd_val:.3f}', ha='center', va='bottom', 
                               fontsize=wd_fontsize, fontweight='bold', color='#333333')
                        
                        # Add count inside bar (if bar is tall enough) - only if there's room
                        if height > max_wd * 0.18:
                            ax.text(x_pos, height * 0.5, 
                                   f'n={count}', ha='center', va='center', 
                                   fontsize=count_fontsize, color='white', fontweight='bold')
                
                # X-axis styling (tribes) - adjust font size and rotation based on number of tribes
                ax.set_xticks(range(num_tribes))
                if num_tribes > 20:
                    font_size = 8
                    rotation = 85
                elif num_tribes > 15:
                    font_size = 9
                    rotation = 80
                else:
                    font_size = 10
                    rotation = 70
                ax.set_xticklabels(tribe_labels, fontsize=font_size, fontweight='600', rotation=rotation, ha='right')
                ax.set_xlabel('Tribe', fontsize=14, fontweight='bold', color='#333333', labelpad=25)
                
                # Y-axis styling (WD)
                ax.set_ylabel('Mean WD', fontsize=14, fontweight='bold', color='#333333', labelpad=20)
                if tribe_wd_vals:
                    ax.set_ylim(0, max(tribe_wd_vals) * 1.25)  # More top margin for labels
                    ax.set_yticks(np.linspace(0, max(tribe_wd_vals), 6))
                    ax.set_yticklabels([f'{y:.2f}' for y in np.linspace(0, max(tribe_wd_vals), 6)], 
                                      fontsize=12)
                
                # Title styling with average WD
                cluster_num = cluster_id.replace('cluster_', '') if cluster_id else '?'
                avg_wd = cluster_avg_wd.get(cluster_id, 0.0)
                # Adjust padding based on number of tribes
                if num_tribes > 20:
                    title_fontsize = 14
                    title_pad = 25
                elif num_tribes > 15:
                    title_fontsize = 15
                    title_pad = 22
                else:
                    title_fontsize = 15
                    title_pad = 20
                ax.set_title(f'Cluster {cluster_num} ({len(tribe_labels)} tribes, Avg WD: {avg_wd:.3f})', 
                            fontsize=title_fontsize, fontweight='bold', pad=title_pad, color='#2c3e50')
                
                # Grid and spines
                ax.grid(True, alpha=0.2, axis='x', linestyle='-', linewidth=0.8, color='#cccccc')
                ax.set_axisbelow(True)
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                ax.spines['left'].set_color('#e0e0e0')
                ax.spines['bottom'].set_color('#e0e0e0')
                ax.spines['left'].set_linewidth(1)
                ax.spines['bottom'].set_linewidth(1)
                
                # Background color
                ax.set_facecolor('#fafafa')
                
                # Adjust margins to prevent overlapping labels - generous margins
                if num_tribes > 20:
                    ax.margins(x=0.12, y=0.08)
                elif num_tribes > 15:
                    ax.margins(x=0.08, y=0.07)
                else:
                    ax.margins(x=0.05, y=0.06)
            
            # Overall title
            plt.suptitle(f'Per-Tribe WD by Cluster: Real vs Synthetic\n{prefix.replace("_", " ").title()}', 
                        fontsize=20, fontweight='bold', y=0.98, color='#2c3e50')
            plt.tight_layout(rect=[0, 0, 1, 0.95], pad=4.0)
            filename = f"per_tribe_wd.png"
            plt.savefig(artifact_output_dir / filename, dpi=300, bbox_inches='tight', facecolor='white')
            plt.close()
            logger.info(f"Saved: {filename}")
            
            # Save per-tribe statistics
            tribe_stats = {
                tribe: {
                    'mean_wd': float(np.mean(wds)),
                    'median_wd': float(np.median(wds)),
                    'std_wd': float(np.std(wds)),
                    'review_count': len(wds),
                    'min_wd': float(np.min(wds)),
                    'max_wd': float(np.max(wds))
                }
                for tribe, wds in tribe_wd.items() if len(wds) >= 3
            }
            tribe_stats_file = artifact_output_dir / "per_tribe_wd_statistics.json"
            with open(tribe_stats_file, 'w', encoding='utf-8') as f:
                json.dump(tribe_stats, f, indent=2)
            logger.info(f"Saved: per_tribe_wd_statistics.json")
        
        # ========================================================================
        # Graph 3: Real vs Synthetic Theme Distributions Comparison
        # ========================================================================
        if actual_sum > 0 and predicted_sum > 0:
            # Get top 25 themes for better visualization
            top_actual_25 = dict(actual_sorted[:25])
            top_predicted_25 = dict(predicted_sorted[:25])
            all_themes_vis_25 = sorted(set(top_actual_25.keys()) | set(top_predicted_25.keys()))
            
            # Side-by-side bar chart comparison
            fig, ax = plt.subplots(figsize=(max(16, len(all_themes_vis_25) * 0.5), 8))
            x_pos = np.arange(len(all_themes_vis_25))
            width = 0.35
            
            actual_values = [top_actual_25.get(theme, 0.0) for theme in all_themes_vis_25]
            predicted_values = [top_predicted_25.get(theme, 0.0) for theme in all_themes_vis_25]
            
            # Truncate long theme names
            display_themes = [t[:40] + '...' if len(t) > 40 else t for t in all_themes_vis_25]
            
            bars1 = ax.barh(x_pos - width/2, actual_values, width, label='Real (Actual)', 
                          alpha=0.8, color='#51CF66', edgecolor='#2F7D32', linewidth=1.2)
            bars2 = ax.barh(x_pos + width/2, predicted_values, width, label='Synthetic (Predicted)', 
                          alpha=0.8, color='#4A7FB5', edgecolor='#2B5A82', linewidth=1.2)
            
            ax.set_yticks(x_pos)
            ax.set_yticklabels(display_themes, fontsize=9)
            ax.set_xlabel('Probability', fontsize=13, fontweight='bold')
            ax.set_title(f'Theme Distributions: Real vs Synthetic\n{prefix.replace("_", " ").title()} (n={total_reviews_for_dist:,} reviews, Mean WD: {mean_wd:.4f})', 
                        fontsize=14, fontweight='bold', pad=15)
            ax.legend(loc='lower right', fontsize=12, framealpha=0.9)
            ax.grid(True, alpha=0.3, axis='x', linestyle=':', linewidth=0.8)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.invert_yaxis()
            plt.tight_layout()
            filename = f"real_vs_synthetic_themes.png"
            plt.savefig(artifact_output_dir / filename, dpi=300, bbox_inches='tight', facecolor='white')
            plt.close()
            logger.info(f"Saved: {filename}")
        
        # Approach 1 - Statistics
        stats_approach1 = {
            'approach': 'per_review_then_aggregate',
            'mean': float(mean_wd),
            'median': float(median_wd),
            'std': float(std_wd),
            'min': float(np.min(wd_values)),
            'max': float(np.max(wd_values)),
            '25th_percentile': float(np.percentile(wd_values, 25)),
            '75th_percentile': float(np.percentile(wd_values, 75)),
            'count': len(wd_values),
            'num_tribes': len(tribe_mean_wd) if tribe_mean_wd else 0
        }
        
        stats_file = artifact_output_dir / "statistics.json"
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(stats_approach1, f, indent=2)
        logger.info(f"Saved: statistics.json")
    
    # Print summary
    logger.info("\n" + "=" * 80)
    logger.info(f"WD Results Summary - {prefix.replace('_', ' ').title()}")
    logger.info("=" * 80)
    if results:
        logger.info(f"Per-Review WD: Mean = {mean_wd:.6f}, Median = {median_wd:.6f}")


def get_artifact_display_name(artifact_key: str) -> str:
    """
    Get clean display name for artifact key, removing 'context' and formatting nicely.
    
    Args:
        artifact_key: Artifact key (e.g., "pre_sgo_context", "post_sgo_context")
        
    Returns:
        Clean display name (e.g., "Pre-SGO", "Post-SGO")
    """
    # Remove _context suffix and format
    clean_key = artifact_key.replace('_context', '').replace('_', ' ')
    # Capitalize properly
    if 'pre sgo' in clean_key.lower():
        return 'Pre-SGO'
    elif 'post sgo' in clean_key.lower():
        return 'Post-SGO'
    elif 'user history' in clean_key.lower():
        return 'User History'
    elif 'user backstory' in clean_key.lower():
        return 'User Backstory'
    else:
        return clean_key.title()


def create_cluster_wise_comparison(
    all_results: Dict[str, List[Dict[str, Any]]],
    output_dir: Path,
    artifact_colors: List[str]
):
    """
    Create cluster-wise WD comparison across all methods.
    
    Args:
        all_results: Dictionary mapping artifact_key -> list of results
        output_dir: Output directory for graphs
        artifact_colors: List of colors for each artifact
    """
    logger.info("\nCreating cluster-wise comparison...")
    
    # Aggregate WD by cluster for each artifact
    cluster_data = defaultdict(lambda: defaultdict(list))  # cluster_id -> artifact_key -> [wds]
    cluster_labels = []
    
    for artifact_key, results in all_results.items():
        for result in results:
            cluster_id = result.get('cluster_id')
            wd = result.get('wd')
            if cluster_id and wd is not None and not np.isnan(wd) and np.isfinite(wd):
                cluster_data[cluster_id][artifact_key].append(wd)
    
    # Get all clusters and sort them
    all_clusters = sorted(cluster_data.keys())
    artifact_keys = list(all_results.keys())
    
    if not all_clusters:
        logger.warning("No cluster data found for comparison")
        return
    
    # Calculate individual averages for each artifact (across all clusters)
    avg_text_parts = []
    for artifact_key in artifact_keys:
        artifact_all_wds = []
        for cluster_id in all_clusters:
            wds = cluster_data[cluster_id][artifact_key]
            artifact_all_wds.extend(wds)
        artifact_avg = np.mean(artifact_all_wds) if artifact_all_wds else 0.0
        display_name = get_artifact_display_name(artifact_key)
        avg_text_parts.append(f"{display_name} Avg: {artifact_avg:.4f}")
    avg_text = "\n".join(avg_text_parts)
    
    # Create comparison graph
    fig, ax = plt.subplots(figsize=(max(12, len(all_clusters) * 2), 8))
    
    x = np.arange(len(all_clusters))
    width = 0.25 if len(artifact_keys) == 3 else (0.8 / len(artifact_keys))
    
    for idx, artifact_key in enumerate(artifact_keys):
        cluster_means = []
        for cluster_id in all_clusters:
            wds = cluster_data[cluster_id][artifact_key]
            cluster_means.append(np.mean(wds) if wds else 0.0)
        
        offset = (idx - len(artifact_keys) / 2 + 0.5) * width
        bars = ax.bar(x + offset, cluster_means, width, 
                     label=get_artifact_display_name(artifact_key),
                     color=artifact_colors[idx % len(artifact_colors)],
                     alpha=0.8, edgecolor='black', linewidth=1)
        
        # Add value labels
        for bar, val in zip(bars, cluster_means):
            if val > 0:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{val:.3f}', ha='center', va='bottom',
                       fontsize=9, fontweight='bold')
    
    ax.set_xlabel('Cluster', fontsize=12, fontweight='bold')
    ax.set_ylabel('Mean WD', fontsize=12, fontweight='bold')
    ax.set_title(f'WD Comparison by Cluster\n{avg_text}', 
                fontsize=14, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels([c.replace('cluster_', 'Cluster ') for c in all_clusters])
    ax.legend(loc='upper right', fontsize=11)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    plt.tight_layout()
    plt.savefig(output_dir / "cluster_wise_comparison.png", dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved cluster-wise comparison graph")
    
    # Save cluster statistics
    cluster_stats = {}
    for cluster_id in all_clusters:
        cluster_stats[cluster_id] = {}
        for artifact_key in artifact_keys:
            wds = cluster_data[cluster_id][artifact_key]
            if wds:
                cluster_stats[cluster_id][artifact_key] = {
                    'mean_wd': float(np.mean(wds)),
                    'median_wd': float(np.median(wds)),
                    'std_wd': float(np.std(wds)),
                    'count': len(wds)
                }
    
    stats_file = output_dir / "cluster_wise_stats.json"
    with open(stats_file, 'w', encoding='utf-8') as f:
        json.dump(cluster_stats, f, indent=2)
    logger.info(f"Saved cluster-wise statistics to: {stats_file}")


def create_tribe_wise_comparison(
    all_results: Dict[str, List[Dict[str, Any]]],
    output_dir: Path,
    artifact_colors: List[str]
):
    """
    Create tribe-wise WD comparison within each cluster.
    
    Args:
        all_results: Dictionary mapping artifact_key -> list of results
        output_dir: Output directory for graphs
        artifact_colors: List of colors for each artifact
    """
    logger.info("\nCreating tribe-wise comparison...")
    
    # Aggregate WD by cluster and tribe for each artifact
    cluster_tribe_data = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    # cluster_id -> tribe_id -> artifact_key -> [wds]
    tribe_names_map = {}  # tribe_id -> tribe_name
    
    for artifact_key, results in all_results.items():
        for result in results:
            cluster_id = result.get('cluster_id')
            tribe_id = result.get('tribe_id')
            tribe_name = result.get('tribe_name')
            wd = result.get('wd')
            if cluster_id and tribe_id and wd is not None and not np.isnan(wd) and np.isfinite(wd):
                cluster_tribe_data[cluster_id][tribe_id][artifact_key].append(wd)
                if tribe_name and tribe_id not in tribe_names_map:
                    tribe_names_map[tribe_id] = tribe_name
    
    artifact_keys = list(all_results.keys())
    
    # Create one graph per cluster
    for cluster_id in sorted(cluster_tribe_data.keys()):
        tribe_data = cluster_tribe_data[cluster_id]
        tribes = sorted(tribe_data.keys())
        
        if not tribes:
            continue
        
        fig, ax = plt.subplots(figsize=(max(14, len(tribes) * 0.8), 8))
        
        x = np.arange(len(tribes))
        width = 0.25 if len(artifact_keys) == 3 else (0.8 / len(artifact_keys))
        
        # Calculate individual averages for each artifact in this cluster
        avg_text_parts = []
        for artifact_key in artifact_keys:
            artifact_cluster_wds = []
            for tribe_id in tribes:
                wds = tribe_data[tribe_id][artifact_key]
                artifact_cluster_wds.extend(wds)
            artifact_avg = np.mean(artifact_cluster_wds) if artifact_cluster_wds else 0.0
            display_name = get_artifact_display_name(artifact_key)
            avg_text_parts.append(f"{display_name} Avg: {artifact_avg:.4f}")
        avg_text = "\n".join(avg_text_parts)
        
        for idx, artifact_key in enumerate(artifact_keys):
            tribe_means = []
            for tribe_id in tribes:
                wds = tribe_data[tribe_id][artifact_key]
                tribe_means.append(np.mean(wds) if wds else 0.0)
            
            offset = (idx - len(artifact_keys) / 2 + 0.5) * width
            bars = ax.bar(x + offset, tribe_means, width,
                         label=get_artifact_display_name(artifact_key),
                         color=artifact_colors[idx % len(artifact_colors)],
                         alpha=0.8, edgecolor='black', linewidth=1)
            
            # Add value labels on bars
            for bar, val in zip(bars, tribe_means):
                if val > 0:
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height,
                           f'{val:.3f}', ha='center', va='bottom',
                           fontsize=8, fontweight='bold')
        
        # Extract tribe names (use persona_name if available, otherwise fallback)
        tribe_labels = []
        for tribe_id in tribes:
            tribe_name = tribe_names_map.get(tribe_id)
            if tribe_name:
                # Truncate long names if needed
                if len(tribe_name) > 30:
                    tribe_labels.append(tribe_name[:27] + '...')
                else:
                    tribe_labels.append(tribe_name)
            else:
                # Fallback to micro number
                micro_match = re.search(r'micro_(\d+)', tribe_id)
                if micro_match:
                    tribe_labels.append(f"Tribe {micro_match.group(1)}")
                else:
                    tribe_labels.append(tribe_id.split('/')[-1][:20])
        
        ax.set_xlabel('Tribe', fontsize=12, fontweight='bold')
        ax.set_ylabel('Mean WD', fontsize=12, fontweight='bold')
        ax.set_title(f'WD Comparison by Tribe: {cluster_id.replace("cluster_", "Cluster ")}\n{avg_text}', 
                    fontsize=14, fontweight='bold', pad=20)
        ax.set_xticks(x)
        ax.set_xticklabels(tribe_labels, rotation=45, ha='right', fontsize=9)
        ax.legend(loc='upper right', fontsize=10)
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        plt.tight_layout()
        
        cluster_num = cluster_id.replace('cluster_', '')
        plt.savefig(output_dir / f"tribe_wise_comparison_{cluster_num}.png", 
                   dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Saved tribe-wise comparison for {cluster_id}")
    
    # Save tribe statistics
    tribe_stats = {}
    for cluster_id in sorted(cluster_tribe_data.keys()):
        tribe_stats[cluster_id] = {}
        for tribe_id in sorted(cluster_tribe_data[cluster_id].keys()):
            tribe_stats[cluster_id][tribe_id] = {}
            for artifact_key in artifact_keys:
                wds = cluster_tribe_data[cluster_id][tribe_id][artifact_key]
                if wds:
                    tribe_stats[cluster_id][tribe_id][artifact_key] = {
                        'mean_wd': float(np.mean(wds)),
                        'median_wd': float(np.median(wds)),
                        'std_wd': float(np.std(wds)),
                        'count': len(wds)
                    }
    
    stats_file = output_dir / "tribe_wise_stats.json"
    with open(stats_file, 'w', encoding='utf-8') as f:
        json.dump(tribe_stats, f, indent=2)
    logger.info(f"Saved tribe-wise statistics to: {stats_file}")


def create_pre_post_sgo_tribe_comparison(
    all_results: Dict[str, List[Dict[str, Any]]],
    output_dir: Path,
    pre_sgo_key: str = 'pre_sgo_context',
    post_sgo_key: str = 'post_sgo_context'
):
    """
    Create pre-SGO vs post-SGO tribe-wise comparison graphs for each cluster.
    Excludes baseline from comparison.
    
    Args:
        all_results: Dictionary mapping artifact_key -> list of results
        output_dir: Output directory for graphs
        pre_sgo_key: Key for pre-SGO artifact (from config)
        post_sgo_key: Key for post-SGO artifact (from config)
    """
    logger.info("\nCreating pre-SGO vs post-SGO tribe-wise comparison...")
    
    # Get pre-SGO and post-SGO results (exclude baseline)
    
    if pre_sgo_key not in all_results or post_sgo_key not in all_results:
        logger.warning(f"Missing pre-SGO or post-SGO results. Pre-SGO: {pre_sgo_key in all_results}, Post-SGO: {post_sgo_key in all_results}")
        return
    
    pre_sgo_results = all_results[pre_sgo_key]
    post_sgo_results = all_results[post_sgo_key]
    
    # Aggregate WD by cluster and tribe
    cluster_tribe_pre = defaultdict(lambda: defaultdict(list))  # cluster_id -> tribe_id -> [wds]
    cluster_tribe_post = defaultdict(lambda: defaultdict(list))  # cluster_id -> tribe_id -> [wds]
    tribe_names_map = {}  # tribe_id -> tribe_name
    
    for result in pre_sgo_results:
        cluster_id = result.get('cluster_id')
        tribe_id = result.get('tribe_id')
        tribe_name = result.get('tribe_name')
        wd = result.get('wd')
        if cluster_id and tribe_id and wd is not None and not np.isnan(wd) and np.isfinite(wd):
            cluster_tribe_pre[cluster_id][tribe_id].append(wd)
            if tribe_name and tribe_id not in tribe_names_map:
                tribe_names_map[tribe_id] = tribe_name
    
    for result in post_sgo_results:
        cluster_id = result.get('cluster_id')
        tribe_id = result.get('tribe_id')
        tribe_name = result.get('tribe_name')
        wd = result.get('wd')
        if cluster_id and tribe_id and wd is not None and not np.isnan(wd) and np.isfinite(wd):
            cluster_tribe_post[cluster_id][tribe_id].append(wd)
            if tribe_name and tribe_id not in tribe_names_map:
                tribe_names_map[tribe_id] = tribe_name
    
    # Create one graph per cluster
    all_clusters = sorted(set(cluster_tribe_pre.keys()) | set(cluster_tribe_post.keys()))
    
    for cluster_id in all_clusters:
        pre_tribes = cluster_tribe_pre.get(cluster_id, {})
        post_tribes = cluster_tribe_post.get(cluster_id, {})
        
        # Get all tribes that appear in either pre or post
        all_tribes = sorted(set(pre_tribes.keys()) | set(post_tribes.keys()))
        
        if not all_tribes:
            continue
        
        # Calculate individual averages for pre-SGO and post-SGO in this cluster
        pre_cluster_wds = []
        post_cluster_wds = []
        for tribe_id in all_tribes:
            pre_cluster_wds.extend(pre_tribes.get(tribe_id, []))
            post_cluster_wds.extend(post_tribes.get(tribe_id, []))
        pre_avg = np.mean(pre_cluster_wds) if pre_cluster_wds else 0.0
        post_avg = np.mean(post_cluster_wds) if post_cluster_wds else 0.0
        avg_text = f"Pre-SGO Avg: {pre_avg:.4f}\nPost-SGO Avg: {post_avg:.4f}"
        
        # Prepare data - only include tribes that have both pre and post data
        tribe_names = []
        pre_means = []
        post_means = []
        
        for tribe_id in all_tribes:
            pre_wds = pre_tribes.get(tribe_id, [])
            post_wds = post_tribes.get(tribe_id, [])
            
            # Only include if both have data
            if pre_wds and post_wds:
                # Use tribe name if available, otherwise fallback to micro number
                tribe_name = tribe_names_map.get(tribe_id)
                if tribe_name:
                    if len(tribe_name) > 30:
                        tribe_names.append(tribe_name[:27] + '...')
                    else:
                        tribe_names.append(tribe_name)
                else:
                    micro_match = re.search(r'micro_(\d+)', tribe_id)
                    if micro_match:
                        tribe_names.append(f"Tribe {micro_match.group(1)}")
                    else:
                        tribe_names.append(tribe_id.split('/')[-1][:20])
                
                pre_means.append(np.mean(pre_wds))
                post_means.append(np.mean(post_wds))
        
        if not tribe_names:
            continue
        
        # Create figure
        fig, ax = plt.subplots(figsize=(max(14, len(tribe_names) * 0.8), 8))
        
        x = np.arange(len(tribe_names))
        width = 0.35
        
        # Create grouped bar chart
        bars_pre = ax.bar(x - width/2, pre_means, width,
                         label='Pre-SGO', color='#3498DB', alpha=0.8, edgecolor='black', linewidth=1)
        bars_post = ax.bar(x + width/2, post_means, width,
                          label='Post-SGO', color='#E74C3C', alpha=0.8, edgecolor='black', linewidth=1)
        
        # Add value labels on bars
        for i, (pre_val, post_val) in enumerate(zip(pre_means, post_means)):
            # Pre-SGO value
            ax.text(i - width/2, pre_val, f'{pre_val:.3f}',
                   ha='center', va='bottom', fontsize=8, fontweight='bold')
            # Post-SGO value
            ax.text(i + width/2, post_val, f'{post_val:.3f}',
                   ha='center', va='bottom', fontsize=8, fontweight='bold')
        
        ax.set_xlabel('Tribe', fontsize=12, fontweight='bold')
        ax.set_ylabel('Mean WD', fontsize=12, fontweight='bold')
        ax.set_title(f'Pre-SGO vs Post-SGO WD by Tribe: {cluster_id.replace("cluster_", "Cluster ")}\n{avg_text}', 
                    fontsize=14, fontweight='bold', pad=20)
        ax.set_xticks(x)
        ax.set_xticklabels(tribe_names, rotation=45, ha='right', fontsize=9)
        ax.legend(loc='upper right', fontsize=11)
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        plt.tight_layout()
        
        cluster_num = cluster_id.replace('cluster_', '')
        plt.savefig(output_dir / f"pre_post_sgo_tribe_comparison_{cluster_num}.png", 
                   dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Saved pre-SGO vs post-SGO tribe comparison for {cluster_id}")
    
    logger.info("Completed pre-SGO vs post-SGO tribe-wise comparison")


def process_single_artifact(
    artifact_key: str,
    artifact_name: str,
    run,
    artifact_type: str,
    max_reviews: Optional[int],
    output_dir: Path,
    is_baseline: bool = False,
    add_wd_to_files: bool = True
) -> Optional[List[Dict[str, Any]]]:
    """
    Process a single artifact and return results.
    
    Args:
        artifact_key: Key name for the artifact (e.g., "user_history")
        artifact_name: Full artifact name from WandB (e.g., "baseline_predictions_o3_history_logprobs_v4:latest")
        run: WandB run object
        artifact_type: Type of artifact
        max_reviews: Maximum number of reviews to process
        output_dir: Output directory for results
        is_baseline: Whether this is a baseline artifact (different file structure)
        add_wd_to_files: Whether to add WD to original files
        
    Returns:
        List of processed review results, or None if processing failed
    """
    logger.info(f"\n{'='*80}")
    logger.info(f"Processing {artifact_key.upper().replace('_', ' ')}: {artifact_name}")
    logger.info(f"{'='*80}")
    
    artifact_path = None
    
    # For baseline files, prioritize local files first
    if is_baseline:
        logger.info("Checking for local baseline file first...")
        artifact_base = artifact_name.split(':')[0]
        # Try to construct the JSON filename from artifact name
        json_filename = artifact_base.replace('baseline_predictions_', 'baseline_predictions_o3_') + '.json'
        if not json_filename.endswith('.json'):
            json_filename = f"{artifact_base}.json"
        
        local_paths = [
            BASE_DIR / "09_baselines" / "artifacts" / artifact_base / json_filename,
            BASE_DIR / "09_baselines" / "artifacts" / "baseline_predictions_o3_history_logprobs_v4" / "baseline_predictions_o3_history_logprobs.json",
            BASE_DIR / "09_baselines" / "artifacts" / "baseline_predictions_o3_backstory_logprobs_v4" / "baseline_predictions_o3_backstory_logprobs.json",
            BASE_DIR / "09_baselines" / "artifacts" / "baseline_predictions_v4" / "baseline_predictions_o3_history.json",
        ]
        
        for local_path in local_paths:
            if local_path.exists():
                # For baseline files, the artifact_path should be the directory containing the JSON file
                artifact_path = local_path.parent
                logger.info(f"✓ Found local baseline file: {local_path}")
                logger.info(f"Artifact path set to: {artifact_path}")
                break
        
        # If still not found, try to find any JSON file in baseline artifacts directory
        if not artifact_path or not artifact_path.exists():
            baseline_dir = BASE_DIR / "09_baselines" / "artifacts"
            if baseline_dir.exists():
                json_files = list(baseline_dir.rglob("baseline_predictions*.json"))
                if json_files:
                    artifact_path = json_files[0].parent
                    logger.info(f"✓ Found local baseline file: {json_files[0]}")
                    logger.info(f"Artifact path set to: {artifact_path}")
        
        # If local file not found, try WandB download
        if not artifact_path or not artifact_path.exists():
            logger.info("Local file not found, trying WandB download...")
            if run:
                artifact_path = use_artifact(run, artifact_name, artifact_type=artifact_type)
                if artifact_path and artifact_path.exists():
                    logger.info(f"Artifact downloaded from WandB to: {artifact_path}")
    else:
        # For non-baseline files, try local first, then WandB
        local_paths = [
            BASE_DIR / "07_sgo_training" / "artifacts" / artifact_name.split(':')[0],
            BASE_DIR / "06_pre_sgo" / "artifacts" / artifact_name.split(':')[0],
        ]
        
        for local_path in local_paths:
            if local_path.exists():
                artifact_path = local_path
                logger.info(f"✓ Using local artifact directory: {artifact_path}")
                break
        
        # If local not found, try WandB download
        if not artifact_path or not artifact_path.exists():
            logger.info("Local artifact not found, trying WandB download...")
            if run:
                artifact_path = use_artifact(run, artifact_name, artifact_type=artifact_type)
                if artifact_path and artifact_path.exists():
                    logger.info(f"Artifact downloaded from WandB to: {artifact_path}")
    
    if not artifact_path or not artifact_path.exists():
        logger.error(f"Failed to download or find artifact: {artifact_name}")
        logger.error(f"Tried WandB download and local file fallback, but artifact not found.")
        return None
    
    results, updated_files = process_artifact_directory(
        artifact_path, max_reviews, add_wd_to_files=add_wd_to_files, is_baseline=is_baseline
    )
    
    if not results:
        logger.warning(f"No results processed for {artifact_key}")
        return None
    
    # Save per-review results with proper naming based on artifact_key
    prefix = artifact_key
    output_file = output_dir / f"approach1_per_review_{prefix}.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)
    logger.info(f"Saved per-review results to: {output_file}")
    
    if add_wd_to_files:
        logger.info(f"Added WD metrics to {len(updated_files)} artifact files")
        
        # Upload updated artifact back to WandB with proper naming
        if run and len(updated_files) > 0:
            # Use artifact_key for naming (e.g., "user_history_with_wd" instead of "baseline_predictions_o3_history_logprobs_v4_with_wd")
            updated_artifact_name = f"{artifact_key}_with_wd"
            logger.info(f"Uploading updated artifact to WandB: {updated_artifact_name}")
            try:
                log_artifact(
                    run=run,
                    artifact_name=updated_artifact_name,
                    artifact_type=artifact_type,
                    artifact_path=artifact_path,
                    metadata={
                        "description": f"{artifact_key} with WD metrics added to each review",
                        "original_artifact": artifact_name,
                        "artifact_key": artifact_key,
                        "num_files_updated": len(updated_files),
                        "num_reviews": len(results),
                        "mean_wd": float(np.mean([r['wd'] for r in results])) if results else 0.0
                    },
                    aliases=["latest"]
                )
                logger.info(f"✓ Successfully uploaded updated artifact to WandB: {updated_artifact_name}")
            except Exception as e:
                logger.warning(f"Failed to upload artifact to WandB: {e}")
    
    return results


def main():
    """Main execution function."""
    logger.info("=" * 80)
    logger.info("WD CALCULATION FOR PRE-SGO PREDICTIONS")
    logger.info("=" * 80)
    
    # Initialize W&B run
    run = init_wandb_run(
        run_name="calculate_wd_pre_sgo",
        stage="Metrics and analysis",
        config={"description": "Calculate WD for pre-SGO predictions"}
    )
    
    if run is None:
        logger.warning("W&B run initialization failed - running in local mode")
    
    try:
        # Load configuration
        # Try to load from stage config, fallback to direct file load
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
                return
        
        input_artifacts = config.get('input_artifacts', {})
        artifact_type = config.get('artifact_type', 'model')
        output_config = config.get('output', {})
        wd_config = config.get('wd', {})
        viz_config = config.get('visualization', {})
        processing_config = config.get('processing', {})
        
        # Set global epsilon
        global EPSILON
        EPSILON = float(wd_config.get('epsilon', 1e-10))
        
        # Create output directory (WD-only; uploaded as wd_metrics_and_graphs in wandb)
        output_dir = BASE_DIR / output_config.get('wd_directory', 'Metrics and analysis/artifacts/wd')
        output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Output directory (graphs and results): {output_dir.absolute()}")
        
        max_reviews = processing_config.get('max_reviews')
        if max_reviews:
            max_reviews = int(max_reviews)
        
        # Get mode and artifacts list from config
        wd_config_mode = wd_config.get('mode', '1')
        artifacts_list = wd_config.get('artifacts', [])
        
        if not artifacts_list:
            logger.error("No artifacts specified in config.wd.artifacts. Please specify which artifacts to process.")
            return
        
        # Determine which artifacts are baseline (different file structure)
        baseline_artifact_keys = ['user_history', 'user_backstory', 'baseline_predictions', 'baseline_backstory', 'baseline_history']
        
        # Validate mode and artifacts count match
        expected_count = {'1': 1, '2': 2, '3': 3, '4': 4}.get(wd_config_mode, 1)
        if len(artifacts_list) != expected_count:
            logger.warning(f"Mode '{wd_config_mode}' expects {expected_count} artifact(s), but {len(artifacts_list)} specified. Proceeding anyway.")
        
        logger.info(f"Mode: {wd_config_mode}, Processing artifacts: {artifacts_list}")
        
        # Process all specified artifacts
        all_results = {}
        for artifact_key in artifacts_list:
            artifact_name = input_artifacts.get(artifact_key)
            if not artifact_name:
                logger.error(f"Artifact '{artifact_key}' not found in input_artifacts config")
                continue
            
            is_baseline = artifact_key in baseline_artifact_keys
            results = process_single_artifact(
                artifact_key=artifact_key,
                artifact_name=artifact_name,
                run=run,
                artifact_type=artifact_type,
                max_reviews=max_reviews,
                output_dir=output_dir,
                is_baseline=is_baseline,
                add_wd_to_files=(wd_config_mode == '1')  # Only add to files in mode 1
            )
            
            if results:
                all_results[artifact_key] = results
                
                # For mode 1, create individual visualizations
                if wd_config_mode == '1':
                    prefix = artifact_key
                    create_visualizations(results, output_dir, prefix=prefix)
        
        # Generate pre-SGO vs post-SGO tribe comparison when mode is "2" with pre/post artifacts
        if wd_config_mode == '2' and 'pre_sgo_context' in all_results and 'post_sgo_context' in all_results:
            create_pre_post_sgo_tribe_comparison(all_results, output_dir, 'pre_sgo_context', 'post_sgo_context')
        
        # For comparison modes, generate comparison graphs
        if wd_config_mode in ['2', '3', '4'] and len(all_results) >= 2:
            logger.info("\n" + "=" * 80)
            logger.info(f"GENERATING WD COMPARISON GRAPHS: {len(all_results)}-way comparison")
            logger.info("=" * 80)
            
            comparison_dir = output_dir / f"wd_comparison_{wd_config_mode}way"
            comparison_dir.mkdir(parents=True, exist_ok=True)
            
            # Create overall comparison graph
            artifact_labels = []
            artifact_means = []
            artifact_colors = ['#E74C3C', '#3498DB', '#E67E22', '#9B59B6']  # Colors for up to 4 artifacts
            
            for idx, (artifact_key, results) in enumerate(all_results.items()):
                wd_values = [r.get('wd') for r in results if r.get('wd') is not None]
                mean_wd = np.mean(wd_values) if wd_values else 0.0
                artifact_labels.append(get_artifact_display_name(artifact_key))
                artifact_means.append(mean_wd)
                logger.info(f"{artifact_key}: Mean WD = {mean_wd:.6f} (n={len(wd_values)})")
            
            # Calculate individual averages for each artifact
            avg_text_parts = []
            for artifact_key, results in all_results.items():
                wd_values = [r.get('wd') for r in results if r.get('wd') is not None]
                artifact_avg = np.mean(wd_values) if wd_values else 0.0
                display_name = get_artifact_display_name(artifact_key)
                avg_text_parts.append(f"{display_name} Avg: {artifact_avg:.4f}")
            avg_text = "\n".join(avg_text_parts)
            
            # Overall comparison graph
            fig, ax = plt.subplots(figsize=(max(10, len(artifact_labels) * 2.5), 7))
            bars = ax.bar(artifact_labels, artifact_means, 
                         color=artifact_colors[:len(artifact_labels)], 
                         alpha=0.8, edgecolor='black', linewidth=1.5)
            for bar, val in zip(bars, artifact_means):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height, f'{val:.4f}',
                       ha='center', va='bottom', fontsize=12, fontweight='bold')
            ax.set_ylabel('Wasserstein Distance', fontsize=12, fontweight='bold')
            ax.set_title(f'WD Comparison: {len(artifact_labels)}-way (Overall)\n{avg_text}', 
                        fontsize=14, fontweight='bold', pad=20)
            ax.set_ylim(0, max(artifact_means) * 1.2 if artifact_means else 1.0)
            ax.grid(axis='y', alpha=0.3, linestyle='--')
            plt.tight_layout()
            plt.savefig(comparison_dir / "overall_wd_comparison.png", dpi=300, bbox_inches='tight')
            plt.close()
            logger.info(f"Saved overall comparison graph")
            
            # Cluster-wise comparison
            create_cluster_wise_comparison(all_results, comparison_dir, artifact_colors)
            
            # Tribe-wise comparison (within clusters)
            create_tribe_wise_comparison(all_results, comparison_dir, artifact_colors)
            
            # Save comparison statistics
            stats = {}
            for artifact_key, results in all_results.items():
                wd_values = [r.get('wd') for r in results if r.get('wd') is not None]
                stats[artifact_key] = {
                    'mean_wd': float(np.mean(wd_values)) if wd_values else 0.0,
                    'median_wd': float(np.median(wd_values)) if wd_values else 0.0,
                    'std_wd': float(np.std(wd_values)) if wd_values else 0.0,
                    'count': len(wd_values)
                }
            stats_file = comparison_dir / "comparison_stats.json"
            with open(stats_file, 'w', encoding='utf-8') as f:
                json.dump(stats, f, indent=2)
            logger.info(f"Saved comparison statistics to: {stats_file}")
        
        # Upload all WD graphs and results to WandB
        if run and output_dir.exists():
            logger.info("Uploading WD metrics and graphs to WandB...")
            artifact_type = config.get("artifact_type", "result")
            try:
                log_artifact(
                    run=run,
                    artifact_name=config.get("output_artifacts", {}).get("wd_metrics", "wd_metrics_and_graphs"),
                    artifact_type=artifact_type,
                    artifact_path=str(output_dir),
                    metadata={"description": "WD metrics, distribution graphs, per-cluster/tribe comparisons, and comparison results"},
                    aliases=["latest"],
                )
                logger.info("✓ Uploaded WD metrics and graphs to WandB")
            except Exception as e:
                logger.warning(f"Failed to upload WD artifact to WandB: {e}")
        
        # OLD CODE REMOVED - All processing is now handled by the flexible artifact system above
        
        logger.info("\n" + "=" * 80)
        logger.info("WD CALCULATION COMPLETE")
        logger.info("=" * 80)
        
    except Exception as e:
        logger.error(f"Error in main execution: {e}", exc_info=True)
    finally:
        if run:
            finish_run(run)




if __name__ == "__main__":
    main()
