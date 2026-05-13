#!/usr/bin/env python3
"""
Calculate JSD and WD for Delta Files
=====================================

This script:
1. Processes all delta files in sgo_train_final_predictions_refined_chars/deltas/
2. For each review, calculates JSD and WD between:
   - prediction.predicted_themes (predicted probability distribution)
   - actual.topic_probabilities (ground truth probability distribution)
3. Adds metrics to each review in the file
4. Calculates mean JSD/WD per micro cluster and per cluster
5. Saves updated files and summary statistics

Usage:
    python Metrics and analysis/scripts/calculate_jsd_wd_deltas.py
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
    logging.warning("Falling back to L1 distance approximation")

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
    # Get all unique themes from both distributions
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
    if len(P) == 0 or len(Q) == 0:
        return float('nan')
    
    if len(P) != len(Q):
        return float('nan')
    
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


def calculate_wasserstein_1_distance(
    actual_array: np.ndarray,
    predicted_array: np.ndarray
) -> float:
    """
    Calculate Wasserstein-1 distance (p=1) using optimal transport formulation.
    
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


def calculate_metrics_for_review(review: Dict[str, Any]) -> Tuple[Optional[float], Optional[float]]:
    """
    Calculate JSD and WD for a single review.
    
    Args:
        review: Review dictionary with 'prediction' and 'actual' keys
        
    Returns:
        Tuple of (jsd, wd) or (None, None) if calculation failed
    """
    try:
        prediction = review.get('prediction', {})
        actual = review.get('actual', {})
        
        if not prediction or not actual:
            return None, None
        
        # Get theme distributions
        actual_themes = actual.get('topic_probabilities', {})
        predicted_themes = prediction.get('predicted_themes', {})
        
        # Handle different formats (list vs dict)
        if isinstance(actual_themes, list):
            if not actual_themes:
                return None, None
            actual_themes = {theme: 1.0 / len(actual_themes) for theme in actual_themes}
        
        if isinstance(predicted_themes, list):
            if not predicted_themes:
                return None, None
            predicted_themes = {theme: 1.0 / len(predicted_themes) for theme in predicted_themes}
        
        # Validate that we have distributions
        if not actual_themes or not predicted_themes:
            return None, None
        
        # Normalize distributions to ensure they sum to 1.0
        actual_themes = normalize_distribution(actual_themes)
        predicted_themes = normalize_distribution(predicted_themes)
        
        if not actual_themes or not predicted_themes:
            return None, None
        
        # Align distributions (will also normalize again as safety check)
        actual_array, predicted_array = align_distributions(actual_themes, predicted_themes)
        
        if len(actual_array) == 0 or len(predicted_array) == 0:
            return None, None
        
        # Calculate JSD
        jsd = compute_jsd(actual_array, predicted_array)
        
        # Calculate WD
        wd = calculate_wasserstein_1_distance(actual_array, predicted_array)
        
        # Check for NaN or infinite values
        if np.isnan(jsd) or not np.isfinite(jsd):
            jsd = None
        if np.isnan(wd) or not np.isfinite(wd):
            wd = None
        
        return jsd, wd
        
    except Exception as e:
        logger.warning(f"Error calculating metrics for review: {e}")
        return None, None


def process_delta_file(file_path: Path) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Process a single delta file and calculate metrics for all reviews.
    
    Args:
        file_path: Path to the delta JSON file
        
    Returns:
        Tuple of (updated_data, statistics_dict)
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        logger.error(f"Error reading file {file_path}: {e}")
        return None, None
    
    cluster_id = data.get('cluster_id', 'unknown')
    micro_cluster_id = data.get('micro_cluster_id', 'unknown')
    tribe_id = data.get('tribe_id', 'unknown')
    
    deltas = data.get('deltas', [])
    if not deltas:
        logger.warning(f"No deltas found in {file_path}")
        return data, {}
    
    # Calculate metrics for each review
    jsd_values = []
    wd_values = []
    valid_reviews = 0
    
    for review in deltas:
        jsd, wd = calculate_metrics_for_review(review)
        
        # Add metrics to review
        if 'metrics' not in review:
            review['metrics'] = {}
        
        if jsd is not None:
            review['metrics']['jsd'] = float(jsd)
            review['metrics']['wd'] = float(wd) if wd is not None else None
            jsd_values.append(jsd)
            if wd is not None:
                wd_values.append(wd)
            valid_reviews += 1
        elif wd is not None:
            review['metrics']['wd'] = float(wd)
            review['metrics']['jsd'] = None
            wd_values.append(wd)
            valid_reviews += 1
        else:
            review['metrics']['jsd'] = None
            review['metrics']['wd'] = None
    
    # Calculate statistics
    stats = {
        'cluster_id': cluster_id,
        'micro_cluster_id': micro_cluster_id,
        'tribe_id': tribe_id,
        'total_reviews': len(deltas),
        'valid_reviews': valid_reviews,
        'jsd': {
            'mean': float(np.mean(jsd_values)) if jsd_values else None,
            'median': float(np.median(jsd_values)) if jsd_values else None,
            'std': float(np.std(jsd_values)) if jsd_values else None,
            'min': float(np.min(jsd_values)) if jsd_values else None,
            'max': float(np.max(jsd_values)) if jsd_values else None,
            'count': len(jsd_values)
        },
        'wd': {
            'mean': float(np.mean(wd_values)) if wd_values else None,
            'median': float(np.median(wd_values)) if wd_values else None,
            'std': float(np.std(wd_values)) if wd_values else None,
            'min': float(np.min(wd_values)) if wd_values else None,
            'max': float(np.max(wd_values)) if wd_values else None,
            'count': len(wd_values)
        }
    }
    
    return data, stats


def process_all_delta_files(deltas_dir: Path) -> Dict[str, Any]:
    """
    Process all delta files in the deltas directory.
    
    Args:
        deltas_dir: Path to the deltas directory
        
    Returns:
        Dictionary with cluster-level and micro-level statistics
    """
    # Find all delta files
    delta_files = list(deltas_dir.rglob("micro_*_all_reviews_deltas.json"))
    logger.info(f"Found {len(delta_files)} delta files to process")
    
    # Statistics structure
    cluster_stats = defaultdict(lambda: {
        'micro_clusters': {},
        'all_jsd': [],
        'all_wd': []
    })
    
    # Process each file
    for delta_file in tqdm(delta_files, desc="Processing delta files"):
        updated_data, stats = process_delta_file(delta_file)
        
        if updated_data is None:
            continue
        
        # Save updated file
        try:
            with open(delta_file, 'w', encoding='utf-8') as f:
                json.dump(updated_data, f, indent=2, ensure_ascii=False)
            logger.debug(f"Updated file: {delta_file}")
        except Exception as e:
            logger.error(f"Error saving file {delta_file}: {e}")
            continue
        
        # Aggregate statistics
        cluster_id = stats['cluster_id']
        micro_cluster_id = stats['micro_cluster_id']
        
        cluster_stats[cluster_id]['micro_clusters'][micro_cluster_id] = stats
        
        # Collect all individual JSD and WD values from reviews for cluster-level statistics
        deltas = updated_data.get('deltas', [])
        for review in deltas:
            metrics = review.get('metrics', {})
            jsd = metrics.get('jsd')
            wd = metrics.get('wd')
            
            if jsd is not None:
                cluster_stats[cluster_id]['all_jsd'].append(jsd)
            if wd is not None:
                cluster_stats[cluster_id]['all_wd'].append(wd)
    
    # Calculate cluster-level statistics
    for cluster_id, cluster_data in cluster_stats.items():
        all_jsd = cluster_data['all_jsd']
        all_wd = cluster_data['all_wd']
        
        cluster_data['cluster_jsd'] = {
            'mean': float(np.mean(all_jsd)) if all_jsd else None,
            'median': float(np.median(all_jsd)) if all_jsd else None,
            'std': float(np.std(all_jsd)) if all_jsd else None,
            'min': float(np.min(all_jsd)) if all_jsd else None,
            'max': float(np.max(all_jsd)) if all_jsd else None,
            'count': len(all_jsd)
        }
        
        cluster_data['cluster_wd'] = {
            'mean': float(np.mean(all_wd)) if all_wd else None,
            'median': float(np.median(all_wd)) if all_wd else None,
            'std': float(np.std(all_wd)) if all_wd else None,
            'min': float(np.min(all_wd)) if all_wd else None,
            'max': float(np.max(all_wd)) if all_wd else None,
            'count': len(all_wd)
        }
    
    return dict(cluster_stats)


def main():
    """Main execution function."""
    logger.info("=" * 80)
    logger.info("JSD AND WD CALCULATION FOR DELTA FILES")
    logger.info("=" * 80)
    
    # Set paths
    deltas_dir = BASE_DIR / "07_sgo_training" / "artifacts" / "sgo_train_final_predictions_refined_chars" / "deltas"
    
    if not deltas_dir.exists():
        logger.error(f"Deltas directory not found: {deltas_dir}")
        return
    
    logger.info(f"Processing delta files in: {deltas_dir}")
    
    # Process all delta files
    cluster_stats = process_all_delta_files(deltas_dir)
    
    # Save summary statistics
    output_dir = BASE_DIR / "07_sgo_training" / "artifacts" / "sgo_train_final_predictions_refined_chars"
    summary_file = output_dir / "deltas_jsd_wd_summary.json"
    
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(cluster_stats, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Saved summary statistics to: {summary_file}")
    
    # Print summary
    logger.info("\n" + "=" * 80)
    logger.info("SUMMARY STATISTICS")
    logger.info("=" * 80)
    
    for cluster_id, stats in sorted(cluster_stats.items()):
        logger.info(f"\n{cluster_id}:")
        logger.info(f"  Micro clusters: {len(stats['micro_clusters'])}")
        
        if stats['cluster_jsd']['mean'] is not None:
            logger.info(f"  JSD - Mean: {stats['cluster_jsd']['mean']:.6f}, "
                       f"Median: {stats['cluster_jsd']['median']:.6f}, "
                       f"Std: {stats['cluster_jsd']['std']:.6f}")
        
        if stats['cluster_wd']['mean'] is not None:
            logger.info(f"  WD  - Mean: {stats['cluster_wd']['mean']:.6f}, "
                       f"Median: {stats['cluster_wd']['median']:.6f}, "
                       f"Std: {stats['cluster_wd']['std']:.6f}")
        
        # Print top and bottom micro clusters by JSD
        micro_clusters = stats['micro_clusters']
        if micro_clusters:
            sorted_by_jsd = sorted(
                [(mid, mstats) for mid, mstats in micro_clusters.items() 
                 if mstats['jsd']['mean'] is not None],
                key=lambda x: x[1]['jsd']['mean']
            )
            
            if sorted_by_jsd:
                logger.info(f"\n  Top 3 micro clusters (lowest JSD):")
                for mid, mstats in sorted_by_jsd[:3]:
                    wd_str = f"{mstats['wd']['mean']:.6f}" if mstats['wd']['mean'] is not None else 'N/A'
                    logger.info(f"    {mid}: JSD={mstats['jsd']['mean']:.6f}, "
                               f"WD={wd_str}, "
                               f"n={mstats['valid_reviews']}")
                
                logger.info(f"\n  Bottom 3 micro clusters (highest JSD):")
                for mid, mstats in sorted_by_jsd[-3:]:
                    wd_str = f"{mstats['wd']['mean']:.6f}" if mstats['wd']['mean'] is not None else 'N/A'
                    logger.info(f"    {mid}: JSD={mstats['jsd']['mean']:.6f}, "
                               f"WD={wd_str}, "
                               f"n={mstats['valid_reviews']}")
    
    logger.info("\n" + "=" * 80)
    logger.info("CALCULATION COMPLETE")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()

