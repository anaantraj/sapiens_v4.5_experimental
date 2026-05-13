#!/usr/bin/env python3
"""
Calculate JSD and WD for specific artifact directories by processing delta files.
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

EPSILON = 1e-10


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
    """Align two theme distributions to the same set of themes and normalize."""
    all_themes = sorted(set(actual_themes.keys()) | set(predicted_themes.keys()))
    if not all_themes:
        return np.array([]), np.array([])
    
    actual_array = np.array([actual_themes.get(theme, 0.0) for theme in all_themes])
    predicted_array = np.array([predicted_themes.get(theme, 0.0) for theme in all_themes])
    
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
    """Compute Jensen-Shannon Divergence between two probability distributions."""
    if len(P) == 0 or len(Q) == 0:
        return float('nan')
    if len(P) != len(Q):
        return float('nan')
    
    P = P + epsilon
    Q = Q + epsilon
    P = P / P.sum()
    Q = Q / Q.sum()
    M = 0.5 * (P + Q)
    
    kl_pm = entropy(P, M, base=2)
    kl_qm = entropy(Q, M, base=2)
    jsd = 0.5 * kl_pm + 0.5 * kl_qm
    
    return float(jsd)


def calculate_wasserstein_1_distance(
    actual_array: np.ndarray,
    predicted_array: np.ndarray
) -> float:
    """Calculate Wasserstein-1 distance using optimal transport."""
    if len(actual_array) == 0 or len(predicted_array) == 0:
        return float('nan')
    if len(actual_array) != len(predicted_array):
        return float('nan')
    
    actual_sum = actual_array.sum()
    predicted_sum = predicted_array.sum()
    
    if actual_sum < EPSILON or predicted_sum < EPSILON:
        return float('nan')
    
    P = predicted_array / predicted_sum
    Q = actual_array / actual_sum
    n = len(P)
    cost_matrix = np.ones((n, n)) - np.eye(n)
    
    if HAS_OT:
        try:
            wd = ot.emd2(P, Q, cost_matrix, numItermax=1000000)
            return float(wd)
        except Exception as e:
            logger.warning(f"Error in optimal transport calculation: {e}")
            wd = np.sum(np.abs(P - Q))
            return float(wd)
    else:
        wd = np.sum(np.abs(P - Q))
        return float(wd)


def calculate_metrics_for_review(review: Dict[str, Any]) -> Tuple[Optional[float], Optional[float]]:
    """Calculate JSD and WD for a single review."""
    try:
        prediction = review.get('prediction', {})
        actual = review.get('actual', {})
        
        if not prediction or not actual:
            return None, None
        
        actual_themes = actual.get('topic_probabilities', {})
        predicted_themes = prediction.get('predicted_themes', {})
        
        if isinstance(actual_themes, list):
            if not actual_themes:
                return None, None
            actual_themes = {theme: 1.0 / len(actual_themes) for theme in actual_themes}
        
        if isinstance(predicted_themes, list):
            if not predicted_themes:
                return None, None
            predicted_themes = {theme: 1.0 / len(predicted_themes) for theme in predicted_themes}
        
        if not actual_themes or not predicted_themes:
            return None, None
        
        actual_themes = normalize_distribution(actual_themes)
        predicted_themes = normalize_distribution(predicted_themes)
        
        if not actual_themes or not predicted_themes:
            return None, None
        
        actual_array, predicted_array = align_distributions(actual_themes, predicted_themes)
        
        if len(actual_array) == 0 or len(predicted_array) == 0:
            return None, None
        
        jsd = compute_jsd(actual_array, predicted_array)
        wd = calculate_wasserstein_1_distance(actual_array, predicted_array)
        
        if np.isnan(jsd) or not np.isfinite(jsd):
            jsd = None
        if np.isnan(wd) or not np.isfinite(wd):
            wd = None
        
        return jsd, wd
        
    except Exception as e:
        logger.debug(f"Error calculating metrics for review: {e}")
        return None, None


def process_delta_file(file_path: Path) -> Dict[str, Any]:
    """Process a single delta file and calculate metrics for all reviews."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        logger.error(f"Error reading file {file_path}: {e}")
        return {}
    
    cluster_id = data.get('cluster_id', 'unknown')
    micro_cluster_id = data.get('micro_cluster_id', 'unknown')
    tribe_id = data.get('tribe_id', 'unknown')
    
    deltas = data.get('deltas', [])
    if not deltas:
        logger.warning(f"No deltas found in {file_path}")
        return {}
    
    jsd_values = []
    wd_values = []
    valid_reviews = 0
    
    for review in deltas:
        jsd, wd = calculate_metrics_for_review(review)
        
        if jsd is not None:
            jsd_values.append(jsd)
            if wd is not None:
                wd_values.append(wd)
            valid_reviews += 1
        elif wd is not None:
            wd_values.append(wd)
            valid_reviews += 1
    
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
    
    return stats


def get_display_name(dir_name: str) -> str:
    """
    Get proper display name for directory based on actual content.
    
    Mapping:
    - sgo_train_final_predictions_user_seed_backstory → sgo_test_final_predictions_user_seed_backstory (user_backstory + seed_backstory for test)
    - sgo_train_final_predictions_seed_backstory_only → sgo_test_final_predictions_seed_backstory_only (seed_backstory only for test)
    - sgo_train_final_predictions_user_seed_memory → sgo_test_final_predictions_user_seed_memory (seed_backstory + user_backstory + memory for test)
    """
    name_mapping = {
        'sgo_train_final_predictions_user_seed_backstory': 'sgo_test_final_predictions_user_seed_backstory',
        'sgo_train_final_predictions_seed_backstory_only': 'sgo_test_final_predictions_seed_backstory_only',
        'sgo_train_final_predictions_user_seed_memory': 'sgo_test_final_predictions_user_seed_memory'
    }
    return name_mapping.get(dir_name, dir_name)


def process_directory(artifacts_dir: Path, dir_name: str, output_dir: Path):
    """Process all delta files in a directory."""
    display_name = get_display_name(dir_name)
    
    logger.info(f"\n{'='*80}")
    logger.info(f"Processing: {dir_name}")
    logger.info(f"Display Name: {display_name}")
    logger.info(f"{'='*80}")
    
    deltas_dir = artifacts_dir / dir_name / "deltas"
    
    if not deltas_dir.exists():
        logger.warning(f"Deltas directory not found: {deltas_dir}")
        return
    
    # Find all delta files
    delta_files = list(deltas_dir.rglob("*_all_reviews_deltas.json"))
    logger.info(f"Found {len(delta_files)} delta files")
    
    if not delta_files:
        logger.warning(f"No delta files found in {deltas_dir}")
        return
    
    all_stats = []
    cluster_stats = defaultdict(lambda: {'jsd': [], 'wd': []})
    micro_stats = defaultdict(lambda: {'jsd': [], 'wd': []})
    
    for delta_file in tqdm(delta_files, desc=f"Processing {dir_name}"):
        stats = process_delta_file(delta_file)
        if stats:
            all_stats.append(stats)
            
            cluster_id = stats['cluster_id']
            micro_id = stats['micro_cluster_id']
            
            if stats['jsd']['mean'] is not None:
                cluster_stats[cluster_id]['jsd'].append(stats['jsd']['mean'])
                micro_stats[f"{cluster_id}/{micro_id}"]['jsd'].append(stats['jsd']['mean'])
            
            if stats['wd']['mean'] is not None:
                cluster_stats[cluster_id]['wd'].append(stats['wd']['mean'])
                micro_stats[f"{cluster_id}/{micro_id}"]['wd'].append(stats['wd']['mean'])
    
    # Calculate summary statistics
    all_jsd = []
    all_wd = []
    for stats in all_stats:
        if stats['jsd']['mean'] is not None:
            all_jsd.append(stats['jsd']['mean'])
        if stats['wd']['mean'] is not None:
            all_wd.append(stats['wd']['mean'])
    
    # Get display name for output
    display_name = get_display_name(dir_name)
    
    summary = {
        'directory': dir_name,
        'display_name': display_name,
        'total_delta_files': len(delta_files),
        'total_micro_clusters': len(all_stats),
        'overall': {
            'jsd': {
                'mean': float(np.mean(all_jsd)) if all_jsd else None,
                'median': float(np.median(all_jsd)) if all_jsd else None,
                'std': float(np.std(all_jsd)) if all_jsd else None,
                'min': float(np.min(all_jsd)) if all_jsd else None,
                'max': float(np.max(all_jsd)) if all_jsd else None,
                'count': len(all_jsd)
            },
            'wd': {
                'mean': float(np.mean(all_wd)) if all_wd else None,
                'median': float(np.median(all_wd)) if all_wd else None,
                'std': float(np.std(all_wd)) if all_wd else None,
                'min': float(np.min(all_wd)) if all_wd else None,
                'max': float(np.max(all_wd)) if all_wd else None,
                'count': len(all_wd)
            }
        },
        'by_cluster': {
            cluster_id: {
                'jsd': {
                    'mean': float(np.mean(vals['jsd'])) if vals['jsd'] else None,
                    'count': len(vals['jsd'])
                },
                'wd': {
                    'mean': float(np.mean(vals['wd'])) if vals['wd'] else None,
                    'count': len(vals['wd'])
                }
            }
            for cluster_id, vals in cluster_stats.items()
        },
        'by_micro_cluster': {
            micro_id: {
                'jsd': {
                    'mean': float(np.mean(vals['jsd'])) if vals['jsd'] else None,
                    'count': len(vals['jsd'])
                },
                'wd': {
                    'mean': float(np.mean(vals['wd'])) if vals['wd'] else None,
                    'count': len(vals['wd'])
                }
            }
            for micro_id, vals in micro_stats.items()
        },
        'detailed_stats': all_stats
    }
    
    # Save results - use display name for output directory
    output_subdir = output_dir / display_name
    output_subdir.mkdir(parents=True, exist_ok=True)
    
    summary_file = output_subdir / "jsd_wd_summary.json"
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    logger.info(f"Saved summary to: {summary_file}")
    
    # Print summary
    logger.info(f"\nSummary for {display_name}:")
    if all_jsd:
        logger.info(f"  JSD: Mean = {np.mean(all_jsd):.6f}, Median = {np.median(all_jsd):.6f}, Count = {len(all_jsd)}")
    if all_wd:
        logger.info(f"  WD:  Mean = {np.mean(all_wd):.6f}, Median = {np.median(all_wd):.6f}, Count = {len(all_wd)}")


def main():
    """Main execution function."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Calculate JSD and WD for artifact directories')
    parser.add_argument('--artifacts-dir', type=str, default=None,
                       help='Base artifacts directory (default: 07_sgo_training/artifacts)')
    parser.add_argument('--output-dir', type=str, default=None,
                       help='Output directory for results (default: artifacts/jsd_wd_results)')
    parser.add_argument('--directories', type=str, nargs='+',
                       default=['sgo_train_final_predictions_seed_backstory_only', 
                               'sgo_train_final_predictions_user_seed_backstory'],
                       help='Directory names to process (actual directory names in artifacts folder)')
    
    args = parser.parse_args()
    
    # Determine artifacts directory
    if args.artifacts_dir:
        artifacts_base = Path(args.artifacts_dir)
    else:
        artifacts_base = BASE_DIR / "07_sgo_training" / "artifacts"
    
    # Determine output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = artifacts_base / "jsd_wd_results"
    
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory: {output_dir}")
    
    # Process each directory
    for dir_name in args.directories:
        process_directory(artifacts_base, dir_name, output_dir)
    
    logger.info("\n" + "=" * 80)
    logger.info("JSD and WD calculation complete!")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
