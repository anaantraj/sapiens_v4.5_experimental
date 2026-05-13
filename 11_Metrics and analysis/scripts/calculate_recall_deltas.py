#!/usr/bin/env python3
"""
Calculate Recall@max(3,k) and Recall@max(4,k) for Delta Files
=============================================================

This script:
1. Loads delta files from sgo_train_final_predictions_refined_chars/deltas/
2. Matches each review with ground truth from ground_truth_test_micro_cluster_details
3. Uses predicted_themes from ground truth as the ground truth themes
4. Calculates Recall@max(3,k) and Recall@max(4,k) by comparing:
   - Top max(3,k) or max(4,k) predicted themes (sorted by probability) from delta file
   - With predicted_themes list from ground truth
5. Updates delta files with recall metrics

Usage:
    python Metrics and analysis/scripts/calculate_recall_deltas.py
"""

import json
import sys
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict
from tqdm import tqdm

# Add project root to path
BASE_DIR = Path(__file__).parent.parent.parent
sys.path.insert(0, str(BASE_DIR))
sys.path.insert(0, str(BASE_DIR / "Metrics and analysis" / "scripts"))

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import theme mapping function
try:
    from map_predicted_themes_setb_to_seta import map_themes_setb_to_seta
except ImportError:
    logger.error("Could not import map_themes_setb_to_seta. Make sure the file exists.")
    sys.exit(1)

# Import recall calculation functions
try:
    from calculate_jsd_wd_predictions_vs_ground_truth import (
        calculate_recall_at_max3k,
        calculate_recall_at_max4k,
        create_match_key
    )
except ImportError:
    logger.error("Could not import recall calculation functions.")
    sys.exit(1)


def load_ground_truth(ground_truth_dir: Path) -> Dict[str, Dict[str, Any]]:
    """
    Load all ground truth files from cluster/micro directories.
    
    Args:
        ground_truth_dir: Path to ground truth directory
        
    Returns:
        Dictionary mapping match_key -> ground truth review data
    """
    logger.info(f"Loading ground truth from: {ground_truth_dir}")
    
    ground_truth = {}
    total_reviews = 0
    match_keys_seen = set()  # Track duplicates
    
    # Find all cluster directories
    cluster_dirs = sorted([d for d in ground_truth_dir.iterdir() if d.is_dir() and d.name.startswith('cluster_')])
    logger.info(f"Found {len(cluster_dirs)} cluster directories")
    
    for cluster_dir in tqdm(cluster_dirs, desc="Loading clusters"):
        cluster_id = cluster_dir.name
        
        # Find all micro detail files
        micro_files = sorted(cluster_dir.glob("micro_*_details.json"))
        
        for micro_file in micro_files:
            micro_id = micro_file.stem.replace('_details', '')
            
            try:
                with open(micro_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # Extract members_grouped_by_user
                members_grouped = data.get('members_grouped_by_user', {})
                persona_name = data.get('persona_name', 'Unknown')
                
                # Process each user's reviews
                for user_id, reviews in members_grouped.items():
                    if not isinstance(reviews, list):
                        continue
                    
                    for review in reviews:
                        if not isinstance(review, dict):
                            continue
                        
                        product_description = review.get('product_description', '')
                        category = review.get('category', '')
                        
                        # Skip if missing required fields
                        if not product_description or not category:
                            continue
                        
                        # Create match key
                        match_key = create_match_key(user_id, product_description, category)
                        
                        # Skip if we've already seen this match key (duplicate)
                        if match_key in match_keys_seen:
                            continue
                        match_keys_seen.add(match_key)
                        
                        # Store ground truth data - use predicted_themes as ground truth
                        ground_truth[match_key] = {
                            'user_id': user_id,
                            'product_description': product_description,
                            'category': category,
                            'predicted_themes': review.get('predicted_themes', []),  # Use predicted_themes as ground truth
                            'cluster_id': cluster_id,
                            'micro_id': micro_id,
                            'persona_name': persona_name
                        }
                        total_reviews += 1
                        
            except Exception as e:
                logger.warning(f"Error loading {micro_file}: {e}")
                continue
    
    logger.info(f"Loaded {total_reviews} ground truth reviews")
    logger.info(f"Created {len(ground_truth)} unique match keys")
    return ground_truth


def calculate_recall_for_review(
    review: Dict[str, Any],
    ground_truth: Dict[str, Dict[str, Any]],
    debug: bool = False
) -> Tuple[Optional[float], Optional[float], Optional[int], Optional[int], Optional[int]]:
    """
    Calculate Recall@max(3,k) and Recall@max(4,k) for a single review.
    
    Uses predicted_themes from ground truth as the ground truth themes.
    
    Args:
        review: Review dictionary from delta file
        ground_truth: Dictionary of ground truth reviews keyed by match_key
        debug: Whether to print debug information
        
    Returns:
        Tuple of (recall_at_max3k, recall_at_max4k, k_actual, top_k_used_3, top_k_used_4)
    """
    user_id = review.get('user_id', '')
    product_description = review.get('product_description', '')
    category = review.get('category', '')
    
    if not user_id or not product_description or not category:
        if debug:
            logger.debug(f"Missing fields: user_id={bool(user_id)}, product_description={bool(product_description)}, category={bool(category)}")
        return None, None, None, None, None
    
    # Create match key
    match_key = create_match_key(user_id, product_description, category)
    
    # Find ground truth
    gt_review = ground_truth.get(match_key)
    if not gt_review:
        if debug:
            logger.debug(f"No ground truth match for key: {match_key[:100]}...")
        return None, None, None, None, None
    
    # Get predicted themes from delta file (predictions)
    pred_obj = review.get('prediction', {})
    pred_themes = pred_obj.get('predicted_themes', {})
    
    if not pred_themes:
        return None, None, None, None, None
    
    # Get predicted_themes from ground truth (actual ground truth themes)
    gt_predicted_themes = gt_review.get('predicted_themes', [])
    
    # Convert to set if it's a list
    if isinstance(gt_predicted_themes, list):
        gt_themes_set = set(gt_predicted_themes)
    elif isinstance(gt_predicted_themes, dict):
        # If it's a dict, use the keys
        gt_themes_set = set(gt_predicted_themes.keys())
    else:
        gt_themes_set = set()
    
    if not gt_themes_set:
        return None, None, None, None, None
    
    # Calculate recall
    recall_at_max3k, k_actual, top_k_used_3 = calculate_recall_at_max3k(
        pred_themes,
        gt_themes_set
    )
    
    recall_at_max4k, _, top_k_used_4 = calculate_recall_at_max4k(
        pred_themes,
        gt_themes_set
    )
    
    return recall_at_max3k, recall_at_max4k, k_actual, top_k_used_3, top_k_used_4


def process_delta_file(
    file_path: Path,
    ground_truth: Dict[str, Dict[str, Any]]
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Process a single delta file and calculate recall metrics.
    
    Uses predicted_themes from ground truth as the ground truth themes.
    
    Args:
        file_path: Path to delta file
        ground_truth: Dictionary of ground truth reviews
        
    Returns:
        Tuple of (updated_data, statistics)
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        logger.error(f"Error reading {file_path}: {e}")
        return None, None
    
    deltas = data.get('deltas', [])
    stats = {
        'file_path': str(file_path),
        'total_reviews': len(deltas),
        'reviews_with_recall': 0,
        'reviews_without_recall': 0,
        'failure_reasons': defaultdict(int)
    }
    
    for review in deltas:
        # Check why recall might fail
        user_id = review.get('user_id', '')
        product_description = review.get('product_description', '')
        category = review.get('category', '')
        pred_obj = review.get('prediction', {})
        pred_themes = pred_obj.get('predicted_themes', {})
        
        # Track failure reasons
        if not user_id or not product_description or not category:
            missing_fields = []
            if not user_id:
                missing_fields.append('user_id')
            if not product_description:
                missing_fields.append('product_description')
            if not category:
                missing_fields.append('category')
            stats['failure_reasons'][f'missing_fields: {", ".join(missing_fields)}'] += 1
        elif not pred_themes:
            stats['failure_reasons']['empty_predicted_themes'] += 1
        else:
            # Create match key to check if ground truth exists
            match_key = create_match_key(user_id, product_description, category)
            if match_key not in ground_truth:
                stats['failure_reasons']['no_ground_truth_match'] += 1
            else:
                gt_review = ground_truth[match_key]
                gt_predicted_themes = gt_review.get('predicted_themes', [])
                if not gt_predicted_themes:
                    stats['failure_reasons']['missing_predicted_themes_in_ground_truth'] += 1
                else:
                    # Should have recall - track as unknown failure
                    stats['failure_reasons']['unknown_failure'] += 1
        
        recall_at_max3k, recall_at_max4k, k_actual, top_k_used_3, top_k_used_4 = calculate_recall_for_review(
            review,
            ground_truth,
            debug=False
        )
        
        # Update metrics in review
        if 'metrics' not in review:
            review['metrics'] = {}
        
        if recall_at_max3k is not None:
            review['metrics']['recall_at_max3k'] = float(recall_at_max3k)
            review['metrics']['recall_at_max4k'] = float(recall_at_max4k)
            review['metrics']['k_actual'] = k_actual
            review['metrics']['top_k_used_3'] = top_k_used_3
            review['metrics']['top_k_used_4'] = top_k_used_4
            stats['reviews_with_recall'] += 1
        else:
            stats['reviews_without_recall'] += 1
    
    return data, stats


def process_all_delta_files(
    deltas_dir: Path,
    ground_truth_dir: Path
) -> Dict[str, Any]:
    """
    Process all delta files and calculate recall metrics.
    
    Uses predicted_themes from ground truth as the ground truth themes.
    
    Args:
        deltas_dir: Path to deltas directory
        ground_truth_dir: Path to ground truth directory
        
    Returns:
        Dictionary with overall statistics
    """
    logger.info(f"Processing delta files from: {deltas_dir}")
    logger.info(f"Using predicted_themes from ground truth as ground truth themes")
    
    # Load ground truth
    ground_truth = load_ground_truth(ground_truth_dir)
    
    # Find all delta files
    delta_files = list(deltas_dir.rglob("micro_*_all_reviews_deltas.json"))
    logger.info(f"Found {len(delta_files)} delta files")
    
    overall_stats = {
        'total_files': len(delta_files),
        'total_reviews': 0,
        'reviews_with_recall': 0,
        'reviews_without_recall': 0,
        'file_stats': [],
        'failure_reasons': defaultdict(int)
    }
    
    recall_values_3 = []
    recall_values_4 = []
    k_values = []
    
    # Process each file
    for delta_file in tqdm(delta_files, desc="Processing delta files"):
        data, stats = process_delta_file(delta_file, ground_truth)
        
        if data is None:
            continue
        
        # Save updated file
        try:
            with open(delta_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Error saving {delta_file}: {e}")
            continue
        
        overall_stats['file_stats'].append(stats)
        overall_stats['total_reviews'] += stats['total_reviews']
        overall_stats['reviews_with_recall'] += stats['reviews_with_recall']
        overall_stats['reviews_without_recall'] += stats['reviews_without_recall']
        
        # Aggregate failure reasons
        for reason, count in stats.get('failure_reasons', {}).items():
            overall_stats['failure_reasons'][reason] += count
        
        # Collect recall values for statistics - ONLY include reviews with valid recall (not None)
        deltas = data.get('deltas', [])
        for review in deltas:
            metrics = review.get('metrics', {})
            
            # Only include reviews that have valid recall values (explicitly check for not None)
            recall_3 = metrics.get('recall_at_max3k')
            recall_4 = metrics.get('recall_at_max4k')
            k_actual = metrics.get('k_actual')
            
            # Only add to statistics if recall values are not None
            # This ensures reviews without recall are excluded from average calculations
            if recall_3 is not None:
                recall_values_3.append(recall_3)
            if recall_4 is not None:
                recall_values_4.append(recall_4)
            if k_actual is not None:
                k_values.append(k_actual)
    
    # Calculate overall statistics
    if recall_values_3:
        import numpy as np
        overall_stats['recall_at_max3k'] = {
            'mean': float(np.mean(recall_values_3)),
            'median': float(np.median(recall_values_3)),
            'std': float(np.std(recall_values_3)),
            'min': float(np.min(recall_values_3)),
            'max': float(np.max(recall_values_3)),
            'count': len(recall_values_3)
        }
    
    if recall_values_4:
        import numpy as np
        overall_stats['recall_at_max4k'] = {
            'mean': float(np.mean(recall_values_4)),
            'median': float(np.median(recall_values_4)),
            'std': float(np.std(recall_values_4)),
            'min': float(np.min(recall_values_4)),
            'max': float(np.max(recall_values_4)),
            'count': len(recall_values_4)
        }
    
    if k_values:
        import numpy as np
        overall_stats['k_actual'] = {
            'mean': float(np.mean(k_values)),
            'median': float(np.median(k_values)),
            'std': float(np.std(k_values)),
            'min': float(np.min(k_values)),
            'max': float(np.max(k_values)),
            'count': len(k_values)
        }
    
    return overall_stats


def main():
    """Main execution function."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Calculate Recall@max(3,k) and Recall@max(4,k) for delta files',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('--deltas-dir', type=str,
                       default=str(BASE_DIR / "07_sgo_training" / "artifacts" / "sgo_train_final_predictions_refined_chars" / "deltas"),
                       help='Path to deltas directory')
    parser.add_argument('--ground-truth-dir', type=str,
                       default=str(BASE_DIR / "ground_truth_test_micro_cluster_details"),
                       help='Path to ground truth directory')
    parser.add_argument('--output-summary', type=str,
                       help='Path to save summary statistics JSON file')
    
    args = parser.parse_args()
    
    deltas_dir = Path(args.deltas_dir)
    ground_truth_dir = Path(args.ground_truth_dir)
    
    if not deltas_dir.exists():
        logger.error(f"Deltas directory not found: {deltas_dir}")
        return
    
    if not ground_truth_dir.exists():
        logger.error(f"Ground truth directory not found: {ground_truth_dir}")
        return
    
    logger.info("=" * 80)
    logger.info("Recall Calculation for Delta Files")
    logger.info("=" * 80)
    logger.info(f"Deltas directory: {deltas_dir}")
    logger.info(f"Ground truth directory: {ground_truth_dir}")
    logger.info(f"Using predicted_themes from ground truth as ground truth themes")
    logger.info("=" * 80)
    
    # Process all delta files
    overall_stats = process_all_delta_files(deltas_dir, ground_truth_dir)
    
    # Print summary
    logger.info("\n" + "=" * 80)
    logger.info("SUMMARY STATISTICS")
    logger.info("=" * 80)
    logger.info(f"Total files processed: {overall_stats['total_files']}")
    logger.info(f"Total reviews: {overall_stats['total_reviews']:,}")
    logger.info(f"Reviews with recall: {overall_stats['reviews_with_recall']:,} ({100 * overall_stats['reviews_with_recall'] / overall_stats['total_reviews']:.2f}%)")
    logger.info(f"Reviews without recall: {overall_stats['reviews_without_recall']:,} ({100 * overall_stats['reviews_without_recall'] / overall_stats['total_reviews']:.2f}%)")
    
    if overall_stats['reviews_without_recall'] > 0 and overall_stats['failure_reasons']:
        logger.info(f"\nFailure reasons for reviews without recall:")
        for reason, count in sorted(overall_stats['failure_reasons'].items(), key=lambda x: -x[1]):
            logger.info(f"  {reason}: {count:,} ({100 * count / overall_stats['reviews_without_recall']:.2f}%)")
    
    if 'recall_at_max3k' in overall_stats:
        r3 = overall_stats['recall_at_max3k']
        logger.info(f"\nRecall@max(3,k):")
        logger.info(f"  Mean: {r3['mean']:.6f}")
        logger.info(f"  Median: {r3['median']:.6f}")
        logger.info(f"  Std: {r3['std']:.6f}")
        logger.info(f"  Min: {r3['min']:.6f}")
        logger.info(f"  Max: {r3['max']:.6f}")
        logger.info(f"  Count: {r3['count']:,}")
    
    if 'recall_at_max4k' in overall_stats:
        r4 = overall_stats['recall_at_max4k']
        logger.info(f"\nRecall@max(4,k):")
        logger.info(f"  Mean: {r4['mean']:.6f}")
        logger.info(f"  Median: {r4['median']:.6f}")
        logger.info(f"  Std: {r4['std']:.6f}")
        logger.info(f"  Min: {r4['min']:.6f}")
        logger.info(f"  Max: {r4['max']:.6f}")
        logger.info(f"  Count: {r4['count']:,}")
    
    if 'k_actual' in overall_stats:
        k = overall_stats['k_actual']
        logger.info(f"\nK (actual themes):")
        logger.info(f"  Mean: {k['mean']:.2f}")
        logger.info(f"  Median: {k['median']:.2f}")
        logger.info(f"  Std: {k['std']:.2f}")
        logger.info(f"  Min: {k['min']}")
        logger.info(f"  Max: {k['max']}")
        logger.info(f"  Count: {k['count']:,}")
    
    logger.info("=" * 80)
    
    # Save summary if requested
    if args.output_summary:
        summary_file = Path(args.output_summary)
        summary_file.parent.mkdir(parents=True, exist_ok=True)
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(overall_stats, f, indent=2, ensure_ascii=False)
        logger.info(f"\nSaved summary statistics to: {summary_file}")
    else:
        # Save to default location
        summary_file = deltas_dir.parent / "recall_calculation_summary.json"
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(overall_stats, f, indent=2, ensure_ascii=False)
        logger.info(f"\nSaved summary statistics to: {summary_file}")


if __name__ == "__main__":
    main()

