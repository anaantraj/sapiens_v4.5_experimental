#!/usr/bin/env python3
"""
Merge Ground Truth with BEFM Predictions
==========================================

This script:
1. Loads all prediction files from befm_results_history
2. Matches each prediction with ground truth using (user_id, category, product_description)
3. Merges ground truth data (review_text, predicted_themes, topic_probabilities, etc.) under each prediction
4. Saves updated prediction files

Usage:
    python Metrics and analysis/scripts/merge_ground_truth_with_befm_predictions.py
"""

import json
import sys
import logging
from pathlib import Path
from typing import Dict, Any, Tuple
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


def create_match_key(user_id: str, product_description: str, category: str) -> str:
    """
    Create a unique match key from user_id, product_description, and category.
    
    Args:
        user_id: User ID
        product_description: Product description
        category: Category name
        
    Returns:
        Match key string
    """
    # Normalize by stripping whitespace and converting to lowercase
    normalized_user = user_id.strip().lower() if user_id else ""
    normalized_product = product_description.strip().lower() if product_description else ""
    normalized_category = category.strip().lower() if category else ""
    
    return f"{normalized_user}|||{normalized_product}|||{normalized_category}"


def load_ground_truth(ground_truth_dir: Path) -> Dict[str, Dict[str, Any]]:
    """
    Load ground truth reviews from ground_truth_test_micro_cluster_details.
    
    Args:
        ground_truth_dir: Path to ground truth directory
        
    Returns:
        Dictionary of ground truth reviews keyed by match_key
    """
    logger.info(f"Loading ground truth from: {ground_truth_dir}")
    
    ground_truth = {}
    total_reviews = 0
    match_keys_seen = set()
    
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
                        if not product_description or not category or not user_id:
                            continue
                        
                        # Create match key
                        match_key = create_match_key(user_id, product_description, category)
                        
                        # Skip duplicates (same user, product, category)
                        if match_key in match_keys_seen:
                            continue
                        match_keys_seen.add(match_key)
                        
                        # Store ground truth data
                        ground_truth[match_key] = {
                            'user_id': user_id,
                            'product_description': product_description,
                            'category': category,
                            'review_text': review.get('review_text', ''),
                            'rating': review.get('rating'),
                            'sentiment': review.get('sentiment'),
                            'predicted_themes': review.get('predicted_themes', []),
                            'topic_probabilities': review.get('topic_probabilities', {}),
                            'topic_probabilities_before_normalisation': review.get('topic_probabilities_before_normalisation', {}),
                            'topic_logprobs': review.get('topic_logprobs', {}),
                            'cluster_id': cluster_id,
                            'micro_id': micro_id,
                            'persona_name': persona_name,
                            'cluster': review.get('cluster'),
                            'timestamp': review.get('timestamp'),
                            'asin': review.get('asin'),
                            'main_category': review.get('main_category')
                        }
                        total_reviews += 1
                        
            except Exception as e:
                logger.warning(f"Error loading {micro_file}: {e}")
                continue
    
    logger.info(f"Loaded {total_reviews} ground truth reviews")
    logger.info(f"Created {len(ground_truth)} unique match keys")
    return ground_truth


def process_prediction_file(
    file_path: Path,
    ground_truth: Dict[str, Dict[str, Any]]
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Process a single prediction file and merge ground truth data.
    
    Args:
        file_path: Path to prediction file
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
    
    user_id = data.get('user_id', '')
    if not user_id:
        logger.warning(f"No user_id found in {file_path}")
        return None, None
    
    stats = {
        'file_path': str(file_path),
        'total_predictions': 0,
        'matched_predictions': 0,
        'unmatched_predictions': 0
    }
    
    predictions = data.get('predictions', [])
    stats['total_predictions'] = len(predictions)
    
    # Process each prediction
    for pred in predictions:
        product_description = pred.get('product_description', '')
        original_category = pred.get('original_category', '')
        
        if not product_description or not original_category:
            stats['unmatched_predictions'] += 1
            continue
        
        # Create match key
        match_key = create_match_key(user_id, product_description, original_category)
        
        # Find ground truth
        gt_review = ground_truth.get(match_key)
        
        if gt_review:
            # Merge ground truth data under the prediction
            pred['ground_truth'] = {
                'review_text': gt_review.get('review_text', ''),
                'rating': gt_review.get('rating'),
                'sentiment': gt_review.get('sentiment'),
                'predicted_themes': gt_review.get('predicted_themes', []),
                'topic_probabilities': gt_review.get('topic_probabilities', {}),
                'topic_probabilities_before_normalisation': gt_review.get('topic_probabilities_before_normalisation', {}),
                'topic_logprobs': gt_review.get('topic_logprobs', {}),
                'cluster_id': gt_review.get('cluster_id'),
                'micro_id': gt_review.get('micro_id'),
                'persona_name': gt_review.get('persona_name'),
                'cluster': gt_review.get('cluster'),
                'timestamp': gt_review.get('timestamp'),
                'asin': gt_review.get('asin'),
                'main_category': gt_review.get('main_category')
            }
            stats['matched_predictions'] += 1
        else:
            stats['unmatched_predictions'] += 1
    
    return data, stats


def process_all_prediction_files(
    predictions_dir: Path,
    ground_truth_dir: Path
) -> Dict[str, Any]:
    """
    Process all prediction files and merge ground truth data.
    
    Args:
        predictions_dir: Path to predictions directory
        ground_truth_dir: Path to ground truth directory
        
    Returns:
        Dictionary with overall statistics
    """
    logger.info(f"Processing prediction files from: {predictions_dir}")
    
    # Load ground truth
    ground_truth = load_ground_truth(ground_truth_dir)
    
    # Find all prediction files
    prediction_files = sorted(predictions_dir.glob("prediction_*.json"))
    logger.info(f"Found {len(prediction_files)} prediction files")
    
    overall_stats = {
        'total_files': len(prediction_files),
        'total_predictions': 0,
        'matched_predictions': 0,
        'unmatched_predictions': 0,
        'file_stats': []
    }
    
    # Process each file
    for pred_file in tqdm(prediction_files, desc="Processing prediction files"):
        data, stats = process_prediction_file(pred_file, ground_truth)
        
        if data is None:
            continue
        
        # Save updated file
        try:
            with open(pred_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=4, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Error saving {pred_file}: {e}")
            continue
        
        overall_stats['total_predictions'] += stats['total_predictions']
        overall_stats['matched_predictions'] += stats['matched_predictions']
        overall_stats['unmatched_predictions'] += stats['unmatched_predictions']
        overall_stats['file_stats'].append(stats)
    
    return overall_stats


def main():
    """Main execution function."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Merge ground truth with BEFM predictions',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('--predictions-dir', type=str,
                       default=str(BASE_DIR / "befm_results_history"),
                       help='Path to predictions directory')
    parser.add_argument('--ground-truth-dir', type=str,
                       default=str(BASE_DIR / "ground_truth_train_micro_cluster_details_converted"),
                       help='Path to ground truth directory')
    parser.add_argument('--output-summary', type=str,
                       help='Path to save summary statistics JSON file')
    
    args = parser.parse_args()
    
    predictions_dir = Path(args.predictions_dir)
    ground_truth_dir = Path(args.ground_truth_dir)
    
    if not predictions_dir.exists():
        logger.error(f"Predictions directory not found: {predictions_dir}")
        return
    
    if not ground_truth_dir.exists():
        logger.error(f"Ground truth directory not found: {ground_truth_dir}")
        return
    
    logger.info("=" * 80)
    logger.info("Merge Ground Truth with BEFM Predictions")
    logger.info("=" * 80)
    logger.info(f"Predictions directory: {predictions_dir}")
    logger.info(f"Ground truth directory: {ground_truth_dir}")
    logger.info("=" * 80)
    
    # Process all prediction files
    overall_stats = process_all_prediction_files(predictions_dir, ground_truth_dir)
    
    # Print summary
    logger.info("\n" + "=" * 80)
    logger.info("SUMMARY STATISTICS")
    logger.info("=" * 80)
    logger.info(f"Total files processed: {overall_stats['total_files']}")
    logger.info(f"Total predictions: {overall_stats['total_predictions']:,}")
    logger.info(f"Matched predictions: {overall_stats['matched_predictions']:,} ({100.0 * overall_stats['matched_predictions'] / max(overall_stats['total_predictions'], 1):.2f}%)")
    logger.info(f"Unmatched predictions: {overall_stats['unmatched_predictions']:,} ({100.0 * overall_stats['unmatched_predictions'] / max(overall_stats['total_predictions'], 1):.2f}%)")
    logger.info("=" * 80)
    
    # Save summary if requested
    if args.output_summary:
        summary_file = Path(args.output_summary)
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(overall_stats, f, indent=2, ensure_ascii=False)
        logger.info(f"\nSaved summary statistics to: {summary_file}")
    else:
        # Save to default location
        summary_file = predictions_dir / "ground_truth_merge_summary.json"
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(overall_stats, f, indent=2, ensure_ascii=False)
        logger.info(f"\nSaved summary statistics to: {summary_file}")


if __name__ == "__main__":
    main()

