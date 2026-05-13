#!/usr/bin/env python3
"""
Calculate Recall@max(3,k) and Recall@max(4,k) for test_summary_enhanced_persona_micro_cluster_accuracy.json files.
Uses ground truth from ground_truth_test_micro_cluster_details_converted.
"""

import json
import numpy as np
import sys
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict

# Add project root to path
BASE_DIR = Path(__file__).parent.parent.parent
sys.path.insert(0, str(BASE_DIR))

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def get_ground_truth_themes(actual: Dict[str, Any], threshold: float = 0.5) -> set:
    """
    Extract ground truth themes from topic_probabilities_before_normalisation using threshold.
    
    Priority (no fallbacks):
    1. topic_probabilities_before_normalisation (British spelling)
    2. topic_probs_before_normalization (American spelling)
    3. topic_probabilities_without_normalisation
    
    Args:
        actual: Actual review dictionary containing topic probability fields
        threshold: Minimum prob_yes to include theme (default: 0.5)
        
    Returns:
        Set of theme names that are in ground truth (prob_yes >= threshold)
    """
    # Priority 1: topic_probabilities_before_normalisation (British spelling)
    topic_probs_before = actual.get('topic_probabilities_before_normalisation', {})
    
    # Priority 2: topic_probs_before_normalization (American spelling)
    if not topic_probs_before:
        topic_probs_before = actual.get('topic_probs_before_normalization', {})
    
    # Priority 3: topic_probabilities_without_normalisation
    if not topic_probs_before:
        topic_probs_before = actual.get('topic_probabilities_without_normalisation', {})
    
    if not topic_probs_before:
        return set()
    
    if isinstance(topic_probs_before, list):
        return set(topic_probs_before)
    
    ground_truth_themes = set()
    for theme, prob_data in topic_probs_before.items():
        if isinstance(prob_data, dict):
            # Handle nested dict format: {"prob_yes": float, "prob_no": float}
            prob_yes = prob_data.get('prob_yes')
            if prob_yes is not None and isinstance(prob_yes, (int, float)) and prob_yes >= threshold:
                ground_truth_themes.add(theme)
        elif isinstance(prob_data, (int, float)):
            # Handle direct probability value (before normalization)
            if not np.isnan(prob_data) and np.isfinite(prob_data) and prob_data >= threshold:
                ground_truth_themes.add(theme)
    
    return ground_truth_themes


def get_top_k_predicted(predicted_themes: Dict[str, float], k: int) -> List[str]:
    """Get top-k predicted themes sorted by probability."""
    if not predicted_themes:
        return []
    
    if isinstance(predicted_themes, list):
        return predicted_themes[:k]
    
    sorted_themes = sorted(
        predicted_themes.items(),
        key=lambda x: x[1],
        reverse=True
    )
    return [theme for theme, _ in sorted_themes[:k]]


def calculate_recall_at_k(
    predicted_themes: Dict[str, float],
    ground_truth_themes: set,
    k: int
) -> float:
    """
    Calculate Recall@k.
    
    Recall@k = |top_k_predicted ∩ ground_truth| / |ground_truth|
    """
    if not ground_truth_themes:
        return 0.0
    
    top_k_predicted = set(get_top_k_predicted(predicted_themes, k))
    intersection = top_k_predicted & ground_truth_themes
    
    recall = len(intersection) / len(ground_truth_themes)
    return recall


def calculate_recall_at_max3k(
    predicted_themes: Dict[str, float],
    ground_truth_themes: set
) -> Tuple[float, int, int]:
    """Calculate Recall@max(3, k) where k is the number of ground truth themes."""
    if not ground_truth_themes:
        return 0.0, 0, 0
    
    k = len(ground_truth_themes)
    top_k = max(3, k)
    
    recall = calculate_recall_at_k(predicted_themes, ground_truth_themes, top_k)
    return recall, k, top_k


def calculate_recall_at_max4k(
    predicted_themes: Dict[str, float],
    ground_truth_themes: set
) -> Tuple[float, int, int]:
    """Calculate Recall@max(4, k) where k is the number of ground truth themes."""
    if not ground_truth_themes:
        return 0.0, 0, 0
    
    k = len(ground_truth_themes)
    top_k = max(4, k)
    
    recall = calculate_recall_at_k(predicted_themes, ground_truth_themes, top_k)
    return recall, k, top_k


def load_ground_truth_file(cluster_id: str, micro_id: str) -> Dict[str, Dict[str, Any]]:
    """Load ground truth file from ground_truth_test_micro_cluster_details_converted."""
    possible_gt_dirs = [
        BASE_DIR / "ground_truth_test_micro_cluster_details_converted",
        Path("ground_truth_test_micro_cluster_details_converted"),
        Path("/media/samanvitha/Seagate HDD/vectorial_sapiens_v4/ground_truth_test_micro_cluster_details_converted"),
    ]
    
    ground_truth_file = None
    for gt_dir in possible_gt_dirs:
        if gt_dir.exists():
            gt_file = gt_dir / cluster_id / f"{micro_id}_details.json"
            if gt_file.exists():
                ground_truth_file = gt_file
                break
    
    if not ground_truth_file:
        logger.warning(f"Ground truth file not found for {cluster_id}/{micro_id}")
        return {}
    
    try:
        with open(ground_truth_file, 'r', encoding='utf-8') as f:
            gt_data = json.load(f)
        
        lookup = {}
        
        # Handle members_grouped_by_user structure - user_id is the key
        if 'members_grouped_by_user' in gt_data and isinstance(gt_data['members_grouped_by_user'], dict):
            for user_id_key, user_reviews in gt_data['members_grouped_by_user'].items():
                if isinstance(user_reviews, list):
                    for review in user_reviews:
                        user_id = user_id_key
                        product_description = review.get('product_description', '').strip()
                        
                        # Primary key: user_id + product_description
                        if user_id and product_description:
                            key = f"{user_id}_{product_description}"
                            lookup[key] = review
        
        logger.info(f"Loaded {len(lookup)} ground truth reviews from {ground_truth_file}")
        return lookup
    except Exception as e:
        logger.warning(f"Error loading ground truth file {ground_truth_file}: {e}")
        return {}


def process_test_summary_file(file_path: Path, threshold: float = 0.5) -> Dict[str, Any]:
    """Process a test_summary_enhanced_persona_micro_cluster_accuracy.json file and calculate recall metrics."""
    logger.info(f"Processing: {file_path}")
    
    # Extract cluster and micro IDs from file path
    cluster_match = None
    micro_match = None
    
    # Try to extract from path
    path_str = str(file_path)
    import re
    cluster_match = re.search(r'cluster_(\d+)', path_str)
    micro_match = re.search(r'micro_(\d+)', path_str)
    
    cluster_id = f"cluster_{cluster_match.group(1)}" if cluster_match else None
    micro_id = f"micro_{micro_match.group(1)}" if micro_match else None
    
    if not cluster_id or not micro_id:
        logger.error(f"Could not extract cluster_id and micro_id from {file_path}")
        return {}
    
    # Load ground truth file
    ground_truth_lookup = load_ground_truth_file(cluster_id, micro_id)
    
    # Load test summary file
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        logger.error(f"Error reading file {file_path}: {e}")
        return {}
    
    user_predictions = data.get('user_predictions', {})
    if not user_predictions:
        logger.warning(f"No user_predictions found in {file_path}")
        return {}
    
    # Collections for aggregation
    all_recall_3k = []
    all_recall_4k = []
    total_reviews = 0
    processed_reviews = 0
    
    # Process each review
    for user_id, reviews in user_predictions.items():
        if not isinstance(reviews, list):
            continue
        
        for review in reviews:
            total_reviews += 1
            
            prediction = review.get('prediction', {})
            predicted_themes = prediction.get('predicted_themes', {})
            
            if not predicted_themes:
                continue
            
            # Match with ground truth
            product_description = review.get('product_description', '').strip()
            match_key = f"{user_id}_{product_description}"
            
            actual_for_recall = None
            if ground_truth_lookup and match_key in ground_truth_lookup:
                actual_for_recall = ground_truth_lookup[match_key]
            else:
                # Try to use actual from review if available
                actual_for_recall = review.get('actual', {})
            
            if not actual_for_recall:
                continue
            
            # Get ground truth themes
            gt_themes = get_ground_truth_themes(actual_for_recall, threshold=threshold)
            
            if not gt_themes:
                continue
            
            # Calculate recall metrics
            recall_3k, k_actual, top_k_3 = calculate_recall_at_max3k(predicted_themes, gt_themes)
            recall_4k, k_actual, top_k_4 = calculate_recall_at_max4k(predicted_themes, gt_themes)
            
            # Store in review metrics if metrics dict exists
            if 'metrics' not in review:
                review['metrics'] = {}
            
            review['metrics']['recall_at_max3k'] = recall_3k
            review['metrics']['recall_at_max4k'] = recall_4k
            review['metrics']['k_actual'] = k_actual
            review['metrics']['top_k_3'] = top_k_3
            review['metrics']['top_k_4'] = top_k_4
            
            # Collect for aggregation
            if not np.isnan(recall_3k):
                all_recall_3k.append(recall_3k)
            if not np.isnan(recall_4k):
                all_recall_4k.append(recall_4k)
            
            processed_reviews += 1
    
    # Calculate statistics
    def calculate_stats(values):
        if values:
            return {
                'mean': float(np.mean(values)),
                'std': float(np.std(values)),
                'median': float(np.median(values)),
                'min': float(np.min(values)),
                'max': float(np.max(values)),
                'count': len(values)
            }
        return None
    
    metrics = {
        'file_path': str(file_path),
        'cluster_id': cluster_id,
        'micro_id': micro_id,
        'total_reviews': total_reviews,
        'processed_reviews': processed_reviews,
        'threshold_used': threshold,
        'recall_3k': calculate_stats(all_recall_3k),
        'recall_4k': calculate_stats(all_recall_4k),
    }
    
    # Save updated file
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        logger.info(f"Updated file: {file_path}")
    except Exception as e:
        logger.warning(f"Error saving file {file_path}: {e}")
    
    return metrics


def main():
    """Main execution function."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Calculate Recall@max(3,k) and Recall@max(4,k) for test_summary files')
    parser.add_argument('--files', type=str, nargs='+',
                       default=[
                           '07_sgo_training/artifacts/sgo_train_final_predictions_refined_chars/cluster_4/micro_13_test_summary_enhanced_persona_micro_cluster_accuracy.json',
                           '07_sgo_training/artifacts/sgo_train_final_predictions_refined_chars/cluster_4/micro_15_test_summary_enhanced_persona_micro_cluster_accuracy.json'
                       ],
                       help='Paths to test_summary files to process')
    parser.add_argument('--threshold', type=float, default=0.5,
                       help='Threshold for ground truth theme extraction (default: 0.5)')
    
    args = parser.parse_args()
    
    logger.info("=" * 80)
    logger.info("CALCULATING RECALL@MAX(3,K) AND RECALL@MAX(4,K) FOR TEST SUMMARY FILES")
    logger.info("=" * 80)
    logger.info(f"Threshold: {args.threshold}")
    
    all_results = []
    
    for file_path_str in args.files:
        file_path = Path(file_path_str)
        if not file_path.is_absolute():
            file_path = BASE_DIR / file_path_str
        
        if not file_path.exists():
            logger.error(f"File not found: {file_path}")
            continue
        
        result = process_test_summary_file(file_path, threshold=args.threshold)
        if result:
            all_results.append(result)
    
    # Calculate overall statistics across all tribes
    all_recall_3k_values = []
    all_recall_4k_values = []
    total_reviews_all = 0
    processed_reviews_all = 0
    
    for result in all_results:
        # Load individual values from files for overall calculation
        file_path = Path(result['file_path'])
        if not file_path.is_absolute():
            file_path = BASE_DIR / result['file_path']
        
        if file_path.exists():
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            user_predictions = data.get('user_predictions', {})
            for user_id, reviews in user_predictions.items():
                if not isinstance(reviews, list):
                    continue
                for review in reviews:
                    metrics = review.get('metrics', {})
                    recall_3k = metrics.get('recall_at_max3k')
                    recall_4k = metrics.get('recall_at_max4k')
                    
                    if recall_3k is not None and not np.isnan(recall_3k):
                        all_recall_3k_values.append(recall_3k)
                    if recall_4k is not None and not np.isnan(recall_4k):
                        all_recall_4k_values.append(recall_4k)
        
        total_reviews_all += result['total_reviews']
        processed_reviews_all += result['processed_reviews']
    
    # Calculate overall statistics
    def calculate_stats(values):
        if values:
            return {
                'mean': float(np.mean(values)),
                'std': float(np.std(values)),
                'median': float(np.median(values)),
                'min': float(np.min(values)),
                'max': float(np.max(values)),
                'count': len(values)
            }
        return None
    
    overall_stats = {
        'total_reviews': total_reviews_all,
        'processed_reviews': processed_reviews_all,
        'threshold_used': args.threshold,
        'recall_3k': calculate_stats(all_recall_3k_values),
        'recall_4k': calculate_stats(all_recall_4k_values),
    }
    
    # Create comprehensive analysis JSON
    analysis = {
        'threshold_used': args.threshold,
        'tribes': all_results,
        'overall': overall_stats
    }
    
    # Save analysis to JSON file
    output_file = BASE_DIR / "07_sgo_training" / "artifacts" / "recall_analysis_micro_13_micro_15.json"
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(analysis, f, indent=2, ensure_ascii=False)
    
    logger.info(f"\nSaved analysis to: {output_file}")
    
    # Print summary
    logger.info("\n" + "=" * 80)
    logger.info("SUMMARY")
    logger.info("=" * 80)
    
    for result in all_results:
        logger.info(f"\n{result['cluster_id']}/{result['micro_id']}:")
        logger.info(f"  Total reviews: {result['total_reviews']}")
        logger.info(f"  Processed reviews: {result['processed_reviews']}")
        if result['recall_3k']:
            logger.info(f"  Recall@max(3,k): Mean = {result['recall_3k']['mean']:.6f}, Median = {result['recall_3k']['median']:.6f}, Count = {result['recall_3k']['count']}")
        if result['recall_4k']:
            logger.info(f"  Recall@max(4,k): Mean = {result['recall_4k']['mean']:.6f}, Median = {result['recall_4k']['median']:.6f}, Count = {result['recall_4k']['count']}")
    
    logger.info(f"\nOVERALL (Both Tribes Combined):")
    logger.info(f"  Total reviews: {overall_stats['total_reviews']}")
    logger.info(f"  Processed reviews: {overall_stats['processed_reviews']}")
    if overall_stats['recall_3k']:
        logger.info(f"  Recall@max(3,k): Mean = {overall_stats['recall_3k']['mean']:.6f}, Median = {overall_stats['recall_3k']['median']:.6f}, Count = {overall_stats['recall_3k']['count']}")
    if overall_stats['recall_4k']:
        logger.info(f"  Recall@max(4,k): Mean = {overall_stats['recall_4k']['mean']:.6f}, Median = {overall_stats['recall_4k']['median']:.6f}, Count = {overall_stats['recall_4k']['count']}")
    
    logger.info("=" * 80)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

