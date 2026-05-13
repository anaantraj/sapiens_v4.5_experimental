#!/usr/bin/env python3
"""
Map Ground Truth Theme Names from Set B to Set A
=================================================

This script maps theme names in ground truth files from Set B to Set A for:
- topic_logprobs
- topic_probabilities_before_normalisation
- topic_probabilities
- predicted_themes

It processes both:
- ground_truth_train_micro_cluster_details_converted
- ground_truth_test_micro_cluster_details

Usage:
    python Metrics and analysis/scripts/map_ground_truth_setb_to_seta.py
"""

import json
import sys
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Any
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
    from map_predicted_themes_setb_to_seta import (
        map_themes_setb_to_seta,
        get_category_mapping_key,
        THEME_MAPPINGS
    )
except ImportError:
    logger.error("Could not import theme mapping functions. Make sure map_predicted_themes_setb_to_seta.py exists.")
    sys.exit(1)


def map_logprobs_setb_to_seta(
    topic_logprobs: Dict[str, Dict[str, float]],
    category: str
) -> Dict[str, Dict[str, float]]:
    """
    Map topic_logprobs from Set B to Set A.
    
    When multiple Set B themes map to the same Set A theme, we need to merge the logprobs.
    For logprobs, we use log-sum-exp trick: log(exp(a) + exp(b)) = max(a,b) + log(1 + exp(-|a-b|))
    But for simplicity, we'll take the maximum logprob_yes and logprob_no separately.
    
    Args:
        topic_logprobs: Dictionary of theme -> {logprob_yes: float, logprob_no: float}
        category: Category name from the review
        
    Returns:
        Dictionary of mapped theme names (Set A) to logprobs
    """
    if not topic_logprobs:
        return {}
    
    # Get the mapping for this category
    mapping_key = get_category_mapping_key(category)
    if not mapping_key or mapping_key not in THEME_MAPPINGS:
        # No mapping found, return as-is
        return topic_logprobs
    
    theme_mapping = THEME_MAPPINGS[mapping_key]
    
    # Create new dictionary with mapped themes
    # When multiple Set B themes map to same Set A theme, merge logprobs
    mapped_logprobs = defaultdict(lambda: {'logprob_yes': [], 'logprob_no': []})
    unmapped_logprobs = {}
    
    for theme_b, logprobs in topic_logprobs.items():
        if not isinstance(logprobs, dict):
            continue
        
        if theme_b in theme_mapping:
            # Map to Set A theme
            theme_a = theme_mapping[theme_b]
            mapped_logprobs[theme_a]['logprob_yes'].append(logprobs.get('logprob_yes', float('-inf')))
            mapped_logprobs[theme_a]['logprob_no'].append(logprobs.get('logprob_no', float('-inf')))
        else:
            # Theme not in mapping - keep as-is (might already be Set A)
            unmapped_logprobs[theme_b] = logprobs
    
    # Merge logprobs for themes that were mapped from multiple Set B themes
    # Use log-sum-exp: log(exp(a) + exp(b)) ≈ max(a,b) + log(1 + exp(-|a-b|))
    # For simplicity, we'll use max for now (or we could implement proper log-sum-exp)
    import math
    
    final_mapped = {}
    for theme_a, logprob_lists in mapped_logprobs.items():
        if logprob_lists['logprob_yes']:
            # Use maximum logprob_yes (most positive, least negative)
            logprob_yes = max(logprob_lists['logprob_yes'])
        else:
            logprob_yes = float('-inf')
        
        if logprob_lists['logprob_no']:
            # Use maximum logprob_no
            logprob_no = max(logprob_lists['logprob_no'])
        else:
            logprob_no = float('-inf')
        
        final_mapped[theme_a] = {
            'logprob_yes': logprob_yes,
            'logprob_no': logprob_no
        }
    
    # Add unmapped themes (they might already be Set A)
    final_mapped.update(unmapped_logprobs)
    
    return final_mapped


def map_probabilities_before_normalization_setb_to_seta(
    topic_probs_before_norm: Dict[str, float],
    category: str
) -> Dict[str, float]:
    """
    Map topic_probabilities_before_normalisation from Set B to Set A.
    
    When multiple Set B themes map to the same Set A theme, sum their probabilities.
    
    Args:
        topic_probs_before_norm: Dictionary of theme -> probability
        category: Category name from the review
        
    Returns:
        Dictionary of mapped theme names (Set A) to probabilities (summed if multiple Set B themes map to same Set A)
    """
    if not topic_probs_before_norm:
        return {}
    
    # Use the existing mapping function which handles probability summation
    return map_themes_setb_to_seta(topic_probs_before_norm, category)


def map_predicted_themes_list_setb_to_seta(
    predicted_themes: List[str],
    category: str
) -> List[str]:
    """
    Map predicted_themes list from Set B to Set A.
    
    Args:
        predicted_themes: List of theme names (Set B)
        category: Category name from the review
        
    Returns:
        List of mapped theme names (Set A), preserving order and removing duplicates
    """
    if not predicted_themes:
        return []
    
    # Get the mapping for this category
    mapping_key = get_category_mapping_key(category)
    if not mapping_key or mapping_key not in THEME_MAPPINGS:
        # No mapping found, return as-is
        return predicted_themes
    
    theme_mapping = THEME_MAPPINGS[mapping_key]
    
    # Map each theme
    mapped_themes = []
    seen_set_a_themes = set()
    
    for theme_b in predicted_themes:
        if theme_b in theme_mapping:
            # Map to Set A theme
            theme_a = theme_mapping[theme_b]
            # Only add if we haven't seen it before (avoid duplicates)
            if theme_a not in seen_set_a_themes:
                mapped_themes.append(theme_a)
                seen_set_a_themes.add(theme_a)
        else:
            # Theme not in mapping - keep as-is (might already be Set A)
            if theme_b not in seen_set_a_themes:
                mapped_themes.append(theme_b)
                seen_set_a_themes.add(theme_b)
    
    return mapped_themes


def process_ground_truth_file(file_path: Path) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Process a single ground truth file and map theme names.
    
    Args:
        file_path: Path to the ground truth file
        
    Returns:
        Tuple of (updated_data, statistics)
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        logger.error(f"Error reading {file_path}: {e}")
        return None, None
    
    stats = {
        'file_path': str(file_path),
        'total_reviews': 0,
        'reviews_mapped_logprobs': 0,
        'reviews_mapped_probs_before_norm': 0,
        'reviews_mapped_topic_probabilities': 0,
        'reviews_mapped_predicted_themes': 0,
        'categories_found': set()
    }
    
    # Extract members_grouped_by_user
    members_grouped = data.get('members_grouped_by_user', {})
    
    for user_id, reviews in members_grouped.items():
        if not isinstance(reviews, list):
            continue
        
        for review in reviews:
            if not isinstance(review, dict):
                continue
            
            stats['total_reviews'] += 1
            category = review.get('category', '')
            # Also check main_category if category doesn't match
            main_category = review.get('main_category', '')
            # Use main_category if available, otherwise use category
            mapping_category = main_category if main_category else category
            if category:
                stats['categories_found'].add(category)
            if main_category:
                stats['categories_found'].add(f"main_category: {main_category}")
            
            # Map topic_logprobs
            topic_logprobs = review.get('topic_logprobs', {})
            if topic_logprobs:
                original_logprobs_themes = set(topic_logprobs.keys())
                mapped_logprobs = map_logprobs_setb_to_seta(topic_logprobs, mapping_category)
                mapped_logprobs_themes = set(mapped_logprobs.keys())
                
                if original_logprobs_themes != mapped_logprobs_themes:
                    stats['reviews_mapped_logprobs'] += 1
                    review['topic_logprobs'] = mapped_logprobs
            
            # Map topic_probabilities_before_normalisation
            topic_probs_before_norm = review.get('topic_probabilities_before_normalisation', {})
            if topic_probs_before_norm:
                original_probs_themes = set(topic_probs_before_norm.keys())
                mapped_probs = map_probabilities_before_normalization_setb_to_seta(
                    topic_probs_before_norm,
                    mapping_category
                )
                mapped_probs_themes = set(mapped_probs.keys())
                
                if original_probs_themes != mapped_probs_themes:
                    stats['reviews_mapped_probs_before_norm'] += 1
                    review['topic_probabilities_before_normalisation'] = mapped_probs
            
            # Map topic_probabilities (normalized probabilities)
            topic_probs = review.get('topic_probabilities', {})
            if topic_probs:
                original_topic_probs_themes = set(topic_probs.keys())
                mapped_topic_probs = map_probabilities_before_normalization_setb_to_seta(
                    topic_probs,
                    mapping_category
                )
                mapped_topic_probs_themes = set(mapped_topic_probs.keys())
                
                if original_topic_probs_themes != mapped_topic_probs_themes:
                    stats['reviews_mapped_topic_probabilities'] = stats.get('reviews_mapped_topic_probabilities', 0) + 1
                    review['topic_probabilities'] = mapped_topic_probs
            
            # Map predicted_themes (list)
            predicted_themes = review.get('predicted_themes', [])
            if predicted_themes and isinstance(predicted_themes, list):
                original_predicted_themes = predicted_themes.copy()
                mapped_predicted_themes = map_predicted_themes_list_setb_to_seta(
                    predicted_themes,
                    mapping_category
                )
                
                if original_predicted_themes != mapped_predicted_themes:
                    stats['reviews_mapped_predicted_themes'] += 1
                    review['predicted_themes'] = mapped_predicted_themes
    
    stats['categories_found'] = list(stats['categories_found'])
    
    return data, stats


def process_ground_truth_directory(gt_dir: Path) -> Dict[str, Any]:
    """
    Process all ground truth files in a directory.
    
    Args:
        gt_dir: Path to ground truth directory
        
    Returns:
        Dictionary with overall statistics
    """
    logger.info(f"Processing ground truth directory: {gt_dir}")
    
    overall_stats = {
        'total_files': 0,
        'total_reviews': 0,
        'reviews_mapped_logprobs': 0,
        'reviews_mapped_probs_before_norm': 0,
        'reviews_mapped_topic_probabilities': 0,
        'reviews_mapped_predicted_themes': 0,
        'file_stats': []
    }
    
    # Find all cluster directories
    cluster_dirs = sorted([d for d in gt_dir.iterdir() if d.is_dir() and d.name.startswith('cluster_')])
    logger.info(f"Found {len(cluster_dirs)} cluster directories")
    
    # Find all micro detail files
    micro_files = []
    for cluster_dir in cluster_dirs:
        micro_files.extend(cluster_dir.glob("micro_*_details.json"))
    
    logger.info(f"Found {len(micro_files)} micro detail files")
    
    # Process each file
    for micro_file in tqdm(micro_files, desc="Processing files"):
        data, stats = process_ground_truth_file(micro_file)
        
        if data is None:
            continue
        
        # Save updated file
        try:
            with open(micro_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=4, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Error saving {micro_file}: {e}")
            continue
        
        overall_stats['total_files'] += 1
        overall_stats['file_stats'].append(stats)
        overall_stats['total_reviews'] += stats['total_reviews']
        overall_stats['reviews_mapped_logprobs'] += stats['reviews_mapped_logprobs']
        overall_stats['reviews_mapped_probs_before_norm'] += stats['reviews_mapped_probs_before_norm']
        overall_stats['reviews_mapped_topic_probabilities'] += stats.get('reviews_mapped_topic_probabilities', 0)
        overall_stats['reviews_mapped_predicted_themes'] += stats['reviews_mapped_predicted_themes']
    
    return overall_stats


def main():
    """Main execution function."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Map ground truth theme names from Set B to Set A',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('--train-dir', type=str,
                       default=str(BASE_DIR / "ground_truth_train_micro_cluster_details_converted"),
                       help='Path to train ground truth directory')
    parser.add_argument('--test-dir', type=str,
                       default=str(BASE_DIR / "ground_truth_test_micro_cluster_details"),
                       help='Path to test ground truth directory')
    parser.add_argument('--train-only', action='store_true',
                       help='Only process train directory')
    parser.add_argument('--test-only', action='store_true',
                       help='Only process test directory')
    
    args = parser.parse_args()
    
    logger.info("=" * 80)
    logger.info("Ground Truth Theme Mapping: Set B → Set A")
    logger.info("=" * 80)
    
    # Process train directory
    if not args.test_only:
        train_dir = Path(args.train_dir)
        if train_dir.exists():
            logger.info(f"\nProcessing TRAIN directory: {train_dir}")
            train_stats = process_ground_truth_directory(train_dir)
            
            logger.info("\n" + "=" * 80)
            logger.info("TRAIN DIRECTORY STATISTICS")
            logger.info("=" * 80)
            logger.info(f"Total files processed: {train_stats['total_files']}")
            logger.info(f"Total reviews: {train_stats['total_reviews']:,}")
            logger.info(f"Reviews with mapped logprobs: {train_stats['reviews_mapped_logprobs']:,}")
            logger.info(f"Reviews with mapped probs_before_norm: {train_stats['reviews_mapped_probs_before_norm']:,}")
            logger.info(f"Reviews with mapped topic_probabilities: {train_stats['reviews_mapped_topic_probabilities']:,}")
            logger.info(f"Reviews with mapped predicted_themes: {train_stats['reviews_mapped_predicted_themes']:,}")
        else:
            logger.warning(f"Train directory not found: {train_dir}")
    
    # Process test directory
    if not args.train_only:
        test_dir = Path(args.test_dir)
        if test_dir.exists():
            logger.info(f"\nProcessing TEST directory: {test_dir}")
            test_stats = process_ground_truth_directory(test_dir)
            
            logger.info("\n" + "=" * 80)
            logger.info("TEST DIRECTORY STATISTICS")
            logger.info("=" * 80)
            logger.info(f"Total files processed: {test_stats['total_files']}")
            logger.info(f"Total reviews: {test_stats['total_reviews']:,}")
            logger.info(f"Reviews with mapped logprobs: {test_stats['reviews_mapped_logprobs']:,}")
            logger.info(f"Reviews with mapped probs_before_norm: {test_stats['reviews_mapped_probs_before_norm']:,}")
            logger.info(f"Reviews with mapped topic_probabilities: {test_stats['reviews_mapped_topic_probabilities']:,}")
            logger.info(f"Reviews with mapped predicted_themes: {test_stats['reviews_mapped_predicted_themes']:,}")
        else:
            logger.warning(f"Test directory not found: {test_dir}")
    
    logger.info("\n" + "=" * 80)
    logger.info("Mapping complete!")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()

