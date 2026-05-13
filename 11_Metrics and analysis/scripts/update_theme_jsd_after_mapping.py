#!/usr/bin/env python3
"""
Update theme_jsd and theme_delta After Theme Mapping
====================================================

This script updates the theme_jsd and theme_delta fields in delta files
to match the new JSD values calculated after theme mapping from Set B to Set A.

Usage:
    python Metrics and analysis/scripts/update_theme_jsd_after_mapping.py
"""

import json
import sys
import logging
from pathlib import Path
from typing import Dict, List, Any
from tqdm import tqdm

# Add project root to path
BASE_DIR = Path(__file__).parent.parent.parent
sys.path.insert(0, str(BASE_DIR))

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def update_theme_jsd_in_file(file_path: Path) -> Dict[str, Any]:
    """
    Update theme_jsd and theme_delta fields in a delta file.
    
    Args:
        file_path: Path to the delta JSON file
        
    Returns:
        Dictionary with update statistics
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        logger.error(f"Error reading file {file_path}: {e}")
        return None
    
    stats = {
        'file_path': str(file_path),
        'total_reviews': 0,
        'updated_reviews': 0,
        'skipped_reviews': 0
    }
    
    deltas = data.get('deltas', [])
    stats['total_reviews'] = len(deltas)
    
    for review in deltas:
        metrics = review.get('metrics', {})
        new_jsd = metrics.get('jsd')
        
        if new_jsd is not None:
            # Update theme_jsd to match the new JSD value
            old_theme_jsd = review.get('theme_jsd')
            review['theme_jsd'] = float(new_jsd)
            
            # Update theme_delta to match theme_jsd (they should be the same)
            review['theme_delta'] = float(new_jsd)
            
            # Recalculate overall_delta if it exists
            if 'overall_delta' in review and 'delta_weights' in review:
                text_delta = review.get('text_delta', 0.0)
                theme_delta = float(new_jsd)
                weights = review.get('delta_weights', {})
                text_weight = weights.get('text', 0.7)
                theme_weight = weights.get('theme', 0.3)
                
                overall_delta = (text_weight * text_delta) + (theme_weight * theme_delta)
                review['overall_delta'] = float(overall_delta)
            
            stats['updated_reviews'] += 1
            
            if old_theme_jsd is not None and abs(old_theme_jsd - new_jsd) > 0.001:
                logger.debug(f"Updated theme_jsd from {old_theme_jsd:.6f} to {new_jsd:.6f}")
        else:
            stats['skipped_reviews'] += 1
    
    return data, stats


def process_all_delta_files(deltas_dir: Path) -> Dict[str, Any]:
    """
    Process all delta files and update theme_jsd values.
    
    Args:
        deltas_dir: Path to the deltas directory
        
    Returns:
        Dictionary with overall statistics
    """
    # Find all delta files
    delta_files = list(deltas_dir.rglob("micro_*_all_reviews_deltas.json"))
    logger.info(f"Found {len(delta_files)} delta files to process")
    
    overall_stats = {
        'total_files': len(delta_files),
        'total_reviews': 0,
        'updated_reviews': 0,
        'skipped_reviews': 0,
        'file_stats': []
    }
    
    # Process each file
    for delta_file in tqdm(delta_files, desc="Processing delta files"):
        updated_data, stats = update_theme_jsd_in_file(delta_file)
        
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
        overall_stats['total_reviews'] += stats['total_reviews']
        overall_stats['updated_reviews'] += stats['updated_reviews']
        overall_stats['skipped_reviews'] += stats['skipped_reviews']
        overall_stats['file_stats'].append(stats)
    
    return overall_stats


def main():
    """Main execution function."""
    logger.info("=" * 80)
    logger.info("UPDATING THEME_JSD AFTER THEME MAPPING")
    logger.info("=" * 80)
    
    # Set paths
    deltas_dir = BASE_DIR / "07_sgo_training" / "artifacts" / "sgo_train_final_predictions_refined_chars" / "deltas"
    
    if not deltas_dir.exists():
        logger.error(f"Deltas directory not found: {deltas_dir}")
        return
    
    logger.info(f"Processing delta files in: {deltas_dir}")
    
    # Process all delta files
    stats = process_all_delta_files(deltas_dir)
    
    # Save summary statistics
    output_dir = BASE_DIR / "07_sgo_training" / "artifacts" / "sgo_train_final_predictions_refined_chars"
    summary_file = output_dir / "theme_jsd_update_summary.json"
    
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Saved summary statistics to: {summary_file}")
    
    # Print summary
    logger.info("\n" + "=" * 80)
    logger.info("SUMMARY STATISTICS")
    logger.info("=" * 80)
    logger.info(f"Total files processed: {stats['total_files']}")
    logger.info(f"Total reviews: {stats['total_reviews']}")
    logger.info(f"Reviews updated: {stats['updated_reviews']}")
    logger.info(f"Reviews skipped (no JSD metric): {stats['skipped_reviews']}")
    
    if stats['total_reviews'] > 0:
        percentage = (stats['updated_reviews'] / stats['total_reviews'] * 100)
        logger.info(f"Update percentage: {percentage:.2f}%")
    
    logger.info("\n" + "=" * 80)
    logger.info("UPDATE COMPLETE")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()


