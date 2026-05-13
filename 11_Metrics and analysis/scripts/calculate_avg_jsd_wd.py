#!/usr/bin/env python3
"""
Calculate Average JSD and WD
============================

This script calculates the average JSD and WD across all reviews in delta files.

Usage:
    python Metrics and analysis/scripts/calculate_avg_jsd_wd.py
"""

import json
import numpy as np
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


def calculate_averages(deltas_dir: Path) -> Dict[str, Any]:
    """
    Calculate average JSD and WD across all reviews.
    
    Args:
        deltas_dir: Path to the deltas directory
        
    Returns:
        Dictionary with statistics
    """
    # Find all delta files
    delta_files = list(deltas_dir.rglob("micro_*_all_reviews_deltas.json"))
    logger.info(f"Found {len(delta_files)} delta files to process")
    
    jsd_values = []
    wd_values = []
    
    total_reviews = 0
    reviews_with_jsd = 0
    reviews_with_wd = 0
    
    # Process each file
    for delta_file in tqdm(delta_files, desc="Processing delta files"):
        try:
            with open(delta_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except Exception as e:
            logger.error(f"Error reading file {delta_file}: {e}")
            continue
        
        deltas = data.get('deltas', [])
        
        for review in deltas:
            total_reviews += 1
            metrics = review.get('metrics', {})
            
            jsd = metrics.get('jsd')
            wd = metrics.get('wd')
            
            if jsd is not None:
                jsd_values.append(jsd)
                reviews_with_jsd += 1
            
            if wd is not None:
                wd_values.append(wd)
                reviews_with_wd += 1
    
    # Calculate statistics
    stats = {
        'total_reviews': total_reviews,
        'reviews_with_jsd': reviews_with_jsd,
        'reviews_with_wd': reviews_with_wd,
        'jsd': {},
        'wd': {}
    }
    
    if jsd_values:
        stats['jsd'] = {
            'mean': float(np.mean(jsd_values)),
            'median': float(np.median(jsd_values)),
            'std': float(np.std(jsd_values)),
            'min': float(np.min(jsd_values)),
            'max': float(np.max(jsd_values)),
            'count': len(jsd_values)
        }
    
    if wd_values:
        stats['wd'] = {
            'mean': float(np.mean(wd_values)),
            'median': float(np.median(wd_values)),
            'std': float(np.std(wd_values)),
            'min': float(np.min(wd_values)),
            'max': float(np.max(wd_values)),
            'count': len(wd_values)
        }
    
    return stats


def main():
    """Main execution function."""
    logger.info("=" * 80)
    logger.info("CALCULATING AVERAGE JSD AND WD")
    logger.info("=" * 80)
    
    # Set paths
    deltas_dir = BASE_DIR / "07_sgo_training" / "artifacts" / "sgo_train_final_predictions_refined_chars" / "deltas"
    
    if not deltas_dir.exists():
        logger.error(f"Deltas directory not found: {deltas_dir}")
        return
    
    # Calculate averages
    stats = calculate_averages(deltas_dir)
    
    # Print results
    logger.info("\n" + "=" * 80)
    logger.info("RESULTS")
    logger.info("=" * 80)
    logger.info(f"Total reviews: {stats['total_reviews']}")
    logger.info(f"Reviews with JSD: {stats['reviews_with_jsd']}")
    logger.info(f"Reviews with WD: {stats['reviews_with_wd']}")
    
    if stats['jsd']:
        logger.info("\nJSD Statistics:")
        logger.info(f"  Average JSD: {stats['jsd']['mean']:.6f}")
        logger.info(f"  Median JSD:  {stats['jsd']['median']:.6f}")
        logger.info(f"  Std JSD:     {stats['jsd']['std']:.6f}")
        logger.info(f"  Min JSD:     {stats['jsd']['min']:.6f}")
        logger.info(f"  Max JSD:     {stats['jsd']['max']:.6f}")
    
    if stats['wd']:
        logger.info("\nWD Statistics:")
        logger.info(f"  Average WD:  {stats['wd']['mean']:.6f}")
        logger.info(f"  Median WD:   {stats['wd']['median']:.6f}")
        logger.info(f"  Std WD:      {stats['wd']['std']:.6f}")
        logger.info(f"  Min WD:      {stats['wd']['min']:.6f}")
        logger.info(f"  Max WD:      {stats['wd']['max']:.6f}")
    
    logger.info("\n" + "=" * 80)
    logger.info("CALCULATION COMPLETE")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()


