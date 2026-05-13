#!/usr/bin/env python3
"""
Analyze JSD Threshold Statistics
=================================

This script analyzes reviews with JSD <= threshold and calculates statistics.

Usage:
    python Metrics and analysis/scripts/analyze_jsd_threshold.py
"""

import json
import numpy as np
import sys
import logging
from pathlib import Path
from typing import Dict, List, Any
from collections import defaultdict
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


def analyze_jsd_threshold(deltas_dir: Path, threshold: float = 0.63) -> Dict[str, Any]:
    """
    Analyze reviews with JSD <= threshold.
    
    Args:
        deltas_dir: Path to the deltas directory
        threshold: JSD threshold value
        
    Returns:
        Dictionary with statistics
    """
    # Find all delta files
    delta_files = list(deltas_dir.rglob("micro_*_all_reviews_deltas.json"))
    logger.info(f"Found {len(delta_files)} delta files to process")
    
    # Statistics
    stats = {
        'threshold': threshold,
        'total_reviews': 0,
        'reviews_with_jsd': 0,
        'reviews_below_threshold': 0,
        'jsd_values_below_threshold': [],
        'jsd_values_all': [],
        'wd_values_below_threshold': [],
        'wd_values_all': [],
        'avg_jsd_below_threshold': None,
        'avg_jsd_all': None,
        'avg_wd_below_threshold': None,
        'avg_wd_all': None,
        'min_jsd': None,
        'max_jsd': None,
        'min_wd': None,
        'max_wd': None,
        'median_jsd_below_threshold': None,
        'median_jsd_all': None,
        'median_wd_below_threshold': None,
        'median_wd_all': None,
        'std_jsd_below_threshold': None,
        'std_wd_below_threshold': None,
        'by_cluster': defaultdict(lambda: {
            'total_reviews': 0,
            'reviews_with_jsd': 0,
            'reviews_below_threshold': 0,
            'jsd_values_below_threshold': [],
            'wd_values_below_threshold': [],
            'avg_jsd_below_threshold': None,
            'avg_wd_below_threshold': None
        }),
        'by_micro_cluster': defaultdict(lambda: {
            'total_reviews': 0,
            'reviews_with_jsd': 0,
            'reviews_below_threshold': 0,
            'jsd_values_below_threshold': [],
            'wd_values_below_threshold': [],
            'avg_jsd_below_threshold': None,
            'avg_wd_below_threshold': None
        })
    }
    
    # Process each file
    for delta_file in tqdm(delta_files, desc="Processing delta files"):
        try:
            with open(delta_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except Exception as e:
            logger.error(f"Error reading file {delta_file}: {e}")
            continue
        
        cluster_id = data.get('cluster_id', 'unknown')
        micro_cluster_id = data.get('micro_cluster_id', 'unknown')
        
        deltas = data.get('deltas', [])
        
        for review in deltas:
            stats['total_reviews'] += 1
            stats['by_cluster'][cluster_id]['total_reviews'] += 1
            stats['by_micro_cluster'][micro_cluster_id]['total_reviews'] += 1
            
            metrics = review.get('metrics', {})
            jsd = metrics.get('jsd')
            wd = metrics.get('wd')
            
            if jsd is not None:
                stats['reviews_with_jsd'] += 1
                stats['by_cluster'][cluster_id]['reviews_with_jsd'] += 1
                stats['by_micro_cluster'][micro_cluster_id]['reviews_with_jsd'] += 1
                
                stats['jsd_values_all'].append(jsd)
                
                if wd is not None:
                    stats['wd_values_all'].append(wd)
                
                if jsd <= threshold:
                    stats['reviews_below_threshold'] += 1
                    stats['by_cluster'][cluster_id]['reviews_below_threshold'] += 1
                    stats['by_micro_cluster'][micro_cluster_id]['reviews_below_threshold'] += 1
                    
                    stats['jsd_values_below_threshold'].append(jsd)
                    stats['by_cluster'][cluster_id]['jsd_values_below_threshold'].append(jsd)
                    stats['by_micro_cluster'][micro_cluster_id]['jsd_values_below_threshold'].append(jsd)
                    
                    if wd is not None:
                        stats['wd_values_below_threshold'].append(wd)
                        stats['by_cluster'][cluster_id]['wd_values_below_threshold'].append(wd)
                        stats['by_micro_cluster'][micro_cluster_id]['wd_values_below_threshold'].append(wd)
    
    # Calculate statistics
    if stats['jsd_values_below_threshold']:
        stats['avg_jsd_below_threshold'] = float(np.mean(stats['jsd_values_below_threshold']))
        stats['median_jsd_below_threshold'] = float(np.median(stats['jsd_values_below_threshold']))
        stats['min_jsd'] = float(np.min(stats['jsd_values_below_threshold']))
        stats['max_jsd'] = float(np.max(stats['jsd_values_below_threshold']))
        stats['std_jsd_below_threshold'] = float(np.std(stats['jsd_values_below_threshold']))
    
    if stats['wd_values_below_threshold']:
        stats['avg_wd_below_threshold'] = float(np.mean(stats['wd_values_below_threshold']))
        stats['median_wd_below_threshold'] = float(np.median(stats['wd_values_below_threshold']))
        stats['min_wd'] = float(np.min(stats['wd_values_below_threshold']))
        stats['max_wd'] = float(np.max(stats['wd_values_below_threshold']))
        stats['std_wd_below_threshold'] = float(np.std(stats['wd_values_below_threshold']))
    
    if stats['jsd_values_all']:
        stats['avg_jsd_all'] = float(np.mean(stats['jsd_values_all']))
        stats['median_jsd_all'] = float(np.median(stats['jsd_values_all']))
    
    if stats['wd_values_all']:
        stats['avg_wd_all'] = float(np.mean(stats['wd_values_all']))
        stats['median_wd_all'] = float(np.median(stats['wd_values_all']))
    
    # Calculate cluster-level statistics
    for cluster_id, cluster_data in stats['by_cluster'].items():
        if cluster_data['jsd_values_below_threshold']:
            cluster_data['avg_jsd_below_threshold'] = float(np.mean(cluster_data['jsd_values_below_threshold']))
            cluster_data['percentage_below_threshold'] = (cluster_data['reviews_below_threshold'] / cluster_data['reviews_with_jsd'] * 100) if cluster_data['reviews_with_jsd'] > 0 else 0
        if cluster_data['wd_values_below_threshold']:
            cluster_data['avg_wd_below_threshold'] = float(np.mean(cluster_data['wd_values_below_threshold']))
    
    # Calculate micro-cluster-level statistics
    for micro_id, micro_data in stats['by_micro_cluster'].items():
        if micro_data['jsd_values_below_threshold']:
            micro_data['avg_jsd_below_threshold'] = float(np.mean(micro_data['jsd_values_below_threshold']))
            micro_data['percentage_below_threshold'] = (micro_data['reviews_below_threshold'] / micro_data['reviews_with_jsd'] * 100) if micro_data['reviews_with_jsd'] > 0 else 0
        if micro_data['wd_values_below_threshold']:
            micro_data['avg_wd_below_threshold'] = float(np.mean(micro_data['wd_values_below_threshold']))
    
    # Convert defaultdicts to regular dicts for JSON serialization
    stats['by_cluster'] = dict(stats['by_cluster'])
    stats['by_micro_cluster'] = dict(stats['by_micro_cluster'])
    
    return stats


def main():
    """Main execution function."""
    logger.info("=" * 80)
    logger.info("ANALYZING JSD THRESHOLD STATISTICS")
    logger.info("=" * 80)
    
    # Set paths
    deltas_dir = BASE_DIR / "07_sgo_training" / "artifacts" / "sgo_train_final_predictions_refined_chars" / "deltas"
    
    if not deltas_dir.exists():
        logger.error(f"Deltas directory not found: {deltas_dir}")
        return
    
    threshold = 0.63
    logger.info(f"Analyzing reviews with JSD <= {threshold}")
    
    # Analyze
    stats = analyze_jsd_threshold(deltas_dir, threshold)
    
    # Save results
    output_dir = BASE_DIR / "07_sgo_training" / "artifacts" / "sgo_train_final_predictions_refined_chars"
    output_file = output_dir / f"jsd_threshold_{threshold}_analysis.json"
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Saved analysis to: {output_file}")
    
    # Print summary
    logger.info("\n" + "=" * 80)
    logger.info("SUMMARY STATISTICS")
    logger.info("=" * 80)
    logger.info(f"Threshold: JSD <= {threshold}")
    logger.info(f"Total reviews: {stats['total_reviews']}")
    logger.info(f"Reviews with JSD metric: {stats['reviews_with_jsd']}")
    logger.info(f"Reviews with JSD <= {threshold}: {stats['reviews_below_threshold']}")
    
    if stats['reviews_below_threshold'] > 0:
        percentage = (stats['reviews_below_threshold'] / stats['reviews_with_jsd'] * 100) if stats['reviews_with_jsd'] > 0 else 0
        logger.info(f"Percentage of reviews with JSD <= {threshold}: {percentage:.2f}%")
        logger.info(f"\nJSD Statistics for reviews with JSD <= {threshold}:")
        logger.info(f"  Average JSD: {stats['avg_jsd_below_threshold']:.6f}")
        logger.info(f"  Median JSD:  {stats['median_jsd_below_threshold']:.6f}")
        logger.info(f"  Std JSD:      {stats.get('std_jsd_below_threshold', 0):.6f}")
        logger.info(f"  Min JSD:      {stats['min_jsd']:.6f}")
        logger.info(f"  Max JSD:      {stats['max_jsd']:.6f}")
        
        if stats.get('avg_wd_below_threshold') is not None:
            logger.info(f"\nWD Statistics for reviews with JSD <= {threshold}:")
            logger.info(f"  Average WD:  {stats['avg_wd_below_threshold']:.6f}")
            logger.info(f"  Median WD:   {stats['median_wd_below_threshold']:.6f}")
            logger.info(f"  Std WD:      {stats.get('std_wd_below_threshold', 0):.6f}")
            logger.info(f"  Min WD:      {stats['min_wd']:.6f}")
            logger.info(f"  Max WD:      {stats['max_wd']:.6f}")
            logger.info(f"  Count:       {len(stats['wd_values_below_threshold'])} reviews")
    
    if stats['avg_jsd_all']:
        logger.info(f"\nOverall Statistics (all reviews):")
        logger.info(f"  Average JSD: {stats['avg_jsd_all']:.6f}")
        logger.info(f"  Median JSD:  {stats['median_jsd_all']:.6f}")
        if stats.get('avg_wd_all'):
            logger.info(f"  Average WD:  {stats['avg_wd_all']:.6f}")
            logger.info(f"  Median WD:   {stats['median_wd_all']:.6f}")
    
    # Top micro clusters by percentage below threshold
    if stats['by_micro_cluster']:
        logger.info("\n" + "=" * 80)
        logger.info("TOP 10 MICRO CLUSTERS BY PERCENTAGE BELOW THRESHOLD")
        logger.info("=" * 80)
        
        sorted_micro = sorted(
            [(mid, mdata) for mid, mdata in stats['by_micro_cluster'].items() 
             if mdata.get('reviews_below_threshold', 0) > 0],
            key=lambda x: x[1].get('percentage_below_threshold', 0),
            reverse=True
        )[:10]
        
        for micro_id, micro_data in sorted_micro:
            logger.info(f"{micro_id}: {micro_data['reviews_below_threshold']}/{micro_data['reviews_with_jsd']} "
                       f"({micro_data.get('percentage_below_threshold', 0):.1f}%), "
                       f"Avg JSD: {micro_data.get('avg_jsd_below_threshold', 0):.6f}")
    
    logger.info("\n" + "=" * 80)
    logger.info("ANALYSIS COMPLETE")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()

