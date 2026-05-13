#!/usr/bin/env python3
"""
Analyze micro clusters to find those with:
1. Higher number of reviews
2. Better improvement metrics (JSD, WD, Recall improvements)
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional
import argparse
import logging
import re

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_micro_cluster_metrics(file_path: Path) -> Optional[Dict[str, Any]]:
    """Load tribe_metrics from a micro cluster file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        tribe_metrics = data.get('tribe_metrics', {})
        if not tribe_metrics:
            return None
        
        # Extract cluster and micro IDs
        cluster_match = re.search(r'cluster_(\d+)', str(file_path))
        micro_match = re.search(r'micro_(\d+)', str(file_path))
        cluster_id = f"cluster_{cluster_match.group(1)}" if cluster_match else "unknown"
        micro_id = f"micro_{micro_match.group(1)}" if micro_match else "unknown"
        
        return {
            'cluster_id': cluster_id,
            'micro_id': micro_id,
            'file_path': str(file_path),
            'total_reviews': tribe_metrics.get('total_reviews', 0),
            'processed_reviews': tribe_metrics.get('processed_reviews', 0),
            'threshold_used': tribe_metrics.get('threshold_used', 0.3),
            'initial_jsd': tribe_metrics.get('initial_jsd', {}),
            'initial_wd': tribe_metrics.get('initial_wd', {}),
            'jsd': tribe_metrics.get('jsd', {}),  # Post-SGO
            'wd': tribe_metrics.get('wd', {}),  # Post-SGO
            'best_jsd': tribe_metrics.get('best_jsd', {}),
            'best_wd': tribe_metrics.get('best_wd', {}),
            'improvement_in_jsd': tribe_metrics.get('improvement_in_jsd', {}),
            'improvement_in_wd': tribe_metrics.get('improvement_in_wd', {}),
            'improvement_in_recall_3k': tribe_metrics.get('improvement_in_recall_3k', {}),
            'improvement_in_recall_4k': tribe_metrics.get('improvement_in_recall_4k', {}),
            'initial_recall_3k': tribe_metrics.get('initial_recall_3k', {}),
            'initial_recall_4k': tribe_metrics.get('initial_recall_4k', {}),
            'recall_3k': tribe_metrics.get('recall_3k', {}),  # Post-SGO
            'recall_4k': tribe_metrics.get('recall_4k', {}),  # Post-SGO
        }
    except Exception as e:
        logger.warning(f"Error loading {file_path}: {e}")
        return None


def calculate_composite_score(metrics: Dict[str, Any]) -> float:
    """
    Calculate a composite improvement score.
    Higher is better.
    """
    score = 0.0
    weight = 0.0
    
    # Improvement in JSD (percentage improvement, positive = better, higher is better)
    if metrics.get('improvement_in_jsd') and metrics['improvement_in_jsd'].get('mean') is not None:
        jsd_improvement = metrics['improvement_in_jsd']['mean']
        if not np.isnan(jsd_improvement):
            score += jsd_improvement * 0.3  # Weight: 30%
            weight += 0.3
    
    # Improvement in WD (percentage improvement, positive = better, higher is better)
    if metrics.get('improvement_in_wd') and metrics['improvement_in_wd'].get('mean') is not None:
        wd_improvement = metrics['improvement_in_wd']['mean']
        if not np.isnan(wd_improvement):
            score += wd_improvement * 0.3  # Weight: 30%
            weight += 0.3
    
    # Improvement in Recall@max(3,k) (percentage improvement, positive = better, higher is better)
    if metrics.get('improvement_in_recall_3k') and metrics['improvement_in_recall_3k'].get('mean') is not None:
        recall_3k_improvement = metrics['improvement_in_recall_3k']['mean']
        if not np.isnan(recall_3k_improvement):
            score += recall_3k_improvement * 0.2  # Weight: 20%
            weight += 0.2
    
    # Improvement in Recall@max(4,k) (percentage improvement, positive = better, higher is better)
    if metrics.get('improvement_in_recall_4k') and metrics['improvement_in_recall_4k'].get('mean') is not None:
        recall_4k_improvement = metrics['improvement_in_recall_4k']['mean']
        if not np.isnan(recall_4k_improvement):
            score += recall_4k_improvement * 0.2  # Weight: 20%
            weight += 0.2
    
    # Normalize by weight
    if weight > 0:
        return score / weight
    return 0.0


def analyze_all_micro_clusters(results_dir: Path) -> List[Dict[str, Any]]:
    """Analyze all micro cluster files and return sorted results."""
    all_metrics = []
    
    # Find all micro cluster summary files
    for cluster_dir in sorted(results_dir.glob('cluster_*')):
        if not cluster_dir.is_dir():
            continue
        
        for summary_file in sorted(cluster_dir.glob('micro_*_summary_*.json')):
            metrics = load_micro_cluster_metrics(summary_file)
            if metrics:
                # Calculate composite score
                metrics['composite_score'] = calculate_composite_score(metrics)
                all_metrics.append(metrics)
    
    return all_metrics


def main():
    parser = argparse.ArgumentParser(
        description='Analyze micro clusters by review count and improvement metrics',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('--results-dir', type=str, default=None,
                       help='Path to sgo_training_results directory (default: script directory)')
    parser.add_argument('--min-reviews', type=int, default=10,
                       help='Minimum number of reviews to consider (default: 10)')
    parser.add_argument('--top-n', type=int, default=20,
                       help='Number of top micro clusters to show (default: 20)')
    parser.add_argument('--sort-by', type=str, default='composite',
                       choices=['composite', 'reviews', 'jsd_improvement', 'wd_improvement', 
                               'recall_3k_improvement', 'recall_4k_improvement'],
                       help='Sort by: composite score, review count, or specific improvement metric')
    parser.add_argument('--output', type=str, default=None,
                       help='Output JSON file path (optional)')
    
    args = parser.parse_args()
    
    # Determine results directory
    if args.results_dir:
        results_dir = Path(args.results_dir)
    else:
        results_dir = Path(__file__).parent.parent / "artifacts" / "sgo_training_results_v6"
    
    if not results_dir.exists():
        logger.error(f"ERROR: Results directory not found: {results_dir}")
        return 1
    
    logger.info("=" * 80)
    logger.info("ANALYZING MICRO CLUSTERS BY REVIEW COUNT AND IMPROVEMENT METRICS")
    logger.info("=" * 80)
    logger.info(f"Results directory: {results_dir}")
    logger.info(f"Minimum reviews: {args.min_reviews}")
    logger.info(f"Sort by: {args.sort_by}")
    
    # Load all metrics
    logger.info("Loading micro cluster metrics...")
    all_metrics = analyze_all_micro_clusters(results_dir)
    logger.info(f"Loaded metrics from {len(all_metrics)} micro clusters")
    
    # Filter by minimum reviews
    filtered_metrics = [
        m for m in all_metrics 
        if m['total_reviews'] >= args.min_reviews
    ]
    logger.info(f"Filtered to {len(filtered_metrics)} micro clusters with >= {args.min_reviews} reviews")
    
    # Sort
    if args.sort_by == 'composite':
        filtered_metrics.sort(key=lambda x: (x['composite_score'], x['total_reviews']), reverse=True)
    elif args.sort_by == 'reviews':
        filtered_metrics.sort(key=lambda x: x['total_reviews'], reverse=True)
    elif args.sort_by == 'jsd_improvement':
        filtered_metrics.sort(
            key=lambda x: x['improvement_in_jsd'].get('mean', 0) if x['improvement_in_jsd'] else 0,
            reverse=True
        )
    elif args.sort_by == 'wd_improvement':
        filtered_metrics.sort(
            key=lambda x: x['improvement_in_wd'].get('mean', 0) if x['improvement_in_wd'] else 0,
            reverse=True
        )
    elif args.sort_by == 'recall_3k_improvement':
        filtered_metrics.sort(
            key=lambda x: x['improvement_in_recall_3k'].get('mean', 0) if x['improvement_in_recall_3k'] else 0,
            reverse=True
        )
    elif args.sort_by == 'recall_4k_improvement':
        filtered_metrics.sort(
            key=lambda x: x['improvement_in_recall_4k'].get('mean', 0) if x['improvement_in_recall_4k'] else 0,
            reverse=True
        )
    
    # Display top N
    top_n = filtered_metrics[:args.top_n]
    
    logger.info("\n" + "=" * 80)
    logger.info(f"TOP {len(top_n)} MICRO CLUSTERS (sorted by {args.sort_by})")
    logger.info("=" * 80)
    logger.info(f"{'Rank':<6} {'Cluster':<15} {'Reviews':<10} {'Composite':<12} {'JSD Imp %':<12} {'WD Imp %':<12} {'R@3k Imp %':<12} {'R@4k Imp %':<12}")
    logger.info("-" * 100)
    
    for rank, metrics in enumerate(top_n, 1):
        cluster_micro = f"{metrics['cluster_id']}/{metrics['micro_id']}"
        reviews = metrics['total_reviews']
        composite = metrics['composite_score']
        
        jsd_imp = metrics['improvement_in_jsd'].get('mean', 0) if metrics['improvement_in_jsd'] else 0
        wd_imp = metrics['improvement_in_wd'].get('mean', 0) if metrics['improvement_in_wd'] else 0
        recall_3k_imp = metrics['improvement_in_recall_3k'].get('mean', 0) if metrics['improvement_in_recall_3k'] else 0
        recall_4k_imp = metrics['improvement_in_recall_4k'].get('mean', 0) if metrics['improvement_in_recall_4k'] else 0
        
        logger.info(
            f"{rank:<6} {cluster_micro:<15} {reviews:<10} {composite:<12.6f} "
            f"{jsd_imp:<12.6f} {wd_imp:<12.6f} {recall_3k_imp:<12.6f} {recall_4k_imp:<12.6f}"
        )
    
    # Save to JSON if requested
    if args.output:
        output_data = {
            'sort_by': args.sort_by,
            'min_reviews': args.min_reviews,
            'total_micro_clusters': len(all_metrics),
            'filtered_micro_clusters': len(filtered_metrics),
            'top_n': top_n
        }
        
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"\nResults saved to: {args.output}")
    
    logger.info("=" * 80)
    
    return 0


if __name__ == "__main__":
    exit(main())

