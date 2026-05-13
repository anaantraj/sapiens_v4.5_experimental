#!/usr/bin/env python3
"""
Analyze JSD Summary
==================

Analyzes all delta files to count reviews with JSD < 0.35 and calculate their average.
Generates a grand summary file.

Usage:
    python 07_sgo_training/scripts/analyze_jsd_summary.py --cluster cluster_4
    python 07_sgo_training/scripts/analyze_jsd_summary.py --cluster cluster_4 --jsd-threshold 0.35
"""

import sys
import os
import json
from pathlib import Path
import argparse
from collections import defaultdict
import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Simple JSON utilities
def load_json_file(filename, default_value=None):
    if default_value is None:
        default_value = {}
    if not filename or not Path(filename).exists():
        return default_value
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            return json.load(f)
    except (json.JSONDecodeError, FileNotFoundError):
        return default_value

def save_json_file(data, filename):
    directory = Path(filename).parent
    if directory:
        directory.mkdir(parents=True, exist_ok=True)
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def analyze_delta_files(deltas_dir, cluster_id, jsd_threshold=0.35):
    """
    Analyze all delta files for a cluster and generate summary statistics.
    
    Args:
        deltas_dir: Base directory containing delta files
        cluster_id: Cluster ID to analyze (e.g., "cluster_4")
        jsd_threshold: JSD threshold (default: 0.35)
    
    Returns:
        Dictionary with summary statistics
    """
    cluster_dir = Path(deltas_dir) / cluster_id
    if not cluster_dir.exists():
        print(f"❌ Cluster directory not found: {cluster_dir}")
        return None
    
    # Find all delta files
    delta_files = sorted(cluster_dir.glob("micro_*_all_reviews_deltas.json"))
    
    if not delta_files:
        print(f"❌ No delta files found in {cluster_dir}")
        return None
    
    print(f"📊 Analyzing {len(delta_files)} delta files for {cluster_id}...")
    
    # Statistics
    all_reviews = []
    reviews_below_threshold = []
    micro_cluster_stats = []
    
    for delta_file in delta_files:
        print(f"  Processing: {delta_file.name}")
        data = load_json_file(str(delta_file), {})
        
        if not data or 'deltas' not in data:
            print(f"    ⚠️  Skipping {delta_file.name}: no deltas found")
            continue
        
        micro_id = data.get('micro_cluster_id', delta_file.stem.replace('_all_reviews_deltas', ''))
        total_reviews = len(data['deltas'])
        
        # Filter reviews with JSD < threshold
        micro_reviews_below = []
        for review in data['deltas']:
            jsd = review.get('theme_jsd')
            if jsd is not None:
                all_reviews.append({
                    'micro_id': micro_id,
                    'review_key': review.get('review_key', ''),
                    'jsd': jsd,
                    'theme_delta': review.get('theme_delta'),
                    'overall_delta': review.get('overall_delta')
                })
                
                if jsd <= jsd_threshold:
                    reviews_below_threshold.append({
                        'micro_id': micro_id,
                        'review_key': review.get('review_key', ''),
                        'jsd': jsd,
                        'theme_delta': review.get('theme_delta'),
                        'overall_delta': review.get('overall_delta')
                    })
                    micro_reviews_below.append(jsd)
        
        # Micro cluster statistics
        if micro_reviews_below:
            micro_avg_jsd = np.mean(micro_reviews_below)
            micro_cluster_stats.append({
                'micro_id': micro_id,
                'total_reviews': total_reviews,
                'reviews_below_threshold': len(micro_reviews_below),
                'percentage': (len(micro_reviews_below) / total_reviews * 100) if total_reviews > 0 else 0,
                'avg_jsd_below_threshold': micro_avg_jsd,
                'min_jsd': np.min(micro_reviews_below),
                'max_jsd': np.max(micro_reviews_below)
            })
        else:
            micro_cluster_stats.append({
                'micro_id': micro_id,
                'total_reviews': total_reviews,
                'reviews_below_threshold': 0,
                'percentage': 0.0,
                'avg_jsd_below_threshold': None,
                'min_jsd': None,
                'max_jsd': None
            })
    
    # Overall statistics
    total_reviews_count = len(all_reviews)
    reviews_below_count = len(reviews_below_threshold)
    
    if reviews_below_threshold:
        jsd_values = [r['jsd'] for r in reviews_below_threshold]
        avg_jsd_below = np.mean(jsd_values)
        min_jsd_below = np.min(jsd_values)
        max_jsd_below = np.max(jsd_values)
        median_jsd_below = np.median(jsd_values)
        std_jsd_below = np.std(jsd_values)
    else:
        avg_jsd_below = None
        min_jsd_below = None
        max_jsd_below = None
        median_jsd_below = None
        std_jsd_below = None
    
    # Overall JSD statistics (all reviews)
    if all_reviews:
        all_jsd_values = [r['jsd'] for r in all_reviews]
        overall_avg_jsd = np.mean(all_jsd_values)
        overall_min_jsd = np.min(all_jsd_values)
        overall_max_jsd = np.max(all_jsd_values)
        overall_median_jsd = np.median(all_jsd_values)
        overall_std_jsd = np.std(all_jsd_values)
    else:
        overall_avg_jsd = None
        overall_min_jsd = None
        overall_max_jsd = None
        overall_median_jsd = None
        overall_std_jsd = None
    
    summary = {
        'cluster_id': cluster_id,
        'jsd_threshold': jsd_threshold,
        'total_reviews_analyzed': total_reviews_count,
        'reviews_with_jsd_at_or_below_threshold': {
            'count': reviews_below_count,
            'percentage': (reviews_below_count / total_reviews_count * 100) if total_reviews_count > 0 else 0,
            'avg_jsd': float(avg_jsd_below) if avg_jsd_below is not None else None,
            'min_jsd': float(min_jsd_below) if min_jsd_below is not None else None,
            'max_jsd': float(max_jsd_below) if max_jsd_below is not None else None,
            'median_jsd': float(median_jsd_below) if median_jsd_below is not None else None,
            'std_jsd': float(std_jsd_below) if std_jsd_below is not None else None
        },
        'overall_statistics_all_reviews': {
            'avg_jsd': float(overall_avg_jsd) if overall_avg_jsd is not None else None,
            'min_jsd': float(overall_min_jsd) if overall_min_jsd is not None else None,
            'max_jsd': float(overall_max_jsd) if overall_max_jsd is not None else None,
            'median_jsd': float(overall_median_jsd) if overall_median_jsd is not None else None,
            'std_jsd': float(overall_std_jsd) if overall_std_jsd is not None else None
        },
        'micro_cluster_breakdown': micro_cluster_stats,
        'num_micro_clusters': len(micro_cluster_stats)
    }
    
    return summary


def main():
    parser = argparse.ArgumentParser(description='Analyze JSD statistics from delta files')
    parser.add_argument('--cluster', type=str, required=True, help='Cluster ID (e.g., cluster_4)')
    parser.add_argument('--deltas-dir', type=str, 
                       default='07_sgo_training/artifacts/sgo_train_final_predictions_user_seed_memory/deltas',
                       help='Base directory containing delta files')
    parser.add_argument('--jsd-threshold', type=float, default=0.35,
                       help='JSD threshold (default: 0.35)')
    parser.add_argument('--output-dir', type=str,
                       default='07_sgo_training/artifacts/sgo_train_final_predictions_user_seed_memory',
                       help='Output directory for summary file')
    
    args = parser.parse_args()
    
    # Analyze delta files
    summary = analyze_delta_files(args.deltas_dir, args.cluster, args.jsd_threshold)
    
    if summary is None:
        print("❌ Failed to generate summary")
        return 1
    
    # Print summary
    print("\n" + "="*80)
    print("📊 GRAND SUMMARY")
    print("="*80)
    print(f"Cluster: {summary['cluster_id']}")
    print(f"JSD Threshold: {summary['jsd_threshold']}")
    print(f"Total Reviews Analyzed: {summary['total_reviews_analyzed']}")
    print(f"\nReviews with JSD <= {summary['jsd_threshold']}:")
    print(f"  Count: {summary['reviews_with_jsd_at_or_below_threshold']['count']}")
    print(f"  Percentage: {summary['reviews_with_jsd_at_or_below_threshold']['percentage']:.2f}%")
    if summary['reviews_with_jsd_at_or_below_threshold']['avg_jsd'] is not None:
        print(f"  Average JSD: {summary['reviews_with_jsd_at_or_below_threshold']['avg_jsd']:.6f}")
        print(f"  Min JSD: {summary['reviews_with_jsd_at_or_below_threshold']['min_jsd']:.6f}")
        print(f"  Max JSD: {summary['reviews_with_jsd_at_or_below_threshold']['max_jsd']:.6f}")
        print(f"  Median JSD: {summary['reviews_with_jsd_at_or_below_threshold']['median_jsd']:.6f}")
        print(f"  Std Dev JSD: {summary['reviews_with_jsd_at_or_below_threshold']['std_jsd']:.6f}")
    
    print(f"\nOverall Statistics (All Reviews):")
    if summary['overall_statistics_all_reviews']['avg_jsd'] is not None:
        print(f"  Average JSD: {summary['overall_statistics_all_reviews']['avg_jsd']:.6f}")
        print(f"  Min JSD: {summary['overall_statistics_all_reviews']['min_jsd']:.6f}")
        print(f"  Max JSD: {summary['overall_statistics_all_reviews']['max_jsd']:.6f}")
        print(f"  Median JSD: {summary['overall_statistics_all_reviews']['median_jsd']:.6f}")
        print(f"  Std Dev JSD: {summary['overall_statistics_all_reviews']['std_jsd']:.6f}")
    
    print(f"\nMicro Clusters Analyzed: {summary['num_micro_clusters']}")
    print("="*80)
    
    # Save summary to file
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"{args.cluster}_jsd_grand_summary.json"
    
    save_json_file(summary, str(output_file))
    print(f"\n✅ Summary saved to: {output_file}")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())

