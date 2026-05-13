#!/usr/bin/env python3
"""
Calculate JSD Summary for Final Predictions
===========================================

This script:
1. Reads delta files from deltas/cluster_X/micro_Y_all_reviews_deltas.json
2. Calculates mean JSD for each micro cluster
3. Calculates mean JSD for each cluster (aggregating all micro clusters)
4. Updates grand_summary_enhanced_persona_micro_cluster.json with JSD metrics

Usage:
    python 07_sgo_training/scripts/calculate_jsd_summary.py
    python 07_sgo_training/scripts/calculate_jsd_summary.py --only-cluster cluster_1
"""

import sys
import logging
import json
import argparse
from pathlib import Path
from collections import defaultdict
import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def load_delta_file(delta_file: Path) -> dict:
    """Load a delta file and return its contents."""
    try:
        with open(delta_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        logging.error(f"Error loading delta file {delta_file}: {e}")
        return None


def calculate_micro_cluster_jsd(delta_data: dict) -> dict:
    """
    Calculate JSD statistics for a micro cluster from delta data.
    
    Returns:
        Dictionary with mean, std, count, min, max, median JSD
    """
    if not delta_data or 'deltas' not in delta_data:
        return None
    
    jsd_values = []
    for delta_entry in delta_data['deltas']:
        # Prefer theme_jsd, fallback to theme_delta
        jsd = delta_entry.get('theme_jsd') or delta_entry.get('theme_delta')
        if jsd is not None:
            jsd_values.append(float(jsd))
    
    if not jsd_values:
        return None
    
    return {
        'mean': float(np.mean(jsd_values)),
        'std': float(np.std(jsd_values)),
        'count': len(jsd_values),
        'min': float(np.min(jsd_values)),
        'max': float(np.max(jsd_values)),
        'median': float(np.median(jsd_values))
    }


def process_cluster_deltas(artifacts_dir: Path, cluster_id: str) -> dict:
    """
    Process all delta files for a cluster and calculate JSD statistics.
    
    Returns:
        Dictionary with:
        - micro_clusters: {micro_id: jsd_stats}
        - cluster_jsd: overall cluster JSD stats
    """
    deltas_dir = artifacts_dir / "deltas" / cluster_id
    
    if not deltas_dir.exists():
        logging.warning(f"  ⚠️  Deltas directory not found: {deltas_dir}")
        return None
    
    # Find all delta files for this cluster
    delta_files = list(deltas_dir.glob("micro_*_all_reviews_deltas.json"))
    
    if not delta_files:
        logging.warning(f"  ⚠️  No delta files found in {deltas_dir}")
        return None
    
    logging.info(f"  Found {len(delta_files)} delta files for {cluster_id}")
    
    micro_cluster_jsd = {}
    all_jsd_values = []
    
    # Process each micro cluster
    for delta_file in sorted(delta_files):
        # Extract micro_id from filename (e.g., "micro_0_all_reviews_deltas.json" -> "micro_0")
        micro_id = delta_file.stem.replace('_all_reviews_deltas', '')
        
        delta_data = load_delta_file(delta_file)
        if not delta_data:
            continue
        
        jsd_stats = calculate_micro_cluster_jsd(delta_data)
        if jsd_stats:
            micro_cluster_jsd[micro_id] = jsd_stats
            # Collect all JSD values for cluster-level aggregation
            for delta_entry in delta_data.get('deltas', []):
                jsd = delta_entry.get('theme_jsd') or delta_entry.get('theme_delta')
                if jsd is not None:
                    all_jsd_values.append(float(jsd))
            
            logging.info(f"    {micro_id}: mean JSD = {jsd_stats['mean']:.6f} (n={jsd_stats['count']})")
    
    # Calculate cluster-level JSD statistics
    cluster_jsd = None
    if all_jsd_values:
        cluster_jsd = {
            'mean': float(np.mean(all_jsd_values)),
            'std': float(np.std(all_jsd_values)),
            'count': len(all_jsd_values),
            'min': float(np.min(all_jsd_values)),
            'max': float(np.max(all_jsd_values)),
            'median': float(np.median(all_jsd_values))
        }
        logging.info(f"  📊 Cluster {cluster_id} overall: mean JSD = {cluster_jsd['mean']:.6f} (n={cluster_jsd['count']})")
    
    return {
        'micro_clusters': micro_cluster_jsd,
        'cluster_jsd': cluster_jsd
    }


def update_grand_summary(artifacts_dir: Path, cluster_id: str, jsd_stats: dict):
    """
    Update grand summary file with JSD statistics.
    
    Args:
        artifacts_dir: Base artifacts directory
        cluster_id: Cluster ID (e.g., "cluster_1")
        jsd_stats: Dictionary with micro_clusters and cluster_jsd
    """
    cluster_dir = artifacts_dir / cluster_id
    grand_summary_file = cluster_dir / "grand_summary_enhanced_persona_micro_cluster.json"
    
    if not grand_summary_file.exists():
        logging.warning(f"  ⚠️  Grand summary file not found: {grand_summary_file}")
        return
    
    # Load existing grand summary
    try:
        with open(grand_summary_file, 'r', encoding='utf-8') as f:
            grand_summary = json.load(f)
    except Exception as e:
        logging.error(f"  ❌ Error loading grand summary: {e}")
        return
    
    # Add JSD statistics to final_summary
    if 'final_summary' not in grand_summary:
        grand_summary['final_summary'] = {}
    
    # Add cluster-level JSD
    if jsd_stats.get('cluster_jsd'):
        grand_summary['final_summary']['theme_jsd'] = jsd_stats['cluster_jsd']
        logging.info(f"  ✅ Added cluster-level JSD to grand summary")
    
    # Add micro cluster JSD breakdown (optional, for reference)
    if jsd_stats.get('micro_clusters'):
        if 'micro_cluster_jsd' not in grand_summary:
            grand_summary['micro_cluster_jsd'] = {}
        grand_summary['micro_cluster_jsd'] = jsd_stats['micro_clusters']
        logging.info(f"  ✅ Added micro cluster JSD breakdown to grand summary")
    
    # Save updated grand summary
    try:
        with open(grand_summary_file, 'w', encoding='utf-8') as f:
            json.dump(grand_summary, f, indent=2, ensure_ascii=False)
        logging.info(f"  💾 Updated grand summary: {grand_summary_file}")
    except Exception as e:
        logging.error(f"  ❌ Error saving grand summary: {e}")


def main():
    parser = argparse.ArgumentParser(description='Calculate JSD summary for final predictions')
    parser.add_argument('--only-cluster', type=str, help='Process only this specific cluster (e.g., cluster_1)')
    parser.add_argument('--artifacts-dir', type=str, help='Path to artifacts directory (default: 07_sgo_training/artifacts/sgo_train_final_predictions)')
    args = parser.parse_args()
    
    # Determine artifacts directory
    if args.artifacts_dir:
        artifacts_dir = Path(args.artifacts_dir)
    else:
        # Default: use the final predictions artifacts directory
        stage_dir = Path(__file__).parent.parent
        artifacts_dir = stage_dir / "artifacts" / "sgo_train_final_predictions"
    
    if not artifacts_dir.exists():
        logging.error(f"❌ Artifacts directory not found: {artifacts_dir}")
        return
    
    logging.info(f"Processing JSD summaries from: {artifacts_dir}")
    
    # Find all clusters
    deltas_base_dir = artifacts_dir / "deltas"
    if not deltas_base_dir.exists():
        logging.error(f"❌ Deltas directory not found: {deltas_base_dir}")
        return
    
    cluster_dirs = [d for d in deltas_base_dir.iterdir() if d.is_dir() and d.name.startswith('cluster_')]
    
    if args.only_cluster:
        cluster_dirs = [d for d in cluster_dirs if d.name == args.only_cluster]
        if not cluster_dirs:
            logging.error(f"❌ Cluster {args.only_cluster} not found in deltas directory")
            return
    
    if not cluster_dirs:
        logging.warning("⚠️  No cluster directories found")
        return
    
    logging.info(f"Found {len(cluster_dirs)} cluster(s) to process")
    
    # Process each cluster
    for cluster_dir in sorted(cluster_dirs):
        cluster_id = cluster_dir.name
        logging.info(f"\n{'='*60}")
        logging.info(f"Processing {cluster_id}")
        logging.info(f"{'='*60}")
        
        # Calculate JSD statistics
        jsd_stats = process_cluster_deltas(artifacts_dir, cluster_id)
        
        if jsd_stats:
            # Update grand summary
            update_grand_summary(artifacts_dir, cluster_id, jsd_stats)
        else:
            logging.warning(f"  ⚠️  No JSD statistics calculated for {cluster_id}")
    
    logging.info("\n" + "="*60)
    logging.info("JSD Summary Calculation Complete")
    logging.info("="*60)


if __name__ == "__main__":
    main()

