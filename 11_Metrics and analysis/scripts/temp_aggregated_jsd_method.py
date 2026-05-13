#!/usr/bin/env python3
"""
Temporary script to test AGGREGATED DISTRIBUTION JSD method
==================================================================
This script:
1. Loads pre_sgo_context artifact
2. Aggregates all actual themes across all reviews
3. Aggregates all predicted themes across all reviews
4. Calculates ONE JSD between the two aggregated distributions
5. Saves results to temp folder

This is different from the main script which calculates per-review JSD.
"""

import json
import numpy as np
import sys
import logging
import re
from pathlib import Path
from typing import Dict, List, Any
from collections import defaultdict
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.spatial.distance import jensenshannon

# Add project root to path
BASE_DIR = Path(__file__).parent.parent.parent
sys.path.insert(0, str(BASE_DIR))

from utils.wandb_utils import (
    init_wandb_run,
    get_stage_config,
    use_artifact,
    finish_run,
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global epsilon for numerical stability
EPSILON = 1e-10


def normalize_distribution(theme_dict: Dict[str, float]) -> Dict[str, float]:
    """Normalize a theme probability distribution to sum to 1.0."""
    if not theme_dict:
        return {}
    total = sum(theme_dict.values())
    if total == 0:
        return {}
    return {theme: prob / total for theme, prob in theme_dict.items()}


def compute_jsd(P: np.ndarray, Q: np.ndarray, epsilon: float = EPSILON) -> float:
    """
    Compute Jensen-Shannon Divergence between two probability distributions.
    
    Uses scipy's jensenshannon (same as the other script's method).
    """
    # Add epsilon to avoid zeros
    P = P + epsilon
    Q = Q + epsilon
    
    # Re-normalize after adding epsilon
    P = P / P.sum()
    Q = Q / Q.sum()
    
    # JSD (scipy returns sqrt, we square it - same as other script)
    jsd = jensenshannon(P, Q, base=2) ** 2
    
    return float(jsd)


def process_artifact_directory(artifact_path: Path, filter_cluster: str = None) -> Dict[str, Any]:
    """
    Process all JSON files and build aggregated distributions at multiple levels.
    
    Args:
        artifact_path: Path to artifact directory
        filter_cluster: Optional cluster ID to filter (e.g., "cluster_1"). If None, processes all clusters.
    
    Returns:
        Dictionary with aggregated distributions at overall, cluster, and tribe levels
    """
    # Overall aggregation
    aggregated_actual = defaultdict(float)
    aggregated_predicted = defaultdict(float)
    
    # Per-cluster aggregation
    cluster_actual = defaultdict(lambda: defaultdict(float))
    cluster_predicted = defaultdict(lambda: defaultdict(float))
    cluster_review_counts = defaultdict(int)
    
    # Per-tribe aggregation
    tribe_actual = defaultdict(lambda: defaultdict(float))
    tribe_predicted = defaultdict(lambda: defaultdict(float))
    tribe_review_counts = defaultdict(int)
    tribe_names_map = {}  # tribe_id -> persona_name
    
    total_reviews = 0
    
    # Find all JSON files recursively
    json_files = list(artifact_path.rglob("*.json"))
    logger.info(f"Found {len(json_files)} JSON files in artifact")
    
    if filter_cluster:
        # Filter files to only include the specified cluster
        json_files = [f for f in json_files if filter_cluster in str(f)]
        logger.info(f"Filtered to {len(json_files)} files for {filter_cluster}")
    
    for json_file in tqdm(json_files, desc="Processing files"):
        try:
            # Extract cluster and micro from path
            file_path_str = str(json_file)
            cluster_id = None
            micro_id = None
            tribe_id = None
            
            cluster_match = re.search(r'cluster_(\d+)', file_path_str)
            micro_match = re.search(r'micro_(\d+)', file_path_str)
            
            if cluster_match:
                cluster_id = f"cluster_{cluster_match.group(1)}"
            if micro_match:
                micro_id = f"micro_{micro_match.group(1)}"
            
            # Skip if cluster filter is set and doesn't match
            if filter_cluster and cluster_id != filter_cluster:
                continue
            
            if cluster_id and micro_id:
                tribe_id = f"{cluster_id}/{micro_id}"
            
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Extract persona_name (tribe name)
            persona_name = data.get('persona_name', None)
            if not persona_name:
                metadata = data.get('metadata', {})
                persona_name = metadata.get('persona_name', None)
            if persona_name and tribe_id:
                tribe_names_map[tribe_id] = persona_name
            
            user_predictions = data.get('user_predictions', {})
            
            for user_id, reviews in user_predictions.items():
                if not isinstance(reviews, list):
                    continue
                
                for review in reviews:
                    prediction = review.get('prediction', {})
                    actual = review.get('actual', {})
                    
                    if not prediction or not actual:
                        continue
                    
                    actual_themes = actual.get('predicted_themes', {})
                    predicted_themes = prediction.get('predicted_themes', {})
                    
                    # Handle list format
                    if isinstance(actual_themes, list):
                        if not actual_themes:
                            continue
                        actual_themes = {theme: 1.0 / len(actual_themes) for theme in actual_themes}
                    
                    if isinstance(predicted_themes, list):
                        if not predicted_themes:
                            continue
                        predicted_themes = {theme: 1.0 / len(predicted_themes) for theme in predicted_themes}
                    
                    if not actual_themes or not predicted_themes:
                        continue
                    
                    # Normalize distributions
                    actual_themes = normalize_distribution(actual_themes)
                    predicted_themes = normalize_distribution(predicted_themes)
                    
                    if not actual_themes or not predicted_themes:
                        continue
                    
                    # Aggregate at all levels
                    total_reviews += 1
                    
                    # Overall aggregation
                    for theme, prob in actual_themes.items():
                        aggregated_actual[theme] += prob
                    for theme, prob in predicted_themes.items():
                        aggregated_predicted[theme] += prob
                    
                    # Cluster aggregation
                    if cluster_id:
                        cluster_review_counts[cluster_id] += 1
                        for theme, prob in actual_themes.items():
                            cluster_actual[cluster_id][theme] += prob
                        for theme, prob in predicted_themes.items():
                            cluster_predicted[cluster_id][theme] += prob
                    
                    # Tribe aggregation
                    if tribe_id:
                        tribe_review_counts[tribe_id] += 1
                        for theme, prob in actual_themes.items():
                            tribe_actual[tribe_id][theme] += prob
                        for theme, prob in predicted_themes.items():
                            tribe_predicted[tribe_id][theme] += prob
                        
        except Exception as e:
            logger.warning(f"Error processing file {json_file}: {e}")
            continue
    
    logger.info(f"Processed {total_reviews} reviews")
    
    # Normalize overall aggregated distributions
    actual_sum = sum(aggregated_actual.values())
    predicted_sum = sum(aggregated_predicted.values())
    
    if actual_sum > 0:
        aggregated_actual = {theme: prob / actual_sum for theme, prob in aggregated_actual.items()}
    if predicted_sum > 0:
        aggregated_predicted = {theme: prob / predicted_sum for theme, prob in aggregated_predicted.items()}
    
    # Normalize cluster distributions
    cluster_actual_normalized = {}
    cluster_predicted_normalized = {}
    for cluster_id in cluster_actual.keys():
        actual_sum = sum(cluster_actual[cluster_id].values())
        predicted_sum = sum(cluster_predicted[cluster_id].values())
        if actual_sum > 0:
            cluster_actual_normalized[cluster_id] = {theme: prob / actual_sum 
                                                    for theme, prob in cluster_actual[cluster_id].items()}
        else:
            cluster_actual_normalized[cluster_id] = {}
        if predicted_sum > 0:
            cluster_predicted_normalized[cluster_id] = {theme: prob / predicted_sum 
                                                       for theme, prob in cluster_predicted[cluster_id].items()}
        else:
            cluster_predicted_normalized[cluster_id] = {}
    
    # Normalize tribe distributions (only for tribes with >= 3 reviews)
    tribe_actual_normalized = {}
    tribe_predicted_normalized = {}
    for tribe_id in tribe_actual.keys():
        if tribe_review_counts[tribe_id] >= 3:  # Minimum threshold
            actual_sum = sum(tribe_actual[tribe_id].values())
            predicted_sum = sum(tribe_predicted[tribe_id].values())
            if actual_sum > 0:
                tribe_actual_normalized[tribe_id] = {theme: prob / actual_sum 
                                                    for theme, prob in tribe_actual[tribe_id].items()}
            else:
                tribe_actual_normalized[tribe_id] = {}
            if predicted_sum > 0:
                tribe_predicted_normalized[tribe_id] = {theme: prob / predicted_sum 
                                                       for theme, prob in tribe_predicted[tribe_id].items()}
            else:
                tribe_predicted_normalized[tribe_id] = {}
    
    return {
        'aggregated_actual': aggregated_actual,
        'aggregated_predicted': aggregated_predicted,
        'cluster_actual': cluster_actual_normalized,
        'cluster_predicted': cluster_predicted_normalized,
        'cluster_review_counts': dict(cluster_review_counts),
        'tribe_actual': tribe_actual_normalized,
        'tribe_predicted': tribe_predicted_normalized,
        'tribe_review_counts': {k: v for k, v in tribe_review_counts.items() if v >= 3},
        'tribe_names_map': tribe_names_map,
        'total_reviews': total_reviews
    }


def calculate_aggregated_jsd(aggregated_actual: Dict[str, float], 
                            aggregated_predicted: Dict[str, float]) -> float:
    """
    Calculate JSD between aggregated distributions using UNION method.
    
    This uses the union of all themes (same as your script), but compares
    aggregated distributions instead of per-review.
    """
    # Get union of all themes (same as your script's method)
    all_themes = sorted(set(aggregated_actual.keys()) | set(aggregated_predicted.keys()))
    
    if not all_themes:
        return 0.0
    
    # Create aligned arrays
    vec1 = np.array([aggregated_actual.get(theme, 0.0) for theme in all_themes])
    vec2 = np.array([aggregated_predicted.get(theme, 0.0) for theme in all_themes])
    
    # Normalize (should already be normalized, but safety check)
    vec1_sum = vec1.sum()
    vec2_sum = vec2.sum()
    if vec1_sum > EPSILON:
        vec1 = vec1 / vec1_sum
    else:
        vec1 = np.zeros_like(vec1)
    if vec2_sum > EPSILON:
        vec2 = vec2 / vec2_sum
    else:
        vec2 = np.zeros_like(vec2)
    
    # Calculate JSD using scipy (same as other script)
    jsd = compute_jsd(vec1, vec2)
    
    return jsd


def create_visualizations(data: Dict[str, Any], output_dir: Path, prefix: str = "pre_sgo_context"):
    """Create all visualizations for aggregated JSD method."""
    import seaborn as sns
    sns.set_style("whitegrid")
    plt.rcParams['figure.dpi'] = 300
    plt.rcParams['savefig.dpi'] = 300
    
    # Overall aggregated distribution visualization
    aggregated_actual = data['aggregated_actual']
    aggregated_predicted = data['aggregated_predicted']
    overall_jsd = calculate_aggregated_jsd(aggregated_actual, aggregated_predicted)
    total_reviews = data['total_reviews']
    
    # Graph 1: Overall aggregated distribution
    actual_sorted = sorted(aggregated_actual.items(), key=lambda x: x[1], reverse=True)
    predicted_sorted = sorted(aggregated_predicted.items(), key=lambda x: x[1], reverse=True)
    
    top_actual = dict(actual_sorted[:25])
    top_predicted = dict(predicted_sorted[:25])
    all_themes_vis = sorted(set(top_actual.keys()) | set(top_predicted.keys()))
    
    fig, ax = plt.subplots(figsize=(max(16, len(all_themes_vis) * 0.5), 8))
    x_pos = np.arange(len(all_themes_vis))
    width = 0.35
    
    actual_values = [top_actual.get(theme, 0.0) for theme in all_themes_vis]
    predicted_values = [top_predicted.get(theme, 0.0) for theme in all_themes_vis]
    
    display_themes = [t[:40] + '...' if len(t) > 40 else t for t in all_themes_vis]
    
    bars1 = ax.barh(x_pos - width/2, actual_values, width, label='Real (Actual)', 
                  alpha=0.8, color='#51CF66', edgecolor='#2F7D32', linewidth=1.2)
    bars2 = ax.barh(x_pos + width/2, predicted_values, width, label='Synthetic (Predicted)', 
                  alpha=0.8, color='#4A7FB5', edgecolor='#2B5A82', linewidth=1.2)
    
    ax.set_yticks(x_pos)
    ax.set_yticklabels(display_themes, fontsize=9)
    ax.set_xlabel('Probability', fontsize=13, fontweight='bold')
    ax.set_title(f'Aggregated Distribution JSD: Real vs Synthetic\n{prefix.replace("_", " ").title()} (n={total_reviews:,} reviews, JSD: {overall_jsd:.4f})', 
                fontsize=14, fontweight='bold', pad=15)
    ax.legend(loc='lower right', fontsize=12, framealpha=0.9)
    ax.grid(True, alpha=0.3, axis='x', linestyle=':', linewidth=0.8)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.invert_yaxis()
    plt.tight_layout()
    
    filename = output_dir / f"aggregated_distribution_{prefix}.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    logger.info(f"Saved: {filename}")
    
    # Graph 2: P, Q, M Comparison (Overall)
    all_themes_pqm = sorted(set(aggregated_actual.keys()) | set(aggregated_predicted.keys()))
    if all_themes_pqm:
        # Calculate mixture M = (P + Q) / 2
        mixture = {}
        for theme in all_themes_pqm:
            p_val = aggregated_actual.get(theme, 0.0)
            q_val = aggregated_predicted.get(theme, 0.0)
            mixture[theme] = (p_val + q_val) / 2.0
        
        # Normalize mixture
        mix_sum = sum(mixture.values())
        if mix_sum > 0:
            mixture = {k: v / mix_sum for k, v in mixture.items()}
        
        # Get top themes for visualization
        top_themes_pqm = sorted(all_themes_pqm, 
                               key=lambda t: max(aggregated_actual.get(t, 0), 
                                                aggregated_predicted.get(t, 0),
                                                mixture.get(t, 0)), 
                               reverse=True)[:30]
        
        fig, axes = plt.subplots(1, 3, figsize=(24, 10), facecolor='white')
        
        x_pos = np.arange(len(top_themes_pqm))
        width = 0.6
        
        # P (Actual) distribution
        p_values = [aggregated_actual.get(theme, 0.0) for theme in top_themes_pqm]
        axes[0].barh(x_pos, p_values, width, alpha=0.85, color='#51CF66', 
                    edgecolor='#2F7D32', linewidth=1.5, label='P (Actual)')
        axes[0].set_yticks(x_pos)
        axes[0].set_yticklabels([t[:35] + '...' if len(t) > 35 else t for t in top_themes_pqm], 
                                fontsize=9)
        axes[0].set_xlabel('Probability', fontsize=12, fontweight='bold')
        axes[0].set_title(f'P (Actual Distribution)\n{len(aggregated_actual)} themes', 
                         fontsize=13, fontweight='bold', pad=10)
        axes[0].grid(True, alpha=0.3, axis='x', linestyle=':', linewidth=0.8)
        axes[0].invert_yaxis()
        axes[0].spines['top'].set_visible(False)
        axes[0].spines['right'].set_visible(False)
        
        # Q (Predicted) distribution
        q_values = [aggregated_predicted.get(theme, 0.0) for theme in top_themes_pqm]
        axes[1].barh(x_pos, q_values, width, alpha=0.85, color='#4A7FB5', 
                    edgecolor='#2B5A82', linewidth=1.5, label='Q (Predicted)')
        axes[1].set_yticks(x_pos)
        axes[1].set_yticklabels([t[:35] + '...' if len(t) > 35 else t for t in top_themes_pqm], 
                                fontsize=9)
        axes[1].set_xlabel('Probability', fontsize=12, fontweight='bold')
        axes[1].set_title(f'Q (Predicted Distribution)\n{len(aggregated_predicted)} themes', 
                         fontsize=13, fontweight='bold', pad=10)
        axes[1].grid(True, alpha=0.3, axis='x', linestyle=':', linewidth=0.8)
        axes[1].invert_yaxis()
        axes[1].spines['top'].set_visible(False)
        axes[1].spines['right'].set_visible(False)
        
        # M (Mixture) distribution
        m_values = [mixture.get(theme, 0.0) for theme in top_themes_pqm]
        axes[2].barh(x_pos, m_values, width, alpha=0.85, color='#FFA726', 
                    edgecolor='#E65100', linewidth=1.5, label='M (Mixture)')
        axes[2].set_yticks(x_pos)
        axes[2].set_yticklabels([t[:35] + '...' if len(t) > 35 else t for t in top_themes_pqm], 
                               fontsize=9)
        axes[2].set_xlabel('Probability', fontsize=12, fontweight='bold')
        axes[2].set_title(f'M (Mixture = (P+Q)/2)\nJSD: {overall_jsd:.4f}', 
                         fontsize=13, fontweight='bold', pad=10)
        axes[2].grid(True, alpha=0.3, axis='x', linestyle=':', linewidth=0.8)
        axes[2].invert_yaxis()
        axes[2].spines['top'].set_visible(False)
        axes[2].spines['right'].set_visible(False)
        
        plt.suptitle(f'P, Q, M Distribution Comparison: {prefix.replace("_", " ").title()}\n(n={total_reviews:,} reviews)', 
                    fontsize=16, fontweight='bold', y=0.98)
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        
        filename = output_dir / f"pqm_comparison_overall_{prefix}.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        logger.info(f"Saved: {filename}")
    
    # Graph 3: P, Q, M Comparison (Per-Cluster)
    cluster_actual = data['cluster_actual']
    cluster_predicted = data['cluster_predicted']
    
    if cluster_actual and cluster_predicted:
        clusters_sorted = sorted(cluster_actual.keys())
        num_clusters = len(clusters_sorted)
        
        if num_clusters > 0:
            cols = min(3, num_clusters)
            rows = int(np.ceil(num_clusters / cols))
            
            fig = plt.figure(figsize=(cols * 10, rows * 12), facecolor='white')
            gs = GridSpec(rows, cols, figure=fig, hspace=0.5, wspace=0.4)
            
            for idx, cluster_id in enumerate(clusters_sorted):
                cluster_num = cluster_id.replace('cluster_', '')
                actual_dist = cluster_actual.get(cluster_id, {})
                predicted_dist = cluster_predicted.get(cluster_id, {})
                
                if not actual_dist or not predicted_dist:
                    continue
                
                # Calculate mixture
                all_themes_cluster = sorted(set(actual_dist.keys()) | set(predicted_dist.keys()))
                mixture_cluster = {}
                for theme in all_themes_cluster:
                    p_val = actual_dist.get(theme, 0.0)
                    q_val = predicted_dist.get(theme, 0.0)
                    mixture_cluster[theme] = (p_val + q_val) / 2.0
                
                mix_sum = sum(mixture_cluster.values())
                if mix_sum > 0:
                    mixture_cluster = {k: v / mix_sum for k, v in mixture_cluster.items()}
                
                # Get top themes
                top_themes_cluster = sorted(all_themes_cluster,
                                          key=lambda t: max(actual_dist.get(t, 0),
                                                           predicted_dist.get(t, 0),
                                                           mixture_cluster.get(t, 0)),
                                          reverse=True)[:20]
                
                if not top_themes_cluster:
                    continue
                
                row = idx // cols
                col = idx % cols
                ax = fig.add_subplot(gs[row, col])
                
                x_pos_cluster = np.arange(len(top_themes_cluster))
                width_cluster = 0.25
                
                p_vals = [actual_dist.get(theme, 0.0) for theme in top_themes_cluster]
                q_vals = [predicted_dist.get(theme, 0.0) for theme in top_themes_cluster]
                m_vals = [mixture_cluster.get(theme, 0.0) for theme in top_themes_cluster]
                
                ax.bar(x_pos_cluster - width_cluster, p_vals, width_cluster, 
                      alpha=0.85, color='#51CF66', edgecolor='#2F7D32', 
                      linewidth=1.2, label='P (Actual)')
                ax.bar(x_pos_cluster, q_vals, width_cluster, 
                      alpha=0.85, color='#4A7FB5', edgecolor='#2B5A82', 
                      linewidth=1.2, label='Q (Predicted)')
                ax.bar(x_pos_cluster + width_cluster, m_vals, width_cluster, 
                      alpha=0.85, color='#FFA726', edgecolor='#E65100', 
                      linewidth=1.2, label='M (Mixture)')
                
                cluster_jsd_val = calculate_aggregated_jsd(actual_dist, predicted_dist)
                cluster_review_count = data['cluster_review_counts'].get(cluster_id, 0)
                
                ax.set_xticks(x_pos_cluster)
                ax.set_xticklabels([t[:15] + '...' if len(t) > 15 else t 
                                   for t in top_themes_cluster], 
                                   fontsize=7, rotation=75, ha='right')
                ax.set_ylabel('Probability', fontsize=11, fontweight='bold')
                ax.set_title(f'Cluster {cluster_num} P, Q, M Comparison\nJSD: {cluster_jsd_val:.4f}, n={cluster_review_count}', 
                           fontsize=12, fontweight='bold', pad=10)
                ax.legend(loc='upper right', fontsize=9, framealpha=0.9)
                ax.grid(True, alpha=0.2, axis='y', linestyle=':', linewidth=0.8)
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                ax.set_facecolor('#fafafa')
            
            plt.suptitle(f'P, Q, M Distribution Comparison by Cluster: {prefix.replace("_", " ").title()}', 
                        fontsize=16, fontweight='bold', y=0.995)
            plt.tight_layout(rect=[0, 0, 1, 0.97])
            
            filename = output_dir / f"pqm_comparison_per_cluster_{prefix}.png"
            plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
            plt.close()
            logger.info(f"Saved: {filename}")
    
    # Graph 4: Per-Tribe Aggregated JSD (grouped by cluster)
    cluster_actual = data['cluster_actual']
    cluster_predicted = data['cluster_predicted']
    tribe_actual = data['tribe_actual']
    tribe_predicted = data['tribe_predicted']
    tribe_review_counts = data['tribe_review_counts']
    tribe_names_map = data['tribe_names_map']
    
    # Calculate JSD per tribe
    tribe_jsd = {}
    cluster_tribe_jsd = defaultdict(dict)
    
    for tribe_id in tribe_actual.keys():
        if tribe_id in tribe_predicted:
            jsd_val = calculate_aggregated_jsd(tribe_actual[tribe_id], tribe_predicted[tribe_id])
            tribe_jsd[tribe_id] = jsd_val
            
            # Group by cluster
            cluster_match = re.search(r'(cluster_\d+)', tribe_id)
            if cluster_match:
                cluster_id = cluster_match.group(1)
                cluster_tribe_jsd[cluster_id][tribe_id] = jsd_val
    
    # Calculate JSD per cluster
    cluster_jsd = {}
    for cluster_id in cluster_actual.keys():
        if cluster_id in cluster_predicted:
            cluster_jsd[cluster_id] = calculate_aggregated_jsd(
                cluster_actual[cluster_id], cluster_predicted[cluster_id]
            )
    
    if cluster_tribe_jsd and len(cluster_tribe_jsd) > 0:
        clusters = sorted(cluster_tribe_jsd.keys())
        num_clusters = len(clusters)
        
        # Determine grid layout
        if num_clusters == 1:
            rows, cols = 1, 1
        elif 'cluster_2' in clusters:
            cols = max(4, num_clusters)
            rows = 2
        else:
            cols = int(np.ceil(np.sqrt(num_clusters)))
            rows = int(np.ceil(num_clusters / cols))
        
        fig = plt.figure(figsize=(cols * 12, rows * 8), facecolor='white')
        gs = GridSpec(rows, cols, figure=fig, hspace=0.8, wspace=0.6)
        
        colors = plt.cm.viridis(np.linspace(0.2, 0.8, 20))
        
        # Calculate average JSD per cluster
        cluster_avg_jsd = {}
        for cluster_id in clusters:
            if cluster_id in cluster_jsd:
                cluster_avg_jsd[cluster_id] = cluster_jsd[cluster_id]
        
        # Plot each cluster
        for idx, cluster_id in enumerate(clusters):
            cluster_tribes_data = cluster_tribe_jsd[cluster_id]
            
            if not cluster_tribes_data:
                continue
            
            # Determine subplot position
            if 'cluster_2' in clusters and cluster_id == 'cluster_2':
                ax = fig.add_subplot(gs[0, 1:3])
            elif 'cluster_2' in clusters:
                cluster_num = int(cluster_id.replace('cluster_', ''))
                if cluster_num == 1:
                    ax = fig.add_subplot(gs[0, 0])
                elif cluster_num == 3:
                    ax = fig.add_subplot(gs[0, 3])
                else:
                    row = 1
                    col = cluster_num - 4
                    if col < 0:
                        col = 0
                    if col >= cols:
                        col = cols - 1
                    ax = fig.add_subplot(gs[row, col])
            else:
                row = idx // cols
                col = idx % cols
                ax = fig.add_subplot(gs[row, col])
            
            # Sort tribes by JSD
            sorted_tribes = sorted(cluster_tribes_data.items(), key=lambda x: x[1])
            tribe_ids = [t for t, _ in sorted_tribes]
            
            # Get tribe names
            tribe_labels = []
            for tribe_id in tribe_ids:
                tribe_name = tribe_names_map.get(tribe_id)
                if tribe_name:
                    if len(tribe_name) > 30:
                        tribe_labels.append(tribe_name[:27] + '...')
                    else:
                        tribe_labels.append(tribe_name)
                else:
                    micro_match = re.search(r'micro_(\d+)', tribe_id)
                    if micro_match:
                        tribe_labels.append(f'Tribe {micro_match.group(1)}')
                    else:
                        tribe_labels.append(f'Tribe {tribe_id.split("/")[-1]}')
            
            tribe_jsd_vals = [v for _, v in sorted_tribes]
            tribe_counts = [tribe_review_counts.get(t, 0) for t in tribe_ids]
            
            num_tribes = len(tribe_labels)
            bar_colors = [colors[i % len(colors)] for i in range(num_tribes)]
            
            if num_tribes > 20:
                bar_width = 0.5
            elif num_tribes > 15:
                bar_width = 0.6
            else:
                bar_width = 0.7
            
            bars = ax.bar(range(num_tribes), tribe_jsd_vals, alpha=0.9, 
                          color=bar_colors, edgecolor='white', linewidth=2, width=bar_width)
            
            # Add labels
            if tribe_jsd_vals:
                max_jsd = max(tribe_jsd_vals)
                y_padding = max_jsd * 0.05
                
                if num_tribes > 20:
                    jsd_fontsize = 9
                    count_fontsize = 8
                elif num_tribes > 15:
                    jsd_fontsize = 10
                    count_fontsize = 9
                else:
                    jsd_fontsize = 11
                    count_fontsize = 10
                
                for i, (bar, jsd_val, count) in enumerate(zip(bars, tribe_jsd_vals, tribe_counts)):
                    height = bar.get_height()
                    x_pos = bar.get_x() + bar.get_width()/2
                    
                    ax.text(x_pos, height + y_padding, 
                           f'{jsd_val:.3f}', ha='center', va='bottom', 
                           fontsize=jsd_fontsize, fontweight='bold', color='#333333')
                    
                    if height > max_jsd * 0.18:
                        ax.text(x_pos, height * 0.5, 
                               f'n={count}', ha='center', va='center', 
                               fontsize=count_fontsize, color='white', fontweight='bold')
            
            # Styling
            ax.set_xticks(range(num_tribes))
            if num_tribes > 20:
                font_size = 8
                rotation = 85
            elif num_tribes > 15:
                font_size = 9
                rotation = 80
            else:
                font_size = 10
                rotation = 70
            ax.set_xticklabels(tribe_labels, fontsize=font_size, fontweight='600', rotation=rotation, ha='right')
            ax.set_xlabel('Tribe', fontsize=14, fontweight='bold', color='#333333', labelpad=25)
            ax.set_ylabel('Aggregated JSD', fontsize=14, fontweight='bold', color='#333333', labelpad=20)
            
            if tribe_jsd_vals:
                ax.set_ylim(0, max(tribe_jsd_vals) * 1.25)
                ax.set_yticks(np.linspace(0, max(tribe_jsd_vals), 6))
                ax.set_yticklabels([f'{y:.3f}' for y in np.linspace(0, max(tribe_jsd_vals), 6)], 
                                  fontsize=12)
            
            cluster_num = cluster_id.replace('cluster_', '') if cluster_id else '?'
            cluster_agg_jsd = cluster_avg_jsd.get(cluster_id, 0.0)
            
            # Calculate average of tribe JSDs (what the bars actually show)
            if tribe_jsd_vals:
                avg_tribe_jsd = np.mean(tribe_jsd_vals)
            else:
                avg_tribe_jsd = 0.0
            
            if num_tribes > 20:
                title_fontsize = 13
                title_pad = 25
            elif num_tribes > 15:
                title_fontsize = 14
                title_pad = 22
            else:
                title_fontsize = 14
                title_pad = 20
            
            # Show both metrics: cluster aggregated and average of tribe JSDs
            ax.set_title(f'Cluster {cluster_num} ({len(tribe_labels)} tribes)\nCluster Agg: {cluster_agg_jsd:.3f}, Avg Tribe: {avg_tribe_jsd:.3f}', 
                        fontsize=title_fontsize, fontweight='bold', pad=title_pad, color='#2c3e50')
            
            ax.grid(True, alpha=0.2, axis='x', linestyle='-', linewidth=0.8, color='#cccccc')
            ax.set_axisbelow(True)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.set_facecolor('#fafafa')
            
            if num_tribes > 20:
                ax.margins(x=0.12, y=0.08)
            elif num_tribes > 15:
                ax.margins(x=0.08, y=0.07)
            else:
                ax.margins(x=0.05, y=0.06)
        
        plt.suptitle(f'Per-Tribe Aggregated JSD by Cluster: Real vs Synthetic\n{prefix.replace("_", " ").title()}', 
                    fontsize=20, fontweight='bold', y=0.98, color='#2c3e50')
        plt.tight_layout(rect=[0, 0, 1, 0.95], pad=4.0)
        filename = output_dir / f"per_tribe_aggregated_jsd_{prefix}.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        logger.info(f"Saved: {filename}")
    
    # Graph 5: P, Q, M Comparison (Per-Tribe - Selected Tribes)
    tribe_actual = data['tribe_actual']
    tribe_predicted = data['tribe_predicted']
    tribe_review_counts = data['tribe_review_counts']
    tribe_names_map = data['tribe_names_map']
    
    if tribe_actual and tribe_predicted:
        # Select top tribes by review count (one per cluster if possible)
        selected_tribes = []
        tribes_by_cluster = defaultdict(list)
        
        for tribe_id in tribe_actual.keys():
            if tribe_id in tribe_predicted and tribe_id in tribe_review_counts:
                cluster_match = re.search(r'(cluster_\d+)', tribe_id)
                if cluster_match:
                    cluster_id = cluster_match.group(1)
                    tribes_by_cluster[cluster_id].append((tribe_id, tribe_review_counts[tribe_id]))
        
        # Select top 2-3 tribes per cluster by review count
        for cluster_id, tribe_list in tribes_by_cluster.items():
            tribe_list.sort(key=lambda x: x[1], reverse=True)
            selected_tribes.extend([t[0] for t in tribe_list[:3]])  # Top 3 per cluster
        
        # Limit to 15 tribes total for visualization
        selected_tribes = selected_tribes[:15]
        
        if selected_tribes:
            num_tribes_vis = len(selected_tribes)
            cols = 3
            rows = int(np.ceil(num_tribes_vis / cols))
            
            fig = plt.figure(figsize=(cols * 8, rows * 6), facecolor='white')
            gs = GridSpec(rows, cols, figure=fig, hspace=0.6, wspace=0.4)
            
            for idx, tribe_id in enumerate(selected_tribes):
                actual_dist = tribe_actual.get(tribe_id, {})
                predicted_dist = tribe_predicted.get(tribe_id, {})
                
                if not actual_dist or not predicted_dist:
                    continue
                
                # Calculate mixture
                all_themes_tribe = sorted(set(actual_dist.keys()) | set(predicted_dist.keys()))
                mixture_tribe = {}
                for theme in all_themes_tribe:
                    p_val = actual_dist.get(theme, 0.0)
                    q_val = predicted_dist.get(theme, 0.0)
                    mixture_tribe[theme] = (p_val + q_val) / 2.0
                
                mix_sum = sum(mixture_tribe.values())
                if mix_sum > 0:
                    mixture_tribe = {k: v / mix_sum for k, v in mixture_tribe.items()}
                
                # Get top themes
                top_themes_tribe = sorted(all_themes_tribe,
                                         key=lambda t: max(actual_dist.get(t, 0),
                                                          predicted_dist.get(t, 0),
                                                          mixture_tribe.get(t, 0)),
                                         reverse=True)[:10]
                
                if not top_themes_tribe:
                    continue
                
                row = idx // cols
                col = idx % cols
                ax = fig.add_subplot(gs[row, col])
                
                x_pos_tribe = np.arange(len(top_themes_tribe))
                width_tribe = 0.25
                
                p_vals = [actual_dist.get(theme, 0.0) for theme in top_themes_tribe]
                q_vals = [predicted_dist.get(theme, 0.0) for theme in top_themes_tribe]
                m_vals = [mixture_tribe.get(theme, 0.0) for theme in top_themes_tribe]
                
                ax.bar(x_pos_tribe - width_tribe, p_vals, width_tribe, 
                      alpha=0.85, color='#51CF66', edgecolor='#2F7D32', 
                      linewidth=1.0, label='P')
                ax.bar(x_pos_tribe, q_vals, width_tribe, 
                      alpha=0.85, color='#4A7FB5', edgecolor='#2B5A82', 
                      linewidth=1.0, label='Q')
                ax.bar(x_pos_tribe + width_tribe, m_vals, width_tribe, 
                      alpha=0.85, color='#FFA726', edgecolor='#E65100', 
                      linewidth=1.0, label='M')
                
                tribe_jsd_val = calculate_aggregated_jsd(actual_dist, predicted_dist)
                tribe_review_count = tribe_review_counts.get(tribe_id, 0)
                tribe_name = tribe_names_map.get(tribe_id, tribe_id.split('/')[-1])
                if len(tribe_name) > 25:
                    tribe_name = tribe_name[:22] + '...'
                
                ax.set_xticks(x_pos_tribe)
                ax.set_xticklabels([t[:12] + '...' if len(t) > 12 else t 
                                   for t in top_themes_tribe], 
                                   fontsize=6, rotation=60, ha='right')
                ax.set_ylabel('Prob', fontsize=9, fontweight='bold')
                ax.set_title(f'{tribe_name}\nJSD: {tribe_jsd_val:.3f}, n={tribe_review_count}', 
                           fontsize=10, fontweight='bold', pad=8)
                ax.legend(loc='upper right', fontsize=7, framealpha=0.9)
                ax.grid(True, alpha=0.2, axis='y', linestyle=':', linewidth=0.6)
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                ax.set_facecolor('#fafafa')
            
            plt.suptitle(f'P, Q, M Distribution Comparison: Selected Tribes\n{prefix.replace("_", " ").title()}', 
                        fontsize=14, fontweight='bold', y=0.995)
            plt.tight_layout(rect=[0, 0, 1, 0.97])
            
            filename = output_dir / f"pqm_comparison_per_tribe_{prefix}.png"
            plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
            plt.close()
            logger.info(f"Saved: {filename}")
    
    return {
        'overall_jsd': overall_jsd,
        'cluster_jsd': cluster_jsd,
        'tribe_jsd': tribe_jsd
    }


def main():
    """Main execution function."""
    logger.info("=" * 80)
    logger.info("AGGREGATED DISTRIBUTION JSD METHOD (Temporary Test)")
    logger.info("=" * 80)
    
    # Initialize W&B run
    run = init_wandb_run(
        run_name="temp_aggregated_jsd",
        stage="Metrics and analysis",
        config={"description": "Test aggregated distribution JSD method"}
    )
    
    if run is None:
        logger.warning("W&B run initialization failed - running in local mode")
    
    try:
        # Load configuration
        config = get_stage_config("Metrics and analysis")
        if not config:
            config_path = BASE_DIR / "Metrics and analysis" / "config.yaml"
            if config_path.exists():
                import yaml
                with open(config_path, 'r', encoding='utf-8') as f:
                    config = yaml.safe_load(f) or {}
            else:
                logger.error("Failed to load configuration")
                return
        
        input_artifacts = config.get('input_artifacts', {})
        artifact_type = config.get('artifact_type', 'model')
        
        # Create temp output directory
        output_dir = BASE_DIR / "Metrics and analysis" / "artifacts" / "temp_aggregated_method"
        output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Output directory: {output_dir.absolute()}")
        
        # Process post_sgo_context (for cluster-1)
        context_artifact_name = input_artifacts.get('post_sgo_context')
        filter_cluster = "cluster_1"  # Filter to only cluster-1
        
        if context_artifact_name:
            logger.info(f"\n{'='*80}")
            logger.info(f"Processing POST-SGO CONTEXT: {context_artifact_name}")
            logger.info(f"Filtering to: {filter_cluster}")
            logger.info(f"{'='*80}")
            
            if run:
                artifact_path = use_artifact(run, context_artifact_name, artifact_type=artifact_type)
            else:
                artifact_path = BASE_DIR / "artifacts" / context_artifact_name.split(':')[0].split('/')[-1]
                if not artifact_path.exists():
                    # Try alternative location
                    artifact_path = BASE_DIR / "07_post_sgo_predictions" / "artifacts" / context_artifact_name.split(':')[0]
            
            if artifact_path and artifact_path.exists():
                logger.info(f"Artifact path: {artifact_path}")
                
                # Build aggregated distributions at all levels (filtered to cluster-1)
                aggregated_data = process_artifact_directory(artifact_path, filter_cluster=filter_cluster)
                
                # Create visualizations and get JSD values
                prefix = f"post_sgo_context_{filter_cluster}"
                jsd_results = create_visualizations(aggregated_data, output_dir, prefix=prefix)
                
                overall_jsd = jsd_results['overall_jsd']
                cluster_jsd = jsd_results['cluster_jsd']
                tribe_jsd = jsd_results['tribe_jsd']
                
                logger.info(f"\n{'='*80}")
                logger.info("AGGREGATED DISTRIBUTION JSD RESULTS")
                logger.info(f"{'='*80}")
                logger.info(f"Total reviews: {aggregated_data['total_reviews']:,}")
                logger.info(f"Overall Aggregated JSD: {overall_jsd:.6f}")
                logger.info(f"\nPer-Cluster Aggregated JSD:")
                for cluster_id in sorted(cluster_jsd.keys()):
                    logger.info(f"  {cluster_id}: {cluster_jsd[cluster_id]:.6f} (n={aggregated_data['cluster_review_counts'].get(cluster_id, 0)})")
                logger.info(f"\nPer-Tribe Aggregated JSD: {len(tribe_jsd)} tribes")
                
                # Save results
                results = {
                    'method': 'aggregated_distribution_jsd',
                    'overall_jsd': float(overall_jsd),
                    'total_reviews': aggregated_data['total_reviews'],
                    'num_actual_themes': len(aggregated_data['aggregated_actual']),
                    'num_predicted_themes': len(aggregated_data['aggregated_predicted']),
                    'num_union_themes': len(set(aggregated_data['aggregated_actual'].keys()) | set(aggregated_data['aggregated_predicted'].keys())),
                    'cluster_jsd': {k: float(v) for k, v in cluster_jsd.items()},
                    'cluster_review_counts': aggregated_data['cluster_review_counts'],
                    'tribe_jsd': {k: float(v) for k, v in tribe_jsd.items()},
                    'tribe_review_counts': aggregated_data['tribe_review_counts'],
                    'aggregated_actual': aggregated_data['aggregated_actual'],
                    'aggregated_predicted': aggregated_data['aggregated_predicted']
                }
                
                results_file = output_dir / "aggregated_jsd_results.json"
                with open(results_file, 'w', encoding='utf-8') as f:
                    json.dump(results, f, indent=2)
                logger.info(f"Saved results to: {results_file}")
            else:
                logger.error(f"Failed to find artifact: {context_artifact_name}")
        
        logger.info("\n" + "=" * 80)
        logger.info("AGGREGATED DISTRIBUTION JSD CALCULATION COMPLETE")
        logger.info("=" * 80)
        
    except Exception as e:
        logger.error(f"Error in main execution: {e}", exc_info=True)
    finally:
        if run:
            finish_run(run)


if __name__ == "__main__":
    main()

