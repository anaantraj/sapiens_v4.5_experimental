#!/usr/bin/env python3
"""
Create clean visualization graphs for delta convergence analysis.

This script creates:
1. Aggregate convergence curves (text, theme, overall deltas)
2. Per-cluster convergence panels (all three deltas per tribe)
3. Per-tribe individual graphs (all three deltas)
4. Improvement distribution histograms
5. Tribe comparison charts with names
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
from pathlib import Path
from collections import defaultdict
import sys

# Add scripts directory to path
scripts_dir = Path(__file__).parent.parent
if str(scripts_dir) not in sys.path:
    sys.path.insert(0, str(scripts_dir))

# Set matplotlib style for clean, professional graphs
try:
    plt.style.use('seaborn-v0_8-whitegrid')
except:
    try:
        plt.style.use('seaborn-whitegrid')
    except:
        plt.style.use('default')
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'DejaVu Sans', 'Liberation Sans'],
    'mathtext.fontset': 'stix',
    'font.size': 12,
    'axes.labelsize': 13,
    'axes.titlesize': 14,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'legend.fontsize': 11,
    'figure.titlesize': 16,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.linewidth': 1.2,
    'axes.grid': True,
    'grid.alpha': 0.15,
    'grid.linestyle': '--',
    'grid.linewidth': 0.5,
    'lines.linewidth': 2.5,
    'lines.markersize': 8,
    'patch.linewidth': 0.5,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'figure.dpi': 150,
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
})

# Color scheme - professional and clean
COLORS = {
    'text': '#4A90E2',      # Clean blue
    'theme': '#E74C3C',     # Clean red
    'overall': '#27AE60',   # Clean green
    'improved': '#27AE60',
    'worse': '#E74C3C',
}

def load_delta_matrices(base_path: Path):
    """Load delta matrices from CSV files"""
    output_dir = base_path.parent / "delta_convergence_analysis"
    
    if not output_dir.exists():
        raise FileNotFoundError(f"Delta matrices not found. Please run build_delta_convergence_matrix.py first.")
    
    print("Loading delta matrices from CSV files...")
    
    matrices = {}
    for matrix_type in ['text', 'theme', 'overall']:
        csv_path = output_dir / f"delta_matrix_{matrix_type}.csv"
        if csv_path.exists():
            df = pd.read_csv(csv_path)
            iter_cols = [col for col in df.columns if col.startswith('I') and col[1:].isdigit()]
            iter_cols = sorted(iter_cols, key=lambda x: int(x[1:]))
            matrices[matrix_type] = {
                'data': df[iter_cols].values,
                'metadata': df[['review_key', 'cluster_id', 'micro_cluster_id']].copy()
            }
            print(f"  Loaded {matrix_type.upper()} matrix: {df.shape[0]} reviews, {len(iter_cols)} iterations")
        else:
            print(f"  WARNING: {csv_path} not found")
    
    return matrices

def load_tribe_names(base_path: Path) -> dict:
    """Load tribe names (persona_name) from summary files"""
    print("\nLoading tribe names from summary files...")
    tribe_names = {}
    
    # Look for summary files in cluster directories
    for cluster_dir in sorted(base_path.iterdir()):
        if not cluster_dir.is_dir() or not cluster_dir.name.startswith('cluster'):
            continue
        
        cluster_id = cluster_dir.name
        summary_files = list(cluster_dir.glob("micro_*_summary_*.json"))
        
        for summary_file in summary_files:
            try:
                with open(summary_file, 'r') as f:
                    data = json.load(f)
                
                # Extract micro_cluster from filename
                filename = summary_file.stem
                micro_match = filename.split('_')
                if len(micro_match) >= 2:
                    micro_cluster_id = f"micro_{micro_match[1]}"
                else:
                    continue
                
                # Get persona_name from metadata
                persona_name = None
                if 'metadata' in data:
                    persona_name = data['metadata'].get('persona_name')
                elif 'persona_name' in data:
                    persona_name = data['persona_name']
                
                if persona_name:
                    tribe_id = f"{cluster_id}/{micro_cluster_id}"
                    tribe_names[tribe_id] = persona_name
                    
            except Exception as e:
                continue
    
    print(f"  Loaded {len(tribe_names)} tribe names")
    return tribe_names

def load_convergence_metrics(base_path: Path):
    """Load convergence metrics CSV"""
    output_dir = base_path.parent / "delta_convergence_analysis"
    metrics_path = output_dir / "convergence_metrics.csv"
    
    if metrics_path.exists():
        df = pd.read_csv(metrics_path)
        print(f"  Loaded convergence metrics: {len(df)} rows")
        return df
    return None

def create_aggregate_convergence_plot(matrices: dict, save_dir: Path):
    """Create clean aggregate convergence curve showing all three deltas"""
    print("\nCreating aggregate convergence plot...")
    
    fig, ax = plt.subplots(figsize=(13, 7.5))
    fig.patch.set_facecolor('white')
    
    iterations = list(range(13))  # I0 to I12
    x_iter = np.array(iterations)
    
    # Plot all three deltas with clean styling - ALL SOLID LINES
    for matrix_type, color_key, marker, linestyle, label, zorder in [
        ('overall', 'overall', 'o', '-', 'Overall Delta', 5),
        ('theme', 'theme', 's', '-', 'Theme Delta', 4),
        ('text', 'text', '^', '-', 'Text Delta', 3),
    ]:
        if matrix_type not in matrices:
            continue
        
        matrix_data = matrices[matrix_type]['data']
        
        # Calculate means and confidence intervals
        means = np.nanmean(matrix_data, axis=0)
        stds = np.nanstd(matrix_data, axis=0)
        n = np.sum(~np.isnan(matrix_data), axis=0)
        ci = 1.96 * stds / np.sqrt(n)  # 95% CI
        
        color = COLORS[color_key]
        
        # Subtle confidence interval
        ax.fill_between(x_iter, means - ci, means + ci, alpha=0.1, color=color, linewidth=0)
        
        # Clean solid line with markers at EVERY iteration (matching notebook style)
        # Use format string like 'o-', 's-', '^-' for markers at every point
        marker_style = {'o': 'o-', 's': 's-', '^': '^-'}.get(marker, 'o-')
        ax.plot(x_iter, means, marker_style, color=color, 
               linewidth=2.5, markersize=10, 
               markerfacecolor=color, markeredgecolor='white', markeredgewidth=1.8,
               label=label, zorder=zorder, alpha=0.95)
        
        # Annotate I0 and I12 for ALL three deltas - positioned directly above the points
        improvement_pct = (means[0] - means[-1]) / means[0] * 100 if means[0] > 0 else 0
        
        # Calculate y-axis range for proper spacing
        y_range = ax.get_ylim()[1] - ax.get_ylim()[0]
        offset_above = y_range * 0.03  # 3% of y-range above the point
        
        # I0 annotation - directly above the point
        ax.text(0, means[0] + offset_above, f'I0: {means[0]:.3f}',
               fontsize=10, fontweight='bold', color=color,
               bbox=dict(boxstyle='round,pad=0.4', facecolor='white', 
                        edgecolor=color, alpha=0.9, linewidth=1.5),
               verticalalignment='bottom', horizontalalignment='center')
        
        # I12 annotation with improvement percentage - directly above the point
        ax.text(12, means[-1] + offset_above, f'I12: {means[-1]:.3f}\n({improvement_pct:+.1f}%)',
               fontsize=10, fontweight='bold', color=color,
               bbox=dict(boxstyle='round,pad=0.4', facecolor='white', 
                        edgecolor=color, alpha=0.9, linewidth=1.5),
               verticalalignment='bottom', horizontalalignment='center')
    
    ax.set_xlabel('SGO Iteration', fontweight='bold', fontsize=14, labelpad=10)
    ax.set_ylabel('Delta (Δ)', fontweight='bold', fontsize=14, labelpad=10)
    ax.set_xticks(x_iter)
    ax.set_xticklabels([f'I{i}' for i in range(13)], fontsize=11)
    ax.set_xlim(-0.5, 12.5)
    ax.legend(loc='upper right', frameon=True, fancybox=True, shadow=True,
             edgecolor='#ddd', framealpha=0.98, fontsize=12, 
             borderpad=0.8, handlelength=2.5)
    ax.set_title('Delta Convergence Across All Reviews (I0 → I12)', 
                fontweight='bold', pad=20, fontsize=16)
    ax.grid(True, alpha=0.2, linestyle='--', linewidth=0.8)
    
    plt.tight_layout()
    save_path = save_dir / "aggregate_convergence.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    print(f"  Saved: {save_path}")
    plt.close()

def create_per_cluster_plots(matrices: dict, metrics_df: pd.DataFrame, 
                             tribe_names: dict, save_dir: Path):
    """Create per-cluster convergence panels - all three deltas per tribe"""
    print("\nCreating per-cluster convergence plots...")
    
    if metrics_df is None:
        print("  Skipping: No metrics data available")
        return
    
    # Filter to tribe-level metrics for OVERALL delta only (for sorting)
    tribe_metrics = metrics_df[
        (metrics_df['cluster_id'] != 'ALL') & 
        (metrics_df['micro_cluster_id'] != 'ALL') &
        (metrics_df['matrix_type'] == 'OVERALL')
    ].copy()
    
    clusters = sorted(tribe_metrics['cluster_id'].unique())
    
    for cluster in clusters:
        cluster_tribes = tribe_metrics[tribe_metrics['cluster_id'] == cluster].copy()
        cluster_tribes = cluster_tribes.sort_values('improvement_pct', ascending=False)
        
        n_tribes = len(cluster_tribes)
        if n_tribes == 0:
            continue
        
        # Create grid layout
        n_cols = 2
        n_rows = int(np.ceil(n_tribes / n_cols))
        fig_h = max(5, n_rows * 2.5)
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, fig_h), squeeze=False)
        fig.patch.set_facecolor('white')
        fig.suptitle(f'{cluster} — {n_tribes} Tribes (sorted by improvement)', 
                     fontsize=16, fontweight='bold', y=0.998)
        
        for idx, (_, row) in enumerate(cluster_tribes.iterrows()):
            r, c = idx // n_cols, idx % n_cols
            ax = axes[r][c]
            
            micro_cluster = row['micro_cluster_id']
            tribe_id = f"{cluster}/{micro_cluster}"
            tribe_name = tribe_names.get(tribe_id, tribe_id)
            
            # Get data for this tribe - all three deltas
            metadata = matrices['overall']['metadata']
            tribe_mask = (metadata['cluster_id'] == cluster) & \
                        (metadata['micro_cluster_id'] == micro_cluster)
            
            if tribe_mask.sum() == 0:
                ax.text(0.5, 0.5, 'No data', ha='center', va='center', 
                       transform=ax.transAxes, fontsize=12)
                continue
            
            iterations = list(range(13))
            x_iter = np.array(iterations)
            
            # Plot all three deltas with clean styling - ALL SOLID LINES
            for matrix_type, color_key, marker, linestyle, label, zorder in [
                ('overall', 'overall', 'o', '-', 'Overall', 5),
                ('theme', 'theme', 's', '-', 'Theme', 4),
                ('text', 'text', '^', '-', 'Text', 3),
            ]:
                if matrix_type not in matrices:
                    continue
                
                tribe_data = matrices[matrix_type]['data'][tribe_mask]
                means = np.nanmean(tribe_data, axis=0)
                
                color = COLORS[color_key]
                # Clean solid lines with markers at EVERY iteration (matching notebook style)
                marker_style = {'o': 'o-', 's': 's-', '^': '^-'}.get(marker, 'o-')
                ax.plot(x_iter, means, marker_style, color=color, 
                       linewidth=2, markersize=6, 
                       markerfacecolor=color, markeredgecolor='white', markeredgewidth=1.2,
                       label=label, alpha=0.95, zorder=zorder)
            
            # Improvement annotation
            improvement = row['improvement_pct']
            delta_color = COLORS['improved'] if improvement > 0 else COLORS['worse']
            delta_symbol = '↓' if improvement > 0 else '↑'
            ax.text(0.97, 0.95, f'{delta_symbol} {abs(improvement):.1f}%',
                   transform=ax.transAxes, ha='right', va='top',
                   fontsize=10, fontweight='bold', color=delta_color,
                   bbox=dict(boxstyle='round,pad=0.4', facecolor='white',
                            edgecolor=delta_color, alpha=0.95, linewidth=1.2))
            
            # Title with tribe name
            n_reviews = int(row['num_reviews'])
            title = f'{tribe_name[:38]}' + ('...' if len(tribe_name) > 38 else '')
            ax.set_title(f'{title}\n({cluster}/{micro_cluster}, n={n_reviews})', 
                        fontsize=10, fontweight='bold', pad=8)
            
            ax.set_xticks(x_iter[::2])  # Show every other iteration
            ax.set_xticklabels([f'I{i}' for i in range(0, 13, 2)], fontsize=9)
            ax.set_xlim(-0.5, 12.5)
            ax.set_xlabel('Iteration', fontsize=10, fontweight='bold')
            ax.set_ylabel('Delta', fontsize=10, fontweight='bold')
            ax.grid(True, alpha=0.15, linestyle='--', linewidth=0.6)
            ax.tick_params(axis='y', labelsize=9)
            
            # Legend on first subplot only
            if idx == 0:
                ax.legend(loc='upper right', fontsize=9, framealpha=0.95, 
                         shadow=True, edgecolor='#ddd', borderpad=0.6)
        
        # Hide unused axes
        for idx in range(n_tribes, n_rows * n_cols):
            r, c = idx // n_cols, idx % n_cols
            axes[r][c].set_visible(False)
        
        plt.tight_layout(rect=[0, 0, 1, 0.99])
        save_path = save_dir / f"convergence_{cluster}.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
        print(f"  Saved: {save_path}")
        plt.close()

def create_per_tribe_individual_plots(matrices: dict, metrics_df: pd.DataFrame,
                                      tribe_names: dict, save_dir: Path):
    """Create individual plots for each tribe - all three deltas"""
    print("\nCreating individual tribe plots...")
    
    if metrics_df is None:
        print("  Skipping: No metrics data available")
        return
    
    tribe_metrics = metrics_df[
        (metrics_df['cluster_id'] != 'ALL') & 
        (metrics_df['micro_cluster_id'] != 'ALL') &
        (metrics_df['matrix_type'] == 'OVERALL')
    ].copy()
    
    tribe_dir = save_dir / "tribes"
    tribe_dir.mkdir(exist_ok=True)
    
    for _, row in tribe_metrics.iterrows():
        cluster = row['cluster_id']
        micro_cluster = row['micro_cluster_id']
        tribe_id = f"{cluster}/{micro_cluster}"
        tribe_name = tribe_names.get(tribe_id, tribe_id)
        
        # Get data for this tribe
        metadata = matrices['overall']['metadata']
        tribe_mask = (metadata['cluster_id'] == cluster) & \
                    (metadata['micro_cluster_id'] == micro_cluster)
        
        if tribe_mask.sum() == 0:
            continue
        
        fig, ax = plt.subplots(figsize=(11, 6.5))
        fig.patch.set_facecolor('white')
        
        iterations = list(range(13))
        x_iter = np.array(iterations)
        
        # Plot all three deltas with clean styling - ALL SOLID LINES
        for matrix_type, color_key, marker, linestyle, label, zorder in [
            ('overall', 'overall', 'o', '-', 'Overall Delta', 5),
            ('theme', 'theme', 's', '-', 'Theme Delta', 4),
            ('text', 'text', '^', '-', 'Text Delta', 3),
        ]:
            if matrix_type not in matrices:
                continue
            
            tribe_data = matrices[matrix_type]['data'][tribe_mask]
            means = np.nanmean(tribe_data, axis=0)
            stds = np.nanstd(tribe_data, axis=0)
            n = tribe_data.shape[0]
            ci = 1.96 * stds / np.sqrt(n) if n > 0 else 0
            
            color = COLORS[color_key]
            # Subtle confidence interval
            ax.fill_between(x_iter, means - ci, means + ci, alpha=0.1, color=color, linewidth=0)
            # Clean solid line with markers at EVERY iteration (matching notebook style)
            marker_style = {'o': 'o-', 's': 's-', '^': '^-'}.get(marker, 'o-')
            ax.plot(x_iter, means, marker_style, color=color, 
                   linewidth=2.5, markersize=10, 
                   markerfacecolor=color, markeredgecolor='white', markeredgewidth=1.8,
                   label=label, zorder=zorder, alpha=0.95)
        
        # Improvement annotation
        improvement = row['improvement_pct']
        delta_color = COLORS['improved'] if improvement > 0 else COLORS['worse']
        delta_symbol = '↓' if improvement > 0 else '↑'
        ax.text(0.97, 0.95, f'{delta_symbol} {abs(improvement):.1f}%',
               transform=ax.transAxes, ha='right', va='top',
               fontsize=12, fontweight='bold', color=delta_color,
               bbox=dict(boxstyle='round,pad=0.5', facecolor='white',
                        edgecolor=delta_color, alpha=0.95, linewidth=1.5))
        
        ax.set_xlabel('SGO Iteration', fontweight='bold', fontsize=13, labelpad=10)
        ax.set_ylabel('Delta (Δ)', fontweight='bold', fontsize=13, labelpad=10)
        ax.set_xticks(x_iter)
        ax.set_xticklabels([f'I{i}' for i in range(13)], fontsize=11)
        ax.set_xlim(-0.5, 12.5)
        ax.legend(loc='upper right', frameon=True, fancybox=True, shadow=True,
                 edgecolor='#ddd', framealpha=0.98, fontsize=12, borderpad=0.8)
        ax.set_title(f'{tribe_name}\n{cluster}/{micro_cluster} (n={int(row["num_reviews"])})', 
                    fontweight='bold', pad=18, fontsize=14)
        ax.grid(True, alpha=0.2, linestyle='--', linewidth=0.8)
        
        plt.tight_layout()
        # Sanitize filename
        safe_name = "".join(c for c in tribe_name if c.isalnum() or c in (' ', '-', '_')).strip()[:50]
        save_path = tribe_dir / f"{cluster}_{micro_cluster}_{safe_name}.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
        plt.close()
    
    print(f"  Saved {len(tribe_metrics)} individual tribe plots to {tribe_dir}")

def create_improvement_distribution(metrics_df: pd.DataFrame, save_dir: Path):
    """Create improvement distribution histograms"""
    print("\nCreating improvement distribution plots...")
    
    if metrics_df is None:
        return
    
    tribe_metrics = metrics_df[
        (metrics_df['cluster_id'] != 'ALL') & 
        (metrics_df['micro_cluster_id'] != 'ALL') &
        (metrics_df['matrix_type'] == 'OVERALL')
    ].copy()
    
    if len(tribe_metrics) == 0:
        return
    
    improvements = tribe_metrics['improvement_pct'].values
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6.5))
    fig.patch.set_facecolor('white')
    
    # Histogram
    ax1 = axes[0]
    n, bins, patches = ax1.hist(improvements, bins=30, edgecolor='white', 
                                alpha=0.75, color=COLORS['overall'], linewidth=1.2)
    ax1.axvline(x=0, color='#E74C3C', linestyle='--', linewidth=2.5, label='No change', zorder=5)
    ax1.axvline(x=np.mean(improvements), color='#27AE60', linestyle='--', linewidth=2.5, 
               label=f'Mean: {np.mean(improvements):.1f}%', zorder=5)
    ax1.set_xlabel('Improvement %', fontsize=14, fontweight='bold', labelpad=10)
    ax1.set_ylabel('Number of Tribes', fontsize=14, fontweight='bold', labelpad=10)
    ax1.set_title('Distribution of Tribe Improvements (I0 → I12)', 
                 fontsize=15, fontweight='bold', pad=15)
    ax1.legend(fontsize=12, framealpha=0.95, shadow=True, edgecolor='#ddd')
    ax1.grid(True, alpha=0.2, linestyle='--', linewidth=0.8)
    
    # Bar chart (sorted)
    ax2 = axes[1]
    sorted_imp = sorted(improvements, reverse=True)
    colors = [COLORS['improved'] if i > 0 else COLORS['worse'] for i in sorted_imp]
    ax2.bar(range(len(sorted_imp)), sorted_imp, color=colors, alpha=0.8, 
           edgecolor='white', linewidth=1.2)
    ax2.axhline(y=0, color='black', linewidth=1.5, zorder=5)
    ax2.set_xlabel('Tribe (ranked by improvement)', fontsize=14, fontweight='bold', labelpad=10)
    ax2.set_ylabel('Improvement %', fontsize=14, fontweight='bold', labelpad=10)
    ax2.set_title(f'All {len(sorted_imp)} Tribes Ranked by Improvement', 
                 fontsize=15, fontweight='bold', pad=15)
    ax2.grid(True, alpha=0.2, linestyle='--', linewidth=0.8, axis='y')
    
    plt.tight_layout()
    save_path = save_dir / "improvement_distribution.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    print(f"  Saved: {save_path}")
    plt.close()

def create_tribe_comparison_chart(metrics_df: pd.DataFrame, tribe_names: dict, 
                                 save_dir: Path, top_n: int = 20):
    """Create horizontal bar chart comparing top/bottom tribes with names"""
    print(f"\nCreating tribe comparison chart (top/bottom {top_n})...")
    
    if metrics_df is None:
        return
    
    tribe_metrics = metrics_df[
        (metrics_df['cluster_id'] != 'ALL') & 
        (metrics_df['micro_cluster_id'] != 'ALL') &
        (metrics_df['matrix_type'] == 'OVERALL')
    ].copy()
    
    if len(tribe_metrics) == 0:
        return
    
    # Add tribe names
    tribe_metrics['tribe_name'] = tribe_metrics.apply(
        lambda row: tribe_names.get(f"{row['cluster_id']}/{row['micro_cluster_id']}", 
                                   f"{row['cluster_id']}/{row['micro_cluster_id']}"),
        axis=1
    )
    
    tribe_metrics = tribe_metrics.sort_values('improvement_pct', ascending=True)
    
    top_tribes = tribe_metrics.tail(top_n)
    bottom_tribes = tribe_metrics.head(top_n)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, max(10, top_n * 0.5)))
    fig.patch.set_facecolor('white')
    
    # Top tribes
    y_pos1 = np.arange(len(top_tribes))
    colors1 = [COLORS['improved'] if imp > 0 else COLORS['worse'] 
               for imp in top_tribes['improvement_pct']]
    ax1.barh(y_pos1, top_tribes['improvement_pct'], color=colors1, alpha=0.85, 
            edgecolor='white', linewidth=1.5)
    ax1.set_yticks(y_pos1)
    ax1.set_yticklabels([f"{row['tribe_name'][:42]}" for _, row in top_tribes.iterrows()], 
                        fontsize=10)
    ax1.set_xlabel('Improvement %', fontsize=13, fontweight='bold', labelpad=10)
    ax1.set_title(f'Top {top_n} Improving Tribes', fontsize=15, fontweight='bold', pad=15)
    ax1.axvline(x=0, color='black', linewidth=1.5, zorder=5)
    ax1.grid(True, alpha=0.2, linestyle='--', linewidth=0.8, axis='x')
    
    # Bottom tribes
    y_pos2 = np.arange(len(bottom_tribes))
    colors2 = [COLORS['improved'] if imp > 0 else COLORS['worse'] 
               for imp in bottom_tribes['improvement_pct']]
    ax2.barh(y_pos2, bottom_tribes['improvement_pct'], color=colors2, alpha=0.85, 
            edgecolor='white', linewidth=1.5)
    ax2.set_yticks(y_pos2)
    ax2.set_yticklabels([f"{row['tribe_name'][:42]}" for _, row in bottom_tribes.iterrows()], 
                        fontsize=10)
    ax2.set_xlabel('Improvement %', fontsize=13, fontweight='bold', labelpad=10)
    ax2.set_title(f'Bottom {top_n} Tribes', fontsize=15, fontweight='bold', pad=15)
    ax2.axvline(x=0, color='black', linewidth=1.5, zorder=5)
    ax2.grid(True, alpha=0.2, linestyle='--', linewidth=0.8, axis='x')
    
    plt.tight_layout()
    save_path = save_dir / "tribe_comparison.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    print(f"  Saved: {save_path}")
    plt.close()

def main():
    """Main execution function"""
    base_path = Path("/media/samanvitha/Seagate HDD/vectorial_sapiens_v4/07_sgo_training/artifacts/sgo_training_results_v6")
    
    if not base_path.exists():
        print(f"Error: Base path does not exist: {base_path}")
        return
    
    print("="*70)
    print("DELTA CONVERGENCE VISUALIZATION - CLEAN GRAPHS")
    print("="*70)
    
    save_dir = base_path.parent / "delta_convergence_analysis" / "figures"
    save_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {save_dir}")
    
    # Load data
    try:
        matrices = load_delta_matrices(base_path)
        metrics_df = load_convergence_metrics(base_path)
        tribe_names = load_tribe_names(base_path)
    except Exception as e:
        print(f"Error loading data: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Create visualizations
    create_aggregate_convergence_plot(matrices, save_dir)
    create_per_cluster_plots(matrices, metrics_df, tribe_names, save_dir)
    create_per_tribe_individual_plots(matrices, metrics_df, tribe_names, save_dir)
    create_improvement_distribution(metrics_df, save_dir)
    create_tribe_comparison_chart(metrics_df, tribe_names, save_dir, top_n=20)
    
    print("\n" + "="*70)
    print("VISUALIZATION COMPLETE")
    print("="*70)
    print(f"All figures saved to: {save_dir}")
    print(f"  - Aggregate convergence: aggregate_convergence.png")
    print(f"  - Per-cluster panels: convergence_cluster_X.png")
    print(f"  - Individual tribes: tribes/*.png")
    print(f"  - Improvement distribution: improvement_distribution.png")
    print(f"  - Tribe comparison: tribe_comparison.png")

if __name__ == "__main__":
    main()
