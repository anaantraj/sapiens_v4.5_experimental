#!/usr/bin/env python3
"""
Build delta convergence matrices from I0 to I12 for all reviews across all clusters and micro-clusters (tribes).

This script:
1. Loads I0 deltas from _delta/cluster_X/micro_Y_all_reviews_deltas.json
2. Loads I1-I12 deltas from _journey/cluster_X/micro_Y_journey.json
3. Builds sparse matrices with forward fill for missing iterations
4. Saves to CSV and Excel with cluster/tribe information
5. Calculates convergence metrics per tribe
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple, Optional
import sys

# Add scripts directory to path
scripts_dir = Path(__file__).parent.parent
if str(scripts_dir) not in sys.path:
    sys.path.insert(0, str(scripts_dir))

def load_i0_deltas(base_path: Path) -> Dict[str, Dict[int, float]]:
    """
    Load I0 (initial) deltas from _delta folder.
    
    Returns:
        Dict with keys: 'text', 'theme', 'overall'
        Each value is a dict: {review_key: {0: delta_value}}
    """
    delta_path = base_path / "_delta"
    sparse_matrices = {
        'text': {},
        'theme': {},
        'overall': {}
    }
    
    print("Loading I0 deltas from _delta folder...")
    total_reviews = 0
    
    for cluster_dir in sorted(delta_path.iterdir()):
        if not cluster_dir.is_dir() or not cluster_dir.name.startswith('cluster'):
            continue
        
        cluster_id = cluster_dir.name
        delta_files = list(cluster_dir.glob("micro_*_all_reviews_deltas.json"))
        
        for delta_file in delta_files:
            try:
                with open(delta_file, 'r') as f:
                    data = json.load(f)
                
                cluster_id_from_file = data.get('cluster_id', cluster_id)
                micro_cluster_id = data.get('micro_cluster_id', '')
                
                for delta_entry in data.get('deltas', []):
                    review_key = delta_entry.get('review_key', '')
                    if not review_key:
                        continue
                    
                    # Store with cluster and micro_cluster info in key
                    full_key = f"{cluster_id_from_file}|{micro_cluster_id}|{review_key}"
                    
                    text_delta = delta_entry.get('text_delta')
                    theme_delta = delta_entry.get('theme_delta')
                    overall_delta = delta_entry.get('overall_delta')
                    
                    if text_delta is not None:
                        sparse_matrices['text'][full_key] = {0: float(text_delta)}
                    if theme_delta is not None:
                        sparse_matrices['theme'][full_key] = {0: float(theme_delta)}
                    if overall_delta is not None:
                        sparse_matrices['overall'][full_key] = {0: float(overall_delta)}
                    else:
                        # Calculate overall_delta if not present: 0.7 * text + 0.3 * theme
                        if text_delta is not None and theme_delta is not None:
                            overall_delta = 0.7 * float(text_delta) + 0.3 * float(theme_delta)
                            sparse_matrices['overall'][full_key] = {0: overall_delta}
                    
                    total_reviews += 1
                    
            except Exception as e:
                print(f"  Error loading {delta_file}: {e}")
                continue
    
    print(f"  Loaded I0 deltas for {total_reviews} reviews")
    print(f"  Unique reviews in text matrix: {len(sparse_matrices['text'])}")
    print(f"  Unique reviews in theme matrix: {len(sparse_matrices['theme'])}")
    print(f"  Unique reviews in overall matrix: {len(sparse_matrices['overall'])}")
    
    return sparse_matrices

def load_journey_deltas(base_path: Path, sparse_matrices: Dict[str, Dict[str, Dict[int, float]]]) -> Dict[str, set]:
    """
    Load I1-I12 deltas from _journey folder and update sparse matrices.
    
    Returns:
        Set of review keys that have journey data
    """
    journey_path = base_path / "_journey"
    reviews_in_journey = set()
    
    print("\nLoading I1-I12 deltas from _journey folder...")
    journey_files = list(journey_path.glob("**/micro_*_journey.json"))
    print(f"  Found {len(journey_files)} journey files")
    
    for journey_file in journey_files:
        try:
            with open(journey_file, 'r') as f:
                journey_data = json.load(f)
            
            # Extract cluster and micro_cluster from path
            parts = journey_file.parts
            cluster_idx = None
            for i, part in enumerate(parts):
                if part.startswith('cluster_'):
                    cluster_idx = i
                    break
            
            if cluster_idx is None:
                continue
            
            cluster_id = parts[cluster_idx]
            # Extract micro_cluster from filename: micro_X_journey.json
            filename = journey_file.name
            micro_match = filename.split('_')
            if len(micro_match) >= 2:
                micro_cluster_id = f"micro_{micro_match[1]}"
            else:
                micro_cluster_id = ""
            
            for review_key, review_data in journey_data.items():
                # Create full key with cluster and micro_cluster info
                full_key = f"{cluster_id}|{micro_cluster_id}|{review_key}"
                reviews_in_journey.add(full_key)
                
                # Initialize if not already present
                for matrix_type in ['text', 'theme', 'overall']:
                    if full_key not in sparse_matrices[matrix_type]:
                        sparse_matrices[matrix_type][full_key] = {}
                
                # Get initial deltas from journey file (I0) - may override delta file values
                if 'initial_prediction_deltas' in review_data:
                    initial = review_data['initial_prediction_deltas']
                    if 'text_delta' in initial:
                        sparse_matrices['text'][full_key][0] = float(initial['text_delta'])
                    if 'theme_delta' in initial:
                        sparse_matrices['theme'][full_key][0] = float(initial['theme_delta'])
                    # Recalculate overall_delta from text and theme
                    if 0 in sparse_matrices['text'][full_key] and 0 in sparse_matrices['theme'][full_key]:
                        sparse_matrices['overall'][full_key][0] = (
                            0.7 * sparse_matrices['text'][full_key][0] + 
                            0.3 * sparse_matrices['theme'][full_key][0]
                        )
                
                # Process correction_journey for iterations 1-12
                if 'correction_journey' in review_data and review_data['correction_journey']:
                    # Store deltas for each iteration
                    for entry in review_data['correction_journey']:
                        iteration = entry.get('iteration', 0)
                        if iteration < 1 or iteration > 12:
                            continue
                        
                        new_text_delta = entry.get('new_text_delta')
                        new_theme_delta = entry.get('new_theme_delta')
                        new_overall_delta = entry.get('new_overall_delta')
                        
                        if new_text_delta is not None:
                            sparse_matrices['text'][full_key][iteration] = float(new_text_delta)
                        if new_theme_delta is not None:
                            sparse_matrices['theme'][full_key][iteration] = float(new_theme_delta)
                        if new_overall_delta is not None:
                            sparse_matrices['overall'][full_key][iteration] = float(new_overall_delta)
                        else:
                            # Calculate overall_delta if not present
                            if new_text_delta is not None and new_theme_delta is not None:
                                overall_delta = 0.7 * float(new_text_delta) + 0.3 * float(new_theme_delta)
                                sparse_matrices['overall'][full_key][iteration] = overall_delta
                
        except Exception as e:
            print(f"  Error loading {journey_file}: {e}")
            continue
    
    print(f"  Reviews in journey files: {len(reviews_in_journey)}")
    return reviews_in_journey

def forward_fill_matrices(sparse_matrices: Dict[str, Dict[str, Dict[int, float]]], 
                         max_iteration: int = 12) -> Dict[str, np.ndarray]:
    """
    Forward fill sparse matrices and convert to dense numpy arrays.
    For each iteration, uses the MINIMUM delta seen so far (from I0 up to that iteration).
    This ensures that if a delta gets worse, we keep the best value achieved.
    
    Returns:
        Dict with keys: 'text', 'theme', 'overall'
        Each value is a numpy array of shape (num_reviews, max_iteration + 1)
    """
    print(f"\nForward filling matrices up to iteration {max_iteration}...")
    print("  Using MINIMUM delta seen so far at each iteration (best performance tracking)")
    
    # Get all unique review keys
    all_keys = set()
    for matrix_type in ['text', 'theme', 'overall']:
        all_keys.update(sparse_matrices[matrix_type].keys())
    
    all_keys = sorted(list(all_keys))
    num_reviews = len(all_keys)
    num_iterations = max_iteration + 1  # I0 to I12 = 13 iterations
    
    print(f"  Total unique reviews: {num_reviews}")
    print(f"  Iterations: I0 to I{max_iteration} ({num_iterations} total)")
    
    matrices = {}
    for matrix_type in ['text', 'theme', 'overall']:
        matrix = np.full((num_reviews, num_iterations), np.nan)
        
        for i, review_key in enumerate(all_keys):
            if review_key in sparse_matrices[matrix_type]:
                deltas = sparse_matrices[matrix_type][review_key]
                
                # Store raw deltas first
                raw_deltas = {}
                for iteration, delta_value in deltas.items():
                    if 0 <= iteration <= max_iteration:
                        raw_deltas[iteration] = delta_value
                
                # For each iteration, use the MINIMUM delta seen so far (from I0 to that iteration)
                # This ensures we track the best performance achieved
                for j in range(num_iterations):
                    # Collect all raw deltas from I0 up to iteration j
                    available_deltas = []
                    for k in range(j + 1):  # From 0 to j (inclusive)
                        if k in raw_deltas:
                            available_deltas.append(raw_deltas[k])
                    
                    if available_deltas:
                        # Use the minimum (best) delta seen so far
                        matrix[i, j] = min(available_deltas)
                    elif j > 0:
                        # If no raw data available for this iteration, forward fill from previous
                        # But we still want the minimum, so use the previous iteration's min value
                        matrix[i, j] = matrix[i, j-1]
                    else:
                        # I0 has no data - will be handled by NaN fill later
                        pass
        
        matrices[matrix_type] = matrix
        
        # Verify no NaN values remain
        nan_count = np.sum(np.isnan(matrix))
        if nan_count > 0:
            print(f"  WARNING: {matrix_type} matrix has {nan_count} NaN values after processing")
            # Fill any remaining NaNs with the last known value or 0
            for i in range(num_reviews):
                for j in range(num_iterations):
                    if np.isnan(matrix[i, j]):
                        # Try to find last non-NaN value
                        for k in range(j-1, -1, -1):
                            if not np.isnan(matrix[i, k]):
                                matrix[i, j] = matrix[i, k]
                                break
                        # If still NaN, use 0
                        if np.isnan(matrix[i, j]):
                            matrix[i, j] = 0.0
        
        print(f"  {matrix_type.upper()} matrix: {matrix.shape}, NaN count: {np.sum(np.isnan(matrix))}")
    
    return matrices, all_keys

def calculate_convergence_metrics(matrices: Dict[str, np.ndarray], 
                                  review_keys: List[str]) -> pd.DataFrame:
    """
    Calculate convergence metrics per cluster, micro_cluster (tribe), and overall.
    """
    print("\nCalculating convergence metrics...")
    
    # Parse cluster and micro_cluster from review keys
    cluster_micro_map = {}
    for key in review_keys:
        parts = key.split('|')
        if len(parts) >= 3:
            cluster_id = parts[0]
            micro_cluster_id = parts[1]
            cluster_micro_map[key] = (cluster_id, micro_cluster_id)
        else:
            cluster_micro_map[key] = ('unknown', 'unknown')
    
    metrics_data = []
    
    # Overall metrics
    for matrix_type in ['text', 'theme', 'overall']:
        matrix = matrices[matrix_type]
        i0_mean = np.mean(matrix[:, 0])
        i12_mean = np.mean(matrix[:, 12])
        improvement = i0_mean - i12_mean
        improvement_pct = (improvement / i0_mean * 100) if i0_mean > 0 else 0.0
        
        metrics_data.append({
            'cluster_id': 'ALL',
            'micro_cluster_id': 'ALL',
            'matrix_type': matrix_type.upper(),
            'i0_mean': i0_mean,
            'i12_mean': i12_mean,
            'improvement': improvement,
            'improvement_pct': improvement_pct,
            'num_reviews': len(review_keys)
        })
    
    # Per-cluster metrics
    clusters = set(c for c, _ in cluster_micro_map.values())
    for cluster_id in sorted(clusters):
        cluster_keys = [k for k, (c, _) in cluster_micro_map.items() if c == cluster_id]
        cluster_indices = [i for i, k in enumerate(review_keys) if k in cluster_keys]
        
        if not cluster_indices:
            continue
        
        cluster_matrix_text = matrices['text'][cluster_indices, :]
        cluster_matrix_theme = matrices['theme'][cluster_indices, :]
        cluster_matrix_overall = matrices['overall'][cluster_indices, :]
        
        for matrix_type, matrix in [('text', cluster_matrix_text), 
                                   ('theme', cluster_matrix_theme),
                                   ('overall', cluster_matrix_overall)]:
            i0_mean = np.mean(matrix[:, 0])
            i12_mean = np.mean(matrix[:, 12])
            improvement = i0_mean - i12_mean
            improvement_pct = (improvement / i0_mean * 100) if i0_mean > 0 else 0.0
            
            metrics_data.append({
                'cluster_id': cluster_id,
                'micro_cluster_id': 'ALL',
                'matrix_type': matrix_type.upper(),
                'i0_mean': i0_mean,
                'i12_mean': i12_mean,
                'improvement': improvement,
                'improvement_pct': improvement_pct,
                'num_reviews': len(cluster_indices)
            })
    
    # Per-micro-cluster (tribe) metrics
    cluster_micro_pairs = set(cluster_micro_map.values())
    for cluster_id, micro_cluster_id in sorted(cluster_micro_pairs):
        if micro_cluster_id == 'ALL':
            continue
        
        tribe_keys = [k for k, (c, m) in cluster_micro_map.items() 
                     if c == cluster_id and m == micro_cluster_id]
        tribe_indices = [i for i, k in enumerate(review_keys) if k in tribe_keys]
        
        if not tribe_indices:
            continue
        
        tribe_matrix_text = matrices['text'][tribe_indices, :]
        tribe_matrix_theme = matrices['theme'][tribe_indices, :]
        tribe_matrix_overall = matrices['overall'][tribe_indices, :]
        
        for matrix_type, matrix in [('text', tribe_matrix_text), 
                                   ('theme', tribe_matrix_theme),
                                   ('overall', tribe_matrix_overall)]:
            i0_mean = np.mean(matrix[:, 0])
            i12_mean = np.mean(matrix[:, 12])
            improvement = i0_mean - i12_mean
            improvement_pct = (improvement / i0_mean * 100) if i0_mean > 0 else 0.0
            
            metrics_data.append({
                'cluster_id': cluster_id,
                'micro_cluster_id': micro_cluster_id,
                'matrix_type': matrix_type.upper(),
                'i0_mean': i0_mean,
                'i12_mean': i12_mean,
                'improvement': improvement,
                'improvement_pct': improvement_pct,
                'num_reviews': len(tribe_indices)
            })
    
    metrics_df = pd.DataFrame(metrics_data)
    return metrics_df

def save_matrices_to_files(matrices: Dict[str, np.ndarray], 
                           review_keys: List[str],
                           output_dir: Path,
                           metrics_df: pd.DataFrame):
    """
    Save matrices to CSV and Excel files.
    """
    print(f"\nSaving matrices to {output_dir}...")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Parse review keys to extract components
    parsed_keys = []
    for key in review_keys:
        parts = key.split('|')
        if len(parts) >= 3:
            cluster_id = parts[0]
            micro_cluster_id = parts[1]
            review_key = parts[2]
            parsed_keys.append({
                'cluster_id': cluster_id,
                'micro_cluster_id': micro_cluster_id,
                'review_key': review_key
            })
        else:
            parsed_keys.append({
                'cluster_id': 'unknown',
                'micro_cluster_id': 'unknown',
                'review_key': key
            })
    
    # Create DataFrames for each matrix type and save to CSV
    for matrix_type in ['text', 'theme', 'overall']:
        matrix = matrices[matrix_type]
        
        # Create column names
        columns = [f'I{i}' for i in range(13)]  # I0 to I12
        
        # Create DataFrame
        df = pd.DataFrame(matrix, columns=columns)
        
        # Add metadata columns
        df.insert(0, 'review_key', [pk['review_key'] for pk in parsed_keys])
        df.insert(1, 'cluster_id', [pk['cluster_id'] for pk in parsed_keys])
        df.insert(2, 'micro_cluster_id', [pk['micro_cluster_id'] for pk in parsed_keys])
        
        # Save to CSV
        csv_path = output_dir / f"delta_matrix_{matrix_type}.csv"
        df.to_csv(csv_path, index=False)
        print(f"  Saved {csv_path}")
    
    # Save all matrices to Excel in one file (if openpyxl is available)
    excel_path = output_dir / "delta_matrices_all.xlsx"
    try:
        print(f"  Saving all matrices to {excel_path}...")
        with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
            # Write each matrix type
            for matrix_type in ['text', 'theme', 'overall']:
                matrix = matrices[matrix_type]
                columns = [f'I{i}' for i in range(13)]  # I0 to I12
                df = pd.DataFrame(matrix, columns=columns)
                df.insert(0, 'review_key', [pk['review_key'] for pk in parsed_keys])
                df.insert(1, 'cluster_id', [pk['cluster_id'] for pk in parsed_keys])
                df.insert(2, 'micro_cluster_id', [pk['micro_cluster_id'] for pk in parsed_keys])
                df.to_excel(writer, sheet_name=f'{matrix_type.upper()}_DELTA', index=False)
            
            # Write metrics
            metrics_df.to_excel(writer, sheet_name='CONVERGENCE_METRICS', index=False)
        
        print(f"  Saved {excel_path}")
    except ImportError:
        print(f"  WARNING: openpyxl not installed. Skipping Excel export.")
        print(f"  Install with: pip install openpyxl")
    except Exception as e:
        print(f"  WARNING: Error saving Excel file: {e}")
        print(f"  CSV files are available as alternative.")
    
    # Also save metrics to CSV
    metrics_csv = output_dir / "convergence_metrics.csv"
    metrics_df.to_csv(metrics_csv, index=False)
    print(f"  Saved {metrics_csv}")
    print(f"  Saved {excel_path}")

def main():
    """Main execution function."""
    # Set base path
    base_path = Path("/media/samanvitha/Seagate HDD/vectorial_sapiens_v4/07_sgo_training/artifacts/sgo_training_results_v6")
    
    if not base_path.exists():
        print(f"Error: Base path does not exist: {base_path}")
        return
    
    print("="*70)
    print("DELTA CONVERGENCE MATRIX BUILDER")
    print("="*70)
    
    # Step 1: Load I0 deltas
    sparse_matrices = load_i0_deltas(base_path)
    
    # Step 2: Load I1-I12 deltas from journey files
    reviews_in_journey = load_journey_deltas(base_path, sparse_matrices)
    
    # Step 3: Forward fill and create dense matrices
    matrices, review_keys = forward_fill_matrices(sparse_matrices, max_iteration=12)
    
    # Step 4: Calculate convergence metrics
    metrics_df = calculate_convergence_metrics(matrices, review_keys)
    
    # Step 5: Save to files
    output_dir = base_path.parent / "delta_convergence_analysis"
    save_matrices_to_files(matrices, review_keys, output_dir, metrics_df)
    
    # Print summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"Total reviews processed: {len(review_keys)}")
    print(f"Reviews with journey data: {len(reviews_in_journey)}")
    print(f"\nOverall Convergence (I0 -> I12):")
    for matrix_type in ['text', 'theme', 'overall']:
        matrix = matrices[matrix_type]
        i0_mean = np.mean(matrix[:, 0])
        i12_mean = np.mean(matrix[:, 12])
        improvement = i0_mean - i12_mean
        improvement_pct = (improvement / i0_mean * 100) if i0_mean > 0 else 0.0
        print(f"  {matrix_type.upper()} DELTA: {i0_mean:.4f} -> {i12_mean:.4f} "
              f"(improvement: {improvement:.4f}, {improvement_pct:.2f}%)")
    
    print(f"\nOutput files saved to: {output_dir}")
    print("="*70)

if __name__ == "__main__":
    main()

