#!/usr/bin/env python3
"""
Calculate JSD and WD from Journey Files
========================================

This script:
1. Reads all journey files from the specified directory
2. For each review, extracts:
   - Initial delta and its predicted_themes probabilities
   - Best delta (minimum theme_delta) and its predicted_themes probabilities
3. Calculates JSD and WD for both initial and best predictions against actual
4. Saves results to output directory

Usage:
    python 07_sgo_training/scripts/calculate_jsd_wd_from_journey.py
"""

import json
import numpy as np
import sys
import logging
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict
from tqdm import tqdm
from scipy.stats import entropy

# Add project root to path
BASE_DIR = Path(__file__).parent.parent.parent
sys.path.insert(0, str(BASE_DIR))

try:
    import ot  # Python Optimal Transport library
    HAS_OT = True
except ImportError:
    ot = None
    HAS_OT = False
    logging.warning("POT (Python Optimal Transport) library not found. Install with: pip install POT")
    logging.warning("Falling back to L1 distance approximation")

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global epsilon for numerical stability
EPSILON = 1e-10


def normalize_distribution(theme_dict: Dict[str, float]) -> Dict[str, float]:
    """
    Normalize a theme probability distribution to sum to 1.0.
    
    Args:
        theme_dict: Dictionary mapping theme names to probabilities
        
    Returns:
        Normalized dictionary
    """
    if not theme_dict:
        return {}
    
    total = sum(theme_dict.values())
    if total == 0:
        return {}
    
    return {theme: prob / total for theme, prob in theme_dict.items()}


def align_distributions(
    actual_themes: Dict[str, float],
    predicted_themes: Dict[str, float]
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Align two theme distributions to the same set of themes and normalize.
    
    Args:
        actual_themes: Ground truth theme distribution
        predicted_themes: Predicted theme distribution
        
    Returns:
        Tuple of (actual_array, predicted_array) with aligned themes, normalized to sum to 1.0
    """
    # Get all unique themes from both distributions (union method)
    all_themes = sorted(set(actual_themes.keys()) | set(predicted_themes.keys()))
    
    if not all_themes:
        return np.array([]), np.array([])
    
    # Create aligned arrays
    actual_array = np.array([actual_themes.get(theme, 0.0) for theme in all_themes])
    predicted_array = np.array([predicted_themes.get(theme, 0.0) for theme in all_themes])
    
    # Normalize to ensure they sum to 1.0
    actual_sum = actual_array.sum()
    predicted_sum = predicted_array.sum()
    
    if actual_sum > EPSILON:
        actual_array = actual_array / actual_sum
    else:
        actual_array = np.zeros_like(actual_array)
    
    if predicted_sum > EPSILON:
        predicted_array = predicted_array / predicted_sum
    else:
        predicted_array = np.zeros_like(predicted_array)
    
    return actual_array, predicted_array


def compute_jsd(P: np.ndarray, Q: np.ndarray, epsilon: float = EPSILON) -> float:
    """
    Compute Jensen-Shannon Divergence between two probability distributions.
    
    JSD(P||Q) = 0.5 * KL(P||M) + 0.5 * KL(Q||M)
    where M = 0.5 * (P + Q)
    
    Args:
        P: First probability distribution (numpy array) - already normalized
        Q: Second probability distribution (numpy array) - already normalized
        epsilon: Small value to avoid log(0)
        
    Returns:
        JSD value (float)
    """
    # Add epsilon to avoid zeros (but keep original distribution shape)
    P = P + epsilon
    Q = Q + epsilon
    
    # Re-normalize after adding epsilon to maintain probability distribution
    P = P / P.sum()
    Q = Q / Q.sum()
    
    # Compute mixture distribution
    M = 0.5 * (P + Q)
    
    # Compute KL divergences
    kl_pm = entropy(P, M, base=2)
    kl_qm = entropy(Q, M, base=2)
    
    # JSD is the average of the two KL divergences
    jsd = 0.5 * kl_pm + 0.5 * kl_qm
    
    return float(jsd)


def calculate_wasserstein_1_distance(
    actual_array: np.ndarray,
    predicted_array: np.ndarray
) -> float:
    """
    Calculate Wasserstein-1 distance (p=1) using optimal transport formulation.
    
    Formula: W_1(P, Q) = min_γ sum_i sum_j γ_ij * C_ij
    where:
    - P = predicted distribution (source)
    - Q = actual distribution (target)
    - C_ij = cost matrix: C[i,j] = 1 if i != j, else 0 (unit cost)
    - γ = optimal transport plan
    
    Args:
        actual_array: Q - Ground truth probability distribution (normalized, sums to 1.0)
        predicted_array: P - Predicted probability distribution (normalized, sums to 1.0)
        
    Returns:
        Wasserstein-1 distance (float) using optimal transport
    """
    if len(actual_array) == 0 or len(predicted_array) == 0:
        return float('nan')
    
    if len(actual_array) != len(predicted_array):
        return float('nan')
    
    # Ensure they sum to 1.0 (safety check)
    actual_sum = actual_array.sum()
    predicted_sum = predicted_array.sum()
    
    if actual_sum < EPSILON or predicted_sum < EPSILON:
        return float('nan')
    
    # Normalize again as safety check
    P = predicted_array / predicted_sum  # Predicted distribution (source)
    Q = actual_array / actual_sum  # Actual distribution (target)
    
    n = len(P)
    
    # Build cost matrix: C[i,j] = 1 if i != j, else 0
    cost_matrix = np.ones((n, n)) - np.eye(n)  # 1 everywhere except diagonal (0)
    
    # Use optimal transport to compute Wasserstein-1 distance
    if HAS_OT:
        try:
            wd = ot.emd2(P, Q, cost_matrix, numItermax=1000000)
            return float(wd)
        except Exception as e:
            logger.warning(f"Error in optimal transport calculation: {e}")
            # Fallback to L1 distance
            wd = np.sum(np.abs(P - Q))
            return float(wd)
    else:
        # Fallback: Use L1 distance if POT library is not available
        wd = np.sum(np.abs(P - Q))
        return float(wd)


def calculate_recall_at_k(
    actual_themes: Dict[str, float],
    predicted_themes: Dict[str, float],
    k: int,
    threshold: float = 0.3
) -> float:
    """
    Calculate recall@k: proportion of actual themes found in top-k predicted themes.
    
    Args:
        actual_themes: Actual theme distribution (ground truth)
        predicted_themes: Predicted theme distribution
        k: Number of top predictions to consider
        threshold: Minimum probability threshold for predicted themes (default 0.3)
        
    Returns:
        Recall@k value (float between 0 and 1)
    """
    if not actual_themes or not predicted_themes:
        return 0.0
    
    # Get actual theme names (themes with non-zero probability)
    actual_theme_names = set(theme for theme, prob in actual_themes.items() if prob > EPSILON)
    
    if not actual_theme_names:
        return 0.0
    
    # Sort predicted themes by probability (descending)
    sorted_predictions = sorted(
        [(theme, prob) for theme, prob in predicted_themes.items() if prob >= threshold],
        key=lambda x: x[1],
        reverse=True
    )
    
    # Get top-k predicted theme names
    top_k_predicted = set(theme for theme, _ in sorted_predictions[:k])
    
    # Calculate recall: intersection / total actual themes
    intersection = len(actual_theme_names & top_k_predicted)
    recall = intersection / len(actual_theme_names) if actual_theme_names else 0.0
    
    return float(recall)


def find_best_delta(journey_entry: Dict[str, Any]) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
    """
    Find the best delta (minimum theme_delta) from correction_journey.
    
    Args:
        journey_entry: Journey entry for a review
        
    Returns:
        Tuple of (best_deltas_dict, best_prediction_data) or (None, None) if not found
    """
    initial_deltas = journey_entry.get('initial_prediction_deltas', {})
    initial_theme_delta = initial_deltas.get('theme_delta', float('inf'))
    initial_overall_delta = initial_deltas.get('overall_delta', float('inf'))
    initial_data = journey_entry.get('initial_prediction_data', {})
    
    # Start with initial as best
    best_theme_delta = initial_theme_delta
    best_overall_delta = initial_overall_delta
    best_prediction = initial_data
    
    # Check all corrections in the journey
    correction_journey = journey_entry.get('correction_journey', [])
    
    for correction in correction_journey:
        current_theme_delta = correction.get('new_theme_delta')
        current_overall_delta = correction.get('new_overall_delta')
        current_pred_data = correction.get('corrected_prediction_data')
        
        if current_pred_data is None:
            continue
        
        # Select best based on: theme_delta improvement OR (same theme_delta AND overall_delta improvement)
        theme_improved = current_theme_delta is not None and current_theme_delta < best_theme_delta
        theme_same_but_overall_improved = (
            current_theme_delta is not None and 
            abs(current_theme_delta - best_theme_delta) < 0.001 and  # Essentially same theme_delta
            current_overall_delta is not None and 
            current_overall_delta < best_overall_delta  # But overall improved
        )
        
        if theme_improved or theme_same_but_overall_improved:
            best_theme_delta = current_theme_delta if current_theme_delta is not None else best_theme_delta
            best_overall_delta = current_overall_delta if current_overall_delta is not None else best_overall_delta
            best_prediction = current_pred_data
    
    best_deltas = {
        'theme_delta': best_theme_delta,
        'overall_delta': best_overall_delta,
        'text_delta': None  # Not always available in corrections
    }
    
    return best_deltas, best_prediction


def process_journey_file(journey_file: Path) -> List[Dict[str, Any]]:
    """
    Process a single journey file and extract JSD/WD metrics.
    
    Args:
        journey_file: Path to journey JSON file
        
    Returns:
        List of results for each review in the file
    """
    results = []
    
    # Extract cluster and micro from path
    file_path_str = str(journey_file)
    cluster_id = None
    micro_id = None
    
    cluster_match = re.search(r'cluster_(\d+)', file_path_str)
    micro_match = re.search(r'micro_(\d+)', file_path_str)
    
    if cluster_match:
        cluster_id = f"cluster_{cluster_match.group(1)}"
    if micro_match:
        micro_id = f"micro_{micro_match.group(1)}"
    
    try:
        with open(journey_file, 'r', encoding='utf-8') as f:
            journey_data = json.load(f)
    except Exception as e:
        logger.error(f"Error reading journey file {journey_file}: {e}")
        return results
    
    for review_key, journey_entry in journey_data.items():
        try:
            # Extract actual review data
            actual_data = journey_entry.get('actual_review_data', {})
            actual_themes = actual_data.get('predicted_themes', {})
            
            # Handle list format for actual themes
            if isinstance(actual_themes, list):
                if not actual_themes:
                    continue
                actual_themes = {theme: 1.0 / len(actual_themes) for theme in actual_themes}
            
            if not actual_themes:
                continue
            
            # Normalize actual themes
            actual_themes = normalize_distribution(actual_themes)
            if not actual_themes:
                continue
            
            # Extract initial prediction
            initial_deltas = journey_entry.get('initial_prediction_deltas', {})
            initial_data = journey_entry.get('initial_prediction_data', {})
            initial_predicted_themes = initial_data.get('predicted_themes', {})
            
            # Handle list format
            if isinstance(initial_predicted_themes, list):
                if not initial_predicted_themes:
                    continue
                initial_predicted_themes = {theme: 1.0 / len(initial_predicted_themes) for theme in initial_predicted_themes}
            
            if not initial_predicted_themes:
                continue
            
            # Normalize initial predicted themes
            initial_predicted_themes = normalize_distribution(initial_predicted_themes)
            if not initial_predicted_themes:
                continue
            
            # Align and calculate metrics for initial
            initial_actual_array, initial_predicted_array = align_distributions(actual_themes, initial_predicted_themes)
            
            if len(initial_actual_array) == 0 or len(initial_predicted_array) == 0:
                continue
            
            initial_jsd = compute_jsd(initial_predicted_array, initial_actual_array)
            initial_wd = calculate_wasserstein_1_distance(initial_actual_array, initial_predicted_array)
            
            # Calculate recall@max(3,k) and recall@max(4,k) for initial (pre-SGO)
            num_actual = len(actual_themes)
            k_initial_3 = max(3, num_actual)
            k_initial_4 = max(4, num_actual)
            initial_recall_3k = calculate_recall_at_k(actual_themes, initial_predicted_themes, k_initial_3, threshold=0.3)
            initial_recall_4k = calculate_recall_at_k(actual_themes, initial_predicted_themes, k_initial_4, threshold=0.3)
            
            # Find best delta
            best_deltas, best_prediction = find_best_delta(journey_entry)
            
            best_jsd = None
            best_wd = None
            best_predicted_themes = None
            best_recall_3k = None
            best_recall_4k = None
            
            if best_prediction:
                best_predicted_themes = best_prediction.get('predicted_themes', {})
                
                # Handle list format
                if isinstance(best_predicted_themes, list):
                    if best_predicted_themes:
                        best_predicted_themes = {theme: 1.0 / len(best_predicted_themes) for theme in best_predicted_themes}
                    else:
                        best_predicted_themes = {}
                
                if best_predicted_themes:
                    # Normalize best predicted themes
                    best_predicted_themes = normalize_distribution(best_predicted_themes)
                    
                    if best_predicted_themes:
                        # Align and calculate metrics for best
                        best_actual_array, best_predicted_array = align_distributions(actual_themes, best_predicted_themes)
                        
                        if len(best_actual_array) > 0 and len(best_predicted_array) > 0:
                            best_jsd = compute_jsd(best_predicted_array, best_actual_array)
                            best_wd = calculate_wasserstein_1_distance(best_actual_array, best_predicted_array)
                        
                        # Calculate recall@max(3,k) and recall@max(4,k) for best (post-SGO)
                        k_best_3 = max(3, num_actual)
                        k_best_4 = max(4, num_actual)
                        best_recall_3k = calculate_recall_at_k(actual_themes, best_predicted_themes, k_best_3, threshold=0.3)
                        best_recall_4k = calculate_recall_at_k(actual_themes, best_predicted_themes, k_best_4, threshold=0.3)
            
            # Create result entry
            result = {
                'review_key': review_key,
                'user_id': journey_entry.get('user_id', ''),
                'review_idx': journey_entry.get('review_idx', 0),
                'category': journey_entry.get('category', 'Unknown'),
                'cluster_id': cluster_id,
                'micro_id': micro_id,
                'initial_deltas': {
                    'theme_delta': initial_deltas.get('theme_delta'),
                    'text_delta': initial_deltas.get('text_delta'),
                    'overall_delta': initial_deltas.get('overall_delta')
                },
                'initial_jsd': initial_jsd if not np.isnan(initial_jsd) else None,
                'initial_wd': initial_wd if not np.isnan(initial_wd) else None,
                'initial_recall_at_max3k': initial_recall_3k,
                'initial_recall_at_max4k': initial_recall_4k,
                'best_deltas': best_deltas,
                'best_jsd': best_jsd if best_jsd is not None and not np.isnan(best_jsd) else None,
                'best_wd': best_wd if best_wd is not None and not np.isnan(best_wd) else None,
                'best_recall_at_max3k': best_recall_3k,
                'best_recall_at_max4k': best_recall_4k,
                'improvement_jsd': (initial_jsd - best_jsd) if best_jsd is not None and not np.isnan(best_jsd) else None,
                'improvement_wd': (initial_wd - best_wd) if best_wd is not None and not np.isnan(best_wd) else None,
                'improvement_recall_at_max3k': (best_recall_3k - initial_recall_3k) if best_recall_3k is not None else None,
                'improvement_recall_at_max4k': (best_recall_4k - initial_recall_4k) if best_recall_4k is not None else None,
                'num_actual_themes': len(actual_themes),
                'num_initial_predicted_themes': len(initial_predicted_themes),
                'num_best_predicted_themes': len(best_predicted_themes) if best_predicted_themes else 0
            }
            
            results.append(result)
            
        except Exception as e:
            logger.warning(f"Error processing review {review_key} in {journey_file}: {e}")
            continue
    
    return results


def process_all_journey_files(journey_dir: Path) -> List[Dict[str, Any]]:
    """
    Process all journey files in the directory.
    
    Args:
        journey_dir: Directory containing journey files
        
    Returns:
        List of all results
    """
    all_results = []
    
    # Find all journey JSON files
    journey_files = list(journey_dir.rglob("*_journey.json"))
    
    logger.info(f"Found {len(journey_files)} journey files")
    
    for journey_file in tqdm(journey_files, desc="Processing journey files"):
        results = process_journey_file(journey_file)
        all_results.extend(results)
    
    logger.info(f"Processed {len(all_results)} reviews from {len(journey_files)} journey files")
    
    return all_results


def calculate_statistics(values: List[float]) -> Dict[str, Optional[float]]:
    """Calculate mean, median, std, min, max for a list of values."""
    if not values:
        return {
            'mean': None,
            'median': None,
            'std': None,
            'min': None,
            'max': None,
            'count': 0
        }
    return {
        'mean': float(np.mean(values)),
        'median': float(np.median(values)),
        'std': float(np.std(values)),
        'min': float(np.min(values)),
        'max': float(np.max(values)),
        'count': len(values)
    }


def save_results(results: List[Dict[str, Any]], output_dir: Path):
    """
    Save results to JSON file and generate summary statistics per micro cluster.
    
    Args:
        results: List of result dictionaries
        output_dir: Output directory
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save all results
    output_file = output_dir / "journey_jsd_wd_results.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    logger.info(f"Saved results to: {output_file}")
    
    # Group by micro cluster
    micro_cluster_results = defaultdict(list)
    for result in results:
        micro_id = result.get('micro_id')
        if micro_id:
            micro_cluster_results[micro_id].append(result)
    
    # Calculate statistics for each micro cluster
    micro_cluster_stats = {}
    
    for micro_id, micro_results in sorted(micro_cluster_results.items()):
        # Extract all metrics
        initial_jsds = [r['initial_jsd'] for r in micro_results if r.get('initial_jsd') is not None]
        initial_wds = [r['initial_wd'] for r in micro_results if r.get('initial_wd') is not None]
        initial_recalls_3k = [r['initial_recall_at_max3k'] for r in micro_results if r.get('initial_recall_at_max3k') is not None]
        initial_recalls_4k = [r['initial_recall_at_max4k'] for r in micro_results if r.get('initial_recall_at_max4k') is not None]
        
        best_jsds = [r['best_jsd'] for r in micro_results if r.get('best_jsd') is not None]
        best_wds = [r['best_wd'] for r in micro_results if r.get('best_wd') is not None]
        best_recalls_3k = [r['best_recall_at_max3k'] for r in micro_results if r.get('best_recall_at_max3k') is not None]
        best_recalls_4k = [r['best_recall_at_max4k'] for r in micro_results if r.get('best_recall_at_max4k') is not None]
        
        improvement_jsds = [r['improvement_jsd'] for r in micro_results if r.get('improvement_jsd') is not None]
        improvement_wds = [r['improvement_wd'] for r in micro_results if r.get('improvement_wd') is not None]
        improvement_recall_3k = [r['improvement_recall_at_max3k'] for r in micro_results if r.get('improvement_recall_at_max3k') is not None]
        improvement_recall_4k = [r['improvement_recall_at_max4k'] for r in micro_results if r.get('improvement_recall_at_max4k') is not None]
        
        micro_cluster_stats[micro_id] = {
            'total_reviews': len(micro_results),
            'cluster_id': micro_results[0].get('cluster_id') if micro_results else None,
            'initial': {
                'jsd': calculate_statistics(initial_jsds),
                'wd': calculate_statistics(initial_wds),
                'recall_at_max3k': calculate_statistics(initial_recalls_3k),
                'recall_at_max4k': calculate_statistics(initial_recalls_4k)
            },
            'best': {
                'jsd': calculate_statistics(best_jsds),
                'wd': calculate_statistics(best_wds),
                'recall_at_max3k': calculate_statistics(best_recalls_3k),
                'recall_at_max4k': calculate_statistics(best_recalls_4k)
            },
            'improvement': {
                'jsd': calculate_statistics(improvement_jsds),
                'wd': calculate_statistics(improvement_wds),
                'recall_at_max3k': calculate_statistics(improvement_recall_3k),
                'recall_at_max4k': calculate_statistics(improvement_recall_4k)
            }
        }
    
    # Calculate overall summary statistics
    initial_jsds = [r['initial_jsd'] for r in results if r.get('initial_jsd') is not None]
    initial_wds = [r['initial_wd'] for r in results if r.get('initial_wd') is not None]
    initial_recalls_3k = [r['initial_recall_at_max3k'] for r in results if r.get('initial_recall_at_max3k') is not None]
    initial_recalls_4k = [r['initial_recall_at_max4k'] for r in results if r.get('initial_recall_at_max4k') is not None]
    best_jsds = [r['best_jsd'] for r in results if r.get('best_jsd') is not None]
    best_wds = [r['best_wd'] for r in results if r.get('best_wd') is not None]
    best_recalls_3k = [r['best_recall_at_max3k'] for r in results if r.get('best_recall_at_max3k') is not None]
    best_recalls_4k = [r['best_recall_at_max4k'] for r in results if r.get('best_recall_at_max4k') is not None]
    improvement_jsds = [r['improvement_jsd'] for r in results if r.get('improvement_jsd') is not None]
    improvement_wds = [r['improvement_wd'] for r in results if r.get('improvement_wd') is not None]
    improvement_recall_3k = [r['improvement_recall_at_max3k'] for r in results if r.get('improvement_recall_at_max3k') is not None]
    improvement_recall_4k = [r['improvement_recall_at_max4k'] for r in results if r.get('improvement_recall_at_max4k') is not None]
    
    summary = {
        'total_reviews': len(results),
        'total_micro_clusters': len(micro_cluster_stats),
        'overall': {
            'initial': {
                'jsd': calculate_statistics(initial_jsds),
                'wd': calculate_statistics(initial_wds),
                'recall_at_max3k': calculate_statistics(initial_recalls_3k),
                'recall_at_max4k': calculate_statistics(initial_recalls_4k)
            },
            'best': {
                'jsd': calculate_statistics(best_jsds),
                'wd': calculate_statistics(best_wds),
                'recall_at_max3k': calculate_statistics(best_recalls_3k),
                'recall_at_max4k': calculate_statistics(best_recalls_4k)
            },
            'improvement': {
                'jsd': calculate_statistics(improvement_jsds),
                'wd': calculate_statistics(improvement_wds),
                'recall_at_max3k': calculate_statistics(improvement_recall_3k),
                'recall_at_max4k': calculate_statistics(improvement_recall_4k)
            }
        },
        'per_micro_cluster': micro_cluster_stats
    }
    
    summary_file = output_dir / "journey_jsd_wd_summary.json"
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    logger.info(f"Saved summary to: {summary_file}")
    
    # Print summary
    logger.info("\n" + "=" * 80)
    logger.info("JSD and WD Summary from Journey Files")
    logger.info("=" * 80)
    logger.info(f"\nOverall Statistics (across all {len(micro_cluster_stats)} micro clusters):")
    if initial_jsds:
        logger.info(f"  Initial JSD: Mean = {np.mean(initial_jsds):.6f}, Median = {np.median(initial_jsds):.6f}")
    if initial_wds:
        logger.info(f"  Initial WD: Mean = {np.mean(initial_wds):.6f}, Median = {np.median(initial_wds):.6f}")
    if initial_recalls_3k:
        logger.info(f"  Initial Recall@max(3,k): Mean = {np.mean(initial_recalls_3k):.6f}, Median = {np.median(initial_recalls_3k):.6f}")
    if initial_recalls_4k:
        logger.info(f"  Initial Recall@max(4,k): Mean = {np.mean(initial_recalls_4k):.6f}, Median = {np.median(initial_recalls_4k):.6f}")
    if best_jsds:
        logger.info(f"  Best JSD: Mean = {np.mean(best_jsds):.6f}, Median = {np.median(best_jsds):.6f}")
    if best_wds:
        logger.info(f"  Best WD: Mean = {np.mean(best_wds):.6f}, Median = {np.median(best_wds):.6f}")
    if best_recalls_3k:
        logger.info(f"  Best Recall@max(3,k): Mean = {np.mean(best_recalls_3k):.6f}, Median = {np.median(best_recalls_3k):.6f}")
    if best_recalls_4k:
        logger.info(f"  Best Recall@max(4,k): Mean = {np.mean(best_recalls_4k):.6f}, Median = {np.median(best_recalls_4k):.6f}")
    if improvement_jsds:
        logger.info(f"  JSD Improvement: Mean = {np.mean(improvement_jsds):.6f}, Median = {np.median(improvement_jsds):.6f}")
    if improvement_wds:
        logger.info(f"  WD Improvement: Mean = {np.mean(improvement_wds):.6f}, Median = {np.median(improvement_wds):.6f}")
    if improvement_recall_3k:
        logger.info(f"  Recall@max(3,k) Improvement: Mean = {np.mean(improvement_recall_3k):.6f}, Median = {np.median(improvement_recall_3k):.6f}")
    if improvement_recall_4k:
        logger.info(f"  Recall@max(4,k) Improvement: Mean = {np.mean(improvement_recall_4k):.6f}, Median = {np.median(improvement_recall_4k):.6f}")
    
    logger.info(f"\nPer-Micro-Cluster Statistics:")
    for micro_id, stats in sorted(micro_cluster_stats.items()):
        logger.info(f"\n  {micro_id} ({stats['total_reviews']} reviews):")
        if stats['initial']['jsd']['mean'] is not None:
            logger.info(f"    Initial JSD: {stats['initial']['jsd']['mean']:.6f} (median: {stats['initial']['jsd']['median']:.6f})")
        if stats['initial']['wd']['mean'] is not None:
            logger.info(f"    Initial WD: {stats['initial']['wd']['mean']:.6f} (median: {stats['initial']['wd']['median']:.6f})")
        if stats['initial']['recall_at_max3k']['mean'] is not None:
            logger.info(f"    Initial Recall@max(3,k): {stats['initial']['recall_at_max3k']['mean']:.6f} (median: {stats['initial']['recall_at_max3k']['median']:.6f})")
        if stats['initial']['recall_at_max4k']['mean'] is not None:
            logger.info(f"    Initial Recall@max(4,k): {stats['initial']['recall_at_max4k']['mean']:.6f} (median: {stats['initial']['recall_at_max4k']['median']:.6f})")
        if stats['best']['jsd']['mean'] is not None:
            logger.info(f"    Best JSD: {stats['best']['jsd']['mean']:.6f} (median: {stats['best']['jsd']['median']:.6f})")
        if stats['best']['wd']['mean'] is not None:
            logger.info(f"    Best WD: {stats['best']['wd']['mean']:.6f} (median: {stats['best']['wd']['median']:.6f})")
        if stats['best']['recall_at_max3k']['mean'] is not None:
            logger.info(f"    Best Recall@max(3,k): {stats['best']['recall_at_max3k']['mean']:.6f} (median: {stats['best']['recall_at_max3k']['median']:.6f})")
        if stats['best']['recall_at_max4k']['mean'] is not None:
            logger.info(f"    Best Recall@max(4,k): {stats['best']['recall_at_max4k']['mean']:.6f} (median: {stats['best']['recall_at_max4k']['median']:.6f})")
        if stats['improvement']['jsd']['mean'] is not None:
            logger.info(f"    JSD Improvement: {stats['improvement']['jsd']['mean']:.6f} (median: {stats['improvement']['jsd']['median']:.6f})")
        if stats['improvement']['wd']['mean'] is not None:
            logger.info(f"    WD Improvement: {stats['improvement']['wd']['mean']:.6f} (median: {stats['improvement']['wd']['median']:.6f})")
        if stats['improvement']['recall_at_max3k']['mean'] is not None:
            logger.info(f"    Recall@max(3,k) Improvement: {stats['improvement']['recall_at_max3k']['mean']:.6f} (median: {stats['improvement']['recall_at_max3k']['median']:.6f})")
        if stats['improvement']['recall_at_max4k']['mean'] is not None:
            logger.info(f"    Recall@max(4,k) Improvement: {stats['improvement']['recall_at_max4k']['mean']:.6f} (median: {stats['improvement']['recall_at_max4k']['median']:.6f})")


def main():
    """Main execution function."""
    logger.info("=" * 80)
    logger.info("JSD and WD Calculation from Journey Files")
    logger.info("=" * 80)
    
    # Set paths
    journey_dir = BASE_DIR / "07_sgo_training" / "artifacts" / "sgo_training_results_v3_sample" / "_journey"
    output_dir = BASE_DIR / "07_sgo_training" / "artifacts" / "sgo_training_results_v3_sample" / "journey_jsd_wd_analysis"
    
    if not journey_dir.exists():
        logger.error(f"Journey directory not found: {journey_dir}")
        return
    
    logger.info(f"Journey directory: {journey_dir}")
    logger.info(f"Output directory: {output_dir}")
    
    # Process all journey files
    results = process_all_journey_files(journey_dir)
    
    if not results:
        logger.warning("No results found. Exiting.")
        return
    
    # Save results
    save_results(results, output_dir)
    
    logger.info("\n" + "=" * 80)
    logger.info("JSD and WD Calculation Complete")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()

