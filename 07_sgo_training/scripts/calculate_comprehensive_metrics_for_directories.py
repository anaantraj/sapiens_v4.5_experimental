#!/usr/bin/env python3
"""
Calculate JSD, WD, Recall@max(3,k), and Recall@max(4,k) for artifact directories.
Uses threshold-based ground truth extraction from topic_probabilities_before_normalisation 
(like calculate_recall_at_k.py which uses topic_probs_before_normalization).
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

EPSILON = 1e-10


def normalize_distribution(theme_dict: Dict[str, float]) -> Dict[str, float]:
    """Normalize a theme probability distribution to sum to 1.0."""
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
    """Align two theme distributions to the same set of themes and normalize."""
    all_themes = sorted(set(actual_themes.keys()) | set(predicted_themes.keys()))
    if not all_themes:
        return np.array([]), np.array([])
    
    actual_array = np.array([actual_themes.get(theme, 0.0) for theme in all_themes])
    predicted_array = np.array([predicted_themes.get(theme, 0.0) for theme in all_themes])
    
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
    """Compute Jensen-Shannon Divergence between two probability distributions."""
    if len(P) == 0 or len(Q) == 0:
        return float('nan')
    if len(P) != len(Q):
        return float('nan')
    
    P = P + epsilon
    Q = Q + epsilon
    P = P / P.sum()
    Q = Q / Q.sum()
    M = 0.5 * (P + Q)
    
    kl_pm = entropy(P, M, base=2)
    kl_qm = entropy(Q, M, base=2)
    jsd = 0.5 * kl_pm + 0.5 * kl_qm
    
    return float(jsd)


def calculate_wasserstein_1_distance(
    actual_array: np.ndarray,
    predicted_array: np.ndarray
) -> float:
    """Calculate Wasserstein-1 distance using optimal transport."""
    if len(actual_array) == 0 or len(predicted_array) == 0:
        return float('nan')
    if len(actual_array) != len(predicted_array):
        return float('nan')
    
    actual_sum = actual_array.sum()
    predicted_sum = predicted_array.sum()
    
    if actual_sum < EPSILON or predicted_sum < EPSILON:
        return float('nan')
    
    P = predicted_array / predicted_sum
    Q = actual_array / actual_sum
    n = len(P)
    cost_matrix = np.ones((n, n)) - np.eye(n)
    
    if HAS_OT:
        try:
            wd = ot.emd2(P, Q, cost_matrix, numItermax=1000000)
            return float(wd)
        except Exception as e:
            logger.warning(f"Error in optimal transport calculation: {e}")
            wd = np.sum(np.abs(P - Q))
            return float(wd)
    else:
        wd = np.sum(np.abs(P - Q))
        return float(wd)


def get_ground_truth_themes(actual: Dict[str, Any], threshold: float = 0.5) -> set:
    """
    Extract ground truth themes from topic_probabilities_before_normalisation using threshold.
    This matches the logic from calculate_recall_at_k.py which uses topic_probs_before_normalization.
    
    Priority (no fallbacks):
    1. topic_probabilities_before_normalisation (British spelling)
    2. topic_probs_before_normalization (American spelling)
    3. topic_probabilities_without_normalisation
    
    Args:
        actual: Actual review dictionary containing topic probability fields
        threshold: Minimum prob_yes to include theme (default: 0.5)
        
    Returns:
        Set of theme names that are in ground truth (prob_yes >= threshold)
    """
    # Priority 1: topic_probabilities_before_normalisation (British spelling)
    topic_probs_before = actual.get('topic_probabilities_before_normalisation', {})
    
    # Priority 2: topic_probs_before_normalization (American spelling)
    if not topic_probs_before:
        topic_probs_before = actual.get('topic_probs_before_normalization', {})
    
    # Priority 3: topic_probabilities_without_normalisation
    if not topic_probs_before:
        topic_probs_before = actual.get('topic_probabilities_without_normalisation', {})
    
    if not topic_probs_before:
        return set()
    
    if isinstance(topic_probs_before, list):
        return set(topic_probs_before)
    
    ground_truth_themes = set()
    for theme, prob_data in topic_probs_before.items():
        if isinstance(prob_data, dict):
            # Handle nested dict format: {"prob_yes": float, "prob_no": float}
            prob_yes = prob_data.get('prob_yes')
            if prob_yes is not None and isinstance(prob_yes, (int, float)) and prob_yes >= threshold:
                ground_truth_themes.add(theme)
        elif isinstance(prob_data, (int, float)):
            # Handle direct probability value (before normalization)
            if not np.isnan(prob_data) and np.isfinite(prob_data) and prob_data >= threshold:
                ground_truth_themes.add(theme)
    
    return ground_truth_themes


def get_top_k_predicted(predicted_themes: Dict[str, float], k: int) -> List[str]:
    """Get top-k predicted themes sorted by probability."""
    if not predicted_themes:
        return []
    
    if isinstance(predicted_themes, list):
        return predicted_themes[:k]
    
    sorted_themes = sorted(
        predicted_themes.items(),
        key=lambda x: x[1],
        reverse=True
    )
    return [theme for theme, _ in sorted_themes[:k]]


def calculate_recall_at_k(
    predicted_themes: Dict[str, float],
    ground_truth_themes: set,
    k: int
) -> float:
    """
    Calculate Recall@k.
    
    Recall@k = |top_k_predicted ∩ ground_truth| / |ground_truth|
    """
    if not ground_truth_themes:
        return 0.0
    
    top_k_predicted = set(get_top_k_predicted(predicted_themes, k))
    intersection = top_k_predicted & ground_truth_themes
    
    recall = len(intersection) / len(ground_truth_themes)
    return recall


def calculate_recall_at_max3k(
    predicted_themes: Dict[str, float],
    ground_truth_themes: set
) -> Tuple[float, int, int]:
    """Calculate Recall@max(3, k) where k is the number of ground truth themes."""
    if not ground_truth_themes:
        return 0.0, 0, 0
    
    k = len(ground_truth_themes)
    top_k = max(3, k)
    
    recall = calculate_recall_at_k(predicted_themes, ground_truth_themes, top_k)
    return recall, k, top_k


def calculate_recall_at_max4k(
    predicted_themes: Dict[str, float],
    ground_truth_themes: set
) -> Tuple[float, int, int]:
    """Calculate Recall@max(4, k) where k is the number of ground truth themes."""
    if not ground_truth_themes:
        return 0.0, 0, 0
    
    k = len(ground_truth_themes)
    top_k = max(4, k)
    
    recall = calculate_recall_at_k(predicted_themes, ground_truth_themes, top_k)
    return recall, k, top_k


def calculate_metrics_for_review(review: Dict[str, Any], threshold: float = 0.5, actual_for_recall: Optional[Dict[str, Any]] = None) -> Tuple[Optional[float], Optional[float], Optional[float], Optional[float], Optional[int], Optional[int]]:
    """
    Calculate JSD, WD, Recall@max(3,k), and Recall@max(4,k) for a single review.
    
    Args:
        review: Review dictionary with prediction and actual
        threshold: Threshold for ground truth extraction
        actual_for_recall: Optional ground truth data from separate ground truth file (overrides review['actual'])
    
    Returns:
        Tuple of (jsd, wd, recall_3k, recall_4k, k_actual, top_k_3) or (None, ...) if calculation failed
    """
    try:
        prediction = review.get('prediction', {})
        actual = review.get('actual', {})
        
        # Use provided actual_for_recall if available, otherwise use actual from review
        if actual_for_recall is not None:
            actual_for_gt = actual_for_recall
        else:
            actual_for_gt = actual
        
        if not prediction or not actual:
            return None, None, None, None, None, None
        
        # Get theme distributions
        actual_themes = actual.get('topic_probabilities', {})
        predicted_themes = prediction.get('predicted_themes', {})
        
        # Handle different formats (list vs dict)
        if isinstance(actual_themes, list):
            if not actual_themes:
                return None, None, None, None, None, None
            actual_themes = {theme: 1.0 / len(actual_themes) for theme in actual_themes}
        
        if isinstance(predicted_themes, list):
            if not predicted_themes:
                return None, None, None, None, None, None
            predicted_themes = {theme: 1.0 / len(predicted_themes) for theme in predicted_themes}
        
        if not actual_themes or not predicted_themes:
            return None, None, None, None, None, None
        
        # For JSD/WD: use actual_themes as distribution (normalized)
        # Extract direct probabilities if it's a nested dict
        actual_themes_for_jsd = {}
        if isinstance(actual_themes, dict):
            for theme, prob_data in actual_themes.items():
                if isinstance(prob_data, dict):
                    # Use prob_yes if available, otherwise use 0.5 as default
                    prob_yes = prob_data.get('prob_yes')
                    if prob_yes is not None and isinstance(prob_yes, (int, float)):
                        actual_themes_for_jsd[theme] = float(prob_yes)
                    else:
                        # Fallback: use 0.5 if prob_yes not available
                        actual_themes_for_jsd[theme] = 0.5
                elif isinstance(prob_data, (int, float)):
                    actual_themes_for_jsd[theme] = float(prob_data)
        
        if not actual_themes_for_jsd:
            # Fallback: if actual_themes is already a simple dict, use it directly
            if isinstance(actual_themes, dict):
                actual_themes_for_jsd = {k: float(v) if isinstance(v, (int, float)) else 0.5 
                                        for k, v in actual_themes.items()}
            else:
                return None, None, None, None, None, None
        
        # Normalize distributions for JSD/WD
        actual_themes_normalized = normalize_distribution(actual_themes_for_jsd)
        predicted_themes_normalized = normalize_distribution(predicted_themes)
        
        if not actual_themes_normalized or not predicted_themes_normalized:
            return None, None, None, None, None, None
        
        actual_array, predicted_array = align_distributions(actual_themes_normalized, predicted_themes_normalized)
        
        if len(actual_array) == 0 or len(predicted_array) == 0:
            return None, None, None, None, None, None
        
        # Calculate JSD
        jsd = compute_jsd(actual_array, predicted_array)
        
        # Calculate WD
        wd = calculate_wasserstein_1_distance(actual_array, predicted_array)
        
        # Check for NaN or infinite values
        if np.isnan(jsd) or not np.isfinite(jsd):
            jsd = None
        if np.isnan(wd) or not np.isfinite(wd):
            wd = None
        
        # Calculate Recall@max(3,k) and Recall@max(4,k) using threshold-based ground truth
        # Use topic_probabilities_before_normalisation (before normalization) for threshold
        # Use actual_for_gt which may come from ground truth file
        gt_themes = get_ground_truth_themes(actual_for_gt, threshold=threshold)
        
        if gt_themes:
            recall_3k, k_actual, top_k_3 = calculate_recall_at_max3k(predicted_themes, gt_themes)
            recall_4k, k_actual, top_k_4 = calculate_recall_at_max4k(predicted_themes, gt_themes)
        else:
            recall_3k = None
            recall_4k = None
            k_actual = 0
            top_k_3 = 0
            top_k_4 = 0
        
        return jsd, wd, recall_3k, recall_4k, k_actual, top_k_3
        
    except Exception as e:
        logger.debug(f"Error calculating metrics for review: {e}")
        return None, None, None, None, None, None


def load_ground_truth_file(cluster_id: str, micro_id: str) -> Dict[str, Dict[str, Any]]:
    """Load ground truth file from ground_truth_test_micro_cluster_details_converted."""
    possible_gt_dirs = [
        BASE_DIR / "ground_truth_test_micro_cluster_details_converted",
        Path("ground_truth_test_micro_cluster_details_converted"),
        Path("/media/samanvitha/Seagate HDD/vectorial_sapiens_v4/ground_truth_test_micro_cluster_details_converted"),
    ]
    
    ground_truth_file = None
    for gt_dir in possible_gt_dirs:
        if gt_dir.exists():
            gt_file = gt_dir / cluster_id / f"{micro_id}_details.json"
            if gt_file.exists():
                ground_truth_file = gt_file
                break
    
    if not ground_truth_file:
        logger.debug(f"Ground truth file not found for {cluster_id}/{micro_id}")
        return {}
    
    try:
        with open(ground_truth_file, 'r', encoding='utf-8') as f:
            gt_data = json.load(f)
        
        lookup = {}
        
        # Handle members_grouped_by_user structure - user_id is the key
        if 'members_grouped_by_user' in gt_data and isinstance(gt_data['members_grouped_by_user'], dict):
            for user_id_key, user_reviews in gt_data['members_grouped_by_user'].items():
                if isinstance(user_reviews, list):
                    for review in user_reviews:
                        user_id = user_id_key
                        product_description = review.get('product_description', '').strip()
                        
                        # Primary key: user_id + product_description
                        if user_id and product_description:
                            key = f"{user_id}_{product_description}"
                            lookup[key] = review
        
        logger.debug(f"Loaded {len(lookup)} ground truth reviews from {ground_truth_file}")
        return lookup
    except Exception as e:
        logger.warning(f"Error loading ground truth file {ground_truth_file}: {e}")
        return {}


def process_delta_file(file_path: Path, threshold: float = 0.5) -> Dict[str, Any]:
    """Process a single delta file and calculate metrics for all reviews."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        logger.error(f"Error reading file {file_path}: {e}")
        return {}
    
    cluster_id = data.get('cluster_id', 'unknown')
    micro_cluster_id = data.get('micro_cluster_id', 'unknown')
    tribe_id = data.get('tribe_id', 'unknown')
    
    # Load ground truth file for this cluster/micro
    ground_truth_lookup = load_ground_truth_file(cluster_id, micro_cluster_id)
    
    deltas = data.get('deltas', [])
    if not deltas:
        logger.warning(f"No deltas found in {file_path}")
        return {}
    
    jsd_values = []
    wd_values = []
    recall_3k_values = []
    recall_4k_values = []
    valid_reviews = 0
    
    for review in deltas:
        # Match with ground truth for recall calculation
        user_id = review.get('user_id', '')
        product_description = review.get('product_description', '').strip()
        match_key = f"{user_id}_{product_description}"
        
        # Get ground truth data if available
        actual_for_recall = review.get('actual', {})
        if ground_truth_lookup and match_key in ground_truth_lookup:
            actual_for_recall = ground_truth_lookup[match_key]
        
        jsd, wd, recall_3k, recall_4k, k_actual, top_k_3 = calculate_metrics_for_review(
            review, threshold=threshold, actual_for_recall=actual_for_recall
        )
        
        if jsd is not None:
            jsd_values.append(jsd)
            if wd is not None:
                wd_values.append(wd)
            if recall_3k is not None:
                recall_3k_values.append(recall_3k)
            if recall_4k is not None:
                recall_4k_values.append(recall_4k)
            valid_reviews += 1
        elif wd is not None:
            wd_values.append(wd)
            if recall_3k is not None:
                recall_3k_values.append(recall_3k)
            if recall_4k is not None:
                recall_4k_values.append(recall_4k)
            valid_reviews += 1
        elif recall_3k is not None or recall_4k is not None:
            if recall_3k is not None:
                recall_3k_values.append(recall_3k)
            if recall_4k is not None:
                recall_4k_values.append(recall_4k)
            valid_reviews += 1
    
    stats = {
        'cluster_id': cluster_id,
        'micro_cluster_id': micro_cluster_id,
        'tribe_id': tribe_id,
        'total_reviews': len(deltas),
        'valid_reviews': valid_reviews,
        'threshold_used': threshold,
        'jsd': {
            'mean': float(np.mean(jsd_values)) if jsd_values else None,
            'median': float(np.median(jsd_values)) if jsd_values else None,
            'std': float(np.std(jsd_values)) if jsd_values else None,
            'min': float(np.min(jsd_values)) if jsd_values else None,
            'max': float(np.max(jsd_values)) if jsd_values else None,
            'count': len(jsd_values)
        },
        'wd': {
            'mean': float(np.mean(wd_values)) if wd_values else None,
            'median': float(np.median(wd_values)) if wd_values else None,
            'std': float(np.std(wd_values)) if wd_values else None,
            'min': float(np.min(wd_values)) if wd_values else None,
            'max': float(np.max(wd_values)) if wd_values else None,
            'count': len(wd_values)
        },
        'recall_3k': {
            'mean': float(np.mean(recall_3k_values)) if recall_3k_values else None,
            'median': float(np.median(recall_3k_values)) if recall_3k_values else None,
            'std': float(np.std(recall_3k_values)) if recall_3k_values else None,
            'min': float(np.min(recall_3k_values)) if recall_3k_values else None,
            'max': float(np.max(recall_3k_values)) if recall_3k_values else None,
            'count': len(recall_3k_values)
        },
        'recall_4k': {
            'mean': float(np.mean(recall_4k_values)) if recall_4k_values else None,
            'median': float(np.median(recall_4k_values)) if recall_4k_values else None,
            'std': float(np.std(recall_4k_values)) if recall_4k_values else None,
            'min': float(np.min(recall_4k_values)) if recall_4k_values else None,
            'max': float(np.max(recall_4k_values)) if recall_4k_values else None,
            'count': len(recall_4k_values)
        }
    }
    
    return stats


def get_display_name(dir_name: str) -> str:
    """
    Get proper display name for directory based on actual content.
    
    Mapping:
    - sgo_train_final_predictions_user_backstory_only → sgo_test_final_predictions_user_seed_backstory (user_backstory + seed_backstory for test)
    - sgo_train_final_predictions_seed_backstory_only → sgo_test_final_predictions_seed_backstory_only (seed_backstory only for test)
    - sgo_train_final_predictions_user_seed_memory → sgo_test_final_predictions_user_seed_memory (seed_backstory + user_backstory + memory for test)
    """
    name_mapping = {
        'sgo_train_final_predictions_user_backstory_only': 'sgo_test_final_predictions_user_seed_backstory',
        'sgo_train_final_predictions_seed_backstory_only': 'sgo_test_final_predictions_seed_backstory_only',
        'sgo_train_final_predictions_user_seed_memory': 'sgo_test_final_predictions_user_seed_memory'
    }
    return name_mapping.get(dir_name, dir_name)


def process_directory(artifacts_dir: Path, dir_name: str, output_dir: Path, threshold: float = 0.5):
    """Process all delta files in a directory."""
    display_name = get_display_name(dir_name)
    
    logger.info(f"\n{'='*80}")
    logger.info(f"Processing: {dir_name}")
    logger.info(f"Display Name: {display_name}")
    logger.info(f"Threshold: {threshold}")
    logger.info(f"{'='*80}")
    
    deltas_dir = artifacts_dir / dir_name / "deltas"
    
    if not deltas_dir.exists():
        logger.warning(f"Deltas directory not found: {deltas_dir}")
        return
    
    # Find all delta files
    delta_files = list(deltas_dir.rglob("*_all_reviews_deltas.json"))
    logger.info(f"Found {len(delta_files)} delta files")
    
    if not delta_files:
        logger.warning(f"No delta files found in {deltas_dir}")
        return
    
    all_stats = []
    cluster_stats = defaultdict(lambda: {'jsd': [], 'wd': [], 'recall_3k': [], 'recall_4k': []})
    micro_stats = defaultdict(lambda: {'jsd': [], 'wd': [], 'recall_3k': [], 'recall_4k': []})
    
    for delta_file in tqdm(delta_files, desc=f"Processing {dir_name}"):
        stats = process_delta_file(delta_file, threshold=threshold)
        if stats:
            all_stats.append(stats)
            
            cluster_id = stats['cluster_id']
            micro_id = stats['micro_cluster_id']
            
            if stats['jsd']['mean'] is not None:
                cluster_stats[cluster_id]['jsd'].append(stats['jsd']['mean'])
                micro_stats[f"{cluster_id}/{micro_id}"]['jsd'].append(stats['jsd']['mean'])
            
            if stats['wd']['mean'] is not None:
                cluster_stats[cluster_id]['wd'].append(stats['wd']['mean'])
                micro_stats[f"{cluster_id}/{micro_id}"]['wd'].append(stats['wd']['mean'])
            
            if stats['recall_3k']['mean'] is not None:
                cluster_stats[cluster_id]['recall_3k'].append(stats['recall_3k']['mean'])
                micro_stats[f"{cluster_id}/{micro_id}"]['recall_3k'].append(stats['recall_3k']['mean'])
            
            if stats['recall_4k']['mean'] is not None:
                cluster_stats[cluster_id]['recall_4k'].append(stats['recall_4k']['mean'])
                micro_stats[f"{cluster_id}/{micro_id}"]['recall_4k'].append(stats['recall_4k']['mean'])
    
    # Calculate summary statistics
    all_jsd = []
    all_wd = []
    all_recall_3k = []
    all_recall_4k = []
    for stats in all_stats:
        if stats['jsd']['mean'] is not None:
            all_jsd.append(stats['jsd']['mean'])
        if stats['wd']['mean'] is not None:
            all_wd.append(stats['wd']['mean'])
        if stats['recall_3k']['mean'] is not None:
            all_recall_3k.append(stats['recall_3k']['mean'])
        if stats['recall_4k']['mean'] is not None:
            all_recall_4k.append(stats['recall_4k']['mean'])
    
    # Get display name for output
    display_name = get_display_name(dir_name)
    
    summary = {
        'directory': dir_name,
        'display_name': display_name,
        'total_delta_files': len(delta_files),
        'total_micro_clusters': len(all_stats),
        'threshold_used': threshold,
        'overall': {
            'jsd': {
                'mean': float(np.mean(all_jsd)) if all_jsd else None,
                'median': float(np.median(all_jsd)) if all_jsd else None,
                'std': float(np.std(all_jsd)) if all_jsd else None,
                'min': float(np.min(all_jsd)) if all_jsd else None,
                'max': float(np.max(all_jsd)) if all_jsd else None,
                'count': len(all_jsd)
            },
            'wd': {
                'mean': float(np.mean(all_wd)) if all_wd else None,
                'median': float(np.median(all_wd)) if all_wd else None,
                'std': float(np.std(all_wd)) if all_wd else None,
                'min': float(np.min(all_wd)) if all_wd else None,
                'max': float(np.max(all_wd)) if all_wd else None,
                'count': len(all_wd)
            },
            'recall_3k': {
                'mean': float(np.mean(all_recall_3k)) if all_recall_3k else None,
                'median': float(np.median(all_recall_3k)) if all_recall_3k else None,
                'std': float(np.std(all_recall_3k)) if all_recall_3k else None,
                'min': float(np.min(all_recall_3k)) if all_recall_3k else None,
                'max': float(np.max(all_recall_3k)) if all_recall_3k else None,
                'count': len(all_recall_3k)
            },
            'recall_4k': {
                'mean': float(np.mean(all_recall_4k)) if all_recall_4k else None,
                'median': float(np.median(all_recall_4k)) if all_recall_4k else None,
                'std': float(np.std(all_recall_4k)) if all_recall_4k else None,
                'min': float(np.min(all_recall_4k)) if all_recall_4k else None,
                'max': float(np.max(all_recall_4k)) if all_recall_4k else None,
                'count': len(all_recall_4k)
            }
        },
        'by_cluster': {
            cluster_id: {
                'jsd': {
                    'mean': float(np.mean(vals['jsd'])) if vals['jsd'] else None,
                    'count': len(vals['jsd'])
                },
                'wd': {
                    'mean': float(np.mean(vals['wd'])) if vals['wd'] else None,
                    'count': len(vals['wd'])
                },
                'recall_3k': {
                    'mean': float(np.mean(vals['recall_3k'])) if vals['recall_3k'] else None,
                    'count': len(vals['recall_3k'])
                },
                'recall_4k': {
                    'mean': float(np.mean(vals['recall_4k'])) if vals['recall_4k'] else None,
                    'count': len(vals['recall_4k'])
                }
            }
            for cluster_id, vals in cluster_stats.items()
        },
        'by_micro_cluster': {
            micro_id: {
                'jsd': {
                    'mean': float(np.mean(vals['jsd'])) if vals['jsd'] else None,
                    'count': len(vals['jsd'])
                },
                'wd': {
                    'mean': float(np.mean(vals['wd'])) if vals['wd'] else None,
                    'count': len(vals['wd'])
                },
                'recall_3k': {
                    'mean': float(np.mean(vals['recall_3k'])) if vals['recall_3k'] else None,
                    'count': len(vals['recall_3k'])
                },
                'recall_4k': {
                    'mean': float(np.mean(vals['recall_4k'])) if vals['recall_4k'] else None,
                    'count': len(vals['recall_4k'])
                }
            }
            for micro_id, vals in micro_stats.items()
        },
        'detailed_stats': all_stats
    }
    
    # Save results - use display name for output directory
    output_subdir = output_dir / display_name
    output_subdir.mkdir(parents=True, exist_ok=True)
    
    summary_file = output_subdir / "comprehensive_metrics_summary.json"
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    logger.info(f"Saved summary to: {summary_file}")
    
    # Print summary
    logger.info(f"\nSummary for {display_name} (threshold={threshold}):")
    if all_jsd:
        logger.info(f"  JSD: Mean = {np.mean(all_jsd):.6f}, Median = {np.median(all_jsd):.6f}, Count = {len(all_jsd)}")
    if all_wd:
        logger.info(f"  WD:  Mean = {np.mean(all_wd):.6f}, Median = {np.median(all_wd):.6f}, Count = {len(all_wd)}")
    if all_recall_3k:
        logger.info(f"  Recall@max(3,k): Mean = {np.mean(all_recall_3k):.6f}, Median = {np.median(all_recall_3k):.6f}, Count = {len(all_recall_3k)}")
    if all_recall_4k:
        logger.info(f"  Recall@max(4,k): Mean = {np.mean(all_recall_4k):.6f}, Median = {np.median(all_recall_4k):.6f}, Count = {len(all_recall_4k)}")


def main():
    """Main execution function."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Calculate JSD, WD, and Recall@max(3,k), Recall@max(4,k) for artifact directories')
    parser.add_argument('--artifacts-dir', type=str, default=None,
                       help='Base artifacts directory (default: 07_sgo_training/artifacts)')
    parser.add_argument('--output-dir', type=str, default=None,
                       help='Output directory for results (default: artifacts/comprehensive_metrics_results)')
    parser.add_argument('--directories', type=str, nargs='+',
                       default=['sgo_train_final_predictions_seed_backstory_only', 
                               'sgo_train_final_predictions_user_backstory_only',
                               'sgo_train_final_predictions_user_seed_memory'],
                       help='Directory names to process (actual directory names in artifacts folder)')
    parser.add_argument('--threshold', type=float, default=0.5,
                       help='Threshold for ground truth theme extraction from topic_probabilities (default: 0.5)')
    
    args = parser.parse_args()
    
    # Determine artifacts directory
    if args.artifacts_dir:
        artifacts_base = Path(args.artifacts_dir)
    else:
        artifacts_base = BASE_DIR / "07_sgo_training" / "artifacts"
    
    # Determine output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = artifacts_base / "comprehensive_metrics_results"
    
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Threshold: {args.threshold}")
    
    # Process each directory
    for dir_name in args.directories:
        process_directory(artifacts_base, dir_name, output_dir, threshold=args.threshold)
    
    logger.info("\n" + "=" * 80)
    logger.info("Comprehensive metrics calculation complete!")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()

