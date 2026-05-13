#!/usr/bin/env python3
"""
Calculate JSD, WD, and Recall Metrics for BEFM Predictions
===========================================================

This script:
1. Loads all BEFM prediction files from befm_results_history
2. For each prediction with ground_truth:
   - Normalizes confidence scores in prediction.predicted_themes
   - Calculates JSD and WD between normalized predictions and ground truth topic_probabilities
   - Calculates Recall@max(3,k) and Recall@max(4,k) using predicted_themes from ground_truth
3. Updates prediction files with metrics

Usage:
    python Metrics and analysis/scripts/calculate_metrics_befm_predictions.py
"""

import json
import numpy as np
import sys
import logging
from pathlib import Path
from typing import Dict, Any, Tuple, Optional, List
from collections import defaultdict
from tqdm import tqdm
from scipy.stats import entropy

# Add project root to path
BASE_DIR = Path(__file__).parent.parent.parent
sys.path.insert(0, str(BASE_DIR))
sys.path.insert(0, str(BASE_DIR / "Metrics and analysis" / "scripts"))

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global epsilon for numerical stability
EPSILON = 1e-10

# Try to import optimal transport library
try:
    import ot  # Python Optimal Transport library
    HAS_OT = True
except ImportError:
    ot = None
    HAS_OT = False
    logger.warning("POT (Python Optimal Transport) library not found. Install with: pip install POT")
    logger.warning("Falling back to L1 distance approximation for WD")


# Mapping from BEFM theme names to Set A (ground truth) theme names
BEFM_TO_SETA_THEME_MAPPING = {
    # All Beauty
    "Product Suitability and Effectiveness": "Product Performance & Effectiveness",
    "Aesthetic and Sensory Experience": "Sensory & Aesthetic Experience",
    "Ease of Use and Application": "Ease of Use, Application & Maintenance",
    "Performance and Effectiveness": "Product Performance & Effectiveness",
    "Performance and Results Consistency": "Product Performance & Effectiveness",
    "Safety and Ingredients Transparency": "Ingredient Safety & Transparency",
    "Safety and Ingredients": "Ingredient Safety & Transparency",
    "Durability and Build Quality": "Quality, Durability & Reliability",
    "Comfort and Ergonomics": "Ease of Use & Ergonomics",
    "Value and Affordability": "Value for Money",
    "Value for Money": "Value for Money",
    "Environmental Sustainability": "Ingredient Safety & Transparency + Sustainability & Eco-Friendliness",
    "Packaging and Sustainability": "Customer Support, Shipping & Service",
    
    # Digital Music
    "Sound quality and remastering": "Sound quality and production/mastering",
    "Artist performance and musicianship": "Songwriting, musicianship and stylistic authenticity",
    "Nostalgia and emotional impact": "Emotional, nostalgic and therapeutic impact",
    "Documentation and packaging quality": "Packaging, liner notes and metadata accuracy",
    "Authenticity and ethical releases": "Authenticity, rarity and collectability + Artist background and historical context",
    "Collector appeal and product listing accuracy": "Authenticity, rarity and collectability",
    "Availability and distribution": "Track selection and album curation",
    
    # Video Games
    "Game Content and Performance": "Content Depth Storyline and Appropriateness",
    "User Interface and Control Usability": "Gameplay Mechanics Controls and Difficulty + Accessibility & Inclusive Design",
    "Audio and Communication Quality": "Audio and Visual Quality",
    "Connectivity and Compatibility": "Performance Stability and Bug Issues",
    "Product Build and Comfort": "Hardware Build Quality Comfort and Connectivity + Battery Life and Portability",
    
    # Health and Personal Care
    "Ease of Use and Setup": "Ease of Use & Convenience",
    "Portability and Versatility": "Ease of Use & Convenience",
    "Use-case and mood suitability": "Ease of Use & Convenience",
    
    # General/Cross-category
    "Setup and Product Information Accuracy": "Customer Support, Shipping & Service",
    "Language accessibility and cultural relevance": "Ease of Use, Application & Maintenance",
    "Meaningful Customization": "Ease of Use, Application & Maintenance",
    "Customer Support and Brand Trust": "Customer Support, Shipping & Service",
}


def map_befm_themes_to_seta(predicted_themes: Dict[str, float], category: str = "") -> Dict[str, float]:
    """
    Map BEFM theme names to Set A (ground truth) theme names.
    
    Args:
        predicted_themes: Dictionary of BEFM theme names to confidence scores
        category: Category name (for future category-specific mappings)
        
    Returns:
        Dictionary of mapped Set A theme names to probabilities
    """
    from collections import defaultdict
    
    mapped_themes = defaultdict(float)
    unmapped_themes = {}
    
    for befm_theme, score in predicted_themes.items():
        if befm_theme in BEFM_TO_SETA_THEME_MAPPING:
            seta_theme = BEFM_TO_SETA_THEME_MAPPING[befm_theme]
            mapped_themes[seta_theme] += score
        else:
            # Theme not in mapping - keep as-is (might already be Set A)
            unmapped_themes[befm_theme] = score
    
    # Add unmapped themes (they might already be Set A)
    for theme, score in unmapped_themes.items():
        mapped_themes[theme] += score
    
    return dict(mapped_themes)


def normalize_distribution(theme_dict: Dict[str, float]) -> Dict[str, float]:
    """
    Normalize a theme probability distribution to sum to 1.0.
    
    Args:
        theme_dict: Dictionary mapping theme names to probabilities/confidence scores
        
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
    # Get all unique themes from both distributions
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


def get_top_k_predicted(predicted_themes: Dict[str, float], k: int) -> List[str]:
    """
    Get top k predicted themes sorted by probability.
    
    Args:
        predicted_themes: Dictionary of theme -> probability
        k: Number of top themes to return
        
    Returns:
        List of top k theme names
    """
    if not predicted_themes:
        return []
    
    # Sort by probability (descending)
    sorted_themes = sorted(predicted_themes.items(), key=lambda x: x[1], reverse=True)
    return [theme for theme, _ in sorted_themes[:k]]


def calculate_recall_at_max_nk(
    predicted_themes: Dict[str, float],
    ground_truth_themes: set,
    n: int = 4
) -> Tuple[float, int, int]:
    """
    Calculate Recall@max(n, k) where k is the number of ground truth themes.
    
    Args:
        predicted_themes: Dictionary of predicted theme -> probability
        ground_truth_themes: Set of ground truth theme names
        n: Minimum number of predictions to consider (e.g., 3 or 4)
        
    Returns:
        Tuple of (recall, k_actual, top_k_used)
    """
    if not ground_truth_themes:
        return 0.0, 0, 0
    
    k_actual = len(ground_truth_themes)
    top_k_used = max(n, k_actual)
    
    # Get top k predicted themes
    top_predicted = get_top_k_predicted(predicted_themes, top_k_used)
    
    # Count how many ground truth themes are in top predicted
    matched = len(set(top_predicted) & ground_truth_themes)
    
    recall = matched / k_actual if k_actual > 0 else 0.0
    
    return recall, k_actual, top_k_used


def calculate_recall_at_max_nh(
    predicted_themes: Dict[str, float],
    ground_truth_themes: set,
    n: int = 4
) -> Tuple[float, int, int]:
    """
    Calculate Recall@max(n, h) where h is the number of predicted themes.
    
    Args:
        predicted_themes: Dictionary of predicted theme -> probability
        ground_truth_themes: Set of ground truth theme names
        n: Minimum number of predictions to consider (e.g., 4)
        
    Returns:
        Tuple of (recall, h_predicted, top_h_used)
    """
    if not ground_truth_themes:
        return 0.0, 0, 0
    
    h_predicted = len(predicted_themes)
    top_h_used = max(n, h_predicted)
    
    # Get top h predicted themes
    top_predicted = get_top_k_predicted(predicted_themes, top_h_used)
    
    # Count how many ground truth themes are in top predicted
    matched = len(set(top_predicted) & ground_truth_themes)
    
    k_actual = len(ground_truth_themes)
    recall = matched / k_actual if k_actual > 0 else 0.0
    
    return recall, h_predicted, top_h_used


def calculate_recall_at_h(
    predicted_themes: Dict[str, float],
    ground_truth_themes: set
) -> Tuple[float, int]:
    """
    Calculate Recall@h where h is the number of predicted themes.
    This uses exactly h predicted themes (no max with a minimum).
    
    Args:
        predicted_themes: Dictionary of predicted theme -> probability
        ground_truth_themes: Set of ground truth theme names
        
    Returns:
        Tuple of (recall, h_predicted)
    """
    if not ground_truth_themes:
        return 0.0, 0
    
    h_predicted = len(predicted_themes)
    
    # Get top h predicted themes (exactly h, not max(4, h))
    top_predicted = get_top_k_predicted(predicted_themes, h_predicted)
    
    # Count how many ground truth themes are in top predicted
    matched = len(set(top_predicted) & ground_truth_themes)
    
    k_actual = len(ground_truth_themes)
    recall = matched / k_actual if k_actual > 0 else 0.0
    
    return recall, h_predicted


def calculate_recall_at_n(
    predicted_themes: Dict[str, float],
    ground_truth_themes: set,
    n: int = 3
) -> float:
    """
    Calculate Recall@n: check if ground truth themes are in top n predicted themes.
    
    Args:
        predicted_themes: Dictionary of predicted theme -> probability
        ground_truth_themes: Set of ground truth theme names
        n: Number of top predicted themes to consider
        
    Returns:
        Recall value (float)
    """
    if not ground_truth_themes:
        return 0.0
    
    # Get top n predicted themes
    top_predicted = get_top_k_predicted(predicted_themes, n)
    
    # Count how many ground truth themes are in top predicted
    matched = len(set(top_predicted) & ground_truth_themes)
    
    k_actual = len(ground_truth_themes)
    recall = matched / k_actual if k_actual > 0 else 0.0
    
    return recall


def calculate_recall_at_max3k(
    predicted_themes: Dict[str, float],
    ground_truth_themes: set
) -> Tuple[float, int, int]:
    """Calculate Recall@max(3, k)."""
    return calculate_recall_at_max_nk(predicted_themes, ground_truth_themes, n=3)


def calculate_recall_at_max4k(
    predicted_themes: Dict[str, float],
    ground_truth_themes: set
) -> Tuple[float, int, int]:
    """Calculate Recall@max(4, k)."""
    return calculate_recall_at_max_nk(predicted_themes, ground_truth_themes, n=4)


def calculate_metrics_for_prediction(pred: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Calculate JSD, WD, and Recall metrics for a single prediction.
    
    Args:
        pred: Prediction dictionary with 'prediction' and optional 'ground_truth'
        
    Returns:
        Dictionary with metrics, or None if calculation failed
    """
    try:
        prediction = pred.get('prediction', {})
        ground_truth = pred.get('ground_truth')
        
        if not prediction or not ground_truth:
            return None
        
        # Get predicted themes (confidence scores)
        pred_themes_raw = prediction.get('predicted_themes', {})
        if not pred_themes_raw:
            return None
        
        # Map BEFM themes to Set A before normalization
        category = pred.get('original_category', '')
        pred_themes_mapped = map_befm_themes_to_seta(pred_themes_raw, category)
        
        # Update the prediction.predicted_themes field with mapped Set A themes
        # Keep the same confidence scores, just update theme names
        prediction['predicted_themes'] = pred_themes_mapped
        
        # Normalize predicted themes (confidence scores -> probabilities)
        pred_themes_normalized = normalize_distribution(pred_themes_mapped)
        if not pred_themes_normalized:
            return None
        
        # Get ground truth topic_probabilities for JSD/WD
        gt_topic_probs = ground_truth.get('topic_probabilities', {})
        if not gt_topic_probs:
            return None
        
        # Normalize ground truth probabilities
        gt_topic_probs_normalized = normalize_distribution(gt_topic_probs)
        if not gt_topic_probs_normalized:
            return None
        
        # Align distributions for JSD/WD calculation
        actual_array, predicted_array = align_distributions(
            gt_topic_probs_normalized,
            pred_themes_normalized
        )
        
        if len(actual_array) == 0 or len(predicted_array) == 0:
            return None
        
        # Calculate JSD
        jsd = compute_jsd(actual_array, predicted_array)
        
        # Calculate WD
        wd = calculate_wasserstein_1_distance(actual_array, predicted_array)
        if np.isnan(wd) or not np.isfinite(wd):
            wd = None
        
        # Get ground truth: top 1 theme from topic_probabilities
        # Sort by probability and get the top theme
        if not gt_topic_probs_normalized:
            return None
        
        # Get top 1 theme from topic_probabilities
        sorted_gt_themes = sorted(gt_topic_probs_normalized.items(), key=lambda x: x[1], reverse=True)
        if not sorted_gt_themes:
            return None
        
        top_gt_theme = sorted_gt_themes[0][0]
        gt_themes_set = {top_gt_theme}  # Single theme set
        
        # Calculate Recall@max(3,k) and Recall@max(4,k) where k = ground truth themes
        recall_at_max3k, k_actual, top_k_used_3 = calculate_recall_at_max3k(
            pred_themes_normalized,
            gt_themes_set
        )
        
        recall_at_max4k, _, top_k_used_4 = calculate_recall_at_max4k(
            pred_themes_normalized,
            gt_themes_set
        )
        
        # Calculate Recall@max(4, h) where h = predicted themes
        recall_at_max4h, h_predicted, top_h_used_4 = calculate_recall_at_max_nh(
            pred_themes_normalized,
            gt_themes_set,
            n=4
        )
        
        # Calculate Recall@h where h = predicted themes (exactly h, no max)
        recall_at_h, _ = calculate_recall_at_h(
            pred_themes_normalized,
            gt_themes_set
        )
        
        # Calculate Recall@3 (top 3 predicted themes)
        recall_at_3 = calculate_recall_at_n(
            pred_themes_normalized,
            gt_themes_set,
            n=3
        )
        
        return {
            'jsd': jsd,
            'wd': wd,
            'recall_at_max3k': recall_at_max3k,
            'recall_at_max4k': recall_at_max4k,
            'recall_at_max4h': recall_at_max4h,
            'recall_at_h': recall_at_h,
            'recall_at_3': recall_at_3,
            'k_actual': k_actual,
            'h_predicted': h_predicted,
            'top_k_used_3': top_k_used_3,
            'top_k_used_4': top_k_used_4,
            'top_h_used_4': top_h_used_4
        }
        
    except Exception as e:
        logger.warning(f"Error calculating metrics: {e}")
        return None


def process_prediction_file(file_path: Path) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Process a single prediction file and calculate metrics.
    
    Args:
        file_path: Path to prediction file
        
    Returns:
        Tuple of (updated_data, statistics)
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        logger.error(f"Error reading {file_path}: {e}")
        return None, None
    
    stats = {
        'file_path': str(file_path),
        'total_predictions': 0,
        'predictions_with_metrics': 0,
        'predictions_without_metrics': 0
    }
    
    predictions = data.get('predictions', [])
    stats['total_predictions'] = len(predictions)
    
    # Process each prediction
    for pred in predictions:
        metrics = calculate_metrics_for_prediction(pred)
        
        if metrics:
            # Add metrics to prediction
            if 'metrics' not in pred:
                pred['metrics'] = {}
            
            # Update/add metrics
            pred['metrics'].update({
                'jsd': metrics['jsd'],
                'wd': metrics['wd'],
                'recall_at_max3k': metrics['recall_at_max3k'],
                'recall_at_max4k': metrics['recall_at_max4k'],
                'recall_at_max4h': metrics['recall_at_max4h'],
                'recall_at_h': metrics['recall_at_h'],
                'recall_at_3': metrics['recall_at_3'],
                'k_actual': metrics['k_actual'],
                'h_predicted': metrics['h_predicted'],
                'top_k_used_3': metrics['top_k_used_3'],
                'top_k_used_4': metrics['top_k_used_4'],
                'top_h_used_4': metrics['top_h_used_4']
            })
            
            stats['predictions_with_metrics'] += 1
        else:
            stats['predictions_without_metrics'] += 1
    
    return data, stats


def process_all_prediction_files(predictions_dir: Path) -> Dict[str, Any]:
    """
    Process all prediction files and calculate metrics.
    
    Args:
        predictions_dir: Path to predictions directory
        
    Returns:
        Dictionary with overall statistics
    """
    logger.info(f"Processing prediction files from: {predictions_dir}")
    
    # Find all prediction files
    prediction_files = sorted(predictions_dir.glob("prediction_*.json"))
    logger.info(f"Found {len(prediction_files)} prediction files")
    
    overall_stats = {
        'total_files': len(prediction_files),
        'total_predictions': 0,
        'predictions_with_metrics': 0,
        'predictions_without_metrics': 0,
        'file_stats': [],
        'jsd_values': [],
        'wd_values': [],
        'recall_at_max3k_values': [],
        'recall_at_max4k_values': [],
        'recall_at_max4h_values': [],
        'recall_at_h_values': [],
        'recall_at_3_values': [],
        'k_actual_values': [],
        'h_predicted_values': []
    }
    
    # Process each file
    for pred_file in tqdm(prediction_files, desc="Processing prediction files"):
        data, stats = process_prediction_file(pred_file)
        
        if data is None:
            continue
        
        # Save updated file
        try:
            with open(pred_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=4, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Error saving {pred_file}: {e}")
            continue
        
        overall_stats['total_predictions'] += stats['total_predictions']
        overall_stats['predictions_with_metrics'] += stats['predictions_with_metrics']
        overall_stats['predictions_without_metrics'] += stats['predictions_without_metrics']
        overall_stats['file_stats'].append(stats)
        
        # Collect metric values for statistics - ONLY for predictions with metrics
        predictions = data.get('predictions', [])
        for pred in predictions:
            metrics = pred.get('metrics', {})
            # Only include if metrics were successfully calculated (has jsd)
            if metrics.get('jsd') is not None:
                overall_stats['jsd_values'].append(metrics['jsd'])
                if metrics.get('wd') is not None:
                    overall_stats['wd_values'].append(metrics['wd'])
                if metrics.get('recall_at_max3k') is not None:
                    overall_stats['recall_at_max3k_values'].append(metrics['recall_at_max3k'])
                if metrics.get('recall_at_max4k') is not None:
                    overall_stats['recall_at_max4k_values'].append(metrics['recall_at_max4k'])
                if metrics.get('recall_at_max4h') is not None:
                    overall_stats['recall_at_max4h_values'].append(metrics['recall_at_max4h'])
                if metrics.get('recall_at_h') is not None:
                    overall_stats['recall_at_h_values'].append(metrics['recall_at_h'])
                if metrics.get('recall_at_3') is not None:
                    overall_stats['recall_at_3_values'].append(metrics['recall_at_3'])
                if metrics.get('k_actual') is not None:
                    overall_stats['k_actual_values'].append(metrics['k_actual'])
                if metrics.get('h_predicted') is not None:
                    overall_stats['h_predicted_values'].append(metrics['h_predicted'])
    
    return overall_stats


def generate_summary_statistics(overall_stats: Dict[str, Any]) -> Dict[str, Any]:
    """
    Generate summary statistics from collected metric values.
    
    Args:
        overall_stats: Dictionary with collected metric values
        
    Returns:
        Dictionary with summary statistics
    """
    summary = {}
    
    # JSD statistics
    jsd_values = overall_stats.get('jsd_values', [])
    if jsd_values:
        summary['jsd'] = {
            'mean': float(np.mean(jsd_values)),
            'median': float(np.median(jsd_values)),
            'std': float(np.std(jsd_values)),
            'min': float(np.min(jsd_values)),
            'max': float(np.max(jsd_values)),
            '25th_percentile': float(np.percentile(jsd_values, 25)),
            '75th_percentile': float(np.percentile(jsd_values, 75)),
            'count': len(jsd_values)
        }
    
    # WD statistics
    wd_values = overall_stats.get('wd_values', [])
    if wd_values:
        summary['wd'] = {
            'mean': float(np.mean(wd_values)),
            'median': float(np.median(wd_values)),
            'std': float(np.std(wd_values)),
            'min': float(np.min(wd_values)),
            'max': float(np.max(wd_values)),
            '25th_percentile': float(np.percentile(wd_values, 25)),
            '75th_percentile': float(np.percentile(wd_values, 75)),
            'count': len(wd_values)
        }
    
    # Recall@max(3,k) statistics
    recall_3k_values = overall_stats.get('recall_at_max3k_values', [])
    if recall_3k_values:
        summary['recall_at_max3k'] = {
            'mean': float(np.mean(recall_3k_values)),
            'median': float(np.median(recall_3k_values)),
            'std': float(np.std(recall_3k_values)),
            'min': float(np.min(recall_3k_values)),
            'max': float(np.max(recall_3k_values)),
            '25th_percentile': float(np.percentile(recall_3k_values, 25)),
            '75th_percentile': float(np.percentile(recall_3k_values, 75)),
            'count': len(recall_3k_values)
        }
    
    # Recall@max(4,k) statistics
    recall_4k_values = overall_stats.get('recall_at_max4k_values', [])
    if recall_4k_values:
        summary['recall_at_max4k'] = {
            'mean': float(np.mean(recall_4k_values)),
            'median': float(np.median(recall_4k_values)),
            'std': float(np.std(recall_4k_values)),
            'min': float(np.min(recall_4k_values)),
            'max': float(np.max(recall_4k_values)),
            '25th_percentile': float(np.percentile(recall_4k_values, 25)),
            '75th_percentile': float(np.percentile(recall_4k_values, 75)),
            'count': len(recall_4k_values)
        }
    
    # Recall@max(4,h) statistics
    recall_4h_values = overall_stats.get('recall_at_max4h_values', [])
    if recall_4h_values:
        summary['recall_at_max4h'] = {
            'mean': float(np.mean(recall_4h_values)),
            'median': float(np.median(recall_4h_values)),
            'std': float(np.std(recall_4h_values)),
            'min': float(np.min(recall_4h_values)),
            'max': float(np.max(recall_4h_values)),
            '25th_percentile': float(np.percentile(recall_4h_values, 25)),
            '75th_percentile': float(np.percentile(recall_4h_values, 75)),
            'count': len(recall_4h_values)
        }
    
    # Recall@h statistics
    recall_h_values = overall_stats.get('recall_at_h_values', [])
    if recall_h_values:
        summary['recall_at_h'] = {
            'mean': float(np.mean(recall_h_values)),
            'median': float(np.median(recall_h_values)),
            'std': float(np.std(recall_h_values)),
            'min': float(np.min(recall_h_values)),
            'max': float(np.max(recall_h_values)),
            '25th_percentile': float(np.percentile(recall_h_values, 25)),
            '75th_percentile': float(np.percentile(recall_h_values, 75)),
            'count': len(recall_h_values)
        }
    
    # Recall@3 statistics
    recall_3_values = overall_stats.get('recall_at_3_values', [])
    if recall_3_values:
        summary['recall_at_3'] = {
            'mean': float(np.mean(recall_3_values)),
            'median': float(np.median(recall_3_values)),
            'std': float(np.std(recall_3_values)),
            'min': float(np.min(recall_3_values)),
            'max': float(np.max(recall_3_values)),
            '25th_percentile': float(np.percentile(recall_3_values, 25)),
            '75th_percentile': float(np.percentile(recall_3_values, 75)),
            'count': len(recall_3_values)
        }
    
    # K actual statistics
    k_values = overall_stats.get('k_actual_values', [])
    if k_values:
        summary['k_actual'] = {
            'mean': float(np.mean(k_values)),
            'median': float(np.median(k_values)),
            'std': float(np.std(k_values)),
            'min': float(np.min(k_values)),
            'max': float(np.max(k_values)),
            '25th_percentile': float(np.percentile(k_values, 25)),
            '75th_percentile': float(np.percentile(k_values, 75)),
            'count': len(k_values)
        }
    
    # H predicted statistics
    h_values = overall_stats.get('h_predicted_values', [])
    if h_values:
        summary['h_predicted'] = {
            'mean': float(np.mean(h_values)),
            'median': float(np.median(h_values)),
            'std': float(np.std(h_values)),
            'min': float(np.min(h_values)),
            'max': float(np.max(h_values)),
            '25th_percentile': float(np.percentile(h_values, 25)),
            '75th_percentile': float(np.percentile(h_values, 75)),
            'count': len(h_values)
        }
    
    return summary


def main():
    """Main execution function."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Calculate JSD, WD, and Recall metrics for BEFM predictions',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('--predictions-dir', type=str,
                       default=str(BASE_DIR / "befm_results_history"),
                       help='Path to predictions directory')
    parser.add_argument('--output-summary', type=str,
                       help='Path to save summary statistics JSON file')
    
    args = parser.parse_args()
    
    predictions_dir = Path(args.predictions_dir)
    
    if not predictions_dir.exists():
        logger.error(f"Predictions directory not found: {predictions_dir}")
        return
    
    logger.info("=" * 80)
    logger.info("Calculate Metrics for BEFM Predictions")
    logger.info("=" * 80)
    logger.info(f"Predictions directory: {predictions_dir}")
    logger.info("=" * 80)
    
    # Process all prediction files
    overall_stats = process_all_prediction_files(predictions_dir)
    
    # Generate summary statistics
    summary_stats = generate_summary_statistics(overall_stats)
    overall_stats['summary_statistics'] = summary_stats
    
    # Print summary
    logger.info("\n" + "=" * 80)
    logger.info("SUMMARY STATISTICS")
    logger.info("=" * 80)
    logger.info(f"Total files processed: {overall_stats['total_files']}")
    logger.info(f"Total predictions: {overall_stats['total_predictions']:,}")
    logger.info(f"Predictions with metrics: {overall_stats['predictions_with_metrics']:,} ({100.0 * overall_stats['predictions_with_metrics'] / max(overall_stats['total_predictions'], 1):.2f}%)")
    logger.info(f"Predictions without metrics: {overall_stats['predictions_without_metrics']:,} ({100.0 * overall_stats['predictions_without_metrics'] / max(overall_stats['total_predictions'], 1):.2f}%)")
    
    if summary_stats.get('jsd'):
        jsd_stats = summary_stats['jsd']
        logger.info(f"\nJSD Statistics:")
        logger.info(f"  Mean: {jsd_stats['mean']:.6f}")
        logger.info(f"  Median: {jsd_stats['median']:.6f}")
        logger.info(f"  Std: {jsd_stats['std']:.6f}")
        logger.info(f"  Min: {jsd_stats['min']:.6f}")
        logger.info(f"  Max: {jsd_stats['max']:.6f}")
        logger.info(f"  Count: {jsd_stats['count']:,}")
    
    if summary_stats.get('wd'):
        wd_stats = summary_stats['wd']
        logger.info(f"\nWD Statistics:")
        logger.info(f"  Mean: {wd_stats['mean']:.6f}")
        logger.info(f"  Median: {wd_stats['median']:.6f}")
        logger.info(f"  Std: {wd_stats['std']:.6f}")
        logger.info(f"  Min: {wd_stats['min']:.6f}")
        logger.info(f"  Max: {wd_stats['max']:.6f}")
        logger.info(f"  Count: {wd_stats['count']:,}")
    
    if summary_stats.get('recall_at_max3k'):
        recall_3k_stats = summary_stats['recall_at_max3k']
        logger.info(f"\nRecall@max(3,k) Statistics:")
        logger.info(f"  Mean: {recall_3k_stats['mean']:.6f}")
        logger.info(f"  Median: {recall_3k_stats['median']:.6f}")
        logger.info(f"  Std: {recall_3k_stats['std']:.6f}")
        logger.info(f"  Min: {recall_3k_stats['min']:.6f}")
        logger.info(f"  Max: {recall_3k_stats['max']:.6f}")
        logger.info(f"  Count: {recall_3k_stats['count']:,}")
    
    if summary_stats.get('recall_at_max4k'):
        recall_4k_stats = summary_stats['recall_at_max4k']
        logger.info(f"\nRecall@max(4,k) Statistics (k = ground truth themes):")
        logger.info(f"  Mean: {recall_4k_stats['mean']:.6f}")
        logger.info(f"  Median: {recall_4k_stats['median']:.6f}")
        logger.info(f"  Std: {recall_4k_stats['std']:.6f}")
        logger.info(f"  Min: {recall_4k_stats['min']:.6f}")
        logger.info(f"  Max: {recall_4k_stats['max']:.6f}")
        logger.info(f"  Count: {recall_4k_stats['count']:,}")
    
    if summary_stats.get('recall_at_max4h'):
        recall_4h_stats = summary_stats['recall_at_max4h']
        logger.info(f"\nRecall@max(4,h) Statistics (h = predicted themes):")
        logger.info(f"  Mean: {recall_4h_stats['mean']:.6f}")
        logger.info(f"  Median: {recall_4h_stats['median']:.6f}")
        logger.info(f"  Std: {recall_4h_stats['std']:.6f}")
        logger.info(f"  Min: {recall_4h_stats['min']:.6f}")
        logger.info(f"  Max: {recall_4h_stats['max']:.6f}")
        logger.info(f"  Count: {recall_4h_stats['count']:,}")
    
    if summary_stats.get('recall_at_h'):
        recall_h_stats = summary_stats['recall_at_h']
        logger.info(f"\nRecall@h Statistics (h = predicted themes, exactly h):")
        logger.info(f"  Mean: {recall_h_stats['mean']:.6f}")
        logger.info(f"  Median: {recall_h_stats['median']:.6f}")
        logger.info(f"  Std: {recall_h_stats['std']:.6f}")
        logger.info(f"  Min: {recall_h_stats['min']:.6f}")
        logger.info(f"  Max: {recall_h_stats['max']:.6f}")
        logger.info(f"  Count: {recall_h_stats['count']:,}")
    
    if summary_stats.get('recall_at_3'):
        recall_3_stats = summary_stats['recall_at_3']
        logger.info(f"\nRecall@3 Statistics (top 3 predicted themes vs top 1 ground truth):")
        logger.info(f"  Mean: {recall_3_stats['mean']:.6f}")
        logger.info(f"  Median: {recall_3_stats['median']:.6f}")
        logger.info(f"  Std: {recall_3_stats['std']:.6f}")
        logger.info(f"  Min: {recall_3_stats['min']:.6f}")
        logger.info(f"  Max: {recall_3_stats['max']:.6f}")
        logger.info(f"  Count: {recall_3_stats['count']:,}")
    
    if summary_stats.get('k_actual'):
        k_stats = summary_stats['k_actual']
        logger.info(f"\nK (ground truth themes) Statistics:")
        logger.info(f"  Mean: {k_stats['mean']:.2f}")
        logger.info(f"  Median: {k_stats['median']:.2f}")
        logger.info(f"  Std: {k_stats['std']:.2f}")
        logger.info(f"  Min: {k_stats['min']:.0f}")
        logger.info(f"  Max: {k_stats['max']:.0f}")
        logger.info(f"  Count: {k_stats['count']:,}")
    
    if summary_stats.get('h_predicted'):
        h_stats = summary_stats['h_predicted']
        logger.info(f"\nH (predicted themes) Statistics:")
        logger.info(f"  Mean: {h_stats['mean']:.2f}")
        logger.info(f"  Median: {h_stats['median']:.2f}")
        logger.info(f"  Std: {h_stats['std']:.2f}")
        logger.info(f"  Min: {h_stats['min']:.0f}")
        logger.info(f"  Max: {h_stats['max']:.0f}")
        logger.info(f"  Count: {h_stats['count']:,}")
    
    logger.info("=" * 80)
    
    # Save summary if requested
    if args.output_summary:
        summary_file = Path(args.output_summary)
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(overall_stats, f, indent=2, ensure_ascii=False)
        logger.info(f"\nSaved summary statistics to: {summary_file}")
    else:
        # Save to default location
        summary_file = predictions_dir / "befm_metrics_summary.json"
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(overall_stats, f, indent=2, ensure_ascii=False)
        logger.info(f"\nSaved summary statistics to: {summary_file}")


if __name__ == "__main__":
    main()

