#!/usr/bin/env python3
"""
Add JSD, WD, Recall@max(3,k), Recall@max(4,k), initial_deltas, and best_delta metrics 
to each review in micro cluster files.
Calculates tribe-wise (micro cluster) averages and saves them at the end of each file.
Uses threshold=0.5 by default for ground truth theme determination.
"""

import json
import numpy as np
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Any, Optional, Tuple
import sys
import argparse
import logging
import re

# Add project root to path
BASE_DIR = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(BASE_DIR))

# Setup logging first
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import calculation functions from category_wise_analysis_v2.py
category_analysis_path = BASE_DIR / "07_sgo_training" / "artifacts" / "sgo_train_final_predictions_user_seed_memory"
sys.path.insert(0, str(category_analysis_path))
try:
    # Import the module
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "category_wise_analysis_v2",
        category_analysis_path / "category_wise_analysis_v2.py"
    )
    category_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(category_module)
    
    # Extract functions
    normalize_distribution = category_module.normalize_distribution
    align_distributions_jsd = category_module.align_distributions_jsd
    align_distributions_wd = category_module.align_distributions_wd
    compute_jsd = category_module.compute_jsd
    calculate_wasserstein_1_distance = category_module.calculate_wasserstein_1_distance
    calculate_recall_at_max3k = category_module.calculate_recall_at_max3k
    calculate_recall_at_max4k = category_module.calculate_recall_at_max4k
    get_ground_truth_themes = category_module.get_ground_truth_themes
    EPSILON = getattr(category_module, 'EPSILON', 1e-10)
    HAS_IMPORTS = True
except (ImportError, AttributeError, FileNotFoundError) as e:
    logger.warning(f"Could not import from category_wise_analysis_v2.py: {e}. Using fallback implementations.")
    # Fallback implementations
    HAS_IMPORTS = False
    EPSILON = 1e-10
    
    def normalize_distribution(theme_dict: Dict[str, float]) -> Dict[str, float]:
        """Normalize a theme probability distribution to sum to 1.0."""
        if not theme_dict:
            return {}
        filtered_dict = {k: float(v) for k, v in theme_dict.items() 
                        if v is not None and not (isinstance(v, float) and np.isnan(v))}
        if not filtered_dict:
            return {}
        total = sum(filtered_dict.values())
        if total == 0:
            return {}
        return {theme: prob / total for theme, prob in filtered_dict.items()}
    
    def align_distributions_jsd(actual_themes, predicted_themes):
        """Align two theme distributions to the same set of themes and normalize."""
        all_themes = sorted(set(actual_themes.keys()) | set(predicted_themes.keys()))
        if not all_themes:
            return np.array([]), np.array([])
        actual_array = np.array([actual_themes.get(theme, 0.0) for theme in all_themes])
        predicted_array = np.array([predicted_themes.get(theme, 0.0) for theme in all_themes])
        
        # Normalize
        actual_sum = actual_array.sum()
        predicted_sum = predicted_array.sum()
        if actual_sum > EPSILON:
            actual_array = actual_array / actual_sum
        if predicted_sum > EPSILON:
            predicted_array = predicted_array / predicted_sum
            
        return actual_array, predicted_array
    
    def align_distributions_wd(actual_themes, predicted_themes):
        """Align distributions for WD calculation."""
        return align_distributions_jsd(actual_themes, predicted_themes)
    
    def compute_jsd(P: np.ndarray, Q: np.ndarray, epsilon: float = EPSILON) -> float:
        """Compute Jensen-Shannon Divergence between two probability distributions."""
        P = P + epsilon
        Q = Q + epsilon
        P = P / P.sum()
        Q = Q / Q.sum()
        M = 0.5 * (P + Q)
        from scipy.stats import entropy
        kl_pm = entropy(P, M, base=2)
        kl_qm = entropy(Q, M, base=2)
        return 0.5 * (kl_pm + kl_qm)
    
    def calculate_wasserstein_1_distance(actual_array, predicted_array):
        """Calculate Wasserstein-1 distance using optimal transport or L1 fallback."""
        if len(actual_array) == 0 or len(predicted_array) == 0:
            return float('nan')
        if len(actual_array) != len(predicted_array):
            return float('nan')
        
        # Try to use optimal transport if available
        try:
            import ot
            P = predicted_array / predicted_array.sum() if predicted_array.sum() > EPSILON else predicted_array
            Q = actual_array / actual_array.sum() if actual_array.sum() > EPSILON else actual_array
            n = len(P)
            cost_matrix = np.ones((n, n)) - np.eye(n)
            wd = ot.emd2(P, Q, cost_matrix, numItermax=1000000)
            return float(wd)
        except:
            # Fallback to L1 distance
            return float(np.sum(np.abs(actual_array - predicted_array)))
    
    def get_top_k_predicted(predicted_themes: Dict[str, float], k: int) -> List[str]:
        """Get top k predicted themes sorted by probability in descending order."""
        if not predicted_themes:
            return []
        sorted_themes = sorted(
            predicted_themes.items(),
            key=lambda x: x[1],
            reverse=True
        )
        return [theme for theme, _ in sorted_themes[:k]]
    
    def calculate_recall_at_max3k(
        predicted_themes: Dict[str, float],
        ground_truth_themes: set,
        debug: bool = False
    ) -> Tuple[float, int, int]:
        """Calculate Recall@max(3, k) where k is the number of ground truth themes."""
        if not ground_truth_themes:
            return 0.0, 0, 0
        
        k = len(ground_truth_themes)
        top_k = max(3, k)
        
        # Get top_k predictions sorted by probability (descending)
        top_k_predicted = set(get_top_k_predicted(predicted_themes, top_k))
        
        # Find intersection (matches)
        intersection = top_k_predicted & ground_truth_themes
        num_matches = len(intersection)
        
        # Recall = matches / k (number of ground truth themes)
        recall = num_matches / k if k > 0 else 0.0
        
        return recall, k, top_k
    
    def calculate_recall_at_max4k(
        predicted_themes: Dict[str, float],
        ground_truth_themes: set,
        debug: bool = False
    ) -> Tuple[float, int, int]:
        """Calculate Recall@max(4, k) where k is the number of ground truth themes."""
        if not ground_truth_themes:
            return 0.0, 0, 0
        
        k = len(ground_truth_themes)
        top_k = max(4, k)
        
        # Get top_k predictions sorted by probability (descending)
        top_k_predicted = set(get_top_k_predicted(predicted_themes, top_k))
        
        # Find intersection (matches)
        intersection = top_k_predicted & ground_truth_themes
        num_matches = len(intersection)
        
        # Recall = matches / k (number of ground truth themes)
        recall = num_matches / k if k > 0 else 0.0
        
        return recall, k, top_k
    
    def get_ground_truth_themes(actual_predicted_themes: Dict[str, float], threshold: float = 0.0) -> set:
        """Extract ground truth themes from actual predicted_themes using threshold."""
        result = set()
        for theme, prob in actual_predicted_themes.items():
            try:
                if prob is not None and not (isinstance(prob, float) and np.isnan(prob)):
                    if prob > threshold:
                        result.add(theme)
            except (TypeError, ValueError):
                continue
        return result


def get_ground_truth_themes_from_logprobs_or_probs(
    actual: Dict[str, Any], 
    threshold: float = 0.3
) -> set:
    """
    Extract ground truth themes from topic_probs_before_normalization, topic_probabilities, 
    or topic_logprobs BEFORE normalization.
    
    This function follows the same logic as get_ground_truth_themes in calculate_recall_at_k.py:
    - Uses topic_probs_before_normalization with structure: {"theme": {"prob_yes": float, "prob_no": float}}
    - Extracts prob_yes and applies threshold on it (before normalization)
    
    Priority order:
    1. topic_probs_before_normalization (with prob_yes/prob_no structure) - EXACTLY like calculate_recall_at_k.py
    2. topic_probabilities (with prob_yes/prob_no structure or direct probabilities)
    3. topic_logprobs (convert logprobs to probabilities: e^logprob_yes / (e^logprob_yes + e^logprob_no))
    4. predicted_themes (fallback - already normalized)
    
    Args:
        actual: Actual review dictionary
        threshold: Probability threshold (default: 0.3) - applied on prob_yes BEFORE normalization
        
    Returns:
        Set of ground truth theme names
    """
    result = set()
    
    # First try: topic_probs_before_normalization (EXACTLY like calculate_recall_at_k.py)
    # Structure: {"theme": {"prob_yes": float, "prob_no": float}}
    topic_probs_before_normalization = actual.get('topic_probs_before_normalization', {})
    if topic_probs_before_normalization and isinstance(topic_probs_before_normalization, dict):
        for theme, probs in topic_probs_before_normalization.items():
            if isinstance(probs, dict):
                prob_yes = probs.get('prob_yes')
                # Handle None/null values - skip if prob_yes is None
                if prob_yes is not None and isinstance(prob_yes, (int, float)) and prob_yes >= threshold:
                    result.add(theme)
            elif isinstance(probs, (int, float)):
                # Fallback: if it's a direct probability value
                if probs >= threshold:
                    result.add(theme)
    
    # Second try: topic_probabilities (may have prob_yes/prob_no structure or direct probabilities)
    if not result:
        topic_probabilities = actual.get('topic_probabilities', {})
        if topic_probabilities and isinstance(topic_probabilities, dict):
            for theme, prob_data in topic_probabilities.items():
                if isinstance(prob_data, dict):
                    # Extract prob_yes from nested dict (this is pre-normalization)
                    prob_yes = prob_data.get('prob_yes')
                    if prob_yes is not None and isinstance(prob_yes, (int, float)) and prob_yes >= threshold:
                        result.add(theme)
                elif isinstance(prob_data, (int, float)):
                    # Direct probability value (assumed to be pre-normalization)
                    if prob_data >= threshold:
                        result.add(theme)
    
    # Third try: topic_logprobs (convert logprobs to probabilities using e^logprob)
    # Convert logprobs to probabilities: e^logprob_yes / (e^logprob_yes + e^logprob_no)
    if not result:
        topic_logprobs = actual.get('topic_logprobs', {})
        if topic_logprobs and isinstance(topic_logprobs, dict):
            for theme, logprob_data in topic_logprobs.items():
                if isinstance(logprob_data, dict):
                    logprob_yes = logprob_data.get('yes')
                    logprob_no = logprob_data.get('no')
                    
                    if logprob_yes is not None and logprob_no is not None:
                        try:
                            # Convert logprobs to probabilities
                            prob_yes = np.exp(logprob_yes)
                            prob_no = np.exp(logprob_no)
                            # Normalize to get probability of "yes" (this is the pre-normalization step)
                            prob_yes_normalized = prob_yes / (prob_yes + prob_no) if (prob_yes + prob_no) > 0 else 0.0
                            
                            # Apply threshold on the normalized probability
                            if prob_yes_normalized >= threshold:
                                result.add(theme)
                        except (ValueError, TypeError, OverflowError):
                            continue
    
    # Fallback: use predicted_themes (already normalized) - apply threshold on normalized values
    # NOTE: This is not ideal as these are already normalized, but we use it as fallback
    if not result:
        predicted_themes = actual.get('predicted_themes', {})
        if predicted_themes and isinstance(predicted_themes, dict):
            for theme, prob in predicted_themes.items():
                try:
                    if prob is not None and not (isinstance(prob, float) and np.isnan(prob)):
                        # Apply threshold on normalized probability (fallback case)
                        if prob > threshold:
                            result.add(theme)
                except (TypeError, ValueError):
                    continue
    
    return result


def calculate_initial_metrics_from_delta(delta_entry: Dict[str, Any], threshold: float = 0.3) -> Optional[Dict[str, Any]]:
    """
    Calculate initial (pre-SGO) JSD, WD, and recall metrics from delta entry.
    
    Args:
        delta_entry: Delta entry with prediction and actual
        threshold: Probability threshold for ground truth themes
        
    Returns:
        Dictionary with initial_jsd, initial_wd, initial_recall_3k, initial_recall_4k or None
    """
    try:
        prediction = delta_entry.get('prediction', {})
        actual = delta_entry.get('actual', {})
        
        predicted_themes = prediction.get('predicted_themes', {})
        actual_themes = actual.get('predicted_themes', {})
        
        if not predicted_themes or not actual_themes:
            return None
        
        # Handle list format
        if isinstance(predicted_themes, list):
            if not predicted_themes:
                return None
            predicted_themes = {theme: 1.0 / len(predicted_themes) for theme in predicted_themes}
        
        if isinstance(actual_themes, list):
            if not actual_themes:
                return None
            actual_themes = {theme: 1.0 / len(actual_themes) for theme in actual_themes}
        
        # Filter and normalize
        def safe_float_convert(value):
            if value is None:
                return None
            try:
                fval = float(value)
                if np.isnan(fval) or np.isinf(fval):
                    return None
                return fval
            except (ValueError, TypeError):
                return None
        
        predicted_themes = {k: v for k, v in [(k, safe_float_convert(v)) for k, v in predicted_themes.items()] 
                           if v is not None}
        actual_themes = {k: v for k, v in [(k, safe_float_convert(v)) for k, v in actual_themes.items()] 
                        if v is not None}
        
        if not predicted_themes or not actual_themes:
            return None
        
        # Normalize
        try:
            predicted_themes = normalize_distribution(predicted_themes)
            actual_themes = normalize_distribution(actual_themes)
        except Exception:
            return None
        
        if not predicted_themes or not actual_themes:
            return None
        
        # Get initial JSD from delta entry if available
        initial_jsd = delta_entry.get('theme_jsd')
        if initial_jsd is None:
            # Calculate JSD
            try:
                actual_array_jsd, pred_array_jsd = align_distributions_jsd(actual_themes, predicted_themes)
                if len(actual_array_jsd) > 0 and len(pred_array_jsd) > 0:
                    initial_jsd = compute_jsd(actual_array_jsd, pred_array_jsd, epsilon=EPSILON)
                    if np.isnan(initial_jsd) or not np.isfinite(initial_jsd):
                        initial_jsd = None
                else:
                    initial_jsd = None
            except Exception:
                initial_jsd = None
        else:
            try:
                initial_jsd = float(initial_jsd)
                if np.isnan(initial_jsd) or not np.isfinite(initial_jsd):
                    initial_jsd = None
            except (ValueError, TypeError):
                initial_jsd = None
        
        # Calculate initial WD
        try:
            actual_array_wd, pred_array_wd = align_distributions_wd(actual_themes, predicted_themes)
            if len(actual_array_wd) > 0 and len(pred_array_wd) > 0:
                initial_wd = calculate_wasserstein_1_distance(actual_array_wd, pred_array_wd)
                if np.isnan(initial_wd) or not np.isfinite(initial_wd):
                    initial_wd = None
            else:
                initial_wd = None
        except Exception:
            initial_wd = None
        
        # Calculate initial recall metrics
        try:
            actual_dict = delta_entry.get('actual', {})
            gt_themes = get_ground_truth_themes_from_logprobs_or_probs(actual_dict, threshold=threshold)
            if gt_themes:
                initial_recall_3k, k_actual, top_k_3 = calculate_recall_at_max3k(predicted_themes, gt_themes, debug=False)
                initial_recall_4k, k_actual, top_k_4 = calculate_recall_at_max4k(predicted_themes, gt_themes, debug=False)
            else:
                initial_recall_3k = None
                initial_recall_4k = None
                k_actual = 0
                top_k_3 = 0
                top_k_4 = 0
        except Exception:
            initial_recall_3k = None
            initial_recall_4k = None
            k_actual = 0
            top_k_3 = 0
            top_k_4 = 0
        
        return {
            'initial_jsd': initial_jsd,
            'initial_wd': initial_wd,
            'initial_recall_3k': initial_recall_3k,
            'initial_recall_4k': initial_recall_4k,
            'k_actual': k_actual,
            'top_k_3': top_k_3,
            'top_k_4': top_k_4
        }
    except Exception as e:
        logger.debug(f"Error calculating initial metrics: {e}")
        return None


def find_best_metrics_from_journey(review: Dict[str, Any], threshold: float = 0.3) -> Optional[Dict[str, Any]]:
    """
    Find best metrics from correction_journey (minimum delta iterations).
    
    Args:
        review: Review dictionary with correction_journey
        threshold: Probability threshold for ground truth themes
        
    Returns:
        Dictionary with best_jsd, best_wd, best_recall_3k, best_recall_4k or None
    """
    try:
        actual = review.get('actual', {})
        actual_themes = actual.get('predicted_themes', {})
        
        if not actual_themes:
            return None
        
        # Handle list format
        if isinstance(actual_themes, list):
            if not actual_themes:
                return None
            actual_themes = {theme: 1.0 / len(actual_themes) for theme in actual_themes}
        
        # Normalize actual
        def safe_float_convert(value):
            if value is None:
                return None
            try:
                fval = float(value)
                if np.isnan(fval) or np.isinf(fval):
                    return None
                return fval
            except (ValueError, TypeError):
                return None
        
        actual_themes = {k: v for k, v in [(k, safe_float_convert(v)) for k, v in actual_themes.items()] 
                        if v is not None}
        
        if not actual_themes:
            return None
        
        try:
            actual_themes = normalize_distribution(actual_themes)
        except Exception:
            return None
        
        if not actual_themes:
            return None
        
        # Check initial prediction
        prediction = review.get('prediction', {})
        initial_predicted_themes = prediction.get('predicted_themes', {})
        
        best_jsd = None
        best_wd = None
        best_recall_3k = None
        best_recall_4k = None
        best_theme_delta = float('inf')
        
        # Process initial prediction
        if initial_predicted_themes:
            if isinstance(initial_predicted_themes, list):
                if initial_predicted_themes:
                    initial_predicted_themes = {theme: 1.0 / len(initial_predicted_themes) for theme in initial_predicted_themes}
                else:
                    initial_predicted_themes = {}
            
            initial_predicted_themes = {k: v for k, v in [(k, safe_float_convert(v)) for k, v in initial_predicted_themes.items()] 
                                       if v is not None}
            
            if initial_predicted_themes:
                try:
                    initial_predicted_themes = normalize_distribution(initial_predicted_themes)
                    if initial_predicted_themes:
                        # Calculate metrics for initial
                        actual_array, pred_array = align_distributions_jsd(actual_themes, initial_predicted_themes)
                        if len(actual_array) > 0 and len(pred_array) > 0:
                            jsd = compute_jsd(actual_array, pred_array, epsilon=EPSILON)
                            if not (np.isnan(jsd) or not np.isfinite(jsd)):
                                best_jsd = jsd
                                best_theme_delta = abs(jsd)  # Use JSD as proxy for theme_delta
                        
                        actual_array_wd, pred_array_wd = align_distributions_wd(actual_themes, initial_predicted_themes)
                        if len(actual_array_wd) > 0 and len(pred_array_wd) > 0:
                            wd = calculate_wasserstein_1_distance(actual_array_wd, pred_array_wd)
                            if not (np.isnan(wd) or not np.isfinite(wd)):
                                best_wd = wd
                        
                        actual_dict = review.get('actual', {})
                        gt_themes = get_ground_truth_themes_from_logprobs_or_probs(actual_dict, threshold=threshold)
                        if gt_themes:
                            recall_3k, _, _ = calculate_recall_at_max3k(initial_predicted_themes, gt_themes, debug=False)
                            recall_4k, _, _ = calculate_recall_at_max4k(initial_predicted_themes, gt_themes, debug=False)
                            if not np.isnan(recall_3k):
                                best_recall_3k = recall_3k
                            if not np.isnan(recall_4k):
                                best_recall_4k = recall_4k
                except Exception:
                    pass
        
        # Check correction_journey for better metrics
        correction_journey = review.get('correction_journey', [])
        for correction in correction_journey:
            try:
                corrected_data = correction.get('corrected_prediction_data', {})
                corrected_themes = corrected_data.get('predicted_themes', {})
                new_theme_delta = correction.get('new_theme_delta')
                
                if not corrected_themes or new_theme_delta is None:
                    continue
                
                # Convert theme_delta to float for comparison
                try:
                    new_theme_delta = float(new_theme_delta)
                    if np.isnan(new_theme_delta) or not np.isfinite(new_theme_delta):
                        continue
                except (ValueError, TypeError):
                    continue
                
                # Only update if this is better (lower delta)
                if new_theme_delta < best_theme_delta:
                    if isinstance(corrected_themes, list):
                        if corrected_themes:
                            corrected_themes = {theme: 1.0 / len(corrected_themes) for theme in corrected_themes}
                        else:
                            continue
                    
                    corrected_themes = {k: v for k, v in [(k, safe_float_convert(v)) for k, v in corrected_themes.items()] 
                                       if v is not None}
                    
                    if not corrected_themes:
                        continue
                    
                    try:
                        corrected_themes = normalize_distribution(corrected_themes)
                        if not corrected_themes:
                            continue
                        
                        # Calculate metrics
                        actual_array, pred_array = align_distributions_jsd(actual_themes, corrected_themes)
                        if len(actual_array) > 0 and len(pred_array) > 0:
                            jsd = compute_jsd(actual_array, pred_array, epsilon=EPSILON)
                            if not (np.isnan(jsd) or not np.isfinite(jsd)):
                                best_jsd = jsd
                                best_theme_delta = new_theme_delta
                        
                        actual_array_wd, pred_array_wd = align_distributions_wd(actual_themes, corrected_themes)
                        if len(actual_array_wd) > 0 and len(pred_array_wd) > 0:
                            wd = calculate_wasserstein_1_distance(actual_array_wd, pred_array_wd)
                            if not (np.isnan(wd) or not np.isfinite(wd)):
                                best_wd = wd
                        
                        actual_dict = review.get('actual', {})
                        gt_themes = get_ground_truth_themes_from_logprobs_or_probs(actual_dict, threshold=threshold)
                        if gt_themes:
                            recall_3k, _, _ = calculate_recall_at_max3k(corrected_themes, gt_themes, debug=False)
                            recall_4k, _, _ = calculate_recall_at_max4k(corrected_themes, gt_themes, debug=False)
                            if not np.isnan(recall_3k):
                                best_recall_3k = recall_3k
                            if not np.isnan(recall_4k):
                                best_recall_4k = recall_4k
                    except Exception:
                        continue
            except Exception:
                continue
        
        if best_jsd is None and best_wd is None:
            return None
        
        return {
            'best_jsd': best_jsd,
            'best_wd': best_wd,
            'best_recall_3k': best_recall_3k,
            'best_recall_4k': best_recall_4k
        }
    except Exception as e:
        logger.debug(f"Error finding best metrics: {e}")
        return None


def calculate_review_metrics(review: Dict[str, Any], threshold: float = 0.3) -> Optional[Dict[str, Any]]:
    """
    Calculate JSD, WD, Recall@max(3,k), and Recall@max(4,k) for a single review.
    
    Args:
        review: Review dictionary with 'prediction' and 'actual' keys
        threshold: Probability threshold for ground truth themes (default: 0.5)
        
    Returns:
        Dictionary with jsd, wd, recall_3k, recall_4k, k_actual, top_k_3, top_k_4 or None if processing fails
    """
    try:
        prediction = review.get('prediction', {})
        actual = review.get('actual', {})
        
        # Get predicted and actual theme distributions
        predicted_themes = prediction.get('predicted_themes', {})
        actual_themes = actual.get('predicted_themes', {})
        
        if not predicted_themes or not actual_themes:
            return None
        
        # Handle list format - convert to uniform distribution
        if isinstance(predicted_themes, list):
            if not predicted_themes:
                return None
            predicted_themes = {theme: 1.0 / len(predicted_themes) for theme in predicted_themes}
        
        if isinstance(actual_themes, list):
            if not actual_themes:
                return None
            actual_themes = {theme: 1.0 / len(actual_themes) for theme in actual_themes}
        
        # Filter out None and invalid values
        def safe_float_convert(value):
            """Safely convert value to float, returning None if conversion fails."""
            if value is None:
                return None
            try:
                fval = float(value)
                if np.isnan(fval) or np.isinf(fval):
                    return None
                return fval
            except (ValueError, TypeError):
                return None
        
        predicted_themes = {k: v for k, v in [(k, safe_float_convert(v)) for k, v in predicted_themes.items()] 
                           if v is not None}
        actual_themes = {k: v for k, v in [(k, safe_float_convert(v)) for k, v in actual_themes.items()] 
                        if v is not None}
        
        if not predicted_themes or not actual_themes:
            return None
        
        # Normalize distributions
        try:
            predicted_themes = normalize_distribution(predicted_themes)
            actual_themes = normalize_distribution(actual_themes)
        except Exception as e:
            logger.debug(f"Normalization failed: {e}")
            return None
        
        if not predicted_themes or not actual_themes:
            return None
        
        # Align distributions for JSD calculation
        try:
            actual_array_jsd, pred_array_jsd = align_distributions_jsd(actual_themes, predicted_themes)
        except Exception as e:
            logger.debug(f"JSD alignment failed: {e}")
            return None
        
        if len(actual_array_jsd) == 0 or len(pred_array_jsd) == 0:
            return None
        
        # Calculate JSD
        try:
            jsd = compute_jsd(actual_array_jsd, pred_array_jsd, epsilon=EPSILON)
            if np.isnan(jsd) or not np.isfinite(jsd):
                jsd = float('nan')
        except Exception as e:
            logger.debug(f"JSD calculation failed: {e}")
            jsd = float('nan')
        
        # Align distributions for WD calculation
        try:
            actual_array_wd, pred_array_wd = align_distributions_wd(actual_themes, predicted_themes)
        except Exception as e:
            logger.debug(f"WD alignment failed: {e}")
            actual_array_wd, pred_array_wd = actual_array_jsd, pred_array_jsd
        
        # Calculate WD
        try:
            wd = calculate_wasserstein_1_distance(actual_array_wd, pred_array_wd)
            if np.isnan(wd) or not np.isfinite(wd):
                wd = float('nan')
        except Exception as e:
            logger.debug(f"WD calculation failed: {e}")
            wd = float('nan')
        
        # Calculate Recall@max(3,k) and Recall@max(4,k) where k is determined by threshold
        try:
            gt_themes = get_ground_truth_themes_from_logprobs_or_probs(actual, threshold=threshold)
            if gt_themes:
                recall_3k, k_actual, top_k_3 = calculate_recall_at_max3k(predicted_themes, gt_themes, debug=False)
                recall_4k, k_actual, top_k_4 = calculate_recall_at_max4k(predicted_themes, gt_themes, debug=False)
            else:
                recall_3k = float('nan')
                recall_4k = float('nan')
                k_actual = 0
                top_k_3 = 0
                top_k_4 = 0
        except Exception as e:
            logger.debug(f"Recall calculation failed: {e}")
            recall_3k = float('nan')
            recall_4k = float('nan')
            k_actual = 0
            top_k_3 = 0
            top_k_4 = 0
        
        return {
            'jsd': jsd,
            'wd': wd,
            'recall_3k': recall_3k,
            'recall_4k': recall_4k,
            'k_actual': k_actual,
            'top_k_3': top_k_3,
            'top_k_4': top_k_4
        }
    except Exception as e:
        logger.debug(f"Error processing review: {e}")
        return None


def load_initial_deltas(results_dir: Path, cluster_id: str, micro_id: str) -> Optional[Dict[str, Any]]:
    """
    Load initial deltas from _delta directory.
    
    Args:
        results_dir: Path to results directory
        cluster_id: Cluster ID (e.g., "cluster_1")
        micro_id: Micro cluster ID (e.g., "micro_0")
        
    Returns:
        Dictionary with delta data or None if not found
    """
    delta_file = results_dir / "_delta" / cluster_id / f"{micro_id}_all_reviews_deltas.json"
    
    if not delta_file.exists():
        logger.debug(f"  Delta file not found: {delta_file}")
        return None
    
    try:
        with open(delta_file, 'r', encoding='utf-8') as f:
            delta_data = json.load(f)
        return delta_data
    except Exception as e:
        logger.warning(f"  Error loading delta file {delta_file}: {e}")
        return None


def create_review_key(review: Dict[str, Any]) -> str:
    """Create a review key from review data for matching."""
    user_id = review.get('user_id', '')
    asin = review.get('asin', '')
    # Try to find review index from context or use 0
    review_idx = 0  # Default, will be matched by user_id and asin
    return f"{user_id}_{asin}"


def process_micro_cluster_file(file_path: Path, threshold: float = 0.3) -> bool:
    """
    Process a single micro cluster file with comprehensive metrics:
    1. Calculate initial (pre-SGO) metrics from delta files
    2. Calculate post-SGO metrics (current prediction)
    3. Find best metrics from correction_journey
    4. Calculate improvements
    5. Aggregate all at micro cluster level
    
    Args:
        file_path: Path to micro cluster summary file
        threshold: Probability threshold for ground truth themes (default: 0.3)
        
    Returns:
        True if successful, False otherwise
    """
    try:
        logger.info(f"Processing: {file_path}")
        
        # Load file
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        user_predictions = data.get('user_predictions', {})
        if not user_predictions:
            logger.warning(f"  No user_predictions found in {file_path}")
            return False
        
        # Extract cluster and micro IDs from file path
        cluster_match = re.search(r'cluster_(\d+)', str(file_path))
        micro_match = re.search(r'micro_(\d+)', str(file_path))
        cluster_id = f"cluster_{cluster_match.group(1)}" if cluster_match else None
        micro_id = f"micro_{micro_match.group(1)}" if micro_match else None
        
        # Load initial deltas and create lookup
        results_dir = file_path.parent.parent
        initial_delta_data = None
        delta_lookup = {}  # review_key -> delta_entry
        if cluster_id and micro_id:
            initial_delta_data = load_initial_deltas(results_dir, cluster_id, micro_id)
            if initial_delta_data:
                for delta_entry in initial_delta_data.get('deltas', []):
                    review_key = delta_entry.get('review_key', '')
                    if review_key:
                        delta_lookup[review_key] = delta_entry
        
        # Collections for tribe-wise aggregation
        all_initial_jsd = []
        all_initial_wd = []
        all_initial_recall_3k = []
        all_initial_recall_4k = []
        all_post_jsd = []
        all_post_wd = []
        all_post_recall_3k = []
        all_post_recall_4k = []
        all_best_jsd = []
        all_best_wd = []
        all_best_recall_3k = []
        all_best_recall_4k = []
        all_improvement_jsd = []
        all_improvement_wd = []
        all_improvement_recall_3k = []
        all_improvement_recall_4k = []
        total_reviews = 0
        processed_reviews = 0
        
        # Process each review
        for user_id, reviews in user_predictions.items():
            if not isinstance(reviews, list):
                continue
            
            for review_idx, review in enumerate(reviews):
                total_reviews += 1
                
                # Create review key for matching with delta
                review_key = f"{user_id}_review_{review_idx}"
                asin = review.get('asin', '')
                # Try alternative key format
                alt_key = f"{user_id}_{asin}"
                
                # Find matching delta entry
                delta_entry = delta_lookup.get(review_key) or delta_lookup.get(alt_key)
                
                # Calculate initial (pre-SGO) metrics
                initial_metrics = None
                if delta_entry:
                    initial_metrics = calculate_initial_metrics_from_delta(delta_entry, threshold=threshold)
                
                # Calculate post-SGO metrics (current prediction)
                post_metrics = calculate_review_metrics(review, threshold=threshold)
                
                # Find best metrics from journey
                best_metrics = find_best_metrics_from_journey(review, threshold=threshold)
                
                # Calculate improvements
                improvement_jsd = None
                improvement_wd = None
                improvement_recall_3k = None
                improvement_recall_4k = None
                
                if initial_metrics and post_metrics:
                    # Calculate percentage improvements
                    if initial_metrics.get('initial_jsd') is not None and post_metrics.get('jsd') is not None:
                        if not np.isnan(initial_metrics['initial_jsd']) and not np.isnan(post_metrics['jsd']):
                            if initial_metrics['initial_jsd'] > 0:
                                # Percentage improvement: (initial - post) / initial * 100 (lower is better)
                                improvement_jsd = ((initial_metrics['initial_jsd'] - post_metrics['jsd']) / initial_metrics['initial_jsd']) * 100.0
                            else:
                                improvement_jsd = 0.0
                    
                    if initial_metrics.get('initial_wd') is not None and post_metrics.get('wd') is not None:
                        if not np.isnan(initial_metrics['initial_wd']) and not np.isnan(post_metrics['wd']):
                            if initial_metrics['initial_wd'] > 0:
                                # Percentage improvement: (initial - post) / initial * 100 (lower is better)
                                improvement_wd = ((initial_metrics['initial_wd'] - post_metrics['wd']) / initial_metrics['initial_wd']) * 100.0
                            else:
                                improvement_wd = 0.0
                    
                    if initial_metrics.get('initial_recall_3k') is not None and post_metrics.get('recall_3k') is not None:
                        if not np.isnan(initial_metrics['initial_recall_3k']) and not np.isnan(post_metrics['recall_3k']):
                            if initial_metrics['initial_recall_3k'] > 0:
                                # Percentage improvement: (post - initial) / initial * 100 (higher is better)
                                improvement_recall_3k = ((post_metrics['recall_3k'] - initial_metrics['initial_recall_3k']) / initial_metrics['initial_recall_3k']) * 100.0
                            elif initial_metrics['initial_recall_3k'] == 0 and post_metrics['recall_3k'] > 0:
                                # If initial is 0 and post > 0, improvement is infinite percentage-wise
                                # Use a large fixed value (e.g., 1000%) to represent "infinite" improvement
                                # This ensures it's included in mean calculation but doesn't skew it too much
                                improvement_recall_3k = 1000.0
                            else:
                                # Both are 0, no improvement
                                improvement_recall_3k = 0.0
                    
                    if initial_metrics.get('initial_recall_4k') is not None and post_metrics.get('recall_4k') is not None:
                        if not np.isnan(initial_metrics['initial_recall_4k']) and not np.isnan(post_metrics['recall_4k']):
                            if initial_metrics['initial_recall_4k'] > 0:
                                # Percentage improvement: (post - initial) / initial * 100 (higher is better)
                                improvement_recall_4k = ((post_metrics['recall_4k'] - initial_metrics['initial_recall_4k']) / initial_metrics['initial_recall_4k']) * 100.0
                            elif initial_metrics['initial_recall_4k'] == 0 and post_metrics['recall_4k'] > 0:
                                # If initial is 0 and post > 0, improvement is infinite percentage-wise
                                # Use a large fixed value (e.g., 1000%) to represent "infinite" improvement
                                # This ensures it's included in mean calculation but doesn't skew it too much
                                improvement_recall_4k = 1000.0
                            else:
                                # Both are 0, no improvement
                                improvement_recall_4k = 0.0
                
                # Add all metrics to review
                if 'metrics' not in review:
                    review['metrics'] = {}
                
                # Initial metrics
                if initial_metrics:
                    review['metrics']['initial_jsd'] = initial_metrics.get('initial_jsd')
                    review['metrics']['initial_wd'] = initial_metrics.get('initial_wd')
                    review['metrics']['initial_recall_3k'] = initial_metrics.get('initial_recall_3k')
                    review['metrics']['initial_recall_4k'] = initial_metrics.get('initial_recall_4k')
                else:
                    review['metrics']['initial_jsd'] = None
                    review['metrics']['initial_wd'] = None
                    review['metrics']['initial_recall_3k'] = None
                    review['metrics']['initial_recall_4k'] = None
                
                # Post-SGO metrics
                if post_metrics:
                    review['metrics']['jsd'] = post_metrics.get('jsd')  # Post-SGO JSD
                    review['metrics']['wd'] = post_metrics.get('wd')  # Post-SGO WD
                    review['metrics']['recall_3k'] = post_metrics.get('recall_3k')  # Post-SGO recall@max(3,k)
                    review['metrics']['recall_4k'] = post_metrics.get('recall_4k')  # Post-SGO recall@max(4,k)
                    review['metrics']['k_actual'] = post_metrics.get('k_actual')
                    review['metrics']['top_k_3'] = post_metrics.get('top_k_3')
                    review['metrics']['top_k_4'] = post_metrics.get('top_k_4')
                
                # Best metrics
                if best_metrics:
                    review['metrics']['best_jsd'] = best_metrics.get('best_jsd')
                    review['metrics']['best_wd'] = best_metrics.get('best_wd')
                    review['metrics']['best_recall_3k'] = best_metrics.get('best_recall_3k')
                    review['metrics']['best_recall_4k'] = best_metrics.get('best_recall_4k')
                else:
                    review['metrics']['best_jsd'] = None
                    review['metrics']['best_wd'] = None
                    review['metrics']['best_recall_3k'] = None
                    review['metrics']['best_recall_4k'] = None
                
                # Improvements
                review['metrics']['improvement_in_jsd'] = improvement_jsd
                review['metrics']['improvement_in_wd'] = improvement_wd
                review['metrics']['improvement_in_recall_3k'] = improvement_recall_3k
                review['metrics']['improvement_in_recall_4k'] = improvement_recall_4k
                
                # Collect for aggregation
                if initial_metrics:
                    if initial_metrics.get('initial_jsd') is not None and not np.isnan(initial_metrics['initial_jsd']):
                        all_initial_jsd.append(initial_metrics['initial_jsd'])
                    if initial_metrics.get('initial_wd') is not None and not np.isnan(initial_metrics['initial_wd']):
                        all_initial_wd.append(initial_metrics['initial_wd'])
                    if initial_metrics.get('initial_recall_3k') is not None and not np.isnan(initial_metrics['initial_recall_3k']):
                        all_initial_recall_3k.append(initial_metrics['initial_recall_3k'])
                    if initial_metrics.get('initial_recall_4k') is not None and not np.isnan(initial_metrics['initial_recall_4k']):
                        all_initial_recall_4k.append(initial_metrics['initial_recall_4k'])
                
                if post_metrics:
                    if post_metrics.get('jsd') is not None and not np.isnan(post_metrics['jsd']):
                        all_post_jsd.append(post_metrics['jsd'])
                    if post_metrics.get('wd') is not None and not np.isnan(post_metrics['wd']):
                        all_post_wd.append(post_metrics['wd'])
                    if post_metrics.get('recall_3k') is not None and not np.isnan(post_metrics['recall_3k']):
                        all_post_recall_3k.append(post_metrics['recall_3k'])
                    if post_metrics.get('recall_4k') is not None and not np.isnan(post_metrics['recall_4k']):
                        all_post_recall_4k.append(post_metrics['recall_4k'])
                
                if best_metrics:
                    if best_metrics.get('best_jsd') is not None and not np.isnan(best_metrics['best_jsd']):
                        all_best_jsd.append(best_metrics['best_jsd'])
                    if best_metrics.get('best_wd') is not None and not np.isnan(best_metrics['best_wd']):
                        all_best_wd.append(best_metrics['best_wd'])
                    if best_metrics.get('best_recall_3k') is not None and not np.isnan(best_metrics['best_recall_3k']):
                        all_best_recall_3k.append(best_metrics['best_recall_3k'])
                    if best_metrics.get('best_recall_4k') is not None and not np.isnan(best_metrics['best_recall_4k']):
                        all_best_recall_4k.append(best_metrics['best_recall_4k'])
                
                if improvement_jsd is not None and not np.isnan(improvement_jsd):
                    all_improvement_jsd.append(improvement_jsd)
                if improvement_wd is not None and not np.isnan(improvement_wd):
                    all_improvement_wd.append(improvement_wd)
                if improvement_recall_3k is not None and not np.isnan(improvement_recall_3k):
                    all_improvement_recall_3k.append(improvement_recall_3k)
                if improvement_recall_4k is not None and not np.isnan(improvement_recall_4k):
                    all_improvement_recall_4k.append(improvement_recall_4k)
                
                processed_reviews += 1
        
        # Calculate tribe-wise (micro cluster) averages for all metrics
        tribe_metrics = {
            'total_reviews': total_reviews,
            'processed_reviews': processed_reviews,
            'threshold_used': threshold
        }
        
        def calculate_stats(values):
            """Helper to calculate statistics for a list of values."""
            if values:
                return {
                    'mean': float(np.mean(values)),
                    'std': float(np.std(values)),
                    'median': float(np.median(values)),
                    'min': float(np.min(values)),
                    'max': float(np.max(values)),
                    'count': len(values)
                }
            return None
        
        # Initial (pre-SGO) metrics
        tribe_metrics['initial_jsd'] = calculate_stats(all_initial_jsd)
        tribe_metrics['initial_wd'] = calculate_stats(all_initial_wd)
        tribe_metrics['initial_recall_3k'] = calculate_stats(all_initial_recall_3k)
        tribe_metrics['initial_recall_4k'] = calculate_stats(all_initial_recall_4k)
        
        # Post-SGO metrics
        tribe_metrics['jsd'] = calculate_stats(all_post_jsd)  # Post-SGO JSD
        tribe_metrics['wd'] = calculate_stats(all_post_wd)  # Post-SGO WD
        tribe_metrics['recall_3k'] = calculate_stats(all_post_recall_3k)  # Post-SGO recall@max(3,k)
        tribe_metrics['recall_4k'] = calculate_stats(all_post_recall_4k)  # Post-SGO recall@max(4,k)
        
        # Best metrics
        tribe_metrics['best_jsd'] = calculate_stats(all_best_jsd)
        tribe_metrics['best_wd'] = calculate_stats(all_best_wd)
        tribe_metrics['best_recall_3k'] = calculate_stats(all_best_recall_3k)
        tribe_metrics['best_recall_4k'] = calculate_stats(all_best_recall_4k)
        
        # Improvements
        tribe_metrics['improvement_in_jsd'] = calculate_stats(all_improvement_jsd)
        tribe_metrics['improvement_in_wd'] = calculate_stats(all_improvement_wd)
        tribe_metrics['improvement_in_recall_3k'] = calculate_stats(all_improvement_recall_3k)
        tribe_metrics['improvement_in_recall_4k'] = calculate_stats(all_improvement_recall_4k)
        
        # Load and process initial deltas
        initial_deltas_list = []
        if initial_delta_data:
            for delta_entry in initial_delta_data.get('deltas', []):
                overall_delta = delta_entry.get('overall_delta')
                if overall_delta is not None:
                    try:
                        initial_deltas_list.append(float(overall_delta))
                    except (ValueError, TypeError):
                        continue
        
        # Calculate initial_deltas statistics
        if initial_deltas_list:
            tribe_metrics['initial_deltas'] = {
                'mean': float(np.mean(initial_deltas_list)),
                'std': float(np.std(initial_deltas_list)),
                'median': float(np.median(initial_deltas_list)),
                'min': float(np.min(initial_deltas_list)),
                'max': float(np.max(initial_deltas_list)),
                'count': len(initial_deltas_list)
            }
            # Best delta is the minimum (lowest) overall_delta
            tribe_metrics['best_delta'] = float(np.min(initial_deltas_list))
        else:
            tribe_metrics['initial_deltas'] = None
            tribe_metrics['best_delta'] = None
        
        # Add tribe_metrics to the data
        data['tribe_metrics'] = tribe_metrics
        
        # Save updated file (replace NaN with null for valid JSON)
        def replace_nan(obj):
            """Recursively replace NaN with null in JSON-serializable objects."""
            if isinstance(obj, dict):
                return {k: replace_nan(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [replace_nan(item) for item in obj]
            elif isinstance(obj, float) and (np.isnan(obj) or np.isinf(obj)):
                return None
            return obj
        
        # Clean data before saving
        data_clean = replace_nan(data)
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data_clean, f, indent=2, ensure_ascii=False)
        
        logger.info(f"  ✅ Processed {processed_reviews}/{total_reviews} reviews")
        if tribe_metrics['initial_jsd']:
            logger.info(f"     Initial JSD: mean={tribe_metrics['initial_jsd']['mean']:.6f}, count={tribe_metrics['initial_jsd']['count']}")
        if tribe_metrics['initial_wd']:
            logger.info(f"     Initial WD: mean={tribe_metrics['initial_wd']['mean']:.6f}, count={tribe_metrics['initial_wd']['count']}")
        if tribe_metrics['best_jsd']:
            logger.info(f"     Best JSD: mean={tribe_metrics['best_jsd']['mean']:.6f}, count={tribe_metrics['best_jsd']['count']}")
        if tribe_metrics['best_wd']:
            logger.info(f"     Best WD: mean={tribe_metrics['best_wd']['mean']:.6f}, count={tribe_metrics['best_wd']['count']}")
        if tribe_metrics['improvement_in_jsd']:
            logger.info(f"     Improvement in JSD: mean={tribe_metrics['improvement_in_jsd']['mean']:.6f}, count={tribe_metrics['improvement_in_jsd']['count']}")
        if tribe_metrics['improvement_in_wd']:
            logger.info(f"     Improvement in WD: mean={tribe_metrics['improvement_in_wd']['mean']:.6f}, count={tribe_metrics['improvement_in_wd']['count']}")
        
        return True
        
    except Exception as e:
        logger.error(f"Error processing {file_path}: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description='Add JSD, WD, and Recall@max(4,k) metrics to micro cluster files',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('--results-dir', type=str, default=None,
                       help='Path to sgo_training_results directory (default: script directory)')
    parser.add_argument('--threshold', type=float, default=0.3,
                       help='Probability threshold for determining ground truth themes (default: 0.3)')
    parser.add_argument('--only-cluster', type=str, default=None,
                       help='Process only this cluster (e.g., cluster_4)')
    parser.add_argument('--only-micro', type=str, default=None,
                       help='Process only this micro cluster (e.g., micro_15). Requires --only-cluster.')
    
    args = parser.parse_args()
    
    # Determine results directory
    if args.results_dir:
        results_dir = Path(args.results_dir)
    else:
        results_dir = Path(__file__).parent.parent / "artifacts" / "sgo_train_final_predictions_user_seed_memory"
    
    if not results_dir.exists():
        logger.error(f"ERROR: Results directory not found: {results_dir}")
        return 1
    
    threshold = args.threshold
    
    logger.info("=" * 80)
    logger.info("ADDING JSD, WD, AND RECALL@MAX(4,K) METRICS TO MICRO CLUSTER FILES")
    logger.info("=" * 80)
    logger.info(f"Results directory: {results_dir}")
    logger.info(f"Threshold: {threshold}")
    
    # Find all micro cluster summary files
    micro_cluster_files = []
    for cluster_dir in sorted(results_dir.glob('cluster_*')):
        if not cluster_dir.is_dir():
            continue
        
        cluster_id = cluster_dir.name
        
        # Filter by cluster if specified
        if args.only_cluster and cluster_id != args.only_cluster:
            continue
        
        for summary_file in sorted(cluster_dir.glob('micro_*_summary_*.json')):
            # Extract micro_id from filename
            micro_match = re.search(r'micro_(\d+)', summary_file.name)
            if micro_match:
                micro_id = f"micro_{micro_match.group(1)}"
                
                # Filter by micro if specified
                if args.only_micro and micro_id != args.only_micro:
                    continue
                
                micro_cluster_files.append(summary_file)
    
    if not micro_cluster_files:
        logger.warning("No micro cluster files found")
        return 1
    
    logger.info(f"Found {len(micro_cluster_files)} micro cluster files to process")
    
    # Process each file
    success_count = 0
    fail_count = 0
    
    for file_path in micro_cluster_files:
        if process_micro_cluster_file(file_path, threshold=threshold):
            success_count += 1
        else:
            fail_count += 1
    
    logger.info("=" * 80)
    logger.info(f"Processing complete: {success_count} succeeded, {fail_count} failed")
    logger.info("=" * 80)
    
    return 0 if fail_count == 0 else 1


if __name__ == "__main__":
    sys.exit(main())

