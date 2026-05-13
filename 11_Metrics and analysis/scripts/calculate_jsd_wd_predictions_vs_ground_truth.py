#!/usr/bin/env python3
"""
Calculate JSD and WD for Predictions vs Ground Truth
====================================================

This script:
1. Loads predictions from all_predictions_o3_full_history.json
2. Loads ground truth from ground_truth_train_micro_cluster_details/
3. Matches by user_id, product_description, and category
4. Calculates JSD and WD between predicted_themes and topic_probabilities
5. Creates visualization graphs
6. Saves unmatched predictions to failed-predictions.json

Usage:
    python Metrics and analysis/scripts/calculate_jsd_wd_predictions_vs_ground_truth.py
"""

import json
import numpy as np
import sys
import logging
import re
import yaml
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import seaborn as sns
from scipy.stats import entropy

# Add project root to path
BASE_DIR = Path(__file__).parent.parent.parent
sys.path.insert(0, str(BASE_DIR))

from utils.wandb_utils import (
    init_wandb_run,
    finish_run,
    log_artifact,
    get_stage_config,
)

try:
    import ot  # Python Optimal Transport library
    HAS_OT = True
except ImportError:
    ot = None
    HAS_OT = False
    logging.warning("POT (Python Optimal Transport) library not found. Install with: pip install POT")
    logging.warning("Falling back to L1 distance approximation for WD")

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
    
    Args:
        actual_array: Ground truth probability distribution (normalized, sums to 1.0)
        predicted_array: Predicted probability distribution (normalized, sums to 1.0)
        
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


def normalize_string(s: str) -> str:
    """
    Normalize string for matching (lowercase, strip whitespace).
    
    Args:
        s: Input string
        
    Returns:
        Normalized string
    """
    if not s:
        return ""
    return s.lower().strip()


def create_match_key(user_id: str, product_description: str, category: str) -> str:
    """
    Create a normalized match key for matching predictions with ground truth.
    
    Args:
        user_id: User ID
        product_description: Product description
        category: Category
        
    Returns:
        Normalized match key
    """
    # Normalize all components
    norm_user = normalize_string(user_id)
    norm_product = normalize_string(product_description)
    norm_category = normalize_string(category)
    
    # Create key
    return f"{norm_user}|||{norm_product}|||{norm_category}"


def load_predictions(predictions_file: Path) -> List[Dict[str, Any]]:
    """
    Load predictions from reviews_with_topic_classifications.json file.
    
    Args:
        predictions_file: Path to predictions JSON file
        
    Returns:
        List of prediction dictionaries
    """
    logger.info(f"Loading predictions from: {predictions_file}")
    
    # Try standard JSON loading first
    try:
        with open(predictions_file, 'r', encoding='utf-8', errors='replace') as f:
            predictions = json.load(f)
        logger.info(f"Loaded {len(predictions)} predictions")
        return predictions
    except json.JSONDecodeError as e:
        logger.error(f"JSON parsing error at line {e.lineno}, column {e.colno}: {e.msg}")
        logger.warning("Attempting to load with error recovery...")
        original_error = e
        
        # Try loading with error recovery using ijson if available
        try:
            import ijson
            logger.info("Using ijson for streaming JSON parsing...")
            predictions = []
            with open(predictions_file, 'rb') as f:
                parser = ijson.items(f, 'item')
                for item in parser:
                    predictions.append(item)
            logger.info(f"Loaded {len(predictions)} predictions using ijson")
            return predictions
        except ImportError:
            logger.warning("ijson not available. Install with: pip install ijson")
        except Exception as e2:
            logger.error(f"ijson parsing also failed: {e2}")
        
        # Fallback: try to fix common encoding issues and reload
        logger.warning("Attempting to fix encoding issues and reload...")
        try:
            with open(predictions_file, 'r', encoding='utf-8', errors='replace') as f:
                content = f.read()
            
            # Try to fix common JSON issues
            # Replace common encoding artifacts
            content = content.replace('\x00', '')  # Remove null bytes
            
            try:
                predictions = json.loads(content)
                logger.info(f"Loaded {len(predictions)} predictions after encoding fix")
                return predictions
            except json.JSONDecodeError as e3:
                logger.error(f"Still failed after encoding fix: {e3}")
                raise RuntimeError(
                    f"Failed to parse JSON file. Original error at line {original_error.lineno}, column {original_error.colno}: {original_error.msg}\n"
                    f"Please check the JSON file for syntax errors or encoding issues."
                ) from e3
        except Exception as e_fallback:
            logger.error(f"Fallback encoding fix also failed: {e_fallback}")
            raise RuntimeError(
                f"Failed to parse JSON file. Original error at line {original_error.lineno}, column {original_error.colno}: {original_error.msg}\n"
                f"Please check the JSON file for syntax errors or encoding issues."
            ) from original_error
    except Exception as e:
        logger.error(f"Unexpected error loading predictions: {e}")
        raise


def load_test_prediction_accuracy_format(test_accuracy_dir: Path) -> List[Dict[str, Any]]:
    """
    Load predictions from Test_Prediction_Accuracy format.
    
    This format has:
    - Files in cluster_X/micro_Y_summary_enhanced_persona_micro_cluster_accuracy.json
    - Structure: user_predictions[user_id] = [list of reviews]
    - Each review has prediction.predicted_themes (dict) and actual.predicted_themes (list)
    
    Args:
        test_accuracy_dir: Path to Test_Prediction_Accuracy directory
        
    Returns:
        List of review dictionaries with prediction and actual already paired
    """
    logger.info(f"Loading Test_Prediction_Accuracy format from: {test_accuracy_dir}")
    
    all_reviews = []
    total_files = 0
    
    # Find all cluster directories
    cluster_dirs = sorted([d for d in test_accuracy_dir.iterdir() if d.is_dir() and d.name.startswith('cluster_')])
    logger.info(f"Found {len(cluster_dirs)} cluster directories")
    
    for cluster_dir in tqdm(cluster_dirs, desc="Loading clusters"):
        cluster_id = cluster_dir.name
        
        # Find all accuracy files
        accuracy_files = sorted(cluster_dir.glob("micro_*_summary_enhanced_persona_micro_cluster_accuracy.json"))
        
        for accuracy_file in accuracy_files:
            micro_id = accuracy_file.stem.replace('_summary_enhanced_persona_micro_cluster_accuracy', '')
            
            try:
                with open(accuracy_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # Extract metadata
                metadata = data.get('metadata', {})
                persona_name = metadata.get('persona_name', 'Unknown')
                
                # Extract user_predictions
                user_predictions = data.get('user_predictions', {})
                
                # Process each user's reviews
                for user_id, reviews in user_predictions.items():
                    if not isinstance(reviews, list):
                        continue
                    
                    for review in reviews:
                        if not isinstance(review, dict):
                            continue
                        
                        product_description = review.get('product_description', '')
                        category = review.get('category', '')
                        prediction_obj = review.get('prediction', {})
                        actual_obj = review.get('actual', {})
                        
                        # Skip if missing required fields
                        if not product_description or not category:
                            continue
                        
                        # Extract predicted themes (should be a dict with normalized scores)
                        pred_themes = prediction_obj.get('predicted_themes', {})
                        if not isinstance(pred_themes, dict):
                            continue
                        
                        # Extract actual themes (can be a list or dict)
                        actual_themes_raw = actual_obj.get('predicted_themes', [])
                        
                        # Convert actual themes list to uniform distribution
                        if isinstance(actual_themes_raw, list):
                            if actual_themes_raw:
                                # Create uniform distribution
                                actual_themes = {theme: 1.0 / len(actual_themes_raw) for theme in actual_themes_raw}
                            else:
                                actual_themes = {}
                        elif isinstance(actual_themes_raw, dict):
                            # Already a dict, use as is
                            actual_themes = actual_themes_raw
                        else:
                            actual_themes = {}
                        
                        # Create review entry
                        review_entry = {
                            'user_id': user_id,
                            'product_description': product_description,
                            'category': category,
                            'prediction': {
                                'review_text': prediction_obj.get('review_text', ''),
                                'rating': prediction_obj.get('rating'),
                                'sentiment': prediction_obj.get('sentiment'),
                                'predicted_themes': pred_themes
                            },
                            'actual': {
                                'review_text': actual_obj.get('review_text', ''),
                                'rating': actual_obj.get('rating'),
                                'sentiment': actual_obj.get('sentiment'),
                                'predicted_themes': actual_themes
                            },
                            'cluster_id': cluster_id,
                            'micro_id': micro_id,
                            'persona_name': persona_name
                        }
                        
                        all_reviews.append(review_entry)
                
                total_files += 1
                
            except Exception as e:
                logger.warning(f"Error loading {accuracy_file}: {e}")
                continue
    
    logger.info(f"Loaded {len(all_reviews)} reviews from {total_files} accuracy files")
    return all_reviews


def process_test_prediction_accuracy_format(
    reviews: List[Dict[str, Any]],
    output_file: Optional[Path] = None,
    prob_threshold: float = 0.0
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Process Test_Prediction_Accuracy format reviews and calculate JSD, WD, Recall@max(3,k), and Recall@max(4,k).
    
    In this format, prediction and actual are already paired, so we just need to
    calculate metrics directly.
    
    Args:
        reviews: List of review dictionaries with prediction and actual already paired
        output_file: Optional file path to write results incrementally (JSONL format)
        prob_threshold: Probability threshold for filtering ground truth themes from 
                       topic_probabilities_before_normalisation. Themes with probability >= threshold are included.
        
    Returns:
        Tuple of (matched_results, failed_predictions)
    """
    logger.info("Processing Test_Prediction_Accuracy format reviews...")
    
    matched_results = []
    failed_predictions = []
    
    match_stats = {
        'total_predictions': len(reviews),
        'matched': 0,
        'failed': 0,
        'failed_reasons': defaultdict(int)
    }
    
    # Open output file for incremental writing if provided
    output_fp = None
    if output_file:
        output_fp = open(output_file, 'w', encoding='utf-8')
        logger.info(f"Writing results incrementally to: {output_file}")
    
    try:
        for review in tqdm(reviews, desc="Processing reviews"):
            user_id = review.get('user_id', '')
            product_description = review.get('product_description', '')
            category = review.get('category', '')
            
            # Extract theme distributions
            pred_obj = review.get('prediction', {})
            actual_obj = review.get('actual', {})
            
            pred_themes = pred_obj.get('predicted_themes', {})
            actual_themes = actual_obj.get('predicted_themes', {})
            
            # Validate distributions
            if not pred_themes or not actual_themes:
                match_stats['failed'] += 1
                match_stats['failed_reasons']['empty_distributions'] += 1
                failed_predictions.append({
                    **review,
                    'match_failure_reason': 'empty_distributions'
                })
                continue
            
            # Normalize distributions to ensure they sum to 1.0
            pred_themes_norm = normalize_distribution(pred_themes)
            actual_themes_norm = normalize_distribution(actual_themes)
            
            if not pred_themes_norm or not actual_themes_norm:
                match_stats['failed'] += 1
                match_stats['failed_reasons']['normalization_failed'] += 1
                failed_predictions.append({
                    **review,
                    'match_failure_reason': 'normalization_failed'
                })
                continue
            
            # Align distributions to the same set of themes (union of both)
            # align_distributions returns (actual_array, predicted_array) = (gt_array, pred_array)
            actual_array, pred_array = align_distributions(actual_themes_norm, pred_themes_norm)
            
            if len(pred_array) == 0 or len(actual_array) == 0:
                match_stats['failed'] += 1
                match_stats['failed_reasons']['alignment_failed'] += 1
                failed_predictions.append({
                    **review,
                    'match_failure_reason': 'alignment_failed'
                })
                continue
            
            # Calculate JSD: Jensen-Shannon Divergence between actual and predicted
            jsd = compute_jsd(actual_array, pred_array)
            
            # Calculate WD: Wasserstein-1 Distance between actual and predicted
            wd = calculate_wasserstein_1_distance(actual_array, pred_array)
            
            if np.isnan(jsd) or np.isnan(wd) or not np.isfinite(jsd) or not np.isfinite(wd):
                match_stats['failed'] += 1
                match_stats['failed_reasons']['invalid_metrics'] += 1
                failed_predictions.append({
                    **review,
                    'match_failure_reason': 'invalid_metrics'
                })
                continue
            
            # Calculate Recall@max(3, k) and Recall@max(4, k) using topic_probabilities_before_normalisation if available
            recall_at_max3k = None
            recall_at_max4k = None
            k_actual = None
            top_k_used_3 = None
            top_k_used_4 = None
            
            # For Test_Prediction_Accuracy format, we might not have topic_probabilities_before_normalisation
            # In that case, we can use the actual_themes (which might be a list) or actual_themes_norm
            # For now, we'll use actual_themes_norm keys as ground truth themes if no before_norm available
            topic_probs_before_norm = actual_obj.get('topic_probabilities_before_normalisation', {})
            if topic_probs_before_norm:
                # Use probability threshold directly on topic_probabilities_before_normalisation
                # Map Set B to Set A before filtering
                gt_themes_set = get_ground_truth_themes_from_before_normalization(
                    topic_probs_before_norm,
                    prob_threshold,
                    category=category
                )
                
                if gt_themes_set:
                    recall_at_max3k, k_actual, top_k_used_3 = calculate_recall_at_max3k(
                        pred_themes_norm,
                        gt_themes_set
                    )
                    recall_at_max4k, _, top_k_used_4 = calculate_recall_at_max4k(
                        pred_themes_norm,
                        gt_themes_set
                    )
            else:
                # Fallback: use actual_themes_norm keys as ground truth (all themes with non-zero probability)
                # This is less ideal but works if before_normalisation is not available
                gt_themes_set = set(actual_themes_norm.keys())
                if gt_themes_set:
                    recall_at_max3k, k_actual, top_k_used_3 = calculate_recall_at_max3k(
                        pred_themes_norm,
                        gt_themes_set
                    )
                    recall_at_max4k, _, top_k_used_4 = calculate_recall_at_max4k(
                        pred_themes_norm,
                        gt_themes_set
                    )
            
            # Create result in the required format
            result = {
                'user_id': user_id,
                'product_description': product_description,
                'category': category,
                'prediction': {
                    'review_text': pred_obj.get('review_text', ''),
                    'topic_probabilities': pred_themes_norm,
                    'rating': pred_obj.get('rating'),
                    'sentiment': pred_obj.get('sentiment')
                },
                'actual': {
                    'review_text': actual_obj.get('review_text', ''),
                    'topic_probabilities': actual_themes_norm,
                    'rating': actual_obj.get('rating'),
                    'sentiment': actual_obj.get('sentiment')
                },
                'jsd': float(jsd),
                'wd': float(wd),
                'recall_at_max3k': float(recall_at_max3k) if recall_at_max3k is not None else None,
                'recall_at_max4k': float(recall_at_max4k) if recall_at_max4k is not None else None,
                'k_actual': k_actual,
                'top_k_used_3': top_k_used_3,
                'top_k_used_4': top_k_used_4,
                'cluster_id': review.get('cluster_id'),
                'micro_id': review.get('micro_id'),
                'persona_name': review.get('persona_name')
            }
            
            matched_results.append(result)
            match_stats['matched'] += 1
            
            # Write result immediately to file if output file is provided
            if output_fp:
                json.dump(result, output_fp, ensure_ascii=False)
                output_fp.write('\n')
                output_fp.flush()  # Ensure it's written to disk immediately
        
        # Print statistics
        logger.info("\n" + "=" * 80)
        logger.info("PROCESSING STATISTICS")
        logger.info("=" * 80)
        logger.info(f"Total reviews: {match_stats['total_predictions']:,}")
        logger.info(f"Matched: {match_stats['matched']:,} ({100 * match_stats['matched'] / match_stats['total_predictions']:.2f}%)")
        logger.info(f"Failed: {match_stats['failed']:,} ({100 * match_stats['failed'] / match_stats['total_predictions']:.2f}%)")
        logger.info("\nFailure reasons:")
        for reason, count in sorted(match_stats['failed_reasons'].items(), key=lambda x: -x[1]):
            logger.info(f"  {reason}: {count:,} ({100 * count / match_stats['failed']:.2f}%)")
        logger.info("=" * 80)
    
    finally:
        # Close output file if it was opened
        if output_fp:
            output_fp.close()
            logger.info(f"Finished writing results to: {output_file}")
    
    return matched_results, failed_predictions


def load_ground_truth(ground_truth_dir: Path) -> Dict[str, Dict[str, Any]]:
    """
    Load all ground truth files from cluster/micro directories.
    
    Args:
        ground_truth_dir: Path to ground_truth_train_micro_cluster_details directory
        
    Returns:
        Dictionary mapping match_key -> ground truth review data
    """
    logger.info(f"Loading ground truth from: {ground_truth_dir}")
    
    ground_truth = {}
    total_reviews = 0
    
    # Find all cluster directories
    cluster_dirs = sorted([d for d in ground_truth_dir.iterdir() if d.is_dir() and d.name.startswith('cluster_')])
    logger.info(f"Found {len(cluster_dirs)} cluster directories")
    
    for cluster_dir in tqdm(cluster_dirs, desc="Loading clusters"):
        cluster_id = cluster_dir.name
        
        # Find all micro detail files
        micro_files = sorted(cluster_dir.glob("micro_*_details.json"))
        
        for micro_file in micro_files:
            micro_id = micro_file.stem.replace('_details', '')
            
            try:
                with open(micro_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # Extract members_grouped_by_user
                members_grouped = data.get('members_grouped_by_user', {})
                persona_name = data.get('persona_name', 'Unknown')
                
                # Process each user's reviews
                for user_id, reviews in members_grouped.items():
                    if not isinstance(reviews, list):
                        continue
                    
                    for review in reviews:
                        if not isinstance(review, dict):
                            continue
                        
                        product_description = review.get('product_description', '')
                        category = review.get('category', '')
                        topic_probabilities = review.get('topic_probabilities', {})
                        
                        # Skip if missing required fields
                        if not product_description or not category:
                            continue
                        
                        # Create match key
                        match_key = create_match_key(user_id, product_description, category)
                        
                        # Store ground truth data
                        ground_truth[match_key] = {
                            'user_id': user_id,
                            'product_description': product_description,
                            'category': category,
                            'topic_probabilities': topic_probabilities,
                            'topic_probabilities_before_normalisation': review.get('topic_probabilities_before_normalisation', {}),
                            'topic_logprobs': review.get('topic_logprobs', {}),
                            'predicted_themes': review.get('predicted_themes', []),
                            'review_text': review.get('review_text', ''),
                            'rating': review.get('rating'),
                            'sentiment': review.get('sentiment'),
                            'cluster_id': cluster_id,
                            'micro_id': micro_id,
                            'persona_name': persona_name,
                            'cluster': review.get('cluster')
                        }
                        total_reviews += 1
                        
            except Exception as e:
                logger.warning(f"Error loading {micro_file}: {e}")
                continue
    
    logger.info(f"Loaded {total_reviews} ground truth reviews")
    logger.info(f"Created {len(ground_truth)} unique match keys")
    return ground_truth


def get_ground_truth_themes_from_before_normalization(
    topic_probs_before_norm: Dict[str, float],
    threshold: float,
    category: str = ""
) -> set:
    """
    Extract ground truth themes from topic_probabilities_before_normalisation using threshold.
    Maps Set B theme names to Set A before filtering.
    
    Args:
        topic_probs_before_norm: Dictionary of theme -> probability (before normalization, Set B names)
        threshold: Minimum probability to include theme
        category: Category name for theme mapping
        
    Returns:
        Set of theme names that are in ground truth (probability >= threshold, Set A names)
    """
    if not topic_probs_before_norm:
        return set()
    
    if isinstance(topic_probs_before_norm, list):
        return set(topic_probs_before_norm)
    
    # Import theme mapping function
    try:
        # Try importing from the same directory
        sys.path.insert(0, str(Path(__file__).parent))
        from map_predicted_themes_setb_to_seta import map_themes_setb_to_seta
    except ImportError:
        logger.warning("Could not import theme mapping function, using themes as-is")
        # Fallback: no mapping
        ground_truth_themes = set()
        for theme, prob in topic_probs_before_norm.items():
            if isinstance(prob, (int, float)) and prob >= threshold:
                ground_truth_themes.add(theme)
            elif isinstance(prob, dict):
                prob_yes = prob.get('prob_yes')
                if prob_yes is not None and isinstance(prob_yes, (int, float)) and prob_yes >= threshold:
                    ground_truth_themes.add(theme)
        return ground_truth_themes
    
    # First, map Set B themes to Set A
    mapped_probs = map_themes_setb_to_seta(topic_probs_before_norm, category)
    
    # Then filter by threshold
    ground_truth_themes = set()
    for theme, prob in mapped_probs.items():
        # Handle direct probability values
        if isinstance(prob, (int, float)) and prob >= threshold:
            ground_truth_themes.add(theme)
        # Handle nested dict format (if it exists)
        elif isinstance(prob, dict):
            prob_yes = prob.get('prob_yes')
            if prob_yes is not None and isinstance(prob_yes, (int, float)) and prob_yes >= threshold:
                ground_truth_themes.add(theme)
    
    return ground_truth_themes


def get_top_k_predicted(predicted_themes: Dict[str, float], k: int) -> List[str]:
    """
    Get top-k predicted themes sorted by probability.
    
    Args:
        predicted_themes: Dictionary of theme -> probability
        k: Number of top themes to return
        
    Returns:
        List of top-k theme names (sorted by probability, descending)
    """
    if not predicted_themes:
        return []
    
    if isinstance(predicted_themes, list):
        return predicted_themes[:k]
    
    # Sort by probability (descending) and return top-k
    sorted_themes = sorted(
        predicted_themes.items(),
        key=lambda x: x[1],
        reverse=True
    )
    return [theme for theme, _ in sorted_themes[:k]]


def calculate_recall_at_max_nk(
    predicted_themes: Dict[str, float],
    ground_truth_themes: set,
    n: int = 4
) -> Tuple[float, int, int]:
    """
    Calculate Recall@max(n, k) where k is the number of ground truth themes.
    
    Recall@max(n, k) = |top_max(n,k)_predicted ∩ ground_truth| / |ground_truth|
    
    Args:
        predicted_themes: Dictionary of theme -> probability (predictions)
        ground_truth_themes: Set of ground truth theme names
        n: Minimum number of predictions to consider (default: 4)
        
    Returns:
        Tuple of (recall, k, top_k)
        - recall: Recall@max(n, k) value (0.0 to 1.0)
        - k: Number of ground truth themes
        - top_k: max(n, k) - number of top predictions considered
    """
    if not ground_truth_themes:
        return 0.0, 0, 0
    
    k = len(ground_truth_themes)
    top_k = max(n, k)
    
    top_k_predicted = set(get_top_k_predicted(predicted_themes, top_k))
    intersection = top_k_predicted & ground_truth_themes
    
    recall = len(intersection) / len(ground_truth_themes) if ground_truth_themes else 0.0
    return recall, k, top_k


def calculate_recall_at_max4k(
    predicted_themes: Dict[str, float],
    ground_truth_themes: set
) -> Tuple[float, int, int]:
    """
    Calculate Recall@max(4, k) where k is the number of ground truth themes.
    
    This is a convenience wrapper around calculate_recall_at_max_nk with n=4.
    """
    return calculate_recall_at_max_nk(predicted_themes, ground_truth_themes, n=4)


def calculate_recall_at_max3k(
    predicted_themes: Dict[str, float],
    ground_truth_themes: set
) -> Tuple[float, int, int]:
    """
    Calculate Recall@max(3, k) where k is the number of ground truth themes.
    
    This is a convenience wrapper around calculate_recall_at_max_nk with n=3.
    """
    return calculate_recall_at_max_nk(predicted_themes, ground_truth_themes, n=3)


def match_and_calculate_metrics(
    predictions: List[Dict[str, Any]],
    ground_truth: Dict[str, Dict[str, Any]],
    prob_threshold: float = 0.0
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Match predictions with ground truth and calculate JSD, WD, Recall@max(3,k), and Recall@max(4,k).
    
    Args:
        predictions: List of prediction dictionaries
        ground_truth: Dictionary of ground truth reviews keyed by match_key
        prob_threshold: Probability threshold for filtering ground truth themes from 
                       topic_probabilities_before_normalisation. Themes with probability >= threshold are included.
        
    Returns:
        Tuple of (matched_results, failed_predictions)
    """
    logger.info("Matching predictions with ground truth...")
    
    matched_results = []
    failed_predictions = []
    
    match_stats = {
        'total_predictions': len(predictions),
        'matched': 0,
        'failed': 0,
        'failed_reasons': defaultdict(int)
    }
    
    for pred in tqdm(predictions, desc="Matching predictions"):
        user_id = pred.get('user_id', '')
        product_description = pred.get('product_description', '')
        category = pred.get('category', '')
        
        # Check if required fields are present
        if not user_id:
            match_stats['failed'] += 1
            match_stats['failed_reasons']['missing_user_id'] += 1
            failed_predictions.append({
                **pred,
                'match_failure_reason': 'missing_user_id'
            })
            continue
        
        if not product_description:
            match_stats['failed'] += 1
            match_stats['failed_reasons']['missing_product_description'] += 1
            failed_predictions.append({
                **pred,
                'match_failure_reason': 'missing_product_description'
            })
            continue
        
        if not category:
            match_stats['failed'] += 1
            match_stats['failed_reasons']['missing_category'] += 1
            failed_predictions.append({
                **pred,
                'match_failure_reason': 'missing_category'
            })
            continue
        
        # Create match key
        match_key = create_match_key(user_id, product_description, category)
        
        # Try to find match
        gt_review = ground_truth.get(match_key)
        
        if not gt_review:
            match_stats['failed'] += 1
            match_stats['failed_reasons']['no_ground_truth_match'] += 1
            failed_predictions.append({
                **pred,
                'match_failure_reason': 'no_ground_truth_match',
                'match_key': match_key
            })
            continue
        
        # Extract theme distributions for JSD and WD calculation
        # Support multiple file formats:
        # 1. reviews_with_topic_classifications.json: topic_probabilities at top level
        # 2. all_predictions_o3_full_history.json: prediction.predicted_themes (dict)
        pred_themes = pred.get('topic_probabilities', {})  # Format 1
        if not pred_themes:
            # Try format 2: prediction.predicted_themes
            pred_obj = pred.get('prediction', {})
            if isinstance(pred_obj, dict):
                pred_themes = pred_obj.get('predicted_themes', {})
                # If it's a list, convert to uniform distribution
                if isinstance(pred_themes, list):
                    if pred_themes:
                        pred_themes = {theme: 1.0 / len(pred_themes) for theme in pred_themes}
                    else:
                        pred_themes = {}
        
        gt_themes = gt_review.get('topic_probabilities', {})  # From ground truth
        
        # Validate distributions
        if not pred_themes or not gt_themes:
            match_stats['failed'] += 1
            match_stats['failed_reasons']['empty_distributions'] += 1
            failed_predictions.append({
                **pred,
                'match_failure_reason': 'empty_distributions'
            })
            continue
        
        # Normalize distributions to ensure they sum to 1.0
        pred_themes_norm = normalize_distribution(pred_themes)
        gt_themes_norm = normalize_distribution(gt_themes)
        
        if not pred_themes_norm or not gt_themes_norm:
            match_stats['failed'] += 1
            match_stats['failed_reasons']['normalization_failed'] += 1
            failed_predictions.append({
                **pred,
                'match_failure_reason': 'normalization_failed'
            })
            continue
        
        # Align distributions to the same set of themes (union of both)
        # align_distributions returns (actual_array, predicted_array) = (gt_array, pred_array)
        gt_array, pred_array = align_distributions(gt_themes_norm, pred_themes_norm)
        
        if len(pred_array) == 0 or len(gt_array) == 0:
            match_stats['failed'] += 1
            match_stats['failed_reasons']['alignment_failed'] += 1
            failed_predictions.append({
                **pred,
                'match_failure_reason': 'alignment_failed'
            })
            continue
        
        # Calculate JSD: Jensen-Shannon Divergence between ground truth and predicted topic_probabilities
        jsd = compute_jsd(gt_array, pred_array)
        
        # Calculate WD: Wasserstein-1 Distance between ground truth and predicted topic_probabilities
        wd = calculate_wasserstein_1_distance(gt_array, pred_array)
        
        if np.isnan(jsd) or np.isnan(wd) or not np.isfinite(jsd) or not np.isfinite(wd):
            match_stats['failed'] += 1
            match_stats['failed_reasons']['invalid_metrics'] += 1
            failed_predictions.append({
                **pred,
                'match_failure_reason': 'invalid_metrics'
            })
            continue
        
        # Calculate Recall@max(3, k) and Recall@max(4, k) using topic_probabilities_before_normalisation
        recall_at_max3k = None
        recall_at_max4k = None
        k_actual = None
        top_k_used_3 = None
        top_k_used_4 = None
        
        topic_probs_before_norm = gt_review.get('topic_probabilities_before_normalisation', {})
        if topic_probs_before_norm:
            # Get ground truth themes using probability threshold directly on before-normalization probabilities
            # Map Set B to Set A before filtering
            gt_themes_set = get_ground_truth_themes_from_before_normalization(
                topic_probs_before_norm,
                prob_threshold,
                category=category
            )
            
            if gt_themes_set:
                recall_at_max3k, k_actual, top_k_used_3 = calculate_recall_at_max3k(
                    pred_themes_norm,
                    gt_themes_set
                )
                recall_at_max4k, _, top_k_used_4 = calculate_recall_at_max4k(
                    pred_themes_norm,
                    gt_themes_set
                )
        
        # Extract prediction and actual data
        # Support multiple formats
        pred_obj = pred.get('prediction', {})
        if isinstance(pred_obj, dict):
            pred_review_text = pred_obj.get('review_text', '')
        else:
            pred_review_text = pred.get('review_text', '')
        
        pred_topic_logprobs = pred.get('topic_logprobs', {})
        
        actual_review_text = gt_review.get('review_text', '')
        actual_topic_logprobs = gt_review.get('topic_logprobs', {})
        
        # Create result in the required format
        result = {
            'user_id': user_id,
            'product_description': product_description,
            'category': category,
            'prediction': {
                'review_text': pred_review_text,
                'topic_probabilities': pred_themes_norm,
                'topic_logprobs': pred_topic_logprobs
            },
            'actual': {
                'review_text': actual_review_text,
                'topic_probabilities': gt_themes_norm,
                'topic_logprobs': actual_topic_logprobs
            },
            'jsd': float(jsd),
            'wd': float(wd),
            'recall_at_max3k': float(recall_at_max3k) if recall_at_max3k is not None else None,
            'recall_at_max4k': float(recall_at_max4k) if recall_at_max4k is not None else None,
            'k_actual': k_actual,
            'top_k_used_3': top_k_used_3,
            'top_k_used_4': top_k_used_4,
            'cluster_id': gt_review.get('cluster_id'),
            'micro_id': gt_review.get('micro_id'),
            'persona_name': gt_review.get('persona_name'),
            'cluster': gt_review.get('cluster'),
            'review_index': pred.get('review_index'),
            'review_type': pred.get('review_type')
        }
        
        matched_results.append(result)
        match_stats['matched'] += 1
    
    # Print statistics
    logger.info("\n" + "=" * 80)
    logger.info("MATCHING STATISTICS")
    logger.info("=" * 80)
    logger.info(f"Total predictions: {match_stats['total_predictions']:,}")
    logger.info(f"Matched: {match_stats['matched']:,} ({100 * match_stats['matched'] / match_stats['total_predictions']:.2f}%)")
    logger.info(f"Failed: {match_stats['failed']:,} ({100 * match_stats['failed'] / match_stats['total_predictions']:.2f}%)")
    logger.info("\nFailure reasons:")
    for reason, count in sorted(match_stats['failed_reasons'].items(), key=lambda x: -x[1]):
        logger.info(f"  {reason}: {count:,} ({100 * count / match_stats['failed']:.2f}%)")
    
    # Show sample match keys for debugging
    if matched_results:
        logger.info("\nSample matched keys (first 3):")
        for i, result in enumerate(matched_results[:3]):
            match_key = create_match_key(
                result['user_id'],
                result['product_description'],
                result['category']
            )
            logger.info(f"  {i+1}. {match_key[:100]}...")
    
    if failed_predictions and len(failed_predictions) > 0:
        logger.info("\nSample failed predictions (first 3):")
        for i, failed in enumerate(failed_predictions[:3]):
            if 'match_key' in failed:
                logger.info(f"  {i+1}. Key: {failed['match_key'][:100]}... | Reason: {failed.get('match_failure_reason', 'unknown')}")
            else:
                logger.info(f"  {i+1}. User: {failed.get('user_id', 'N/A')[:30]}... | Reason: {failed.get('match_failure_reason', 'unknown')}")
    
    logger.info("=" * 80)
    
    return matched_results, failed_predictions


def create_visualizations(
    results: List[Dict[str, Any]],
    jsd_output_dir: Path,
    wd_output_dir: Path,
    filename_suffix: str = "backstory"
):
    """
    Create visualization graphs for JSD and WD analysis.
    JSD graphs go to jsd_output_dir (for jsd_metrics_and_graphs artifact);
    WD graphs go to wd_output_dir (for wd_metrics_and_graphs artifact).
    
    Args:
        results: List of matched results with JSD and WD
        jsd_output_dir: Output directory for JSD graphs (and combined stats)
        wd_output_dir: Output directory for WD graphs
    """
    logger.info("Creating visualizations...")
    
    jsd_output_dir.mkdir(parents=True, exist_ok=True)
    wd_output_dir.mkdir(parents=True, exist_ok=True)
    
    # Set style
    sns.set_style("whitegrid")
    plt.rcParams['figure.dpi'] = 300
    plt.rcParams['savefig.dpi'] = 300
    
    if not results:
        logger.warning("No results to visualize")
        return
    
    # Extract metrics
    jsd_values = np.array([r['jsd'] for r in results])
    wd_values = np.array([r['wd'] for r in results if not np.isnan(r['wd'])])
    
    # Calculate per-cluster statistics
    cluster_jsd = defaultdict(list)
    cluster_wd = defaultdict(list)
    cluster_tribe_jsd = defaultdict(lambda: defaultdict(list))
    cluster_tribe_wd = defaultdict(lambda: defaultdict(list))
    tribe_names_map = {}
    
    for r in results:
        cluster_id = r.get('cluster_id')
        micro_id = r.get('micro_id')
        persona_name = r.get('persona_name')
        
        if cluster_id:
            cluster_jsd[cluster_id].append(r['jsd'])
            if not np.isnan(r['wd']):
                cluster_wd[cluster_id].append(r['wd'])
            
            if micro_id:
                tribe_id = f"{cluster_id}/{micro_id}"
                cluster_tribe_jsd[cluster_id][tribe_id].append(r['jsd'])
                if not np.isnan(r['wd']):
                    cluster_tribe_wd[cluster_id][tribe_id].append(r['wd'])
                if persona_name and tribe_id not in tribe_names_map:
                    tribe_names_map[tribe_id] = persona_name
    
    # ========================================================================
    # Graph 1: JSD Distribution Histogram
    # ========================================================================
    fig, ax = plt.subplots(figsize=(12, 7))
    bins = np.linspace(jsd_values.min(), jsd_values.max(), 30)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    hist, _ = np.histogram(jsd_values, bins=bins)
    hist_percent = (hist / len(jsd_values)) * 100
    
    bars = ax.bar(bin_centers, hist_percent, width=(bins[1]-bins[0])*0.8, 
                 alpha=0.8, color='#4A7FB5', edgecolor='#2B5A82', linewidth=1.2)
    
    mean_jsd = np.mean(jsd_values)
    median_jsd = np.median(jsd_values)
    ax.axvline(mean_jsd, color='#FF6B6B', linestyle='--', linewidth=2.5, 
              label=f'Mean: {mean_jsd:.4f}', zorder=5)
    ax.axvline(median_jsd, color='#51CF66', linestyle='--', linewidth=2.5, 
              label=f'Median: {median_jsd:.4f}', zorder=5)
    
    ax.set_xlabel('Jensen-Shannon Divergence', fontsize=13, fontweight='bold')
    ax.set_ylabel('Percentage (%)', fontsize=13, fontweight='bold')
    ax.set_title(f'JSD Distribution: Predictions vs Ground Truth\n(n={len(jsd_values):,} matched reviews)', 
                 fontsize=14, fontweight='bold', pad=15)
    ax.legend(loc='upper right', fontsize=11, framealpha=0.9)
    ax.grid(True, alpha=0.3, axis='y', linestyle=':', linewidth=0.8)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    plt.savefig(jsd_output_dir / f"jsd_distribution_{filename_suffix}.png", dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    logger.info(f"Saved: jsd_distribution_{filename_suffix}.png")
    
    # ========================================================================
    # Graph 2: WD Distribution Histogram
    # ========================================================================
    if len(wd_values) > 0:
        fig, ax = plt.subplots(figsize=(12, 7))
        bins = np.linspace(wd_values.min(), wd_values.max(), 30)
        bin_centers = (bins[:-1] + bins[1:]) / 2
        hist, _ = np.histogram(wd_values, bins=bins)
        hist_percent = (hist / len(wd_values)) * 100
        
        bars = ax.bar(bin_centers, hist_percent, width=(bins[1]-bins[0])*0.8, 
                     alpha=0.8, color='#E74C3C', edgecolor='#C0392B', linewidth=1.2)
        
        mean_wd = np.mean(wd_values)
        median_wd = np.median(wd_values)
        ax.axvline(mean_wd, color='#FF6B6B', linestyle='--', linewidth=2.5, 
                  label=f'Mean: {mean_wd:.4f}', zorder=5)
        ax.axvline(median_wd, color='#51CF66', linestyle='--', linewidth=2.5, 
                  label=f'Median: {median_wd:.4f}', zorder=5)
        
        ax.set_xlabel('Wasserstein Distance', fontsize=13, fontweight='bold')
        ax.set_ylabel('Percentage (%)', fontsize=13, fontweight='bold')
        ax.set_title(f'WD Distribution: Predictions vs Ground Truth\n(n={len(wd_values):,} matched reviews)', 
                     fontsize=14, fontweight='bold', pad=15)
        ax.legend(loc='upper right', fontsize=11, framealpha=0.9)
        ax.grid(True, alpha=0.3, axis='y', linestyle=':', linewidth=0.8)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        plt.tight_layout()
        plt.savefig(wd_output_dir / f"wd_distribution_{filename_suffix}.png", dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        logger.info(f"Saved: wd_distribution_{filename_suffix}.png")
    
    # ========================================================================
    # Graph 3: Per-Cluster JSD Comparison
    # ========================================================================
    if cluster_jsd:
        clusters = sorted(cluster_jsd.keys())
        cluster_means = [np.mean(cluster_jsd[c]) for c in clusters]
        cluster_counts = [len(cluster_jsd[c]) for c in clusters]
        
        fig, ax = plt.subplots(figsize=(max(12, len(clusters) * 2), 8))
        x = np.arange(len(clusters))
        bars = ax.bar(x, cluster_means, alpha=0.8, color='#4A7FB5', edgecolor='#2B5A82', linewidth=1.2)
        
        for bar, val, count in zip(bars, cluster_means, cluster_counts):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{val:.3f}\n(n={count})', ha='center', va='bottom',
                   fontsize=10, fontweight='bold')
        
        ax.set_xlabel('Cluster', fontsize=12, fontweight='bold')
        ax.set_ylabel('Mean JSD', fontsize=12, fontweight='bold')
        ax.set_title(f'JSD by Cluster\nOverall Mean: {np.mean(jsd_values):.4f}', 
                    fontsize=14, fontweight='bold', pad=20)
        ax.set_xticks(x)
        ax.set_xticklabels([c.replace('cluster_', 'Cluster ') for c in clusters])
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        plt.tight_layout()
        plt.savefig(jsd_output_dir / f"jsd_by_cluster_{filename_suffix}.png", dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Saved: jsd_by_cluster_{filename_suffix}.png")
    
    # ========================================================================
    # Graph 4: Per-Cluster WD Comparison
    # ========================================================================
    if cluster_wd:
        clusters = sorted(cluster_wd.keys())
        cluster_means = [np.mean(cluster_wd[c]) for c in clusters]
        cluster_counts = [len(cluster_wd[c]) for c in clusters]
        
        fig, ax = plt.subplots(figsize=(max(12, len(clusters) * 2), 8))
        x = np.arange(len(clusters))
        bars = ax.bar(x, cluster_means, alpha=0.8, color='#E74C3C', edgecolor='#C0392B', linewidth=1.2)
        
        for bar, val, count in zip(bars, cluster_means, cluster_counts):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{val:.3f}\n(n={count})', ha='center', va='bottom',
                   fontsize=10, fontweight='bold')
        
        ax.set_xlabel('Cluster', fontsize=12, fontweight='bold')
        ax.set_ylabel('Mean WD', fontsize=12, fontweight='bold')
        ax.set_title(f'WD by Cluster\nOverall Mean: {np.mean(wd_values):.4f}', 
                    fontsize=14, fontweight='bold', pad=20)
        ax.set_xticks(x)
        ax.set_xticklabels([c.replace('cluster_', 'Cluster ') for c in clusters])
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        plt.tight_layout()
        plt.savefig(wd_output_dir / f"wd_by_cluster_{filename_suffix}.png", dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Saved: wd_by_cluster_{filename_suffix}.png")
    
    # Save statistics (both metrics) to both dirs so each artifact has full stats
    stats = {
        'jsd': {
            'mean': float(np.mean(jsd_values)),
            'median': float(np.median(jsd_values)),
            'std': float(np.std(jsd_values)),
            'min': float(np.min(jsd_values)),
            'max': float(np.max(jsd_values)),
            '25th_percentile': float(np.percentile(jsd_values, 25)),
            '75th_percentile': float(np.percentile(jsd_values, 75)),
            'count': len(jsd_values)
        },
        'wd': {
            'mean': float(np.mean(wd_values)) if len(wd_values) > 0 else None,
            'median': float(np.median(wd_values)) if len(wd_values) > 0 else None,
            'std': float(np.std(wd_values)) if len(wd_values) > 0 else None,
            'min': float(np.min(wd_values)) if len(wd_values) > 0 else None,
            'max': float(np.max(wd_values)) if len(wd_values) > 0 else None,
            'count': len(wd_values)
        },
        'per_cluster': {
            cluster_id: {
                'jsd_mean': float(np.mean(jsds)),
                'jsd_count': len(jsds),
                'wd_mean': float(np.mean(wds)) if wds else None,
                'wd_count': len(wds) if wds else 0
            }
            for cluster_id, jsds in cluster_jsd.items()
            for wds in [cluster_wd.get(cluster_id, [])]
        }
    }
    
    for out_dir in (jsd_output_dir, wd_output_dir):
        stats_file = out_dir / f"statistics_{filename_suffix}.json"
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2)
    logger.info(f"Saved: statistics_{filename_suffix}.json")


def generate_summary_statistics(
    results: List[Dict[str, Any]],
    output_dir: Path,
    filename_suffix: str = "backstory"
) -> Dict[str, Any]:
    """
    Generate comprehensive summary statistics.
    
    Args:
        results: List of matched results with JSD and WD
        output_dir: Output directory for summary file
        
    Returns:
        Dictionary containing summary statistics
    """
    if not results:
        logger.warning("No results to generate statistics for")
        return {}
    
    jsd_values = np.array([r['jsd'] for r in results])
    wd_values = np.array([r['wd'] for r in results if not np.isnan(r['wd'])])
    
    # Calculate per-cluster statistics
    cluster_jsd = defaultdict(list)
    cluster_wd = defaultdict(list)
    cluster_recall_3 = defaultdict(list)
    cluster_recall_4 = defaultdict(list)
    cluster_counts = defaultdict(int)
    
    for r in results:
        cluster_id = r.get('cluster_id')
        if cluster_id:
            cluster_jsd[cluster_id].append(r['jsd'])
            if not np.isnan(r['wd']):
                cluster_wd[cluster_id].append(r['wd'])
            if r.get('recall_at_max3k') is not None:
                cluster_recall_3[cluster_id].append(r['recall_at_max3k'])
            if r.get('recall_at_max4k') is not None:
                cluster_recall_4[cluster_id].append(r['recall_at_max4k'])
            cluster_counts[cluster_id] += 1
    
    # Calculate per-category statistics
    category_jsd = defaultdict(list)
    category_wd = defaultdict(list)
    category_recall_3 = defaultdict(list)
    category_recall_4 = defaultdict(list)
    category_counts = defaultdict(int)
    
    for r in results:
        category = r.get('category', 'unknown')
        category_jsd[category].append(r['jsd'])
        if not np.isnan(r['wd']):
            category_wd[category].append(r['wd'])
        if r.get('recall_at_max3k') is not None:
            category_recall_3[category].append(r['recall_at_max3k'])
        if r.get('recall_at_max4k') is not None:
            category_recall_4[category].append(r['recall_at_max4k'])
        category_counts[category] += 1
    
    # Calculate recall statistics
    recall_values_3 = np.array([r['recall_at_max3k'] for r in results if r.get('recall_at_max3k') is not None])
    recall_values_4 = np.array([r['recall_at_max4k'] for r in results if r.get('recall_at_max4k') is not None])
    k_values = np.array([r['k_actual'] for r in results if r.get('k_actual') is not None])
    
    summary = {
        'overall': {
            'total_matched': len(results),
            'jsd': {
                'mean': float(np.mean(jsd_values)),
                'median': float(np.median(jsd_values)),
                'std': float(np.std(jsd_values)),
                'min': float(np.min(jsd_values)),
                'max': float(np.max(jsd_values)),
                '25th_percentile': float(np.percentile(jsd_values, 25)),
                '75th_percentile': float(np.percentile(jsd_values, 75)),
                'count': len(jsd_values)
            },
            'wd': {
                'mean': float(np.mean(wd_values)) if len(wd_values) > 0 else None,
                'median': float(np.median(wd_values)) if len(wd_values) > 0 else None,
                'std': float(np.std(wd_values)) if len(wd_values) > 0 else None,
                'min': float(np.min(wd_values)) if len(wd_values) > 0 else None,
                'max': float(np.max(wd_values)) if len(wd_values) > 0 else None,
                '25th_percentile': float(np.percentile(wd_values, 25)) if len(wd_values) > 0 else None,
                '75th_percentile': float(np.percentile(wd_values, 75)) if len(wd_values) > 0 else None,
                'count': len(wd_values)
            },
            'recall_at_max3k': {
                'mean': float(np.mean(recall_values_3)) if len(recall_values_3) > 0 else None,
                'median': float(np.median(recall_values_3)) if len(recall_values_3) > 0 else None,
                'std': float(np.std(recall_values_3)) if len(recall_values_3) > 0 else None,
                'min': float(np.min(recall_values_3)) if len(recall_values_3) > 0 else None,
                'max': float(np.max(recall_values_3)) if len(recall_values_3) > 0 else None,
                '25th_percentile': float(np.percentile(recall_values_3, 25)) if len(recall_values_3) > 0 else None,
                '75th_percentile': float(np.percentile(recall_values_3, 75)) if len(recall_values_3) > 0 else None,
                'count': len(recall_values_3)
            },
            'recall_at_max4k': {
                'mean': float(np.mean(recall_values_4)) if len(recall_values_4) > 0 else None,
                'median': float(np.median(recall_values_4)) if len(recall_values_4) > 0 else None,
                'std': float(np.std(recall_values_4)) if len(recall_values_4) > 0 else None,
                'min': float(np.min(recall_values_4)) if len(recall_values_4) > 0 else None,
                'max': float(np.max(recall_values_4)) if len(recall_values_4) > 0 else None,
                '25th_percentile': float(np.percentile(recall_values_4, 25)) if len(recall_values_4) > 0 else None,
                '75th_percentile': float(np.percentile(recall_values_4, 75)) if len(recall_values_4) > 0 else None,
                'count': len(recall_values_4)
            },
            'k_actual': {
                'mean': float(np.mean(k_values)) if len(k_values) > 0 else None,
                'median': float(np.median(k_values)) if len(k_values) > 0 else None,
                'std': float(np.std(k_values)) if len(k_values) > 0 else None,
                'min': float(np.min(k_values)) if len(k_values) > 0 else None,
                'max': float(np.max(k_values)) if len(k_values) > 0 else None,
                'count': len(k_values)
            }
        },
        'per_cluster': {
            cluster_id: {
                'count': cluster_counts[cluster_id],
                'jsd_mean': float(np.mean(jsds)),
                'jsd_median': float(np.median(jsds)),
                'jsd_std': float(np.std(jsds)),
                'wd_mean': float(np.mean(wds)) if wds else None,
                'wd_median': float(np.median(wds)) if wds else None,
                'wd_std': float(np.std(wds)) if wds else None,
                'recall_at_max3k_mean': float(np.mean(recalls_3)) if recalls_3 else None,
                'recall_at_max3k_median': float(np.median(recalls_3)) if recalls_3 else None,
                'recall_at_max3k_std': float(np.std(recalls_3)) if recalls_3 else None,
                'recall_at_max4k_mean': float(np.mean(recalls_4)) if recalls_4 else None,
                'recall_at_max4k_median': float(np.median(recalls_4)) if recalls_4 else None,
                'recall_at_max4k_std': float(np.std(recalls_4)) if recalls_4 else None
            }
            for cluster_id, jsds in cluster_jsd.items()
            for wds in [cluster_wd.get(cluster_id, [])]
            for recalls_3 in [cluster_recall_3.get(cluster_id, [])]
            for recalls_4 in [cluster_recall_4.get(cluster_id, [])]
        },
        'per_category': {
            category: {
                'count': category_counts[category],
                'jsd_mean': float(np.mean(jsds)),
                'jsd_median': float(np.median(jsds)),
                'jsd_std': float(np.std(jsds)),
                'wd_mean': float(np.mean(wds)) if wds else None,
                'wd_median': float(np.median(wds)) if wds else None,
                'wd_std': float(np.std(wds)) if wds else None,
                'recall_at_max3k_mean': float(np.mean(recalls_3)) if recalls_3 else None,
                'recall_at_max3k_median': float(np.median(recalls_3)) if recalls_3 else None,
                'recall_at_max3k_std': float(np.std(recalls_3)) if recalls_3 else None,
                'recall_at_max4k_mean': float(np.mean(recalls_4)) if recalls_4 else None,
                'recall_at_max4k_median': float(np.median(recalls_4)) if recalls_4 else None,
                'recall_at_max4k_std': float(np.std(recalls_4)) if recalls_4 else None
            }
            for category, jsds in category_jsd.items()
            for wds in [category_wd.get(category, [])]
            for recalls_3 in [category_recall_3.get(category, [])]
            for recalls_4 in [category_recall_4.get(category, [])]
        }
    }
    
    # Save summary
    summary_file = output_dir / f"summary_statistics_{filename_suffix}.json"
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    logger.info(f"Saved summary statistics to: {summary_file}")
    
    return summary


def detect_input_format(input_path: Path) -> str:
    """
    Detect the input format based on the path structure.
    
    Returns:
        'test_accuracy' if Test_Prediction_Accuracy format
        'standard' if standard format (file + ground truth dir)
    """
    if input_path.is_dir():
        # Check if it looks like Test_Prediction_Accuracy format
        # Should have cluster_X subdirectories with accuracy files
        cluster_dirs = [d for d in input_path.iterdir() if d.is_dir() and d.name.startswith('cluster_')]
        if cluster_dirs:
            # Check if any cluster has accuracy files
            for cluster_dir in cluster_dirs[:3]:  # Check first 3 clusters
                accuracy_files = list(cluster_dir.glob("micro_*_summary_enhanced_persona_micro_cluster_accuracy.json"))
                if accuracy_files:
                    return 'test_accuracy'
        return 'unknown'
    elif input_path.is_file():
        return 'standard'
    else:
        return 'unknown'


def detect_dataset_type_and_method(input_path: Path, predictions_file: Optional[Path] = None) -> Tuple[str, str]:
    """
    Detect dataset type (train/test) and method/model name from input paths.
    
    Args:
        input_path: Path to input directory or file
        predictions_file: Optional predictions file path (for standard format)
        
    Returns:
        Tuple of (dataset_type, method_name)
        dataset_type: 'train' or 'test'
        method_name: e.g., 'enhanced_persona_micro_cluster', 'backstory', 'o3', etc.
    """
    dataset_type = "unknown"
    method_name = "unknown"
    
    # Check input path name
    path_str = str(input_path).lower()
    
    # Detect dataset type
    if "test" in path_str:
        dataset_type = "test"
    elif "train" in path_str:
        dataset_type = "train"
    
    # Detect method from path
    if "test_prediction_accuracy" in path_str or "prediction_accuracy" in path_str:
        method_name = "enhanced_persona_micro_cluster"
    elif "backstory" in path_str:
        method_name = "backstory"
    elif "o3" in path_str:
        method_name = "o3"
    elif "gpt" in path_str:
        if "4o" in path_str or "4_o" in path_str:
            method_name = "gpt_4o"
        elif "4_turbo" in path_str or "4-turbo" in path_str:
            method_name = "gpt_4_turbo"
        else:
            method_name = "gpt"
    elif "claude" in path_str:
        method_name = "claude"
    
    # Also check predictions file if provided
    if predictions_file:
        pred_str = str(predictions_file).lower()
        if "test" in pred_str and dataset_type == "unknown":
            dataset_type = "test"
        elif "train" in pred_str and dataset_type == "unknown":
            dataset_type = "train"
        
        if method_name == "unknown":
            if "backstory" in pred_str:
                method_name = "backstory"
            elif "o3" in pred_str:
                method_name = "o3"
    
    # Check ground truth directory name if it's a standard format
    if "ground_truth" in path_str:
        if "test" in path_str:
            dataset_type = "test"
        elif "train" in path_str:
            dataset_type = "train"
    
    return dataset_type, method_name


def main():
    """Main execution function."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Calculate JSD and WD for predictions vs ground truth',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Standard format (predictions file + ground truth directory)
  python script.py --predictions all_predictions.json --ground-truth ground_truth_dir/
  
  # Test_Prediction_Accuracy format (directory with cluster subdirectories)
  python script.py --test-accuracy Test_Prediction_Accuracy/
        """
    )
    parser.add_argument('--predictions', type=str, help='Path to predictions JSON file (standard format)')
    parser.add_argument('--ground-truth', type=str, help='Path to ground truth directory (standard format)')
    parser.add_argument('--test-accuracy', type=str, help='Path to Test_Prediction_Accuracy directory')
    parser.add_argument('--output', type=str, help='Output directory (default: auto-detect based on input)')
    parser.add_argument('--prob-threshold', type=float, default=0.0,
                       help='Probability threshold for ground truth themes (default: 0.0). '
                            'Themes with probability >= threshold in topic_probabilities_before_normalisation are included.')
    
    args = parser.parse_args()
    
    logger.info("=" * 80)
    logger.info("JSD and WD Calculation: Predictions vs Ground Truth")
    logger.info("=" * 80)
    
    # Load config and init W&B so outputs and config are uploaded
    config = get_stage_config("Metrics and analysis")
    if not config:
        config_path = BASE_DIR / "Metrics and analysis" / "config.yaml"
        if config_path.exists():
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f) or {}
        else:
            config = {}
    run = init_wandb_run(
        run_name="calculate_jsd_wd_predictions_vs_ground_truth",
        stage="Metrics and analysis",
        config={"description": "JSD and WD for predictions vs ground truth"},
        job_type=config.get("job_type"),
    )
    if run is None:
        logger.warning("W&B run initialization failed - outputs will not be uploaded to W&B")
    
    # Determine which format to use
    use_test_accuracy_format = False
    test_accuracy_dir = None
    predictions_file = None
    ground_truth_dir = None
    
    if args.test_accuracy:
        # Explicitly specified Test_Prediction_Accuracy format
        test_accuracy_dir = Path(args.test_accuracy)
        use_test_accuracy_format = True
    elif args.predictions and args.ground_truth:
        # Explicitly specified standard format
        predictions_file = Path(args.predictions)
        ground_truth_dir = Path(args.ground_truth)
        use_test_accuracy_format = False
    else:
        # Auto-detect: try Test_Prediction_Accuracy first
        test_accuracy_dir_candidate = BASE_DIR / "Test_Prediction_Accuracy"
        if test_accuracy_dir_candidate.exists():
            format_type = detect_input_format(test_accuracy_dir_candidate)
            if format_type == 'test_accuracy':
                test_accuracy_dir = test_accuracy_dir_candidate
                use_test_accuracy_format = True
                logger.info(f"Auto-detected Test_Prediction_Accuracy format: {test_accuracy_dir}")
        
        # Fall back to standard format if not detected
        if not use_test_accuracy_format:
            predictions_file = BASE_DIR / "all_predictions_o3_backstory_test.json"
            ground_truth_dir = BASE_DIR / "ground_truth_test_micro_cluster_details"
            logger.info(f"Using standard format: predictions={predictions_file}, ground_truth={ground_truth_dir}")
    
    # Detect dataset type and method
    if use_test_accuracy_format:
        dataset_type, method_name = detect_dataset_type_and_method(test_accuracy_dir)
    else:
        dataset_type, method_name = detect_dataset_type_and_method(ground_truth_dir, predictions_file)
    
    # Default to "test" if not detected
    if dataset_type == "unknown":
        dataset_type = "test"
    if method_name == "unknown":
        method_name = "backstory" if not use_test_accuracy_format else "enhanced_persona_micro_cluster"
    
    logger.info(f"Detected dataset type: {dataset_type}, method: {method_name}")
    logger.info(f"Using probability threshold: {args.prob_threshold}")
    
    # Use config for JSD/WD output dirs (already loaded above for W&B)
    output_config = config.get('output', {})
    jsd_base = output_config.get('jsd_directory', 'Metrics and analysis/artifacts/jsd')
    wd_base = output_config.get('wd_directory', 'Metrics and analysis/artifacts/wd')
    
    subdir_name = f"predictions_vs_ground_truth_{method_name}_{dataset_type}"
    if args.output:
        base = Path(args.output)
        jsd_out = base / "jsd" / subdir_name
        wd_out = base / "wd" / subdir_name
    else:
        jsd_out = BASE_DIR / jsd_base / subdir_name
        wd_out = BASE_DIR / wd_base / subdir_name
    
    jsd_out.mkdir(parents=True, exist_ok=True)
    wd_out.mkdir(parents=True, exist_ok=True)
    logger.info(f"JSD outputs -> {jsd_out}")
    logger.info(f"WD outputs -> {wd_out}")
    
    try:
        if use_test_accuracy_format:
            # Test_Prediction_Accuracy format
            if not test_accuracy_dir or not test_accuracy_dir.exists():
                logger.error(f"Test_Prediction_Accuracy directory not found: {test_accuracy_dir}")
                return
            
            logger.info(f"Processing Test_Prediction_Accuracy format from: {test_accuracy_dir}")
            
            # Load reviews from Test_Prediction_Accuracy format
            reviews = load_test_prediction_accuracy_format(test_accuracy_dir)
            
            # Set up output file for incremental writing (JSONL format - one JSON object per line)
            # Write to JSD dir; we'll copy/save to WD dir below so both artifacts have data
            incremental_output_file = jsd_out / f"matched_results_{method_name}_{dataset_type}.jsonl"
            
            # Process and calculate metrics (prediction and actual already paired)
            # Results are written incrementally to the file as each review is processed
            matched_results, failed_predictions = process_test_prediction_accuracy_format(
                reviews, 
                output_file=incremental_output_file,
                prob_threshold=args.prob_threshold
            )
            
            # Save matched and failed to both JSD and WD dirs so each wandb artifact has full data
            for out_dir in (jsd_out, wd_out):
                matched_file = out_dir / f"matched_results_{method_name}_{dataset_type}.json"
                with open(matched_file, 'w', encoding='utf-8') as f:
                    json.dump(matched_results, f, indent=2, ensure_ascii=False)
                failed_file = out_dir / f"failed_predictions_{method_name}_{dataset_type}.json"
                with open(failed_file, 'w', encoding='utf-8') as f:
                    json.dump(failed_predictions, f, indent=2, ensure_ascii=False)
            logger.info(f"Saved matched results and failed predictions to JSD and WD output dirs")
            logger.info(f"Incremental results (JSONL) saved to: {incremental_output_file}")
            logger.info(f"Total failed predictions: {len(failed_predictions):,}")
            
        else:
            # Standard format (predictions file + ground truth directory)
            if not predictions_file or not predictions_file.exists():
                logger.error(f"Predictions file not found: {predictions_file}")
                return
            
            if not ground_truth_dir or not ground_truth_dir.exists():
                logger.error(f"Ground truth directory not found: {ground_truth_dir}")
                return
            
            logger.info(f"Processing standard format: predictions={predictions_file}, ground_truth={ground_truth_dir}")
            
            # Load data
            predictions = load_predictions(predictions_file)
            ground_truth = load_ground_truth(ground_truth_dir)
            
            # Match and calculate metrics
            matched_results, failed_predictions = match_and_calculate_metrics(
                predictions, 
                ground_truth,
                prob_threshold=args.prob_threshold
            )
            
            # Save results to both JSD and WD dirs so each wandb artifact has full data
            logger.info("\nSaving results...")
            for out_dir in (jsd_out, wd_out):
                matched_file = out_dir / f"matched_results_{method_name}_{dataset_type}.json"
                with open(matched_file, 'w', encoding='utf-8') as f:
                    json.dump(matched_results, f, indent=2, ensure_ascii=False)
                failed_file = out_dir / f"failed_predictions_{method_name}_{dataset_type}.json"
                with open(failed_file, 'w', encoding='utf-8') as f:
                    json.dump(failed_predictions, f, indent=2, ensure_ascii=False)
            logger.info(f"Saved matched results and failed predictions to JSD and WD output dirs")
            logger.info(f"Total failed predictions: {len(failed_predictions):,}")
            
        # Generate summary statistics (write to both JSD and WD dirs)
        if matched_results:
            suffix = f"{method_name}_{dataset_type}"
            generate_summary_statistics(matched_results, jsd_out, filename_suffix=suffix)
            generate_summary_statistics(matched_results, wd_out, filename_suffix=suffix)
            
            # Create visualizations: JSD graphs -> jsd_out, WD graphs -> wd_out
            create_visualizations(matched_results, jsd_out, wd_out, filename_suffix=suffix)
        
        logger.info("\n" + "=" * 80)
        logger.info("CALCULATION COMPLETE")
        logger.info("=" * 80)
        logger.info(f"Matched: {len(matched_results):,} reviews")
        logger.info(f"Failed: {len(failed_predictions):,} reviews")
        if matched_results:
            jsd_values = [r['jsd'] for r in matched_results]
            wd_values = [r['wd'] for r in matched_results if not np.isnan(r['wd'])]
            recall_values_3 = [r['recall_at_max3k'] for r in matched_results if r.get('recall_at_max3k') is not None]
            recall_values_4 = [r['recall_at_max4k'] for r in matched_results if r.get('recall_at_max4k') is not None]
            logger.info(f"Mean JSD: {np.mean(jsd_values):.6f}")
            if wd_values:
                logger.info(f"Mean WD: {np.mean(wd_values):.6f}")
            if recall_values_3:
                logger.info(f"Mean Recall@max(3,k): {np.mean(recall_values_3):.6f} (n={len(recall_values_3)})")
                logger.info(f"Median Recall@max(3,k): {np.median(recall_values_3):.6f}")
            if recall_values_4:
                logger.info(f"Mean Recall@max(4,k): {np.mean(recall_values_4):.6f} (n={len(recall_values_4)})")
                logger.info(f"Median Recall@max(4,k): {np.median(recall_values_4):.6f}")
            k_values = [r['k_actual'] for r in matched_results if r.get('k_actual') is not None]
            if k_values:
                logger.info(f"Mean k (actual themes): {np.mean(k_values):.2f}")
                logger.info(f"Median k: {np.median(k_values):.2f}")
        
        # Upload outputs to W&B (separate artifact names for this script)
        out_artifacts = config.get("output_artifacts", {})
        artifact_type = config.get("artifact_type", "result")
        if run and jsd_out.exists():
            try:
                log_artifact(
                    run=run,
                    artifact_name=out_artifacts.get("jsd_predictions_vs_gt", "jsd_predictions_vs_ground_truth"),
                    artifact_type=artifact_type,
                    artifact_path=str(jsd_out),
                    metadata={"description": "JSD metrics: predictions vs ground truth", "method": method_name, "dataset_type": dataset_type},
                    aliases=["latest"],
                )
                logger.info("✓ Uploaded JSD outputs to W&B")
            except Exception as e:
                logger.warning(f"Failed to upload JSD artifact to W&B: {e}")
        if run and wd_out.exists():
            try:
                log_artifact(
                    run=run,
                    artifact_name=out_artifacts.get("wd_predictions_vs_gt", "wd_predictions_vs_ground_truth"),
                    artifact_type=artifact_type,
                    artifact_path=str(wd_out),
                    metadata={"description": "WD metrics: predictions vs ground truth", "method": method_name, "dataset_type": dataset_type},
                    aliases=["latest"],
                )
                logger.info("✓ Uploaded WD outputs to W&B")
            except Exception as e:
                logger.warning(f"Failed to upload WD artifact to W&B: {e}")
    except Exception as e:
        logger.error(f"Error in main execution: {e}", exc_info=True)
    finally:
        if run:
            finish_run(run)


if __name__ == "__main__":
    main()

