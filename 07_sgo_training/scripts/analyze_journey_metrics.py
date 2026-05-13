#!/usr/bin/env python3
"""
Analyze journey files to calculate JSD, WD, Recall@max(3,k), Recall@max(4,k) metrics
for initial (pre-SGO) and best (post-SGO) predictions.

Approach:
1. Get initial JSD/WD from delta file (all reviews)
2. Get best JSD/WD from journey file (reviews that went through SGO)
3. For reviews in delta but NOT in journey, use initial predictions as "best"
4. Calculate averages across ALL reviews
5. Use threshold 0.5 on topic_probabilities_before_normalisation for recall
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import sys
import argparse
import logging
import re

# Add project root to path
BASE_DIR = Path(__file__).parent.parent.parent  # Should be vectorial_sapiens_v4
sys.path.insert(0, str(BASE_DIR))

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import calculation functions from category_wise_analysis_v2.py
category_analysis_path = BASE_DIR / "07_sgo_training" / "artifacts" / "sgo_training_results_v6"
sys.path.insert(0, str(category_analysis_path))
try:
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "category_wise_analysis_v2",
        category_analysis_path / "category_wise_analysis_v2.py"
    )
    category_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(category_module)
    
    normalize_distribution = category_module.normalize_distribution
    align_distributions_jsd = category_module.align_distributions_jsd
    align_distributions_wd = category_module.align_distributions_wd
    compute_jsd = category_module.compute_jsd
    calculate_wasserstein_1_distance = category_module.calculate_wasserstein_1_distance
    calculate_recall_at_max3k = category_module.calculate_recall_at_max3k
    calculate_recall_at_max4k = category_module.calculate_recall_at_max4k
    EPSILON = getattr(category_module, 'EPSILON', 1e-10)
    HAS_IMPORTS = True
except (ImportError, AttributeError, FileNotFoundError) as e:
    logger.warning(f"Could not import from category_wise_analysis_v2.py: {e}. Using fallback implementations.")
    HAS_IMPORTS = False
    EPSILON = 1e-10
    
    def normalize_distribution(theme_dict: Dict[str, float]) -> Dict[str, float]:
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
        all_themes = sorted(set(actual_themes.keys()) | set(predicted_themes.keys()))
        if not all_themes:
            return np.array([]), np.array([])
        actual_array = np.array([actual_themes.get(theme, 0.0) for theme in all_themes])
        predicted_array = np.array([predicted_themes.get(theme, 0.0) for theme in all_themes])
        actual_sum = actual_array.sum()
        predicted_sum = predicted_array.sum()
        if actual_sum > EPSILON:
            actual_array = actual_array / actual_sum
        if predicted_sum > EPSILON:
            predicted_array = predicted_array / predicted_sum
        return actual_array, predicted_array
    
    def align_distributions_wd(actual_themes, predicted_themes):
        return align_distributions_jsd(actual_themes, predicted_themes)
    
    def compute_jsd(P: np.ndarray, Q: np.ndarray, epsilon: float = EPSILON) -> float:
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
        if len(actual_array) == 0 or len(predicted_array) == 0:
            return float('nan')
        if len(actual_array) != len(predicted_array):
            return float('nan')
        try:
            import ot
            P = predicted_array / predicted_array.sum() if predicted_array.sum() > EPSILON else predicted_array
            Q = actual_array / actual_array.sum() if actual_array.sum() > EPSILON else actual_array
            n = len(P)
            cost_matrix = np.ones((n, n)) - np.eye(n)
            wd = ot.emd2(P, Q, cost_matrix, numItermax=1000000)
            return float(wd)
        except:
            return float(np.sum(np.abs(actual_array - predicted_array)))
    
    def get_top_k_predicted(predicted_themes: Dict[str, float], k: int) -> List[str]:
        if not predicted_themes:
            return []
        sorted_themes = sorted(predicted_themes.items(), key=lambda x: x[1], reverse=True)
        return [theme for theme, _ in sorted_themes[:k]]
    
    def calculate_recall_at_max3k(predicted_themes: Dict[str, float], ground_truth_themes: set, debug: bool = False) -> Tuple[float, int, int]:
        if not ground_truth_themes:
            return 0.0, 0, 0
        k = len(ground_truth_themes)
        top_k = max(3, k)
        top_k_predicted = set(get_top_k_predicted(predicted_themes, top_k))
        
        # Handle combined theme names (e.g., "Theme A + Theme B")
        num_matches = 0
        for gt_theme in ground_truth_themes:
            # Check direct match
            if gt_theme in top_k_predicted:
                num_matches += 1
            # Check if it's a combined theme (has '+')
            elif '+' in gt_theme:
                # Split combined theme and check if all parts are in predicted
                parts = [p.strip() for p in gt_theme.split('+')]
                if all(part in top_k_predicted for part in parts):
                    num_matches += 1
        
        recall = num_matches / k if k > 0 else 0.0
        return recall, k, top_k
    
    def calculate_recall_at_max4k(predicted_themes: Dict[str, float], ground_truth_themes: set, debug: bool = False) -> Tuple[float, int, int]:
        if not ground_truth_themes:
            return 0.0, 0, 0
        k = len(ground_truth_themes)
        top_k = max(4, k)
        top_k_predicted = set(get_top_k_predicted(predicted_themes, top_k))
        
        # Handle combined theme names (e.g., "Theme A + Theme B")
        num_matches = 0
        for gt_theme in ground_truth_themes:
            # Check direct match
            if gt_theme in top_k_predicted:
                num_matches += 1
            # Check if it's a combined theme (has '+')
            elif '+' in gt_theme:
                # Split combined theme and check if all parts are in predicted
                parts = [p.strip() for p in gt_theme.split('+')]
                if all(part in top_k_predicted for part in parts):
                    num_matches += 1
        
        recall = num_matches / k if k > 0 else 0.0
        return recall, k, top_k


def map_setb_to_seta_themes(topic_probs_before: Dict[str, Any], category: str) -> Dict[str, float]:
    """Map Set B themes to Set A themes using the conversion mapping."""
    try:
        # Import mapping function - try multiple paths (note: directory has spaces)
        mapping_paths = [
            BASE_DIR / "Metrics and analysis" / "scripts" / "map_predicted_themes_setb_to_seta.py",
            Path("Metrics and analysis") / "scripts" / "map_predicted_themes_setb_to_seta.py",
            Path(__file__).parent.parent.parent / "Metrics and analysis" / "scripts" / "map_predicted_themes_setb_to_seta.py",
        ]
        
        mapping_file = None
        for path in mapping_paths:
            if path.exists():
                mapping_file = path
                break
        
        if mapping_file:
            import importlib.util
            spec = importlib.util.spec_from_file_location("map_predicted_themes_setb_to_seta", mapping_file)
            map_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(map_module)
            map_themes_setb_to_seta = map_module.map_themes_setb_to_seta
        else:
            raise ImportError("Could not find map_predicted_themes_setb_to_seta.py")
        
        # Convert dict values to float if needed
        probs_dict = {}
        for theme, prob in topic_probs_before.items():
            if isinstance(prob, dict):
                # Extract prob_yes if it's a nested dict
                prob_yes = prob.get('prob_yes')
                if prob_yes is not None and isinstance(prob_yes, (int, float)):
                    probs_dict[theme] = float(prob_yes)
            elif isinstance(prob, (int, float)):
                probs_dict[theme] = float(prob)
        
        # Map Set B to Set A
        mapped = map_themes_setb_to_seta(probs_dict, category)
        return mapped
    except ImportError:
        logger.warning("Could not import map_themes_setb_to_seta, using themes as-is")
        # Fallback: return as-is
        probs_dict = {}
        for theme, prob in topic_probs_before.items():
            if isinstance(prob, (int, float)):
                probs_dict[theme] = float(prob)
            elif isinstance(prob, dict):
                prob_yes = prob.get('prob_yes')
                if prob_yes is not None and isinstance(prob_yes, (int, float)):
                    probs_dict[theme] = float(prob_yes)
        return probs_dict


def get_ground_truth_themes_from_before_normalisation(actual: Dict[str, Any], threshold: float = 0.5, category: str = "") -> set:
    """
    Extract ground truth themes from topic_probabilities_before_normalisation.
    Maps Set B themes to Set A first, then applies threshold.
    Uses threshold 0.5 by default as specified.
    
    Priority:
    1. topic_probabilities_before_normalisation (British spelling) - Map Set B → Set A, then apply threshold
    2. topic_probabilities_without_normalisation
    3. topic_probs_before_normalization
    4. topic_probabilities (with prob_yes/prob_no structure) - already Set A
    """
    result = set()
    
    # Priority 1: topic_probabilities_before_normalisation (British spelling)
    topic_probs_before = actual.get('topic_probabilities_before_normalisation', {})
    if not topic_probs_before:
        # Priority 2: topic_probabilities_without_normalisation
        topic_probs_before = actual.get('topic_probabilities_without_normalisation', {})
    if not topic_probs_before:
        # Priority 3: topic_probs_before_normalization (American spelling)
        topic_probs_before = actual.get('topic_probs_before_normalization', {})
    
    if topic_probs_before and isinstance(topic_probs_before, dict):
        # Map Set B → Set A first (if category provided)
        if category:
            mapped_probs = map_setb_to_seta_themes(topic_probs_before, category)
        else:
            # No category, use as-is (might already be Set A)
            mapped_probs = {}
            for theme, prob in topic_probs_before.items():
                if isinstance(prob, (int, float)):
                    mapped_probs[theme] = float(prob)
                elif isinstance(prob, dict):
                    prob_yes = prob.get('prob_yes')
                    if prob_yes is not None and isinstance(prob_yes, (int, float)):
                        mapped_probs[theme] = float(prob_yes)
        
        # Apply threshold on mapped Set A themes
        for theme, prob in mapped_probs.items():
            if isinstance(prob, (int, float)) and prob >= threshold:
                result.add(theme)
    
    # Priority 4: topic_probabilities (with prob_yes/prob_no structure) - already Set A
    if not result:
        topic_probabilities = actual.get('topic_probabilities', {})
        if topic_probabilities and isinstance(topic_probabilities, dict):
            for theme, prob_data in topic_probabilities.items():
                if isinstance(prob_data, dict):
                    prob_yes = prob_data.get('prob_yes')
                    if prob_yes is not None and isinstance(prob_yes, (int, float)) and prob_yes >= threshold:
                        result.add(theme)
                elif isinstance(prob_data, (int, float)) and prob_data >= threshold:
                    result.add(theme)
    
    return result


def calculate_metrics_for_prediction(
    predicted_themes: Dict[str, float], 
    actual_themes: Dict[str, float], 
    actual_for_recall: Dict[str, Any], 
    threshold: float = 0.5,
    category: str = ""
) -> Optional[Dict[str, float]]:
    """Calculate JSD, WD, Recall@max(3,k), Recall@max(4,k) for a prediction."""
    try:
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
        
        predicted_themes = {k: v for k, v in [(k, safe_float_convert(v)) for k, v in predicted_themes.items()] if v is not None}
        actual_themes = {k: v for k, v in [(k, safe_float_convert(v)) for k, v in actual_themes.items()] if v is not None}
        
        if not predicted_themes or not actual_themes:
            return None
        
        # Normalize
        predicted_themes = normalize_distribution(predicted_themes)
        actual_themes = normalize_distribution(actual_themes)
        
        if not predicted_themes or not actual_themes:
            return None
        
        # Calculate JSD
        actual_array_jsd, pred_array_jsd = align_distributions_jsd(actual_themes, predicted_themes)
        if len(actual_array_jsd) == 0 or len(pred_array_jsd) == 0:
            return None
        
        jsd = compute_jsd(actual_array_jsd, pred_array_jsd, epsilon=EPSILON)
        if np.isnan(jsd) or not np.isfinite(jsd):
            jsd = float('nan')
        
        # Calculate WD
        actual_array_wd, pred_array_wd = align_distributions_wd(actual_themes, predicted_themes)
        wd = calculate_wasserstein_1_distance(actual_array_wd, pred_array_wd)
        if np.isnan(wd) or not np.isfinite(wd):
            wd = float('nan')
        
        # Calculate Recall@max(3,k) and Recall@max(4,k) using ground truth from actual_for_recall
        gt_themes = get_ground_truth_themes_from_before_normalisation(actual_for_recall, threshold=threshold, category=category)
        if gt_themes:
            recall_3k, k_actual, top_k_3 = calculate_recall_at_max3k(predicted_themes, gt_themes, debug=False)
            recall_4k, k_actual, top_k_4 = calculate_recall_at_max4k(predicted_themes, gt_themes, debug=False)
        else:
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
        logger.debug(f"Error calculating metrics: {e}")
        return None


def find_best_prediction_from_journey(review_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Find best prediction from correction_journey (minimum overall_delta)."""
    initial_deltas = review_data.get('initial_prediction_deltas', {})
    initial_overall_delta = initial_deltas.get('overall_delta', float('inf'))
    initial_prediction = review_data.get('initial_prediction_data', {})
    
    best_overall_delta = initial_overall_delta
    best_prediction = initial_prediction
    
    correction_journey = review_data.get('correction_journey', [])
    for correction in correction_journey:
        new_overall_delta = correction.get('new_overall_delta')
        if new_overall_delta is not None:
            try:
                new_overall_delta = float(new_overall_delta)
                if new_overall_delta < best_overall_delta:
                    best_overall_delta = new_overall_delta
                    best_prediction = correction.get('corrected_prediction_data', {})
            except (ValueError, TypeError):
                continue
    
    return best_prediction if best_prediction else None


def load_ground_truth_file(results_dir: Path, cluster_id: str, micro_id: str) -> Dict[str, Dict[str, Any]]:
    """Load ground truth file to get detailed ground truth data."""
    # Try multiple possible locations for ground truth directory
    possible_gt_dirs = [
        BASE_DIR / "ground_truth_train_micro_cluster_details_converted_v2",
        results_dir.parent / "ground_truth_train_micro_cluster_details_converted_v2",
        results_dir.parent.parent / "ground_truth_train_micro_cluster_details_converted_v2",
        results_dir.parent.parent.parent / "ground_truth_train_micro_cluster_details_converted_v2",
        Path("ground_truth_train_micro_cluster_details_converted_v2"),
        Path("/media/samanvitha/Seagate HDD/vectorial_sapiens_v4/ground_truth_train_micro_cluster_details_converted_v2"),
    ]
    
    ground_truth_file = None
    for gt_dir in possible_gt_dirs:
        if gt_dir.exists():
            gt_file = gt_dir / cluster_id / f"{micro_id}_details.json"
            if gt_file.exists():
                ground_truth_file = gt_file
                break
    
    if not ground_truth_file:
        logger.warning(f"Ground truth file not found for {cluster_id}/{micro_id}. Tried paths: {[str(d / cluster_id / f'{micro_id}_details.json') for d in possible_gt_dirs]}")
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
        
        logger.info(f"Loaded {len(lookup)} ground truth reviews from {ground_truth_file}")
        return lookup
    except Exception as e:
        logger.warning(f"Error loading ground truth file {ground_truth_file}: {e}")
        return {}


def load_delta_file(results_dir: Path, cluster_id: str, micro_id: str) -> Dict[str, Dict[str, Any]]:
    """Load delta file to get all initial reviews."""
    delta_file = results_dir / "_delta" / cluster_id / f"{micro_id}_all_reviews_deltas.json"
    
    if not delta_file.exists():
        return {}
    
    try:
        with open(delta_file, 'r', encoding='utf-8') as f:
            delta_data = json.load(f)
        
        # Create lookup: (user_id, product_description) -> delta_entry
        lookup = {}
        for delta_entry in delta_data.get('deltas', []):
            user_id = delta_entry.get('user_id', '')
            product_description = delta_entry.get('product_description', '').strip()
            
            # Primary key: user_id + product_description
            if user_id and product_description:
                key = f"{user_id}_{product_description}"
                lookup[key] = delta_entry
        
        logger.info(f"Loaded {len(lookup)} delta entries from {delta_file}")
        return lookup
    except Exception as e:
        logger.warning(f"Error loading delta file {delta_file}: {e}")
        return {}


def load_journey_file(journey_file: Path) -> Dict[str, Dict[str, Any]]:
    """Load journey file and create lookup by user_id + product_description."""
    try:
        with open(journey_file, 'r', encoding='utf-8') as f:
            journey_data = json.load(f)
        
        # Create lookup: (user_id, product_description) -> review_data
        lookup = {}
        for review_key, review_data in journey_data.items():
            actual_review_data = review_data.get('actual_review_data', {})
            user_id = review_data.get('user_id', '')
            product_description = actual_review_data.get('product_description', '').strip()
            
            # Primary key: user_id + product_description
            if user_id and product_description:
                key = f"{user_id}_{product_description}"
                lookup[key] = review_data
        
        logger.info(f"Loaded {len(lookup)} journey reviews from {journey_file}")
        return lookup
    except Exception as e:
        logger.warning(f"Error loading journey file {journey_file}: {e}")
        return {}


def process_all_reviews(
    delta_lookup: Dict[str, Dict[str, Any]],
    journey_lookup: Dict[str, Dict[str, Any]],
    ground_truth_lookup: Dict[str, Dict[str, Any]],
    threshold: float = 0.5
) -> Dict[str, Any]:
    """Process all reviews from delta file, using journey for best predictions when available."""
    
    # Collections for aggregation
    all_initial_jsd = []
    all_initial_wd = []
    all_initial_recall_3k = []
    all_initial_recall_4k = []
    all_best_jsd = []
    all_best_wd = []
    all_best_recall_3k = []
    all_best_recall_4k = []
    
    total_reviews = len(delta_lookup)
    processed_reviews = 0
    
    # Process each review from delta file
    for delta_key, delta_entry in delta_lookup.items():
        try:
            prediction = delta_entry.get('prediction', {})
            actual = delta_entry.get('actual', {})
            
            if not prediction or not actual:
                continue
            
            # Get actual themes from delta (list format)
            actual_themes_list = actual.get('predicted_themes', [])
            if isinstance(actual_themes_list, list):
                if not actual_themes_list:
                    continue
                actual_themes = {theme: 1.0 / len(actual_themes_list) for theme in actual_themes_list}
            elif isinstance(actual_themes_list, dict):
                actual_themes = actual_themes_list
            else:
                continue
            
            # Get predicted themes from delta (initial prediction)
            initial_predicted_themes = prediction.get('predicted_themes', {})
            if not initial_predicted_themes:
                continue
            
            # Match with ground truth for recall calculation
            user_id = delta_entry.get('user_id', '')
            product_description = delta_entry.get('product_description', '').strip()
            actual_for_recall = actual
            
            if ground_truth_lookup:
                match_key = f"{user_id}_{product_description}"
                if match_key in ground_truth_lookup:
                    actual_for_recall = ground_truth_lookup[match_key]
            
            # Get normalized category for Set B → Set A mapping
            category_for_mapping = delta_entry.get('category', '')
            # Load category mapping
            try:
                with open('category_mapping_to_7_main.json', 'r') as f:
                    cat_mapping_data = json.load(f)
                category_mapping = cat_mapping_data.get('category_to_main_mapping', {})
                main_category = category_mapping.get(category_for_mapping, category_for_mapping)
                # Normalize
                category_normalize = {
                    'Appliances': 'Appliances',
                    'All Beauty': 'All_Beauty',
                    'Digital Music': 'Digital_Music',
                    'Video Games': 'Video_Games',
                    'Health & Personal Care': 'Health_and_Personal_Care',
                    'Software': 'Software',
                    'Clothing Shoes & Jewelry': 'Clothing_Shoes_and_Jewelry',
                }
                normalized_category = category_normalize.get(main_category, main_category)
            except Exception:
                normalized_category = ""
            
            # Calculate initial metrics
            initial_metrics = calculate_metrics_for_prediction(
                initial_predicted_themes,
                actual_themes,
                actual_for_recall,
                threshold=threshold,
                category=normalized_category
            )
            
            # Use delta file's pre-calculated JSD if available
            if initial_metrics and delta_entry.get('theme_jsd') is not None:
                try:
                    theme_jsd = float(delta_entry.get('theme_jsd'))
                    if not np.isnan(theme_jsd) and np.isfinite(theme_jsd):
                        initial_metrics['jsd'] = theme_jsd
                except (ValueError, TypeError):
                    pass
            
            # Find best prediction
            best_predicted_themes = initial_predicted_themes  # Default: use initial as best
            
            # Check if this review is in journey file
            if delta_key in journey_lookup:
                journey_review = journey_lookup[delta_key]
                best_prediction = find_best_prediction_from_journey(journey_review)
                if best_prediction:
                    best_predicted_themes = best_prediction.get('predicted_themes', {})
                    if not best_predicted_themes:
                        best_predicted_themes = initial_predicted_themes
            
            # Calculate best metrics
            best_metrics = calculate_metrics_for_prediction(
                best_predicted_themes,
                actual_themes,
                actual_for_recall,
                threshold=threshold,
                category=normalized_category
            )
            
            # Collect metrics
            if initial_metrics:
                if initial_metrics.get('jsd') is not None and not np.isnan(initial_metrics['jsd']):
                    all_initial_jsd.append(initial_metrics['jsd'])
                if initial_metrics.get('wd') is not None and not np.isnan(initial_metrics['wd']):
                    all_initial_wd.append(initial_metrics['wd'])
                if initial_metrics.get('recall_3k') is not None and not np.isnan(initial_metrics['recall_3k']):
                    all_initial_recall_3k.append(initial_metrics['recall_3k'])
                if initial_metrics.get('recall_4k') is not None and not np.isnan(initial_metrics['recall_4k']):
                    all_initial_recall_4k.append(initial_metrics['recall_4k'])
            
            if best_metrics:
                if best_metrics.get('jsd') is not None and not np.isnan(best_metrics['jsd']):
                    all_best_jsd.append(best_metrics['jsd'])
                if best_metrics.get('wd') is not None and not np.isnan(best_metrics['wd']):
                    all_best_wd.append(best_metrics['wd'])
                if best_metrics.get('recall_3k') is not None and not np.isnan(best_metrics['recall_3k']):
                    all_best_recall_3k.append(best_metrics['recall_3k'])
                if best_metrics.get('recall_4k') is not None and not np.isnan(best_metrics['recall_4k']):
                    all_best_recall_4k.append(best_metrics['recall_4k'])
            
            processed_reviews += 1
        except Exception as e:
            logger.debug(f"Error processing review {delta_key}: {e}")
            continue
    
    # Calculate statistics
    def calculate_stats(values):
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
    
    metrics = {
        'total_reviews': total_reviews,
        'processed_reviews': processed_reviews,
        'threshold_used': threshold,
        'initial_jsd': calculate_stats(all_initial_jsd),
        'initial_wd': calculate_stats(all_initial_wd),
        'initial_recall_3k': calculate_stats(all_initial_recall_3k),
        'initial_recall_4k': calculate_stats(all_initial_recall_4k),
        'best_jsd': calculate_stats(all_best_jsd),
        'best_wd': calculate_stats(all_best_wd),
        'best_recall_3k': calculate_stats(all_best_recall_3k),
        'best_recall_4k': calculate_stats(all_best_recall_4k),
    }
    
    # Calculate improvements
    if metrics['initial_jsd'] and metrics['best_jsd']:
        initial_jsd_mean = metrics['initial_jsd']['mean']
        best_jsd_mean = metrics['best_jsd']['mean']
        if initial_jsd_mean > 0:
            improvement_jsd = ((initial_jsd_mean - best_jsd_mean) / initial_jsd_mean) * 100.0
            metrics['improvement_jsd'] = improvement_jsd
    
    if metrics['initial_wd'] and metrics['best_wd']:
        initial_wd_mean = metrics['initial_wd']['mean']
        best_wd_mean = metrics['best_wd']['mean']
        if initial_wd_mean > 0:
            improvement_wd = ((initial_wd_mean - best_wd_mean) / initial_wd_mean) * 100.0
            metrics['improvement_wd'] = improvement_wd
    
    if metrics['initial_recall_3k'] and metrics['best_recall_3k']:
        initial_recall_3k_mean = metrics['initial_recall_3k']['mean']
        best_recall_3k_mean = metrics['best_recall_3k']['mean']
        if initial_recall_3k_mean > 0:
            improvement_recall_3k = ((best_recall_3k_mean - initial_recall_3k_mean) / initial_recall_3k_mean) * 100.0
            metrics['improvement_recall_3k'] = improvement_recall_3k
    
    if metrics['initial_recall_4k'] and metrics['best_recall_4k']:
        initial_recall_4k_mean = metrics['initial_recall_4k']['mean']
        best_recall_4k_mean = metrics['best_recall_4k']['mean']
        if initial_recall_4k_mean > 0:
            improvement_recall_4k = ((best_recall_4k_mean - initial_recall_4k_mean) / initial_recall_4k_mean) * 100.0
            metrics['improvement_recall_4k'] = improvement_recall_4k
    
    return metrics


def process_journey_file(journey_file: Path, threshold: float = 0.5, results_dir: Optional[Path] = None) -> Dict[str, Any]:
    """Process a journey file and calculate all metrics."""
    logger.info(f"Processing journey file: {journey_file}")
    
    # Extract cluster and micro IDs from journey file path
    cluster_match = re.search(r'cluster_(\d+)', str(journey_file))
    micro_match = re.search(r'micro_(\d+)', str(journey_file))
    cluster_id = f"cluster_{cluster_match.group(1)}" if cluster_match else None
    micro_id = f"micro_{micro_match.group(1)}" if micro_match else None
    
    if not cluster_id or not micro_id:
        logger.error(f"Could not extract cluster_id and micro_id from {journey_file}")
        return {}
    
    # Load delta file (all initial reviews)
    delta_lookup = {}
    if results_dir and cluster_id and micro_id:
        delta_lookup = load_delta_file(results_dir, cluster_id, micro_id)
    
    if not delta_lookup:
        logger.error(f"No delta file found for {cluster_id}/{micro_id}")
        return {}
    
    # Load journey file (reviews that went through SGO)
    journey_lookup = load_journey_file(journey_file)
    
    # Load ground truth file (for recall calculation)
    ground_truth_lookup = {}
    if results_dir and cluster_id and micro_id:
        ground_truth_lookup = load_ground_truth_file(results_dir, cluster_id, micro_id)
    
    # Process all reviews
    metrics = process_all_reviews(
        delta_lookup,
        journey_lookup,
        ground_truth_lookup,
        threshold=threshold
    )
    
    return metrics


def main():
    parser = argparse.ArgumentParser(
        description='Analyze journey files to calculate metrics for initial and best predictions',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('--journey-file', type=str, required=True,
                       help='Path to journey file (e.g., cluster_5/micro_5_journey.json)')
    parser.add_argument('--results-dir', type=str, default=None,
                       help='Path to results directory containing _journey folder')
    parser.add_argument('--threshold', type=float, default=0.5,
                       help='Probability threshold for ground truth themes (default: 0.5)')
    parser.add_argument('--output', type=str, default=None,
                       help='Output JSON file path (default: print to stdout)')
    
    args = parser.parse_args()
    
    # Determine journey file path
    if args.results_dir:
        results_dir = Path(args.results_dir)
    else:
        results_dir = Path(__file__).parent.parent / "artifacts" / "sgo_training_results_v3_sample"
    
    journey_file = results_dir / "_journey" / args.journey_file
    if not journey_file.exists():
        # Try as absolute path
        journey_file = Path(args.journey_file)
        if not journey_file.exists():
            logger.error(f"Journey file not found: {journey_file}")
            return 1
    
    threshold = args.threshold
    
    logger.info("=" * 80)
    logger.info("ANALYZING JOURNEY FILE FOR METRICS")
    logger.info("=" * 80)
    logger.info(f"Journey file: {journey_file}")
    logger.info(f"Threshold: {threshold}")
    
    # Process journey file
    metrics = process_journey_file(journey_file, threshold=threshold, results_dir=results_dir)
    
    if not metrics:
        logger.error("Failed to process journey file")
        return 1
    
    # Output results
    if args.output:
        output_file = Path(args.output)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)
        logger.info(f"\nResults saved to: {output_file}")
    else:
        print(json.dumps(metrics, indent=2, ensure_ascii=False))
    
    # Print summary
    logger.info("\n" + "=" * 80)
    logger.info("SUMMARY")
    logger.info("=" * 80)
    logger.info(f"Total reviews: {metrics['total_reviews']}")
    logger.info(f"Processed reviews: {metrics['processed_reviews']}")
    
    logger.info(f"\nInitial (Pre-SGO) Metrics:")
    if metrics['initial_jsd']:
        logger.info(f"  JSD: mean={metrics['initial_jsd']['mean']:.6f}, count={metrics['initial_jsd']['count']}")
    if metrics['initial_wd']:
        logger.info(f"  WD: mean={metrics['initial_wd']['mean']:.6f}, count={metrics['initial_wd']['count']}")
    if metrics['initial_recall_3k']:
        logger.info(f"  Recall@max(3,k): mean={metrics['initial_recall_3k']['mean']:.6f}, count={metrics['initial_recall_3k']['count']}")
    if metrics['initial_recall_4k']:
        logger.info(f"  Recall@max(4,k): mean={metrics['initial_recall_4k']['mean']:.6f}, count={metrics['initial_recall_4k']['count']}")
    
    logger.info(f"\nBest (Post-SGO) Metrics:")
    if metrics['best_jsd']:
        logger.info(f"  JSD: mean={metrics['best_jsd']['mean']:.6f}, count={metrics['best_jsd']['count']}")
    if metrics['best_wd']:
        logger.info(f"  WD: mean={metrics['best_wd']['mean']:.6f}, count={metrics['best_wd']['count']}")
    if metrics['best_recall_3k']:
        logger.info(f"  Recall@max(3,k): mean={metrics['best_recall_3k']['mean']:.6f}, count={metrics['best_recall_3k']['count']}")
    if metrics['best_recall_4k']:
        logger.info(f"  Recall@max(4,k): mean={metrics['best_recall_4k']['mean']:.6f}, count={metrics['best_recall_4k']['count']}")
    
    logger.info(f"\nImprovements:")
    if 'improvement_jsd' in metrics:
        logger.info(f"  JSD: {metrics['improvement_jsd']:.2f}%")
    if 'improvement_wd' in metrics:
        logger.info(f"  WD: {metrics['improvement_wd']:.2f}%")
    if 'improvement_recall_3k' in metrics:
        logger.info(f"  Recall@max(3,k): {metrics['improvement_recall_3k']:.2f}%")
    if 'improvement_recall_4k' in metrics:
        logger.info(f"  Recall@max(4,k): {metrics['improvement_recall_4k']:.2f}%")
    
    logger.info("=" * 80)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
