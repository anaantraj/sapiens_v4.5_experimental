import numpy as np
from collections import defaultdict
from typing import List

def aggregate_metrics(metrics_list: List[dict], include_values: bool = False) -> dict:
    """
    Aggregate metrics from multiple reviews into summary statistics.
    
    Args:
        metrics_list: List of metric dictionaries from individual reviews
        include_values: If True, include raw values array (for micro-cluster level)
    
    Returns:
        Dict with aggregated metrics (mean, std, min, max, etc.) for each metric type
    """
    if not metrics_list:
        return {}
    
    # Collect all metric values
    metric_values = defaultdict(list)
    for metrics in metrics_list:
        for key, value in metrics.items():
            if key not in ['weights_used'] and isinstance(value, (int, float)):
                metric_values[key].append(value)
    
    # Calculate summary statistics for each metric
    aggregated = {}
    for metric_name, values in metric_values.items():
        if values:
            result = {
                'mean': float(np.mean(values)),
                'std': float(np.std(values)),
                'count': len(values)
            }
            if include_values:
                # Include detailed stats for micro-cluster level
                result.update({
                    'min': float(np.min(values)),
                    'max': float(np.max(values)),
                    'median': float(np.median(values)),
                    'values': values
                })
            aggregated[metric_name] = result
    
    return aggregated

def calculate_review_metrics(prediction: dict, actual: dict) -> dict:
    """
    Calculate metrics for a single review comparing prediction vs actual.
    
    Args:
        prediction: Dict with 'rating', 'sentiment', 'predicted_themes' (dict of theme: score)
        actual: Dict with 'rating', 'sentiment', 'predicted_themes' (list of theme names)
    
    Returns:
        Dict with all metrics
    """
    # Get actual themes (list)
    actual_themes_list = actual.get('predicted_themes', [])
    if not isinstance(actual_themes_list, list):
        actual_themes_list = []
    actual_themes_set = set(actual_themes_list)
    num_actual_themes = len(actual_themes_set)
    
    if num_actual_themes == 0:
        return None
    
    # Get predicted themes (dict of theme: score) and sort by score
    predicted_themes_dict = prediction.get('predicted_themes', {})
    if not isinstance(predicted_themes_dict, dict):
        predicted_themes_dict = {}
    
    sorted_predicted = sorted(
        predicted_themes_dict.items(),
        key=lambda x: x[1],
        reverse=True
    )
    predicted_themes_list = [theme for theme, score in sorted_predicted]
    
    # Rating score
    actual_rating = float(actual.get('rating', 3.0))
    predicted_rating = float(prediction.get('rating', 3.0))
    rating_diff = abs(predicted_rating - actual_rating)
    rating_score = max(0.0, 1.0 - (rating_diff / 4.0))
    
    # Sentiment score
    sentiment_score = 1.0 if prediction.get('sentiment') == actual.get('sentiment') else 0.0
    
    # Recall@1
    top_1 = set(predicted_themes_list[:1]) if predicted_themes_list else set()
    recall_at_1 = len(top_1 & actual_themes_set) / num_actual_themes if num_actual_themes > 0 else 0.0
    
    # Recall@3
    top_3 = set(predicted_themes_list[:3]) if len(predicted_themes_list) >= 3 else set(predicted_themes_list)
    recall_at_3 = len(top_3 & actual_themes_set) / num_actual_themes if num_actual_themes > 0 else 0.0
    
    # Recall@5
    top_5 = set(predicted_themes_list[:5]) if len(predicted_themes_list) >= 5 else set(predicted_themes_list)
    recall_at_5 = len(top_5 & actual_themes_set) / num_actual_themes if num_actual_themes > 0 else 0.0
    
    # Recall@k (where k = num_actual_themes)
    k = num_actual_themes
    top_k = set(predicted_themes_list[:k]) if len(predicted_themes_list) >= k else set(predicted_themes_list)
    recall_at_k = len(top_k & actual_themes_set) / num_actual_themes if num_actual_themes > 0 else 0.0
    
    # Recall@max(3,k) - Baseline
    k_baseline = max(3, k)
    top_k_baseline = set(predicted_themes_list[:k_baseline]) if len(predicted_themes_list) >= k_baseline else set(predicted_themes_list)
    recall_at_max_3k = len(top_k_baseline & actual_themes_set) / num_actual_themes if num_actual_themes > 0 else 0.0
    
    # Recall@max(3,k) with 0.8 threshold
    top_k_08 = set(predicted_themes_list[:k_baseline]) if len(predicted_themes_list) >= k_baseline else set(predicted_themes_list)
    additional_themes_08 = set()
    num_themes_above_08 = 0
    for idx, (theme, score) in enumerate(sorted_predicted):
        if score >= 0.8:
            num_themes_above_08 += 1
            if idx >= k_baseline:
                additional_themes_08.add(theme)
    expanded_themes_08 = top_k_08.union(additional_themes_08)
    recall_at_max_3k_08 = len(expanded_themes_08 & actual_themes_set) / num_actual_themes if num_actual_themes > 0 else 0.0
    
    # Recall@max(3,k) with 0.85 threshold
    top_k_085 = set(predicted_themes_list[:k_baseline]) if len(predicted_themes_list) >= k_baseline else set(predicted_themes_list)
    additional_themes_085 = set()
    num_themes_above_085 = 0
    for idx, (theme, score) in enumerate(sorted_predicted):
        if score >= 0.85:
            num_themes_above_085 += 1
            if idx >= k_baseline:
                additional_themes_085.add(theme)
    expanded_themes_085 = top_k_085.union(additional_themes_085)
    recall_at_max_3k_085 = len(expanded_themes_085 & actual_themes_set) / num_actual_themes if num_actual_themes > 0 else 0.0
    
    # Overall accuracy (weighted combination)
    WEIGHTS = {'rating': 0.4, 'sentiment': 0.3, 'theme_recall': 0.3}
    theme_score_for_overall = recall_at_max_3k
    overall_accuracy = (rating_score * WEIGHTS['rating']) + (sentiment_score * WEIGHTS['sentiment']) + (theme_score_for_overall * WEIGHTS['theme_recall'])
    
    return {
        'rating_score': rating_score,
        'sentiment_score': sentiment_score,
        'recall@1': recall_at_1,
        'recall@3': recall_at_3,
        'recall@5': recall_at_5,
        'recall@k': recall_at_k,
        'recall@max(3,k)': recall_at_max_3k,
        'recall@max(3,k)_threshold_0.8': recall_at_max_3k_08,
        'num_themes_above_0.8': num_themes_above_08,
        'num_additional_themes_0.8': len(additional_themes_08),
        'recall@max(3,k)_threshold_0.85': recall_at_max_3k_085,
        'num_themes_above_0.85': num_themes_above_085,
        'num_additional_themes_0.85': len(additional_themes_085),
        'overall_accuracy': overall_accuracy,
        'weights_used': WEIGHTS,
        'num_actual_themes': num_actual_themes
    }
