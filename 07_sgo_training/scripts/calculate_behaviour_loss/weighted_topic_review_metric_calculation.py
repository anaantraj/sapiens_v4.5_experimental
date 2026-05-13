from scipy.spatial.distance import cosine
from scipy.stats import entropy
import numpy as np

EPSILON = 1e-10

def calculate_jsd(predicted_dist: dict, actual_dist: dict, all_themes: list = None) -> float:
    """
    Calculate Jensen-Shannon Divergence between two theme probability distributions.
    
    JSD(P||Q) = 0.5 * KL(P||M) + 0.5 * KL(Q||M)
    where M = 0.5 * (P + Q)
    
    Args:
        predicted_dist: dict with theme names as keys and probabilities as values
        actual_dist: dict with theme names as keys and probabilities as values, OR list of theme names
        all_themes: Optional list of all possible themes (if None, uses union of both distributions)
    
    Returns:
        JSD value (float, range 0-1)
    """
    if not actual_dist:
        return 0.0
    if not predicted_dist or not isinstance(predicted_dist, dict):
        return 1.0
    
    # Handle actual_dist: can be dict (distribution) or list
    if isinstance(actual_dist, dict):
        actual_dict = actual_dist
    else:
        # Convert list to uniform distribution
        actual_dict = {str(t).strip(): 1.0 / len(actual_dist) if actual_dist else 0.0 for t in actual_dist}
    
    # Get all unique themes
    # If all_themes is None or empty list, use union of keys from both distributions
    # This ensures JSD can be calculated even when category lookup fails
    if all_themes is None or (isinstance(all_themes, list) and len(all_themes) == 0):
        all_themes = list(set(list(predicted_dist.keys()) + list(actual_dict.keys())))
    
    # After trying to get themes, if still empty, return 0.0
    # (This would only happen if both distributions are empty)
    if not all_themes:
        return 0.0
    
    # Create probability vectors
    P = np.array([predicted_dist.get(theme, 0.0) for theme in all_themes])
    Q = np.array([actual_dict.get(theme, 0.0) for theme in all_themes])
    
    # Normalize to ensure they sum to 1
    P_sum = P.sum()
    Q_sum = Q.sum()
    if P_sum > 0:
        P = P / P_sum
    else:
        P = np.ones_like(P) / len(P)  # Uniform if empty
    
    if Q_sum > 0:
        Q = Q / Q_sum
    else:
        Q = np.ones_like(Q) / len(Q)  # Uniform if empty
    
    # Add epsilon to avoid zeros
    P = P + EPSILON
    Q = Q + EPSILON
    
    # Re-normalize after adding epsilon
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

def calculate_text_delta(embedding1, embedding2):
    return cosine(embedding1, embedding2)

def calculate_theme_delta(predicted_themes, actual_themes):
    if not actual_themes: return 0.0
    if not predicted_themes or not isinstance(predicted_themes, dict): return 1.0
    
    cleaned_actual = [str(t).strip().lower() for t in actual_themes]
    k = len(cleaned_actual)
    n = max(3, k)
    
    # Sort predicted by score
    top_predicted = sorted(predicted_themes.keys(), key=lambda x: predicted_themes[x], reverse=True)[:n]
    cleaned_predicted = [str(t).strip().lower() for t in top_predicted]
    
    matches = sum(1 for t in cleaned_actual if t in cleaned_predicted)
    recall = matches / k if k > 0 else 0
    return 1.0 - recall

def calculate_theme_delta_logprobs(predicted_themes, actual_themes, all_themes=None):
    """
    Calculate theme delta for logprobs mode using Jensen-Shannon Divergence (JSD).
    Both predicted and actual are probability distributions.
    
    Args:
        predicted_themes: dict with theme names as keys and probabilities as values
        actual_themes: dict with theme names as keys and probabilities as values, OR list of theme names
        all_themes: Optional list of all possible themes (if None, uses union of both distributions)
    
    Returns:
        float: JSD value (range 0-1, where 0 = identical distributions, 1 = maximum divergence)
    """
    return calculate_jsd(predicted_themes, actual_themes, all_themes)

def calculate_review_metrics(prediction, actual, all_themes=None):
    """
    Calculate metrics for a single review.
    For logprobs mode: uses JSD for theme metrics.
    For confidence mode: uses rating, sentiment, and text delta only.
    """
    if not isinstance(prediction, dict) or not isinstance(actual, dict):
        return None

    # Initialize metrics dictionary
    metrics = {}

    # --- Rating and Sentiment Scores ---
    actual_rating = actual.get('rating', 3.0)
    predicted_rating = prediction.get('rating')
    rating_score = 0.0
    if predicted_rating is not None:
        try:
            rating_diff = abs(float(predicted_rating) - float(actual_rating))
            rating_score = max(0.0, 1.0 - (rating_diff / 4.0))
        except (ValueError, TypeError):
            rating_score = 0.0
    
    predicted_sentiment = str(prediction.get('sentiment', "")).strip().lower()
    actual_sentiment = str(actual.get('sentiment', "")).strip().lower()
    sentiment_score = 1.0 if predicted_sentiment and predicted_sentiment == actual_sentiment else 0.0

    metrics['rating_score'] = rating_score
    metrics['sentiment_score'] = sentiment_score

    # --- Theme JSD (for logprobs mode) ---
    predicted_themes = prediction.get('predicted_themes', {})
    actual_themes = actual.get('predicted_themes', [])
    
    # Calculate JSD for theme distributions
    theme_jsd = calculate_jsd(predicted_themes, actual_themes, all_themes)
    metrics['theme_jsd'] = theme_jsd
    
    # For backward compatibility, also include num_actual_themes
    if isinstance(actual_themes, list):
        metrics['num_actual_themes'] = len(actual_themes)
    elif isinstance(actual_themes, dict):
        metrics['num_actual_themes'] = len([t for t, p in actual_themes.items() if p > 0])
    else:
        metrics['num_actual_themes'] = 0

    # --- Overall Accuracy (using JSD for theme component) ---
    # Lower JSD is better, so convert to score: 1 - JSD (clamped to 0-1)
    theme_score = max(0.0, 1.0 - theme_jsd)
    WEIGHTS = {'rating': 0.4, 'sentiment': 0.3, 'theme': 0.3}
    overall_accuracy = (rating_score * WEIGHTS['rating']) + \
                       (sentiment_score * WEIGHTS['sentiment']) + \
                       (theme_score * WEIGHTS['theme'])

    metrics['overall_accuracy'] = overall_accuracy
    metrics['weights_used'] = WEIGHTS

    return metrics
def recalculate_all_review_metrics(data, all_themes_map=None, category_mapping=None):
    """
    Recalculate metrics for all reviews.
    
    Args:
        data: Data dictionary with user_predictions
        all_themes_map: Optional dict mapping category to list of themes (for JSD calculation)
    """
    if not data or 'user_predictions' not in data: return data
    for user_reviews in data['user_predictions'].values():
        if not isinstance(user_reviews, list): continue
        for review_entry in user_reviews:
            prediction = review_entry.get('prediction', {})
            actual = review_entry.get('actual', {})
            
            # Get category themes if available
            # Map category to main category first (same logic as initial delta calculation)
            all_themes = None
            if all_themes_map:
                category = review_entry.get('category', '')
                if category:
                    # Map to main category first (same as initial delta calculation)
                    main_category = category
                    if category_mapping and category:
                        if isinstance(category_mapping, dict) and 'category_to_main_mapping' in category_mapping:
                            main_category = category_mapping['category_to_main_mapping'].get(category, category)
                        else:
                            main_category = category_mapping.get(category, category)
                    
                    # Try direct lookup with main category
                    all_themes = all_themes_map.get(main_category, [])
                    
                    # If not found, try normalized category name
                    if not all_themes:
                        cat_normalized = main_category.replace(' & ', '_and_').replace('&', 'and').replace(' ', '_').replace('-', '_').lower()
                        all_themes = all_themes_map.get(cat_normalized, [])
                    
                    # If still not found, try case-insensitive search
                    if not all_themes:
                        category_lower = main_category.lower()
                        for key in all_themes_map.keys():
                            if key.lower() == category_lower:
                                all_themes = all_themes_map[key]
                                break
            
            metrics = calculate_review_metrics(prediction, actual, all_themes=all_themes)
            if metrics: review_entry['metrics'] = metrics
            elif 'metrics' in review_entry: review_entry.pop('metrics', None)
    return data


def calculate_quantitative_summary_from_data(data):
    """
    Calculate quantitative summary (ratings, sentiment) from prediction data.
    """
    if not data or 'user_predictions' not in data:
        return {
            "average_rating": "0.0",
            "sentiment_distribution_percent": {
                "Positive": 0.0,
                "Neutral": 0.0,
                "Negative": 0.0
            }
        }
    
    ratings = []
    sentiments = {"Positive": 0, "Neutral": 0, "Negative": 0}
    total_reviews = 0
    
    for user_id, user_reviews in data['user_predictions'].items():
        if isinstance(user_reviews, list):
            for review in user_reviews:
                actual = review.get('actual', {})
                if 'rating' in actual:
                    try:
                        ratings.append(float(actual['rating']))
                    except (ValueError, TypeError):
                        pass
                if 'sentiment' in actual:
                    sent = actual['sentiment']
                    if sent in sentiments:
                        sentiments[sent] += 1
                    total_reviews += 1
    
    avg_rating = np.mean(ratings) if ratings else 0.0
    sentiment_dist = {}
    if total_reviews > 0:
        for key, count in sentiments.items():
            sentiment_dist[key] = round((count / total_reviews) * 100, 1)
    else:
        sentiment_dist = {"Positive": 0.0, "Neutral": 0.0, "Negative": 0.0}
    
    return {
        "average_rating": f"{avg_rating:.2f}",
        "sentiment_distribution_percent": sentiment_dist
    }
    
def calculate_and_append_final_metrics(data, all_themes_map=None, category_mapping=None):
    """
    Calculates final aggregate metrics using JSD for theme distributions.
    
    Args:
        data: Data dictionary with user_predictions
        all_themes_map: Optional dict mapping category to list of themes (for JSD calculation)
        category_mapping: Optional dict mapping sub-category to main category (for proper theme lookup)
    """
    print("\nCalculating final aggregate metrics...")
    
    # Initialize metric collectors (removed all recall metrics, using JSD instead)
    all_metrics = {
        'rating_score': [],
        'sentiment_score': [],
        'theme_jsd': [],
        'overall_accuracy': [],
        'num_actual_themes': []
    }

    if 'user_predictions' not in data or not isinstance(data['user_predictions'], dict):
        print("Warning: 'user_predictions' not found. Cannot calculate final metrics.")
        return data

    for user_id, user_data in data['user_predictions'].items():
        if not isinstance(user_data, list) or len(user_data) == 0:
            continue
        
        # Process ALL reviews for this user
        for review_data in user_data:
            prediction = review_data.get('prediction', {})
            actual = review_data.get('actual', {})
            
            # Get category themes if available
            # Map category to main category first (same logic as initial delta calculation)
            all_themes = None
            if all_themes_map:
                category = review_data.get('category', '')
                if category:
                    # Map to main category first (same as initial delta calculation)
                    main_category = category
                    if category_mapping and category:
                        if isinstance(category_mapping, dict) and 'category_to_main_mapping' in category_mapping:
                            main_category = category_mapping['category_to_main_mapping'].get(category, category)
                        else:
                            main_category = category_mapping.get(category, category)
                    
                    # Try direct lookup with main category
                    all_themes = all_themes_map.get(main_category, [])
                    cat_normalized = None
                    
                    # If not found, try normalized category name
                    # Normalize: "Health & Personal Care" -> "health_and_personal_care"
                    if not all_themes:
                        cat_normalized = main_category.replace(' & ', '_and_').replace('&', 'and').replace(' ', '_').replace('-', '_').lower()
                        all_themes = all_themes_map.get(cat_normalized, [])
                    
                    # If still not found, try case-insensitive search
                    if not all_themes:
                        category_lower = main_category.lower()
                        for key in all_themes_map.keys():
                            if key.lower() == category_lower:
                                all_themes = all_themes_map[key]
                                break
                    
                    # Debug: Log if themes not found (only log first few to avoid spam)
                    if not all_themes and len(all_metrics['theme_jsd']) < 3:
                        print(f"  WARNING: No themes found for category '{category}' -> main '{main_category}' (tried: {main_category}, normalized: {cat_normalized or 'N/A'})")
            
            # Use the metrics calculation (now uses JSD for themes)
            metrics_result = calculate_review_metrics(prediction, actual, all_themes=all_themes)
            
            if metrics_result:
                # Collect all metrics
                for metric_name, metric_value in metrics_result.items():
                    if metric_name != 'weights_used' and metric_name in all_metrics:
                        all_metrics[metric_name].append(metric_value)

    # Check if we have any data
    if not all_metrics['overall_accuracy']:
        print("No review data to calculate metrics from.")
        return data

    total_reviews = len(all_metrics['overall_accuracy'])
    print(f"Calculated metrics for {total_reviews} reviews (all reviews included: changed and unchanged).")
    
    # Build final analysis with all metrics
    final_analysis = {
        "aggregate_scores": all_metrics,
        "final_metrics": {}
    }
    
    # Calculate mean, std, count for each metric
    for metric_name, scores in all_metrics.items():
        if scores:
            final_analysis['final_metrics'][metric_name] = {
                'mean': float(np.mean(scores)),
                'std': float(np.std(scores)),
                'count': len(scores)
            }
    
    data.update(final_analysis)
    
    # Print summary to console
    if 'overall_accuracy' in final_analysis['final_metrics']:
        print(f"Final metrics calculated and added to the data. Total reviews processed: {total_reviews}")
        print(f"  Overall Accuracy: {final_analysis['final_metrics']['overall_accuracy']['mean']:.4f}")
        if 'theme_jsd' in final_analysis['final_metrics']:
            print(f"  Theme JSD (mean): {final_analysis['final_metrics']['theme_jsd']['mean']:.4f}")
        
    return data