from .prompt_generation import create_enhanced_prompt
from .llm_client import get_llm_prediction
from .metrics import calculate_review_metrics

def process_single_review(args_tuple):
    """
    Process a single review prediction. Designed for parallel execution.
    
    Args:
        args_tuple: (actual_review, persona_context, user_char_summary, category_char_summary,
                    main_category, category_themes, client, rate_limiter, prompt_template, model_name, max_tokens, temperature, max_retries, scoring_mode, theme_prediction_model)
    
    Returns:
        dict with prediction, actual, and metrics data, or None if error
    """
    (actual_review, persona_context, user_char_summary, category_char_summary,
     main_category, category_themes, client, rate_limiter, prompt_template, model_name, max_tokens, temperature, max_retries, scoring_mode, theme_prediction_model) = args_tuple
    
    prompt = create_enhanced_prompt(
        persona_context=persona_context,
        user_char_summary=user_char_summary,
        category_char_summary=category_char_summary,
        product_description=actual_review.get('product_description', ''),
        category=main_category,
        category_themes=category_themes,
        prompt_template=prompt_template,
        scoring_mode=scoring_mode
    )
    
    product_description = actual_review.get('product_description', '')
    prediction = get_llm_prediction(prompt, client, rate_limiter, category_themes, model_name, max_tokens, temperature, max_retries, scoring_mode, theme_prediction_model, product_description)
    
    # Check if LLM call failed (returns None)
    if prediction is None:
        # Return failed review data for separate saving
        return {
            'status': 'failed',
            'error': 'LLM call failed after all retries',
            'product_description': actual_review.get('product_description', ''),
            'category': actual_review.get('category'),
            'asin': actual_review.get('asin'),
            'timestamp': actual_review.get('timestamp'),
            'actual': {
                'review_text': actual_review.get('review_text', ''),
                'rating': actual_review.get('rating'),
                'sentiment': actual_review.get('sentiment'),
                'predicted_themes': actual_review.get('themes', actual_review.get('predicted_themes', []))
            },
            # Include context for reprocessing
            'persona_context': persona_context,
            'user_char_summary': user_char_summary,
            'category_char_summary': category_char_summary,
            'main_category': main_category,
            'category_themes': category_themes
        }
    
    # Calculate metrics for successful prediction
    metrics = calculate_review_metrics(prediction, {
        'rating': actual_review.get('rating'),
        'sentiment': actual_review.get('sentiment'),
        'predicted_themes': actual_review.get('themes', actual_review.get('predicted_themes', []))
    })
    
    # Return format expected by SGO training: each review has prediction, actual, and metrics
    result = {
        'status': 'success',
        'product_description': actual_review.get('product_description', ''),
        'category': actual_review.get('category'),
        'asin': actual_review.get('asin'),
        'timestamp': actual_review.get('timestamp'),
        'prediction': prediction,  # Contains: review_text, rating, sentiment, predicted_themes
        'actual': {
            'review_text': actual_review.get('review_text', ''),
            'rating': actual_review.get('rating'),
            'sentiment': actual_review.get('sentiment'),
            'predicted_themes': actual_review.get('themes', actual_review.get('predicted_themes', []))
        }
    }
    
    # Always include metrics field (required by schema)
    # If metrics calculation failed (None), create default/empty metrics
    if metrics:
        result['metrics'] = metrics
    else:
        # Create default metrics when calculation returns None (e.g., no actual themes)
        # Use minimal default values that satisfy the schema
        # IMPORTANT: Include all required fields from InitialPredictionsReviewMetrics schema
        result['metrics'] = {
            'rating_score': 0.0,
            'sentiment_score': 0.0,
            'recall@1': 0.0,
            'recall@3': 0.0,
            'recall@5': 0.0,
            'recall@k': 0.0,
            'recall@max(3,k)': 0.0,
            'recall@max(3,k)_threshold_0.8': 0.0,
            'num_themes_above_0.8': 0,
            'num_additional_themes_0.8': 0,
            'recall@max(3,k)_threshold_0.85': 0.0,
            'num_themes_above_0.85': 0,
            'num_additional_themes_0.85': 0,  # Fixed: was missing underscore
            'overall_accuracy': 0.0,
            'weights_used': {'rating': 0.4, 'sentiment': 0.3, 'theme_recall': 0.3}
            # Note: num_actual_themes is NOT included for logprobs mode (removed from default metrics)
        }
    
    return result
