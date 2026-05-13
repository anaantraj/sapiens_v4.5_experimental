"""
Utility functions for managing journey log entries.
Eliminates code duplication between batch_processor.py and cluster_processor.py.
"""
import copy


def create_journey_log_entry(
    review_key,
    user_id,
    review_idx,
    review_entry,
    initial_deltas,
    initial_prediction_data=None
):
    """
    Creates a standardized journey log entry for a review.
    
    Args:
        review_key: Unique identifier for the review (e.g., "user123_review_0")
        user_id: User ID
        review_idx: Review index
        review_entry: Full review entry dict (must have 'actual' and optionally 'prediction', 'product_description', 'category')
        initial_deltas: Dict with 'text_delta', 'theme_delta', 'overall_delta'
        initial_prediction_data: Optional initial prediction data (if None, uses review_entry['prediction'])
    
    Returns:
        Dict: Journey log entry structure
    """
    # Extract product description and category
    product_description = review_entry.get('product_description', 'N/A')
    review_category = review_entry.get('category', 'N/A')
    
    # Use provided initial_prediction_data or extract from review_entry
    if initial_prediction_data is None:
        initial_prediction_data = copy.deepcopy(review_entry.get('prediction', {}))
    else:
        initial_prediction_data = copy.deepcopy(initial_prediction_data)
    
    # Build journey log entry
    journey_entry = {
        "user_id": user_id,
        "review_idx": review_idx,
        "product_description": product_description,
        "category": review_category,
        "actual_review_data": {
            **review_entry.get('actual', {}),
            "product_description": product_description,
            "category": review_category
        },
        "initial_prediction_deltas": {
            "text_delta": initial_deltas.get('text', 0.0),
            "theme_delta": initial_deltas.get('theme', 0.0),
            "overall_delta": initial_deltas.get('overall', 0.0)
        },
        "initial_prediction_data": initial_prediction_data,
        "correction_journey": []
    }
    
    return journey_entry


def ensure_journey_log_entry(
    journey_log,
    review_key,
    user_id,
    review_idx,
    review_entry,
    initial_deltas,
    initial_prediction_data=None,
    lock=None
):
    """
    Ensures a journey log entry exists for a review. Creates it if it doesn't exist.
    Thread-safe if lock is provided.
    
    Args:
        journey_log: The journey log dict to update
        review_key: Unique identifier for the review
        user_id: User ID
        review_idx: Review index
        review_entry: Full review entry dict
        initial_deltas: Dict with delta values
        initial_prediction_data: Optional initial prediction data
        lock: Optional lock for thread-safe access
    
    Returns:
        bool: True if entry was created, False if it already existed
    """
    if lock:
        with lock:
            if review_key not in journey_log:
                journey_log[review_key] = create_journey_log_entry(
                    review_key, user_id, review_idx, review_entry,
                    initial_deltas, initial_prediction_data
                )
                return True
            return False
    else:
        if review_key not in journey_log:
            journey_log[review_key] = create_journey_log_entry(
                review_key, user_id, review_idx, review_entry,
                initial_deltas, initial_prediction_data
            )
            return True
        return False





