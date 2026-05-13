import os
import copy
import re
import time
import numpy as np
from threading import Lock
from concurrent.futures import ThreadPoolExecutor, as_completed
import sys
from pathlib import Path

# Set up package structure for sgo_training imports
scripts_dir = Path(__file__).parent.parent
if str(scripts_dir) not in sys.path:
    sys.path.insert(0, str(scripts_dir))
import _package_setup

from sgo_training.config import settings
from sgo_training.utils import io_utils, logging_utils, journey_utils
# Import batch_processor directly from local package to avoid placeholder issues
from feedback_loop.batch_correction_pipeline import process_single_batch
from sgo_training.utils import embeddings
from calculate_behaviour_loss.weighted_topic_review_metric_calculation import (
    calculate_text_delta, 
    calculate_theme_delta,
    calculate_theme_delta_logprobs,
    recalculate_all_review_metrics,
    calculate_quantitative_summary_from_data,
    calculate_and_append_final_metrics
)
file_lock = Lock()

def get_theme_delta(predicted_themes, actual_themes, all_themes=None):
    """
    Helper function that selects the appropriate theme delta calculation based on scoring mode.
    For logprobs mode: uses calculate_theme_delta_logprobs (JSD based)
    For confidence mode: uses calculate_theme_delta (original implementation)
    
    Args:
        predicted_themes: dict with theme names and probabilities
        actual_themes: dict or list of actual themes
        all_themes: Optional list of all possible themes (for JSD calculation in logprobs mode)
    """
    scoring_mode = getattr(settings, 'SCORING_MODE', 'confidence')
    if scoring_mode in ["logprobs", "logprobs-without-persona-context"]:
        return calculate_theme_delta_logprobs(predicted_themes, actual_themes, all_themes)
    else:
        return calculate_theme_delta(predicted_themes, actual_themes)

def select_best_prediction_by_theme_delta(current_data, journey_log, original_data):
    """
    After all iterations, this function selects the *best* prediction for each review
    based on the lowest recorded overall_delta in the journey_log.
    It preserves original predictions for reviews not in journey_log.
    It ensures we always keep the best (lowest overall_delta) prediction.
    """
    if not journey_log:
        print(" - No journey log found. Using original predictions.")
        return copy.deepcopy(original_data) # Return original data as-is
    
    # Start with original data to preserve reviews not in journey_log
    best_data = copy.deepcopy(original_data)
    processed_reviews = 0
    improved_reviews = 0
    kept_original_reviews = 0
    worse_corrections = 0

    for review_key, journey in journey_log.items():
        # Parse review_key to get user_id and review_idx
        if "_review_" in review_key:
            user_id, review_idx_str = review_key.rsplit("_review_", 1)
            try:
                review_idx = int(review_idx_str)
            except ValueError:
                print(f" - WARNING: Invalid review key format {review_key}. Skipping.")
                continue
        else:
            # Backward compatibility: treat as user_id, use first review
            user_id = review_key
            review_idx = 0
        
        if user_id not in best_data['user_predictions']:
            continue
        
        user_reviews = best_data['user_predictions'][user_id]
        if not isinstance(user_reviews, list) or review_idx >= len(user_reviews):
            print(f" - WARNING: review_idx {review_idx} out of range for user {user_id}. Skipping.")
            continue

        # 1. Get the initial prediction as the starting "best"
        initial_deltas = journey.get('initial_prediction_deltas', {})
        initial_data = journey.get('initial_prediction_data', {})
        
        if 'overall_delta' not in initial_deltas or not initial_data:
             # print(f" - WARNING: Skipping {review_key}, missing initial prediction data in journey.")
             continue
             
        # Use overall_delta as the primary metric (combines text and theme)
        best_overall_delta = initial_deltas.get('overall_delta', 1.0)
        best_theme_delta = initial_deltas.get('theme_delta', 1.0)
        best_prediction = copy.deepcopy(initial_data)
        
        # 2. Check all corrected predictions in the journey
        for correction_attempt in journey.get('correction_journey', []):
            current_overall_delta = correction_attempt.get('new_overall_delta')
            current_theme_delta = correction_attempt.get('new_theme_delta')
            current_pred_data = correction_attempt.get('corrected_prediction_data')
            
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
                # Found a new best!
                best_overall_delta = current_overall_delta if current_overall_delta is not None else best_overall_delta
                best_theme_delta = current_theme_delta if current_theme_delta is not None else best_theme_delta
                best_prediction = copy.deepcopy(current_pred_data)
        
        # 3. Update the main data object
        original_prediction = user_reviews[review_idx].get('prediction', {})
        initial_overall = initial_deltas.get('overall_delta', 1.0)
        initial_theme = initial_deltas.get('theme_delta', 1.0)
        
        # Add metadata for transparency
        user_reviews[review_idx]['correction_journey'] = journey.get('correction_journey', [])
        user_reviews[review_idx]['initial_deltas'] = initial_deltas
        user_reviews[review_idx]['best_deltas'] = {
            'overall_delta': best_overall_delta,
            'theme_delta': best_theme_delta
        }
        if original_prediction != best_prediction:
            # Check if we actually improved (theme improved OR overall improved with same/better theme)
            theme_improved = best_theme_delta < initial_theme
            theme_same_but_overall_improved = (
                abs(best_theme_delta - initial_theme) < 0.001 and  # Essentially same theme_delta
                best_overall_delta < initial_overall  # But overall improved
            )
            
            if theme_improved or theme_same_but_overall_improved:
                # Update with improved prediction
                user_reviews[review_idx]['prediction'] = best_prediction
                improved_reviews += 1
                if theme_improved:
                    print(f"  - IMPROVED {review_key}: Theme delta {initial_theme:.4f} -> {best_theme_delta:.4f} (Overall: {initial_overall:.4f} -> {best_overall_delta:.4f})")
                else:
                    print(f"  - IMPROVED {review_key}: Overall delta {initial_overall:.4f} -> {best_overall_delta:.4f} (Theme delta same: {best_theme_delta:.4f})")
            else:
                # Correction didn't improve, keep original
                worse_corrections += 1
                print(f"  - REJECTED {review_key}: No improvement (Theme: {initial_theme:.4f} -> {best_theme_delta:.4f}, Overall: {initial_overall:.4f} -> {best_overall_delta:.4f}), keeping original")
        else:
            kept_original_reviews += 1
        processed_reviews += 1

    print(f" - Selected best predictions for {processed_reviews} reviews: {improved_reviews} improved, {kept_original_reviews} kept original, {worse_corrections} corrections rejected (worse than original).")
    return best_data

        
        # Determine if we updated or kept original
        # Note: We compare prediction dicts, but simple equality might fail due to float precision
        # better to rely on the deltas we calculated.
        
def process_micro_cluster_file(filepath, client, user_chars_data, category_mapping):
    """
    Wrapper function that handles path parsing and logging context setup.
    """
    try:
        # Robust path parsing
        relative_path = os.path.relpath(filepath, settings.PATHS['PREDICTION_INPUT_DIR'])
        path_parts = relative_path.split(os.sep)
        cluster_id = path_parts[0]
        filename = os.path.basename(filepath)
        
        micro_match = re.search(r'micro_(\d+)', filename)
        if not micro_match: return None
        
        micro_cluster_id = f"micro_{micro_match.group(1)}"
        file_basename = filename.replace('.json', '')
        
    except Exception as e:
        print(f"Error parsing path {filepath}: {e}")
        return None
    
    # Set up cluster-specific logging (captures all terminal output)
    # Using the TeeOutput context manager from logging_utils
    with logging_utils.TeeOutput(cluster_id, settings.PATHS['OUTPUT_BASE_DIR']):
        return _process_micro_cluster_file_internal(
            filepath, 
            client, 
            user_chars_data, 
            cluster_id, 
            micro_cluster_id, 
            file_basename, 
            category_mapping
        )
def _process_micro_cluster_file_internal(filepath, client, user_chars_data, cluster_id, micro_id, file_basename, category_mapping):
    """
    The core logic for processing a cluster: Iteration, Batching, Correction.
    """
    print(f"Processing {cluster_id} / {micro_id}")
    
    # Define Output Paths
    corrected_file = os.path.join(settings.PATHS['OUTPUT_DIRS']['corrected'], cluster_id, f"{file_basename}_corrected.json")
    batch_analyses_file = os.path.join(settings.PATHS['OUTPUT_DIRS']['memory'], cluster_id, f"{micro_id}_batch_analyses.json")
    journey_file = os.path.join(settings.PATHS['OUTPUT_DIRS']['journey'], cluster_id, f"{micro_id}_journey.json")
    cache_file = os.path.join(settings.PATHS['OUTPUT_DIRS']['cache'], cluster_id, f"{micro_id}_cache.json")
    delta_file = os.path.join(settings.PATHS['OUTPUT_DIRS']['delta'], cluster_id, f"{micro_id}_delta.json")
    failure_file = os.path.join(settings.PATHS['OUTPUT_DIRS']['failures'], cluster_id, f"{micro_id}_failures.json")

    # Load Data
    data = io_utils.load_json_file(filepath)
    if not data or 'user_predictions' not in data: return None

    # Load Metadata
    meta = io_utils.load_micro_cluster_summary(cluster_id, micro_id, data)
    baseline_persona = meta.get('qualitative_summary', {})
    persona_name = meta.get('persona_name', f'persona_for_{micro_id}')
    
    # Load Theme Map (topic universe)
    theme_map_raw = io_utils.load_json_file(settings.PATHS['THEMES_FILE'], {})
    
    # Extract category themes map from topic universe
    # Handle different structures: {category: [themes]} or {category: {topics: [themes]}}
    theme_map = {}
    if isinstance(theme_map_raw, dict):
        for category, themes_data in theme_map_raw.items():
            if isinstance(themes_data, list):
                # Direct list of themes
                theme_map[category] = [
                    t.get('topic_name', t) if isinstance(t, dict) else str(t)
                    for t in themes_data
                ]
            elif isinstance(themes_data, dict) and 'topics' in themes_data:
                # Nested structure with 'topics' key
                topics = themes_data['topics']
                if isinstance(topics, list):
                    theme_map[category] = [
                        t.get('topic_name', t) if isinstance(t, dict) else str(t)
                        for t in topics
                    ]
    
    # Normalize category names in theme_map to match category mapping format
    # Category mapping uses "Clothing Shoes & Jewelry" but theme map might use "Fashion" or variations
    # Create normalized lookup that handles both
    theme_map_normalized = {}
    for cat, themes in theme_map.items():
        # Store with original key
        theme_map_normalized[cat] = themes
        # Also store with normalized key (spaces/underscores, ampersand to "and", case-insensitive)
        # This handles: "Health & Personal Care" -> "health_and_personal_care" and "Health_and_Personal_Care" -> "health_and_personal_care"
        # Note: Replace " & " with "_and_" first, then replace remaining "&" with "and", then normalize spaces/underscores
        cat_normalized = cat.replace(' & ', '_and_').replace('&', 'and').replace(' ', '_').replace('-', '_').lower()
        if cat_normalized not in theme_map_normalized:
            theme_map_normalized[cat_normalized] = themes
    
    theme_map = theme_map_normalized
    
    # Load State (Resume capability)
    batch_analyses_log = io_utils.load_json_file(batch_analyses_file, {})
    all_batch_analyses = []
    for v in batch_analyses_log.values():
        if isinstance(v, dict) and 'analyses' in v:
            all_batch_analyses.extend(v['analyses'])
    
    # Load memory from v4 training results if available (from settings, set via config)
    # This allows using learned context from previous training runs
    v4_memory_base_dir = getattr(settings, 'V4_MEMORY_BASE_DIR', None)
    if v4_memory_base_dir:
        v4_memory_file = os.path.join(v4_memory_base_dir, cluster_id, f"{micro_id}_batch_analyses.json")
        if os.path.exists(v4_memory_file):
            v4_batch_analyses_log = io_utils.load_json_file(v4_memory_file, {})
            v4_batch_analyses = []
            for v in v4_batch_analyses_log.values():
                if isinstance(v, dict) and 'analyses' in v:
                    v4_batch_analyses.extend(v['analyses'])
            # Prepend v4 analyses to use them as context (they were learned from training)
            all_batch_analyses = v4_batch_analyses + all_batch_analyses
            logging_utils.log_thread_safe(f"[{cluster_id}/{micro_id}] Loaded {len(v4_batch_analyses)} batch analyses from v4 memory", 'info')
    
    # Create persona_memory structure
    persona_memory = {
        persona_name: {
            'batch_analyses': all_batch_analyses
        }
    }
    
    journey_log = io_utils.load_json_file(journey_file, {})
    high_delta_log = io_utils.load_json_file(delta_file, {})
    embeddings_cache = io_utils.load_json_file(cache_file, {})
    
    # Working Copy
    fixed_data = copy.deepcopy(data)
    original_data = copy.deepcopy(data)
    
    # Pre-filter: Identify reviews with empty actual themes or empty predicted themes and save to failure file
    reviews_with_empty_actual_themes = []
    reviews_with_empty_predicted_themes = []
    for user_id, user_reviews in fixed_data['user_predictions'].items():
        if isinstance(user_reviews, list):
            for review_idx, review_entry in enumerate(user_reviews):
                review_key = f"{user_id}_review_{review_idx}"
                
                # Check if actual themes are empty or missing
                actual = review_entry.get('actual', {})
                actual_themes = actual.get('predicted_themes', [])
                if not actual_themes or (isinstance(actual_themes, list) and len(actual_themes) == 0):
                    reviews_with_empty_actual_themes.append({
                        'review_key': review_key,
                        'user_id': user_id,
                        'review_idx': review_idx,
                        'reason': 'empty_actual_themes',
                        'review_data': review_entry
                    })
                    continue  # Skip to next review if actual themes are empty
                
                # Check if predicted themes are empty or missing
                prediction = review_entry.get('prediction', {})
                predicted_themes = prediction.get('predicted_themes', {})
                # Check if predicted_themes is empty dict or None
                if not predicted_themes or (isinstance(predicted_themes, dict) and len(predicted_themes) == 0):
                    reviews_with_empty_predicted_themes.append({
                        'review_key': review_key,
                        'user_id': user_id,
                        'review_idx': review_idx,
                        'reason': 'empty_predicted_themes',
                        'review_data': review_entry
                    })
    
    # Combine all excluded reviews
    all_excluded_reviews = reviews_with_empty_actual_themes + reviews_with_empty_predicted_themes
    
    # Save failures to file
    if all_excluded_reviews:
        failure_dir = os.path.dirname(failure_file)
        os.makedirs(failure_dir, exist_ok=True)
        
        # Load existing failures if any
        existing_failures = io_utils.load_json_file(failure_file, [])
        if not isinstance(existing_failures, list):
            existing_failures = []
        
        # Merge with existing (avoid duplicates)
        existing_keys = {f['review_key'] for f in existing_failures if isinstance(f, dict) and 'review_key' in f}
        new_failures = [f for f in all_excluded_reviews if f['review_key'] not in existing_keys]
        all_failures = existing_failures + new_failures
        
        io_utils.save_json_file(all_failures, failure_file)
        print(f"\n⚠️  Found {len(reviews_with_empty_actual_themes)} reviews with empty actual themes")
        print(f"⚠️  Found {len(reviews_with_empty_predicted_themes)} reviews with empty predicted themes")
        print(f"   Total excluded: {len(new_failures)} new reviews (total: {len(all_failures)})")
        print(f"   Saved to: {failure_file}")
        if new_failures:
            print(f"   Sample review keys: {[f['review_key'] for f in new_failures[:3]]}")
    
    # --- PRE-ITERATION: CALCULATE DELTAS FOR THIS MICRO CLUSTER ---
    print(f"\n{'='*70}")
    print(f"  PRE-ITERATION: CALCULATING DELTAS FOR {cluster_id}/{micro_id}")
    print(f"{'='*70}")
    
    all_reviews_deltas_file = os.path.join(settings.PATHS['OUTPUT_DIRS']['delta'], cluster_id, f"{micro_id}_all_reviews_deltas.json")
    all_reviews_deltas_dir = os.path.dirname(all_reviews_deltas_file)
    os.makedirs(all_reviews_deltas_dir, exist_ok=True)
    
    # Check if deltas already calculated and saved
    existing_all_deltas = io_utils.load_json_file(all_reviews_deltas_file, {})
    if existing_all_deltas and isinstance(existing_all_deltas, dict) and 'deltas' in existing_all_deltas:
        print(f"  - Found existing deltas file: {all_reviews_deltas_file}")
        print(f"  - Loading {len(existing_all_deltas.get('deltas', []))} pre-calculated deltas for {cluster_id}/{micro_id}")
        all_reviews_deltas = existing_all_deltas.get('deltas', [])
    else:
        # Calculate deltas for reviews in this micro cluster only
        print(f"  - Calculating deltas for reviews in {cluster_id}/{micro_id} from pre_sgo predictions...")
        all_reviews_deltas = []
        excluded_keys = {f['review_key'] for f in all_excluded_reviews}
        
        for user_id, user_reviews in fixed_data['user_predictions'].items():
            if isinstance(user_reviews, list):
                for review_idx, review_entry in enumerate(user_reviews):
                    review_key = f"{user_id}_review_{review_idx}"
                    
                    # Skip reviews with empty actual themes or empty predicted themes
                    if review_key in excluded_keys:
                        continue
                    
                    try:
                        prediction = review_entry.get('prediction', {})
                        actual = review_entry.get('actual', {})
                        
                        # Check if actual themes are empty - skip these reviews
                        actual_themes = actual.get('predicted_themes', [])
                        if not actual_themes or (isinstance(actual_themes, list) and len(actual_themes) == 0):
                            continue
                        
                        # Check if predicted themes are empty - skip these reviews
                        predicted_themes = prediction.get('predicted_themes', {})
                        if not predicted_themes or (isinstance(predicted_themes, dict) and len(predicted_themes) == 0):
                            continue
                        
                        # 1. Text Delta
                        pred_text = prediction.get('review_text', '').strip() if prediction.get('review_text') else ''
                        act_text = actual.get('review_text', '').strip() if actual.get('review_text') else ''
                        
                        if not pred_text or not act_text:
                            print(f"  - WARNING: Skipping {review_key} - missing text (pred: {bool(pred_text)}, actual: {bool(act_text)})")
                            continue
                        
                        # Calculate text_delta with retries
                        text_delta = None
                        max_embedding_retries = 3
                        last_error = None
                        
                        for retry in range(max_embedding_retries):
                            try:
                                pred_emb = embeddings.get_or_compute_embedding(
                                    pred_text,
                                    review_key,
                                    'pred_initial',
                                    client,
                                    embeddings_cache
                                )
                                act_emb = embeddings.get_or_compute_embedding(
                                    act_text,
                                    review_key,
                                    'actual',
                                    client,
                                    embeddings_cache
                                )
                                
                                if pred_emb and act_emb:
                                    text_delta = calculate_text_delta(pred_emb, act_emb)
                                    break
                                else:
                                    last_error = f"Failed to get embeddings - pred_emb: {pred_emb is not None}, act_emb: {act_emb is not None}"
                                    if retry < max_embedding_retries - 1:
                                        time.sleep(1)
                            except Exception as e:
                                last_error = f"Embedding calculation exception: {str(e)}"
                                if retry < max_embedding_retries - 1:
                                    time.sleep(1)
                        
                        if text_delta is None:
                            print(f"  - WARNING: Failed to calculate text_delta for {review_key} after {max_embedding_retries} attempts: {last_error}")
                            # Skip this review if we can't calculate text_delta
                            continue
                        
                        # 2. Theme Delta (uses appropriate function based on scoring mode)
                        # Get category themes for JSD calculation (if logprobs mode)
                        category_themes_list = None
                        if getattr(settings, 'SCORING_MODE', 'confidence') in ["logprobs", "logprobs-without-persona-context"]:
                            category = review_entry.get('category', '')
                            if category and theme_map:
                                # Map to main category
                                if category_mapping and category:
                                    if isinstance(category_mapping, dict) and 'category_to_main_mapping' in category_mapping:
                                        main_category = category_mapping['category_to_main_mapping'].get(category, category)
                                    else:
                                        main_category = category_mapping.get(category, category)
                                else:
                                    main_category = category
                                # Get themes for this category
                                category_themes_list = theme_map.get(main_category, [])
                                # Also try normalized category name
                                if not category_themes_list:
                                    cat_normalized = main_category.replace(' & ', '_and_').replace('&', 'and').replace(' ', '_').replace('-', '_').lower()
                                    category_themes_list = theme_map.get(cat_normalized, [])
                        
                        theme_delta = get_theme_delta(
                            prediction.get('predicted_themes', {}),
                            actual.get('predicted_themes', []),
                            all_themes=category_themes_list
                        )
                        
                        # 3. Overall Delta
                        overall_delta = (text_delta * settings.WEIGHT_TEXT_DELTA) + (theme_delta * settings.WEIGHT_THEME_DELTA)
                        
                        # Build comprehensive delta info
                        # For logprobs mode, theme_delta is JSD; for confidence mode, it's 1-recall
                        delta_info = {
                            'review_key': review_key,
                            'user_id': user_id,
                            'review_idx': review_idx,
                            'text_delta': text_delta,
                            'theme_delta': theme_delta,  # JSD for logprobs mode, 1-recall for confidence mode
                            'theme_jsd': theme_delta if getattr(settings, 'SCORING_MODE', 'confidence') in ["logprobs", "logprobs-without-persona-context"] else None,  # Explicit JSD field for logprobs
                            'overall_delta': overall_delta,
                            'prediction': {
                                'review_text': pred_text,
                                'rating': prediction.get('rating'),
                                'sentiment': prediction.get('sentiment'),
                                'predicted_themes': prediction.get('predicted_themes', {})
                            },
                            'actual': {
                                'review_text': act_text,
                                'rating': actual.get('rating'),
                                'sentiment': actual.get('sentiment'),
                                'predicted_themes': actual.get('predicted_themes', [])
                            },
                            'category': review_entry.get('category'),
                            'product_description': review_entry.get('product_description', '')
                        }
                        all_reviews_deltas.append(delta_info)
                        
                    except Exception as e:
                        print(f"  - ERROR calculating delta for {review_key}: {e}. Skipping.")
                        continue
        
        # Save all deltas to file
        all_deltas_data = {
            'cluster_id': cluster_id,
            'micro_cluster_id': micro_id,
            'total_reviews': len(all_reviews_deltas),
            'calculation_timestamp': time.time(),
            'deltas': all_reviews_deltas
        }
        io_utils.save_json_file(all_deltas_data, all_reviews_deltas_file)
        print(f"  - ✅ Calculated and saved deltas for {len(all_reviews_deltas)} reviews")
        print(f"  - Saved to: {all_reviews_deltas_file}")
    
    # Print delta statistics
    if all_reviews_deltas:
        theme_deltas = [d['theme_delta'] for d in all_reviews_deltas]
        text_deltas = [d['text_delta'] for d in all_reviews_deltas]
        overall_deltas = [d['overall_delta'] for d in all_reviews_deltas]
        
        print(f"\n  📊 DELTA STATISTICS (All Reviews):")
        print(f"     Theme Delta:   min={min(theme_deltas):.4f}, max={max(theme_deltas):.4f}, mean={np.mean(theme_deltas):.4f}, median={np.median(theme_deltas):.4f}")
        print(f"     Text Delta:    min={min(text_deltas):.4f}, max={max(text_deltas):.4f}, mean={np.mean(text_deltas):.4f}, median={np.median(text_deltas):.4f}")
        print(f"     Overall Delta: min={min(overall_deltas):.4f}, max={max(overall_deltas):.4f}, mean={np.mean(overall_deltas):.4f}, median={np.median(overall_deltas):.4f}")
        
        # Count reviews above threshold
        reviews_above_threshold = [d for d in all_reviews_deltas if d['theme_delta'] > settings.DELTA_THRESHOLD]
        print(f"     Reviews with theme_delta > {settings.DELTA_THRESHOLD}: {len(reviews_above_threshold)}/{len(all_reviews_deltas)} ({len(reviews_above_threshold)/len(all_reviews_deltas)*100:.1f}%)")
    
    print(f"{'='*70}\n")
    
    # Create a lookup map from pre-calculated deltas
    deltas_lookup = {d['review_key']: d for d in all_reviews_deltas}
    
    # Iteration Loop
    for i in range(settings.NUM_ITERATIONS):
        iteration_number = i + 1
        print(f"\n{'='*20} Starting Iteration #{iteration_number} {'='*20}")
        
        # --- STEP 0: DETERMINE REVIEWS TO PROCESS (GRANULAR) ---
        keys_to_scan = []
        
        # Get set of review keys with empty actual themes or empty predicted themes (to exclude)
        excluded_keys = {f['review_key'] for f in all_excluded_reviews}
        
        if iteration_number == 1:
            # First iteration: process reviews with theme_delta > threshold from pre-calculated deltas
            # (Stopping criteria: theme_delta <= threshold means we're done with that review)
            for delta_info in all_reviews_deltas:
                review_key = delta_info['review_key']
                if review_key not in excluded_keys and delta_info['theme_delta'] > settings.DELTA_THRESHOLD:
                    keys_to_scan.append(review_key)
        else:
            # Subsequent: process only specific reviews marked as failing (exclude empty actual themes and LLM failures)
            # Also include LLM-failed reviews from previous iterations for retry
            failed_predictions_file = os.path.join(settings.PATHS['OUTPUT_DIRS']['failures'], cluster_id, f"{micro_id}_llm_failed_predictions.json")
            llm_failed_keys = set()
            if os.path.exists(failed_predictions_file):
                all_llm_failed = io_utils.load_json_file(failed_predictions_file, [])
                if isinstance(all_llm_failed, list):
                    llm_failed_keys = {f['review_key'] for f in all_llm_failed if isinstance(f, dict) and 'review_key' in f}
                    if llm_failed_keys:
                        print(f"  - Found {len(llm_failed_keys)} LLM-failed reviews from previous iterations - will retry")
            
            # Combine high_delta_log keys and LLM-failed keys (both need retry)
            keys_to_retry = set(high_delta_log.keys()) | llm_failed_keys
            keys_to_scan = [key for key in keys_to_retry if key not in excluded_keys]

        if not keys_to_scan:
            print("No reviews to process. Ending loop.")
            break
        
        # --- STEP 0.5: CALCULATE/UPDATE DELTAS FOR CURRENT PREDICTIONS ---
        if iteration_number == 1:
            # First iteration: Use pre-calculated initial deltas (already calculated and saved)
            print(f"\n--- Using pre-calculated initial deltas for {len(keys_to_scan)} reviews ---")
            reviews_needing_correction = []
            all_review_deltas = []
            
            for review_key in keys_to_scan:
                try:
                    # Get pre-calculated delta info
                    if review_key not in deltas_lookup:
                        print(f"  - WARNING: No pre-calculated delta found for {review_key}. Skipping.")
                        continue
                    
                    pre_calc_delta = deltas_lookup[review_key]
                    user_id = pre_calc_delta['user_id']
                    review_idx = pre_calc_delta['review_idx']
                    
                    if user_id not in fixed_data['user_predictions']:
                        print(f"  - WARNING: user_id {user_id} not found in fixed_data. Skipping.")
                        continue
                    
                    user_reviews = fixed_data['user_predictions'][user_id]
                    if not isinstance(user_reviews, list) or review_idx >= len(user_reviews):
                        print(f"  - WARNING: review_idx {review_idx} out of range for user {user_id}. Skipping.")
                        continue
                    
                    review_entry = user_reviews[review_idx]
                    
                    # Use pre-calculated deltas
                    text_delta = pre_calc_delta['text_delta']
                    theme_delta = pre_calc_delta['theme_delta']
                    overall_delta = pre_calc_delta['overall_delta']
                    
                    # Format delta_info to match expected structure
                    delta_info = {
                        'user_id': user_id,
                        'review_idx': review_idx,
                        'text': text_delta,
                        'theme': theme_delta,
                        'overall': overall_delta,
                        'review_key': review_key,
                        'data': review_entry
                    }
                    all_review_deltas.append(delta_info)
                    
                    # Check Threshold - only process reviews with theme_delta > threshold
                    if theme_delta > settings.DELTA_THRESHOLD:
                        reviews_needing_correction.append({
                            'key': review_key,
                            'user_id': user_id,
                            'review_idx': review_idx,
                            'data': review_entry,
                            'delta_info': delta_info
                        })
                except Exception as e:
                    print(f"  - CRITICAL ERROR processing pre-calculated delta for {review_key}: {e}. Skipping.")
                    continue
        else:
            # Subsequent iterations: Recalculate deltas for current predictions (may have been updated)
            print(f"\n--- Recalculating deltas for {len(keys_to_scan)} reviews (using current predictions) ---")
            reviews_needing_correction = []
            all_review_deltas = []
            
            for review_key in keys_to_scan:
                try:
                    # Parse Key
                    if "_review_" in review_key:
                        user_id, review_idx_str = review_key.rsplit("_review_", 1)
                        try:
                            review_idx = int(review_idx_str)
                        except ValueError:
                            print(f"  - WARNING: Invalid review key format {review_key}. Skipping.")
                            continue
                    else:
                        user_id = review_key
                        review_idx = 0

                    if user_id not in fixed_data['user_predictions']:
                        print(f"  - WARNING: user_id {user_id} not found in fixed_data. Skipping.")
                        continue
                    
                    user_reviews = fixed_data['user_predictions'][user_id]
                    if not isinstance(user_reviews, list) or review_idx >= len(user_reviews):
                        print(f"  - WARNING: review_idx {review_idx} out of range for user {user_id}. Skipping.")
                        continue
                    
                    review_entry = user_reviews[review_idx]
                    prediction = review_entry.get('prediction', {})
                    actual = review_entry.get('actual', {})
                    
                    # Check if actual themes are empty - skip these reviews
                    actual_themes = actual.get('predicted_themes', [])
                    if not actual_themes or (isinstance(actual_themes, list) and len(actual_themes) == 0):
                        continue
                    
                    # Check if predicted themes are empty - skip these reviews
                    predicted_themes = prediction.get('predicted_themes', {})
                    if not predicted_themes or (isinstance(predicted_themes, dict) and len(predicted_themes) == 0):
                        continue
                    
                    # Calculate deltas for CURRENT prediction (may have been updated in previous iterations)
                    pred_text = prediction.get('review_text', '').strip() if prediction.get('review_text') else ''
                    act_text = actual.get('review_text', '').strip() if actual.get('review_text') else ''
                    
                    if not pred_text or not act_text:
                        print(f"  - WARNING: Skipping {review_key} - missing text (pred: {bool(pred_text)}, actual: {bool(act_text)})")
                        continue
                    
                    # Calculate text_delta with retries
                    text_delta = None
                    max_embedding_retries = 3
                    last_error = None
                    
                    for retry in range(max_embedding_retries):
                        try:
                            pred_emb = embeddings.get_or_compute_embedding(
                                pred_text,
                                review_key,
                                f'pred_iter_{iteration_number}',
                                client,
                                embeddings_cache
                            )
                            act_emb = embeddings.get_or_compute_embedding(
                                act_text,
                                review_key,
                                'actual',
                                client,
                                embeddings_cache
                            )
                            
                            if pred_emb and act_emb:
                                text_delta = calculate_text_delta(pred_emb, act_emb)
                                break
                            else:
                                last_error = f"Failed to get embeddings - pred_emb: {pred_emb is not None}, act_emb: {act_emb is not None}"
                                if retry < max_embedding_retries - 1:
                                    time.sleep(1)
                        except Exception as e:
                            last_error = f"Embedding calculation exception: {str(e)}"
                            if retry < max_embedding_retries - 1:
                                time.sleep(1)
                    
                    if text_delta is None:
                        print(f"  - WARNING: Failed to calculate text_delta for {review_key} after {max_embedding_retries} attempts: {last_error}")
                        continue
                    
                    # Calculate theme_delta (uses appropriate function based on scoring mode)
                    # Get category themes for JSD calculation (if logprobs mode)
                    category_themes_list = None
                    if getattr(settings, 'SCORING_MODE', 'confidence') in ["logprobs", "logprobs-without-persona-context"]:
                        category = review_entry.get('category', '')
                        if category and theme_map:
                            # Map to main category
                            if category_mapping and category:
                                if isinstance(category_mapping, dict) and 'category_to_main_mapping' in category_mapping:
                                    main_category = category_mapping['category_to_main_mapping'].get(category, category)
                                else:
                                    main_category = category_mapping.get(category, category)
                            else:
                                main_category = category
                            # Get themes for this category
                            category_themes_list = theme_map.get(main_category, [])
                            # Also try normalized category name
                            if not category_themes_list:
                                cat_normalized = main_category.replace(' & ', '_and_').replace('&', 'and').replace(' ', '_').replace('-', '_').lower()
                                category_themes_list = theme_map.get(cat_normalized, [])
                    
                    theme_delta = get_theme_delta(
                        prediction.get('predicted_themes', {}),
                        actual.get('predicted_themes', []),
                        all_themes=category_themes_list
                    )
                    
                    # Calculate overall_delta
                    overall_delta = (text_delta * settings.WEIGHT_TEXT_DELTA) + (theme_delta * settings.WEIGHT_THEME_DELTA)
                    
                    # Format delta_info
                    delta_info = {
                        'user_id': user_id,
                        'review_idx': review_idx,
                        'text': text_delta,
                        'theme': theme_delta,
                        'overall': overall_delta,
                        'review_key': review_key,
                        'data': review_entry
                    }
                    all_review_deltas.append(delta_info)
                    
                    # Check Threshold - process reviews with theme_delta > threshold
                    # (Stopping criteria: theme_delta <= threshold means we're done)
                    if theme_delta > settings.DELTA_THRESHOLD:
                        reviews_needing_correction.append({
                            'key': review_key,
                            'user_id': user_id,
                            'review_idx': review_idx,
                            'data': review_entry,
                            'delta_info': delta_info
                        })

                except Exception as e:
                    print(f"  - CRITICAL ERROR calculating delta for {review_key}: {e}. Skipping.")
                    continue
        
        # --- End Delta Calculation ---
        print(f"  - Calculated deltas for {len(all_review_deltas)} reviews")
        
        # Debug: Show delta distribution
        if all_review_deltas:
            theme_deltas = [d['theme'] for d in all_review_deltas]
            if theme_deltas:
                print(f"  - Theme Delta stats: range=[{min(theme_deltas):.4f}, {max(theme_deltas):.4f}], mean={np.mean(theme_deltas):.4f}")
        
        print(f"  - Found {len(reviews_needing_correction)} reviews with theme_delta > {settings.DELTA_THRESHOLD}")
        
        if reviews_needing_correction:
            print(f"  - Reviews being processed (theme_delta > {settings.DELTA_THRESHOLD}):")
            for r in reviews_needing_correction[:10]:  # Show first 10
                d = r['delta_info']
                print(f"      {r['key']}: theme_delta={d['theme']:.4f}, overall_delta={d['overall']:.4f}")
            if len(reviews_needing_correction) > 10:
                print(f"      ... and {len(reviews_needing_correction) - 10} more")
        
        # Check if ALL reviews have theme_delta == 0 (goal achieved - stop iterations)
        all_theme_deltas = [d['theme'] for d in all_review_deltas] if all_review_deltas else []
        all_reviews_at_zero = all_theme_deltas and all(abs(td) < 0.0001 for td in all_theme_deltas)  # Use small epsilon for float comparison
        
        if not reviews_needing_correction:
            if all_reviews_at_zero:
                print(f"  - ✅ SUCCESS: All reviews have theme_delta == 0. Goal achieved! Stopping iterations.")
                # Even if all reviews are perfect, we should still log them in journey for tracking
                # Add reviews with theme_delta == 0 to journey_log if not already present
                for delta_info in all_review_deltas:
                    review_key = delta_info['review_key']
                    if review_key not in journey_log and delta_info['theme'] == 0:
                        # Get review data
                        user_id = delta_info['user_id']
                        review_idx = delta_info['review_idx']
                        review_entry = delta_info['data']
                        
                        journey_utils.ensure_journey_log_entry(
                            journey_log,
                            review_key,
                            user_id,
                            review_idx,
                            review_entry,
                            {
                                'text': delta_info['text'],
                                'theme': delta_info['theme'],
                                'overall': delta_info['overall']
                            },
                            initial_prediction_data=review_entry.get('prediction')
                        )
                        print(f"  - Added perfect review {review_key} to journey log (theme_delta=0, no correction needed)")
                # Save final journey state before breaking
                io_utils.save_json_file(journey_log, journey_file)
                break  # Stop all iterations - goal achieved
            else:
                # No reviews above threshold, but some still have theme_delta > 0
                # Process reviews with theme_delta > 0 (even if <= threshold) to reach goal of 0
                reviews_with_positive_delta = [
                    {
                        'key': d['review_key'],
                        'user_id': d['user_id'],
                        'review_idx': d['review_idx'],
                        'data': d['data'],
                        'delta_info': d
                    }
                    for d in all_review_deltas
                    if d['theme'] > 0 and d['theme'] <= settings.DELTA_THRESHOLD
                ]
                
                if reviews_with_positive_delta:
                    print(f"  - STATUS: No reviews with theme_delta > {settings.DELTA_THRESHOLD} found.")
                    print(f"  - INFO: Found {len(reviews_with_positive_delta)} reviews with 0 < theme_delta <= {settings.DELTA_THRESHOLD}.")
                    print(f"  - Processing these reviews to reach theme_delta == 0...")
                    # Update reviews_needing_correction to include these
                    reviews_needing_correction = reviews_with_positive_delta
                else:
                    # All reviews are at 0 or below threshold, but not all are exactly 0
                    # Update high_delta_log and continue
                    print(f"  - STATUS: No reviews with theme_delta > {settings.DELTA_THRESHOLD} found.")
                    print(f"  - INFO: {len([td for td in all_theme_deltas if td > 0])} reviews still have theme_delta > 0 (but <= threshold).")
                    print(f"  - Continuing to next iteration...")
                    # Update high_delta_log to include reviews with theme_delta > 0
                    for delta_info in all_review_deltas:
                        if delta_info['theme'] > 0:
                            high_delta_log[delta_info['review_key']] = True
                    # Also add reviews with theme_delta == 0 to journey_log if not already present (for tracking)
                    for delta_info in all_review_deltas:
                        review_key = delta_info['review_key']
                        if review_key not in journey_log and delta_info['theme'] == 0:
                            # Get review data
                            user_id = delta_info['user_id']
                            review_idx = delta_info['review_idx']
                            review_entry = delta_info['data']
                            
                            journey_utils.ensure_journey_log_entry(
                                journey_log,
                                review_key,
                                user_id,
                                review_idx,
                                review_entry,
                                {
                                    'text': delta_info['text'],
                                    'theme': delta_info['theme'],
                                    'overall': delta_info['overall']
                                },
                                initial_prediction_data=review_entry.get('prediction')
                            )
                    # Save journey even if no corrections this iteration
                    io_utils.save_json_file(journey_log, journey_file)
                    continue
        
        avg_theme_delta = np.mean([d['delta_info']['theme'] for d in reviews_needing_correction])
        print(f"  - Average Theme Delta for reviews needing correction: {avg_theme_delta:.4f}")

        # ============================================
        # STEP 1: GROUP BY CATEGORY THEN BATCH
        # ============================================
        # Group reviews by category to ensure each batch uses correct category themes
        from collections import defaultdict
        reviews_by_category = defaultdict(list)
        for review_info in reviews_needing_correction:
            review_data = review_info.get('data', {})
            category = review_data.get('category', 'unknown')
            # Map to main category (same logic as batch_processor.map_category_to_main_category)
            if category_mapping and category:
                # category_mapping is a dict: {sub_category: main_category}
                # Check if it has 'category_to_main_mapping' key (nested structure)
                if isinstance(category_mapping, dict) and 'category_to_main_mapping' in category_mapping:
                    main_category = category_mapping['category_to_main_mapping'].get(category, category)
                else:
                    # Direct mapping structure
                    main_category = category_mapping.get(category, category)
            else:
                main_category = category
            reviews_by_category[main_category].append(review_info)
        
        # Create batches within each category group
        batches = []
        all_review_keys_in_batches = set()  # Track review keys to prevent duplicates
        for main_category, category_reviews in reviews_by_category.items():
            category_batches = [category_reviews[x:x+settings.BATCH_SIZE] for x in range(0, len(category_reviews), settings.BATCH_SIZE)]
            
            # Verify no duplicate reviews in batches (safety check)
            for batch in category_batches:
                batch_keys = {r['key'] for r in batch if 'key' in r}
                duplicates = batch_keys & all_review_keys_in_batches
                if duplicates:
                    print(f"  ⚠️  WARNING: Found duplicate review keys in batches: {duplicates}")
                    # Remove duplicates (keep first occurrence)
                    batch[:] = [r for r in batch if r.get('key') not in all_review_keys_in_batches]
                all_review_keys_in_batches.update(batch_keys)
            
            batches.extend(category_batches)
            if len(category_reviews) > settings.BATCH_SIZE:
                print(f"  - Category '{main_category}': {len(category_reviews)} reviews -> {len(category_batches)} batches")
        
        print(f"\nProcessing {len(reviews_needing_correction)} reviews needing correction in {len(batches)} batches (grouped by category, size up to {settings.BATCH_SIZE}).")
        print(f"  [PARALLEL MODE] Processing batches in parallel (max {settings.MAX_PARALLEL_BATCHES} concurrent batches)")

        still_failing_users = set()
        analyses_to_skip_global = set() 
        batch_analyses_new_global = []
        batch_results_list = []  # Collect batch results for summary
        all_failed_predictions = []  # Collect all failed predictions from all batches

        # Function to run inside thread
        def _run_batch(batch_idx, batch_items):
            return process_single_batch(
                batch_items, batch_idx, iteration_number, baseline_persona, persona_name,
                {'persona_name': {'batch_analyses': all_batch_analyses}},
                user_chars_data, theme_map, embeddings_cache, client,
                batch_analyses_log, batch_analyses_file, all_batch_analyses,
                fixed_data, journey_log, category_mapping, file_lock
            )

        # Process batches in parallel
        max_workers = min(settings.MAX_PARALLEL_BATCHES, len(batches))
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_batch = {
                executor.submit(_run_batch, batch_index, batch_reviews): (batch_index, batch_reviews)
                for batch_index, batch_reviews in enumerate(batches)
            }
            
            for future in as_completed(future_to_batch):
                batch_index, batch_reviews = future_to_batch[future]
                try:
                    res = future.result()
                    if res and res[0]:
                        result_dict, _, batch_analyses_new, analyses_to_revoke, failed_predictions = res
                        # Store batch_index, result_dict, batch_analyses_new, analyses_to_revoke, and failed_predictions
                        batch_results_list.append((batch_index, result_dict, batch_analyses_new, analyses_to_revoke, failed_predictions))
                        
                        # Collect failed predictions (LLM failures after retries)
                        all_failed_predictions.extend(failed_predictions)
                        
                        # Apply Updates Safely (only successful corrections)
                        updates = result_dict.get('updates', [])
                        for update in updates:
                            uid = update['user_id']
                            ridx = update['review_idx']
                            pred = update['prediction']
                            if uid in fixed_data['user_predictions']:
                                fixed_data['user_predictions'][uid][ridx]['prediction'].update(pred)
                        
                        # Collect Memory Updates
                        batch_analyses_new_global.extend(batch_analyses_new)
                        for item in analyses_to_revoke:
                            analyses_to_skip_global.add(item['analysis'])
                            
                        # Track Failing (including LLM failures)
                        still_failing_users.update(result_dict.get('still_failing', set()))
                        
                        # Debug: Check journey_log size after batch processing
                        if batch_index == 0:  # Only log for first batch to avoid spam
                            print(f"  [DEBUG] Journey log size after batch {batch_index + 1}: {len(journey_log)} reviews")
                    else:
                        print(f"  [Batch {batch_index + 1}] WARNING: No result returned")
                except SystemExit as e:
                    # Critical error - stop execution immediately
                    print(f"\n❌ CRITICAL ERROR in Batch {batch_index + 1}: {e}")
                    print("Stopping execution due to critical error.")
                    import traceback
                    traceback.print_exc()
                    # Re-raise to stop all processing
                    raise
                except FileNotFoundError as e:
                    # Critical error: prompt file not found - stop execution
                    if "CRITICAL" in str(e) or "prompt" in str(e).lower():
                        print(f"\n❌ CRITICAL ERROR in Batch {batch_index + 1}: {e}")
                        print("Stopping execution due to missing prompt file.")
                        import traceback
                        traceback.print_exc()
                        raise SystemExit(f"CRITICAL: Prompt file not found. Stopping execution. {str(e)}")
                    else:
                        # Other FileNotFoundError - log and continue (shouldn't happen, but handle gracefully)
                        print(f"  [Batch {batch_index + 1}] ERROR: {e}")
                        import traceback
                        traceback.print_exc()
                except Exception as e:
                    print(f"  [Batch {batch_index + 1}] ERROR in parallel processing: {e}")
                    import traceback
                    traceback.print_exc()
        
        # Sort batch results by batch_index to maintain order
        batch_results_list.sort(key=lambda x: x[0])
        
        # Save failed predictions to separate file (LLM failures after retries)
        # Always save, even if empty, to ensure the file exists
        failed_predictions_file = os.path.join(settings.PATHS['OUTPUT_DIRS']['failures'], cluster_id, f"{micro_id}_llm_failed_predictions.json")
        failed_dir = os.path.dirname(failed_predictions_file)
        os.makedirs(failed_dir, exist_ok=True)
        
        # Load existing failed predictions if any
        existing_failed = io_utils.load_json_file(failed_predictions_file, [])
        if not isinstance(existing_failed, list):
            existing_failed = []
        
        if all_failed_predictions:
            # Merge with existing (avoid duplicates by review_key + iteration)
            existing_keys = {(f['review_key'], f.get('iteration', 0)) for f in existing_failed if isinstance(f, dict) and 'review_key' in f}
            new_failed = [f for f in all_failed_predictions if (f['review_key'], f.get('iteration', 0)) not in existing_keys]
            all_failed_combined = existing_failed + new_failed
            
            io_utils.save_json_file(all_failed_combined, failed_predictions_file)
            print(f"\n⚠️  Found {len(new_failed)} LLM failures after retries (total: {len(all_failed_combined)})")
            print(f"   Saved to: {failed_predictions_file}")
            if new_failed:
                print(f"   Sample review keys: {[f['review_key'] for f in new_failed[:3]]}")
        else:
            # Save empty list if no failures (to ensure file exists)
            io_utils.save_json_file(existing_failed, failed_predictions_file)
            print(f"\n✅ No LLM failures in this iteration (existing failures: {len(existing_failed)})")
        
        # Print per-batch summaries (ordered, after all batches complete)
        # Calculate cumulative memory state for accurate "before" calculations
        cumulative_memory_before = len(all_batch_analyses)
        
        print(f"\n{'='*70}")
        print(f"  PER-BATCH SUMMARIES (Ordered by Batch Number)")
        print(f"{'='*70}\n")
        
        for batch_index, result_dict, batch_analyses_new, analyses_to_revoke, failed_predictions in batch_results_list:
            batch_results = result_dict.get('batch_results', [])
            
            if not batch_results:
                continue
                
            # Calculate batch statistics
            batch_improvements = len([r for r in batch_results if r.get('improvement_status') == 'IMPROVED'])
            batch_worse = len([r for r in batch_results if r.get('improvement_status') == 'WORSE'])
            batch_unchanged = len([r for r in batch_results if r.get('improvement_status') == 'UNCHANGED'])
            total_batch_reviews = len(batch_results)
            
            # ============================================
            # BATCH SUMMARY & MEMORY REVOCATION
            # ============================================
            print(f"\n  [Batch {batch_index + 1}] {'='*70}")
            print(f"  [Batch {batch_index + 1}] BATCH SUMMARY")
            print(f"  [Batch {batch_index + 1}] {'='*70}")
            
            print(f"  [Batch {batch_index + 1}] ┌─ REVIEW STATUS SUMMARY")
            print(f"  [Batch {batch_index + 1}] │  Total Reviews:     {total_batch_reviews}")
            if total_batch_reviews > 0:
                print(f"  [Batch {batch_index + 1}] │  ✅ Improved:        {batch_improvements} ({batch_improvements/total_batch_reviews*100:.1f}%)")
                print(f"  [Batch {batch_index + 1}] │  ❌ Got Worse:       {batch_worse} ({batch_worse/total_batch_reviews*100:.1f}%)")
                print(f"  [Batch {batch_index + 1}] │  ➡️  Unchanged:        {batch_unchanged} ({batch_unchanged/total_batch_reviews*100:.1f}%)")
            else:
                print(f"  [Batch {batch_index + 1}] │  ✅ Improved:        {batch_improvements} (N/A - no reviews)")
                print(f"  [Batch {batch_index + 1}] │  ❌ Got Worse:       {batch_worse} (N/A - no reviews)")
                print(f"  [Batch {batch_index + 1}] │  ➡️  Unchanged:        {batch_unchanged} (N/A - no reviews)")
            if batch_results:
                avg_initial_theme = np.mean([r.get('initial_theme_delta', 0) for r in batch_results])
                avg_new_theme = np.mean([r.get('new_theme_delta', 0) for r in batch_results])
                improved_results = [r for r in batch_results if r.get('improvement_status') == 'IMPROVED']
                if improved_results:
                    avg_improvement = np.mean([r.get('theme_improvement_pct', 0) for r in improved_results])
                    print(f"  [Batch {batch_index + 1}] │  Average Initial Theme Delta: {avg_initial_theme:.4f}")
                    print(f"  [Batch {batch_index + 1}] │  Average New Theme Delta:      {avg_new_theme:.4f}")
                    print(f"  [Batch {batch_index + 1}] │  Average Improvement:         {avg_improvement:.2f}%")
            print(f"  [Batch {batch_index + 1}] └─")
            
            # Memory Information
            print(f"\n  [Batch {batch_index + 1}] ┌─ MEMORY INFORMATION (This Batch)")
            print(f"  [Batch {batch_index + 1}] │  Batch Analyses Generated: {len(batch_analyses_new)}")
            print(f"  [Batch {batch_index + 1}] │  Analyses to Add to Memory: {len(batch_analyses_new) - len(analyses_to_revoke)}")
            print(f"  [Batch {batch_index + 1}] │  Analyses to Revoke:         {len(analyses_to_revoke)}")
            if analyses_to_revoke:
                print(f"  [Batch {batch_index + 1}] │  Revoked Analyses:")
                for revoke in analyses_to_revoke:
                    if isinstance(revoke, dict):
                        review_key = revoke.get('review_key', 'N/A')
                        old_delta = revoke.get('old_theme_delta', 0)
                        new_delta = revoke.get('new_theme_delta', 0)
                        print(f"  [Batch {batch_index + 1}] │    - {review_key}: {old_delta:.4f} -> {new_delta:.4f}")
            # Calculate memory state: before this batch, after this batch
            memory_before_batch = cumulative_memory_before
            memory_after_batch = cumulative_memory_before + len(batch_analyses_new) - len(analyses_to_revoke)
            print(f"  [Batch {batch_index + 1}] │  Total Memory Before:       {memory_before_batch}")
            print(f"  [Batch {batch_index + 1}] │  Total Memory After:         {memory_after_batch}")
            # Update cumulative for next batch
            cumulative_memory_before = memory_after_batch
            if batch_analyses_new:
                print(f"  [Batch {batch_index + 1}] │  Sample Analyses (first 2):")
                for i, analysis in enumerate(batch_analyses_new[:2]):
                    analysis_preview = analysis[:150] + "..." if len(analysis) > 150 else analysis
                    print(f"  [Batch {batch_index + 1}] │    {i+1}. {analysis_preview}")
            print(f"  [Batch {batch_index + 1}] └─")
            
            print(f"\n  [Batch {batch_index + 1}] {'='*70}")
            print(f"  [Batch {batch_index + 1}] BATCH COMPLETE")
            print(f"  [Batch {batch_index + 1}] {'='*70}\n")
        
        print(f"{'='*70}")
        print(f"  END OF PER-BATCH SUMMARIES")
        print(f"{'='*70}\n")

        # --- STEP 2: UPDATE MEMORY ---
        # Update all_batch_analyses (filter out revoked analyses)
        analyses_to_skip = analyses_to_skip_global  # Already a set of analysis strings
        for analysis in batch_analyses_new_global:
            if analysis not in analyses_to_skip:
                all_batch_analyses.append(analysis)
        
        # Update persona_memory
        persona_memory[persona_name]['batch_analyses'] = all_batch_analyses
        
        # Print detailed summary
        all_batch_results = []
        for br in batch_results_list:
            # br is now (batch_index, result_dict, batch_analyses_new, analyses_to_revoke)
            result_dict = br[1]
            all_batch_results.extend(result_dict.get('batch_results', []))
        
        total_improved = len([r for r in all_batch_results if r.get('improvement_status') == 'IMPROVED'])
        total_worse = len([r for r in all_batch_results if r.get('improvement_status') == 'WORSE'])
        total_unchanged = len([r for r in all_batch_results if r.get('improvement_status') == 'UNCHANGED'])
        total_reviews = len(all_batch_results)
        
        print(f"\n{'='*70}")
        print(f"  ITERATION {iteration_number} SUMMARY (Parallel Processing)")
        print(f"{'='*70}")
        print(f"  ┌─ BATCH PROCESSING SUMMARY")
        print(f"  │  Total Batches Processed:  {len(batches)}")
        print(f"  │  Total Reviews Processed:   {total_reviews}")
        if total_reviews > 0:
            print(f"  │  ✅ Improved:               {total_improved} ({total_improved/total_reviews*100:.1f}%)")
            print(f"  │  ❌ Got Worse:              {total_worse} ({total_worse/total_reviews*100:.1f}%)")
            print(f"  │  ➡️  Unchanged:              {total_unchanged} ({total_unchanged/total_reviews*100:.1f}%)")
        else:
            print(f"  │  ✅ Improved:               {total_improved} (N/A - no reviews processed)")
            print(f"  │  ❌ Got Worse:              {total_worse} (N/A - no reviews processed)")
            print(f"  │  ➡️  Unchanged:              {total_unchanged} (N/A - no reviews processed)")
        if all_batch_results:
            avg_initial_theme = np.mean([r.get('initial_theme_delta', 0) for r in all_batch_results])
            avg_new_theme = np.mean([r.get('new_theme_delta', 0) for r in all_batch_results])
            improved_results = [r for r in all_batch_results if r.get('improvement_status') == 'IMPROVED']
            if improved_results:
                avg_improvement = np.mean([r.get('theme_improvement_pct', 0) for r in improved_results])
                print(f"  │  Average Initial Theme Delta: {avg_initial_theme:.4f}")
                print(f"  │  Average New Theme Delta:      {avg_new_theme:.4f}")
                print(f"  │  Average Improvement:           {avg_improvement:.2f}%")
        print(f"  └─")
        
        print(f"\n  ┌─ MEMORY SUMMARY")
        print(f"  │  Total Analyses Generated:  {len(batch_analyses_new_global)}")
        print(f"  │  Analyses Added to Memory:   {len(batch_analyses_new_global) - len(analyses_to_skip)}")
        print(f"  │  Analyses Revoked:           {len(analyses_to_skip)}")
        print(f"  │  Total Memory Before:        {len(all_batch_analyses) - len(batch_analyses_new_global) + len(analyses_to_skip)}")
        print(f"  │  Total Memory After:         {len(all_batch_analyses)}")
        print(f"  └─")
        
        # Refined characteristics summary removed
        
        print(f"\n{'='*70}\n")
        
        # Update logs (using keys, not user_ids)
        # Merge with existing high_delta_log (don't overwrite - preserve reviews from previous iterations)
        for key in still_failing_users:
            high_delta_log[key] = True
        
        # Save Checkpoints (save journey after EACH iteration to preserve progress)
        io_utils.save_json_file(embeddings_cache, cache_file)
        io_utils.save_json_file(journey_log, journey_file)  # Save journey after each iteration
        io_utils.save_json_file(high_delta_log, delta_file)
        
        # Log journey save status
        total_journey_entries = sum(len(j.get('correction_journey', [])) for j in journey_log.values())
        print(f"  💾 Saved journey log: {len(journey_log)} reviews, {total_journey_entries} total correction entries")
        
        valid_new_analyses = [a for a in batch_analyses_new_global if a not in analyses_to_skip]
        print(f"Iteration {iteration_number} complete. Memory updated with {len(valid_new_analyses)} new analyses.")

    print(f"\n{'='*20} All Iterations Complete {'='*20}")

    # Save journey log one final time before finalization (in case it wasn't saved properly)
    print(f"\n💾 Final journey log save: {len(journey_log)} reviews, {sum(len(j.get('correction_journey', [])) for j in journey_log.values())} total correction entries")
    io_utils.save_json_file(journey_log, journey_file)
    
    # 4. Finalize
    print("\nSelecting best predictions...")
    final_best_data = select_best_prediction_by_theme_delta(fixed_data, journey_log, original_data)
    
    # Load all failed predictions (LLM failures after retries) to exclude from main output
    failed_predictions_file = os.path.join(settings.PATHS['OUTPUT_DIRS']['failures'], cluster_id, f"{micro_id}_llm_failed_predictions.json")
    all_llm_failed_keys = set()
    if os.path.exists(failed_predictions_file):
        all_llm_failed = io_utils.load_json_file(failed_predictions_file, [])
        if isinstance(all_llm_failed, list):
            all_llm_failed_keys = {f['review_key'] for f in all_llm_failed if isinstance(f, dict) and 'review_key' in f}
            if all_llm_failed_keys:
                print(f"\n⚠️  Excluding {len(all_llm_failed_keys)} LLM-failed reviews from main output")
    
    # Remove failed predictions from main output (exclude from final_best_data)
    if all_llm_failed_keys:
        for user_id, user_reviews in list(final_best_data.get('user_predictions', {}).items()):
            if isinstance(user_reviews, list):
                # Remove reviews that failed LLM calls
                reviews_to_keep = []
                for review_idx, review_entry in enumerate(user_reviews):
                    review_key = f"{user_id}_review_{review_idx}"
                    if review_key not in all_llm_failed_keys:
                        reviews_to_keep.append(review_entry)
                
                if len(reviews_to_keep) < len(user_reviews):
                    # Update with filtered reviews
                    if reviews_to_keep:
                        final_best_data['user_predictions'][user_id] = reviews_to_keep
                    else:
                        # Remove user if all reviews failed
                        del final_best_data['user_predictions'][user_id]
    
    # Recalculate metrics only for successful predictions (excluded failed ones)
    # Pass theme_map and category_mapping for JSD calculation (maps category to list of themes)
    # category_mapping is needed to map sub-categories to main categories before theme lookup
    final_best_data = recalculate_all_review_metrics(final_best_data, all_themes_map=theme_map, category_mapping=category_mapping)
    final_data_with_metrics = calculate_and_append_final_metrics(final_best_data, all_themes_map=theme_map, category_mapping=category_mapping)
    
    # Ensure output directory exists
    output_cluster_dir = os.path.dirname(corrected_file)
    os.makedirs(output_cluster_dir, exist_ok=True)
    
    # Calculate metadata fields (only for successful predictions)
    total_users = len(final_data_with_metrics.get('user_predictions', {}))
    total_reviews = sum(len(reviews) if isinstance(reviews, list) else 0 
                       for reviews in final_data_with_metrics.get('user_predictions', {}).values())
    quantitative_summary = calculate_quantitative_summary_from_data(final_data_with_metrics)
    
    # Build qualitative_summary from baseline_persona (includes persona_summary, key_motivations, etc.)
    qualitative_summary = copy.deepcopy(baseline_persona) if baseline_persona else {}
    
    if 'metadata' not in final_data_with_metrics: final_data_with_metrics['metadata'] = {}
    
    # Add complete metadata structure matching the desired output format
    final_data_with_metrics['metadata']['persona_name'] = persona_name
    final_data_with_metrics['metadata']['micro_cluster_id'] = micro_id
    final_data_with_metrics['metadata']['total_users_in_cluster'] = total_users
    final_data_with_metrics['metadata']['total_reviews_from_cluster'] = total_reviews
    final_data_with_metrics['metadata']['quantitative_summary'] = quantitative_summary
    final_data_with_metrics['metadata']['qualitative_summary'] = qualitative_summary
    # Refined characteristics removed - no longer generated
    
    # Also include backward compatibility fields
    final_data_with_metrics['metadata']['seed_characteristics'] = baseline_persona  # Original baseline (seed)
    # baseline_persona removed - same as qualitative_summary
    
    # Add model_type_used to match initial predictions format
    final_data_with_metrics['model_type_used'] = 'enhanced_delta_corrected'
    
    # Save main output (only successful predictions)
    io_utils.save_json_file(final_data_with_metrics, corrected_file)
    print(f"Final corrected data saved to '{corrected_file}' (excluded {len(all_llm_failed_keys)} LLM-failed reviews).")
    
    # Save failed predictions in same structure as main output (for reprocessing)
    if all_llm_failed_keys and os.path.exists(failed_predictions_file):
        all_llm_failed = io_utils.load_json_file(failed_predictions_file, [])
        if isinstance(all_llm_failed, list) and all_llm_failed:
            # Build failed predictions output in same structure as main output
            failed_output = {
                'model_type_used': 'llm_failed_predictions',
                'user_predictions': {},
                'metadata': {
                    'persona_name': persona_name,
                    'micro_cluster_id': micro_id,
                    'total_failed_reviews': len(all_llm_failed),
                    'failure_reason': 'llm_call_failed_after_retries',
                    'note': 'These reviews failed LLM calls after all retries and can be reprocessed later'
                }
            }
            
            # Group failed predictions by user_id
            for failed_item in all_llm_failed:
                if isinstance(failed_item, dict):
                    user_id = failed_item.get('user_id')
                    review_data = failed_item.get('review_data', {})
                    if user_id and review_data:
                        if user_id not in failed_output['user_predictions']:
                            failed_output['user_predictions'][user_id] = []
                        failed_output['user_predictions'][user_id].append(review_data)
            
            # Save failed predictions file
            failed_output_file = os.path.join(settings.PATHS['OUTPUT_DIRS']['failures'], cluster_id, f"{micro_id}_llm_failed_predictions_output.json")
            io_utils.save_json_file(failed_output, failed_output_file)
            print(f"Failed predictions output saved to '{failed_output_file}' ({len(all_llm_failed)} reviews).")
     
    return final_data_with_metrics
# ... [Previous imports and code in cluster_processor.py] ...

def generate_cluster_aggregate_summary(cluster_id):
    """
    Generate aggregate summary file for a cluster by aggregating all micro cluster results.
    """
    cluster_output_dir = os.path.join(settings.PATHS['OUTPUT_DIRS']['corrected'], cluster_id)
    if not os.path.exists(cluster_output_dir):
        print(f"  - WARNING: Output directory not found: {cluster_output_dir}")
        return None
    
    # Collect all metrics from all micro clusters in this cluster
    all_metrics = {}
    total_reviews = 0
    total_users = 0
    micro_clusters_processed = []
    
    print(f"Generating aggregate summary for {cluster_id}...")

    # Find all corrected files for this cluster
    for filename in sorted(os.listdir(cluster_output_dir)):
        if not (filename.startswith('micro_') and filename.endswith('_corrected.json')):
            continue
        
        filepath = os.path.join(cluster_output_dir, filename)
        # print(f"  - Loading: {filename}")
        
        data = io_utils.load_json_file(filepath)
        if not data or 'final_metrics' not in data:
            # print(f"    ✗ No final_metrics found")
            continue
        
        # Extract micro cluster info
        metadata = data.get('metadata', {})
        micro_cluster_id = metadata.get('micro_cluster_id', filename.replace('_corrected.json', ''))
        persona_name = metadata.get('persona_name', 'Unknown')
        
        # Get metrics
        final_metrics = data.get('final_metrics', {})
        
        # Collect individual scores if available
        aggregate_scores = data.get('aggregate_scores', {})
        if aggregate_scores:
            # Initialize collectors if first micro cluster
            if not all_metrics:
                for metric_name in aggregate_scores.keys():
                    all_metrics[metric_name] = []
            
            # Add scores from this micro cluster
            for metric_name, scores in aggregate_scores.items():
                if isinstance(scores, list) and metric_name in all_metrics:
                    all_metrics[metric_name].extend(scores)
        
        # Count reviews and users
        user_predictions = data.get('user_predictions', {})
        cluster_reviews = sum(len(reviews) if isinstance(reviews, list) else 0 
                            for reviews in user_predictions.values())
        cluster_users = len(user_predictions)
        
        total_reviews += cluster_reviews
        total_users += cluster_users
        
        micro_clusters_processed.append({
            'micro_cluster_id': micro_cluster_id,
            'persona_name': persona_name,
            'reviews': cluster_reviews,
            'users': cluster_users,
            'final_metrics': final_metrics
        })
    
    if not micro_clusters_processed:
        print(f"  - WARNING: No valid micro clusters found in {cluster_output_dir}")
        return None
    
    # Calculate aggregate metrics (matching grand_summary structure)
    final_summary = {}
    if all_metrics:
        for metric_name, scores in all_metrics.items():
            if scores:
                final_summary[metric_name] = {
                    'mean': float(np.mean(scores)),
                    'std': float(np.std(scores)),
                    'count': len(scores)
                }
    
    # Build aggregate summary
    aggregate_summary = {
        'model_type_used': 'enhanced_delta_corrected',
        'cluster_name': cluster_id,
        'final_summary': final_summary,
        'micro_clusters_included': micro_clusters_processed
    }
    
    # Save aggregate summary
    summary_file = os.path.join(cluster_output_dir, f"grand_summary_enhanced_delta_corrected.json")
    io_utils.save_json_file(aggregate_summary, summary_file)
    print(f"  ✅ Aggregate summary saved: {summary_file}")
    print(f"     Micro Clusters: {len(micro_clusters_processed)} | Reviews: {total_reviews}")
    
    return aggregate_summary