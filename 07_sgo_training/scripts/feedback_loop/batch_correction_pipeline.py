import copy
import numpy as np
import sys
from pathlib import Path

# Set up package structure for sgo_training imports
scripts_dir = Path(__file__).parent.parent
if str(scripts_dir) not in sys.path:
    sys.path.insert(0, str(scripts_dir))
import _package_setup

from sgo_training.config import settings
from generate_synthetic_review_and_memory_analysis.prompt_generation_for_both_parts import (
    generate_part_a_batch_prompt, 
    generate_part_b_batch_prompt,
    generate_part_a_individual_prompt_logprobs
)
from sgo_training.utils import io_utils, logging_utils, journey_utils, embeddings
from calculate_behaviour_loss.weighted_topic_review_metric_calculation import (
    calculate_text_delta, 
    calculate_theme_delta,
    calculate_theme_delta_logprobs
)
# Import llm_client directly from local package
from llm import llm_client
# Import logprobs client for logprobs mode
from llm import llm_client_logprobs

def get_theme_delta(predicted_themes, actual_themes, all_themes=None):
    """
    Helper function that selects the appropriate theme delta calculation based on scoring mode.
    For logprobs mode: uses calculate_theme_delta_logprobs (JSD based)
    For confidence mode: uses calculate_theme_delta (original implementation)
    
    Args:Updating stopping criteria and removing recall metrics. Continuing with the remaining changes:
        predicted_themes: dict with theme names and probabilities
        actual_themes: dict or list of actual themes
        all_themes: Optional list of all possible themes (for JSD calculation in logprobs mode)
    """
    scoring_mode = getattr(settings, 'SCORING_MODE', 'confidence')
    if scoring_mode in ["logprobs", "logprobs-without-persona-context"]:
        return calculate_theme_delta_logprobs(predicted_themes, actual_themes, all_themes)
    else:
        return calculate_theme_delta(predicted_themes, actual_themes)

def get_user_characteristics_for_review(user_chars_data, user_id, category, category_mapping):
    """
    Extracts general and category-specific characteristics for a user.
    Now accepts 'category_mapping' explicitly to avoid global dependencies.
    """
    characteristics = []
    
    # 1. Validation
    if user_id not in user_chars_data: 
        return characteristics
        
    user_data = user_chars_data[user_id]
    
    # 2. General Characteristics
    llm_chars = user_data.get('llm_characteristics', {})
    if isinstance(llm_chars, dict):
        user_char = llm_chars.get('influencing_characteristics_summary', '')
        if user_char: 
            characteristics.append(f"[General Characteristics] {user_char}")
            
    # 3. Category Specific Characteristics
    if category:
        # Use the passed-in mapping instead of a global
        main_category = map_category_to_main_category(category, category_mapping)
        
        if main_category:
            category_chars = user_data.get('category_characteristics', {})
            if isinstance(category_chars, dict) and main_category in category_chars:
                cat_char = category_chars[main_category].get('influencing_characteristics_summary', '')
                if cat_char: 
                    characteristics.append(f"[{main_category} Specific] {cat_char}")
                    
    return characteristics

def map_category_to_main_category(category, mapping):
    """
    Helper to map sub-category to main category safely.
    Handles both nested structures (with 'category_to_main_mapping' key) and direct mappings.
    Also normalizes Fashion/Clothing variations to ensure they're treated as the same.
    
    Args:
        category: The sub-category to map (e.g., "Health & Personal Care")
        mapping: Either the full mapping dict with 'category_to_main_mapping' key, 
                 or the extracted category_to_main_mapping dict itself
    """
    if not category: 
        return None
    
    if not mapping:
        print(f"  ⚠️  WARNING: category_mapping is None or empty - cannot map '{category}'")
        return category
    
    # Handle nested structure (category_mapping has 'category_to_main_mapping' key)
    if isinstance(mapping, dict) and 'category_to_main_mapping' in mapping:
        main_category = mapping['category_to_main_mapping'].get(category, category)
    else:
        # Direct mapping structure (this is what load_category_mapping returns)
        main_category = mapping.get(category, category) if mapping else category
    
    # If direct lookup failed, try normalized variations
    if main_category == category and category not in mapping:
        # Try variations: with/without spaces, with/without underscores, case variations
        variations = [
            category,
            category.replace(' ', '_'),
            category.replace(' & ', '_and_'),
            category.replace(' & ', ' and '),
            category.replace(' & ', '_&_'),
            category.lower(),
            category.upper(),
            category.replace(' ', '_').lower(),
        ]
        
        for variant in variations:
            if variant in mapping:
                main_category = mapping[variant]
                print(f"  ✅ Found category mapping using variant '{variant}': '{category}' -> '{main_category}'")
                break
        
        # If still not found, log warning
        if main_category == category:
            print(f"  ⚠️  WARNING: Category '{category}' not found in category_mapping (tried {len(variations)} variations)")
            print(f"  ⚠️  Available categories in mapping: {list(mapping.keys())[:10]}{'...' if len(mapping) > 10 else ''}")
    
    # Normalize Fashion/Clothing variations - treat them as the same
    # Common variations: "Fashion", "Clothing", "Clothing Shoes & Jewelry", "Clothing_Shoes_and_Jewelry"
    main_category_lower = main_category.lower()
    if 'fashion' in main_category_lower or 'clothing_shoes_and_jewelry' in main_category_lower or 'jewelry' in main_category_lower:
        # Normalize to a standard form - use the first matching key from mapping if available
        if isinstance(mapping, dict) and 'category_to_main_mapping' in mapping:
            # Check if any mapped value matches our category
            for sub_cat, mapped_cat in mapping['category_to_main_mapping'].items():
                if 'fashion' in mapped_cat.lower() or 'clothing' in mapped_cat.lower() or 'jewelry' in mapped_cat.lower():
                    # Use the mapped category as the standard form
                    main_category = mapped_cat
                    break
    
    return main_category

def process_part_a_logprobs_mode(
    reviews_needing_correction,
    baseline_persona,
    persona_name,
    temp_persona_memory,
    batch_analyses,
    user_chars_data,
    category_themes_map,
    category_mapping,
    client
):
    """
    Process Part A in logprobs mode - handles reviews individually (not in batch).
    This is a separate function that doesn't modify existing confidence mode code.
    """
    scoring_mode = getattr(settings, 'SCORING_MODE', 'logprobs')
    corrections = []
    
    for review_info in reviews_needing_correction:
        review_data = review_info.get('data', {})
        user_id = review_info.get('user_id', 'unknown')
        category = review_data.get('category', '')
        
        # Get user characteristics for this review
        user_chars = get_user_characteristics_for_review(user_chars_data, user_id, category, category_mapping)
        user_chars_text = ' '.join(user_chars) if isinstance(user_chars, list) else (user_chars or 'None')
        
        # Get themes for this review's category
        category_themes = []
        if category and category_themes_map:
            # Map category to main category
            if isinstance(category_mapping, dict) and 'category_to_main_mapping' in category_mapping:
                main_category = category_mapping['category_to_main_mapping'].get(category, category)
            else:
                main_category = category_mapping.get(category, category) if category_mapping else category
            
            # Get themes for this category
            category_themes = category_themes_map.get(main_category, [])
            if not category_themes:
                # Try normalized variations
                main_cat_normalized = main_category.replace(' & ', '_and_').replace('&', 'and').replace(' ', '_').replace('-', '_').lower()
                for key in category_themes_map.keys():
                    key_normalized = key.replace(' & ', '_and_').replace('&', 'and').replace(' ', '_').replace('-', '_').lower()
                    if key_normalized == main_cat_normalized:
                        category_themes = category_themes_map[key]
                        break
        
        # Get product description
        product_desc = review_data.get('product_description', '') or review_data.get('prediction', {}).get('product_description', '')
        
        # Generate individual prompt for this review
        review_context = generate_part_a_individual_prompt_logprobs(
            baseline_persona,
            user_chars_text,
            persona_name,
            review_data,
            temp_persona_memory,
            batch_motives=batch_analyses,
            category_mapping=category_mapping
        )
        
        # Process this single review using logprobs
        result = llm_client_logprobs.process_single_review_logprobs(
            review_context=review_context,
            product_description=product_desc,
            category_themes=category_themes,
            client=client,
            scoring_mode=scoring_mode
        )
        corrections.append(result)
    
    return corrections

def process_single_batch(
    batch_reviews, batch_index, iteration_number, baseline_persona, persona_name,
    persona_memory, user_chars_data, category_themes_map,
    embeddings_cache, client, batch_analyses_log, batch_analyses_file, all_batch_analyses,
    fixed_data, journey_log, category_mapping, lock  # Added lock and category_mapping
):
    """
    Process a single batch of reviews. This function can be called in parallel.
    Returns: (batch_index, result_dict, batch_analyses_new, analyses_to_revoke, failed_predictions)
    where result_dict contains: {'batch_results': [...], 'still_failing': set(...), 'updates': [...]}
    and failed_predictions is a list of review entries that failed after all retries
    """
    try:
        # Extract delta info for this batch
        batch_initial_deltas = [r['delta_info'] for r in batch_reviews]
        reviews_needing_correction = batch_reviews
        
        average_batch_theme_delta = np.mean([d['theme'] for d in batch_initial_deltas])
        print(f"  [Batch {batch_index + 1}] Average Theme Delta: {average_batch_theme_delta:.4f}")
        print(f"  [Batch {batch_index + 1}] Processing {len(reviews_needing_correction)} reviews")
        
        # Get Category for this batch
        batch_data_for_llm = [review['data'] for review in reviews_needing_correction]
        current_batch_size = len(batch_data_for_llm)
        
        try:
            # Get category from first review (batch should have same category for all reviews)
            product_category = batch_data_for_llm[0].get('category')
            
            # Validate that all reviews in batch have the same category
            categories_in_batch = set()
            for review_data in batch_data_for_llm:
                cat = review_data.get('category')
                if cat:
                    categories_in_batch.add(cat)
            
            if len(categories_in_batch) > 1:
                print(f"  [Batch {batch_index + 1}] ⚠️  WARNING: Batch contains reviews from multiple categories: {categories_in_batch}")
                print(f"  [Batch {batch_index + 1}]     This may cause incorrect theme assignment. Consider grouping by category.")
            
            # DEBUG: Log category mapping attempt
            if batch_index == 0:  # Only log for first batch to avoid spam
                print(f"  [Batch {batch_index + 1}] DEBUG: Category mapping check")
                print(f"  [Batch {batch_index + 1}]   Product category: '{product_category}'")
                print(f"  [Batch {batch_index + 1}]   Category mapping type: {type(category_mapping)}")
                if isinstance(category_mapping, dict):
                    print(f"  [Batch {batch_index + 1}]   Category mapping keys count: {len(category_mapping)}")
                    # Check if product_category is in mapping
                    if product_category in category_mapping:
                        print(f"  [Batch {batch_index + 1}]   ✅ Found '{product_category}' in category_mapping -> '{category_mapping[product_category]}'")
                    else:
                        print(f"  [Batch {batch_index + 1}]   ❌ '{product_category}' NOT found in category_mapping")
                        # Show sample keys
                        sample_keys = [k for k in category_mapping.keys() if 'health' in k.lower() or 'personal' in k.lower() or 'care' in k.lower()][:5]
                        if sample_keys:
                            print(f"  [Batch {batch_index + 1}]   Similar keys found: {sample_keys}")
                        else:
                            print(f"  [Batch {batch_index + 1}]   Sample mapping keys: {list(category_mapping.keys())[:10]}")
            
            main_category = map_category_to_main_category(product_category, category_mapping)
            
            if batch_index == 0:  # Only log for first batch
                print(f"  [Batch {batch_index + 1}]   Mapped main category: '{main_category}'")
            
            # Try multiple lookup strategies for themes
            # Use exact format from category mapping file (e.g., "Health & Personal Care")
            relevant_themes = category_themes_map.get(main_category, [])
            
            # If not found, try with spaces replaced by underscores and ampersand replaced by "and"
            # This handles: "Health & Personal Care" -> "Health_and_Personal_Care"
            if not relevant_themes:
                main_cat_normalized = main_category.replace(' & ', '_and_').replace(' ', '_').replace('-', '_')
                relevant_themes = category_themes_map.get(main_cat_normalized, [])
                if relevant_themes:
                    print(f"  [Batch {batch_index + 1}] DEBUG: Found themes using normalized key: '{main_cat_normalized}' (from '{main_category}')")
            
            # Also try the fully normalized lowercase version (as stored in theme_map_normalized)
            # This matches the normalization done in cluster_processor.py
            if not relevant_themes:
                main_cat_normalized_lower = main_category.replace(' & ', '_and_').replace('&', 'and').replace(' ', '_').replace('-', '_').lower()
                relevant_themes = category_themes_map.get(main_cat_normalized_lower, [])
                if relevant_themes:
                    print(f"  [Batch {batch_index + 1}] DEBUG: Found themes using fully normalized key: '{main_cat_normalized_lower}' (from '{main_category}')")
            
            # Special handling for Fashion/Clothing categories - treat them as the same
            if not relevant_themes and ('fashion' in main_category.lower() or 'clothing' in main_category.lower() or 'jewelry' in main_category.lower()):
                # Try common variations (Fashion and Clothing are the same category)
                fashion_variations = [
                    'Fashion', 
                    'Clothing', 
                    'Clothing Shoes & Jewelry', 
                    'Clothing_Shoes_and_Jewelry', 
                    'Clothing_Shoes_&_Jewelry',
                    'Clothing Shoes and Jewelry'
                ]
                # Also try case-insensitive and normalized versions
                for cat_key, themes in category_themes_map.items():
                    cat_key_lower = cat_key.lower()
                    if any(var.lower() in cat_key_lower or cat_key_lower in var.lower() for var in fashion_variations):
                        relevant_themes = themes
                        print(f"  [Batch {batch_index + 1}] DEBUG: Found themes using Fashion/Clothing match: '{cat_key}' (main_category: '{main_category}')")
                        break
                
                # If still not found, try exact matches
                if not relevant_themes:
                    for var in fashion_variations:
                        if var in category_themes_map:
                            relevant_themes = category_themes_map[var]
                            print(f"  [Batch {batch_index + 1}] DEBUG: Found themes using Fashion variation: '{var}'")
                            break
            
            if not relevant_themes:
                # Debug: Show what we tried
                normalized1 = main_category.replace(' ', '_').replace('&', 'and')
                normalized2 = main_category.replace(' ', '_').replace('-', '_').replace('&', 'and').lower()
                print(f"  [Batch {batch_index + 1}] ⚠️  CRITICAL WARNING: No themes found for main category '{main_category}' (product_category: '{product_category}')")
                print(f"  [Batch {batch_index + 1}]     Tried keys: '{main_category}', '{normalized1}', '{normalized2}'")
                print(f"  [Batch {batch_index + 1}]     Available categories in theme map: {list(category_themes_map.keys())[:10]}")
                # Check if any key contains "health" and "personal" and "care"
                health_keys = [k for k in category_themes_map.keys() if 'health' in k.lower() and 'personal' in k.lower() and 'care' in k.lower()]
                if health_keys:
                    print(f"  [Batch {batch_index + 1}]     Found potential Health & Personal Care keys: {health_keys}")
                print(f"  [Batch {batch_index + 1}]     This will cause LLM to return empty predicted_themes!")
                print(f"  [Batch {batch_index + 1}]     ACTION: Cannot proceed without themes - skipping batch")
                return None, None, [], []
            else:
                print(f"  [Batch {batch_index + 1}] DEBUG: Found {len(relevant_themes)} relevant themes for category '{main_category}': {relevant_themes[:3]}...")
        except IndexError:
            print(f"  [Batch {batch_index + 1}] ERROR: Empty batch, skipping")
            return None, None, [], []
        
        # STEP 1: BATCH ANALYSIS (Part B)
        print(f"  [Batch {batch_index + 1}] Running Part B (Root Cause Analysis)...")
        try:
            prompt_b_batch = generate_part_b_batch_prompt(
                baseline_persona,
                persona_name,
                reviews_needing_correction,
                persona_memory,
                user_chars_data=user_chars_data,
                iteration_number=iteration_number
            )
        except FileNotFoundError as e:
            # Critical error: prompt file not found - stop execution immediately
            error_msg = f"CRITICAL ERROR in Batch {batch_index + 1}: {str(e)}"
            print(f"  [Batch {batch_index + 1}] {error_msg}")
            logging_utils.log_thread_safe(error_msg, 'error')
            # Re-raise to stop execution
            raise SystemExit(f"CRITICAL: Prompt file not found. Stopping execution. {str(e)}")
        except Exception as e:
            # Other critical errors should also stop execution
            error_msg = f"CRITICAL ERROR in Batch {batch_index + 1}: {str(e)}"
            print(f"  [Batch {batch_index + 1}] {error_msg}")
            logging_utils.log_thread_safe(error_msg, 'error')
            raise SystemExit(f"CRITICAL: Error generating prompt. Stopping execution. {str(e)}")
        
        batch_analyses = llm_client.process_part_b_batch(prompt_b_batch, client, current_batch_size)
        
        # Save batch analyses (thread-safe)
        batch_key = f"iteration_{iteration_number}_batch_{batch_index+1}"
        with lock:
            batch_analyses_log[batch_key] = {
                'reviews': [r['key'] for r in reviews_needing_correction],
                'analyses': batch_analyses,
                'category': product_category
            }
            # Note: In real parallel processing, writing to file inside lock might be slow
            # Ideally, aggregate in memory and write once, but following original logic:
            io_utils.save_json_file(batch_analyses_log, batch_analyses_file)
        
        # Create full memory for Part A (all past analyses + current batch)
        temp_batch_analyses = all_batch_analyses + batch_analyses
        temp_persona_memory = {
            persona_name: {
                'batch_analyses': temp_batch_analyses
            }
        }
        
        # STEP 2: CORRECTION (Part A) - Characteristic extraction removed
        print(f"  [Batch {batch_index + 1}] Running Part A (Correction)...")
        batch_data_llm = [r['data'] for r in reviews_needing_correction]
                # DEBUG: Print inputs for first 3 reviews
        print(f"\n  [Batch {batch_index + 1}] {'='*70}")
        print(f"  [Batch {batch_index + 1}] DEBUG: INPUTS FOR FIRST 3 REVIEWS")
        print(f"  [Batch {batch_index + 1}] {'='*70}")
        for debug_idx in range(min(3, len(reviews_needing_correction))):
            review_info = reviews_needing_correction[debug_idx]
            review_key = review_info['key']
            original_data = review_info['data']
            print(f"\n  [Batch {batch_index + 1}] Review {debug_idx + 1}: {review_key}")
            print(f"  [Batch {batch_index + 1}]   INPUT - Original Prediction:")
            print(f"  [Batch {batch_index + 1}]     Review Text: {original_data['prediction'].get('review_text', '')[:200]}...")
            print(f"  [Batch {batch_index + 1}]     Rating: {original_data['prediction'].get('rating', 'N/A')}")
            print(f"  [Batch {batch_index + 1}]     Sentiment: {original_data['prediction'].get('sentiment', 'N/A')}")
            # Handle both dict and list formats for predicted_themes
            pred_themes = original_data['prediction'].get('predicted_themes', {})
            if isinstance(pred_themes, dict):
                theme_keys = list(pred_themes.keys())[:5]
            elif isinstance(pred_themes, list):
                theme_keys = pred_themes[:5]
            else:
                theme_keys = []
            print(f"  [Batch {batch_index + 1}]     Predicted Themes: {theme_keys if theme_keys else '[] (empty or missing)'}")
            print(f"  [Batch {batch_index + 1}]   INPUT - Actual Review:")
            print(f"  [Batch {batch_index + 1}]     Review Text: {original_data['actual'].get('review_text', '')[:200]}...")
            print(f"  [Batch {batch_index + 1}]     Rating: {original_data['actual'].get('rating', 'N/A')}")
            print(f"  [Batch {batch_index + 1}]     Sentiment: {original_data['actual'].get('sentiment', 'N/A')}")
            # Handle both dict (logprobs) and list (confidence) formats
            actual_themes = original_data['actual'].get('predicted_themes', [])
            if isinstance(actual_themes, dict):
                actual_themes_display = list(actual_themes.keys())[:5]
            elif isinstance(actual_themes, list):
                actual_themes_display = actual_themes[:5]
            else:
                actual_themes_display = []
            print(f"  [Batch {batch_index + 1}]     Actual Themes: {actual_themes_display}")

        
        # DEBUG: Print user characteristics for first 3 reviews
        if batch_index == 0:  # Only print for first batch
            print(f"\n  [Batch {batch_index + 1}] {'='*70}")
            print(f"  [Batch {batch_index + 1}] DEBUG: USER CHARACTERISTICS FOR FIRST 3 REVIEWS")
            print(f"  [Batch {batch_index + 1}] {'='*70}")
            print(f"  [Batch {batch_index + 1}] DEBUG: user_chars_data type: {type(user_chars_data)}, length: {len(user_chars_data) if isinstance(user_chars_data, dict) else 'N/A'}")
            if isinstance(user_chars_data, dict) and len(user_chars_data) > 0:
                sample_user_ids = list(user_chars_data.keys())[:3]
                print(f"  [Batch {batch_index + 1}] DEBUG: Sample user IDs in user_chars_data: {sample_user_ids}")
            for debug_idx in range(min(3, len(reviews_needing_correction))):
                review_info = reviews_needing_correction[debug_idx]
                user_id = review_info['user_id']
                review_data = review_info['data']
                review_category = review_data.get('category', product_category)
                print(f"\n  [Batch {batch_index + 1}] Review {debug_idx + 1} - User: {user_id}, Category: {review_category}")
                print(f"  [Batch {batch_index + 1}]   User ID in user_chars_data: {user_id in user_chars_data if isinstance(user_chars_data, dict) else False}")
                user_chars = get_user_characteristics_for_review(user_chars_data, user_id, review_category, category_mapping)
                if user_chars:
                    for char in user_chars:
                        print(f"  [Batch {batch_index + 1}]   {char}")
                else:
                    print(f"  [Batch {batch_index + 1}]   No user characteristics found")
                    if isinstance(user_chars_data, dict) and user_id in user_chars_data:
                        user_data = user_chars_data[user_id]
                        print(f"  [Batch {batch_index + 1}]   DEBUG: User data keys: {list(user_data.keys()) if isinstance(user_data, dict) else 'N/A'}")
                        llm_chars = user_data.get('llm_characteristics', {}) if isinstance(user_data, dict) else {}
                        print(f"  [Batch {batch_index + 1}]   DEBUG: llm_characteristics type: {type(llm_chars)}, keys: {list(llm_chars.keys()) if isinstance(llm_chars, dict) else 'N/A'}")
        
        # Generate Part A prompt
        # Pass full review info (with user_id) so prompt can include per-review user characteristics
        # Check scoring mode to determine if we need to modify prompt (remove themes section for logprobs mode)
        scoring_mode = getattr(settings, 'SCORING_MODE', 'confidence')
        
        # Check if we should use logprobs mode - if so, call separate function
        if scoring_mode in ["logprobs", "logprobs-without-persona-context"]:
            corrections = process_part_a_logprobs_mode(
                reviews_needing_correction,
                baseline_persona,
                persona_name,
                temp_persona_memory,
                batch_analyses,
                user_chars_data,
                category_themes_map,
                category_mapping,
                client
            )
        else:
            # Original confidence mode - keep existing code unchanged
            prompt_a = generate_part_a_batch_prompt(
                baseline_persona,
                None,  # No aggregated user chars - will be per-review instead
                persona_name,
                reviews_needing_correction,  # Pass full review info, not just data
                temp_persona_memory,
                relevant_themes,
                batch_motives=batch_analyses,
                user_chars_data=user_chars_data,  # Pass for per-review extraction
                category_mapping=category_mapping,  # Pass for category-specific chars
                scoring_mode=scoring_mode  # Pass scoring mode to prompt generation
            )
            
            # Pass required_themes as guidance only (not strict validation) - LLM will return themes it thinks are relevant
            corrections = llm_client.process_part_a_batch(
                prompt_a, 
                client, 
                len(reviews_needing_correction),
                required_themes=None  # Don't enforce strict theme validation - let LLM decide relevant themes
            )
        
        # STEP 4: VALIDATION & COMMIT
        batch_results = []
        analyses_to_revoke = []
        still_failing = set()
        failed_predictions = []  # Initialize list for LLM failures
        
        # Prepare embedding calculations (parallel)
        embedding_tasks = []
        for idx, review_info in enumerate(reviews_needing_correction):
            review_key = review_info['key']
            original_prediction_data = review_info['data']
            improved_prediction = corrections[idx]
            
            if improved_prediction and 'review_text' in improved_prediction:
                # Add embedding tasks
                embedding_tasks.append((
                    original_prediction_data['actual']['review_text'],
                    review_key,
                    'actual'
                ))
                embedding_tasks.append((
                    improved_prediction['review_text'],
                    review_key,
                    f'corrected_iter_{iteration_number}'
                ))
        
        # Compute embeddings in parallel
        if settings.PARALLEL_EMBEDDINGS and embedding_tasks:
            # print(f"  [Batch {batch_index + 1}] Computing {len(embedding_tasks)} embeddings in parallel...")
            print(f"  [Batch {batch_index + 1}] Computing {len(embedding_tasks)} embeddings in parallel...")

            embedding_results = embeddings.compute_embeddings_parallel(embedding_tasks, client, embeddings_cache, lock=lock)
            # Update cache with lock to prevent race conditions
            with lock:
                embeddings_cache.update(embedding_results)

        
        print(f"\n  [Batch {batch_index + 1}] {'='*70}")
        print(f"  [Batch {batch_index + 1}] DETAILED REVIEW-BY-REVIEW VALIDATION")
        print(f"  [Batch {batch_index + 1}] {'='*70}")
        valid_updates = []

        
        # Process validation
        
        for idx, review_info in enumerate(reviews_needing_correction):
            user_id = review_info['user_id']
            review_idx = review_info['review_idx']
            original_prediction_data = review_info['data']
            review_key = review_info['key']
            
            improved_prediction = corrections[idx]
            missed_motive_analysis = batch_analyses[idx]
            
            # Get initial deltas for this review (needed for journey_log entry)
            initial_deltas_for_review = next(
                d for d in batch_initial_deltas
                if d['user_id'] == user_id and d.get('review_idx') == review_idx
            )
            initial_overall_delta = initial_deltas_for_review['overall']
            
            # Debug: Log what we received (first 2 only to avoid spam)
            if idx < 2:
                print(f"  [Batch {batch_index + 1}]     DEBUG: Correction {idx} type: {type(improved_prediction)}")
                if isinstance(improved_prediction, dict):
                    print(f"  [Batch {batch_index + 1}]     DEBUG: Correction {idx} keys: {list(improved_prediction.keys())}")
                    has_pred_themes = 'predicted_themes' in improved_prediction
                    pred_themes_val = improved_prediction.get('predicted_themes', 'MISSING')
                    print(f"  [Batch {batch_index + 1}]     DEBUG: predicted_themes present: {has_pred_themes}, type: {type(pred_themes_val).__name__}, value preview: {str(pred_themes_val)[:100] if has_pred_themes else 'N/A'}")
            
            # Ensure journey_log entry exists even for failed corrections (for tracking)
            journey_utils.ensure_journey_log_entry(
                journey_log,
                review_key,
                user_id,
                review_idx,
                original_prediction_data,
                {
                    'text': initial_deltas_for_review['text'],
                    'theme': initial_deltas_for_review['theme'],
                    'overall': initial_overall_delta
                },
                initial_prediction_data=original_prediction_data.get('prediction'),
                lock=lock
            )
            
            # Check if LLM call failed (empty dict means validation failure or API failure after retries)
            # Note: Empty dict {} can mean either:
            # 1. LLM API call failed after all retries (network/API error)
            # 2. LLM returned response but validation failed (missing themes, invalid format, etc.)
            if not improved_prediction or (isinstance(improved_prediction, dict) and not improved_prediction):
                # Extract specific failure reason if available (from service.py validation)
                failure_reason = 'validation_failed_or_api_failed'
                if isinstance(improved_prediction, dict) and '_failure_reason' in improved_prediction:
                    failure_reason = improved_prediction['_failure_reason']
                    # Remove the internal marker
                    improved_prediction.pop('_failure_reason', None)
                
                print(f"  [Batch {batch_index + 1}] ❌ VALIDATION/API FAILURE: {review_key}")
                print(f"  [Batch {batch_index + 1}]     STATUS: Correction is empty dict - could be validation failure or API failure")
                print(f"  [Batch {batch_index + 1}]     FAILURE REASON: {failure_reason}")
                print(f"  [Batch {batch_index + 1}]     ACTION: Will save to failed predictions file, excluding from main output")
                
                # Store full review data for failed predictions file
                failed_predictions.append({
                    'review_key': review_key,
                    'user_id': user_id,
                    'review_idx': review_idx,
                    'iteration': iteration_number,
                    'failure_reason': failure_reason,  # Specific reason from validation
                    'review_data': copy.deepcopy(original_prediction_data)  # Full review data
                })
                
                # Log failed correction attempt in journey (for tracking)
                with lock:
                    journey_log[review_key]['correction_journey'].append({
                        "iteration": iteration_number,
                        "correction_status": "failed",
                        "failure_reason": failure_reason,  # Use specific reason
                        "corrected_prediction_data": None,
                        "new_text_delta": None,
                        "new_theme_delta": None,
                        "new_overall_delta": None,
                        "analysis_used": None
                    })
                
                still_failing.add(review_key)
                continue
            
            if not isinstance(improved_prediction, dict) or 'review_text' not in improved_prediction:
                print(f"  [Batch {batch_index + 1}] ⚠️  REVIEW {idx+1}/{len(reviews_needing_correction)}: {review_key}")
                print(f"  [Batch {batch_index + 1}]     STATUS: INVALID CORRECTION - No review_text in response")
                print(f"  [Batch {batch_index + 1}]     DEBUG: improved_prediction type: {type(improved_prediction)}, is_dict: {isinstance(improved_prediction, dict)}")
                if isinstance(improved_prediction, dict):
                    print(f"  [Batch {batch_index + 1}]     DEBUG: keys: {list(improved_prediction.keys())}")
                print(f"  [Batch {batch_index + 1}]     ACTION: Keeping original prediction, will re-process")

                failure_reason = "invalid_correction_no_review_text"
                
                # Store full review data for failed predictions file
                failed_predictions.append({
                    'review_key': review_key,
                    'user_id': user_id,
                    'review_idx': review_idx,
                    'iteration': iteration_number,
                    'failure_reason': failure_reason,
                    'review_data': copy.deepcopy(original_prediction_data)
                })

                # Log failed correction attempt in journey
                journey_utils.ensure_journey_log_entry(
                    journey_log,
                    review_key,
                    user_id,
                    review_idx,
                    original_prediction_data,
                    {
                        'text': initial_deltas_for_review['text'],
                        'theme': initial_deltas_for_review['theme'],
                        'overall': initial_overall_delta
                    },
                    initial_prediction_data=original_prediction_data.get('prediction'),
                    lock=lock
                )
                with lock:
                    journey_log[review_key]['correction_journey'].append({
                        "iteration": iteration_number,
                        "correction_status": "failed",
                        "failure_reason": failure_reason,
                        "corrected_prediction_data": None,
                        "new_text_delta": None,
                        "new_theme_delta": None,
                        "new_overall_delta": None,
                        "analysis_used": None
                    })

                still_failing.add(review_key)
                continue
            
            # Validate that predicted_themes is present and contains all required themes
            # This should NEVER happen if validation in service.process_part_a_batch works correctly
            pred_themes = improved_prediction.get('predicted_themes', {})
            if not pred_themes or not isinstance(pred_themes, dict):
                print(f"  [Batch {batch_index + 1}]     ❌ CRITICAL ERROR: Correction missing predicted_themes dict!")
                print(f"  [Batch {batch_index + 1}]     DEBUG: pred_themes value: {pred_themes}, type: {type(pred_themes)}")
                print(f"  [Batch {batch_index + 1}]     DEBUG: Correction keys: {list(improved_prediction.keys())}")
                print(f"  [Batch {batch_index + 1}]     DEBUG: This should have been caught by service.process_part_a_batch validation!")
                print(f"  [Batch {batch_index + 1}]     ACTION: Rejecting correction, will re-process")
                
                failure_reason = "missing_predicted_themes"
                
                # Store full review data for failed predictions file
                failed_predictions.append({
                    'review_key': review_key,
                    'user_id': user_id,
                    'review_idx': review_idx,
                    'iteration': iteration_number,
                    'failure_reason': failure_reason,
                    'review_data': copy.deepcopy(original_prediction_data)
                })
                
                # Log failed correction attempt in journey
                journey_utils.ensure_journey_log_entry(
                    journey_log,
                    review_key,
                    user_id,
                    review_idx,
                    original_prediction_data,
                    {
                        'text': initial_deltas_for_review['text'],
                        'theme': initial_deltas_for_review['theme'],
                        'overall': initial_overall_delta
                    },
                    initial_prediction_data=original_prediction_data.get('prediction'),
                    lock=lock
                )
                with lock:
                    journey_log[review_key]['correction_journey'].append({
                        "iteration": iteration_number,
                        "correction_status": "failed",
                        "failure_reason": failure_reason,
                        "corrected_prediction_data": None,
                        "new_text_delta": None,
                        "new_theme_delta": None,
                        "new_overall_delta": None,
                        "analysis_used": None
                    })
                
                still_failing.add(review_key)
                continue
            
            # Check if all required themes are present - REMOVED: Allow LLM to return themes it thinks are relevant
            # The LLM may not include all required themes if it determines they're not relevant to the review
            # We'll still calculate theme_delta which will show if themes are missing, but won't reject the correction
            
            # Update fixed_data (thread-safe access)
            # Note: This direct modification of shared state `fixed_data` relies on Python's GIL or explicit locks.
            # In a true multiprocessing setup, this wouldn't work. Threading is fine for I/O bound.
            # Ideally, return updates and apply in main thread.
            valid_updates.append({
                'user_id': user_id,
                'review_idx': review_idx,
                'prediction': improved_prediction
            })        
            # Get embeddings (from cache or compute) - thread-safe with lock
            actual_embedding = embeddings.get_or_compute_embedding(
                original_prediction_data['actual']['review_text'],
                review_key, 'actual', client, embeddings_cache, lock=lock
            )
            improved_embedding = embeddings.get_or_compute_embedding(
                improved_prediction['review_text'],
                review_key, f'corrected_iter_{iteration_number}', client, embeddings_cache, lock=lock
            )
            
            new_text_delta = calculate_text_delta(improved_embedding, actual_embedding)
            
            # Get category themes for JSD calculation (if logprobs mode)
            category_themes_list = None
            scoring_mode = getattr(settings, 'SCORING_MODE', 'confidence')
            if scoring_mode in ["logprobs", "logprobs-without-persona-context"]:
                review_data = original_prediction_data.get('data', {})
                category = review_data.get('category', '')
                if category and category_themes_map:
                    # Map to main category
                    if category_mapping and category:
                        if isinstance(category_mapping, dict) and 'category_to_main_mapping' in category_mapping:
                            main_category = category_mapping['category_to_main_mapping'].get(category, category)
                        else:
                            main_category = category_mapping.get(category, category)
                    else:
                        main_category = category
                    # Get themes for this category
                    category_themes_list = category_themes_map.get(main_category, [])
                    # Also try normalized category name
                    if not category_themes_list:
                        cat_normalized = main_category.replace(' & ', '_and_').replace('&', 'and').replace(' ', '_').replace('-', '_').lower()
                        for key in category_themes_map.keys():
                            if key.lower() == cat_normalized:
                                category_themes_list = category_themes_map[key]
                                break
            
            new_theme_delta = get_theme_delta(
                improved_prediction.get('predicted_themes', {}),
                original_prediction_data['actual'].get('predicted_themes', []),
                all_themes=category_themes_list
            )
            new_overall_delta = (new_text_delta * settings.WEIGHT_TEXT_DELTA) + (new_theme_delta * settings.WEIGHT_THEME_DELTA)
            
            # initial_deltas_for_review already calculated at the start of the loop (line 417)
            initial_theme_delta = initial_deltas_for_review['theme']
            initial_text_delta = initial_deltas_for_review['text']
            initial_overall_delta = initial_deltas_for_review['overall']
            
            theme_delta_change = new_theme_delta - initial_theme_delta
            overall_delta_change = new_overall_delta - initial_overall_delta
            theme_improvement_pct = ((initial_theme_delta - new_theme_delta) / initial_theme_delta * 100) if initial_theme_delta > 0 else 0.0

            
            
            # Track improvement/worsening
            if new_theme_delta < initial_theme_delta:
                improvement_status = "IMPROVED"
                status_icon = "✅"
            elif new_theme_delta > initial_theme_delta:
                improvement_status = "WORSE"
                status_icon = "❌"
                analyses_to_revoke.append({
                    'review_key': review_key,
                    'analysis': missed_motive_analysis,
                    'old_theme_delta': initial_theme_delta,
                    'new_theme_delta': new_theme_delta
                })
            else:
                improvement_status = "UNCHANGED"
                status_icon = "➡️"
            print(f"\n  [Batch {batch_index + 1}] {status_icon} REVIEW {idx+1}/{len(reviews_needing_correction)}: {review_key}")
            
            # DEBUG: Print outputs for first 3 reviews
            if idx < 3:
                print(f"  [Batch {batch_index + 1}]   OUTPUT - Corrected Prediction:")
                print(f"  [Batch {batch_index + 1}]     Review Text: {improved_prediction.get('review_text', '')[:200]}...")
                print(f"  [Batch {batch_index + 1}]     Rating: {improved_prediction.get('rating', 'N/A')}")
                print(f"  [Batch {batch_index + 1}]     Sentiment: {improved_prediction.get('sentiment', 'N/A')}")
                print(f"  [Batch {batch_index + 1}]     Predicted Themes: {list(improved_prediction.get('predicted_themes', {}).keys())[:5]}")
            
            print(f"  [Batch {batch_index + 1}]     ┌─ PREVIOUS DELTAS (Before Correction)")
            print(f"  [Batch {batch_index + 1}]     │  Theme Delta:    {initial_theme_delta:.4f}")
            print(f"  [Batch {batch_index + 1}]     │  Text Delta:     {initial_text_delta:.4f}")
            print(f"  [Batch {batch_index + 1}]     │  Overall Delta:   {initial_overall_delta:.4f}")
            print(f"  [Batch {batch_index + 1}]     ├─ CURRENT DELTAS (After Correction)")
            print(f"  [Batch {batch_index + 1}]     │  Theme Delta:    {new_theme_delta:.4f} ({theme_delta_change:+.4f})")
            print(f"  [Batch {batch_index + 1}]     │  Text Delta:     {new_text_delta:.4f} ({new_text_delta - initial_text_delta:+.4f})")
            print(f"  [Batch {batch_index + 1}]     │  Overall Delta:   {new_overall_delta:.4f} ({overall_delta_change:+.4f})")
            print(f"  [Batch {batch_index + 1}]     ├─ IMPROVEMENT METRICS")
            print(f"  [Batch {batch_index + 1}]     │  Theme Improvement: {theme_improvement_pct:+.2f}%")
            print(f"  [Batch {batch_index + 1}]     │  Status: {improvement_status}")
            if new_theme_delta == 0:
                print(f"  [Batch {batch_index + 1}]     │  🎯 PERFECT MATCH (delta=0) - No further correction needed")
            elif new_theme_delta <= settings.DELTA_THRESHOLD:
                print(f"  [Batch {batch_index + 1}]     │  ✓ At or below threshold ({settings.DELTA_THRESHOLD}) - Good improvement, stopping")
            else:
                print(f"  [Batch {batch_index + 1}]     │  ⚠️  Still above threshold ({settings.DELTA_THRESHOLD}) - Will re-process")
            if improvement_status == "WORSE":
                print(f"  [Batch {batch_index + 1}]     │  ⚠️  PREDICTION GOT WORSE - Memory will be revoked for this review")
            print(f"  [Batch {batch_index + 1}]     └─")

            
            
            # Journey log (thread-safe)
            journey_utils.ensure_journey_log_entry(
                journey_log,
                review_key,
                user_id,
                review_idx,
                original_prediction_data,
                {
                    'text': initial_deltas_for_review['text'],
                    'theme': initial_deltas_for_review['theme'],
                    'overall': initial_overall_delta
                },
                initial_prediction_data=original_prediction_data.get('prediction'),
                lock=lock
            )
            with lock:
                
                journey_log[review_key]['correction_journey'].append({
                    "iteration": iteration_number,
                    "corrected_prediction_data": copy.deepcopy(improved_prediction),
                    "new_text_delta": new_text_delta,
                    "new_theme_delta": new_theme_delta,
                    "new_overall_delta": new_overall_delta,
                    "analysis_used": missed_motive_analysis
                })
            
            batch_results.append({
                'review_key': review_key,
                'improvement_status': improvement_status,
                'initial_theme_delta': initial_theme_delta,
                'new_theme_delta': new_theme_delta,
                'initial_overall_delta': initial_overall_delta,
                'new_overall_delta': new_overall_delta,
                'initial_text_delta': initial_text_delta,
                'new_text_delta': new_text_delta,
                'theme_improvement_pct': theme_improvement_pct
            })
            
            # --- Re-processing Logic ---
            # Stopping criteria: theme_delta <= threshold means we're done
            if new_theme_delta == 0:
                print(f"  [Batch {batch_index + 1}]     ✓ OUTCOME: Corrected (Theme Delta is 0). No further processing needed.")
            elif new_theme_delta <= settings.DELTA_THRESHOLD:
                print(f"  [Batch {batch_index + 1}]     ✓ OUTCOME: At or below threshold ({settings.DELTA_THRESHOLD}). No further processing needed.")
            else:
                print(f"  [Batch {batch_index + 1}]     ⚠️  OUTCOME: Still failing (Theme Delta: {new_theme_delta:.4f} > {settings.DELTA_THRESHOLD}). Will re-process in next iteration.")
                still_failing.add(review_key)
            # --- End Re-processing Logic ---
        
        # Note: Batch summary is printed in cluster_processor.py after all batches complete
        # to ensure ordered output and avoid confusion in parallel processing

        
        # Calculate summary stats for print output if needed
        # ...

        return {
            'batch_results': batch_results,
            'still_failing': still_failing,
            'analyses_to_revoke': analyses_to_revoke,
            'updates': valid_updates,
            'failed_predictions': failed_predictions  # LLM failures after retries
        }, None, batch_analyses, analyses_to_revoke, failed_predictions
        
    except Exception as e:
        print(f"  [Batch {batch_index + 1}] ERROR: {e}")
        import traceback
        traceback.print_exc()
        return None, None, [], [], []