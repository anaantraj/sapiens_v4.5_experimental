"""
Logprobs-based processing for SGO training.
This module handles review generation and theme classification using logprobs extraction.
Processes reviews individually (not in batches) for logprobs mode.
"""
import time
import json_repair
import math
import sys
from pathlib import Path
from openai import RateLimitError, APIError

# Set up package structure for sgo_training imports
scripts_dir = Path(__file__).parent.parent
if str(scripts_dir) not in sys.path:
    sys.path.insert(0, str(scripts_dir))
import _package_setup

from sgo_training.config import settings
from sgo_training.utils.cost_tracker import track_api_call, extract_token_usage


def get_theme_logprobs_without_context(product_description: str, review_text: str, theme: str, client, theme_model_name: str, max_retries: int) -> dict:
    """
    Get yes/no logprobs for a single theme using ONLY product_description and review_text.
    NO persona context is included - this is a pure review-based classification.
    
    Similar to get_topic_logprobs in topic_classification_fixed.py.
    """
    system_msg = "You are a precise theme classifier. Answer only Yes or No."
    user_msg = f'Analyze this product review and determine if it discusses the theme "{theme}".\n\nProduct Description: {product_description}\n\nReview: "{review_text}"\n\nDoes this review discuss the theme "{theme}"?\nAnswer with ONLY "Yes" or "No".'
    for attempt in range(max_retries):
        try:
            time.sleep(settings.API_RATE_LIMIT_DELAY)
            
            try:
                # Check if model uses max_completion_tokens (gpt-5) or max_tokens
                request_params = {
                    "model": theme_model_name,
                    "messages": [
                        {"role": "system", "content": system_msg},
                        {"role": "user", "content": user_msg}
                    ],
                    "temperature": 0,
                    "logprobs": True,
                    "top_logprobs": getattr(settings, 'TOP_LOGPROBS', 20)
                }
                
                # Use max_completion_tokens for gpt-5 models, max_tokens for others
                if "gpt-5" in theme_model_name.lower():
                    request_params["max_completion_tokens"] = 5
                else:
                    request_params["max_tokens"] = 5
                
                response = client.chat.completions.create(**request_params)
                
                # Track API call for cost calculation (get_theme_logprobs_without_context)
                input_tokens, output_tokens, cached_tokens = extract_token_usage(response)
                track_api_call(
                    model=theme_model_name,
                    call_type="theme_logprobs_without_context",
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    cached_input_tokens=cached_tokens,
                    prompt_length=len(system_msg) + len(user_msg),
                    metadata={"theme": theme, "attempt": attempt + 1, "function": "get_theme_logprobs_without_context"}
                )
                
                # Find logprob for "yes" and "no"
                # Priority: 1) Actual generated token (use its logprob directly), 2) Highest logprob from alternatives
                yes_logprob = -100.0
                no_logprob = -100.0
                yes_token = None
                no_token = None
                yes_found_in_main = False  # Track if we found yes from actual generated token
                no_found_in_main = False   # Track if we found no from actual generated token
                
                logprobs_data = response.choices[0].logprobs
                
                if logprobs_data and logprobs_data.content:
                    # First pass: Check main tokens (actual generated tokens) - use their logprob directly
                    for token_info in logprobs_data.content:
                        token_clean = token_info.token.lower().strip()
                        
                        if token_clean in ["yes", "y"]:
                            # This is the actual generated token - use its logprob directly
                            yes_logprob = token_info.logprob
                            yes_token = token_info.token
                            yes_found_in_main = True
                        elif token_clean in ["no", "n"]:
                            # This is the actual generated token - use its logprob directly
                            no_logprob = token_info.logprob
                            no_token = token_info.token
                            no_found_in_main = True
                    
                    # Second pass: Check alternatives (top_logprobs) only if we didn't find from main token
                    if not yes_found_in_main or not no_found_in_main:
                        for token_info in logprobs_data.content:
                            if token_info.top_logprobs:
                                for alt in token_info.top_logprobs:
                                    alt_clean = alt.token.lower().strip()
                                    
                                    if alt_clean in ["yes", "y"] and not yes_found_in_main:
                                        # Only update if we haven't found yes from main token
                                        if alt.logprob > yes_logprob:
                                            yes_logprob = alt.logprob
                                            yes_token = alt.token
                                    elif alt_clean in ["no", "n"] and not no_found_in_main:
                                        # Only update if we haven't found no from main token
                                        if alt.logprob > no_logprob:
                                            no_logprob = alt.logprob
                                            no_token = alt.token
                
                # Set defaults if not found
                if yes_logprob == -100.0:
                    yes_logprob = -10.0
                if no_logprob == -100.0:
                    no_logprob = -10.0
                
                return {
                    "yes": yes_logprob,
                    "no": no_logprob,
                    "token_yes": yes_token,
                    "token_no": no_token
                }
            
            except Exception as logprob_error:
                error_str = str(logprob_error)
                if "logprob" in error_str.lower() or "403" in error_str:
                    # Fallback to answer-based classification
                    fallback_params = {
                        "model": theme_model_name,
                        "messages": [
                            {"role": "system", "content": system_msg},
                            {"role": "user", "content": user_msg}
                        ],
                        "temperature": 0
                    }
                    
                    if "gpt-5" in theme_model_name.lower():
                        fallback_params["max_completion_tokens"] = 5
                    else:
                        fallback_params["max_tokens"] = 5
                    
                    response = client.chat.completions.create(**fallback_params)
                    
                    # Track fallback API call for cost calculation (get_theme_logprobs_without_context fallback)
                    input_tokens, output_tokens, cached_tokens = extract_token_usage(response)
                    track_api_call(
                        model=theme_model_name,
                        call_type="theme_logprobs_without_context_fallback",
                        input_tokens=input_tokens,
                        output_tokens=output_tokens,
                        cached_input_tokens=cached_tokens,
                        prompt_length=len(system_msg) + len(user_msg),
                        metadata={"theme": theme, "attempt": attempt + 1, "fallback": True, "function": "get_theme_logprobs_without_context"}
                    )
                    
                    answer_text = response.choices[0].message.content.strip().lower()
                    is_yes = "yes" in answer_text or answer_text.startswith("y")
                    
                    prob_yes = 0.9 if is_yes else 0.1
                    prob_no = 0.1 if is_yes else 0.9
                    
                    return {
                        "yes": math.log(prob_yes) if prob_yes > 0 else -10.0,
                        "no": math.log(prob_no) if prob_no > 0 else -10.0,
                        "token_yes": "Yes" if is_yes else None,
                        "token_no": "No" if not is_yes else None
                    }
                else:
                    raise
        
        except (APIError, RateLimitError) as e:
            if attempt < max_retries - 1:
                wait_time = (2 ** attempt) * 2
                print(f"  - WARNING: Theme logprob error (attempt {attempt+1}/{max_retries}) for '{theme}': {e}. Retrying in {wait_time}s...")
                time.sleep(wait_time)
            else:
                print(f"  - ERROR: Failed to get logprobs for theme '{theme}' after {max_retries} attempts: {e}")
                return {"yes": -10.0, "no": -10.0, "token_yes": None, "token_no": None}
        except Exception as e:
            print(f"  - ERROR: Unexpected error getting logprobs for theme '{theme}' (attempt {attempt+1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                time.sleep(5)
            else:
                return {"yes": -10.0, "no": -10.0, "token_yes": None, "token_no": None}
    
    return {"yes": -10.0, "no": -10.0, "token_yes": None, "token_no": None}


def get_theme_logprobs_with_context(full_context_prompt: str, review_text: str, theme: str, client, theme_model_name: str, max_retries: int) -> dict:
    """
    Get yes/no logprobs for a single theme using the FULL context (persona, user, category, product) 
    from the first LLM call, plus the generated review text.
    
    This is used in logprobs mode where we want to classify themes using the same context
    that was used to generate the review.
    """
    system_msg = "You are a precise theme classifier. Answer only Yes or No."
    user_msg = f"""{full_context_prompt}

---
**GENERATED REVIEW TEXT:**
{review_text}

---
**THEME CLASSIFICATION TASK:**

Based on the complete context above (persona characteristics, user traits, product description, past learnings, and the generated review text), determine if this review discusses the theme "{theme}".

Consider:
- The persona's priorities and concerns (from Micro Persona Characteristics)
- The user's specific traits (from User Characteristics)
- The product description and what aspects would be relevant
- The generated review text and what themes it actually addresses
- Past learnings about what this persona typically focuses on

Does this review discuss the theme "{theme}"?

Answer with ONLY "Yes" or "No"."""
    for attempt in range(max_retries):
        try:
            time.sleep(settings.API_RATE_LIMIT_DELAY)
            
            try:
                # Use max_tokens for theme classification (gpt-4o-mini uses max_tokens)
                request_params = {
                    "model": theme_model_name,
                    "messages": [
                        {"role": "system", "content": system_msg},
                        {"role": "user", "content": user_msg}
                    ],
                    "max_tokens": 5,
                    "temperature": 0,
                    "logprobs": True,
                    "top_logprobs": getattr(settings, 'TOP_LOGPROBS', 20)
                }
                
                response = client.chat.completions.create(**request_params)
                
                # Track API call for cost calculation (get_theme_logprobs_with_context)
                input_tokens, output_tokens, cached_tokens = extract_token_usage(response)
                track_api_call(
                    model=theme_model_name,
                    call_type="theme_logprobs_with_context",
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    cached_input_tokens=cached_tokens,
                    prompt_length=len(system_msg) + len(user_msg),
                    metadata={"theme": theme, "attempt": attempt + 1, "function": "get_theme_logprobs_with_context"}
                )
                
                # Find logprob for "yes" and "no"
                # Priority: 1) Actual generated token (use its logprob directly), 2) Highest logprob from alternatives
                yes_logprob = -100.0
                no_logprob = -100.0
                yes_token = None
                no_token = None
                yes_found_in_main = False  # Track if we found yes from actual generated token
                no_found_in_main = False   # Track if we found no from actual generated token
                
                logprobs_data = response.choices[0].logprobs
                
                if logprobs_data and logprobs_data.content:
                    # First pass: Check main tokens (actual generated tokens) - use their logprob directly
                    for token_info in logprobs_data.content:
                        token_clean = token_info.token.lower().strip()
                        
                        if token_clean in ["yes", "y"]:
                            # This is the actual generated token - use its logprob directly
                            yes_logprob = token_info.logprob
                            yes_token = token_info.token
                            yes_found_in_main = True
                        elif token_clean in ["no", "n"]:
                            # This is the actual generated token - use its logprob directly
                            no_logprob = token_info.logprob
                            no_token = token_info.token
                            no_found_in_main = True
                    
                    # Second pass: Check alternatives (top_logprobs) only if we didn't find from main token
                    if not yes_found_in_main or not no_found_in_main:
                        for token_info in logprobs_data.content:
                            if token_info.top_logprobs:
                                for alt in token_info.top_logprobs:
                                    alt_clean = alt.token.lower().strip()
                                    
                                    if alt_clean in ["yes", "y"] and not yes_found_in_main:
                                        # Only update if we haven't found yes from main token
                                        if alt.logprob > yes_logprob:
                                            yes_logprob = alt.logprob
                                            yes_token = alt.token
                                    elif alt_clean in ["no", "n"] and not no_found_in_main:
                                        # Only update if we haven't found no from main token
                                        if alt.logprob > no_logprob:
                                            no_logprob = alt.logprob
                                            no_token = alt.token
                
                # Set defaults if not found
                if yes_logprob == -100.0:
                    yes_logprob = -10.0
                if no_logprob == -100.0:
                    no_logprob = -10.0
                
                return {
                    "yes": yes_logprob,
                    "no": no_logprob,
                    "token_yes": yes_token,
                    "token_no": no_token
                }
            
            except Exception as logprob_error:
                error_str = str(logprob_error)
                if "logprob" in error_str.lower() or "403" in error_str:
                    # Fallback to answer-based classification
                    fallback_params = {
                        "model": theme_model_name,
                        "messages": [
                            {"role": "system", "content": system_msg},
                            {"role": "user", "content": user_msg}
                        ],
                        "max_tokens": 5,
                        "temperature": 0
                    }
                    
                    response = client.chat.completions.create(**fallback_params)
                    
                    # Track fallback API call for cost calculation (get_theme_logprobs_with_context fallback)
                    input_tokens, output_tokens, cached_tokens = extract_token_usage(response)
                    track_api_call(
                        model=theme_model_name,
                        call_type="theme_logprobs_with_context_fallback",
                        input_tokens=input_tokens,
                        output_tokens=output_tokens,
                        cached_input_tokens=cached_tokens,
                        prompt_length=len(system_msg) + len(user_msg),
                        metadata={"theme": theme, "attempt": attempt + 1, "fallback": True, "function": "get_theme_logprobs_with_context"}
                    )
                    
                    answer_text = response.choices[0].message.content.strip().lower()
                    is_yes = "yes" in answer_text or answer_text.startswith("y")
                    
                    prob_yes = 0.9 if is_yes else 0.1
                    prob_no = 0.1 if is_yes else 0.9
                    
                    return {
                        "yes": math.log(prob_yes) if prob_yes > 0 else -10.0,
                        "no": math.log(prob_no) if prob_no > 0 else -10.0,
                        "token_yes": "Yes" if is_yes else None,
                        "token_no": "No" if not is_yes else None
                    }
                else:
                    raise
        
        except (APIError, RateLimitError) as e:
            if attempt < max_retries - 1:
                wait_time = (2 ** attempt) * 2
                print(f"  - WARNING: Theme logprob error (attempt {attempt+1}/{max_retries}) for '{theme}': {e}. Retrying in {wait_time}s...")
                time.sleep(wait_time)
            else:
                print(f"  - ERROR: Failed to get logprobs for theme '{theme}' after {max_retries} attempts: {e}")
                return {"yes": -10.0, "no": -10.0, "token_yes": None, "token_no": None}
        except Exception as e:
            print(f"  - ERROR: Unexpected error getting logprobs for theme '{theme}' (attempt {attempt+1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                time.sleep(5)
            else:
                return {"yes": -10.0, "no": -10.0, "token_yes": None, "token_no": None}
    
    return {"yes": -10.0, "no": -10.0, "token_yes": None, "token_no": None}


def normalize_logprobs_softmax(theme_logprobs: dict) -> dict:
    """
    Normalize theme logprobs using softmax (similar to topic_classification_fixed.py).
    Takes a dict of {theme: {"yes": logprob, "no": logprob}} and returns normalized probabilities.
    """
    if not theme_logprobs:
        return {}
    
    # Extract yes logprobs for softmax
    logprob_values = [theme_logprobs[theme]["yes"] for theme in theme_logprobs.keys()]
    
    if not logprob_values:
        return {theme: 0.0 for theme in theme_logprobs.keys()}
    
    # Numerically stable softmax: exp(x - max(x)) / sum(exp(x - max(x)))
    max_logprob = max(logprob_values)
    exp_logprobs = [math.exp(logprob - max_logprob) for logprob in logprob_values]
    sum_exp = sum(exp_logprobs)
    
    if sum_exp > 0:
        softmax_probs = [exp_logprob / sum_exp for exp_logprob in exp_logprobs]
    else:
        softmax_probs = [1.0 / len(logprob_values)] * len(logprob_values)
    
    # Create normalized dictionary
    normalized = {
        theme: softmax_prob
        for theme, softmax_prob in zip(theme_logprobs.keys(), softmax_probs)
    }
    
    return normalized


def process_single_review_logprobs(
    review_context: str,
    product_description: str,
    category_themes: list,
    client,
    scoring_mode: str,
    max_retries: int = None
) -> dict:
    """
    Process a single review in logprobs mode:
    1. Generate review_text, rating, sentiment (no themes) using review_prediction_model
    2. Classify each theme separately using logprobs
    3. Normalize using softmax
    4. Return correction with predicted_themes (normalized) and theme_logprobs (raw)
    
    Args:
        review_context: Full context string (persona + user chars + product description) for this review
        product_description: Product description for this review
        category_themes: List of themes for this review's category
        client: OpenAI client
        scoring_mode: "logprobs" or "logprobs-without-persona-context"
        max_retries: Maximum retries for API calls
    
    Returns:
        Dict with: review_text, rating, sentiment, predicted_themes (normalized), theme_logprobs (raw)
    """
    if max_retries is None:
        max_retries = getattr(settings, 'MAX_RETRIES', 3)
    
    # Get model names from settings (loaded from config.yaml)
    # These should be configured in config.yaml under review_prediction_model and theme_prediction_model
    review_model = getattr(settings, 'REVIEW_PREDICTION_MODEL', None)
    theme_model = getattr(settings, 'THEME_PREDICTION_MODEL', None)
    
    if review_model is None:
        raise ValueError("REVIEW_PREDICTION_MODEL must be set in config.yaml (review_prediction_model)")
    if theme_model is None:
        raise ValueError("THEME_PREDICTION_MODEL must be set in config.yaml (theme_prediction_model)")
    
    # Step 1: Generate review (without themes) using review_prediction_model
    retry_delay = 2
    for attempt in range(max_retries):
        try:
            time.sleep(settings.API_RATE_LIMIT_DELAY)
            
            system_message = "You are a persona-based review prediction agent. Your response must be a single, valid JSON object. Follow the instructions in the user message carefully."
            
            request_params = {
                "model": review_model,
                "messages": [
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": review_context}
                ],
            }
            
            # Add response_format based on model (some models like o3 handle JSON naturally)
            # Check if model name contains "o3" to determine JSON handling
            if "o3" in review_model.lower():
                # o3 doesn't need response_format, it handles JSON naturally
                pass
            else:
                request_params["response_format"] = {"type": "json_object"}
                # Add max_tokens/temperature for non-o3 models if needed
                if hasattr(settings, 'MAX_TOKENS'):
                    request_params["max_tokens"] = getattr(settings, 'MAX_TOKENS', 2000)
                if hasattr(settings, 'TEMPERATURE'):
                    request_params["temperature"] = getattr(settings, 'TEMPERATURE', 0.7)
            
            response = client.chat.completions.create(**request_params)
            
            # Track API call for cost calculation
            input_tokens, output_tokens, cached_tokens = extract_token_usage(response)
            track_api_call(
                model=review_model,
                call_type="review_generation_logprobs",
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cached_input_tokens=cached_tokens,
                prompt_length=len(review_context),
                metadata={"scoring_mode": scoring_mode, "attempt": attempt + 1}
            )
            
            text = response.choices[0].message.content.strip()
            
            parsed = json_repair.loads(text)
            
            # Extract review data (should be a single review, not a batch)
            if "corrections" in parsed:
                # Batch format - take first correction
                correction = parsed["corrections"][0] if parsed["corrections"] else {}
            elif "review_text" in parsed:
                # Single review format
                correction = parsed
            else:
                raise ValueError(f"Unexpected response format: {list(parsed.keys())}")
            
            if not isinstance(correction, dict) or 'review_text' not in correction:
                raise ValueError(f"Missing review_text in correction: {correction}")
            
            review_text = correction.get('review_text', '')
            rating = correction.get('rating', 3)
            sentiment = correction.get('sentiment', 'Neutral')
            
            # Step 2: Classify each theme using logprobs
            if not category_themes:
                return {
                    'review_text': review_text,
                    'rating': rating,
                    'sentiment': sentiment,
                    'predicted_themes': {},
                    'theme_logprobs': {}
                }
            
            theme_logprobs = {}
            for theme in category_themes:
                if scoring_mode == "logprobs-without-persona-context":
                    logprob_result = get_theme_logprobs_without_context(
                        product_description, review_text, theme, client, theme_model, max_retries
                    )
                else:  # logprobs mode (with persona context)
                    logprob_result = get_theme_logprobs_with_context(
                        review_context, review_text, theme, client, theme_model, max_retries
                    )
                theme_logprobs[theme] = logprob_result
            
            # Normalize using softmax
            normalized_themes = normalize_logprobs_softmax(theme_logprobs)
            
            # Ensure all themes are present (default to 0.0 if missing)
            processed_themes = {}
            for theme in category_themes:
                processed_themes[theme] = normalized_themes.get(theme, 0.0)
            
            # Build final correction
            return {
                'review_text': review_text,
                'rating': rating,
                'sentiment': sentiment,
                'predicted_themes': processed_themes,
                'theme_logprobs': theme_logprobs  # Include raw logprobs for debugging/analysis
            }
            
        except Exception as e:
            if attempt < max_retries - 1:
                wait_time = retry_delay * (2 ** attempt)
                print(f"  - WARNING: Review generation failed (attempt {attempt+1}/{max_retries}): {e}. Retrying in {wait_time}s...")
                time.sleep(wait_time)
                continue
            else:
                print(f"  - ERROR: Review generation failed after {max_retries} attempts: {e}")
                import traceback
                traceback.print_exc()
                return {}
    
    return {}


def process_reviews_logprobs(
    reviews_data: list,
    client,
    scoring_mode: str
) -> list:
    """
    Process multiple reviews in logprobs mode (one at a time, not batched).
    
    Args:
        reviews_data: List of dicts, each containing:
            - review_context: Full context string for this review
            - product_description: Product description
            - category_themes: List of themes for this review's category
        client: OpenAI client
        scoring_mode: "logprobs" or "logprobs-without-persona-context"
    
    Returns:
        List of corrections (same format as process_single_review_logprobs)
    """
    results = []
    for i, review_data in enumerate(reviews_data):
        print(f"  - Processing review {i+1}/{len(reviews_data)} in logprobs mode...")
        result = process_single_review_logprobs(
            review_context=review_data['review_context'],
            product_description=review_data['product_description'],
            category_themes=review_data['category_themes'],
            client=client,
            scoring_mode=scoring_mode
        )
        results.append(result)
    
    return results

