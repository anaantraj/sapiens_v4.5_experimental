import time
import json
import logging
import threading
import math
from openai import OpenAI, RateLimitError, APIError

class RateLimiter:
    def __init__(self, min_interval):
        self.lock = threading.Lock()
        self.last_request_time = 0.0
        self.min_interval = min_interval

    def wait(self):
        with self.lock:
            current_time = time.time()
            time_since_last = current_time - self.last_request_time
            if time_since_last < self.min_interval:
                time.sleep(self.min_interval - time_since_last)
            self.last_request_time = time.time()

def get_theme_logprobs_without_context(product_description: str, review_text: str, theme: str, client: OpenAI, rate_limiter: RateLimiter, theme_model_name: str, max_retries: int) -> dict:
    """
    Get yes/no logprobs for a single theme using ONLY product_description and review_text.
    NO persona context is included - this is a pure review-based classification.
    
    Similar to get_topic_logprobs in topic_classification_fixed.py.
    """
    system_msg = "You are a precise theme classifier. Answer only Yes or No."
    user_msg = f'Analyze this product review and determine if it discusses the theme "{theme}".\n\nProduct Description: {product_description}\n\nReview: "{review_text}"\n\nDoes this review discuss the theme "{theme}"?\nAnswer with ONLY "Yes" or "No".'
    for attempt in range(max_retries):
        try:
            rate_limiter.wait()
            
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
                    "top_logprobs": 5
                }
                
                # Use max_completion_tokens for gpt-5 models, max_tokens for others
                if "gpt-5" in theme_model_name.lower():
                    request_params["max_completion_tokens"] = 5
                else:
                    request_params["max_tokens"] = 5
                
                response = client.chat.completions.create(**request_params)
                
                # Find highest logprob for "yes" and "no"
                yes_logprob = -100.0
                no_logprob = -100.0
                yes_token = None
                no_token = None
                
                logprobs_data = response.choices[0].logprobs
                
                if logprobs_data and logprobs_data.content:
                    for token_info in logprobs_data.content:
                        token_clean = token_info.token.lower().strip()
                        
                        if token_clean in ["yes", "y"]:
                            if token_info.logprob > yes_logprob:
                                yes_logprob = token_info.logprob
                                yes_token = token_info.token
                        elif token_clean in ["no", "n"]:
                            if token_info.logprob > no_logprob:
                                no_logprob = token_info.logprob
                                no_token = token_info.token
                        
                        if token_info.top_logprobs:
                            for alt in token_info.top_logprobs:
                                alt_clean = alt.token.lower().strip()
                                
                                if alt_clean in ["yes", "y"]:
                                    if alt.logprob > yes_logprob:
                                        yes_logprob = alt.logprob
                                        yes_token = alt.token
                                elif alt_clean in ["no", "n"]:
                                    if alt.logprob > no_logprob:
                                        no_logprob = alt.logprob
                                        no_token = alt.token
                
                # Set defaults if not found - but log a warning
                if yes_logprob == -100.0:
                    response_text = response.choices[0].message.content[:100] if response.choices[0].message.content else 'N/A'
                    logging.warning(f"⚠️  Theme '{theme}': Could not find 'yes' token in logprobs. Using default -10.0. Response: {response_text}")
                    yes_logprob = -10.0
                if no_logprob == -100.0:
                    response_text = response.choices[0].message.content[:100] if response.choices[0].message.content else 'N/A'
                    logging.warning(f"⚠️  Theme '{theme}': Could not find 'no' token in logprobs. Using default -10.0. Response: {response_text}")
                    no_logprob = -10.0
                
                # Log if we're using defaults (indicates a problem)
                if yes_logprob == -10.0 and no_logprob == -10.0:
                    logging.error(f"❌ Theme '{theme}': Both yes and no logprobs are defaults (-10.0). This will result in equal probabilities after softmax!")
                    if logprobs_data and logprobs_data.content:
                        available_tokens = [t.token for t in logprobs_data.content[:10]]
                        logging.error(f"   Available tokens in logprobs: {available_tokens}")
                
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
                logging.warning(f"Theme logprob error (attempt {attempt+1}/{max_retries}) for '{theme}': {e}. Retrying in {wait_time}s...")
                time.sleep(wait_time)
            else:
                logging.error(f"Failed to get logprobs for theme '{theme}' after {max_retries} attempts: {e}")
                return {"yes": -10.0, "no": -10.0, "token_yes": None, "token_no": None}
        except Exception as e:
            logging.error(f"Unexpected error getting logprobs for theme '{theme}' (attempt {attempt+1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                time.sleep(5)
            else:
                return {"yes": -10.0, "no": -10.0, "token_yes": None, "token_no": None}
    
    return {"yes": -10.0, "no": -10.0, "token_yes": None, "token_no": None}

def get_theme_logprobs_with_context(full_context_prompt: str, review_text: str, theme: str, client: OpenAI, rate_limiter: RateLimiter, theme_model_name: str, max_retries: int) -> dict:
    """
    Get yes/no logprobs for a single theme using the FULL context (persona, user, category, product) 
    from the first LLM call, plus the generated review text.
    
    This is used in logprobs mode where we want to classify themes using the same context
    that was used to generate the review.
    
    Similar to get_topic_logprobs in topic_classification_fixed.py, but uses full context.
    """
    for attempt in range(max_retries):
        try:
            rate_limiter.wait()
            
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
                    "top_logprobs": 5
                }
                
                response = client.chat.completions.create(**request_params)
                
                # Find highest logprob for "yes" and "no"
                yes_logprob = -100.0
                no_logprob = -100.0
                yes_token = None
                no_token = None
                
                logprobs_data = response.choices[0].logprobs
                
                if logprobs_data and logprobs_data.content:
                    for token_info in logprobs_data.content:
                        token_clean = token_info.token.lower().strip()
                        
                        if token_clean in ["yes", "y"]:
                            if token_info.logprob > yes_logprob:
                                yes_logprob = token_info.logprob
                                yes_token = token_info.token
                        elif token_clean in ["no", "n"]:
                            if token_info.logprob > no_logprob:
                                no_logprob = token_info.logprob
                                no_token = token_info.token
                        
                        if token_info.top_logprobs:
                            for alt in token_info.top_logprobs:
                                alt_clean = alt.token.lower().strip()
                                
                                if alt_clean in ["yes", "y"]:
                                    if alt.logprob > yes_logprob:
                                        yes_logprob = alt.logprob
                                        yes_token = alt.token
                                elif alt_clean in ["no", "n"]:
                                    if alt.logprob > no_logprob:
                                        no_logprob = alt.logprob
                                        no_token = alt.token
                
                # Set defaults if not found - but log a warning
                if yes_logprob == -100.0:
                    response_text = response.choices[0].message.content[:100] if response.choices[0].message.content else 'N/A'
                    logging.warning(f"⚠️  Theme '{theme}': Could not find 'yes' token in logprobs. Using default -10.0. Response: {response_text}")
                    yes_logprob = -10.0
                if no_logprob == -100.0:
                    response_text = response.choices[0].message.content[:100] if response.choices[0].message.content else 'N/A'
                    logging.warning(f"⚠️  Theme '{theme}': Could not find 'no' token in logprobs. Using default -10.0. Response: {response_text}")
                    no_logprob = -10.0
                
                # Log if we're using defaults (indicates a problem)
                if yes_logprob == -10.0 and no_logprob == -10.0:
                    logging.error(f"❌ Theme '{theme}': Both yes and no logprobs are defaults (-10.0). This will result in equal probabilities after softmax!")
                    if logprobs_data and logprobs_data.content:
                        available_tokens = [t.token for t in logprobs_data.content[:10]]
                        logging.error(f"   Available tokens in logprobs: {available_tokens}")
                
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
                logging.warning(f"Theme logprob error (attempt {attempt+1}/{max_retries}) for '{theme}': {e}. Retrying in {wait_time}s...")
                time.sleep(wait_time)
            else:
                logging.error(f"Failed to get logprobs for theme '{theme}' after {max_retries} attempts: {e}")
                return {"yes": -10.0, "no": -10.0, "token_yes": None, "token_no": None}
        except Exception as e:
            logging.error(f"Unexpected error getting logprobs for theme '{theme}' (attempt {attempt+1}/{max_retries}): {e}")
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

def get_llm_prediction(prompt: str, client: OpenAI, rate_limiter: RateLimiter, category_themes: list, model_name: str, max_tokens: int, temperature: float, max_retries: int, scoring_mode: str = "confidence", theme_prediction_model: str = None, product_description: str = "") -> dict:
    """
    Unified LLM call. Returns ALL theme scores.
    Ensures every theme in category_themes gets a score (defaults to 0.0 if missing).
    
    Args:
        scoring_mode: "confidence", "logprobs", or "logprobs-without-persona-context"
            - "confidence": Model outputs confidence scores directly in JSON
            - "logprobs": Generate review first, then get logprobs for each theme separately (with full persona context)
            - "logprobs-without-persona-context": Generate review with full context, then classify themes using ONLY product_description + review_text (no persona context)
        product_description: Required for "logprobs-without-persona-context" mode
    
    Returns:
        dict with prediction data on success, None on failure (after all retries)
    """
    if scoring_mode == "logprobs-without-persona-context":
        # Step 1: Generate review with full persona context (same as logprobs mode)
        for attempt in range(max_retries):
            try:
                rate_limiter.wait()
                
                request_params = {
                    "model": model_name,
                    "messages": [
                        {"role": "system", "content": "You are an expert at role-playing customer personas and generating accurate product reviews. You must output strict JSON."},
                        {"role": "user", "content": prompt}
                    ],
                    "response_format": {"type": "json_object"}
                }
                
                if model_name != "o3":
                    # gpt-5 models require max_completion_tokens instead of max_tokens
                    if "gpt-5" in model_name.lower():
                        if max_tokens is not None:
                            request_params["max_completion_tokens"] = max_tokens
                    else:
                        if max_tokens is not None:
                            request_params["max_tokens"] = max_tokens
                    if temperature is not None:
                        request_params["temperature"] = temperature
                
                response = client.chat.completions.create(**request_params)
                content = response.choices[0].message.content.strip()
                prediction_json = json.loads(content)
                
                review_text = prediction_json.get('review_text', '')
                rating = prediction_json.get('rating', 3.0)
                sentiment = prediction_json.get('sentiment', 'Neutral')
                
                if not review_text:
                    logging.warning(f"⚠️ First LLM call returned empty review_text. JSON keys: {list(prediction_json.keys())}")
                if not rating or rating == 3.0:
                    logging.warning(f"⚠️ First LLM call returned default rating: {rating}")
                
                rating = max(1, min(5, float(rating)))
                if sentiment not in ['Positive', 'Negative', 'Neutral']:
                    sentiment = 'Neutral'
                
                logging.debug(f"✅ First call generated - Review length: {len(review_text)}, Rating: {rating}, Sentiment: {sentiment}")
                
                # Step 2: Get logprobs for each theme using ONLY product_description + review_text (NO persona context)
                if not product_description:
                    logging.warning("⚠️ product_description is empty for logprobs-without-persona-context mode")
                
                theme_model = theme_prediction_model if theme_prediction_model else model_name
                theme_logprobs = {}
                for theme in category_themes:
                    # Use ONLY product_description + review_text (no persona context)
                    logprob_result = get_theme_logprobs_without_context(product_description, review_text, theme, client, rate_limiter, theme_model, max_retries)
                    theme_logprobs[theme] = logprob_result
                
                # Step 3: Normalize using softmax
                normalized_themes = normalize_logprobs_softmax(theme_logprobs)
                
                # Ensure all themes are present
                processed_themes = {}
                for theme in category_themes:
                    processed_themes[theme] = normalized_themes.get(theme, 0.0)
                
                return {
                    'review_text': review_text,
                    'rating': rating,
                    'sentiment': sentiment,
                    'predicted_themes': processed_themes,
                    'theme_logprobs': {theme: theme_logprobs[theme] for theme in category_themes}  # Include raw logprobs for debugging
                }
                
            except (json.JSONDecodeError, APIError, RateLimitError) as e:
                if attempt < max_retries - 1:
                    wait_time = (2 ** attempt) * 2
                    logging.warning(f"Error (attempt {attempt+1}/{max_retries}): {e}. Retrying in {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    logging.error(f"Failed after {max_retries} attempts: {e}")
                    return None
            except Exception as e:
                logging.error(f"Unexpected error (attempt {attempt+1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    time.sleep(5)
                else:
                    return None
        
        return None
    
    elif scoring_mode == "logprobs":
        # Step 1: Generate review (without theme scores)
        for attempt in range(max_retries):
            try:
                rate_limiter.wait()
                
                request_params = {
                    "model": model_name,
                    "messages": [
                        {"role": "system", "content": "You are an expert at role-playing customer personas and generating accurate product reviews. You must output strict JSON."},
                        {"role": "user", "content": prompt}
                    ],
                    "response_format": {"type": "json_object"}
                }
                
                if model_name != "o3":
                    # gpt-5 models require max_completion_tokens instead of max_tokens
                    if "gpt-5" in model_name.lower():
                        if max_tokens is not None:
                            request_params["max_completion_tokens"] = max_tokens
                    else:
                        if max_tokens is not None:
                            request_params["max_tokens"] = max_tokens
                    if temperature is not None:
                        request_params["temperature"] = temperature
                
                response = client.chat.completions.create(**request_params)
                content = response.choices[0].message.content.strip()
                prediction_json = json.loads(content)
                
                review_text = prediction_json.get('review_text', '')
                rating = prediction_json.get('rating', 3.0)
                sentiment = prediction_json.get('sentiment', 'Neutral')
                
                # Debug logging for first call
                if not review_text:
                    logging.warning(f"⚠️ First LLM call returned empty review_text. JSON keys: {list(prediction_json.keys())}")
                if not rating or rating == 3.0:
                    logging.warning(f"⚠️ First LLM call returned default rating: {rating}")
                
                rating = max(1, min(5, float(rating)))
                if sentiment not in ['Positive', 'Negative', 'Neutral']:
                    sentiment = 'Neutral'
                
                logging.debug(f"✅ First call generated - Review length: {len(review_text)}, Rating: {rating}, Sentiment: {sentiment}")
                
                # Step 2: Get logprobs for each theme using FULL context (persona, user, category, product)
                # This ensures theme classification uses the same context as review generation
                theme_model = theme_prediction_model if theme_prediction_model else model_name
                theme_logprobs = {}
                for theme in category_themes:
                    # Use full context prompt + generated review for theme classification
                    logprob_result = get_theme_logprobs_with_context(prompt, review_text, theme, client, rate_limiter, theme_model, max_retries)
                    theme_logprobs[theme] = logprob_result
                
                # Step 3: Normalize using softmax
                normalized_themes = normalize_logprobs_softmax(theme_logprobs)
                
                # Ensure all themes are present
                processed_themes = {}
                for theme in category_themes:
                    processed_themes[theme] = normalized_themes.get(theme, 0.0)
                
                return {
                    'review_text': review_text,
                    'rating': rating,
                    'sentiment': sentiment,
                    'predicted_themes': processed_themes,
                    'theme_logprobs': {theme: theme_logprobs[theme] for theme in category_themes}  # Include raw logprobs for debugging
                }
                
            except (json.JSONDecodeError, APIError, RateLimitError) as e:
                if attempt < max_retries - 1:
                    wait_time = (2 ** attempt) * 2
                    logging.warning(f"Error (attempt {attempt+1}/{max_retries}): {e}. Retrying in {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    logging.error(f"Failed after {max_retries} attempts: {e}")
                    return None
            except Exception as e:
                logging.error(f"Unexpected error (attempt {attempt+1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    time.sleep(5)
                else:
                    return None
        
        return None
    
    else:
        # Original confidence-based scoring
        for attempt in range(max_retries):
            try:
                rate_limiter.wait()
                
                # Build request parameters
                request_params = {
                    "model": model_name,
                    "messages": [
                        {"role": "system", "content": "You are an expert at role-playing customer personas and generating accurate product reviews. You must output strict JSON. CRITICAL: You MUST provide a confidence score (0.0 to 1.0) for EVERY SINGLE theme listed in the prompt. No theme can be omitted. For theme scoring, carefully analyze your review_text and assign scores that accurately reflect what you wrote. Be precise: if a theme is mentioned, give it an appropriate score (0.4-1.0); if not mentioned, give it 0.0. Every theme in the list must have a score - this is mandatory. Be consistent and accurate in your theme scoring based on what you actually wrote in the review."},
                        {"role": "user", "content": prompt}
                    ],
                    "response_format": {"type": "json_object"}
                }
                
                # Only add max_tokens and temperature for non-o3 models
                if model_name != "o3":
                    if max_tokens is not None:
                        request_params["max_tokens"] = max_tokens
                    if temperature is not None:
                        request_params["temperature"] = temperature
                
                response = client.chat.completions.create(**request_params)
                content = response.choices[0].message.content.strip()
                prediction_json = json.loads(content)
                
                review_text = prediction_json.get('review_text', '')
                rating = prediction_json.get('rating', 3.0)
                sentiment = prediction_json.get('sentiment', 'Neutral')
                themes = prediction_json.get('themes', {})
                
                rating = max(1, min(5, float(rating)))
                if sentiment not in ['Positive', 'Negative', 'Neutral']:
                    sentiment = 'Neutral'
                
                processed_themes = {k: float(v) for k, v in themes.items()}
                
                # CRITICAL: Ensure ALL themes from category_themes get a score
                # If LLM missed any theme, set it to 0.0
                if category_themes:
                    for theme in category_themes:
                        if theme not in processed_themes:
                            processed_themes[theme] = 0.0
                            logging.debug(f"Missing theme '{theme}' in LLM response, defaulting to 0.0")
                
                return {
                    'review_text': review_text,
                    'rating': rating,
                    'sentiment': sentiment,
                    'predicted_themes': processed_themes
                }
                
            except (json.JSONDecodeError, APIError, RateLimitError) as e:
                if attempt < max_retries - 1:
                    wait_time = (2 ** attempt) * 2
                    logging.warning(f"Error (attempt {attempt+1}/{max_retries}): {e}. Retrying in {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    logging.error(f"Failed after {max_retries} attempts: {e}")
                    # Return None to indicate failure (don't return fake data)
                    return None
            except Exception as e:
                logging.error(f"Unexpected error (attempt {attempt+1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    time.sleep(5)
                else:
                    # Return None to indicate failure (don't return fake data)
                    return None
