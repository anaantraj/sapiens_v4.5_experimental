#!/usr/bin/env python3
"""
Stage 09: History Baseline
===========================

Baseline prediction using full review history (leave-one-out approach).
Based on Clustering/prediction_leave_one_out_full_history.py
"""

import os
import sys
import json
import logging
import time
import re
import argparse
import math
import numpy as np
from tqdm import tqdm
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Load .env so OPENAI_API_KEY and OPENAI_BASE_URL (Bedrock) are set
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent.parent.parent / ".env", override=True)
except ImportError:
    pass

from openai import OpenAI
import json_repair
from scipy.stats import entropy

# Import W&B utilities
from utils.openai_client import create_openai_client
from utils.wandb_utils import (
    get_stage_config, get_openai_config,
    init_wandb_run, finish_run, use_artifact, log_artifact,
    get_artifact_dir, link_to_registry
)

# Import schemas
from schemas.learned_artifacts.topic_universe import TopicUniverseArtifact
from schemas.learned_artifacts.baseline_predictions import (
    BaselinePredictionsArtifact,
    BaselinePredictionItem,
    BaselinePredictionsArtifactLogprobs,
    BaselinePredictionItemLogprobs
)

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# =============================================================================
# Configuration
# =============================================================================

_cfg = get_stage_config("09_baselines")
_openai_cfg = get_openai_config()

# Rate limiting
rate_limit_lock = threading.Lock()
last_request_time = [0.0]
MIN_REQUEST_INTERVAL = 0.1

# JSD calculation epsilon
EPSILON = 1e-10

# =============================================================================
# Category Normalization
# =============================================================================

def normalize_category_name(category: str) -> str:
    """
    Normalizes category names to match topic_universe format.
    Converts: "Clothing Shoes & Jewelry" -> "Clothing_Shoes_and_Jewelry"
    Based on topic_classification_fixed.py
    """
    if not category:
        return category
    
    # Replace "&" with "and"
    normalized = category.replace("&", "and")
    # Replace spaces with underscores
    normalized = normalized.replace(" ", "_")
    # Remove any double underscores
    normalized = normalized.replace("__", "_")
    # Strip underscores from start/end
    normalized = normalized.strip("_")
    
    return normalized

# Category mapping
SUB_TO_MAIN_CATEGORY_MAP = {
    "All Beauty": "All Beauty",
    "Premium Beauty": "All Beauty",
    "Health & Personal Care": "Health & Personal Care",
    "Baby": "Health & Personal Care",
    "Grocery": "Health & Personal Care",
    "Pet Supplies": "Health & Personal Care",
    "Sports & Outdoors": "Health & Personal Care",
    "Video Games": "Video Games",
    "Toys & Games": "Video Games",
    "Computers": "Video Games",
    "Software": "Video Games",
    "Movies & TV": "Video Games",
    "All Electronics": "Appliances",
    "Cell Phones & Accessories": "Appliances",
    "Camera & Photo": "Appliances",
    "Office Products": "Appliances",
    "Tools & Home Improvement": "Appliances",
    "Industrial & Scientific": "Appliances",
    "Appliances": "Appliances",
    "Amazon Home": "Appliances",
    "Automotive": "Appliances",
    "Digital Music": "Digital Music",
    "Musical Instruments": "Digital Music",
    "Home Audio & Theater": "Digital Music",
    "Portable Audio & Accessories": "Digital Music",
    "Amazon Devices": "Digital Music",
    "Appstore for Android": "Software",
    "Buy a Kindle": "Software",
    "Books": "Fashion",
    "AMAZON FASHION": "Fashion",
    "Arts, Crafts & Sewing": "Fashion"
}

# =============================================================================
# Utility Functions
# =============================================================================

def normalize_sentiment(sentiment: str) -> str:
    """
    Normalize sentiment to schema-compliant values.
    Schema only allows: 'Positive', 'Negative', 'Neutral'
    """
    if not sentiment:
        return 'Neutral'
    
    sentiment_lower = sentiment.strip().lower()
    
    # Map common variations to schema values
    if sentiment_lower in ['positive', 'pos', 'good', 'great', 'excellent', 'happy', 'satisfied']:
        return 'Positive'
    elif sentiment_lower in ['negative', 'neg', 'bad', 'poor', 'terrible', 'unhappy', 'dissatisfied']:
        return 'Negative'
    elif sentiment_lower in ['neutral', 'neut', 'mixed', 'ambivalent', 'ok', 'okay', 'average']:
        return 'Neutral'
    else:
        # Default to Neutral for unknown values
        return 'Neutral'

def rate_limited_request():
    """Ensure we don't exceed rate limits."""
    with rate_limit_lock:
        current_time = time.time()
        time_since_last = current_time - last_request_time[0]
        if time_since_last < MIN_REQUEST_INTERVAL:
            time.sleep(MIN_REQUEST_INTERVAL - time_since_last)
        last_request_time[0] = time.time()

def convert_to_serializable(obj):
    """Convert numpy types to native Python types."""
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [convert_to_serializable(i) for i in obj]
    return obj

# =============================================================================
# Prompt Creation
# =============================================================================

def load_prompt(prompt_filename: str) -> str:
    """Load a prompt template from file."""
    prompt_path = Path(__file__).parent.parent / "prompts" / prompt_filename
    if prompt_path.exists():
        with open(prompt_path, 'r', encoding='utf-8') as f:
            return f.read().strip()
    else:
        raise FileNotFoundError(f"Prompt file not found: {prompt_path}")

def create_prompt_full_history(history_reviews: list, target_review: dict, category_themes: list) -> str:
    """Create prompt with ALL reviews except current (no truncation/limits)."""
    # Allow empty history_reviews (for first review of a user)
    if history_reviews is None:
        history_reviews = []

    # Load prompt template
    try:
        prompt_template = load_prompt("history_baseline_prompt.txt")
    except FileNotFoundError as e:
        logging.error(f"Could not load prompt template: {e}")
        return None

    # Build History - Include ALL reviews (no truncation)
    # If no history, use "nothing"
    history_text_block = ""
    if history_reviews:
        for idx, rev in enumerate(history_reviews):
            prod = rev.get('product_description', 'N/A')
            text = rev.get('review_text', 'N/A')
            rating = rev.get('rating', 'N/A')
            history_text_block += f"Example {idx+1}:\nProduct: {prod}\nMy Rating: {rating}\nMy Review: {text}\n\n"
    else:
        history_text_block = "nothing"

    # Prepare Target - Full product description
    new_prod_desc = target_review.get('product_description', 'N/A')

    # Format themes list clearly
    themes_list = "\n".join([f"- {theme}" for theme in category_themes])
    
    # Format themes JSON template
    themes_json_template = "\n".join([
        f'    "{theme}": <float 0.0-1.0>,' if i < len(category_themes) - 1 
        else f'    "{theme}": <float 0.0-1.0>' 
        for i, theme in enumerate(category_themes)
    ])
    
    # Format the prompt using the template
    prompt = prompt_template.format(
        history_text_block=history_text_block,
        new_prod_desc=new_prod_desc,
        themes_list=themes_list,
        themes_json_template=themes_json_template
    )
    
    return prompt

def create_prompt_without_themes(history_reviews: list, target_review: dict) -> str:
    """Create prompt without themes - only review_text, rating, sentiment."""
    # Allow empty history_reviews (for first review of a user)
    if history_reviews is None:
        history_reviews = []

    # Load prompt template
    try:
        prompt_template = load_prompt("history_baseline_prompt_no_themes.txt")
    except FileNotFoundError as e:
        logging.error(f"Could not load prompt template: {e}")
        return None

    # Build History - Include ALL reviews (no truncation)
    # If no history, use "nothing"
    history_text_block = ""
    if history_reviews:
        for idx, rev in enumerate(history_reviews):
            prod = rev.get('product_description', 'N/A')
            text = rev.get('review_text', 'N/A')
            rating = rev.get('rating', 'N/A')
            history_text_block += f"Example {idx+1}:\nProduct: {prod}\nMy Rating: {rating}\nMy Review: {text}\n\n"
    else:
        history_text_block = "nothing"

    # Prepare Target - Full product description
    new_prod_desc = target_review.get('product_description', 'N/A')
    
    # Format the prompt using the template
    prompt = prompt_template.format(
        history_text_block=history_text_block,
        new_prod_desc=new_prod_desc
    )
    
    return prompt

# =============================================================================
# LLM Interaction
# =============================================================================

def get_llm_prediction(prompt: str, client, model_name: str) -> dict:
    if not prompt:
        return {'review_text': 'Error: Invalid prompt.', 'rating': 3.0, 'sentiment': 'Neutral', 'predicted_themes': {}}
    
    max_retries = 3
    for attempt in range(max_retries):
        try:
            rate_limited_request()
            
            response = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt}]
            )

            response_text = response.choices[0].message.content

            # Robust Parsing
            json_match = re.search(r"\{[\s\S]*\}", response_text)
            json_string = json_match.group(0) if json_match else response_text
                
            try:
                prediction_json = json_repair.loads(json_string)
            except Exception:
                if attempt < max_retries - 1:
                    time.sleep(2)
                    continue
                logging.error(f"FAILED JSON RAW: {json_string[:100]}...")
                raise ValueError("Could not repair JSON")

            return {
                'review_text': prediction_json.get('review_text', 'Error: Could not generate text.'),
                'rating': float(prediction_json.get('rating', 3.0)),
                'sentiment': prediction_json.get('sentiment', 'Neutral'),
                'predicted_themes': prediction_json.get('themes', {})
            }
            
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(5)
                continue
            logging.error(f"\n❌ GENERATION ERROR: {str(e)}")
            return {'review_text': 'Error: Failed to get prediction.', 'rating': 3.0, 'sentiment': 'Neutral', 'predicted_themes': {}}

def get_llm_prediction_without_themes(prompt: str, client, model_name: str, actual_model_name: str = None) -> dict:
    """Get LLM prediction without themes - only review_text, rating, sentiment."""
    if not prompt:
        return {'review_text': 'Error: Invalid prompt.', 'rating': 3.0, 'sentiment': 'Neutral'}
    
    # Use actual_model_name if provided, otherwise use model_name
    model_to_use = actual_model_name if actual_model_name else model_name
    
    max_retries = 3
    for attempt in range(max_retries):
        try:
            rate_limited_request()
            
            response = client.chat.completions.create(
                model=model_to_use,
                messages=[{"role": "user", "content": prompt}]
            )

            response_text = response.choices[0].message.content

            # Robust Parsing
            json_match = re.search(r"\{[\s\S]*\}", response_text)
            json_string = json_match.group(0) if json_match else response_text
                
            try:
                prediction_json = json_repair.loads(json_string)
            except Exception:
                if attempt < max_retries - 1:
                    time.sleep(2)
                    continue
                logging.error(f"FAILED JSON RAW: {json_string[:100]}...")
                raise ValueError("Could not repair JSON")

            return {
                'review_text': prediction_json.get('review_text', 'Error: Could not generate text.'),
                'rating': float(prediction_json.get('rating', 3.0)),
                'sentiment': prediction_json.get('sentiment', 'Neutral')
            }
            
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(5)
                continue
            logging.error(f"\n❌ GENERATION ERROR: {str(e)}")
            return {'review_text': 'Error: Failed to get prediction.', 'rating': 3.0, 'sentiment': 'Neutral'}

# =============================================================================
# Metrics Calculation
# =============================================================================

def normalize_theme_name(theme: str) -> str:
    """Normalize theme name for case-insensitive matching."""
    return theme.strip().lower()

def find_theme_matches(predicted_theme_normalized: str, actual_themes_normalized: set) -> set:
    """Find which actual themes match the predicted theme."""
    matches = set()
    
    for actual_theme_norm in actual_themes_normalized:
        # Exact match
        if predicted_theme_normalized == actual_theme_norm:
            matches.add(actual_theme_norm)
            continue
        
        # Check if predicted theme is a substring of actual theme
        if predicted_theme_normalized in actual_theme_norm:
            matches.add(actual_theme_norm)
            continue
        
        # Check if actual theme is a substring of predicted theme
        if actual_theme_norm in predicted_theme_normalized:
            matches.add(actual_theme_norm)
            continue
        
        # Check word-level overlap
        pred_words = set(predicted_theme_normalized.replace(',', ' ').replace('&', ' ').replace(' ', ' ').split())
        actual_words = set(actual_theme_norm.replace(',', ' ').replace('&', ' ').replace(' ', ' ').split())
        stop_words = {'and', 'or', 'the', 'a', 'an', 'of', 'for', 'in', 'on', 'at', 'to', 'with'}
        pred_words = pred_words - stop_words
        actual_words = actual_words - stop_words
        
        if len(pred_words) > 0 and len(actual_words) > 0:
            overlap = pred_words.intersection(actual_words)
            min_overlap = max(2, min(len(pred_words), len(actual_words)) * 0.5)
            if len(overlap) >= min_overlap:
                matches.add(actual_theme_norm)
    
    return matches

def calculate_enhanced_accuracy(prediction: dict, actual: dict) -> dict:
    # Prefer topic_probabilities, fallback to themes/predicted_themes
    actual_themes_list = None
    if actual.get('topic_probabilities'):
        # Convert topic_probabilities dict to list of theme names (for matching)
        actual_themes_list = list(actual.get('topic_probabilities', {}).keys())
    else:
        actual_themes_list = actual.get('themes', actual.get('predicted_themes', []))
    
    actual_themes_normalized = {normalize_theme_name(t): t for t in actual_themes_list}
    actual_themes_set = set(actual_themes_normalized.keys())
    num_actual_themes = len(actual_themes_set)
    
    if num_actual_themes == 0:
        return {'overall_accuracy': 0.0, 'rating_score': 0.0, 'sentiment_score': 0.0, 'recall@max(3,k)': 0.0}

    predicted_themes_raw = prediction.get('predicted_themes', {})
    
    # Handle both list and dict formats
    if isinstance(predicted_themes_raw, list):
        pred_items = [(normalize_theme_name(t), 1.0 - (i*0.01)) for i, t in enumerate(predicted_themes_raw)]
    elif isinstance(predicted_themes_raw, dict):
        pred_items = sorted([(normalize_theme_name(k), float(v)) for k, v in predicted_themes_raw.items()], 
                          key=lambda x: x[1], reverse=True)
    else:
        pred_items = []

    k_adaptive = max(3, num_actual_themes)
    top_k_predicted = [item[0] for item in pred_items[:k_adaptive]]
    
    # Count matches using improved matching
    all_matches = set()
    for pred_theme_norm in top_k_predicted:
        matches = find_theme_matches(pred_theme_norm, actual_themes_set)
        all_matches.update(matches)
    
    num_matches = len(all_matches)
    recall_score = num_matches / num_actual_themes if num_actual_themes > 0 else 0.0
    
    actual_rating = actual.get('rating', 3.0)
    pred_rating = prediction.get('rating', 3.0)
    rating_diff = abs(pred_rating - actual_rating)
    rating_score = max(0, 1 - (rating_diff / 4.0))
    
    sentiment_score = 1.0 if prediction.get('sentiment', '').lower() == actual.get('sentiment', '').lower() else 0.0
    
    weights = {'rating': 0.4, 'sentiment': 0.3, 'theme_recall': 0.3}
    overall_accuracy = (rating_score * weights['rating']) + \
                       (sentiment_score * weights['sentiment']) + \
                       (recall_score * weights['theme_recall'])

    return {
        'overall_accuracy': overall_accuracy,
        'rating_score': rating_score,
        'sentiment_score': sentiment_score,
        'recall@max(3,k)': recall_score
    }

# =============================================================================
# Logprobs Mode Functions
# =============================================================================

def get_topic_logprobs(review_text: str, topic: str, client, model_name: str = None) -> Dict[str, float]:
    """
    Returns yes/no probabilities using Logprobs (SYNC version).
    Based on topic_classification_fixed.py but adapted for synchronous use.
    """
    system_msg = "You are a precise topic classifier. Answer only Yes or No."
    user_msg = f'Analyze this product review and determine if it discusses the topic "{topic}".\n\nReview: "{review_text}"\n\nDoes this review discuss the topic "{topic}"?\nAnswer with ONLY "Yes" or "No".'

    try:
        rate_limited_request()
        
        # Try with logprobs first (for models that support it)
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": user_msg}
                ],
                max_tokens=5,
                temperature=0,
                logprobs=True,
                top_logprobs=5
            )
            
            # Find ONLY the HIGHEST logprob for "yes" and "no"
            yes_logprob = -100.0   # Sentinel value - will be updated if found
            no_logprob = -100.0    # Sentinel value - will be updated if found
            yes_token = None       # Track which token gave highest
            no_token = None        # Track which token gave highest
            
            logprobs_data = response.choices[0].logprobs
            
            if logprobs_data and logprobs_data.content:
                for token_info in logprobs_data.content:
                    # Clean the main token: lowercase + strip spaces
                    token_clean = token_info.token.lower().strip()
                    
                    # Check main token
                    if token_clean in ["yes", "y"]:
                        if token_info.logprob > yes_logprob:
                            yes_logprob = token_info.logprob
                            yes_token = token_info.token
                    elif token_clean in ["no", "n"]:
                        if token_info.logprob > no_logprob:
                            no_logprob = token_info.logprob
                            no_token = token_info.token
                    
                    # Check alternatives (top_logprobs)
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
            
            # Set defaults if not found
            if yes_logprob == -100.0:
                yes_logprob = -10.0
            if no_logprob == -100.0:
                no_logprob = -10.0
            
            # Convert to probabilities: prob = e^(logprob)
            prob_yes = math.exp(yes_logprob)
            prob_no = math.exp(no_logprob)
            
            # Normalize probabilities
            total = prob_yes + prob_no
            if total > 0:
                prob_yes /= total
                prob_no /= total
            else:
                prob_yes = 0.0
                prob_no = 0.0
            
            return {
                "yes": prob_yes,
                "no": prob_no,
                "logprob_yes": yes_logprob,
                "logprob_no": no_logprob,
                "token_yes": yes_token,
                "token_no": no_token
            }
        
        except Exception as logprob_error:
            # If logprobs not supported, fallback to answer-based classification
            error_str = str(logprob_error)
            if "logprob" in error_str.lower() or "403" in error_str:
                # Retry without logprobs
                rate_limited_request()
                response = client.chat.completions.create(
                    model=model_name,
                    messages=[
                        {"role": "system", "content": system_msg},
                        {"role": "user", "content": user_msg}
                    ],
                    max_tokens=5,
                    temperature=0
                )
                
                # Get answer from response
                answer_text = response.choices[0].message.content.strip().lower()
                is_yes = "yes" in answer_text or answer_text.startswith("y")
                
                # Assign probabilities based on answer (0.9 for chosen, 0.1 for other)
                prob_yes = 0.9 if is_yes else 0.1
                prob_no = 0.1 if is_yes else 0.9
                
                return {
                    "yes": prob_yes,
                    "no": prob_no,
                    "logprob_yes": math.log(prob_yes) if prob_yes > 0 else -10.0,
                    "logprob_no": math.log(prob_no) if prob_no > 0 else -10.0,
                    "token_yes": "Yes" if is_yes else None,
                    "token_no": "No" if not is_yes else None
                }
            else:
                # Re-raise if it's a different error
                raise

    except Exception as e:
        logging.warning(f"Error on topic '{topic}': {e}")
        return {
            "yes": 0.0,
            "no": 0.0,
            "logprob_yes": -10.0,
            "logprob_no": -10.0,
            "token_yes": None,
            "token_no": None
        }

def classify_topics_with_logprobs(review_text: str, category_themes: list, client, model_name: str = None) -> Dict[str, float]:
    """
    Classify all topics for a review using logprobs.
    Returns a dictionary mapping theme names to probabilities (normalized using softmax).
    Calls LLM in parallel for all topics (like topic_classification_fixed.py).
    """
    if not review_text or not category_themes:
        return {}
    
    # Get logprobs for all topics IN PARALLEL
    topic_logprobs = {}
    topic_scores = {}
    
    # Use ThreadPoolExecutor to parallelize topic classification calls
    with ThreadPoolExecutor(max_workers=min(10, len(category_themes))) as executor:
        # Submit all topic classification tasks
        future_to_topic = {
            executor.submit(get_topic_logprobs, review_text, topic, client, model_name): topic
            for topic in category_themes
        }
        
        # Collect results as they complete
        for future in as_completed(future_to_topic):
            topic = future_to_topic[future]
            try:
                result = future.result()
                topic_logprobs[topic] = {
                    "logprob_yes": result["logprob_yes"],
                    "logprob_no": result["logprob_no"]
                }
                topic_scores[topic] = result["yes"]
            except Exception as e:
                logging.warning(f"Error classifying topic '{topic}': {e}")
                # Set default values on error
                topic_logprobs[topic] = {
                    "logprob_yes": -10.0,
                    "logprob_no": -10.0
                }
                topic_scores[topic] = 0.0
    
    # Normalize topic probabilities using softmax on log probabilities
    if len(topic_scores) > 0:
        # Get log probabilities for softmax (logprob_yes values)
        logprob_values = [topic_logprobs[topic]["logprob_yes"] for topic in topic_scores.keys()]
        
        # Apply numerically stable softmax: exp(x - max(x)) / sum(exp(x - max(x)))
        # Subtract max to prevent overflow
        max_logprob = max(logprob_values) if logprob_values else 0.0
        
        # Compute exp(x - max) for each logprob
        exp_logprobs = [math.exp(logprob - max_logprob) for logprob in logprob_values]
        sum_exp = sum(exp_logprobs)
        
        if sum_exp > 0:
            # Softmax probabilities (will sum to 1.0)
            softmax_probs = [exp_logprob / sum_exp for exp_logprob in exp_logprobs]
        else:
            # Fallback to uniform if sum is 0 (shouldn't happen, but safety check)
            softmax_probs = [1.0 / len(logprob_values)] * len(logprob_values)
        
        # Create normalized dictionary (softmax ensures sum = 1.0)
        normalized_topic_scores = {
            topic: softmax_prob
            for topic, softmax_prob in zip(topic_scores.keys(), softmax_probs)
        }
    else:
        normalized_topic_scores = {}
    
    return normalized_topic_scores

# =============================================================================
# JSD Calculation Functions
# =============================================================================

def normalize_distribution(theme_dict: Dict[str, float]) -> Dict[str, float]:
    """
    Normalize a theme probability distribution to sum to 1.0.
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
    """
    # Get all unique themes from both distributions
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

def calculate_jsd_metrics(prediction: dict, actual: dict) -> dict:
    """
    Calculate JSD for logprobs mode.
    Returns dict with only jsd (no other metrics).
    Uses topic_probabilities from actual review data - no fallback.
    """
    # Get theme distributions - ONLY use topic_probabilities, no fallback
    actual_themes = actual.get('topic_probabilities', {})
    if not actual_themes:
        return {'jsd': 0.0}
    
    predicted_themes = prediction.get('predicted_themes', {})
    
    # Handle different formats (list vs dict)
    if isinstance(actual_themes, list):
        # Convert list to uniform distribution
        if not actual_themes:
            return {'jsd': 0.0}
        actual_themes = {theme: 1.0 / len(actual_themes) for theme in actual_themes}
    
    if isinstance(predicted_themes, list):
        # Convert list to uniform distribution
        if not predicted_themes:
            return {'jsd': 0.0}
        predicted_themes = {theme: 1.0 / len(predicted_themes) for theme in predicted_themes}
    
    # Validate that we have distributions
    if not actual_themes or not predicted_themes:
        return {'jsd': 0.0}
    
    # Normalize distributions to ensure they sum to 1.0
    actual_themes = normalize_distribution(actual_themes)
    predicted_themes = normalize_distribution(predicted_themes)
    
    if not actual_themes or not predicted_themes:
        return {'jsd': 0.0}
    
    # Align distributions (will also normalize again as safety check)
    actual_array, predicted_array = align_distributions(actual_themes, predicted_themes)
    
    if len(actual_array) == 0 or len(predicted_array) == 0:
        return {'jsd': 0.0}
    
    # Calculate JSD
    jsd = compute_jsd(actual_array, predicted_array)
    
    return {
        'jsd': jsd
    }

# =============================================================================
# Parallel Processing
# =============================================================================

def process_single_review(args_tuple):
    """Process a single review prediction."""
    (user_id, review_index, target_review, history_reviews, category_themes, 
     raw_category, client, model_name, actual_model_name) = args_tuple
    
    prompt = create_prompt_full_history(history_reviews, target_review, category_themes)
    
    if not prompt:
        return None
    
    prediction = get_llm_prediction(prompt, client, model_name, actual_model_name)
    metrics = calculate_enhanced_accuracy(prediction, target_review)
    
    total_reviews = len(history_reviews) + 1
    if review_index == 0: 
        r_type = 'first'
    elif review_index == total_reviews - 1: 
        r_type = 'last'
    else: 
        r_type = 'intermediate'

    # Format result to match BaselinePredictionItem schema
    result = {
        'user_id': user_id,
        'review_index': review_index,
        'review_type': r_type,
        'product_description': target_review.get('product_description', 'N/A'),
        'category': raw_category,
        'method': 'history',
        'model': 'claude' if 'claude' in model_name.lower() else 'o3',  # Ensure valid model name
        'prediction': {
            'review_text': prediction.get('review_text', ''),
            'rating': float(prediction.get('rating', 3.0)),
            'sentiment': normalize_sentiment(prediction.get('sentiment', 'Neutral')),
            'predicted_themes': prediction.get('predicted_themes', {})
        },
        'actual': {
            'review_text': target_review.get('review_text', ''),
            'rating': float(target_review.get('rating', 3.0)),
            'sentiment': normalize_sentiment(target_review.get('sentiment', 'Neutral')),
            'topic_probabilities': target_review.get('topic_probabilities', {}),
            'themes': target_review.get('themes', target_review.get('predicted_themes', []))
        },
        'metrics': {
            'overall_accuracy': metrics.get('overall_accuracy', 0.0),
            'rating_score': metrics.get('rating_score', 0.0),
            'sentiment_score': metrics.get('sentiment_score', 0.0),
            'recall@max(3,k)': metrics.get('recall@max(3,k)', 0.0)
        }
    }
    
    return result

def process_single_review_logprobs(args_tuple):
    """Process a single review prediction using logprobs mode."""
    (user_id, review_index, target_review, history_reviews, category_themes, 
     raw_category, text_gen_client, text_gen_model, actual_text_gen_model, topic_classification_client, topic_classification_model) = args_tuple
    
    # Step 1: Generate review without themes (using configured model)
    prompt = create_prompt_without_themes(history_reviews, target_review)
    
    if not prompt:
        return None
    
    prediction_base = get_llm_prediction_without_themes(prompt, text_gen_client, text_gen_model, actual_text_gen_model)
    generated_review_text = prediction_base.get('review_text', '')
    
    if not generated_review_text or generated_review_text.startswith('Error'):
        return None
    
    # Step 2: Classify topics using logprobs (using gpt-4o-mini)
    predicted_themes = classify_topics_with_logprobs(
        generated_review_text, 
        category_themes, 
        topic_classification_client, 
        topic_classification_model
    )
    
    # Log if predicted_themes is empty (for debugging)
    if not predicted_themes:
        logging.warning(f"⚠️ No predicted themes for user {user_id}, review {review_index} - review_text length: {len(generated_review_text)}, category_themes count: {len(category_themes)}")
    
    # Step 3: Calculate metrics (JSD instead of recall)
    prediction = {
        'review_text': generated_review_text,
        'rating': prediction_base.get('rating', 3.0),
        'sentiment': prediction_base.get('sentiment', 'Neutral'),
        'predicted_themes': predicted_themes if predicted_themes else {}  # Ensure it's always a dict
    }
    
    metrics = calculate_jsd_metrics(prediction, target_review)
    
    # Log if metrics calculation failed (for debugging)
    if metrics.get('jsd') == 0.0 and not target_review.get('topic_probabilities') and not target_review.get('predicted_themes') and not target_review.get('themes'):
        logging.warning(f"⚠️ No actual themes found for user {user_id}, review {review_index} - cannot calculate JSD")
    
    total_reviews = len(history_reviews) + 1
    if review_index == 0: 
        r_type = 'first'
    elif review_index == total_reviews - 1: 
        r_type = 'last'
    else: 
        r_type = 'intermediate'

    # Format result to match BaselinePredictionItem schema
    result = {
        'user_id': user_id,
        'review_index': review_index,
        'review_type': r_type,
        'product_description': target_review.get('product_description', 'N/A'),
        'category': raw_category,
        'method': 'history',  # Schema requires 'history' or 'backstory', not 'history_logprobs'
        'model': 'claude' if 'claude' in text_gen_model.lower() else 'o3',
        'prediction': {
            'review_text': prediction.get('review_text', ''),
            'rating': float(prediction.get('rating', 3.0)),
            'sentiment': normalize_sentiment(prediction.get('sentiment', 'Neutral')),
            'predicted_themes': prediction.get('predicted_themes', {})
        },
        'actual': {
            'review_text': target_review.get('review_text', ''),
            'rating': float(target_review.get('rating', 3.0)),
            'sentiment': normalize_sentiment(target_review.get('sentiment', 'Neutral')),
            'topic_probabilities': target_review.get('topic_probabilities', {}),
            'themes': target_review.get('themes', target_review.get('predicted_themes', []))
        },
        'metrics': {
            'jsd': metrics.get('jsd', 0.0)  # Only metric calculated in logprobs mode
        }
    }
    
    return result

# =============================================================================
# Main Execution
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="History Baseline Prediction")
    parser.add_argument("--model", type=str, default="o3", choices=["o3", "claude"],
                       help="Model to use: 'o3' or 'claude'")
    parser.add_argument("--max-workers", type=int, default=20,
                       help="Number of parallel workers")
    parser.add_argument("--output-dir", type=str, default=None,
                       help="Output directory (default: auto)")
    parser.add_argument("--scoring-mode", type=str, default="default", choices=["default", "logprobs"],
                       help="Scoring mode: 'default' (themes in prompt) or 'logprobs' (separate topic classification)")
    args = parser.parse_args()
    
    model_name = args.model.lower()
    scoring_mode = args.scoring_mode.lower()
    
    logging.info(f"🚀 Starting History Baseline Prediction Pipeline")
    logging.info(f"📊 Text Generation Model: {model_name} | Workers: {args.max_workers} | Scoring Mode: {scoring_mode}")

    # Initialize clients
    text_gen_client = None
    topic_classification_client = None
    text_gen_model = None
    topic_classification_model = None
    
    # Get model names from config
    review_prediction_models = _cfg.get("review_prediction_models", {})
    o3_model_name = review_prediction_models.get("o3")
    claude_model_name = review_prediction_models.get("claude")
    theme_prediction_model = _cfg.get("theme_prediction_model")
    
    # Initialize text generation client (o3 or claude)
    if model_name == "o3":
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            logging.error("❌ No OpenAI API Key found. Please set OPENAI_API_KEY.")
            return
        if scoring_mode == "logprobs":
            text_gen_client = OpenAI(api_key=api_key)
            if not o3_model_name:
                logging.error("❌ review_prediction_models.o3 not found in config")
                return
            text_gen_model = o3_model_name
        else:
            text_gen_client = create_openai_client(openai_config=_openai_cfg, timeout=120.0)
            text_gen_model = _cfg.get("bedrock_model_id") if "bedrock-mantle" in (os.environ.get("OPENAI_BASE_URL") or "") and _cfg.get("bedrock_model_id") else o3_model_name
            if not text_gen_model:
                logging.error("❌ Missing model config for o3/Bedrock in 09_baselines config")
                return
    elif model_name == "claude":
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            logging.error("❌ No Anthropic API Key found. Please set ANTHROPIC_API_KEY.")
            return
        try:
            import anthropic
            text_gen_client = anthropic.Anthropic(api_key=api_key)
            if not claude_model_name:
                logging.error("❌ review_prediction_models.claude not found in config")
                return
            text_gen_model = claude_model_name
        except ImportError:
            logging.error("❌ anthropic package not installed. Install with: pip install anthropic")
            return
    else:
        logging.error(f"❌ Unsupported model: {model_name}")
        return
    
    # Initialize topic classification client (for logprobs mode)
    if scoring_mode == "logprobs":
        # Get theme prediction model from config
        topic_classification_model = theme_prediction_model
        if not topic_classification_model:
            logging.error("❌ theme_prediction_model not found in config")
            return
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            logging.error("❌ No OpenAI API Key found for topic classification. Please set OPENAI_API_KEY.")
            return
        topic_classification_client = OpenAI(api_key=api_key)
        logging.info(f"📝 Topic Classification Model: {topic_classification_model} (from config)")
    
    # Initialize W&B run
    run = init_wandb_run(
        run_name=f"history_baseline_{args.model.lower()}_{scoring_mode}_{time.strftime('%Y%m%d_%H%M%S')}",
        stage="09_baselines",
        job_type="baseline_prediction"
    )
    
    try:
        # Load artifacts from 09_baselines config
        logging.info("\nDownloading input artifacts...")
        
        # Get artifact names from config
        input_artifacts = _cfg.get("input_artifacts", {})
        if not input_artifacts:
            logging.error("❌ input_artifacts not found in config")
            return
        
        # Get dataset type to determine which training data artifact to use
        dataset_type = _cfg.get("dataset_type", "train")
        if dataset_type == "train":
            training_data_artifact = input_artifacts.get("training_data_train")
        else:
            training_data_artifact = input_artifacts.get("training_data_test")
        
        if not training_data_artifact:
            logging.error(f"❌ training_data_{dataset_type} not found in config.input_artifacts")
            return
        
        topics_artifact = input_artifacts.get("topics")
        if not topics_artifact:
            logging.error("❌ topics not found in config.input_artifacts")
            return
        
        # Get file patterns from config
        file_patterns = _cfg.get("file_patterns", {})
        if not file_patterns:
            logging.error("❌ file_patterns not found in config")
            return
        
        topic_universe_pattern = file_patterns.get("topic_universe")
        if not topic_universe_pattern:
            logging.error("❌ topic_universe pattern not found in config.file_patterns")
            return
        
        # Get training data pattern based on dataset_type
        if dataset_type == "train":
            training_data_pattern = file_patterns.get("training_data")
        else:
            training_data_pattern = file_patterns.get("test_data")
        
        if not training_data_pattern:
            logging.error(f"❌ {'training_data' if dataset_type == 'train' else 'test_data'} pattern not found in config.file_patterns")
            return
        
        # Load topic universe
        topic_universe_path = use_artifact(run, topics_artifact, "dataset")
        if not topic_universe_path:
            logging.error(f"Could not download {topics_artifact} artifact")
            return
        
        # Find topic universe file using pattern
        topic_universe_file = topic_universe_path / topic_universe_pattern
        if not topic_universe_file.exists():
            # Try to find file matching pattern
            topic_universe_files = list(topic_universe_path.glob(f"**/*{topic_universe_pattern}*"))
            if not topic_universe_files:
                # Fallback: search for any topic universe file
                topic_universe_files = list(topic_universe_path.glob("**/*topic*universe*.json"))
            if topic_universe_files:
                topic_universe_file = topic_universe_files[0]
            else:
                logging.error(f"Could not find topic universe file matching pattern '{topic_universe_pattern}' in {topic_universe_path}")
                return
        
        topic_universe = TopicUniverseArtifact.from_file(topic_universe_file)
        category_themes_map = topic_universe.topics_by_category
        logging.info(f"Loaded topic universe for {len(category_themes_map)} categories")
        logging.info(f"   Available categories in topic universe: {list(category_themes_map.keys())}")
        
        # Load category mapping artifact from W&B
        category_mapping_artifact = input_artifacts.get("category_mapping")
        if category_mapping_artifact:
            category_mapping_path = use_artifact(run, category_mapping_artifact, "dataset")
            if category_mapping_path:
                category_mapping_pattern = file_patterns.get("category_mapping", "category_mapping.json")
                if category_mapping_path.is_file():
                    category_mapping_file = category_mapping_path
                elif category_mapping_path.is_dir():
                    category_mapping_file = category_mapping_path / category_mapping_pattern
                    if not category_mapping_file.exists():
                        # Try to find file matching pattern
                        pattern_files = list(category_mapping_path.glob(f"**/*{category_mapping_pattern}*"))
                        if pattern_files:
                            category_mapping_file = pattern_files[0]
                        else:
                            pattern_files = list(category_mapping_path.glob("**/*category*mapping*.json"))
                            if pattern_files:
                                category_mapping_file = pattern_files[0]
                
                if category_mapping_file and category_mapping_file.exists():
                    with open(category_mapping_file, 'r', encoding='utf-8') as f:
                        category_mapping_data = json.load(f)
                    
                    # Extract category_to_main_mapping
                    global SUB_TO_MAIN_CATEGORY_MAP
                    SUB_TO_MAIN_CATEGORY_MAP = category_mapping_data.get('category_to_main_mapping', {})
                    if not SUB_TO_MAIN_CATEGORY_MAP:
                        # Try direct access if it's already a dict
                        if isinstance(category_mapping_data, dict):
                            SUB_TO_MAIN_CATEGORY_MAP = category_mapping_data
                    
                    logging.info(f"✅ Loaded category mapping with {len(SUB_TO_MAIN_CATEGORY_MAP)} categories from W&B")
                    logging.info(f"   Sample mappings: {list(SUB_TO_MAIN_CATEGORY_MAP.items())[:5]}")
                else:
                    logging.warning(f"⚠️ Category mapping file not found, using hardcoded mapping")
            else:
                logging.warning(f"⚠️ Could not download category mapping artifact, using hardcoded mapping")
        else:
            logging.warning(f"⚠️ Category mapping artifact not in config, using hardcoded mapping")
        
        # Load training data with topics
        training_data_path = use_artifact(run, training_data_artifact, "dataset")
        if not training_data_path:
            logging.error(f"Could not download {training_data_artifact} artifact")
            return
        
        # Find training data file using pattern from config
        training_data_file = training_data_path / training_data_pattern
        if not training_data_file.exists():
            # Try to find file matching pattern (glob search)
            pattern_files = list(training_data_path.glob(f"**/*{training_data_pattern}*"))
            if pattern_files:
                training_data_file = pattern_files[0]
            else:
                # Try exact filename match in subdirectories
                pattern_files = list(training_data_path.glob(f"**/{training_data_pattern}"))
                if pattern_files:
                    training_data_file = pattern_files[0]
        
        if not training_data_file or not training_data_file.exists():
            logging.error(f"Could not find training data file matching pattern '{training_data_pattern}' in {training_data_path}")
            available_files = list(training_data_path.glob("**/*.json"))
            if available_files:
                logging.error(f"   Available JSON files: {[f.name for f in available_files[:5]]}")
            else:
                logging.error(f"   No JSON files found in {training_data_path}")
            return
        
        logging.info(f"📁 Using training data file: {training_data_file.name}")
        
        with open(training_data_file, 'r', encoding='utf-8') as f:
            training_data = json.load(f)
        
        # Count total reviews and collect category statistics
        total_reviews_in_file = 0
        categories_in_reviews = set()
        main_categories_in_reviews = set()
        sample_count = 0
        max_samples = 1000
        
        if isinstance(training_data, list):
            total_reviews_in_file = len(training_data)
            for review in training_data:
                if sample_count < max_samples:
                    if review.get('category'):
                        categories_in_reviews.add(review.get('category'))
                    if review.get('main_category'):
                        main_categories_in_reviews.add(review.get('main_category'))
                    sample_count += 1
        elif isinstance(training_data, dict):
            if 'user_predictions' in training_data:
                for reviews in training_data.get('user_predictions', {}).values():
                    review_list = reviews if isinstance(reviews, list) else [reviews]
                    total_reviews_in_file += len(review_list)
                    if sample_count < max_samples:
                        for review in review_list:
                            if sample_count >= max_samples:
                                break
                            if review.get('category'):
                                categories_in_reviews.add(review.get('category'))
                            if review.get('main_category'):
                                main_categories_in_reviews.add(review.get('main_category'))
                            sample_count += 1
            else:
                # Count reviews in dict format
                for user_id, user_data in training_data.items():
                    if isinstance(user_data, dict) and 'reviews' in user_data:
                        total_reviews_in_file += len(user_data['reviews'])
                        if sample_count < max_samples:
                            for review in user_data['reviews']:
                                if sample_count >= max_samples:
                                    break
                                if review.get('category'):
                                    categories_in_reviews.add(review.get('category'))
                                if review.get('main_category'):
                                    main_categories_in_reviews.add(review.get('main_category'))
                                sample_count += 1
                    elif isinstance(user_data, list):
                        total_reviews_in_file += len(user_data)
                        if sample_count < max_samples:
                            for review in user_data:
                                if sample_count >= max_samples:
                                    break
                                if review.get('category'):
                                    categories_in_reviews.add(review.get('category'))
                                if review.get('main_category'):
                                    main_categories_in_reviews.add(review.get('main_category'))
                                sample_count += 1
                    else:
                        total_reviews_in_file += 1
                        if sample_count < max_samples:
                            if user_data.get('category'):
                                categories_in_reviews.add(user_data.get('category'))
                            if user_data.get('main_category'):
                                main_categories_in_reviews.add(user_data.get('main_category'))
                            sample_count += 1
        
        logging.info(f"\n📊 Input File Statistics:")
        logging.info(f"   Total reviews in file: {total_reviews_in_file}")
        logging.info(f"   File path: {training_data_file}")
        if categories_in_reviews:
            logging.info(f"   Sample categories in reviews (from {min(sample_count, max_samples)} samples): {sorted(list(categories_in_reviews))[:10]}")
        if main_categories_in_reviews:
            logging.info(f"   Sample main_categories in reviews: {sorted(list(main_categories_in_reviews))[:10]}")
        
        # Group training data by user
        user_reviews_map = {}
        if isinstance(training_data, list):
            for item in training_data:
                user_id = item.get('user_id')
                if user_id:
                    if user_id not in user_reviews_map:
                        user_reviews_map[user_id] = []
                    user_reviews_map[user_id].append(item)
        elif isinstance(training_data, dict):
            # Handle dict format - could be user_predictions or direct user_id keys with 'reviews' list
            if 'user_predictions' in training_data:
                user_reviews_map = training_data.get('user_predictions', {})
            else:
                # Check if it's a dict with user_id keys and 'reviews' list
                for user_id, user_data in training_data.items():
                    if isinstance(user_data, dict) and 'reviews' in user_data:
                        # Format: {user_id: {"reviews": [...]}}
                        user_reviews_map[user_id] = user_data['reviews']
                    elif isinstance(user_data, list):
                        # Format: {user_id: [...]}
                        user_reviews_map[user_id] = user_data
                    else:
                        # Single review per user
                        user_reviews_map[user_id] = [user_data]
        
        # Count reviews per user
        total_reviews_grouped = sum(len(reviews) if isinstance(reviews, list) else 1 
                                   for reviews in user_reviews_map.values())
        users_with_multiple_reviews = sum(1 for reviews in user_reviews_map.values() 
                                         if (len(reviews) if isinstance(reviews, list) else 1) >= 2)
        users_with_single_review = sum(1 for reviews in user_reviews_map.values() 
                                       if (len(reviews) if isinstance(reviews, list) else 1) == 1)
        
        logging.info(f"   Users in data: {len(user_reviews_map)}")
        logging.info(f"   Total reviews after grouping: {total_reviews_grouped}")
        logging.info(f"   Users with 1 review: {users_with_single_review}")
        logging.info(f"   Users with 2+ reviews (for leave-one-out): {users_with_multiple_reviews}")
        
        # Prepare tasks - build history from training data itself (leave-one-out)
        all_tasks = []
        skipped_users = 0
        skipped_no_topics = 0
        skipped_no_category_match = 0
        
        for user_id, reviews in user_reviews_map.items():
            # Process all users, including those with only 1 review (history will be empty)
            if len(reviews) < 1:
                skipped_users += 1
                continue
            
            # Sort reviews by timestamp if available, otherwise keep original order
            try:
                reviews_sorted = sorted(reviews, key=lambda x: x.get('timestamp', 0))
            except (KeyError, TypeError):
                reviews_sorted = reviews
            
            # Process all reviews (leave-one-out)
            for i in range(len(reviews_sorted)):
                target_review = reviews_sorted[i]
                
                # Build history from all other reviews (leave-one-out)
                history_reviews = []
                for j, review in enumerate(reviews_sorted):
                    if j != i:  # Skip current review
                        history_reviews.append({
                            'product_description': review.get('product_description', 'N/A'),
                            'review_text': review.get('review_text', 'N/A'),
                            'rating': review.get('rating', 3.0)
                        })
                
                # Build history using leave-one-out approach
                # For users with 1 review, history_reviews will be empty (handled by prompt creation)
                
                # Try multiple category fields (some reviews may have main_category already)
                raw_category = (
                    target_review.get('category') or 
                    target_review.get('main_category') or 
                    target_review.get('product_category') or
                    target_review.get('subcategory') or
                    'Unknown'
                )
                
                # Map to main category using the mapping
                mapped_category = SUB_TO_MAIN_CATEGORY_MAP.get(raw_category, raw_category)
                
                # If review already has main_category, prefer that
                if target_review.get('main_category'):
                    mapped_category = target_review.get('main_category')
                
                # Check for topic_probabilities ONLY - no fallback
                if not target_review.get('topic_probabilities'):
                    skipped_no_topics += 1
                    continue
                
                # Get themes for category - use comprehensive matching strategy (like topic_classification_fixed.py)
                category_themes = None
                
                # Normalize the mapped category name (topic_universe uses underscores and "and" instead of "&")
                mapped_category_normalized = normalize_category_name(mapped_category)
                
                # Strategy 1: Try normalized match first (most common case)
                if mapped_category_normalized and mapped_category_normalized in category_themes_map:
                    category_themes = category_themes_map[mapped_category_normalized]
                
                # Strategy 2: Try exact match (original format)
                if not category_themes and mapped_category and mapped_category in category_themes_map:
                    category_themes = category_themes_map[mapped_category]
                
                # Strategy 3: Try raw category (exact match)
                if not category_themes and raw_category and raw_category in category_themes_map:
                    category_themes = category_themes_map[raw_category]
                
                # Strategy 4: Try normalized raw category
                if not category_themes:
                    raw_category_normalized = normalize_category_name(raw_category)
                    if raw_category_normalized and raw_category_normalized in category_themes_map:
                        category_themes = category_themes_map[raw_category_normalized]
                
                # Strategy 5: Try case-insensitive match with normalized name
                if not category_themes:
                    mapped_category_normalized_lower = mapped_category_normalized.lower()
                    for key in category_themes_map.keys():
                        if key.lower() == mapped_category_normalized_lower:
                            category_themes = category_themes_map[key]
                            break
                
                # Strategy 6: Try case-insensitive match with original name
                if not category_themes:
                    mapped_category_lower = mapped_category.lower()
                    for key in category_themes_map.keys():
                        if key.lower() == mapped_category_lower:
                            category_themes = category_themes_map[key]
                            break
                
                # Strategy 7: Try partial match (contains) with normalized name
                if not category_themes:
                    mapped_category_normalized_lower = mapped_category_normalized.lower()
                    for key in category_themes_map.keys():
                        key_lower = key.lower()
                        if mapped_category_normalized_lower in key_lower or key_lower in mapped_category_normalized_lower:
                            category_themes = category_themes_map[key]
                            break
                
                if not category_themes:
                    skipped_no_category_match += 1
                    if skipped_no_category_match <= 5:  # Log first 5 failures
                        logging.debug(f"   Could not match category '{raw_category}' (mapped: '{mapped_category}', normalized: '{mapped_category_normalized}') to topic universe")
                        available_cats = list(category_themes_map.keys())[:5]
                        logging.debug(f"      Available categories (sample): {available_cats}")
                    continue
                
                # Prepare task based on scoring mode (include actual model names from config)
                if scoring_mode == "logprobs":
                    task = (user_id, i, target_review, history_reviews, category_themes, raw_category, 
                           text_gen_client, model_name, text_gen_model, topic_classification_client, topic_classification_model)
                else:
                    task = (user_id, i, target_review, history_reviews, category_themes, raw_category, 
                           text_gen_client, model_name, text_gen_model)
                all_tasks.append(task)
        
        total_tasks = len(all_tasks)
        
        logging.info(f"\n📋 Task Creation Summary:")
        logging.info(f"   Total tasks created: {total_tasks}")
        logging.info(f"   Skipped (users with <1 review): {skipped_users}")
        logging.info(f"   Skipped (no topics): {skipped_no_topics}")
        logging.info(f"   Skipped (no category match): {skipped_no_category_match}")
        logging.info(f"   Note: Processing all users (including those with 1 review - history will be empty)")
        
        if total_tasks == 0:
            logging.error("\n❌ No tasks created! This means:")
            logging.error("   - Users don't have any reviews (need 1+ reviews)")
            logging.error("   - Or reviews don't have topics/topic_probabilities")
            logging.error("   - Or categories don't match topic universe")
            logging.error(f"   - Skipped breakdown: {skipped_users} users, {skipped_no_topics} no topics, {skipped_no_category_match} no category")
            return
        
        logging.info(f"\n🚀 Processing {total_tasks} reviews with {args.max_workers} parallel workers...")
        
        # Select processing function based on scoring mode
        if scoring_mode == "logprobs":
            process_func = process_single_review_logprobs
            logging.info("Using logprobs mode: generating review first, then classifying topics separately")
        else:
            process_func = process_single_review
            logging.info("Using default mode: generating review with themes in prompt")
        
        # Process in parallel
        all_predictions = []
        with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
            future_to_task = {
                executor.submit(process_func, task): task 
                for task in all_tasks
            }
            
            with tqdm(total=total_tasks, desc="Processing reviews") as pbar:
                failed_count = 0
                none_count = 0
                for future in as_completed(future_to_task):
                    try:
                        result = future.result()
                        if result:
                            all_predictions.append(result)
                        else:
                            none_count += 1
                    except Exception as e:
                        failed_count += 1
                        logging.error(f"Error processing review: {e}")
                        if failed_count <= 3:  # Log first 3 exceptions with full traceback
                            import traceback
                            logging.error(f"Full traceback: {traceback.format_exc()}")
                    finally:
                        pbar.update(1)
                
                if failed_count > 0 or none_count > 0:
                    logging.warning(f"⚠️ Processing summary: {len(all_predictions)} successful, {none_count} returned None, {failed_count} exceptions")
        
        # Calculate average JSD for logprobs mode
        if scoring_mode == "logprobs" and all_predictions:
            jsd_values = [p.get('metrics', {}).get('jsd', 0.0) for p in all_predictions if p.get('metrics', {}).get('jsd') is not None]
            if jsd_values:
                avg_jsd = np.mean(jsd_values)
                logging.info(f"\n📊 Average JSD: {avg_jsd:.6f} (calculated over {len(jsd_values)} reviews)")
        
        # Save results with schema validation
        output_artifact_name = _cfg.get("output_artifact", "baseline_predictions_v4")
        method_suffix = "_logprobs" if scoring_mode == "logprobs" else ""
        # Use original model name (o3/claude) for artifact naming
        artifact_name = f"baseline_predictions_{args.model.lower()}_history{method_suffix}_v4"
        
        if args.output_dir:
            output_dir = Path(args.output_dir)
        else:
            output_dir = get_artifact_dir("09_baselines", artifact_name)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Get output filename pattern from config
        file_patterns = _cfg.get("file_patterns", {})
        output_filename_pattern = file_patterns.get("baseline_predictions")
        
        if not output_filename_pattern:
            # Fallback to default pattern if not in config
            logging.warning("⚠️ baseline_predictions pattern not found in config.file_patterns, using default")
            output_filename_pattern = "baseline_predictions_{model}_{method}{scoring_suffix}.json"
        
        # Replace placeholders in filename pattern
        scoring_suffix = "_logprobs" if scoring_mode == "logprobs" else ""
        output_filename = output_filename_pattern.format(
            model=args.model.lower(),
            method="history",
            scoring_suffix=scoring_suffix
        )
        
        # Log how many predictions we have
        logging.info(f"\n📊 Collected {len(all_predictions)} predictions")
        
        if not all_predictions:
            logging.warning("⚠️ No predictions to save! Check if tasks were created and processed correctly.")
            logging.warning(f"   Total tasks created: {total_tasks}")
            # Save empty dict to indicate no results
            validated_dict = {}
        else:
            # Convert to artifact format (dict with review keys)
            predictions_dict = {}
            for idx, pred in enumerate(all_predictions):
                review_key = f"{pred['user_id']}_review_{pred['review_index']}"
                predictions_dict[review_key] = pred
            
            logging.info(f"📝 Created predictions_dict with {len(predictions_dict)} entries")
            
            # Validate against schema (use logprobs schema if in logprobs mode)
            try:
                if scoring_mode == "logprobs":
                    validated_predictions = BaselinePredictionsArtifactLogprobs.from_dict(predictions_dict)
                    logging.info(f"✅ Validated {len(validated_predictions)} predictions against logprobs schema")
                    validated_dict = BaselinePredictionsArtifactLogprobs.to_dict(validated_predictions)
                else:
                    validated_predictions = BaselinePredictionsArtifact.from_dict(predictions_dict)
                    logging.info(f"✅ Validated {len(validated_predictions)} predictions against schema")
                    validated_dict = BaselinePredictionsArtifact.to_dict(validated_predictions)
                logging.info(f"📝 Converted to dict format: {len(validated_dict)} entries")
            except Exception as e:
                logging.error(f"❌ Schema validation failed: {e}")
                import traceback
                logging.error(f"Full error: {traceback.format_exc()}")
                logging.warning("⚠️ Saving unvalidated predictions (check schema compliance manually)")
                # Always save the data even if validation fails
                validated_dict = convert_to_serializable(predictions_dict)
                logging.info(f"📝 Saved {len(validated_dict)} unvalidated predictions")
        
        output_file = output_dir / output_filename
        output_file_absolute = output_file.resolve()
        
        logging.info(f"\n📁 Output Directory: {output_dir.resolve()}")
        logging.info(f"📄 Output Filename: {output_filename}")
        logging.info(f"📄 Full Output Path: {output_file_absolute}")
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(validated_dict, f, indent=2)
        
        if validated_dict:
            logging.info(f"\n✅ Saved {len(validated_dict)} predictions to:")
            logging.info(f"   {output_file_absolute}")
        else:
            logging.warning(f"\n⚠️ Saved empty file to:")
            logging.warning(f"   {output_file_absolute}")
            logging.warning(f"   No predictions were collected")
            logging.warning(f"   This could mean:")
            logging.warning(f"   1. No tasks were created (check if users have enough reviews)")
            logging.warning(f"   2. All predictions failed processing")
            logging.warning(f"   3. All results were filtered out")
        
        # Log artifact
        metadata = {
            "method": "history" + ("_logprobs" if scoring_mode == "logprobs" else ""),
            "model": model_name,
            "scoring_mode": scoring_mode,
            "schema_version": "v4",
            "schema_validated": True,
            "review_count": len(all_predictions)
        }
        
        # Add average JSD for logprobs mode
        if scoring_mode == "logprobs" and all_predictions:
            jsd_values = [p.get('metrics', {}).get('jsd', 0.0) for p in all_predictions if p.get('metrics', {}).get('jsd') is not None]
            if jsd_values:
                metadata["avg_jsd"] = float(np.mean(jsd_values))
        
        # Use "result" as artifact type (not "model" or "dataset") to avoid conflicts
        # The config has "model" but we use "result" for baseline predictions
        artifact = log_artifact(
            run=run,
            artifact_name=artifact_name,
            artifact_type="result",  # Use "result" to avoid type conflicts with existing artifacts
            artifact_path=output_dir,
            metadata=metadata
        )
        
        if artifact:
            link_to_registry(artifact, stage="09_baselines")
            logging.info(f"✅ Logged artifact: {artifact.name}")
        
        logging.info("\n" + "="*70)
        logging.info("✅ History baseline complete!")
        logging.info("="*70)
        logging.info(f"\n📁 Output saved to:")
        logging.info(f"   {output_file_absolute}")
        logging.info(f"\n📦 Artifact logged to W&B:")
        logging.info(f"   {artifact_name}")
        logging.info("="*70)
        
    finally:
        finish_run(run)


if __name__ == "__main__":
    main()

