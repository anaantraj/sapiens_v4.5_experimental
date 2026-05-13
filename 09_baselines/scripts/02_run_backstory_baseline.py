#!/usr/bin/env python3
"""
Stage 09: Backstory Baseline
=============================

Baseline prediction using persona backstory.
Based on Clustering/prediction_backstory_individual.py
"""

import os
import sys
import json
import logging
import time
import re
import argparse
import numpy as np
import math
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
from schemas.learned_artifacts.user_backstory import UserBackstoryArtifact
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

def create_prompt_backstory(backstory: str, target_review: dict, category_themes: list) -> str:
    """Create prompt using persona backstory."""
    # Load prompt template
    try:
        prompt_template = load_prompt("backstory_baseline_prompt.txt")
    except FileNotFoundError as e:
        logging.error(f"Could not load prompt template: {e}")
        return None
    
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
        backstory=backstory,
        new_prod_desc=new_prod_desc,
        themes_list=themes_list,
        themes_json_template=themes_json_template
    )
    
    return prompt

# =============================================================================
# LLM Interaction
# =============================================================================

def get_llm_prediction_o3(prompt: str, client, model_name: str) -> dict:
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

def get_llm_prediction_claude(prompt: str, client, model_name: str) -> dict:
    if not prompt:
        return {'review_text': 'Error: Invalid prompt.', 'rating': 3.0, 'sentiment': 'Neutral', 'predicted_themes': {}}
    
    max_retries = 3
    for attempt in range(max_retries):
        try:
            rate_limited_request()
            
            response = client.messages.create(
                model=model_name,
                max_tokens=4096,
                messages=[{"role": "user", "content": prompt}]
            )

            response_text = response.content[0].text

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

# =============================================================================
# Logprobs Mode Functions
# =============================================================================

def create_prompt_backstory_no_themes(backstory: str, target_review: dict) -> str:
    """Create prompt using persona backstory without theme prediction."""
    # Load prompt template
    try:
        prompt_template = load_prompt("backstory_baseline_prompt_no_themes.txt")
    except FileNotFoundError as e:
        logging.error(f"Could not load prompt template: {e}")
        return None
    
    new_prod_desc = target_review.get('product_description', 'N/A')
    
    # Format the prompt using the template
    prompt = prompt_template.format(
        backstory=backstory,
        new_prod_desc=new_prod_desc
    )
    
    return prompt

def get_llm_prediction_no_themes_o3(prompt: str, client, model_name: str) -> dict:
    """Get review prediction without themes using O3 (or configured model)."""
    if not prompt:
        return {'review_text': 'Error: Invalid prompt.', 'rating': 3.0, 'sentiment': 'Neutral'}
    
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
                'sentiment': prediction_json.get('sentiment', 'Neutral')
            }
            
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(5)
                continue
            logging.error(f"\n❌ GENERATION ERROR: {str(e)}")
            return {'review_text': 'Error: Failed to get prediction.', 'rating': 3.0, 'sentiment': 'Neutral'}

def get_llm_prediction_no_themes_claude(prompt: str, client, model_name: str) -> dict:
    """Get review prediction without themes using Claude (or configured model)."""
    if not prompt:
        return {'review_text': 'Error: Invalid prompt.', 'rating': 3.0, 'sentiment': 'Neutral'}
    
    max_retries = 3
    for attempt in range(max_retries):
        try:
            rate_limited_request()
            
            response = client.messages.create(
                model=model_name,
                max_tokens=4096,
                messages=[{"role": "user", "content": prompt}]
            )

            response_text = response.content[0].text

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

def get_topic_logprobs_o3(review_text: str, topic: str, client, model_name: str) -> dict:
    """Get yes/no logprobs for a topic using configured model (from config)."""
    system_msg = "You are a precise topic classifier. Answer only Yes or No."
    user_msg = f'Analyze this product review and determine if it discusses the topic "{topic}".\n\nReview: "{review_text}"\n\nDoes this review discuss the topic "{topic}"?\nAnswer with ONLY "Yes" or "No".'
    
    max_retries = 3
    for attempt in range(max_retries):
        try:
            rate_limited_request()
            
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
                
                # Find highest logprob for yes/no
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
                
                # Set defaults if not found
                if yes_logprob == -100.0:
                    yes_logprob = -10.0
                if no_logprob == -100.0:
                    no_logprob = -10.0
                
                # Convert to probabilities using softmax
                prob_yes = math.exp(yes_logprob)
                prob_no = math.exp(no_logprob)
                
                # Normalize
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
                    "logprob_no": no_logprob
                }
            
            except Exception as logprob_error:
                # Fallback to answer-based if logprobs not supported
                error_str = str(logprob_error)
                if "logprob" in error_str.lower() or "403" in error_str:
                    response = client.chat.completions.create(
                        model=model_name,
                        messages=[
                            {"role": "system", "content": system_msg},
                            {"role": "user", "content": user_msg}
                        ],
                        max_tokens=5,
                        temperature=0
                    )
                    
                    answer_text = response.choices[0].message.content.strip().lower()
                    is_yes = "yes" in answer_text or answer_text.startswith("y")
                    
                    prob_yes = 0.9 if is_yes else 0.1
                    prob_no = 0.1 if is_yes else 0.9
                    
                    return {
                        "yes": prob_yes,
                        "no": prob_no,
                        "logprob_yes": math.log(prob_yes) if prob_yes > 0 else -10.0,
                        "logprob_no": math.log(prob_no) if prob_no > 0 else -10.0
                    }
                else:
                    raise
        
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(2)
                continue
            logging.error(f"Error getting logprobs for topic '{topic}': {e}")
            return {
                "yes": 0.0,
                "no": 0.0,
                "logprob_yes": -10.0,
                "logprob_no": -10.0
            }
    
    return {
        "yes": 0.0,
        "no": 0.0,
        "logprob_yes": -10.0,
        "logprob_no": -10.0
    }

def get_topic_logprobs_claude(review_text: str, topic: str, client, model_name: str) -> dict:
    """Get yes/no logprobs for a topic using Claude."""
    # Claude doesn't support logprobs, so we use answer-based classification
    system_msg = "You are a precise topic classifier. Answer only Yes or No."
    user_msg = f'Analyze this product review and determine if it discusses the topic "{topic}".\n\nReview: "{review_text}"\n\nDoes this review discuss the topic "{topic}"?\nAnswer with ONLY "Yes" or "No".'
    
    max_retries = 3
    for attempt in range(max_retries):
        try:
            rate_limited_request()
            
            response = client.messages.create(
                model=model_name,
                max_tokens=5,
                messages=[
                    {"role": "user", "content": f"{system_msg}\n\n{user_msg}"}
                ]
            )
            
            answer_text = response.content[0].text.strip().lower()
            is_yes = "yes" in answer_text or answer_text.startswith("y")
            
            prob_yes = 0.9 if is_yes else 0.1
            prob_no = 0.1 if is_yes else 0.9
            
            return {
                "yes": prob_yes,
                "no": prob_no,
                "logprob_yes": math.log(prob_yes) if prob_yes > 0 else -10.0,
                "logprob_no": math.log(prob_no) if prob_no > 0 else -10.0
            }
        
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(2)
                continue
            logging.error(f"Error getting logprobs for topic '{topic}': {e}")
            return {
                "yes": 0.0,
                "no": 0.0,
                "logprob_yes": -10.0,
                "logprob_no": -10.0
            }
    
    return {
        "yes": 0.0,
        "no": 0.0,
        "logprob_yes": -10.0,
        "logprob_no": -10.0
    }

def normalize_distribution(theme_dict: Dict[str, float]) -> Dict[str, float]:
    """Normalize a theme probability distribution to sum to 1.0."""
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
    """Align two theme distributions to the same set of themes and normalize."""
    all_themes = sorted(set(actual_themes.keys()) | set(predicted_themes.keys()))
    
    if not all_themes:
        return np.array([]), np.array([])
    
    actual_array = np.array([actual_themes.get(theme, 0.0) for theme in all_themes])
    predicted_array = np.array([predicted_themes.get(theme, 0.0) for theme in all_themes])
    
    # Normalize
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
    """Compute Jensen-Shannon Divergence between two probability distributions."""
    P = P + epsilon
    Q = Q + epsilon
    
    P = P / P.sum()
    Q = Q / Q.sum()
    
    M = 0.5 * (P + Q)
    
    kl_pm = entropy(P, M, base=2)
    kl_qm = entropy(Q, M, base=2)
    
    jsd = 0.5 * kl_pm + 0.5 * kl_qm
    
    return float(jsd)

def calculate_jsd_for_review(prediction: dict, actual: dict) -> float:
    """Calculate JSD for a single review. Only uses topic_probabilities - no fallback."""
    # Get theme distributions - ONLY use topic_probabilities, no fallback
    actual_themes = actual.get('topic_probabilities', {})
    if not actual_themes:
        return 0.0
    
    # Must be a dict (topic_probabilities should always be a dict)
    if not isinstance(actual_themes, dict):
        return 0.0
    
    predicted_themes_dict = prediction.get('predicted_themes', {})
    
    # Handle dict format for predicted themes
    if not isinstance(predicted_themes_dict, dict):
        return 0.0
    
    if not actual_themes or not predicted_themes_dict:
        return 0.0
    
    # Normalize distributions
    actual_themes = normalize_distribution(actual_themes)
    predicted_themes_dict = normalize_distribution(predicted_themes_dict)
    
    if not actual_themes or not predicted_themes_dict:
        return 0.0
    
    # Align and compute JSD
    actual_array, predicted_array = align_distributions(actual_themes, predicted_themes_dict)
    
    if len(actual_array) == 0 or len(predicted_array) == 0:
        return 0.0
    
    jsd = compute_jsd(actual_array, predicted_array)
    return jsd

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
        if predicted_theme_normalized == actual_theme_norm:
            matches.add(actual_theme_norm)
            continue
        
        if predicted_theme_normalized in actual_theme_norm:
            matches.add(actual_theme_norm)
            continue
        
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
    
    # Count matches
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
# Parallel Processing
# =============================================================================

def process_single_review(args_tuple):
    """Process a single review prediction."""
    (user_id, review_index, target_review, backstory, category_themes, 
     raw_category, client, model_name, actual_model_name, _) = args_tuple
    
    prompt = create_prompt_backstory(backstory, target_review, category_themes)
    
    if not prompt:
        return None
    
    # Choose LLM call based on model (use actual model name from config)
    if model_name.lower() == "o3":
        prediction = get_llm_prediction_o3(prompt, client, actual_model_name)
    elif model_name.lower() == "claude":
        prediction = get_llm_prediction_claude(prompt, client, actual_model_name)
    else:
        return None
    
    metrics = calculate_enhanced_accuracy(prediction, target_review)
    
    # Format result to match BaselinePredictionItem schema
    result = {
        'user_id': user_id,
        'review_index': review_index,
        'product_description': target_review.get('product_description', 'N/A'),
        'category': raw_category,
        'model': 'claude' if 'claude' in model_name.lower() else 'o3',  # Ensure valid model name
        'method': 'backstory',
        'prediction': {
            'review_text': prediction.get('review_text', ''),
            'rating': float(prediction.get('rating', 3.0)),
            'sentiment': prediction.get('sentiment', 'Neutral'),
            'predicted_themes': prediction.get('predicted_themes', {})
        },
        'actual': {
            'review_text': target_review.get('review_text', ''),
            'rating': float(target_review.get('rating', 3.0)),
            'sentiment': target_review.get('sentiment', 'Neutral'),
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
    """Process a single review using logprobs mode."""
    (user_id, review_index, target_review, backstory, category_themes, 
     raw_category, client, model_name, actual_model_name, theme_model_name) = args_tuple
    
    # Step 1: Generate review without themes
    prompt = create_prompt_backstory_no_themes(backstory, target_review)
    
    if not prompt:
        return None
    
    # Choose LLM call based on model (use actual model name from config)
    if model_name.lower() == "o3":
        review_prediction = get_llm_prediction_no_themes_o3(prompt, client, actual_model_name)
    elif model_name.lower() == "claude":
        review_prediction = get_llm_prediction_no_themes_claude(prompt, client, actual_model_name)
    else:
        return None
    
    generated_review_text = review_prediction.get('review_text', '')
    
    if not generated_review_text or generated_review_text.startswith('Error:'):
        return None
    
    # Step 2: Get logprobs for each topic
    # Always use configured theme model for topic classification, regardless of main model
    # Create OpenAI client for theme model if main model is Claude
    topic_classification_client = client
    if model_name.lower() == "claude":
        # Need OpenAI client for theme model topic classification
        openai_api_key = os.environ.get("OPENAI_API_KEY")
        if not openai_api_key:
            logging.error("❌ No OpenAI API Key found for topic classification. Please set OPENAI_API_KEY.")
            return None
        topic_classification_client = OpenAI(api_key=openai_api_key)
    
    topic_scores = {}
    topic_logprobs = {}
    
    for topic in category_themes:
        # Always use configured theme model for topic classification
        result = get_topic_logprobs_o3(generated_review_text, topic, topic_classification_client, theme_model_name)
        
        topic_scores[topic] = result.get("yes", 0.0)
        topic_logprobs[topic] = {
            "logprob_yes": result.get("logprob_yes", -10.0),
            "logprob_no": result.get("logprob_no", -10.0)
        }
    
    # Step 3: Normalize topic probabilities using softmax on log probabilities
    if len(topic_scores) > 0:
        logprob_values = [topic_logprobs[topic]["logprob_yes"] for topic in topic_scores.keys()]
        
        # Apply numerically stable softmax
        max_logprob = max(logprob_values) if logprob_values else 0.0
        exp_logprobs = [math.exp(logprob - max_logprob) for logprob in logprob_values]
        sum_exp = sum(exp_logprobs)
        
        if sum_exp > 0:
            softmax_probs = [exp_logprob / sum_exp for exp_logprob in exp_logprobs]
        else:
            softmax_probs = [1.0 / len(logprob_values)] * len(logprob_values)
        
        normalized_topic_scores = {
            topic: softmax_prob
            for topic, softmax_prob in zip(topic_scores.keys(), softmax_probs)
        }
    else:
        normalized_topic_scores = {}
    
    # Step 4: Calculate JSD
    prediction_with_themes = {
        'review_text': generated_review_text,
        'rating': float(review_prediction.get('rating', 3.0)),
        'sentiment': review_prediction.get('sentiment', 'Neutral'),
        'predicted_themes': normalized_topic_scores
    }
    
    jsd = calculate_jsd_for_review(prediction_with_themes, target_review)
    
    # Format result
    result = {
        'user_id': user_id,
        'review_index': review_index,
        'product_description': target_review.get('product_description', 'N/A'),
        'category': raw_category,
        'model': 'claude' if 'claude' in model_name.lower() else 'o3',
        'method': 'backstory_logprobs',
        'prediction': {
            'review_text': generated_review_text,
            'rating': float(review_prediction.get('rating', 3.0)),
            'sentiment': review_prediction.get('sentiment', 'Neutral'),
            'predicted_themes': normalized_topic_scores
        },
        'actual': {
            'review_text': target_review.get('review_text', ''),
            'rating': float(target_review.get('rating', 3.0)),
            'sentiment': target_review.get('sentiment', 'Neutral'),
            'topic_probabilities': target_review.get('topic_probabilities', {}),
            'themes': target_review.get('themes', target_review.get('predicted_themes', []))
        },
        'metrics': {
            'jsd': jsd  # Only metric calculated in logprobs mode
        }
    }
    
    return result

# =============================================================================
# Main Execution
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Backstory Baseline Prediction")
    parser.add_argument("--model", type=str, default="o3", choices=["o3", "claude"],
                       help="Model to use: 'o3' or 'claude'")
    parser.add_argument("--max-workers", type=int, default=20,
                       help="Number of parallel workers")
    parser.add_argument("--output-dir", type=str, default=None,
                       help="Output directory (default: auto)")
    parser.add_argument("--scoring-mode", type=str, default="default", choices=["default", "logprobs"],
                       help="Scoring mode: 'default' (standard) or 'logprobs' (logprobs-based)")
    args = parser.parse_args()
    
    model_name = args.model.lower()
    scoring_mode = args.scoring_mode.lower()

    logging.info(f"🚀 Starting Backstory Baseline Prediction Pipeline")
    logging.info(f"📊 Model: {model_name} | Workers: {args.max_workers} | Scoring Mode: {scoring_mode}")

    # Get model names from config
    review_prediction_models = _cfg.get("review_prediction_models", {})
    o3_model_name = review_prediction_models.get("o3")
    claude_model_name = review_prediction_models.get("claude")
    theme_prediction_model = _cfg.get("theme_prediction_model")
    if scoring_mode == "logprobs" and not theme_prediction_model:
        logging.error("❌ theme_prediction_model not found in config")
        return
    
    # Initialize client
    if model_name == "o3":
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            logging.error("❌ No OpenAI API Key found. Please set OPENAI_API_KEY.")
            return
        if scoring_mode == "logprobs":
            client = OpenAI(api_key=api_key)
            if not o3_model_name:
                logging.error("❌ review_prediction_models.o3 not found in config")
                return
            actual_model_name = o3_model_name
        else:
            client = create_openai_client(openai_config=_openai_cfg, timeout=120.0)
            actual_model_name = _cfg.get("bedrock_model_id") if "bedrock-mantle" in (os.environ.get("OPENAI_BASE_URL") or "") and _cfg.get("bedrock_model_id") else o3_model_name
            if not actual_model_name:
                logging.error("❌ Missing model config for o3/Bedrock in 09_baselines config")
                return
    elif model_name == "claude":
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            logging.error("❌ No Anthropic API Key found. Please set ANTHROPIC_API_KEY.")
            return
        try:
            import anthropic
            client = anthropic.Anthropic(api_key=api_key)
            if not claude_model_name:
                logging.error("❌ review_prediction_models.claude not found in config")
                return
            actual_model_name = claude_model_name
        except ImportError:
            logging.error("❌ anthropic package not installed. Install with: pip install anthropic")
            return
    else:
        logging.error(f"❌ Unsupported model: {model_name}")
        return
    
    # Initialize W&B run
    run = init_wandb_run(
        run_name=f"backstory_baseline_{model_name}_{time.strftime('%Y%m%d_%H%M%S')}",
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
        
        # Get file patterns from config
        file_patterns = _cfg.get("file_patterns", {})
        if not file_patterns:
            logging.error("❌ file_patterns not found in config")
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
        
        backstory_artifact = input_artifacts.get("user_backstories")
        if not backstory_artifact:
            logging.error("❌ user_backstories not found in config.input_artifacts")
            return
        
        # Get file patterns
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
        
        user_backstories_pattern = file_patterns.get("user_backstories")
        if not user_backstories_pattern:
            logging.error("❌ user_backstories pattern not found in config.file_patterns")
            return
        
        # Load topic universe
        topic_universe_path = use_artifact(run, topics_artifact, "dataset")
        if not topic_universe_path:
            logging.error(f"Could not download {topics_artifact} artifact")
            return
        
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
        
        # Load user backstories from WandB
        user_backstory_path = use_artifact(run, backstory_artifact, "dataset")
        if not user_backstory_path:
            logging.error(f"Could not download {backstory_artifact} artifact from WandB")
            return
        
        # Find backstory file using pattern from config
        user_backstory_file = user_backstory_path / user_backstories_pattern
        if not user_backstory_file.exists():
            # Try to find file matching pattern (glob search)
            pattern_files = list(user_backstory_path.glob(f"**/*{user_backstories_pattern}*"))
            if pattern_files:
                user_backstory_file = pattern_files[0]
            else:
                # Try exact filename match in subdirectories
                pattern_files = list(user_backstory_path.glob(f"**/{user_backstories_pattern}"))
                if pattern_files:
                    user_backstory_file = pattern_files[0]
                else:
                    # Fallback: search for any backstory file
                    backstory_files = list(user_backstory_path.glob("**/*user*backstory*.json"))
                    if not backstory_files:
                        backstory_files = list(user_backstory_path.glob("**/*overall*characteristics*.json"))
                    if backstory_files:
                        user_backstory_file = backstory_files[0]
        
        if not user_backstory_file or not user_backstory_file.exists():
            logging.error(f"Could not find backstory file matching pattern '{user_backstories_pattern}' in {user_backstory_path}")
            available_files = list(user_backstory_path.glob("**/*.json"))
            if available_files:
                logging.error(f"   Available JSON files: {[f.name for f in available_files[:5]]}")
            else:
                logging.error(f"   No JSON files found in {user_backstory_path}")
            return
        
        logging.info(f"📁 Using backstory file: {user_backstory_file.name}")
        
        user_backstories = UserBackstoryArtifact.from_file(user_backstory_file)
        logging.info(f"Loaded backstories for {len(user_backstories)} users")
        
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
        
        # Prepare tasks
        all_tasks = []
        skipped_users = 0
        skipped_no_topics = 0
        skipped_no_category_match = 0
        skipped_no_backstory = 0
        
        for user_id, reviews in user_reviews_map.items():
            # Ensure reviews is a list (handle dict format from user_predictions)
            if isinstance(reviews, dict):
                # If reviews is a dict (format: {review_key: review_data}), convert to list
                reviews = list(reviews.values())
            elif not isinstance(reviews, list):
                # If it's a single review, wrap in list
                reviews = [reviews]
            
            if len(reviews) < 1:
                skipped_users += 1
                continue
            
            # Get backstory
            if user_id not in user_backstories:
                skipped_no_backstory += 1
                continue
            backstory_artifact = user_backstories[user_id]
            # Extract backstory text from the artifact
            general_chars = backstory_artifact.overall_characteristics.influencing_characteristics_summary
            backstory = general_chars
            
            # Process all reviews per user
            for i in range(len(reviews)):
                target_review = reviews[i]
                
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
                
                # Get themes for category - use comprehensive matching strategy (like history baseline)
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
                
                # Prepare task with model names from config
                # Prepare task with model names from config
                task = (user_id, i, target_review, backstory, category_themes, raw_category, 
                       client, model_name, actual_model_name, theme_prediction_model if scoring_mode == "logprobs" else None)
                all_tasks.append(task)
        
        total_tasks = len(all_tasks)
        
        logging.info(f"\n📋 Task Creation Summary:")
        logging.info(f"   Total tasks created: {total_tasks}")
        logging.info(f"   Skipped (users with <1 review): {skipped_users}")
        logging.info(f"   Skipped (no backstory): {skipped_no_backstory}")
        logging.info(f"   Skipped (no topics): {skipped_no_topics}")
        logging.info(f"   Skipped (no category match): {skipped_no_category_match}")
        
        if total_tasks == 0:
            logging.error("\n❌ No tasks created! This means:")
            logging.error("   - Users don't have any reviews (need 1+ reviews)")
            logging.error("   - Or reviews don't have topic_probabilities")
            logging.error("   - Or categories don't match topic universe")
            logging.error("   - Or users don't have backstories")
            logging.error(f"   - Skipped breakdown: {skipped_users} users, {skipped_no_backstory} no backstory, {skipped_no_topics} no topics, {skipped_no_category_match} no category")
            return
        
        # Set up checkpoint file for resuming
        method_suffix = "logprobs" if scoring_mode == "logprobs" else ""
        artifact_name = f"baseline_predictions_{model_name}_backstory{'_' + method_suffix if method_suffix else ''}_v4"
        
        if args.output_dir:
            output_dir = Path(args.output_dir)
        else:
            output_dir = get_artifact_dir("09_baselines", artifact_name)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        checkpoint_file = output_dir / f"checkpoint_{artifact_name}.json"
        
        # Load existing checkpoint if it exists
        processed_keys = set()
        checkpoint_predictions = {}
        if checkpoint_file.exists():
            try:
                with open(checkpoint_file, 'r', encoding='utf-8') as f:
                    checkpoint_data = json.load(f)
                    checkpoint_predictions = checkpoint_data.get('predictions', {})
                    processed_keys = set(checkpoint_predictions.keys())
                    logging.info(f"📂 Loaded checkpoint: {len(processed_keys)} reviews already processed")
                    logging.info(f"   Checkpoint file: {checkpoint_file}")
            except Exception as e:
                logging.warning(f"⚠️ Could not load checkpoint file: {e}. Starting fresh.")
                checkpoint_predictions = {}
                processed_keys = set()
        
        # Filter out already processed tasks
        remaining_tasks = []
        for task in all_tasks:
            user_id, review_index, _, _, _, _, _, _, _, _ = task
            review_key = f"{user_id}_review_{review_index}"
            if review_key not in processed_keys:
                remaining_tasks.append(task)
        
        remaining_count = len(remaining_tasks)
        skipped_count = len(processed_keys)
        
        if remaining_count == 0:
            logging.info(f"✅ All {total_tasks} reviews already processed. Loading from checkpoint...")
            all_predictions = list(checkpoint_predictions.values())
        else:
            logging.info(f"\n🚀 Processing {remaining_count} remaining reviews (skipping {skipped_count} already processed)")
            logging.info(f"   Total tasks: {total_tasks} | Remaining: {remaining_count} | Already done: {skipped_count}")
            
            # Choose processing function based on scoring mode
            if scoring_mode == "logprobs":
                process_func = process_single_review_logprobs
            else:
                process_func = process_single_review
            
            # Start with existing predictions from checkpoint
            all_predictions = list(checkpoint_predictions.values())
            initial_count = len(all_predictions)
            completed_count = initial_count
            checkpoint_interval = 10  # Save checkpoint every 10 reviews
            last_checkpoint_count = initial_count
            
            def save_checkpoint():
                """Helper function to save checkpoint."""
                checkpoint_dict = {}
                for pred in all_predictions:
                    review_key = f"{pred['user_id']}_review_{pred['review_index']}"
                    checkpoint_dict[review_key] = pred
                
                checkpoint_data = {
                    'predictions': checkpoint_dict,
                    'total_processed': len(all_predictions),
                    'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
                }
                
                with open(checkpoint_file, 'w', encoding='utf-8') as f:
                    json.dump(convert_to_serializable(checkpoint_data), f, indent=2)
                
                logging.info(f"💾 Checkpoint saved: {len(all_predictions)}/{total_tasks} reviews processed")
            
            # Process remaining tasks in parallel
            with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
                future_to_task = {
                    executor.submit(process_func, task): task 
                    for task in remaining_tasks
                }
                
                with tqdm(total=remaining_count, initial=0, desc="Processing reviews") as pbar:
                    for future in as_completed(future_to_task):
                        try:
                            result = future.result()
                            if result:
                                all_predictions.append(result)
                                completed_count += 1
                                
                                # Save checkpoint every N reviews (or immediately if first new review)
                                new_reviews_count = completed_count - initial_count
                                if new_reviews_count == 1 or (new_reviews_count % checkpoint_interval == 0):
                                    save_checkpoint()
                                    last_checkpoint_count = completed_count
                        except Exception as e:
                            logging.error(f"Error processing review: {e}")
                        finally:
                            pbar.update(1)
            
            # Final checkpoint save (only if we processed new reviews)
            if completed_count > last_checkpoint_count:
                # Mark as completed if all tasks are done
                checkpoint_dict = {}
                for pred in all_predictions:
                    review_key = f"{pred['user_id']}_review_{pred['review_index']}"
                    checkpoint_dict[review_key] = pred
                
                checkpoint_data = {
                    'predictions': checkpoint_dict,
                    'total_processed': len(all_predictions),
                    'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                    'completed': len(all_predictions) >= total_tasks
                }
                
                with open(checkpoint_file, 'w', encoding='utf-8') as f:
                    json.dump(convert_to_serializable(checkpoint_data), f, indent=2)
                
                if len(all_predictions) >= total_tasks:
                    logging.info(f"💾 Final checkpoint saved: {len(all_predictions)}/{total_tasks} reviews processed (COMPLETED)")
                else:
                    logging.info(f"💾 Final checkpoint saved: {len(all_predictions)}/{total_tasks} reviews processed")
        
        # Calculate average JSD for logprobs mode
        avg_jsd = None
        if scoring_mode == "logprobs":
            jsd_values = [pred.get('metrics', {}).get('jsd', 0.0) for pred in all_predictions if pred.get('metrics', {}).get('jsd') is not None]
            if jsd_values:
                avg_jsd = float(np.mean(jsd_values))
                logging.info(f"\n📊 Average JSD: {avg_jsd:.6f} (calculated from {len(jsd_values)} reviews)")
        
        # Save results with schema validation
        # (output_dir and artifact_name already set above for checkpoint)
        
        # Convert to artifact format (dict with review keys)
        predictions_dict = {}
        for idx, pred in enumerate(all_predictions):
            review_key = f"{pred['user_id']}_review_{pred['review_index']}"
            predictions_dict[review_key] = pred
        
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
        except Exception as e:
            logging.error(f"❌ Schema validation failed: {e}")
            logging.error("Saving unvalidated predictions (check schema compliance manually)")
            validated_dict = convert_to_serializable(predictions_dict)
        
        output_file = output_dir / f"baseline_predictions_{model_name}_backstory{'_' + method_suffix if method_suffix else ''}.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(validated_dict, f, indent=2)
        
        logging.info(f"\n✅ Saved {len(all_predictions)} predictions to {output_file}")
        
        # Prepare metadata
        metadata = {
            "method": "backstory_logprobs" if scoring_mode == "logprobs" else "backstory",
            "model": model_name,
            "scoring_mode": scoring_mode,
            "schema_version": "v4",
            "schema_validated": True,
            "review_count": len(all_predictions)
        }
        
        if avg_jsd is not None:
            metadata["avg_jsd"] = avg_jsd
        
        # Log artifact
        artifact = log_artifact(
            run=run,
            artifact_name=artifact_name,
            artifact_type=_cfg.get("artifact_type", "result"),
            artifact_path=output_dir,
            metadata=metadata
        )
        
        if artifact:
            link_to_registry(artifact, stage="09_baselines")
            logging.info(f"✅ Logged artifact: {artifact.name}")
        
        logging.info("\n[OK] Backstory baseline complete!")
        
    finally:
        finish_run(run)


if __name__ == "__main__":
    main()

