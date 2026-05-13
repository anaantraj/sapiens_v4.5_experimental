#!/usr/bin/env python3
"""
Stage 09: Baselines
====================

Runs baseline predictions using review history or persona backstory.
- Uses configuration from 09_baselines/config.yaml
- Reads artifacts from modules via W&B
- Validates results against BaselinePredictionsArtifact schema

Supports:
- Methods: history (review history), backstory (persona backstory)
- Models: o3, claude
"""

import os
import sys
import json
import logging
import time
import re
import argparse
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

# Import W&B utilities
from utils.openai_client import create_openai_client
from utils.wandb_utils import (
    get_stage_config, get_openai_config,
    init_wandb_run, finish_run, use_artifact, log_artifact,
    log_metrics, log_summary, link_to_registry, get_artifact_dir
)

# Import schemas
from schemas.learned_artifacts.user_backstory import UserBackstoryArtifact
from schemas.learned_artifacts.user_review_history import UserReviewHistoryArtifact
from schemas.learned_artifacts.topic_universe import TopicUniverseArtifact
from schemas.learned_artifacts.baseline_predictions import (
    BaselinePredictionsArtifact,
    BaselinePredictionItem
)

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# =============================================================================
# Configuration
# =============================================================================

# Load config from 09_baselines
_cfg = get_stage_config("09_baselines")
_openai_cfg = get_openai_config()

# Rate limiting
rate_limit_lock = threading.Lock()
last_request_time = [0.0]
# Rate limiting - from config (no hardcoded default)
hyperparams = _cfg.get("hyperparameters", {})
MIN_REQUEST_INTERVAL = hyperparams.get("min_request_interval")
if MIN_REQUEST_INTERVAL is None:
    logging.warning("⚠️  min_request_interval not specified in hyperparameters, using 0.1")
    MIN_REQUEST_INTERVAL = 0.1

# Category mapping - will be loaded from W&B artifact (no hardcoded values)
SUB_TO_MAIN_CATEGORY_MAP = None  # Will be loaded from W&B

# =============================================================================
# Utility Functions
# =============================================================================

def load_prompt(prompt_name: str) -> Optional[str]:
    """Load a prompt template by name from config-specified location."""
    prompt_config = _cfg.get("prompt", {})
    prompt_dir_name = prompt_config.get("directory", "prompts")  # Default for prompt directory is acceptable
    
    # Try 10_running_simulations/baselines/prompts first (shared location)
    prompt_path = Path(__file__).parent.parent.parent / "10_running_simulations" / "baselines" / "prompts" / f"{prompt_name}_prompt.txt"
    
    if not prompt_path.exists():
        # Try local prompts directory
        prompt_path = Path(__file__).parent.parent / prompt_dir_name / f"{prompt_name}_prompt.txt"
    
    if prompt_path.exists():
        with open(prompt_path, 'r', encoding='utf-8') as f:
            return f.read()
    
    logging.error(f"Prompt file not found: {prompt_name}_prompt.txt")
    logging.error(f"  Tried: {Path(__file__).parent.parent.parent / '10_running_simulations' / 'baselines' / 'prompts'}")
    logging.error(f"  Tried: {Path(__file__).parent.parent / prompt_dir_name}")
    return None

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

def create_prompt_history(history_reviews: List[Dict], target_review: Dict, category_themes: List[str]) -> Optional[str]:
    """Create prompt with review history."""
    if not history_reviews:
        return None
    
    prompt_template = load_prompt("history")
    if not prompt_template:
        logging.error("History prompt template not found")
        return None
    
    # Build history text block
    history_text_block = ""
    for idx, rev in enumerate(history_reviews):
        prod = rev.get('product_description', 'N/A')
        text = rev.get('review_text', 'N/A')
        rating = rev.get('rating', 'N/A')
        history_text_block += f"Example {idx+1}:\nProduct: {prod}\nMy Rating: {rating}\nMy Review: {text}\n\n"
    
    new_prod_desc = target_review.get('product_description', 'N/A')
    themes_list = "\n".join([f"- {theme}" for theme in category_themes])
    themes_json_template = "\n".join([
        f'    "{theme}": <float 0.0-1.0>,' if i < len(category_themes) - 1 
        else f'    "{theme}": <float 0.0-1.0>'
        for i, theme in enumerate(category_themes)
    ])
    
    return prompt_template.format(
        history_text_block=history_text_block.strip(),
        new_prod_desc=new_prod_desc,
        themes_list=themes_list,
        themes_json_template=themes_json_template
    )

def create_prompt_backstory(backstory: str, target_review: Dict, category_themes: List[str]) -> Optional[str]:
    """Create prompt using persona backstory."""
    prompt_template = load_prompt("backstory")
    if not prompt_template:
        logging.error("Backstory prompt template not found")
        return None
    
    new_prod_desc = target_review.get('product_description', 'N/A')
    themes_list = "\n".join([f"- {theme}" for theme in category_themes])
    themes_json_template = "\n".join([
        f'    "{theme}": <float 0.0-1.0>,' if i < len(category_themes) - 1 
        else f'    "{theme}": <float 0.0-1.0>'
        for i, theme in enumerate(category_themes)
    ])
    
    return prompt_template.format(
        backstory=backstory,
        new_prod_desc=new_prod_desc,
        themes_list=themes_list,
        themes_json_template=themes_json_template
    )

# =============================================================================
# LLM Interaction
# =============================================================================

def get_llm_prediction_o3(prompt: str, client: OpenAI, model_name: str) -> Optional[Dict[str, Any]]:
    """Get prediction from configured OpenAI-compatible model. Returns None if all retries fail."""
    if not prompt:
        return None
    
    max_retries = 3
    for attempt in range(max_retries):
        try:
            rate_limited_request()
            
            response = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt}]
            )
            
            response_text = response.choices[0].message.content
            
            # Robust parsing
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
            logging.error(f"❌ GENERATION ERROR after {max_retries} retries: {str(e)}")
            return None  # Return None to indicate failure

def get_llm_prediction_claude(prompt: str, client, model_name: str) -> Optional[Dict[str, Any]]:
    """Get prediction from configured Claude model. Returns None if all retries fail."""
    if not prompt:
        return None
    
    try:
        import anthropic
    except ImportError:
        logging.error("anthropic package not installed. Install with: pip install anthropic")
        return None
    
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
            
            # Robust parsing
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
            logging.error(f"❌ GENERATION ERROR after {max_retries} retries: {str(e)}")
            return None  # Return None to indicate failure

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
        
        # Substring matches
        if predicted_theme_normalized in actual_theme_norm or actual_theme_norm in predicted_theme_normalized:
            matches.add(actual_theme_norm)
            continue
        
        # Word-level overlap
        pred_words = set(predicted_theme_normalized.replace(',', ' ').replace('&', ' ').split())
        actual_words = set(actual_theme_norm.replace(',', ' ').replace('&', ' ').split())
        stop_words = {'and', 'or', 'the', 'a', 'an', 'of', 'for', 'in', 'on', 'at', 'to', 'with'}
        pred_words = pred_words - stop_words
        actual_words = actual_words - stop_words
        
        if len(pred_words) > 0 and len(actual_words) > 0:
            overlap = pred_words.intersection(actual_words)
            min_overlap = max(2, min(len(pred_words), len(actual_words)) * 0.5)
            if len(overlap) >= min_overlap:
                matches.add(actual_theme_norm)
    
    return matches

def calculate_enhanced_accuracy(prediction: Dict, actual: Dict) -> Dict[str, float]:
    """Calculate prediction accuracy metrics."""
    actual_themes_list = actual.get('themes', [])
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
    
    # Rating and sentiment scores
    actual_rating = actual.get('rating', 3.0)
    pred_rating = prediction.get('rating', 3.0)
    rating_diff = abs(pred_rating - actual_rating)
    rating_score = max(0, 1 - (rating_diff / 4.0))
    
    sentiment_score = 1.0 if prediction.get('sentiment', '').lower() == actual.get('sentiment', '').lower() else 0.0
    
    # Overall accuracy
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
# Main Processing
# =============================================================================

def process_single_review(args_tuple: Tuple) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
    """
    Process a single review prediction.
    
    Returns:
        Tuple of (successful_result, failed_prediction_data)
        - successful_result: Dict if prediction succeeded, None if failed
        - failed_prediction_data: Dict with failure info if failed, None if succeeded
    """
    (user_id, review_index, target_review, history_reviews, backstory, category_themes,
     raw_category, client, model_name, actual_model_name, method, user_micro_info) = args_tuple
    
    # Choose prompt based on method
    if method == "history":
        prompt = create_prompt_history(history_reviews, target_review, category_themes)
    elif method == "backstory":
        prompt = create_prompt_backstory(backstory, target_review, category_themes)
    else:
        return None, None
    
    if not prompt:
        return None, None
    
    # Choose LLM call based on model
    if model_name.lower() == "o3":
        prediction = get_llm_prediction_o3(prompt, client, actual_model_name)
    elif model_name.lower() == "claude":
        prediction = get_llm_prediction_claude(prompt, client, actual_model_name)
    else:
        return None, None
    
    # Check if prediction failed (None indicates failure after retries)
    if prediction is None:
        # Return failed prediction data for separate file
        failed_data = {
            'user_id': user_id,
            'review_index': review_index,
            'product_description': target_review.get('product_description', 'N/A'),
            'category': raw_category,
            'method': method,
            'model': 'claude' if 'claude' in model_name.lower() else 'o3',
            'actual': {
                'review_text': target_review.get('review_text', ''),
                'rating': float(target_review.get('rating', 3.0)),
                'sentiment': target_review.get('sentiment', 'Neutral'),
                'themes': target_review.get('themes', target_review.get('predicted_themes', []))
            },
            'failure_reason': 'llm_call_failed_after_retries',
            'timestamp': time.time()
        }
        return None, failed_data
    
    # Calculate metrics
    metrics = calculate_enhanced_accuracy(prediction, target_review)
    
    # Determine review type
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
        'method': method,
        'model': 'claude' if 'claude' in model_name.lower() else 'o3',  # Ensure valid model name
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
    
    # Add cluster info if available
    if user_micro_info:
        result['cluster_name'] = user_micro_info.get('cluster', 'unknown')
        result['micro_cluster_id'] = user_micro_info.get('micro_cluster_id', 'unknown')
        result['persona_name'] = user_micro_info.get('persona_name', 'Unknown')
    
    return result, None

def run_baseline(model_name: str, method: str, max_workers: int, output_dir: Path, run):
    """Run baseline for a specific model and method combination."""
    logging.info(f"🚀 Starting Baseline: Model={model_name}, Method={method}")
    review_prediction_models = _cfg.get("review_prediction_models", {})
    o3_model_name = review_prediction_models.get("o3")
    claude_model_name = review_prediction_models.get("claude")
    
    # Initialize client
    if model_name == "o3":
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            logging.error("❌ No OpenAI API Key found. Please set OPENAI_API_KEY.")
            return None
        client = create_openai_client(openai_config=_openai_cfg, timeout=120.0)
        actual_model_name = _cfg.get("bedrock_model_id") if "bedrock-mantle" in (os.environ.get("OPENAI_BASE_URL") or "") and _cfg.get("bedrock_model_id") else o3_model_name
        if not actual_model_name:
            logging.error("❌ Missing model config for o3/Bedrock in 09_baselines config")
            return None
    elif model_name == "claude":
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            logging.error("❌ No Anthropic API Key found. Please set ANTHROPIC_API_KEY.")
            return None
        try:
            import anthropic
            client = anthropic.Anthropic(api_key=api_key)
            actual_model_name = claude_model_name
            if not actual_model_name:
                logging.error("❌ review_prediction_models.claude not found in config")
                return None
        except ImportError:
            logging.error("❌ anthropic package not installed. Install with: pip install anthropic")
            return None
    else:
        logging.error(f"❌ Unsupported model: {model_name}. Use 'o3' or 'claude'.")
        return None
    
    if method not in ["history", "backstory"]:
        logging.error(f"❌ Unsupported method: {method}. Use 'history' or 'backstory'.")
        return None
    
    try:
        # =====================================================================
        # Load Input Artifacts (from config - W&B ONLY, no local fallbacks)
        # =====================================================================
        logging.info("\nDownloading input artifacts from W&B (NO LOCAL FALLBACKS)...")
        
        input_artifacts = _cfg.get("input_artifacts", {})
        if not input_artifacts:
            logging.error("❌ input_artifacts not found in config.yaml - REQUIRED")
            return None
        
        file_patterns = _cfg.get("file_patterns", {})
        if not file_patterns:
            logging.error("❌ file_patterns not found in config.yaml - REQUIRED")
            return None
        
        dataset_type = _cfg.get("dataset_type")
        if not dataset_type:
            logging.error("❌ dataset_type not found in config.yaml - REQUIRED")
            logging.error("   Set dataset_type: 'train' or 'test' in config.yaml")
            return None
        
        if dataset_type not in ["train", "test"]:
            logging.error(f"❌ Invalid dataset_type: {dataset_type}. Must be 'train' or 'test'")
            return None
        
        # Load category mapping from W&B (REQUIRED)
        category_mapping_artifact = input_artifacts.get("category_mapping")
        if not category_mapping_artifact:
            logging.error("❌ category_mapping not found in config.yaml - REQUIRED")
            logging.error("   Set input_artifacts.category_mapping in config.yaml")
            return None
        
        logging.info(f"Loading category mapping from W&B: {category_mapping_artifact}")
        category_mapping_path = use_artifact(run, category_mapping_artifact, artifact_type="dataset")
        if not category_mapping_path:
            logging.error(f"❌ Could not download category mapping artifact from W&B: {category_mapping_artifact}")
            logging.error("   NO LOCAL FALLBACKS - artifact must be available in W&B")
            return None
        
        # Find category mapping file
        category_file_pattern = file_patterns.get("category_mapping", "category_mapping.json")
        if category_mapping_path.is_file():
            category_mapping_file = category_mapping_path
        elif category_mapping_path.is_dir():
            category_mapping_file = category_mapping_path / category_file_pattern
            if not category_mapping_file.exists():
                logging.error(f"❌ Category mapping file '{category_file_pattern}' not found in artifact: {category_mapping_path}")
                logging.error("   NO LOCAL FALLBACKS - file must exist in W&B artifact")
                return None
        else:
            logging.error(f"❌ Category mapping artifact path is invalid: {category_mapping_path}")
            return None
        
        # Load category mapping
        with open(category_mapping_file, 'r', encoding='utf-8') as f:
            category_mapping_data = json.load(f)
        
        # Extract category_to_main_mapping
        global SUB_TO_MAIN_CATEGORY_MAP
        SUB_TO_MAIN_CATEGORY_MAP = category_mapping_data.get('category_to_main_mapping', {})
        if not SUB_TO_MAIN_CATEGORY_MAP:
            logging.error("❌ category_to_main_mapping not found in category mapping file - REQUIRED")
            return None
        
        logging.info(f"✅ Loaded category mapping with {len(SUB_TO_MAIN_CATEGORY_MAP)} categories from W&B")
        
        # Load topic universe from W&B
        topic_artifact = input_artifacts.get("topics")
        if not topic_artifact:
            logging.error("❌ topics artifact not found in config.yaml - REQUIRED")
            return None
        
        logging.info(f"Loading topic universe from W&B: {topic_artifact}")
        topic_universe_path = use_artifact(run, topic_artifact, artifact_type="dataset")
        if not topic_universe_path:
            logging.error(f"❌ Could not download topic universe artifact from W&B: {topic_artifact}")
            logging.error("   NO LOCAL FALLBACKS - artifact must be available in W&B")
            return None
        
        topic_file_pattern = file_patterns.get("topic_universe", "topic_universe.json")
        if topic_universe_path.is_file():
            topic_universe_file = topic_universe_path
        elif topic_universe_path.is_dir():
            topic_universe_file = topic_universe_path / topic_file_pattern
            if not topic_universe_file.exists():
                logging.error(f"❌ Topic universe file '{topic_file_pattern}' not found in artifact: {topic_universe_path}")
                logging.error("   NO LOCAL FALLBACKS - file must exist in W&B artifact")
                return None
        else:
            logging.error(f"❌ Topic universe artifact path is invalid: {topic_universe_path}")
            return None
        
        topic_universe = TopicUniverseArtifact.from_file(topic_universe_file)
        category_themes_map = topic_universe.topics_by_category
        logging.info(f"✅ Loaded topic universe for {len(category_themes_map)} categories from W&B")
        
        # Load training/test data with topics (based on dataset_type)
        if dataset_type == "train":
            training_data_artifact = input_artifacts.get("training_data_train")
        else:
            training_data_artifact = input_artifacts.get("training_data_test")
        
        if not training_data_artifact:
            logging.error(f"❌ training_data_{dataset_type} artifact not found in config.yaml - REQUIRED")
            return None
        
        logging.info(f"Loading {dataset_type} data from W&B: {training_data_artifact}")
        training_data_path = use_artifact(run, training_data_artifact, artifact_type="dataset")
        if not training_data_path:
            logging.error(f"❌ Could not download {dataset_type} data artifact from W&B: {training_data_artifact}")
            logging.error("   NO LOCAL FALLBACKS - artifact must be available in W&B")
            return None
        
        # Find training data file using config pattern
        if dataset_type == "train":
            data_file_pattern = file_patterns.get("training_data", "train_set_reviews_with_topics_filtered.json")
        else:
            data_file_pattern = file_patterns.get("test_data", "test_set_reviews_with_topics_filtered.json")
        
        if training_data_path.is_file():
            training_data_file = training_data_path
        elif training_data_path.is_dir():
            training_data_file = training_data_path / data_file_pattern
            if not training_data_file.exists():
                logging.error(f"❌ {dataset_type} data file '{data_file_pattern}' not found in artifact: {training_data_path}")
                logging.error("   NO LOCAL FALLBACKS - file must exist in W&B artifact")
                return None
        else:
            logging.error(f"❌ Training data artifact path is invalid: {training_data_path}")
            return None
        
        with open(training_data_file, 'r', encoding='utf-8') as f:
            training_data = json.load(f)
        
        logging.info(f"✅ Loaded {dataset_type} data from W&B")
        logging.info(f"📊 Training data type: {type(training_data).__name__}")
        if isinstance(training_data, dict):
            logging.info(f"📊 Training data keys: {list(training_data.keys())[:10]}")  # Show first 10 keys
        elif isinstance(training_data, list):
            logging.info(f"📊 Training data length: {len(training_data)}")
            if training_data:
                logging.info(f"📊 First item keys: {list(training_data[0].keys()) if isinstance(training_data[0], dict) else 'N/A'}")
        
        # Load method-specific artifacts from W&B (NO LOCAL FALLBACKS)
        if method == "history":
            # Load user review history
            review_history_artifact = input_artifacts.get("user_review_history")
            if not review_history_artifact:
                logging.error("❌ user_review_history artifact not found in config.yaml - REQUIRED for history method")
                return None
            
            logging.info(f"Loading user review history from W&B: {review_history_artifact}")
            review_history_path = use_artifact(run, review_history_artifact, artifact_type="dataset")
            if not review_history_path:
                logging.error(f"❌ Could not download user review history artifact from W&B: {review_history_artifact}")
                logging.error("   NO LOCAL FALLBACKS - artifact must be available in W&B")
                return None
            
            review_history_file_pattern = file_patterns.get("user_review_history", "user_review_history.json")
            if review_history_path.is_file():
                review_history_file = review_history_path
            elif review_history_path.is_dir():
                review_history_file = review_history_path / review_history_file_pattern
                if not review_history_file.exists():
                    logging.error(f"❌ Review history file '{review_history_file_pattern}' not found in artifact: {review_history_path}")
                    logging.error("   NO LOCAL FALLBACKS - file must exist in W&B artifact")
                    return None
            else:
                logging.error(f"❌ Review history artifact path is invalid: {review_history_path}")
                return None
            
            user_review_history = UserReviewHistoryArtifact.from_file(review_history_file)
            logging.info(f"✅ Loaded review history for {len(user_review_history)} users from W&B")
        else:  # backstory
            # Load user backstories
            user_backstory_artifact = input_artifacts.get("user_backstories")
            if not user_backstory_artifact:
                logging.error("❌ user_backstories artifact not found in config.yaml - REQUIRED for backstory method")
                return None
            
            logging.info(f"Loading user backstories from W&B: {user_backstory_artifact}")
            user_backstory_path = use_artifact(run, user_backstory_artifact, artifact_type="dataset")
            if not user_backstory_path:
                logging.error(f"❌ Could not download user backstory artifact from W&B: {user_backstory_artifact}")
                logging.error("   NO LOCAL FALLBACKS - artifact must be available in W&B")
                return None
            
            user_backstory_file_pattern = file_patterns.get("user_backstories", "user_overall_characteristics.json")
            if user_backstory_path.is_file():
                user_backstory_file = user_backstory_path
            elif user_backstory_path.is_dir():
                user_backstory_file = user_backstory_path / user_backstory_file_pattern
                if not user_backstory_file.exists():
                    logging.error(f"❌ User backstory file '{user_backstory_file_pattern}' not found in artifact: {user_backstory_path}")
                    logging.error("   NO LOCAL FALLBACKS - file must exist in W&B artifact")
                    return None
            else:
                logging.error(f"❌ User backstory artifact path is invalid: {user_backstory_path}")
                return None
            
            user_backstories = UserBackstoryArtifact.from_file(user_backstory_file)
            logging.info(f"✅ Loaded user backstories for {len(user_backstories)} users from W&B")
        
        # Load user-to-micro-cluster mapping (optional)
        user_micro_cluster_map = {}
        
        # =====================================================================
        # Prepare Tasks
        # =====================================================================
        all_tasks = []
        all_personas_metrics = {}
        
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
            # Try different possible structures
            if 'user_predictions' in training_data:
                # Structure: {user_predictions: {user_id: {reviews: [...]}}}
                user_predictions = training_data.get('user_predictions', {})
                for user_id, user_data in user_predictions.items():
                    reviews = user_data.get('reviews', [])
                    if reviews:
                        user_reviews_map[user_id] = reviews
            else:
                # Structure from Stage 04 filtered output: {user_id: {reviews: [...]}}
                # This is the standard format from train_set_reviews_with_topics_filtered.json
                for user_id, user_data in training_data.items():
                    if isinstance(user_data, list):
                        # Direct list of reviews: {user_id: [...]}
                        user_reviews_map[user_id] = user_data
                    elif isinstance(user_data, dict):
                        # Dict with reviews key: {user_id: {reviews: [...]}}
                        # This is the format from Stage 04 merge_topics.py filtered output
                        reviews = user_data.get('reviews', [])
                        if reviews:
                            user_reviews_map[user_id] = reviews
        
        logging.info(f"📊 Training data: {len(user_reviews_map)} users with reviews")
        total_training_reviews = sum(len(reviews) for reviews in user_reviews_map.values())
        logging.info(f"📊 Total reviews in training data: {total_training_reviews}")
        
        # Initialize skip counters
        skipped_no_history = 0
        skipped_no_themes = 0
        skipped_no_category = 0
        
        if method == "history":
            logging.info(f"📊 Review history: {len(user_review_history)} users loaded")
            # Check overlap
            training_users = set(user_reviews_map.keys())
            history_users = set(user_review_history.keys())
            overlap = training_users & history_users
            logging.info(f"📊 User overlap: {len(overlap)} users in both training_data and review_history")
            if len(overlap) < len(training_users):
                missing = training_users - history_users
                logging.warning(f"⚠️  {len(missing)} users in training_data but not in review_history")
        
        for user_id, reviews in user_reviews_map.items():
            if len(reviews) < (2 if method == "history" else 1):
                continue
            
            # Get backstory for backstory method
            backstory = None
            if method == "backstory":
                if user_id in user_backstories:
                    backstory_data = user_backstories[user_id]
                    # Combine general and category-specific characteristics
                    general_chars = backstory_data.overall_characteristics.influencing_characteristics_summary
                    backstory = general_chars
                else:
                    backstory = "I am a detailed and honest reviewer on Amazon."
            
            # Process all reviews (leave-one-out for history method)
            for i in range(len(reviews)):
                target_review = reviews[i]
                
                # Get history for history method
                history_reviews = []
                if method == "history":
                    if user_id in user_review_history:
                        history_data = user_review_history[user_id]
                        history_count = len(history_data.review_history)
                        training_count = len(reviews)
                        
                        # Debug: Log mismatch if found
                        if history_count != training_count:
                            logging.debug(f"User {user_id}: review_history has {history_count} reviews, training_data has {training_count} reviews")
                        
                        # Use all history reviews except the current one (by index)
                        # If history has fewer reviews, use all available history
                        if i < history_count:
                            all_history = history_data.review_history[:i] + history_data.review_history[i+1:]
                        else:
                            # If training_data has more reviews than history, use all history
                            all_history = history_data.review_history
                        
                        # Convert to format expected by prompt
                        history_reviews = [
                            {
                                'product_description': r.product_description,
                                'review_text': r.review_text,
                                'rating': getattr(r, 'rating', 3.0)
                            }
                            for r in all_history
                        ]
                    else:
                        logging.debug(f"User {user_id} not found in review_history")
                    if not history_reviews:
                        # Skip if no history (first review with no previous reviews)
                        skipped_no_history += 1
                        continue
                
                raw_category = target_review.get('category')
                # Use category mapping loaded from W&B (no hardcoded values)
                if not SUB_TO_MAIN_CATEGORY_MAP:
                    logging.error("❌ Category mapping not loaded - cannot process reviews")
                    continue
                mapped_category = SUB_TO_MAIN_CATEGORY_MAP.get(raw_category, raw_category)
                
                if not target_review.get('themes') and not target_review.get('predicted_themes'):
                    skipped_no_themes += 1
                    continue
                
                # Get category themes - try multiple category name formats
                category_themes = None
                if raw_category and raw_category in category_themes_map:
                    category_themes = category_themes_map[raw_category]
                elif mapped_category and mapped_category in category_themes_map:
                    category_themes = category_themes_map[mapped_category]
                else:
                    # Try normalized versions (handle Fashion/Clothing variations)
                    # Normalize: "Clothing Shoes & Jewelry" <-> "Clothing_Shoes_and_Jewelry"
                    normalized_mapped = mapped_category.replace(" ", "_").replace("&", "and") if mapped_category else None
                    if normalized_mapped and normalized_mapped in category_themes_map:
                        category_themes = category_themes_map[normalized_mapped]
                    else:
                        # Try reverse normalization
                        denormalized_mapped = mapped_category.replace("_", " ").replace("and", "&") if mapped_category else None
                        if denormalized_mapped and denormalized_mapped in category_themes_map:
                            category_themes = category_themes_map[denormalized_mapped]
                
                if not category_themes:
                    skipped_no_category += 1
                    logging.debug(f"Skipping review {i} for user {user_id}: category '{raw_category}' (mapped: '{mapped_category}') not found in category_themes_map")
                    continue
                
                # Get micro-cluster info
                user_micro_info = user_micro_cluster_map.get(str(user_id), {})
                
                task = (user_id, i, target_review, history_reviews, backstory, category_themes,
                       raw_category, client, model_name, actual_model_name, method, user_micro_info)
                all_tasks.append(task)
        
        total_tasks = len(all_tasks)
        logging.info(f"📊 Task preparation summary:")
        logging.info(f"   - Total reviews in training data: {total_training_reviews}")
        logging.info(f"   - Tasks created: {total_tasks}")
        logging.info(f"   - Skipped (no history): {skipped_no_history}")
        logging.info(f"   - Skipped (no themes): {skipped_no_themes}")
        logging.info(f"   - Skipped (no category): {skipped_no_category}")
        logging.info(f"Processing {total_tasks} reviews with {max_workers} parallel workers...")
        
        # =====================================================================
        # Process Tasks
        # =====================================================================
        all_predictions = []
        failed_predictions = []
        checkpoint_file = output_dir / f"checkpoint_predictions_{model_name}_{method}.json"
        failed_checkpoint_file = output_dir / f"failed_predictions_{model_name}_{method}.json"
        
        # Load checkpoint if exists
        processed_keys = set()
        if checkpoint_file.exists():
            try:
                with open(checkpoint_file, 'r', encoding='utf-8') as f:
                    checkpoint_data = json.load(f)
                if isinstance(checkpoint_data, list):
                    all_predictions = checkpoint_data
                    # Also load metrics from checkpoint into all_personas_metrics
                    for entry in all_predictions:
                        u_id = entry.get('user_id')
                        r_idx = entry.get('review_index')
                        if u_id is not None and r_idx is not None:
                            processed_keys.add((str(u_id), int(r_idx)))
                        
                        # Load metrics from checkpoint
                        metrics = entry.get('metrics', {})
                        for key, value in metrics.items():
                            if isinstance(value, (int, float)):
                                all_personas_metrics.setdefault(key, []).append(value)
                    
                    logging.info(f"✅ Resuming... {len(processed_keys)} samples already processed.")
                    logging.info(f"   Loaded {len(all_predictions)} predictions from checkpoint")
                    logging.info(f"   Loaded metrics: {dict((k, len(v)) for k, v in all_personas_metrics.items())}")
            except Exception as e:
                logging.warning(f"⚠️ Error loading checkpoint: {e}. Starting fresh.")
        
        # Load failed predictions checkpoint if exists
        if failed_checkpoint_file.exists():
            try:
                with open(failed_checkpoint_file, 'r', encoding='utf-8') as f:
                    failed_checkpoint_data = json.load(f)
                if isinstance(failed_checkpoint_data, list):
                    failed_predictions = failed_checkpoint_data
                    logging.info(f"   Loaded {len(failed_predictions)} failed predictions from checkpoint")
            except Exception as e:
                logging.warning(f"⚠️ Error loading failed predictions checkpoint: {e}")
        
        # Filter out already processed tasks
        all_tasks = [t for t in all_tasks if (str(t[0]), int(t[1])) not in processed_keys]
        
        # Process in parallel
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_task = {
                executor.submit(process_single_review, task): task
                for task in all_tasks
            }
            
            save_counter = 0
            failed_save_counter = 0
            
            with tqdm(total=len(all_tasks), desc=f"Processing {model_name}+{method}") as pbar:
                for future in as_completed(future_to_task):
                    try:
                        result, failed_data = future.result()
                        if result:
                            # Successful prediction
                            all_predictions.append(result)
                            
                            # Update metrics
                            for key, value in result['metrics'].items():
                                if isinstance(value, (int, float)):
                                    all_personas_metrics.setdefault(key, []).append(value)
                            
                            save_counter += 1
                            
                            # Save checkpoint every 5 items
                            if save_counter >= 5:
                                with open(checkpoint_file, 'w', encoding='utf-8') as f:
                                    json.dump(convert_to_serializable(all_predictions), f, indent=2)
                                save_counter = 0
                        elif failed_data:
                            # Failed prediction - save to separate file
                            failed_predictions.append(failed_data)
                            failed_save_counter += 1
                            
                            # Save failed predictions checkpoint every 5 items
                            if failed_save_counter >= 5:
                                with open(failed_checkpoint_file, 'w', encoding='utf-8') as f:
                                    json.dump(convert_to_serializable(failed_predictions), f, indent=2)
                                failed_save_counter = 0
                    except Exception as e:
                        logging.error(f"Error processing review: {e}")
                    finally:
                        pbar.update(1)
        
        # =====================================================================
        # Save Results and Validate Schema
        # =====================================================================
        if all_predictions:
            # Convert to artifact format
            predictions_dict = {}
            for pred in all_predictions:
                review_key = f"{pred['user_id']}_review_{pred['review_index']}"
                predictions_dict[review_key] = pred
            
            # Validate against schema (strict)
            try:
                validated_predictions = BaselinePredictionsArtifact.from_dict(predictions_dict)
                logging.info(f"✅ Validated {len(validated_predictions)} predictions against schema")
                
                # Convert validated predictions back to dict for saving
                validated_dict = BaselinePredictionsArtifact.to_dict(validated_predictions)
            except Exception as e:
                logging.error(f"❌ Schema validation failed: {e}")
                logging.error("Cannot proceed without valid schema. Please fix data format.")
                raise ValueError(f"Schema validation failed: {e}") from e
            
            # Save validated predictions (only successful ones)
            output_file = output_dir / f"baseline_predictions_{model_name}_{method}.json"
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(convert_to_serializable(validated_dict), f, indent=2)
            
            # Save failed predictions to separate file
            if failed_predictions:
                failed_output_file = output_dir / f"failed_predictions_{model_name}_{method}.json"
                with open(failed_output_file, 'w', encoding='utf-8') as f:
                    json.dump(convert_to_serializable(failed_predictions), f, indent=2)
                logging.info(f"⚠️  Saved {len(failed_predictions)} failed predictions to {failed_output_file}")
                logging.info(f"   These can be reprocessed later")
            
            # Calculate final summary
            if all_personas_metrics:
                grand_average_results = {
                    'model': model_name,
                    'method': method,
                    'review_count': len(all_predictions),
                    'final_summary': {}
                }
                for key, scores in all_personas_metrics.items():
                    mean_score = np.mean(scores)
                    grand_average_results['final_summary'][key] = {
                        'mean': mean_score,
                        'std': np.std(scores),
                        'count': len(scores)
                    }
                
                summary_file = output_dir / f"grand_summary_{model_name}_{method}.json"
                with open(summary_file, 'w') as f:
                    json.dump(convert_to_serializable(grand_average_results), f, indent=4)
                
                logging.info(f"📊 Final summary: recall@max(3,k) = {grand_average_results['final_summary'].get('recall@max(3,k)', {}).get('mean', 0):.4f}")
            
            # Log summary of failed predictions
            if failed_predictions:
                logging.warning(f"⚠️  {len(failed_predictions)} predictions failed after retries")
                logging.warning(f"   Failed predictions saved to: failed_predictions_{model_name}_{method}.json")
            else:
                logging.info(f"✅ All predictions succeeded (0 failures)")
            
            # Log artifact to W&B
            artifact = log_artifact(
                run=run,
                artifact_name=f"baseline_predictions_{model_name}_{method}_v4",
                artifact_type="result",
                artifact_path=str(output_dir),
                metadata={
                    "model": model_name,
                    "method": method,
                    "num_predictions": len(all_predictions),
                    "num_failed": len(failed_predictions),
                    "schema_version": "v4",
                    "schema_validated": True
                }
            )
            
            if artifact:
                link_to_registry(artifact, stage="09_baselines")
                logging.info(f"✅ Logged artifact: {artifact.name}")
            
            logging.info(f"✅ Saved {len(all_predictions)} successful predictions to {output_file}")
            return output_file
        
        return None
    
    except Exception as e:
        logging.error(f"❌ Error running baseline {model_name}+{method}: {e}")
        import traceback
        traceback.print_exc()
        return None

def main():
    """Main execution logic."""
    parser = argparse.ArgumentParser(description="Baseline Prediction Pipeline (Stage 09)")
    parser.add_argument("--model", type=str, default=None, choices=["o3", "claude"],
                       help="Model to use: 'o3' or 'claude' (if not specified, runs all from config)")
    parser.add_argument("--method", type=str, default=None, choices=["history", "backstory"],
                       help="Method to use: 'history' or 'backstory' (if not specified, runs all from config)")
    parser.add_argument("--max-workers", type=int, default=None,
                       help="Number of parallel workers (default: from config)")
    
    args = parser.parse_args()
    
    # Get config values
    baseline_config = _cfg.get("baseline_config", {})
    models = args.model and [args.model] or baseline_config.get("models", ["o3", "claude"])
    methods = args.method and [args.method] or baseline_config.get("methods", ["history", "backstory"])
    max_workers = args.max_workers or _cfg.get("hyperparameters", {}).get("max_workers", 20)
    
    logging.info("=" * 70)
    logging.info("STAGE 09: Baselines")
    logging.info("=" * 70)
    logging.info(f"Models: {models}")
    logging.info(f"Methods: {methods}")
    logging.info(f"Max workers: {max_workers}")
    
    # Initialize W&B run
    run = init_wandb_run(
        run_name=f"baselines_{time.strftime('%Y%m%d_%H%M%S')}",
        stage="09_baselines",
        job_type="baseline_prediction"
    )
    
    try:
        # Get output directory (from config, no hardcoded default)
        output_artifacts = _cfg.get("output_artifacts", {})
        output_artifact_name = output_artifacts.get("baseline_predictions")
        if not output_artifact_name:
            logging.error("❌ output_artifacts.baseline_predictions not found in config.yaml - REQUIRED")
            return
        
        # Save artifacts in 10_running_simulations/artifacts when called from unified runner
        # Check environment variable set by unified runner
        if os.environ.get("SAVE_TO_STAGE_10", "false").lower() == "true":
            artifact_dir = get_artifact_dir("10_running_simulations", output_artifact_name)
        else:
            artifact_dir = get_artifact_dir("09_baselines", output_artifact_name)
        artifact_dir.mkdir(parents=True, exist_ok=True)
        
        # Run all combinations
        results = {}
        for model in models:
            for method in methods:
                result_file = run_baseline(model, method, max_workers, artifact_dir, run)
                if result_file:
                    results[f"{model}_{method}"] = result_file
        
        if results:
            logging.info(f"\n✅ Completed {len(results)} baseline runs")
            for key, path in results.items():
                logging.info(f"  - {key}: {path}")
        else:
            logging.error("❌ No baseline runs completed successfully")
    
    finally:
        finish_run(run)

if __name__ == "__main__":
    main()

