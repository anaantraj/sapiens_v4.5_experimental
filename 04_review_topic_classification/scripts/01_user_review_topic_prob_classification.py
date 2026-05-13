#!/usr/bin/env python3
"""
Stage 04: Review Topic Classification
=====================================

Parallel theme prediction script for reviews.
- Reads configuration from config.yaml
- Downloads reviews and topics artifacts from W&B
- Processes reviews in batches with async/await
- Logs artifact to W&B: Processed Data collection

Usage:
    python 04_review_topic_classification/scripts/01_user_review_topic_prob_classification.py
"""

import json
import os
import sys
import time
import asyncio
import logging
import hashlib
from collections import defaultdict
from pathlib import Path
from openai import RateLimitError, APIError, APIConnectionError, APITimeoutError

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Load .env so OPENAI_API_KEY and OPENAI_BASE_URL (Bedrock) are set
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent.parent.parent / ".env", override=True)
except ImportError:
    pass

from utils.openai_client import create_async_openai_client
from utils.wandb_utils import (
    load_config, get_stage_config, get_openai_config,
    init_wandb_run, finish_run, use_artifact, log_artifact,
    log_metrics, log_summary, link_to_registry, get_artifact_dir,
    create_comprehensive_artifact_metadata, get_learned_artifact_schema,
    validate_stage_dependencies
)

# Import schemas for validation
from schemas.learned_artifacts import TopicUniverseArtifact, ReviewTopicClassificationArtifact

# Configure Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Prompt directory
PROMPT_DIR = Path(__file__).parent.parent / "prompts"
import math

def extract_token_probabilities_from_response(logprobs_data, predicted_topics):
    """
    Extracts the actual mathematical probability of the tokens that formed 
    the predicted topics.
    
    Args:
        logprobs_data: The 'logprobs' object from the OpenAI response.
        predicted_topics: List of strings (topics) the model found in the JSON.
        
    Returns:
        Dict: { "Topic Name": 0.982, ... } mapping topics to their generation confidence.
    """
    if not logprobs_data or not hasattr(logprobs_data, 'content'):
        return {}

    token_probs = defaultdict(list)
    
    # Flatten predicted topics for easy searching
    # We clean them to match token fragments (lowercase, stripped)
    target_map = {t.lower().strip(): t for t in predicted_topics}
    
    try:
        # Iterate through every token generated in the response
        for token_info in logprobs_data.content:
            token_text = token_info.token.lower().strip()
            
            # Skip structural JSON tokens
            if token_text in ['"', '{', '}', '[', ']', ':', ',']:
                continue
                
            # Check if this token is part of a predicted topic
            # Note: Topics might be split into multiple tokens (e.g., "Battery" + "Life")
            # We assign this token's probability to any topic containing this string
            current_prob = math.exp(token_info.logprob) # Convert log scale to 0-1
            
            for target_lower, original_name in target_map.items():
                if token_text in target_lower:
                    token_probs[original_name].append(current_prob)
                    
    except Exception as e:
        error_msg = f"Error calculating token probabilities: {e}"
        logging.error(error_msg)
        # Don't silently return empty dict - log the error but return empty to allow processing to continue
        # The caller should handle missing probabilities appropriately
        return {}

    # Average the probabilities for multi-token words
    final_scores = {}
    for topic, probs in token_probs.items():
        if probs:
            # We take the average confidence of the tokens that made up the word
            final_scores[topic] = sum(probs) / len(probs)
            
    return final_scores
# Global tracking
processed_count = 0
failed_count = 0
empty_themes_count = 0  # Reviews with empty themes (but saved to output)
polluted_data_count = 0  # Reviews with missing input fields
total_input_tokens = 0
total_output_tokens = 0

def get_processed_ids(output_file: Path) -> set:
    """Reads the output file to find review_ids that are already finished."""
    processed_ids = set()
    if not output_file.exists():
        return processed_ids
    
    logging.info(f"Checking for existing progress in {output_file}...")
    try:
        with open(output_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    try:
                        data = json.loads(line)
                        if 'review_id' in data:
                            processed_ids.add(data['review_id'])
                    except json.JSONDecodeError:
                        continue
        logging.info(f"Found {len(processed_ids)} already processed reviews. Resuming...")
    except Exception as e:
        logging.warning(f"Could not read existing file to resume: {e}")
    
    return processed_ids
def map_category_to_main_category(category_name: str) -> str:
    """
    Map a category name to one of the 7 main categories.
    This matches the mapping used in Stage 01 and Stage 03.
    
    Args:
        category_name: Category name from review data
        
    Returns:
        Main category name or None if no match
    """
    if not category_name:
        return None
    
    category_lower = category_name.lower()
    
    # Mapping rules (matching Stage 01 and Stage 03)
    if any(x in category_lower for x in ['fashion', 'clothing', 'shoes', 'jewelry', 'apparel']):
        return "Clothing_Shoes_and_Jewelry"
    elif any(x in category_lower for x in ['appliance', 'tools & home improvement', 'amazon home', 'industrial & scientific', 'sports & outdoors', 'automotive', 'arts, crafts & sewing']):
        return "Appliances"
    elif any(x in category_lower for x in ['beauty', 'cosmetic', 'makeup', 'skincare', 'premium beauty']):
        return "All_Beauty"
    elif any(x in category_lower for x in ['music', 'digital music', 'audio', 'musical instruments', 'home audio & theater', 'car electronics']):
        return "Digital_Music"
    elif any(x in category_lower for x in ['video game', 'gaming', 'game', 'toys & games']):
        return "Video_Games"
    elif any(x in category_lower for x in ['health', 'personal care', 'wellness', 'baby', 'grocery']):
        return "Health_and_Personal_Care"
    elif any(x in category_lower for x in ['software', 'app', 'application', 'appstore for android', 'computers', 'all electronics', 'cell phones & accessories', 'camera & photo', 'office products', 'buy a kindle', 'books', 'movies & tv', 'portable audio & accessories']):
        return "Software"
    
    return None


def load_review_data(filepath: Path):
    """Loads review data from JSON file."""
    if not filepath.exists():
        error_msg = f"Input file not found at '{filepath}'"
        logging.error(error_msg)
        raise FileNotFoundError(error_msg)
    
    logging.info(f"Loading review data from '{filepath}'...")
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            user_data = json.load(f)
        
        # Flatten reviews with user_id and category info
        all_reviews = []
        for user_id, data in user_data.items():
            reviews_list = data.get('reviews', [])
            for review_obj in reviews_list:
                review_text = review_obj.get('review_text', '')
                review_category = review_obj.get('category', 'Unknown')
                
                if review_text and review_text.strip():
                    # Map original category to main category
                    main_category = map_category_to_main_category(review_category)
                    all_reviews.append({
                        "user_id": user_id,
                        "review": review_text,
                        "category": review_category,  # Keep original for reference
                        "main_category": main_category,  # Mapped main category
                        "rating": review_obj.get('rating'),
                        "asin": review_obj.get('asin'),
                        "timestamp": review_obj.get('timestamp')
                    })
        
        logging.info(f"Loaded {len(all_reviews)} reviews from {len(user_data)} users.")
        return all_reviews
        
    except json.JSONDecodeError as e:
        error_msg = f"Error decoding JSON file '{filepath}': {e}"
        logging.error(error_msg)
        raise ValueError(error_msg) from e
    except Exception as e:
        error_msg = f"Error reading file '{filepath}': {e}"
        logging.error(error_msg)
        raise RuntimeError(error_msg) from e


def load_prompt(prompt_filename: str) -> str:
    """Load a prompt template from file."""
    prompt_path = PROMPT_DIR / prompt_filename
    if prompt_path.exists():
        with open(prompt_path, 'r', encoding='utf-8') as f:
            return f.read().strip()
    else:
        raise FileNotFoundError(f"Prompt file not found: {prompt_path}")


async def extract_topics_from_review(
    review_text: str,
    review_id: str,
    category: str,
    topic_universe: list,
    client,
    model_name: str,
    semaphore: asyncio.Semaphore,
    max_retries: int,
    retry_delay: int
) -> dict:
    global processed_count, failed_count, total_input_tokens, total_output_tokens
    
    # Validate inputs - if invalid, return error result instead of raising (to not break async batch)
    if not review_text or not review_text.strip():
        error_msg = f"Empty review_text for review_id: {review_id}"
        logging.warning(error_msg)
        failed_count += 1
        return {
            "review_id": review_id,
            "error": error_msg,
            "identified_themes": [],
            "theme_token_probabilities": {},
            "sentiment": None  # No sentiment if no review text
        }
    
    if not topic_universe or len(topic_universe) == 0:
        error_msg = f"Empty topic_universe for category: {category}, review_id: {review_id}"
        logging.error(error_msg)
        # This is a configuration error - should be caught earlier, but handle gracefully
        failed_count += 1
        return {
            "review_id": review_id,
            "error": error_msg,
            "identified_themes": [],
            "theme_token_probabilities": {},
            "sentiment": None
        }
    
    async with semaphore:
        # --- MODIFIED PROMPT FOR STRICTNESS ---
        # We explicitly tell it to ignore the "forced" requirement if nothing fits.
        prompt_template = load_prompt("review_classification_user_prompt.txt")
        topic_universe_str = json.dumps(topic_universe, indent=2)
        
        user_prompt = prompt_template.format(
            review_text=review_text,
            topic_universe=topic_universe_str
        )
        
        # Load system prompt from file
        system_prompt = load_prompt("review_classification_system_prompt.txt")
        
        for attempt in range(max_retries):
            try:
                # Prepare API params
                api_params = {
                    "model": model_name,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    "response_format": {"type": "json_object"},
                    "timeout": 60.0
                }

                # --- ENABLE LOGPROBS FOR TOKEN PROBABILITIES (OpenAI-style from API) ---
                # Request logprobs for models that support it: OpenAI (gpt-4o, gpt-4o-mini), Bedrock (gpt-oss on Mantle). Use only such models for this stage.
                models_with_logprobs = ["gpt-4", "gpt-3.5", "gpt-4o", "gpt-4-turbo", "gpt-oss", "oss"]
                if any(m in model_name.lower() for m in models_with_logprobs):
                    api_params["logprobs"] = True
                    api_params["top_logprobs"] = 3
                else:
                    logging.warning(f"Model {model_name} may not support logprobs. Token probabilities may come from JSON or fallback.")
                completion = await client.chat.completions.create(**api_params)
                
                # Metrics handling
                if completion.usage:
                    total_input_tokens += completion.usage.prompt_tokens
                    total_output_tokens += completion.usage.completion_tokens
                
                response_content = completion.choices[0].message.content
                
                # Extract JSON
                start_index = response_content.find('{')
                end_index = response_content.rfind('}')
                
                if start_index == -1 or end_index == -1:
                    raise ValueError("No JSON found in response")
                
                clean_json_string = response_content[start_index:end_index + 1]
                response_json = json.loads(clean_json_string)
                
                # Handle batch format (old prompt) or single review format (new prompt)
                # Batch format: {"results": [{"sentiment": "...", "predicted_themes": [...]}]}
                # Single format: {"identified_themes": [...], "sentiment": "...", "topic_probabilities": {...}}
                if "results" in response_json and isinstance(response_json["results"], list) and len(response_json["results"]) > 0:
                    # Batch format - extract first result (since we're processing one review at a time)
                    result = response_json["results"][0]
                    identified_themes = result.get("predicted_themes", result.get("identified_themes", []))
                    if not identified_themes:
                        identified_themes = []
                    sentiment = result.get("sentiment")
                    if not sentiment:
                        # Missing sentiment - log warning but don't fail, use None
                        logging.warning(f"LLM response missing 'sentiment' field for review_id: {review_id}")
                        sentiment = None
                    json_topic_probs = result.get("topic_probabilities", {})
                else:
                    # Single review format
                    identified_themes = response_json.get("identified_themes", response_json.get("predicted_topics", response_json.get("predicted_themes", [])))
                    if not identified_themes:
                        identified_themes = []
                    sentiment = response_json.get("sentiment")
                    if not sentiment:
                        # Missing sentiment - log warning but don't fail, use None
                        logging.warning(f"LLM response missing 'sentiment' field for review_id: {review_id}")
                        sentiment = None
                    json_topic_probs = response_json.get("topic_probabilities", {})
                
                # --- EXTRACT REAL TOKEN PROBABILITIES ---
                real_token_probs = {}
                if hasattr(completion.choices[0], 'logprobs') and completion.choices[0].logprobs:
                    real_token_probs = extract_token_probabilities_from_response(
                        completion.choices[0].logprobs, 
                        identified_themes
                    )
                
                # Create theme_token_probabilities: map each identified theme to its token probability
                theme_token_probabilities = {}
                
                for theme in identified_themes:
                    if theme in real_token_probs:
                        theme_token_probabilities[theme] = real_token_probs[theme]
                    elif theme in json_topic_probs:
                        # Fallback to JSON probability if token prob not available from API
                        p = json_topic_probs[theme]
                        theme_token_probabilities[theme] = float(p) if p is not None else 0.5
                    else:
                        # Model did not return probability for this theme (e.g. topic_probabilities omitted); use 0.5 so schema accepts
                        logging.warning(f"No probability available for theme '{theme}' in review_id '{review_id}'. Using 0.5.")
                        theme_token_probabilities[theme] = 0.5
                
                processed_count += 1
                # Track empty themes (but still save to output)
                global empty_themes_count
                if not identified_themes or len(identified_themes) == 0:
                    empty_themes_count += 1
                
                return {
                    "review_id": review_id,
                    "identified_themes": identified_themes,  # Themes identified by LLM (can be empty)
                    "theme_token_probabilities": theme_token_probabilities,  # Token prob for each identified theme
                    "sentiment": sentiment
                }
                
            except Exception as e:
                # Error handling block (RateLimit, etc) remains the same as your original code...
                logging.warning(f"Attempt {attempt+1} failed for {review_id}: {e}")
                if attempt == max_retries - 1:
                    failed_count += 1
                    return {
                        "review_id": review_id, 
                        "error": str(e),
                        "identified_themes": [],
                        "theme_token_probabilities": {}
                    }
                await asyncio.sleep(retry_delay * (attempt + 1))

async def process_category_reviews(
    category: str,
    reviews: list,
    topic_universe: list,
    client,
    model_name: str,
    semaphore: asyncio.Semaphore,
    max_retries: int,
    retry_delay: int,
    concurrent_requests: int,
    output_file: Path = None,
    polluted_data_file: Path = None,
    failed_reviews_file: Path = None,
    save_interval: int = 10
) -> list:
    """
    Process reviews for a category and extract topics (one review at a time).
    Uses Bedrock when OPENAI_BASE_URL is set (client from create_async_openai_client).
    """
    logging.info(f"\nProcessing {len(reviews)} reviews for category: {category}")
    
    results = []
    polluted_reviews = []  # Reviews with missing required fields
    
    # Filter out polluted data (missing required fields) before processing
    valid_reviews = []
    for review in reviews:
        review_text = review.get('review', '')
        user_id = review.get('user_id', '')
        
        # Check for missing required fields
        if not review_text or not review_text.strip():
            polluted_reviews.append({
                **review,
                "pollution_reason": "missing_or_empty_review_text"
            })
            continue
        
        if not user_id:
            polluted_reviews.append({
                **review,
                "pollution_reason": "missing_user_id"
            })
            continue
        
        valid_reviews.append(review)
    
    # Save polluted data to separate file
    if polluted_reviews and polluted_data_file:
        global polluted_data_count
        polluted_data_count += len(polluted_reviews)
        try:
            with open(polluted_data_file, 'a', encoding='utf-8') as f:
                for polluted_review in polluted_reviews:
                    json_line = json.dumps(polluted_review, ensure_ascii=False) + '\n'
                    f.write(json_line)
            logging.warning(f"  Saved {len(polluted_reviews)} polluted reviews to {polluted_data_file.name}")
        except Exception as e:
            logging.error(f"  Failed to save polluted data: {e}")
    
    if not valid_reviews:
        logging.warning(f"  No valid reviews to process for category: {category}")
        return results
    
    logging.info(f"  Processing {len(valid_reviews)} valid reviews (filtered {len(polluted_reviews)} polluted)")
    
    # Process reviews in parallel (one review per LLM call)
    tasks = []
    for review in valid_reviews:
        review_text = review.get('review', '')
        user_id = review.get('user_id', '')
        asin = review.get('asin', '')
        review_id = f"{user_id}_{asin}" if user_id and asin else f"{category}_{len(results)}"
        
        task = extract_topics_from_review(
            review_text=review_text,
            review_id=review_id,
            category=category,
            topic_universe=topic_universe,
            client=client,
            model_name=model_name,
            semaphore=semaphore,
            max_retries=max_retries,
            retry_delay=retry_delay
        )
        tasks.append((review, task))
    
    # Execute all tasks in parallel
    logging.info(f"  Processing {len(tasks)} reviews in parallel (max {concurrent_requests} concurrent)...")
    
    # Process in chunks to avoid overwhelming the API
    chunk_size = concurrent_requests
    saved_count = 0  # Track how many have been saved
    
    for i in range(0, len(tasks), chunk_size):
        chunk = tasks[i:i + chunk_size]
        chunk_results = await asyncio.gather(*[task for _, task in chunk])
        
        # Combine results with original review data
        chunk_final_results = []
        failed_reviews_chunk = []  # Reviews that failed after all retries
        
        for (review, _), result in zip(chunk, chunk_results):
            # Check if this review failed after all retries
            if "error" in result:
                # Save failed review for rerun
                failed_review_data = {
                    **review,
                    "error": result.get("error"),
                    "review_id": result.get("review_id")
                }
                failed_reviews_chunk.append(failed_review_data)
                # Still add to results but mark as failed
                final_result = {
                    "user_id": review.get("user_id"),
                    "review": review.get("review"),
                    "category": review.get("category"),
                    "review_id": result.get("review_id"),
                    "identified_themes": result.get("identified_themes", []),
                    "theme_token_probabilities": result.get("theme_token_probabilities", {}),
                    "sentiment": None,  # No sentiment if failed
                    "rating": review.get("rating"),
                    "asin": review.get("asin"),
                    "timestamp": review.get("timestamp"),
                    "error": result.get("error")
                }
                results.append(final_result)
                chunk_final_results.append(final_result)
            else:
                # Successful processing - get sentiment
                sentiment = result.get("sentiment")
                if sentiment is None:
                    logging.warning(f"Missing sentiment for review_id {result.get('review_id')} but no error field")
                
                final_result = {
                    "user_id": review.get("user_id"),
                    "review": review.get("review"),
                    "category": review.get("category"),
                    "review_id": result.get("review_id"),
                    "identified_themes": result.get("identified_themes", []),  # Themes identified by LLM (can be empty)
                    "theme_token_probabilities": result.get("theme_token_probabilities", {}),  # Token prob for each identified theme
                    "sentiment": sentiment,
                    "rating": review.get("rating"),
                    "asin": review.get("asin"),
                    "timestamp": review.get("timestamp")
                }
                results.append(final_result)
                chunk_final_results.append(final_result)
        
        # Save failed reviews to separate file for rerun
        if failed_reviews_chunk and failed_reviews_file:
            try:
                with open(failed_reviews_file, 'a', encoding='utf-8') as f:
                    for failed_review in failed_reviews_chunk:
                        json_line = json.dumps(failed_review, ensure_ascii=False) + '\n'
                        f.write(json_line)
                logging.info(f"  Saved {len(failed_reviews_chunk)} failed reviews to {failed_reviews_file.name} for rerun")
            except Exception as e:
                logging.error(f"  Failed to save failed reviews: {e}")
        
        # Save every save_interval reviews
        if output_file and len(chunk_final_results) > 0:
            # Check if we've accumulated enough to save
            unsaved_count = len(results) - saved_count
            if unsaved_count >= save_interval:
                # Save the unsaved results
                to_save = results[saved_count:saved_count + (unsaved_count // save_interval) * save_interval]
                save_results_incremental(to_save, output_file)
                saved_count += len(to_save)
                logging.info(f"  Saved {saved_count} reviews to {output_file.name}")
        
        logging.info(f"  Processed {min(i + chunk_size, len(tasks))}/{len(tasks)} reviews...")
    
    # Save any remaining results
    if output_file and len(results) > saved_count:
        remaining = results[saved_count:]
        save_results_incremental(remaining, output_file)
        logging.info(f"  Saved remaining {len(remaining)} reviews (total: {len(results)})")
    
    return results


def create_review_id(review_data):
    """Creates a unique identifier for a review."""
    user_id = review_data.get('user_id', '')
    asin = review_data.get('asin', '')
    review_text = review_data.get('review', '')
    timestamp = review_data.get('timestamp', '')
    
    review_hash = hashlib.md5(review_text.encode('utf-8')).hexdigest()[:8]
    return f"{user_id}|{asin}|{timestamp}|{review_hash}"


def save_results_incremental(results, output_file: Path):
    """Saves results incrementally to JSONL file with schema validation."""
    if not results:
        return True
    
    try:
        validated_count = 0
        with open(output_file, 'a', encoding='utf-8') as f:
            for result in results:
                # Validate against schema before saving
                try:
                    validated_result = ReviewTopicClassificationArtifact.from_dict(result)
                    json_line = json.dumps(validated_result.to_dict(), ensure_ascii=False) + '\n'
                    f.write(json_line)
                    f.flush()
                    validated_count += 1
                except Exception as e:
                    logging.warning(f"Schema validation failed for review {result.get('review_id', 'unknown')}: {e}")
                    # Still save the result but log the warning
                    json_line = json.dumps(result, ensure_ascii=False) + '\n'
                    f.write(json_line)
                    f.flush()
        
        if validated_count < len(results):
            logging.warning(f"Only {validated_count}/{len(results)} results passed schema validation")
        
        return True
    except Exception as e:
        error_msg = f"Error saving results to {output_file}: {e}"
        logging.error(error_msg)
        # Raise exception instead of silently returning False - this is critical and should fail the run
        raise RuntimeError(error_msg) from e


async def main():
    """Main execution function."""
    global processed_count, failed_count, total_input_tokens, total_output_tokens
    
    # =========================================================================
    # Step 1: Load configuration from config.yaml
    # =========================================================================
    print("=" * 70)
    print("STAGE 04: Review Topic Classification")
    print("=" * 70)
    
    cfg = get_stage_config("04_review_topic_classification")
    openai_cfg = get_openai_config()
    
    # Get stage directory from config
    stage_directory = cfg.get("stage_directory")
    if not stage_directory:
        raise ValueError("stage_directory must be specified in config.yaml")
    
    hyperparams = cfg.get("hyperparameters")
    if not hyperparams:
        raise ValueError("hyperparameters must be specified in config.yaml")
    
    cost_cfg = cfg.get("cost", {})  # Cost config is optional
    
    # Get dataset_mode first (needed to select correct input artifact)
    dataset_mode = hyperparams.get("dataset_mode")
    if not dataset_mode:
        raise ValueError("hyperparameters.dataset_mode must be specified in config.yaml")
    
    if dataset_mode not in ["train", "test"]:
        raise ValueError(f"hyperparameters.dataset_mode must be 'train' or 'test', got: {dataset_mode}")
    
    # Get input artifact names from config (required, no defaults)
    # Should be a dict with 'train' and 'test' keys, or a pattern string
    input_artifact_reviews_config = cfg.get("input_artifact_reviews")
    if not input_artifact_reviews_config:
        raise ValueError("input_artifact_reviews must be specified in config.yaml")
    
    # Handle both dict format (explicit train/test) and pattern format
    if isinstance(input_artifact_reviews_config, dict):
        # Dict format: {"train": "...", "test": "..."}
        if dataset_mode not in input_artifact_reviews_config:
            raise ValueError(f"input_artifact_reviews.{dataset_mode} must be specified in config.yaml")
        input_artifact_reviews = input_artifact_reviews_config.get(dataset_mode)
        if not input_artifact_reviews:
            raise ValueError(f"input_artifact_reviews.{dataset_mode} must be a non-empty string in config.yaml")
    elif isinstance(input_artifact_reviews_config, str):
        # Pattern format: "{dataset_mode}_set_samples_v5:latest"
        input_artifact_reviews = input_artifact_reviews_config.format(dataset_mode=dataset_mode)
    else:
        raise ValueError("input_artifact_reviews must be either a dict with 'train' and 'test' keys, or a pattern string in config.yaml")
    
    input_artifact_topics = cfg.get("input_artifact_topics")
    if not input_artifact_topics:
        raise ValueError("input_artifact_topics must be specified in config.yaml")
    
    output_artifact_name = cfg.get("output_artifact")
    if not output_artifact_name:
        raise ValueError("output_artifact must be specified in config.yaml")
    
    # dataset_mode is already retrieved and validated above
    
    concurrent_requests = hyperparams.get("concurrent_requests")
    if concurrent_requests is None:
        raise ValueError("hyperparameters.concurrent_requests must be specified in config.yaml")
    
    use_parallel = hyperparams.get("parallel")
    if use_parallel is None:
        raise ValueError("hyperparameters.parallel must be specified in config.yaml")
    
    model_name = hyperparams.get("model")
    if not model_name:
        raise ValueError("hyperparameters.model must be specified in config.yaml")
    base_url = os.environ.get("OPENAI_BASE_URL", "").strip()
    if base_url and ("bedrock-mantle" in base_url or "bedrock" in base_url.lower()):
        # Prefer logprobs_model_id when set. Use a model that returns API logprobs (e.g. gpt-oss on Mantle).
        if cfg.get("logprobs_model_id"):
            model_name = cfg.get("logprobs_model_id")
        elif cfg.get("bedrock_model_id"):
            model_name = cfg.get("bedrock_model_id")
    max_retries = openai_cfg.get("max_retries", 5)  # Optional, has reasonable default
    retry_delay = openai_cfg.get("retry_delay", 10)  # Optional, has reasonable default
    
    cost_per_1m_input = cost_cfg.get("per_1m_input", 0.59)  # Optional, has reasonable default
    cost_per_1m_output = cost_cfg.get("per_1m_output", 0.79)  # Optional, has reasonable default
    
    save_interval = hyperparams.get("save_interval")
    if save_interval is None:
        raise ValueError("hyperparameters.save_interval must be specified in config.yaml")
    
    # Get artifact type from config
    artifact_type = cfg.get("artifact_type")
    if not artifact_type:
        raise ValueError("artifact_type must be specified in config.yaml")
    
    # Get job_type from config
    job_type = cfg.get("job_type")
    if not job_type:
        raise ValueError("job_type must be specified in config.yaml")
    
    # Get paths configuration
    paths_config = cfg.get("paths")
    if not paths_config:
        raise ValueError("paths must be specified in config.yaml")
    
    input_review_filenames = paths_config.get("input_review_filenames")
    if not input_review_filenames:
        raise ValueError("paths.input_review_filenames must be specified in config.yaml")
    
    # Get filenames based on dataset_mode
    if dataset_mode not in input_review_filenames:
        raise ValueError(f"paths.input_review_filenames.{dataset_mode} must be specified in config.yaml")
    
    review_filenames = input_review_filenames.get(dataset_mode)
    if not review_filenames or not isinstance(review_filenames, list) or len(review_filenames) == 0:
        raise ValueError(f"paths.input_review_filenames.{dataset_mode} must be a non-empty list in config.yaml")
    
    input_topic_universe_filename = paths_config.get("input_topic_universe_filename")
    if not input_topic_universe_filename:
        raise ValueError("paths.input_topic_universe_filename must be specified in config.yaml")
    
    output_filename_pattern = paths_config.get("output_filename_pattern")
    if not output_filename_pattern:
        raise ValueError("paths.output_filename_pattern must be specified in config.yaml")
    
    # Create Bedrock (or configured endpoint) client; uses OPENAI_API_KEY + OPENAI_BASE_URL from env
    if not os.environ.get("OPENAI_API_KEY"):
        logging.error("OPENAI_API_KEY environment variable not set (set in .env or export)")
        return
    llm_client = create_async_openai_client(timeout=60.0)
    
    print(f"\n[Config] Dataset mode: {dataset_mode}")
    print(f"[Config] Input artifact (reviews): {input_artifact_reviews} (auto-selected based on dataset_mode)")
    print(f"[Config] Input artifact (topics): {input_artifact_topics}")
    print(f"[Config] Output artifact: {output_artifact_name}")
    print(f"[Config] Model: {model_name}")
    print(f"[Config] Concurrent requests: {concurrent_requests}")
    print(f"[Config] Processing: 1 review at a time (parallel across reviews)")
    
    # =========================================================================
    # Step 2: Initialize W&B run
    # =========================================================================
    print("\n" + "-" * 70)
    print("Step 2: Initialize W&B Run")
    print("-" * 70)
    
    run = init_wandb_run(
        run_name=f"review_topic_classification_{dataset_mode}",
        stage=stage_directory,
        job_type=job_type
    )
    
    # Validate stage dependencies (W&B only - no local fallback)
    required_artifacts = [input_artifact_reviews, input_artifact_topics]
    
    if not validate_stage_dependencies(run, stage_directory, required_artifacts):
        logging.error("Stage 02 and Stage 03 must be completed first!")
        logging.error("Please run Stage 02 and Stage 03 to create required artifacts.")
        return
    
    try:
        # =====================================================================
        # Step 3: Load topic universe (learned artifact) - MUST be shrunk version
        # =====================================================================
        print("\n" + "-" * 70)
        print("Step 3: Load Topic Universe (Learned Artifact)")
        print("-" * 70)
        print(f"[REQUIRED] Input artifact: {input_artifact_topics}")
        print(f"[REQUIRED] Artifact type: {artifact_type}")
        print(f"[NOTE] Must be topic_universe_final_* (final shrunk output from Stage 03)")
        print(f"[INFO] Downloading from W&B (NO local fallback)...")
        
        # Download topic universe artifact from W&B (ONLY - no local fallback)
        topics_path = use_artifact(run, input_artifact_topics, artifact_type=artifact_type)
        
        if topics_path is None:
            print(f"[ERROR] ✗ Could not download topic universe artifact from W&B: {input_artifact_topics}")
            print(f"[ERROR]   Make sure artifact '{input_artifact_topics}' exists in W&B")
            print(f"[ERROR]   Make sure to run Stage 03 (02_shrink_topics.py) first!")
            print(f"[ERROR]   No local fallback available - W&B is the only source")
            return
        
        # Resolve path to handle any symlinks or relative paths
        topics_path = Path(topics_path).resolve()
        print(f"[OK] ✓ Topic universe artifact downloaded to: {topics_path}")
        
        # Debug: Check if path exists and list contents
        if not topics_path.exists():
            print(f"[ERROR] ✗ Artifact directory does not exist: {topics_path}")
            print(f"[ERROR]   This indicates a problem with W&B artifact download")
            print(f"[ERROR]   Artifact name: {input_artifact_topics}")
            raise FileNotFoundError(f"W&B artifact directory does not exist: {topics_path}")
        
        print(f"[DEBUG] Artifact directory contents:")
        for item in topics_path.iterdir():
            print(f"  - {item.name} ({'file' if item.is_file() else 'dir'})")
        
        # Use exact filename from config (no fallback - W&B only)
        topic_universe_file = topics_path / input_topic_universe_filename
        
        if not topic_universe_file.exists():
            print(f"[ERROR] ✗ Topic universe file not found in W&B artifact: {topic_universe_file}")
            print(f"[ERROR]   Expected exact filename: {input_topic_universe_filename}")
            print(f"[ERROR]   Artifact path: {topics_path}")
            print(f"[ERROR]   Make sure Stage 03 (02_shrink_topics.py) uploaded the file correctly")
            print(f"[ERROR]   No local fallback available - W&B is the only source")
            raise FileNotFoundError(f"Required topic universe file '{input_topic_universe_filename}' not found in W&B artifact '{input_artifact_topics}'.")
        
        # Load and validate topic universe
        print(f"  Loading from: {topic_universe_file}")
        topic_universe_artifact = TopicUniverseArtifact.from_file(topic_universe_file)
        print(f"[OK] Loaded topic universe with {len(topic_universe_artifact.topics_by_category)} categories")
        
        # =====================================================================
        # Step 4: Download Reviews Artifact from W&B (ONLY - no local fallback)
        # =====================================================================
        print("\n" + "-" * 70)
        print("Step 4: Download Reviews Artifact from W&B")
        print("-" * 70)
        print(f"[REQUIRED] Input artifact: {input_artifact_reviews}")
        print(f"[REQUIRED] Artifact type: {artifact_type}")
        print(f"[INFO] Downloading from W&B (NO local fallback)...")
        
        # Download reviews artifact from W&B (ONLY - no local fallback)
        reviews_path = use_artifact(run, input_artifact_reviews, artifact_type=artifact_type)
        
        if reviews_path is None:
            print(f"[ERROR] ✗ Could not download reviews artifact from W&B: {input_artifact_reviews}")
            print(f"[ERROR]   Make sure artifact '{input_artifact_reviews}' exists in W&B")
            print(f"[ERROR]   Make sure Stage 02 has been completed and artifact uploaded")
            print(f"[ERROR]   No local fallback available - W&B is the only source")
            return
        
        # Resolve path to handle any symlinks or relative paths
        reviews_path = Path(reviews_path).resolve()
        print(f"[OK] ✓ Reviews artifact downloaded to: {reviews_path}")
        
        # =====================================================================
        # Step 5: Load reviews data
        # =====================================================================
        print("\n" + "-" * 70)
        print("Step 5: Load Reviews Data")
        print("-" * 70)
        
        # Load reviews - use exact filenames from config (no fallback)
        review_file = None
        for filename in review_filenames:
            candidate = reviews_path / filename
            if candidate.exists():
                review_file = candidate
                logging.info(f"Found review file: {filename}")
                break
        
        if review_file is None:
            # List available files for debugging
            error_msg = f"Review file not found in artifact: {reviews_path}\n"
            error_msg += f"Expected filenames (in order): {review_filenames}\n"
            error_msg += f"Artifact path: {reviews_path}\n"
            error_msg += f"[DEBUG] Listing files in artifact directory:\n"
            if reviews_path.exists():
                for item in reviews_path.iterdir():
                    error_msg += f"  - {item.name} ({'file' if item.is_file() else 'dir'})\n"
            else:
                error_msg += f"  [ERROR] Artifact directory does not exist!\n"
            logging.error(error_msg)
            raise FileNotFoundError(f"Required review file not found in W&B artifact. {error_msg}")
        
        try:
            all_reviews = load_review_data(review_file)
        except (FileNotFoundError, ValueError, RuntimeError) as e:
            logging.error(f"Failed to load review data: {e}")
            raise  # Re-raise to fail the run
        
        if not all_reviews:
            logging.warning("Review file loaded but contains no reviews")
            logging.error("No reviews to process")
            return
        
        # TEST MODE: Limit to 10 reviews for testing
        test_limit = int(os.environ.get("TEST_LIMIT", "0"))
        if test_limit > 0:
            logging.info(f"[TEST MODE] Limiting to {test_limit} reviews")
            all_reviews = all_reviews[:test_limit]
        
        # Group reviews by MAIN category (using mapping)
        reviews_by_main_category = defaultdict(list)
        original_category_counts = defaultdict(int)
        
        for review in all_reviews:
            original_category = review.get('category', 'Unknown')
            main_category = review.get('main_category')  # Already mapped in load_review_data
            
            if main_category:
                reviews_by_main_category[main_category].append(review)
                original_category_counts[original_category] += 1
        
        logging.info(f"\nReviews by original category:")
        for cat, count in sorted(original_category_counts.items(), key=lambda x: -x[1]):
            mapped = map_category_to_main_category(cat)
            logging.info(f"  {cat}: {count} reviews → {mapped}")
        
        logging.info(f"\nReviews grouped by main category:")
        for main_cat, reviews in reviews_by_main_category.items():
            logging.info(f"  {main_cat}: {len(reviews)} reviews")
        
        # Get themes for each MAIN category from learned artifact
        themes_by_category = {}
        available_categories = list(topic_universe_artifact.topics_by_category.keys())
        
        for main_category in list(reviews_by_main_category.keys()):
            # Get topics for this main category from the learned artifact
            category_topics = topic_universe_artifact.get_topics_for_category(main_category)
            if category_topics:
                themes_by_category[main_category] = category_topics
                logging.info(f"  Loaded {len(category_topics)} topics for {main_category}")
            else:
                logging.warning(f"No topics found for {main_category}, skipping")
                del reviews_by_main_category[main_category]
        
        # Update variable name for consistency
        reviews_by_category = reviews_by_main_category
        
        if not reviews_by_category:
            logging.error("No categories with topics to process")
            print(f"  Available categories in topic universe: {available_categories}")
            return
        
        # =====================================================================
        # Step 6: Process reviews (saves locally incrementally)
        # =====================================================================
        print("\n" + "-" * 70)
        print(f"Step 6: Process Reviews ({'PARALLEL' if use_parallel else 'SEQUENTIAL'})")
        print("-" * 70)
        print(f"[INFO] Output will be saved locally FIRST (incremental saves)")
        print(f"[INFO] → W&B upload will happen after processing completes")
        
        # Setup output (local directory - always saved)
        output_dir = get_artifact_dir(stage_directory, output_artifact_name)
        output_dir.mkdir(parents=True, exist_ok=True)
        output_filename = output_filename_pattern.format(dataset_mode=dataset_mode)
        output_file = output_dir / output_filename
        
        # Setup files for polluted data and failed reviews
        polluted_data_file = output_dir / f"polluted_data_{dataset_mode}.jsonl"
        failed_reviews_file = output_dir / f"failed_reviews_{dataset_mode}.jsonl"
        
        # Clear these files at start (in case of rerun)
        if polluted_data_file.exists():
            polluted_data_file.unlink()
        if failed_reviews_file.exists():
            failed_reviews_file.unlink()
        
        print(f"[INFO] Output directory: {output_dir}")
        print(f"[INFO] Output file: {output_file}")
        print(f"[INFO] Polluted data file: {polluted_data_file}")
        print(f"[INFO] Failed reviews file (for rerun): {failed_reviews_file}")
        
        # Prepare semaphore for rate limiting
        semaphore = asyncio.Semaphore(concurrent_requests)
        
        total_reviews = sum(len(r) for r in reviews_by_category.values())
        logging.info(f"\nProcessing {total_reviews} reviews across {len(reviews_by_category)} categories")
        logging.info(f"Processing mode: 1 review at a time (parallel across reviews)")
        logging.info(f"Concurrent requests: {concurrent_requests}")
        
        start_time = time.time()
        last_save_count = 0
        
        # Process categories (sequential or parallel)
        if use_parallel and len(reviews_by_category) > 1:
            # Process all categories in parallel
            category_tasks = []
            for category, reviews in reviews_by_category.items():
                topics = themes_by_category.get(category)
                if not topics or not reviews:
                    continue
                
                task = process_category_reviews(
                    category=category,
                    reviews=reviews,
                    topic_universe=topics,
                    client=llm_client,
                    model_name=model_name,
                    semaphore=semaphore,
                    max_retries=max_retries,
                    retry_delay=retry_delay,
                    concurrent_requests=concurrent_requests,
                    output_file=output_file,
                    polluted_data_file=polluted_data_file,
                    failed_reviews_file=failed_reviews_file,
                    save_interval=save_interval
                )
                category_tasks.append((category, task))
            
            # Execute all category tasks in parallel
            all_results = await asyncio.gather(*[task for _, task in category_tasks])
            
            # Results are already saved incrementally inside process_category_reviews
            # No need to save again here (would cause duplicates)
            for (category, _), results in zip(category_tasks, all_results):
                if results:
                    # Results already saved, just update progress metrics
                    if processed_count - last_save_count >= save_interval:
                        elapsed = time.time() - start_time
                        rate = processed_count / elapsed if elapsed > 0 else 0
                        logging.info(f"[Progress] {processed_count}/{total_reviews} "
                                   f"({processed_count/total_reviews*100:.1f}%) "
                                   f"Rate: {rate:.2f}/sec")
                        last_save_count = processed_count
                        
                        # Log progress metrics
                        log_metrics(run, {
                            "reviews_processed": processed_count,
                            "reviews_failed": failed_count,
                            "input_tokens": total_input_tokens,
                            "output_tokens": total_output_tokens,
                        })
        else:
            # Process categories sequentially
            for category, reviews in reviews_by_category.items():
                topics = themes_by_category.get(category)
                if not topics or not reviews:
                    continue
                
                logging.info(f"\nProcessing {len(reviews)} reviews for: {category}")
                
                # Process reviews (one at a time, parallel across reviews)
                # Note: Results are already saved incrementally inside process_category_reviews
                results = await process_category_reviews(
                    category=category,
                    reviews=reviews,
                    topic_universe=topics,
                    client=llm_client,
                    model_name=model_name,
                    semaphore=semaphore,
                    max_retries=max_retries,
                    retry_delay=retry_delay,
                    concurrent_requests=concurrent_requests,
                    output_file=output_file,
                    polluted_data_file=polluted_data_file,
                    failed_reviews_file=failed_reviews_file,
                    save_interval=save_interval
                )
                
                # Results are already saved incrementally inside process_category_reviews
                # No need to save again here (would cause duplicates)
        
        # =====================================================================
        # Step 7: Final summary and upload
        # =====================================================================
        print("\n" + "-" * 70)
        print("Step 7: Final Summary")
        print("-" * 70)
        
        elapsed_time = time.time() - start_time
        cost_input = (total_input_tokens / 1_000_000) * cost_per_1m_input
        cost_output = (total_output_tokens / 1_000_000) * cost_per_1m_output
        total_cost = cost_input + cost_output
        
        logging.info(f"\nTotal reviews processed: {processed_count}")
        logging.info(f"Total reviews failed (after retries): {failed_count}")
        logging.info(f"Reviews with empty themes (saved to output): {empty_themes_count}")
        logging.info(f"Polluted data (missing fields, saved separately): {polluted_data_count}")
        logging.info(f"Processing time: {elapsed_time/60:.1f} minutes")
        logging.info(f"Total cost: ${total_cost:.4f}")
        
        # Log final metrics
        log_summary(run, {
            "final_processed": processed_count,
            "final_failed": failed_count,
            "empty_themes_count": empty_themes_count,
            "polluted_data_count": polluted_data_count,
            "total_cost_usd": total_cost,
            "total_input_tokens": total_input_tokens,
            "total_output_tokens": total_output_tokens,
        })
        
        # Upload artifact
        print("\n" + "-" * 70)
        print("Step 8: Upload Artifact to W&B")
        print("-" * 70)
        print(f"[INFO] ✓ Local files already saved at: {output_dir}")
        print(f"[INFO] → Now uploading to W&B: {output_artifact_name}")
        print(f"[INFO]   Artifact type: {artifact_type}")
        
        artifact_uploaded = False
        wandb_success = True
        try:
            print(f"[INFO] Creating comprehensive artifact metadata...")
            print(f"[INFO] Uploading artifact to W&B (this may take a moment)...")
            artifact = log_artifact(
                run=run,
                artifact_name=output_artifact_name,
                artifact_type=artifact_type,
                artifact_path=output_dir,
                metadata=create_comprehensive_artifact_metadata(
                    stage=stage_directory,
                    artifact_name=output_artifact_name,
                    sample_size=processed_count,
                    model_vendor="Bedrock" if (os.environ.get("OPENAI_BASE_URL") or "").strip().find("bedrock-mantle") >= 0 else "OpenAI",
                    model_name=model_name,
                    model_description="LLM for review topic classification with probability scores",
                    model_params={
                        "temperature": 0.7,
                        "response_format": "json_object",
                    },
                    learned_artifact_schema=get_learned_artifact_schema(stage_directory, output_artifact_name),
                    additional_metadata={
                        "dataset_mode": dataset_mode,
                        "num_reviews_processed": processed_count,
                        "num_failed": failed_count,
                        "total_cost_usd": total_cost,
                        "input_artifact_reviews": input_artifact_reviews,
                        "input_artifact_topics": input_artifact_topics,
                    }
                )
            )
            
            print(f"[INFO] Linking artifact to registry...")
            link_to_registry(artifact, stage=stage_directory)
            artifact_uploaded = True
            print(f"[OK] ✓ Artifact successfully uploaded to W&B: {output_artifact_name}")
            if run:
                print(f"[INFO]   View artifact in W&B run: {run.url}")
        except Exception as e:
            print(f"[ERROR] ✗ Failed to upload artifact to W&B: {e}")
            print(f"[INFO] ✓ Artifact saved locally at: {output_dir}")
            print(f"[INFO]   Local files are available regardless of W&B upload status")
            print(f"[INFO]   You can retry the W&B upload later if needed")
            wandb_success = False
        
        # =====================================================================
        # Final Summary
        # =====================================================================
        print("\n" + "=" * 70)
        print("STAGE 04 COMPLETE")
        print("=" * 70)
        print(f"\nSummary:")
        print(f"  Reviews processed successfully: {processed_count}")
        print(f"  Reviews failed (after all retries): {failed_count}")
        print(f"  Reviews with empty themes (saved to output): {empty_themes_count}")
        print(f"  Polluted data (missing fields): {polluted_data_count}")
        print(f"  Total cost: ${total_cost:.4f}")
        
        print(f"\n" + "=" * 70)
        print("W&B Upload Status")
        print("=" * 70)
        if wandb_success and artifact_uploaded:
            print(f"[OK] ✓ W&B Upload: SUCCESS")
            print(f"  - Artifact: {output_artifact_name}")
            print(f"  - Artifact type: {artifact_type}")
            if run:
                print(f"  - View run: {run.url}")
        else:
            print(f"[WARNING] ✗ W&B Upload: FAILED")
            if artifact_uploaded:
                print(f"  - Artifact: {output_artifact_name} (uploaded but metrics failed)")
            else:
                print(f"  - Artifact: {output_artifact_name} (upload failed)")
            print(f"\n[INFO] ✓ All outputs are saved locally regardless of W&B status")
            print(f"  Local files are available at: {output_dir}")
            print(f"  You can retry the W&B upload later if needed")
        
        print(f"\n[INFO] Local files (always saved):")
        print(f"  - Directory: {output_dir}")
        print(f"  - Output file: {output_file}")
        if output_file.exists():
            file_size = output_file.stat().st_size / (1024 * 1024)  # Size in MB
            print(f"  - File size: {file_size:.2f} MB")
        
        if polluted_data_file.exists():
            polluted_size = polluted_data_file.stat().st_size / (1024 * 1024)  # Size in MB
            print(f"  - Polluted data file: {polluted_data_file.name} ({polluted_size:.2f} MB, {polluted_data_count} reviews)")
        
        if failed_reviews_file.exists():
            failed_size = failed_reviews_file.stat().st_size / (1024 * 1024)  # Size in MB
            print(f"  - Failed reviews file (for rerun): {failed_reviews_file.name} ({failed_size:.2f} MB, {failed_count} reviews)")
            print(f"    → You can rerun these reviews by processing this file separately")
        
        if run and wandb_success and artifact_uploaded:
            print(f"\nView run at: {run.url}")
        
    finally:
        finish_run(run)


if __name__ == "__main__":
    asyncio.run(main())
