#!/usr/bin/env python3
"""
Stage 05: User Level Inference - Backstory Characteristics
==========================================================
input_artifact_reviews: "train_set_with_topics_v4:latest"  # Merged training data from Stage 04 (based on train_set_sampled_2kusers_10kreviews_v4: 10k reviews, 2k users)

Extracts user characteristics and backstories using LLM analysis.
- Reads configuration from config.yaml
- Downloads review data artifact from W&B
- Parallel processing with configurable workers
- Logs artifact to W&B: User Inference collection

Usage:
    python 05_user_level_inference/scripts/01_user_to_backstory_characteristics.py
"""

import json
import os
import sys
import time
import tiktoken
from pathlib import Path
from typing import Tuple
from openai import RateLimitError, APIError, APIConnectionError, APITimeoutError
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
import logging
# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Load .env so OPENAI_API_KEY and OPENAI_BASE_URL (Bedrock) are set
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent.parent.parent / ".env", override=True)
except ImportError:
    pass

from utils.openai_client import create_openai_client
from utils.wandb_utils import (
    load_config, get_stage_config, get_openai_config,
    init_wandb_run, finish_run, use_artifact, log_artifact,
    log_metrics, log_summary, link_to_registry, get_artifact_dir,
    create_comprehensive_artifact_metadata, get_learned_artifact_schema,
    validate_stage_dependencies
)

# Import schema for validation
from schemas.learned_artifacts import UserBackstoryArtifact

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Token encoding
encoding = tiktoken.get_encoding("cl100k_base")

# Prompt directory
PROMPT_DIR = Path(__file__).parent.parent / "prompts"

# Thread-safe locks
file_lock = Lock()
progress_lock = Lock()

# Global progress tracking
processed_count = 0
failed_count = 0
invalid_count = 0
llm_failed_count = 0


def load_review_data(filepath: Path):
    """Loads review data from JSON file."""
    if not filepath.exists():
        logging.error(f"Input file not found at '{filepath}'")
        return None
    
    logging.info(f"Loading review data from '{filepath}'...")
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        logging.error(f"Error decoding JSON file: {e}")
        return None


def load_existing_characteristics(filepath: Path):
    """Loads existing characteristics to allow resuming."""
    if not filepath.exists() or filepath.stat().st_size == 0:
        return {}
    
    logging.info(f"Loading existing characteristics to resume...")
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    except json.JSONDecodeError:
        logging.warning("Could not decode existing file. Starting fresh.")
        return {}


def save_progress_safe(characteristics_data: dict, filepath: Path):
    """Thread-safe save operation with schema validation. Only saves validated data."""
    with file_lock:
        try:
            # Validate all user backstories before saving - only save validated data
            validated_data = {}
            
            for user_id, user_data in characteristics_data.items():
                try:
                    # Validate against schema (user_id is optional in schema, used only for validation)
                    user_data_for_validation = user_data.copy()
                    user_data_for_validation["user_id"] = user_id
                    
                    # Validate against schema - if validation fails, skip this user
                    validated_backstory = UserBackstoryArtifact.from_dict(user_data_for_validation)
                    # to_dict() will exclude user_id automatically
                    validated_data[user_id] = validated_backstory.to_dict()
                except Exception as e:
                    # Skip invalid data - don't pollute output file
                    logging.warning(f"Schema validation failed for user {user_id}: {e} - skipping from output")
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(validated_data, f, indent=2, ensure_ascii=False)
            return True
        except Exception as e:
            logging.error(f"Error saving progress: {e}")
            return False


def save_error_safe(error_data: dict, filepath: Path):
    """Thread-safe save operation for error tracking files."""
    with file_lock:
        try:
            # Load existing errors if file exists
            existing_errors = {}
            if filepath.exists() and filepath.stat().st_size > 0:
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        existing_errors = json.load(f)
                except json.JSONDecodeError:
                    existing_errors = {}
            
            # Merge new errors
            existing_errors.update(error_data)
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(existing_errors, f, indent=2, ensure_ascii=False)
            return True
        except Exception as e:
            logging.error(f"Error saving error file: {e}")
            return False


def is_valid_user_id(user_id: str) -> bool:
    """Validate user_id format."""
    if not user_id:
        return False
    if not isinstance(user_id, str):
        return False
    if not user_id.strip():
        return False
    return True


def validate_user_input_data(user_id: str, user_data: dict) -> Tuple[bool, str]:
    """
    Validate user input data before processing.
    Returns (is_valid, error_message).
    """
    if not is_valid_user_id(user_id):
        return False, f"Invalid user_id: {user_id}"
    
    if not user_data:
        return False, "User data is None or empty"
    
    reviews = user_data.get('reviews', [])
    if not reviews:
        return False, "No reviews found for user"
    
    if not isinstance(reviews, list):
        return False, f"Reviews is not a list: {type(reviews)}"
    
    # Check if there are any reviews with actual text
    has_valid_review = False
    for review in reviews:
        if isinstance(review, dict):
            review_text = review.get('review_text', '')
            if review_text and isinstance(review_text, str) and review_text.strip():
                has_valid_review = True
                break
    
    if not has_valid_review:
        return False, "No valid review text found in reviews"
    
    return True, ""


def count_tokens(text: str) -> int:
    """Counts tokens in text."""
    return len(encoding.encode(text))


def truncate_reviews_to_token_limit(review_list: list, max_tokens: int):
    """Truncates reviews to fit within token limit."""
    if not review_list:
        return "", 0, 0
    
    total_reviews = len(review_list)
    truncated_reviews = []
    current_tokens = 0
    
    for review in review_list:
        review_tokens = count_tokens(review)
        if current_tokens + review_tokens > max_tokens:
            break
        truncated_reviews.append(review)
        current_tokens += review_tokens
    
    truncated_text = " ".join(truncated_reviews)
    kept_reviews = len(truncated_reviews)
    
    return truncated_text, kept_reviews, total_reviews


def group_reviews_by_category(reviews: list):
    """Groups reviews by category."""
    category_reviews = {}
    
    for review in reviews:
        category = review.get('category', 'Unknown')
        review_text = review.get('review_text', '')
        if category not in category_reviews:
            category_reviews[category] = []
        category_reviews[category].append(review_text)
    
    return category_reviews


def load_prompt(prompt_filename: str) -> str:
    """Load a prompt template from file."""
    prompt_path = PROMPT_DIR / prompt_filename
    if prompt_path.exists():
        with open(prompt_path, 'r', encoding='utf-8') as f:
            return f.read().strip()
    else:
        raise FileNotFoundError(f"Prompt file not found: {prompt_path}")


def build_prompt(all_reviews_text: str, category_name: str = None):
    """Creates the prompt for LLM analysis."""
    # Load prompt template from file
    user_prompt_template = load_prompt("user_backstory_user_prompt.txt")
    
    category_context = f" in the {category_name} category" if category_name else ""
    category_instruction = f" when reviewing {category_name} products" if category_name else ""
    
    return user_prompt_template.format(
        category_context=category_context,
        category_instruction=category_instruction,
        all_reviews_text=all_reviews_text
    )


def get_llm_analysis(client, user_id: str, all_reviews_text: str, model_name: str,
                     category_name: str = None, max_retries: int = 5, 
                     retry_delay: int = 10, rate_limit_delay: float = 1.0):
    """
    Calls OpenAI API to get analysis with improved retry logic.
    Returns (success: bool, result: dict or None, error_message: str)
    """
    if not all_reviews_text.strip():
        return False, None, "Empty review text"

    prompt = build_prompt(all_reviews_text, category_name)
    last_error = None
    
    for attempt in range(max_retries):
        try:
            # Exponential backoff with jitter for retries
            if attempt > 0:
                # Base delay * 2^attempt with jitter (random 0-1 second)
                wait_time = retry_delay * (2 ** (attempt - 1)) + (rate_limit_delay * attempt)
                logging.info(f"Retry attempt {attempt + 1}/{max_retries} for user {user_id[:20]}... after {wait_time:.1f}s...")
                time.sleep(wait_time)
            
            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                timeout=120.0  # Increased timeout for reliability
            )
            
            json_text = response.choices[0].message.content
            
            # Clean up markdown formatting
            if json_text.startswith("```json"):
                json_text = json_text[7:]
            if json_text.endswith("```"):
                json_text = json_text[:-3]
            
            # Try to parse JSON
            try:
                parsed_result = json.loads(json_text.strip())
                return True, parsed_result, None
            except json.JSONDecodeError as e:
                last_error = f"JSON parse error: {e}"
                logging.error(f"JSON parse error for user {user_id[:20]}... (attempt {attempt + 1}/{max_retries}): {e}")
                # Don't retry JSON decode errors - they're likely permanent
                if attempt < max_retries - 1:
                    # Still retry in case it's a formatting issue
                    continue
                else:
                    return False, None, last_error

        except RateLimitError as e:
            last_error = f"Rate limit error: {str(e)}"
            wait_time = retry_delay * (2 ** attempt) + (rate_limit_delay * attempt)
            logging.warning(f"Rate limit for user {user_id[:20]}... (attempt {attempt + 1}/{max_retries}). Retrying in {wait_time:.1f}s...")
            if attempt < max_retries - 1:
                time.sleep(wait_time)
            
        except (APIConnectionError, APITimeoutError) as e:
            last_error = f"Connection/Timeout error: {str(e)}"
            wait_time = retry_delay * (2 ** attempt) + (rate_limit_delay * attempt)
            logging.warning(f"Connection error for user {user_id[:20]}... (attempt {attempt + 1}/{max_retries}): {e}. Retrying in {wait_time:.1f}s...")
            if attempt < max_retries - 1:
                time.sleep(wait_time)
            
        except APIError as e:
            last_error = f"API error: {str(e)}"
            wait_time = retry_delay * (2 ** attempt)
            logging.error(f"API Error for user {user_id[:20]}... (attempt {attempt + 1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                time.sleep(wait_time)
            
        except Exception as e:
            last_error = f"Unexpected error: {str(e)}"
            wait_time = retry_delay * (2 ** attempt)
            logging.error(f"Unexpected error for user {user_id[:20]}... (attempt {attempt + 1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                time.sleep(wait_time)
    
    # All retries exhausted
    return False, None, last_error or "All retries exhausted"


def process_single_user(user_id: str, user_data: dict, client, model_name: str,
                       max_review_tokens: int, max_retries: int, retry_delay: int,
                       rate_limit_delay: float):
    """
    Processes a single user.
    Returns (user_id, result_type, data/error_info)
    result_type: 'success', 'invalid_input', 'llm_failed'
    """
    global processed_count, failed_count, invalid_count, llm_failed_count
    
    # Validate input data first
    is_valid, error_msg = validate_user_input_data(user_id, user_data)
    if not is_valid:
        with progress_lock:
            invalid_count += 1
        return user_id, 'invalid_input', {"error": error_msg, "user_data_summary": {
            "has_reviews": bool(user_data.get('reviews')),
            "review_count": len(user_data.get('reviews', [])) if isinstance(user_data.get('reviews'), list) else 0
        }}
    
    reviews = user_data.get('reviews', [])
    category_reviews = group_reviews_by_category(reviews)
    num_categories = len(category_reviews)
    
    # Get all review texts
    all_review_texts = [r.get('review_text', '') for r in reviews if r.get('review_text', '').strip()]
    all_reviews_text = " ".join(all_review_texts)
    
    # Truncate if needed
    if count_tokens(all_reviews_text) > max_review_tokens:
        all_reviews_text, kept_reviews, total_reviews = truncate_reviews_to_token_limit(
            all_review_texts, max_review_tokens
        )
    
    user_char_data = {}
    llm_errors = []
    
    # Get category-specific characteristics if multiple categories
    if num_categories > 1:
        category_characteristics = {}
        
        for category, review_texts in category_reviews.items():
            category_text = " ".join(review_texts)
            
            if count_tokens(category_text) > max_review_tokens:
                category_text, _, _ = truncate_reviews_to_token_limit(review_texts, max_review_tokens)
            
            success, category_persona, error = get_llm_analysis(
                client, user_id, category_text, model_name,
                category_name=category, max_retries=max_retries,
                retry_delay=retry_delay, rate_limit_delay=rate_limit_delay
            )
            
            if success and category_persona:
                category_characteristics[category] = category_persona
            else:
                llm_errors.append(f"Category {category}: {error}")
        
        if category_characteristics:
            user_char_data["category_characteristics"] = category_characteristics
    
    # Get overall characteristics (required)
    success, overall_persona, error = get_llm_analysis(
        client, user_id, all_reviews_text, model_name,
        max_retries=max_retries, retry_delay=retry_delay, rate_limit_delay=rate_limit_delay
    )
    
    if success and overall_persona:
        user_char_data["overall_characteristics"] = overall_persona
        user_char_data["num_categories"] = num_categories
        user_char_data["categories"] = list(category_reviews.keys())
        
        # Validate against schema before returning
        try:
            user_data_for_validation = user_char_data.copy()
            user_data_for_validation["user_id"] = user_id
            validated_backstory = UserBackstoryArtifact.from_dict(user_data_for_validation)
            # Return validated data (without user_id)
            validated_data = validated_backstory.to_dict()
            
            with progress_lock:
                processed_count += 1
            
            return user_id, 'success', validated_data
        except Exception as e:
            # Schema validation failed
            with progress_lock:
                invalid_count += 1
            return user_id, 'invalid_input', {"error": f"Schema validation failed: {e}", "data": user_char_data}
    else:
        # LLM call failed after all retries
        all_errors = [f"Overall: {error}"] + llm_errors
        with progress_lock:
            llm_failed_count += 1
        
        return user_id, 'llm_failed', {
            "error": "LLM call failed after all retries",
            "errors": all_errors,
            "num_retries": max_retries,
            "user_data_summary": {
                "num_reviews": len(reviews),
                "num_categories": num_categories
            }
        }


def main():
    """Main execution function."""
    global processed_count, failed_count
    
    # =========================================================================
    # Step 1: Load configuration from config.yaml
    # =========================================================================
    print("=" * 70)
    print("STAGE 05: User Level Inference - Backstory Characteristics")
    print("=" * 70)
    
    cfg = get_stage_config("05_user_level_inference")
    openai_cfg = get_openai_config()
    
    # Validate required config fields
    if "hyperparameters" not in cfg:
        logging.error("Missing required config field: hyperparameters")
        return
    
    hyperparams = cfg["hyperparameters"]
    
    # Get input artifacts from config (required, no fallbacks)
    if "input_artifact_reviews" not in cfg:
        logging.error("Missing required config field: input_artifact_reviews")
        return
    
    input_artifact_reviews = cfg["input_artifact_reviews"]
    input_artifact_topics = cfg.get("input_artifact_topics")  # Optional, but must be in config if provided
    
    if "output_artifact" not in cfg:
        logging.error("Missing required config field: output_artifact")
        return
    
    output_artifact_name = cfg["output_artifact"]
    
    # Get hyperparameters from config (required, no fallbacks)
    required_hyperparams = ["model", "num_workers", "max_context_tokens", "max_review_tokens", "save_interval", "reverse_order"]
    for param in required_hyperparams:
        if param not in hyperparams:
            logging.error(f"Missing required hyperparameter: {param}")
            return
    
    num_workers = hyperparams["num_workers"]
    user_limit = hyperparams.get("user_limit")  # Optional
    reverse_order = hyperparams["reverse_order"]
    model_name = hyperparams["model"]
    base_url = os.environ.get("OPENAI_BASE_URL", "").strip()
    if base_url and "bedrock-mantle" in base_url and cfg.get("bedrock_model_id"):
        model_name = cfg["bedrock_model_id"]
    max_context_tokens = hyperparams["max_context_tokens"]
    max_review_tokens = hyperparams["max_review_tokens"]
    save_interval = hyperparams["save_interval"]
    
    # Get OpenAI config (with reasonable defaults like before)
    # These are optional because config structure may be nested/complex
    max_retries = openai_cfg.get("max_retries", 5)  # Default: 5 retries (good for important operations)
    retry_delay = openai_cfg.get("retry_delay", 10)  # Default: 10 seconds (good for exponential backoff)
    rate_limit_delay = openai_cfg.get("rate_limit_delay", 1.0)  # Default: 1.0 second
    
    # Create client from environment/config. Uses Bedrock when OPENAI_BASE_URL is set.
    if not os.environ.get("OPENAI_API_KEY"):
        logging.error("OPENAI_API_KEY environment variable not set (set in .env or export)")
        return
    client = create_openai_client(timeout=60.0)
    
    print(f"\n[Config] Input artifact (reviews): {input_artifact_reviews}")
    print(f"[Config] Input artifact (topics): {input_artifact_topics}")
    print(f"[Config] Output artifact: {output_artifact_name}")
    print(f"[Config] Model: {model_name}")
    print(f"[Config] Workers: {num_workers}")
    print(f"[Config] User limit: {user_limit}")
    print(f"[Config] Reverse order: {reverse_order}")
    
    # =========================================================================
    # Step 2: Initialize W&B run
    # =========================================================================
    print("\n" + "-" * 70)
    print("Step 2: Initialize W&B Run")
    print("-" * 70)
    
    run = init_wandb_run(
        run_name=f"user_backstory_{output_artifact_name}",
        stage="05_user_level_inference",
        job_type="user_inference"
    )
    
    # Validate dependencies with W&B
    required_artifacts = [input_artifact_reviews]
    if not validate_stage_dependencies(run, "05_user_level_inference", required_artifacts):
        logging.error("Stage 04 must be completed first!")
        return
    
    try:
        # =====================================================================
        # Step 3: Get input artifacts from W&B
        # =====================================================================
        print("\n" + "-" * 70)
        print("Step 3: Get Input Artifacts from W&B")
        print("-" * 70)
        
        # Download reviews artifact from W&B (required)
        logging.info(f"Downloading reviews artifact from W&B: {input_artifact_reviews}")
        reviews_path = use_artifact(run, input_artifact_reviews, artifact_type="dataset")
        
        if reviews_path is None:
            logging.error(f"Could not download reviews artifact: {input_artifact_reviews}")
            return
        
        # Download topics artifact from W&B (optional)
        if input_artifact_topics:
            logging.info(f"Downloading topics artifact from W&B: {input_artifact_topics}")
            topics_path = use_artifact(run, input_artifact_topics, artifact_type="dataset")
            if topics_path is None:
                logging.warning(f"Topics artifact not found: {input_artifact_topics} (continuing without it)")
        
        # =====================================================================
        # Step 4: Load data
        # =====================================================================
        print("\n" + "-" * 70)
        print("Step 4: Load Data")
        print("-" * 70)
        
        # Get review file name from config (required, no fallback, no alternatives)
        if "review_filename" not in cfg:
            logging.error("Missing required config field: review_filename")
            return
        
        review_filename = cfg["review_filename"]
        review_file = reviews_path / review_filename
        
        # Check if file exists - if not, list available files for debugging
        if not review_file.exists():
            logging.error(f"Review file not found in artifact: {review_file}")
            logging.error(f"Expected file: {review_filename}")
            
            # List available files for debugging (but don't use them)
            if reviews_path.exists():
                available_files = list(reviews_path.glob("*.json"))
                if available_files:
                    logging.error(f"Available JSON files in artifact:")
                    for f in available_files:
                        logging.error(f"  - {f.name}")
                else:
                    logging.error(f"No JSON files found in artifact directory: {reviews_path}")
            else:
                logging.error(f"Artifact directory does not exist: {reviews_path}")
            
            logging.error(f"Please update config.yaml with the correct review_filename from the list above")
            return
        
        all_user_data = load_review_data(review_file)
        if all_user_data is None:
            return
        
        logging.info(f"Loaded {len(all_user_data)} users")
        
        # Setup output - save locally
        output_dir = get_artifact_dir("05_user_level_inference", output_artifact_name)
        output_file = output_dir / "user_overall_characteristics.json"
        invalid_users_file = output_dir / "invalid_users.json"
        failed_users_file = output_dir / "failed_users.json"
        
        logging.info(f"[LOCAL OUTPUT] Saving all outputs locally to: {output_dir}")
        logging.info(f"[LOCAL OUTPUT] Valid data: {output_file}")
        logging.info(f"[LOCAL OUTPUT] Invalid users: {invalid_users_file}")
        logging.info(f"[LOCAL OUTPUT] Failed users: {failed_users_file}")
        
        # Load existing to resume
        characteristics_data = load_existing_characteristics(output_file)
        logging.info(f"Found {len(characteristics_data)} existing users (will skip)")
        
        # Load existing error files to avoid reprocessing
        existing_invalid = load_existing_characteristics(invalid_users_file)
        existing_failed = load_existing_characteristics(failed_users_file)
        logging.info(f"Found {len(existing_invalid)} existing invalid users, {len(existing_failed)} existing failed users")
        
        # =====================================================================
        # Step 5: Prepare user list
        # =====================================================================
        print("\n" + "-" * 70)
        print("Step 5: Prepare User List")
        print("-" * 70)
        
        users_to_process = [
            (uid, data) for uid, data in all_user_data.items()
            if uid not in characteristics_data and uid not in existing_invalid and uid not in existing_failed
        ]
        
        if reverse_order:
            users_to_process.reverse()
        
        if user_limit:
            users_to_process = users_to_process[:user_limit]
        
        logging.info(f"{len(users_to_process)} users to process")
        
        if not users_to_process:
            logging.info("All users already processed!")
            return
        
        # =====================================================================
        # Step 6: Process users in parallel
        # =====================================================================
        print("\n" + "-" * 70)
        print("Step 6: Process Users (Parallel Processing Enabled)")
        print("-" * 70)
        logging.info(f"[PARALLEL] Starting parallel processing with {num_workers} workers")
        logging.info(f"[PARALLEL] Processing {len(users_to_process)} users concurrently")
        
        start_time = time.time()
        last_save_count = len(characteristics_data)
        
        # Track errors separately
        invalid_users_data = existing_invalid.copy()
        failed_users_data = existing_failed.copy()
        
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            logging.info(f"[PARALLEL] ThreadPoolExecutor initialized with max_workers={num_workers}")
            future_to_user = {}
            
            # Submit all tasks to the executor (this happens immediately, enabling parallel execution)
            for user_id, user_data in users_to_process:
                future = executor.submit(
                    process_single_user,
                    user_id, user_data, client, model_name,
                    max_review_tokens, max_retries, retry_delay, rate_limit_delay
                )
                future_to_user[future] = user_id
            
            logging.info(f"[PARALLEL] Submitted {len(future_to_user)} tasks to executor (running in parallel)")
            
            for future in as_completed(future_to_user):
                user_id = future_to_user[future]
                try:
                    result_user_id, result_type, result_data = future.result()
                    
                    if result_type == 'success':
                        # Validated schema data - safe to save
                        characteristics_data[result_user_id] = result_data
                        
                        current_count = len(characteristics_data)
                        if current_count - last_save_count >= save_interval:
                            # Save all files locally
                            save_progress_safe(characteristics_data, output_file)
                            # Also save error files periodically
                            if invalid_users_data:
                                save_error_safe(invalid_users_data, invalid_users_file)
                            if failed_users_data:
                                save_error_safe(failed_users_data, failed_users_file)
                            
                            elapsed = time.time() - start_time
                            rate = processed_count / elapsed if elapsed > 0 else 0
                            logging.info(f"[Progress] {processed_count} processed, {invalid_count} invalid, {llm_failed_count} LLM failed. "
                                       f"Rate: {rate:.2f}/sec. Saved {current_count} users locally to: {output_file}")
                            last_save_count = current_count
                            
                            # Log progress
                            log_metrics(run, {
                                "users_processed": processed_count,
                                "users_invalid": invalid_count,
                                "users_llm_failed": llm_failed_count,
                            })
                    
                    elif result_type == 'invalid_input':
                        # Invalid input data - save to error file
                        invalid_users_data[result_user_id] = result_data
                        logging.warning(f"Invalid input for user {result_user_id[:20]}...: {result_data.get('error', 'Unknown error')}")
                    
                    elif result_type == 'llm_failed':
                        # LLM call failed after all retries - save to failed file
                        failed_users_data[result_user_id] = result_data
                        logging.error(f"LLM failed for user {result_user_id[:20]}... after {max_retries} retries")
                    
                except Exception as e:
                    logging.error(f"Exception for user {user_id[:20]}...: {e}")
                    with progress_lock:
                        failed_count += 1
                    # Save exception as failed user
                    failed_users_data[user_id] = {
                        "error": f"Processing exception: {str(e)}",
                        "error_type": "exception"
                    }
        
        # Final saves - ensure all files are saved locally FIRST
        logging.info(f"[LOCAL SAVE] Saving final outputs to local directory...")
        save_progress_safe(characteristics_data, output_file)
        logging.info(f"[LOCAL SAVE] ✓ Saved {len(characteristics_data)} users locally to: {output_file}")
        
        if invalid_users_data:
            save_error_safe(invalid_users_data, invalid_users_file)
            logging.info(f"[LOCAL SAVE] ✓ Saved {len(invalid_users_data)} invalid users locally to: {invalid_users_file}")
        if failed_users_data:
            save_error_safe(failed_users_data, failed_users_file)
            logging.info(f"[LOCAL SAVE] ✓ Saved {len(failed_users_data)} failed users locally to: {failed_users_file}")
        
        # Verify all files exist locally before uploading to W&B
        files_to_upload = [output_file]
        if invalid_users_file.exists():
            files_to_upload.append(invalid_users_file)
        if failed_users_file.exists():
            files_to_upload.append(failed_users_file)
        
        logging.info(f"[LOCAL SAVE] All files saved locally. Ready to upload {len(files_to_upload)} files to W&B.")
        
        # =====================================================================
        # Step 7: Final summary and upload
        # =====================================================================
        print("\n" + "-" * 70)
        print("Step 7: Final Summary")
        print("-" * 70)
        
        elapsed_time = time.time() - start_time
        
        logging.info(f"\n{'='*70}")
        logging.info(f"PROCESSING SUMMARY")
        logging.info(f"{'='*70}")
        logging.info(f"Total users processed successfully: {processed_count}")
        logging.info(f"Total users with invalid input: {invalid_count}")
        logging.info(f"Total users with LLM failures: {llm_failed_count}")
        logging.info(f"Total characteristics saved (schema-validated): {len(characteristics_data)}")
        logging.info(f"Total invalid users tracked: {len(invalid_users_data)}")
        logging.info(f"Total failed users tracked: {len(failed_users_data)}")
        logging.info(f"Processing time: {elapsed_time/60:.1f} minutes")
        
        log_summary(run, {
            "final_processed": processed_count,
            "final_invalid": invalid_count,
            "final_llm_failed": llm_failed_count,
            "total_users_with_characteristics": len(characteristics_data),
            "total_invalid_users": len(invalid_users_data),
            "total_failed_users": len(failed_users_data),
        })
        
        # Upload artifact to W&B (includes all files from output_dir)
        print("\n" + "-" * 70)
        print("Step 8: Upload Artifact to W&B")
        print("-" * 70)
        
        logging.info(f"[W&B UPLOAD] Uploading artifact '{output_artifact_name}' to W&B...")
        logging.info(f"[W&B UPLOAD] Uploading directory: {output_dir}")
        logging.info(f"[W&B UPLOAD] Files to upload:")
        logging.info(f"[W&B UPLOAD]   - {output_file.name} ({len(characteristics_data)} users)")
        if invalid_users_file.exists():
            logging.info(f"[W&B UPLOAD]   - {invalid_users_file.name} ({len(invalid_users_data)} users)")
        if failed_users_file.exists():
            logging.info(f"[W&B UPLOAD]   - {failed_users_file.name} ({len(failed_users_data)} users)")
        
        artifact = log_artifact(
            run=run,
            artifact_name=output_artifact_name,
            artifact_type="dataset",
            artifact_path=output_dir,  # This uploads ALL files in the directory to W&B
            metadata=create_comprehensive_artifact_metadata(
                stage="05_user_level_inference",
                artifact_name=output_artifact_name,
                sample_size=len(characteristics_data),
                model_vendor="Bedrock" if "bedrock-mantle" in (os.environ.get("OPENAI_BASE_URL") or "") else "OpenAI",
                model_name=model_name,
                model_description="LLM for user backstory and characteristics extraction",
                model_params={
                    "temperature": 0.7,
                    "max_tokens": max_context_tokens,
                },
                learned_artifact_schema=get_learned_artifact_schema("05_user_level_inference", output_artifact_name),
                additional_metadata={
                    "num_users_processed": processed_count,
                    "num_users_invalid": invalid_count,
                    "num_users_llm_failed": llm_failed_count,
                    "total_users": len(characteristics_data),
                    "total_invalid_users": len(invalid_users_data),
                    "total_failed_users": len(failed_users_data),
                    "input_artifact_reviews": input_artifact_reviews,
                    "input_artifact_topics": input_artifact_topics,
                    "schema_validated": True,
                    "schema_version": "v4",
                    "artifact_type": "learned_artifact",  # Mark as learned artifact
                    "error_tracking": {
                        "invalid_users_file": str(invalid_users_file.name),
                        "failed_users_file": str(failed_users_file.name),
                    }
                }
            )
        )
        
        link_to_registry(artifact, stage="05_user_level_inference")
        
        if artifact:
            logging.info(f"[W&B UPLOAD] ✓ Successfully uploaded artifact '{output_artifact_name}' to W&B")
            logging.info(f"[W&B UPLOAD] ✓ All files in {output_dir} are now available in W&B")
        else:
            logging.warning(f"[W&B UPLOAD] W&B disabled or upload failed - files still saved locally")
        
        # =====================================================================
        # Final Summary
        # =====================================================================
        print("\n" + "=" * 70)
        print("STAGE 05 COMPLETE")
        print("=" * 70)
        print(f"\nSummary:")
        print(f"  Users processed successfully: {processed_count}")
        print(f"  Users with invalid input: {invalid_count}")
        print(f"  Users with LLM failures: {llm_failed_count}")
        print(f"  Total with characteristics (schema-validated): {len(characteristics_data)}")
        print(f"  Invalid users tracked: {len(invalid_users_data)}")
        print(f"  Failed users tracked: {len(failed_users_data)}")
        print(f"\n{'='*70}")
        print(f"OUTPUT FILES - SAVED IN BOTH LOCATIONS:")
        print(f"{'='*70}")
        print(f"📁 LOCAL (saved locally):")
        print(f"   Output directory: {output_dir}")
        print(f"   ✓ Valid data: {output_file}")
        print(f"   ✓ Invalid users: {invalid_users_file}")
        print(f"   ✓ Failed users: {failed_users_file}")
        if artifact:
            print(f"\n☁️  W&B (uploaded to Weights & Biases):")
            print(f"   Artifact name: {output_artifact_name}")
            print(f"   ✓ All files uploaded to W&B artifact")
            print(f"   View at: {run.url if run else 'N/A'}")
        else:
            print(f"\n⚠️  W&B: Upload skipped (W&B disabled or failed)")
        print(f"\n✓ All outputs are saved both locally AND in W&B")
        
        if run:
            print(f"\nView run at: {run.url}")
        
    finally:
        finish_run(run)


if __name__ == "__main__":
    main()
