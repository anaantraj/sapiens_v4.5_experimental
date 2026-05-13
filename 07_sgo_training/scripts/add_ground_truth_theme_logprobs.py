"""
Script to add theme probabilities (e^logprobs) for GROUND TRUTH themes.
Downloads train_set_reviews_with_merged_predictions_cleaned from WandB,
and for each review, gets logprobs for the actual/ground truth themes,
then calculates e^logprobs and adds them as actual_theme_probs.

This is the same regardless of prediction method - we're getting logprobs
for the actual review text and actual themes.
"""
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
from openai import OpenAI
from openai import RateLimitError, APIError

# Try to import wandb utilities and config
try:
    script_dir = Path(__file__).parent
    project_root = script_dir.parent.parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    from utils.wandb_utils import download_artifact, use_artifact, init_wandb_run, log_artifact, finish_run
    HAS_WANDB = True
except ImportError as e:
    HAS_WANDB = False
    print(f"Warning: WandB utilities not available ({e})")

# Try to import settings and cost tracker
try:
    from sgo_training.config import settings
    from sgo_training.utils.cost_tracker import track_api_call, extract_token_usage
    HAS_SETTINGS = True
except ImportError:
    HAS_SETTINGS = False
    # Create a simple settings mock
    class Settings:
        API_RATE_LIMIT_DELAY = 0.1
        PATHS = {}
    settings = Settings()
    
    # Create simple cost tracker functions
    def track_api_call(*args, **kwargs):
        pass
    
    def extract_token_usage(response):
        return 0, 0, 0


class RateLimiter:
    """Simple rate limiter for API calls."""
    def __init__(self, min_interval: float = 0.1):
        self.min_interval = min_interval
        self.last_request_time = 0.0
    
    def wait(self):
        """Wait if necessary to respect rate limit."""
        time_since_last = time.time() - self.last_request_time
        if time_since_last < self.min_interval:
            time.sleep(self.min_interval - time_since_last)
        self.last_request_time = time.time()


def get_theme_logprobs_for_ground_truth(
    product_description: str,
    review_text: str,
    theme: str,
    client: OpenAI,
    rate_limiter: RateLimiter,
    theme_model_name: str = "gpt-4o-mini",
    max_retries: int = 3
) -> Dict[str, Any]:
    """
    Get yes/no logprobs for a ground truth theme using ONLY product_description and review_text.
    This is the same regardless of prediction method - we're classifying the actual review.
    
    Args:
        product_description: Product description
        review_text: Actual/ground truth review text
        theme: Theme name to classify
        client: OpenAI client
        rate_limiter: Rate limiter instance
        theme_model_name: Model to use for theme classification
        max_retries: Maximum retry attempts
    
    Returns:
        Dictionary with yes/no logprobs and tokens
    """
    for attempt in range(max_retries):
        try:
            rate_limiter.wait()
            
            system_msg = "You are a precise theme classifier. Answer only Yes or No."
            user_msg = f'Analyze this product review and determine if it discusses the theme "{theme}".\n\nProduct Description: {product_description}\n\nReview: "{review_text}"\n\nDoes this review discuss the theme "{theme}"?\nAnswer with ONLY "Yes" or "No".'
            
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
                    "top_logprobs": 20
                }
                
                # Use max_completion_tokens for gpt-5 models, max_tokens for others
                if "gpt-5" in theme_model_name.lower():
                    request_params["max_completion_tokens"] = 5
                else:
                    request_params["max_tokens"] = 5
                
                response = client.chat.completions.create(**request_params)
                
                # Track API call for cost calculation
                input_tokens, output_tokens, cached_tokens = extract_token_usage(response)
                track_api_call(
                    model=theme_model_name,
                    call_type="ground_truth_theme_logprobs",
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    cached_input_tokens=cached_tokens,
                    prompt_length=len(system_msg) + len(user_msg),
                    metadata={"theme": theme, "attempt": attempt + 1}
                )
                
                # Find logprob for "yes" and "no"
                yes_logprob = -100.0
                no_logprob = -100.0
                yes_token = None
                no_token = None
                yes_found_in_main = False
                no_found_in_main = False
                
                logprobs_data = response.choices[0].logprobs
                
                if logprobs_data and logprobs_data.content:
                    # First pass: Check main tokens
                    for token_info in logprobs_data.content:
                        token_clean = token_info.token.lower().strip()
                        
                        if token_clean in ["yes", "y"]:
                            yes_logprob = token_info.logprob
                            yes_token = token_info.token
                            yes_found_in_main = True
                        elif token_clean in ["no", "n"]:
                            no_logprob = token_info.logprob
                            no_token = token_info.token
                            no_found_in_main = True
                    
                    # Second pass: Check alternatives if needed
                    if not yes_found_in_main or not no_found_in_main:
                        for token_info in logprobs_data.content:
                            if token_info.top_logprobs:
                                for alt in token_info.top_logprobs:
                                    alt_clean = alt.token.lower().strip()
                                    
                                    if alt_clean in ["yes", "y"] and not yes_found_in_main:
                                        if alt.logprob > yes_logprob:
                                            yes_logprob = alt.logprob
                                            yes_token = alt.token
                                    elif alt_clean in ["no", "n"] and not no_found_in_main:
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
            
            except (RateLimitError, APIError) as api_error:
                if attempt < max_retries - 1:
                    wait_time = (2 ** attempt) * 2
                    print(f"  API error (attempt {attempt + 1}/{max_retries}): {api_error}. Retrying in {wait_time}s...")
                    time.sleep(wait_time)
                    continue
                else:
                    raise
        
        except Exception as e:
            if attempt < max_retries - 1:
                wait_time = (2 ** attempt) * 2
                print(f"  Error (attempt {attempt + 1}/{max_retries}): {e}. Retrying in {wait_time}s...")
                time.sleep(wait_time)
                continue
            else:
                print(f"  Failed to get logprobs for theme '{theme}' after {max_retries} attempts: {e}")
                return {"yes": -10.0, "no": -10.0, "token_yes": None, "token_no": None}
    
    return {"yes": -10.0, "no": -10.0, "token_yes": None, "token_no": None}


def calculate_theme_probs(theme_logprobs: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
    """
    Calculate e^logprobs for each theme.
    
    Args:
        theme_logprobs: Dictionary with theme names as keys, each containing:
            - "yes": log probability for yes
            - "no": log probability for no
    
    Returns:
        Dictionary with theme names as keys, each containing:
            - "yes": probability (e^logprob_yes)
            - "no": probability (e^logprob_no)
    """
    theme_probs = {}
    
    for theme_name, logprob_data in theme_logprobs.items():
        if not isinstance(logprob_data, dict):
            continue
            
        yes_logprob = logprob_data.get("yes")
        no_logprob = logprob_data.get("no")
        
        # Calculate e^logprob for both yes and no
        yes_prob = math.exp(yes_logprob) if yes_logprob is not None else None
        no_prob = math.exp(no_logprob) if no_logprob is not None else None
        
        theme_probs[theme_name] = {
            "yes": yes_prob,
            "no": no_prob
        }
    
    return theme_probs


def get_ground_truth_themes(review: Dict[str, Any]) -> List[str]:
    """
    Extract ground truth themes from a review.
    Handles different formats: list, dict, or predicted_themes field.
    
    Args:
        review: Review dictionary
    
    Returns:
        List of theme names
    """
    themes = []
    
    # Check actual section (for prediction files)
    actual = review.get("actual", {})
    if isinstance(actual, dict):
        # Check for themes list
        if "themes" in actual and isinstance(actual["themes"], list):
            themes.extend(actual["themes"])
        # Check for predicted_themes (might be list or dict)
        elif "predicted_themes" in actual:
            pred_themes = actual["predicted_themes"]
            if isinstance(pred_themes, list):
                themes.extend(pred_themes)
            elif isinstance(pred_themes, dict):
                # If it's a dict, extract keys where value is truthy or > threshold
                themes.extend([k for k, v in pred_themes.items() if v])
    
    # Check top-level predicted_themes (for train_set format)
    if "predicted_themes" in review:
        pred_themes = review["predicted_themes"]
        if isinstance(pred_themes, list):
            themes.extend(pred_themes)
        elif isinstance(pred_themes, dict):
            # If it's a dict, extract keys where value is truthy
            themes.extend([k for k, v in pred_themes.items() if v])
    
    # Also check top-level themes
    if "themes" in review and isinstance(review["themes"], list):
        themes.extend(review["themes"])
    
    return list(set(themes))  # Remove duplicates


def process_review_for_ground_truth_logprobs(
    review: Dict[str, Any],
    client: OpenAI,
    rate_limiter: RateLimiter,
    category_themes: List[str],
    theme_model_name: str = "gpt-4o-mini",
    max_retries: int = 3
) -> bool:
    """
    Process a single review to get logprobs for ground truth themes.
    
    Args:
        review: Review dictionary
        client: OpenAI client
        rate_limiter: Rate limiter
        category_themes: List of all possible themes for this category (for validation)
        theme_model_name: Model to use
        max_retries: Max retries per theme
    
    Returns:
        True if theme_probs were added, False otherwise
    """
    # Get ground truth review text and product description
    # Handle both prediction format (with actual section) and train_set format (direct fields)
    actual = review.get("actual", {})
    if isinstance(actual, dict):
        review_text = actual.get("review_text", "")
        product_description = actual.get("product_description", "")
    else:
        review_text = review.get("review_text", "")
        product_description = review.get("product_description", "")
    
    if not review_text:
        return False
    
    if not product_description:
        # Try to get from other fields
        product_description = review.get("product_title", "") or review.get("title", "")
    
    # Get ground truth themes
    ground_truth_themes = get_ground_truth_themes(review)
    
    if not ground_truth_themes:
        return False
    
    # Get logprobs for each ground truth theme
    theme_logprobs = {}
    for theme in ground_truth_themes:
        # Validate theme is in category (optional check)
        if category_themes and theme not in category_themes:
            print(f"  Warning: Theme '{theme}' not in category themes, processing anyway...")
        
        logprob_result = get_theme_logprobs_for_ground_truth(
            product_description,
            review_text,
            theme,
            client,
            rate_limiter,
            theme_model_name,
            max_retries
        )
        theme_logprobs[theme] = logprob_result
    
    if not theme_logprobs:
        return False
    
    # Calculate e^logprobs
    theme_probs = calculate_theme_probs(theme_logprobs)
    
    # Add to review (add to actual section if it exists, otherwise add to top level)
    if "actual" in review and isinstance(review["actual"], dict):
        review["actual"]["theme_logprobs"] = theme_logprobs
        review["actual"]["theme_probs"] = theme_probs
    else:
        # Add to top level
        review["theme_logprobs"] = theme_logprobs
        review["theme_probs"] = theme_probs
    
    return True


def process_reviews_data(
    reviews_data: Dict[str, Any],
    client: OpenAI,
    rate_limiter: RateLimiter,
    category_themes_map: Dict[str, List[str]],
    theme_model_name: str = "gpt-4o-mini",
    max_retries: int = 3,
    max_reviews: Optional[int] = None
) -> Tuple[int, int]:
    """
    Process all reviews in the data structure to add ground truth theme logprobs.
    
    Args:
        reviews_data: Dictionary with user_id -> user_data structure
        client: OpenAI client
        rate_limiter: Rate limiter
        category_themes_map: Map of category -> list of themes
        theme_model_name: Model to use
        max_retries: Max retries per theme
        max_reviews: Optional limit on number of reviews to process
    
    Returns:
        Tuple of (processed_count, total_count)
    """
    processed_count = 0
    total_count = 0
    
    # Handle different data structures
    user_reviews_map = {}
    
    if isinstance(reviews_data, dict):
        # Check if it's {user_id: {reviews: [...]}} format
        for user_id, user_data in reviews_data.items():
            if isinstance(user_data, dict):
                reviews = user_data.get("reviews", [])
                if isinstance(reviews, list):
                    user_reviews_map[user_id] = reviews
                elif isinstance(user_data, list):
                    # Direct list format: {user_id: [...]}
                    user_reviews_map[user_id] = user_data
            elif isinstance(user_data, list):
                # Direct list format: {user_id: [...]}
                user_reviews_map[user_id] = user_data
    
    for user_id, reviews in user_reviews_map.items():
        if not isinstance(reviews, list):
            continue
        
        for review in reviews:
            if max_reviews and total_count >= max_reviews:
                break
            
            total_count += 1
            
            if not isinstance(review, dict):
                continue
            
            # Get category for this review
            category = review.get("category") or review.get("main_category")
            if not category:
                continue
            
            # Get all possible themes for this category (for validation)
            category_themes = category_themes_map.get(category, [])
            
            # Process review
            if process_review_for_ground_truth_logprobs(
                review,
                client,
                rate_limiter,
                category_themes,
                theme_model_name,
                max_retries
            ):
                processed_count += 1
            
            if total_count % 10 == 0:
                print(f"  Processed {total_count} reviews ({processed_count} with themes)...")
    
    return processed_count, total_count


def load_category_themes(topic_universe_path: Optional[Path] = None) -> Dict[str, List[str]]:
    """
    Load category -> themes mapping from topic universe.
    
    Args:
        topic_universe_path: Path to topic universe file
    
    Returns:
        Dictionary mapping category -> list of themes
    """
    if topic_universe_path is None:
        # Try to find topic universe from config or default location
        try:
            topic_universe_path = Path(settings.PATHS.get('THEMES_FILE', ''))
            if not topic_universe_path.exists():
                # Try default location
                project_root = Path(__file__).parent.parent.parent.parent
                topic_universe_path = project_root / "03_topic_universe" / "artifacts" / "topic_universe.json"
        except:
            project_root = Path(__file__).parent.parent.parent.parent
            topic_universe_path = project_root / "03_topic_universe" / "artifacts" / "topic_universe.json"
    
    if not topic_universe_path or not topic_universe_path.exists():
        raise FileNotFoundError(f"Topic universe file not found: {topic_universe_path}")
    
    with open(topic_universe_path, 'r', encoding='utf-8') as f:
        topic_universe = json.load(f)
    
    # Extract category -> themes mapping
    category_themes_map = {}
    if isinstance(topic_universe, dict):
        for category, themes_data in topic_universe.items():
            if isinstance(themes_data, list):
                category_themes_map[category] = themes_data
            elif isinstance(themes_data, dict) and "topics" in themes_data:
                category_themes_map[category] = themes_data["topics"]
            elif isinstance(themes_data, dict) and "themes" in themes_data:
                category_themes_map[category] = themes_data["themes"]
    
    return category_themes_map


def main():
    """Main function to process ground truth theme logprobs."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Add ground truth theme logprobs to reviews")
    parser.add_argument(
        "--artifact",
        type=str,
        default="train_set_reviews_with_merged_predictions_cleaned:latest",
        help="WandB artifact name (default: train_set_reviews_with_merged_predictions_cleaned:latest)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory (default: creates _with_ground_truth_logprobs directory)"
    )
    parser.add_argument(
        "--theme-model",
        type=str,
        default="gpt-4o-mini",
        help="Model to use for theme classification (default: gpt-4o-mini)"
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=3,
        help="Maximum retries per API call (default: 3)"
    )
    parser.add_argument(
        "--max-reviews",
        type=int,
        default=None,
        help="Maximum number of reviews to process (for testing)"
    )
    parser.add_argument(
        "--topic-universe",
        type=str,
        default=None,
        help="Path to topic_universe.json file"
    )
    parser.add_argument(
        "--wandb",
        action="store_true",
        help="Upload results to WandB"
    )
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("Adding Ground Truth Theme Logprobs")
    print("=" * 80)
    
    # Initialize OpenAI client
    try:
        api_key = None
        if HAS_SETTINGS:
            api_key = getattr(settings, 'OPENAI_API_KEY', None)
        if not api_key:
            api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            print("Error: OPENAI_API_KEY not found in config or environment")
            return
        client = OpenAI(api_key=api_key)
    except Exception as e:
        print(f"Error initializing OpenAI client: {e}")
        return
    
    rate_limiter = RateLimiter(min_interval=0.1)
    
    # Load category themes mapping
    print("\nLoading topic universe...")
    try:
        topic_universe_path = Path(args.topic_universe) if args.topic_universe else None
        category_themes_map = load_category_themes(topic_universe_path)
        print(f"  Loaded themes for {len(category_themes_map)} categories")
    except Exception as e:
        print(f"Error loading topic universe: {e}")
        return
    
    # Download artifact from WandB
    print(f"\nDownloading artifact from WandB: {args.artifact}")
    try:
        if HAS_WANDB:
            artifact_path = download_artifact(
                artifact_name=args.artifact.split(":")[0],
                artifact_type="dataset",
                version=args.artifact.split(":")[1] if ":" in args.artifact else "latest"
            )
        else:
            print("Error: WandB utilities not available")
            return
        
        if not artifact_path:
            print("Error: Failed to download artifact")
            return
        
        artifact_path = Path(artifact_path)
        print(f"  Artifact downloaded to: {artifact_path}")
        
        # Find the JSON file
        json_file = None
        if artifact_path.is_file() and artifact_path.suffix == ".json":
            json_file = artifact_path
        else:
            # Try common filenames
            for filename in [
                "train_set_reviews_with_merged_predictions_cleaned.json",
                "cleaned_merged_reviews_by_user.json",
                "reviews.json",
                "data.json"
            ]:
                candidate = artifact_path / filename
                if candidate.exists():
                    json_file = candidate
                    break
        
        if not json_file or not json_file.exists():
            print(f"Error: Could not find JSON file in artifact")
            return
        
        print(f"  Found JSON file: {json_file}")
        
    except Exception as e:
        print(f"Error downloading artifact: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Load review data
    print(f"\nLoading review data from {json_file}...")
    try:
        with open(json_file, 'r', encoding='utf-8') as f:
            reviews_data = json.load(f)
        print(f"  Loaded data with {len(reviews_data)} users")
    except Exception as e:
        print(f"Error loading JSON file: {e}")
        return
    
    # Process reviews
    print(f"\nProcessing reviews to add ground truth theme logprobs...")
    print(f"  Using model: {args.theme_model}")
    print(f"  Max retries: {args.max_retries}")
    if args.max_reviews:
        print(f"  Limiting to {args.max_reviews} reviews")
    
    processed_count, total_count = process_reviews_data(
        reviews_data,
        client,
        rate_limiter,
        category_themes_map,
        args.theme_model,
        args.max_retries,
        args.max_reviews
    )
    
    print(f"\n{'=' * 80}")
    print(f"Processing complete!")
    print(f"  Total reviews: {total_count}")
    print(f"  Processed with themes: {processed_count}")
    print(f"{'=' * 80}")
    
    # Determine output path
    if args.output_dir:
        output_path = Path(args.output_dir)
    else:
        output_path = json_file.parent / f"{json_file.stem}_with_ground_truth_logprobs.json"
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save processed data
    print(f"\nSaving processed data to: {output_path}")
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(reviews_data, f, indent=4, ensure_ascii=False)
        print(f"  ✓ Successfully saved to {output_path}")
    except Exception as e:
        print(f"  ✗ Error saving file: {e}")
        return
    
    # Upload to WandB if requested
    if args.wandb and HAS_WANDB:
        print(f"\nUploading to WandB...")
        try:
            run = init_wandb_run(
                run_name="add_ground_truth_theme_logprobs",
                stage="07_sgo_training",
                config={
                    "description": "Adding ground truth theme logprobs to train set",
                    "input_artifact": args.artifact,
                    "theme_model": args.theme_model,
                    "processed_reviews": processed_count,
                    "total_reviews": total_count
                }
            )
            
            if run:
                artifact_name = f"{args.artifact.split(':')[0]}_with_ground_truth_logprobs"
                artifact = log_artifact(
                    run=run,
                    artifact_name=artifact_name,
                    artifact_type="dataset",
                    artifact_path=output_path,
                    metadata={
                        "processed_reviews": processed_count,
                        "total_reviews": total_count,
                        "theme_model": args.theme_model
                    }
                )
                print(f"  ✓ Uploaded {artifact_name} to WandB")
                finish_run(run)
        except Exception as e:
            print(f"  ✗ Error uploading to WandB: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()

