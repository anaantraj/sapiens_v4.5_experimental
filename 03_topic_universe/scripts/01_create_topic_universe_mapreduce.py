#!/usr/bin/env python3
"""
Stage 03: Topic Universe - Map Reduce
=====================================

Uses LLM map-reduce to discover topics from reviews.
- Reads configuration from config.yaml
- Downloads training data artifact from W&B
- Uses LLM to iteratively discover topics
- Logs artifact to W&B: Topic Universe collection

Usage:
    python 03_topic_universe/scripts/01_create_topic_universe_mapreduce.py
"""

import json
import os
import re
import sys
import random
import asyncio
import tiktoken
from pathlib import Path

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Load .env so OPENAI_API_KEY and OPENAI_BASE_URL (Bedrock) are set
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent.parent.parent / ".env", override=True)
except ImportError:
    pass

from utils.wandb_utils import (
    load_config, get_stage_config, get_openai_config,
    init_wandb_run, finish_run, use_artifact, log_artifact,
    log_metrics, log_summary, link_to_registry, get_artifact_dir,
    validate_stage_dependencies, create_comprehensive_artifact_metadata,
    get_learned_artifact_schema
)
from utils.openai_client import create_async_openai_client

# Import schema for validation
from schemas.learned_artifacts import TopicUniverseArtifact

# Use the appropriate tokenizer for the model
encoding = tiktoken.get_encoding("cl100k_base")

# Prompt directory
PROMPT_DIR = Path(__file__).parent.parent / "prompts"


def load_prompt(prompt_filename: str) -> str:
    """Load a prompt template from file."""
    prompt_path = PROMPT_DIR / prompt_filename
    if prompt_path.exists():
        with open(prompt_path, 'r', encoding='utf-8') as f:
            return f.read().strip()
    else:
        raise FileNotFoundError(f"Prompt file not found: {prompt_path}")


def map_category_to_main_category(category_name: str) -> str:
    """
    Map a category name to one of the 7 main categories.
    This matches the mapping used in Stage 01.
    
    Args:
        category_name: Category name from review data
        
    Returns:
        Main category name or None if no match
    """
    if not category_name:
        return None
    
    category_lower = category_name.lower()
    
    # Mapping rules (matching Stage 01)
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


def read_reviews_from_artifact(artifact_path: Path, category_name: str, input_json_filename: str):
    """
    Reads review texts from the downloaded artifact.
    Extracts reviews for the specified category.
    Uses category mapping to match main categories to actual category names in data.
    
    Args:
        artifact_path: Path to downloaded artifact directory
        category_name: Category to filter reviews
        input_json_filename: Exact JSON filename expected in artifact (no fallback)
    """
    if not input_json_filename:
        raise ValueError("input_json_filename must be provided")
    
    # Resolve path to handle any symlinks or relative paths
    artifact_path = Path(artifact_path).resolve()
    
    # Use exact filename (no fallback)
    review_file = artifact_path / input_json_filename
    
    if not review_file.exists():
        error_msg = f"Review file not found in artifact: {review_file}\n"
        error_msg += f"Expected exact filename: {input_json_filename}\n"
        error_msg += f"Artifact path (resolved): {artifact_path}\n"
        error_msg += f"[DEBUG] Listing files in artifact directory:\n"
        if artifact_path.exists():
            for item in artifact_path.iterdir():
                error_msg += f"  - {item.name} ({'file' if item.is_file() else 'dir'})\n"
        else:
            error_msg += f"  [ERROR] Artifact directory does not exist!\n"
        print(f"[ERROR] {error_msg}")
        raise FileNotFoundError(f"Required input file '{input_json_filename}' not found in W&B artifact. {error_msg}")
    
    print(f"Loading reviews from '{review_file}' for category '{category_name}'...")
    
    try:
        with open(review_file, 'r', encoding='utf-8') as f:
            all_user_data = json.load(f)
        
        all_review_texts = []
        for user_id, user_data in all_user_data.items():
            for review in user_data.get('reviews', []):
                # If category specified, filter by it using category mapping
                if category_name and category_name != "all":
                    review_category = review.get('category', '') or ''
                    # Map the review's category to main category
                    review_main_category = map_category_to_main_category(review_category)
                    # Check if it matches the requested main category (exact match)
                    if not review_main_category or review_main_category != category_name:
                        continue
                
                review_text = review.get('review_text', '')
                if review_text and review_text.strip():
                    all_review_texts.append(review_text.strip())
                    
        print(f"Loaded {len(all_review_texts)} reviews for category '{category_name}'.")
        return all_review_texts
                        
    except json.JSONDecodeError as e:
        error_msg = f"Failed to decode JSON file '{review_file}': {e}"
        print(f"[ERROR] {error_msg}")
        raise ValueError(error_msg) from e
    except Exception as e:
        error_msg = f"Failed to read file '{review_file}': {e}"
        print(f"[ERROR] {error_msg}")
        raise RuntimeError(error_msg) from e


async def discover_themes_with_llm(review_batch, previous_themes, category, round_num,
                                    client, model, max_tokens, prompt_config):
    """
    Uses an LLM (via AWS Bedrock client) to discover themes from a batch of reviews.
    """
    if not review_batch:
        return []
    
    full_review_text = "\n".join(review_batch)
    
    # Load prompt templates (system prompt merged into main prompt)
    if round_num == 1:
        prompt_template = load_prompt("topic_discovery_round1_prompt.txt")
    else:
        previous_themes_str = ", ".join(f"'{theme}'" for theme in previous_themes)
        prompt_template = load_prompt("topic_discovery_subsequent_rounds_prompt.txt")
    
    # Calculate tokens and truncate reviews if needed
    # Estimate tokens for prompt template (without reviews)
    if round_num == 1:
        template_without_reviews = prompt_template.replace("{reviews_text}", "")
        # Format template for token estimation
        template_without_reviews = template_without_reviews.format(
            category=category,
            initial_themes_count=prompt_config.get("initial_themes_count", 10)
        )
    else:
        template_without_reviews = prompt_template.replace("{reviews_text}", "").replace("{previous_themes}", previous_themes_str)
        # Format template for token estimation
        template_without_reviews = template_without_reviews.format(
            category=category,
            previous_themes=previous_themes_str,
            subsequent_rounds_max_themes=prompt_config.get("subsequent_rounds_max_themes", 5)
        )
    
    template_tokens = len(encoding.encode(template_without_reviews))
    review_tokens = len(encoding.encode(full_review_text))
    
    # Reserve tokens for response
    reserved_tokens = template_tokens + 1000
    available_tokens = max_tokens - reserved_tokens
    
    # Truncate reviews if needed
    if review_tokens > available_tokens:
        print(f"   [WARNING] Review text ({review_tokens} tokens) exceeds available space ({available_tokens} tokens). Truncating...")
        encoded_reviews = encoding.encode(full_review_text)
        truncated_encoded = encoded_reviews[:available_tokens]
        full_review_text = encoding.decode(truncated_encoded)
        review_tokens = len(truncated_encoded)
    
    # Format prompt with (possibly truncated) reviews
    if round_num == 1:
        prompt_text = prompt_template.format(
            category=category,
            initial_themes_count=prompt_config.get("initial_themes_count", 10),
            reviews_text=full_review_text
        )
    else:
        prompt_text = prompt_template.format(
            category=category,
            previous_themes=previous_themes_str,
            subsequent_rounds_max_themes=prompt_config.get("subsequent_rounds_max_themes", 5),
            reviews_text=full_review_text
        )
    
    prompt_tokens = len(encoding.encode(prompt_text))
    payload_tokens = prompt_tokens
    print(f"   Total tokens for this LLM call: {payload_tokens}")

    try:
        print(f"   Calling LLM (Bedrock) to discover themes...")
        response = await client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt_text}],
            response_format={"type": "json_object"},
            max_tokens=min(4096, max(256, max_tokens - payload_tokens)),
        )
        
        # Validate API response structure
        if not response.choices:
            error_msg = "LLM API response missing 'choices' field"
            print(f"   [ERROR] {error_msg}")
            raise ValueError(error_msg)
        
        choice = response.choices[0]
        if not choice.message:
            error_msg = "LLM API response missing 'message' field in choice"
            print(f"   [ERROR] {error_msg}")
            raise ValueError(error_msg)
        
        response_content = (choice.message.content or "{}").strip()
        if not response_content:
            error_msg = "LLM API response has empty content"
            print(f"   [ERROR] {error_msg}")
            raise ValueError(error_msg)
        
        # Fix common LLM JSON glitches from Bedrock (extra wrapper/whitespace)
        # e.g. {"{"themes": ...}  or  {\n {"themes": ...
        if response_content.startswith('{"{'):
            response_content = "{" + response_content[3:]
        # Remove leading { plus optional whitespace when followed by {"themes"
        response_content = re.sub(r'^\s*\{\s*(?=\{\s*"themes")', '', response_content)
        
        try:
            response_json = json.loads(response_content)
        except json.JSONDecodeError as e:
            # Fallback: try to extract "themes" array from truncated or malformed JSON
            themes_match = re.search(r'"themes"\s*:\s*\[(.*)\]', response_content, re.DOTALL)
            if themes_match:
                array_str = "[" + themes_match.group(1) + "]"
                try:
                    new_themes = json.loads(array_str)
                    if isinstance(new_themes, list):
                        return new_themes
                except json.JSONDecodeError:
                    pass
            error_msg = f"LLM did not return valid JSON: {e}\nResponse content: {response_content[:200]}"
            print(f"   [ERROR] {error_msg}")
            raise ValueError(error_msg) from e
        
        if "themes" not in response_json:
            error_msg = "LLM response missing 'themes' field"
            print(f"   [ERROR] {error_msg}")
            raise ValueError(error_msg)
        
        new_themes = response_json.get('themes', [])
        
        if not isinstance(new_themes, list):
            error_msg = f"LLM returned invalid themes format (expected list, got {type(new_themes)})"
            print(f"   [ERROR] {error_msg}")
            raise ValueError(error_msg)

        return new_themes

    except Exception as e:
        if "ClientError" in type(e).__name__ or "APIError" in type(e).__name__:
            error_msg = f"LLM (Bedrock) API call failed: {e}"
        else:
            error_msg = f"Error processing LLM response: {e}"
        print(f"   [ERROR] {error_msg}")
        raise RuntimeError(error_msg) from e


def save_themes(themes: list, output_dir: Path, category: str):
    """
    Saves the themes to the artifact directory (legacy single-file format).
    Kept for backward compatibility.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    filename = output_dir / f"final_themes_{category.replace(' ', '_')}.json"
    try:
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(themes, f, indent=4)
        print(f"\n[OK] Saved {len(themes)} themes to '{filename}'")
        return True
    except IOError as e:
        print(f"[ERROR] Could not write to file: {e}")
        return False


def save_and_validate_topic_universe(
    themes: list, 
    output_dir: Path, 
    category: str,
    metadata: dict = None
) -> TopicUniverseArtifact:
    """
    Save themes using merged structure with schema validation.
    
    Args:
        themes: List of topic strings for the category
        output_dir: Directory to save the artifact
        category: Category name
        metadata: Optional metadata (method, model, etc.)
        
    Returns:
        Validated TopicUniverseArtifact instance
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    merged_file = output_dir / "topic_universe.json"
    
    # Load existing merged file if it exists, otherwise create new
    if merged_file.exists():
        try:
            existing = TopicUniverseArtifact.from_file(merged_file)
            print(f"[INFO] Loaded existing topic universe with {len(existing.topics_by_category)} categories")
            # Add/update this category
            existing.topics_by_category[category] = themes
            validated = existing
        except Exception as e:
            print(f"[WARNING] Could not load existing file, creating new: {e}")
            validated = TopicUniverseArtifact(
                topics_by_category={category: themes},
                metadata=metadata
            )
    else:
        # Create new artifact
        validated = TopicUniverseArtifact(
            topics_by_category={category: themes},
            metadata=metadata
        )
    
    # Validate (Pydantic will raise ValidationError if invalid)
    # The validation happens automatically when creating the model
    
    # Save merged file
    validated.to_merged_file(merged_file)
    print(f"[OK] Saved merged topic universe to '{merged_file}'")
    print(f"     Total categories: {len(validated.topics_by_category)}")
    print(f"     Topics for '{category}': {len(themes)}")
    
    return validated


async def process_single_category(
    category_name: str,
    input_path: Path,
    client,
    model: str,
    max_tokens: int,
    sample_size: int,
    max_rounds: int,
    input_artifact: str,
    input_json_filename: str,
    prompt_config: dict
) -> tuple:
    """
    Process a single category to discover topics.
    This function is designed to run in parallel with other categories.
    
    Args:
        category_name: Category to process
        input_path: Path to input artifact
        client: Async OpenAI-compatible client (Bedrock when OPENAI_BASE_URL is set)
        model: Model name
        max_tokens: Max tokens
        sample_size: Sample size per round
        max_rounds: Max rounds
        input_artifact: Input artifact name
        
    Returns:
        Tuple of (category_name, themes_list, rounds_completed, num_reviews)
    """
    print(f"\n{'='*70}")
    print(f"PROCESSING CATEGORY: {category_name}")
    print(f"{'='*70}")
    
    # Read reviews for this category (independent data)
    try:
        all_reviews = read_reviews_from_artifact(input_path, category_name, input_json_filename)
    except (FileNotFoundError, ValueError, RuntimeError) as e:
        # File not found or data loading errors - fail the category
        print(f"[ERROR] Failed to load reviews for category '{category_name}': {e}")
        raise  # Re-raise to fail the category processing
    
    if not all_reviews:
        print(f"[WARNING] No reviews found for category '{category_name}' (file loaded but empty or no matching reviews)")
        return (category_name, [], 0, 0)
    
    print(f"[OK] Loaded {len(all_reviews)} reviews for {category_name}")
    
    # Iterative theme discovery (independent processing)
    all_discovered_themes = []
    reviews_to_sample = list(all_reviews)
    rounds_completed = 0
    
    for round_num in range(1, max_rounds + 1):
        print(f"\n[{category_name}] Round {round_num}/{max_rounds}")
        
        if not reviews_to_sample:
            print(f"[{category_name}] No reviews left to sample.")
            break
        
        k = min(sample_size, len(reviews_to_sample))
        print(f"[{category_name}] Sampling {k} reviews from {len(reviews_to_sample)} remaining...")
        
        review_batch = random.sample(reviews_to_sample, k)
        
        # Remove sampled reviews
        batch_set = set(review_batch)
        reviews_to_sample = [r for r in reviews_to_sample if r not in batch_set]
        
        try:
            new_themes = await discover_themes_with_llm(
                review_batch, all_discovered_themes, category_name, round_num,
                client, model, max_tokens, prompt_config
            )
        except (ValueError, RuntimeError) as e:
            # LLM API errors - fail the category processing
            print(f"[{category_name}] [ERROR] LLM discovery failed: {e}")
            raise  # Re-raise to fail the category
        
        if not new_themes:
            print(f"[{category_name}] No new themes discovered in this round.")
            if not reviews_to_sample:
                break
            continue
        
        truly_new_themes = [theme for theme in new_themes if theme not in all_discovered_themes]
        
        if not truly_new_themes and round_num > 1:
            print(f"[{category_name}] Only previously discovered themes found. Ending process.")
            break
        
        print(f"[{category_name}] Discovered {len(truly_new_themes)} new themes: {truly_new_themes}")
        all_discovered_themes.extend(truly_new_themes)
        rounds_completed = round_num
    
    final_unique_themes = sorted(list(set(all_discovered_themes)))
    print(f"\n[{category_name}] COMPLETE: {len(final_unique_themes)} themes discovered")
    
    return (category_name, final_unique_themes, rounds_completed, len(all_reviews))


async def main():
    """Main execution function - processes all categories in parallel."""
    
    # =========================================================================
    # Step 1: Load configuration from config.yaml
    # =========================================================================
    print("=" * 70)
    print("STAGE 03: Topic Universe - Map Reduce (Parallel Processing)")
    print("=" * 70)
    
    cfg = get_stage_config("03_topic_universe")
    openai_cfg = get_openai_config()
    
    # Get stage directory from config
    stage_directory = cfg.get("stage_directory")
    if not stage_directory:
        raise ValueError("stage_directory must be specified in config.yaml")
    
    hyperparams = cfg.get("hyperparameters")
    if not hyperparams:
        raise ValueError("hyperparameters must be specified in config.yaml")
    
    # Get parameters from config (no defaults - all must be in config)
    input_artifact = cfg.get("input_artifact")
    if not input_artifact:
        raise ValueError("input_artifact must be specified in config.yaml")
    
    output_artifact_name = cfg.get("output_artifact")
    if not output_artifact_name:
        raise ValueError("output_artifact must be specified in config.yaml")
    
    # Get categories from config (required)
    if "categories" in hyperparams:
        categories = hyperparams.get("categories")
        if not categories or not isinstance(categories, list) or len(categories) == 0:
            raise ValueError("hyperparameters.categories must be specified as a non-empty list in config.yaml")
    elif "category" in hyperparams:
        category_config = hyperparams.get("category")
        if not category_config:
            raise ValueError("hyperparameters.category must be specified in config.yaml")
        if category_config == "all":
            raise ValueError("hyperparameters.category='all' is not supported. Please specify categories as a list in hyperparameters.categories")
        categories = [category_config]
    else:
        raise ValueError("Either hyperparameters.categories (list) or hyperparameters.category (string) must be specified in config.yaml")
    
    use_parallel = hyperparams.get("parallel")
    if use_parallel is None:
        raise ValueError("hyperparameters.parallel must be specified in config.yaml")
    
    sample_size = hyperparams.get("sample_size_per_round")
    if sample_size is None:
        raise ValueError("hyperparameters.sample_size_per_round must be specified in config.yaml")
    
    max_rounds = hyperparams.get("max_rounds")
    if max_rounds is None:
        raise ValueError("hyperparameters.max_rounds must be specified in config.yaml")
    
    model = hyperparams.get("model")
    if not model:
        raise ValueError("hyperparameters.model must be specified in config.yaml")
    # When using Bedrock, use config bedrock_model_id if set; otherwise use model
    base_url = os.environ.get("OPENAI_BASE_URL", "").strip()
    if base_url and "bedrock-mantle" in base_url and cfg.get("bedrock_model_id"):
        model = cfg.get("bedrock_model_id")
    max_tokens = hyperparams.get("max_tokens")
    if max_tokens is None:
        raise ValueError("hyperparameters.max_tokens must be specified in config.yaml")
    
    # Get prompt configuration (with defaults)
    prompt_config = hyperparams.get("prompt_config", {})
    # Set defaults if not provided
    if "initial_themes_count" not in prompt_config:
        prompt_config["initial_themes_count"] = 10
    if "subsequent_rounds_max_themes" not in prompt_config:
        prompt_config["subsequent_rounds_max_themes"] = 5
    
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
    
    input_json_filename = paths_config.get("input_json_filename")
    if not input_json_filename:
        raise ValueError("paths.input_json_filename must be specified in config.yaml")
    
    # Create Bedrock (or configured endpoint) client; uses OPENAI_API_KEY + OPENAI_BASE_URL from env
    if not os.environ.get("OPENAI_API_KEY"):
        print("[ERROR] OPENAI_API_KEY environment variable not set (set in .env or export)")
        return
    llm_client = create_async_openai_client()
    
    print(f"\n[Config] Input artifact: {input_artifact}")
    print(f"[Config] Output artifact: {output_artifact_name}")
    print(f"[Config] Categories: {categories}")
    print(f"[Config] Parallel processing: {use_parallel}")
    print(f"[Config] Model: {model}")
    print(f"[Config] Max tokens: {max_tokens}")
    print(f"[Config] Sample size per round: {sample_size}")
    print(f"[Config] Max rounds: {max_rounds}")
    
    # =========================================================================
    # Step 2: Initialize W&B run
    # =========================================================================
    print("\n" + "-" * 70)
    print("Step 2: Initialize W&B Run")
    print("-" * 70)
    
    run = init_wandb_run(
        run_name=f"topic_universe_all_categories",
        stage=stage_directory,
        job_type=job_type
    )
    
    try:
        # =====================================================================
        # Step 3: Validate stage dependencies (sequential execution)
        # =====================================================================
        print("\n" + "-" * 70)
        print("Step 3: Validating Stage Dependencies")
        print("-" * 70)
        
        required_artifacts = [input_artifact] if input_artifact else []
        
        if not validate_stage_dependencies(run, stage_directory, required_artifacts):
            print("[ERROR] Stage 02 must be completed first!")
            print("[ERROR] Please run Stage 02 to create required artifacts.")
            return
        
        # =====================================================================
        # Step 4: Download input artifact (prioritize local)
        # =====================================================================
        print("\n" + "-" * 70)
        print("Step 4: Get Input Artifact")
        print("-" * 70)
        
        # =====================================================================
        # Step 3.5: Check if topic universe already exists
        # =====================================================================
        print("\n" + "-" * 70)
        print("Step 3.5: Check if Topic Universe Already Exists")
        print("-" * 70)
        
        # Initialize variables for summary (used whether discovery runs or not)
        all_category_results = {}
        category_stats = {}
        total_themes = 0
        wandb_success = True
        artifact_uploaded = False
        
        # Check if output artifact already exists (for SKIPPING discovery, not as input fallback)
        # NOTE: This is only for optimization - if output already exists, skip discovery
        # Input artifacts are ALWAYS from W&B (no local fallback)
        output_artifact_base = output_artifact_name.split(":")[0]
        output_dir = get_artifact_dir(stage_directory, output_artifact_base)
        existing_topic_universe_file = output_dir / "topic_universe.json"
        
        topic_universe_exists = False
        if existing_topic_universe_file.exists():
            print(f"[INFO] Topic universe already exists locally at: {existing_topic_universe_file}")
            print(f"[INFO] → Skipping topic discovery (optimization)")
            print(f"[INFO] → Will proceed directly to shrink step...")
            print(f"[NOTE] Input artifacts for shrink step will still be from W&B (no local fallback)")
            topic_universe_exists = True
            # Load existing data for summary
            try:
                existing_artifact = TopicUniverseArtifact.from_file(existing_topic_universe_file)
                all_category_results = existing_artifact.topics_by_category
                total_themes = sum(len(themes) for themes in all_category_results.values())
                for category, themes in all_category_results.items():
                    category_stats[category] = {
                        "num_themes": len(themes),
                        "rounds_completed": 0,  # Unknown if skipped
                        "num_reviews": 0  # Unknown if skipped
                    }
            except Exception as e:
                print(f"[WARNING] Could not load existing topic universe for summary: {e}")
        else:
            # Also check W&B (only specific exceptions are caught)
            try:
                import wandb
                api = wandb.Api()
                # Try to find the artifact (may not exist yet)
                try:
                    artifact_path = f"{run.entity}/{run.project}/{output_artifact_name}"
                    artifact = api.artifact(artifact_path)
                    if artifact:
                        print(f"[INFO] Topic universe exists in W&B: {output_artifact_name}")
                        print(f"[INFO] → Skipping topic discovery (optimization)")
                        print(f"[INFO] → Will proceed directly to shrink step...")
                        print(f"[NOTE] Input artifacts for shrink step will still be from W&B (no local fallback)")
                        topic_universe_exists = True
                except wandb.errors.CommError:
                    # Artifact doesn't exist in W&B (expected), proceed with discovery
                    pass
                except wandb.errors.APIError as e:
                    # W&B API error - log but don't fail (network/auth issues)
                    print(f"[WARNING] Could not check W&B for existing artifact: {e}")
                    print(f"[INFO] Proceeding with topic discovery...")
                except ImportError as e:
                    # wandb not available - log but don't fail
                    print(f"[WARNING] Could not import wandb to check existing artifact: {e}")
                    print(f"[INFO] Proceeding with topic discovery...")
            except Exception as e:
                # Unexpected error - log but proceed (don't fail the whole run)
                print(f"[WARNING] Unexpected error checking W&B for existing artifact: {e}")
                print(f"[INFO] Proceeding with topic discovery...")
        
        if not topic_universe_exists:
            # =====================================================================
            # Step 4: Download input artifact from W&B (ONLY - no local fallback)
            # =====================================================================
            print("\n" + "-" * 70)
            print("Step 4: Download Input Artifact from W&B")
            print("-" * 70)
            print(f"[REQUIRED] Input artifact: {input_artifact}")
            print(f"[REQUIRED] Artifact type: {artifact_type}")
            print(f"[INFO] Downloading from W&B (NO local fallback)...")
            
            # Download input artifact from W&B (ONLY - no local fallback)
            input_path = use_artifact(run, input_artifact, artifact_type=artifact_type)
            
            if input_path is None:
                print(f"[ERROR] ✗ Could not download input artifact from W&B: {input_artifact}")
                print(f"[ERROR]   Make sure artifact '{input_artifact}' exists in W&B")
                print(f"[ERROR]   Make sure Stage 02 has been completed and artifact uploaded")
                print(f"[ERROR]   No local fallback available - W&B is the only source")
                return
            
            # Resolve the path to handle any symlinks or relative paths
            input_path = Path(input_path).resolve()
            print(f"[OK] ✓ Input artifact downloaded to: {input_path}")
            
            # Debug: List files in the downloaded artifact
            if input_path.exists():
                print(f"[DEBUG] Files in artifact directory:")
                for item in input_path.iterdir():
                    print(f"  - {item.name} ({'file' if item.is_file() else 'dir'})")
            else:
                print(f"[ERROR] ✗ Artifact directory does not exist: {input_path}")
                return
            
            # =====================================================================
            # Step 5: Process all categories (parallel or sequential)
            # =====================================================================
            print("\n" + "-" * 70)
            print(f"Step 5: Process {len(categories)} Categories ({'PARALLEL' if use_parallel else 'SEQUENTIAL'})")
            print("-" * 70)
            
            # Create tasks for all categories
            tasks = [
                process_single_category(
                    category_name=cat,
                    input_path=input_path,
                    client=llm_client,
                    model=model,
                    max_tokens=max_tokens,
                    sample_size=sample_size,
                    max_rounds=max_rounds,
                    input_artifact=input_artifact,
                    input_json_filename=input_json_filename,
                    prompt_config=prompt_config
                )
                for cat in categories
            ]
            
            # Execute in parallel or sequential
            if use_parallel:
                print(f"[INFO] Processing {len(categories)} categories in PARALLEL...")
                results = await asyncio.gather(*tasks)
            else:
                print(f"[INFO] Processing {len(categories)} categories SEQUENTIALLY...")
                results = []
                for task in tasks:
                    result = await task
                    results.append(result)
            
            # =====================================================================
            # Step 6: Merge results and save locally (with schema validation)
            # =====================================================================
            print("\n" + "-" * 70)
            print("Step 6: Merge Results and Save Locally (with Schema Validation)")
            print("-" * 70)
            print(f"[INFO] Saving output locally FIRST (before W&B upload)...")
            
            # Collect all results
            all_category_results = {}
            category_stats = {}
            
            for category_name, themes, rounds_completed, num_reviews in results:
                if themes:  # Only add if themes were found
                    all_category_results[category_name] = themes
                    category_stats[category_name] = {
                        "num_themes": len(themes),
                        "rounds_completed": rounds_completed,
                        "num_reviews": num_reviews
                    }
                    print(f"  {category_name}: {len(themes)} themes")
            
            if not all_category_results:
                print("[ERROR] No themes discovered for any category")
                return
            
            # Save with schema validation (LOCAL FIRST - always saved)
            output_dir = get_artifact_dir(stage_directory, output_artifact_name)
            output_dir.mkdir(parents=True, exist_ok=True)
            print(f"[INFO] Output directory: {output_dir}")
            
            metadata = {
                "method": "map_reduce_llm",
                "model": model,
                "input_artifact": input_artifact,
                "categories_processed": list(all_category_results.keys()),
            }
            
            try:
                # Create validated artifact
                print(f"[INFO] Creating validated artifact with schema...")
                validated_artifact = TopicUniverseArtifact(
                    topics_by_category=all_category_results,
                    metadata=metadata
                )
                
                # Save merged file (LOCAL)
                merged_file = output_dir / "topic_universe.json"
                validated_artifact.to_merged_file(merged_file)
                print(f"[OK] ✓ Saved merged topic universe locally: {merged_file}")
                print(f"     Total categories: {len(validated_artifact.topics_by_category)}")
                
                # Also save legacy single-file format for backward compatibility
                for category_name, themes in all_category_results.items():
                    save_themes(themes, output_dir, category_name)
                
                print(f"[OK] ✓ All local files saved successfully")
                print(f"[INFO] → Proceeding to W&B upload in next step...")
                
            except Exception as e:
                print(f"[ERROR] ✗ Schema validation or local save failed: {e}")
                raise
            
            # =====================================================================
            # Step 7: Log metrics and summary (with error handling)
            # =====================================================================
            print("\n" + "-" * 70)
            print("Step 7: Log Metrics")
            print("-" * 70)
            
            total_themes = sum(len(themes) for themes in all_category_results.values())
            
            wandb_success = True
            try:
                log_summary(run, {
                    "categories_processed": len(all_category_results),
                    "total_themes": total_themes,
                    "avg_themes_per_category": total_themes / len(all_category_results) if all_category_results else 0,
                })
                
                # Log per-category metrics
                for category, stats in category_stats.items():
                    log_metrics(run, {
                        f"category/{category}/themes": stats["num_themes"],
                        f"category/{category}/rounds": stats["rounds_completed"],
                        f"category/{category}/reviews": stats["num_reviews"],
                    })
                print("[OK] Metrics logged to W&B")
            except Exception as e:
                print(f"[WARNING] Failed to log metrics to W&B: {e}")
                print("[INFO] Metrics logging failed, but local files are saved")
                wandb_success = False
            
            # =====================================================================
            # Step 8: Upload artifact to W&B (with error handling)
            # =====================================================================
            print("\n" + "-" * 70)
            print("Step 8: Upload Artifact to W&B")
            print("-" * 70)
            print(f"[INFO] ✓ Local files already saved at: {output_dir}")
            print(f"[INFO] → Now uploading to W&B: {output_artifact_name}")
            print(f"[INFO]   Artifact type: {artifact_type}")
            
            artifact_uploaded = False
            try:
                print(f"[INFO] Creating comprehensive artifact metadata...")
                artifact_metadata = {
                    "categories": list(all_category_results.keys()),
                    "total_categories": len(all_category_results),
                    "total_themes": total_themes,
                    "category_stats": category_stats,
                    "method": "map_reduce_llm",
                    "model": model,
                    "input_artifact": input_artifact,
                    "parallel_processing": use_parallel,
                    "schema_validated": True,
                    "schema_version": "v4"
                }
                
                print(f"[INFO] Uploading artifact to W&B (this may take a moment)...")
                artifact = log_artifact(
                    run=run,
                    artifact_name=output_artifact_name,
                    artifact_type=artifact_type,
                    artifact_path=output_dir,
                    metadata=create_comprehensive_artifact_metadata(
                        stage=stage_directory,
                        artifact_name=output_artifact_name,
                        sample_size=total_themes,
                        model_vendor="Bedrock",
                        model_name=model,
                        model_description="LLM for topic discovery via map-reduce",
                        model_params={
                            "max_tokens": max_tokens,
                            "sample_size_per_round": sample_size,
                            "max_rounds": max_rounds,
                        },
                        learned_artifact_schema=get_learned_artifact_schema(stage_directory, output_artifact_name),
                        additional_metadata={
                            "categories_processed": list(all_category_results.keys()),
                            "total_categories": len(all_category_results),
                            "total_themes": total_themes,
                            "category_stats": category_stats,
                            "parallel_processing": use_parallel,
                            "input_artifact": input_artifact,
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
        # Step 9: Run shrink/consolidation step automatically (always runs)
        # =====================================================================
        print("\n" + "-" * 70)
        print("Step 9: Shrink/Consolidate Topics")
        print("-" * 70)
        
        shrink_input_artifact = cfg.get("shrink_input_artifact")
        shrink_output_artifact = cfg.get("shrink_output_artifact")
        
        if not shrink_input_artifact:
            raise ValueError("shrink_input_artifact must be specified in config.yaml")
        if not shrink_output_artifact:
            raise ValueError("shrink_output_artifact must be specified in config.yaml")
        
        print(f"[INFO] Running topic shrinking step...")
        print(f"  Input: {shrink_input_artifact}")
        print(f"  Output: {shrink_output_artifact}")
        
        # Import and run shrink script
        import subprocess
        shrink_script = Path(__file__).parent / "02_shrink_topics.py"
        
        if shrink_script.exists():
            try:
                result = subprocess.run(
                    [
                        sys.executable,
                        str(shrink_script),
                        "--all-categories",
                        "--input-artifact", shrink_input_artifact,
                        "--output-artifact", shrink_output_artifact
                    ],
                    check=True,
                    capture_output=False,
                    text=True
                )
                print(f"[OK] Topic shrinking completed successfully")
            except subprocess.CalledProcessError as e:
                print(f"[ERROR] Topic shrinking failed with exit code {e.returncode}")
                print(f"[WARNING] Continuing despite shrink failure...")
            except Exception as e:
                print(f"[ERROR] Error running shrink script: {e}")
                print(f"[WARNING] Continuing despite shrink failure...")
        else:
            print(f"[WARNING] Shrink script not found at {shrink_script}, skipping...")
        
        # =====================================================================
        # Final Summary
        # =====================================================================
        print("\n" + "=" * 70)
        print("STAGE 03 COMPLETE")
        print("=" * 70)
        print(f"\nSummary:")
        if all_category_results:
            print(f"  Categories processed: {len(all_category_results)}")
            print(f"  Total themes discovered: {total_themes}")
            if not topic_universe_exists:
                print(f"  Processing mode: {'PARALLEL' if use_parallel else 'SEQUENTIAL'}")
            else:
                print(f"  Topic discovery: SKIPPED (already exists)")
            
            print(f"\nOutput (LOCAL - always saved):")
            print(f"  Directory: {output_dir}")
            merged_file = output_dir / "topic_universe.json"
            if merged_file.exists():
                print(f"  Main file: {merged_file}")
            
            for category, stats in category_stats.items():
                print(f"\n  {category}:")
                print(f"    Themes: {stats['num_themes']}")
                if stats['rounds_completed'] > 0:
                    print(f"    Rounds: {stats['rounds_completed']}")
                if stats['num_reviews'] > 0:
                    print(f"    Reviews: {stats['num_reviews']}")
        else:
            print(f"  No categories processed")
        
        # W&B Upload Status (only show if we actually ran discovery)
        if not topic_universe_exists:
            # Check if variables exist (they're only defined if discovery ran)
            try:
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
                merged_file = output_dir / "topic_universe.json"
                if merged_file.exists():
                    print(f"  - Main file: {merged_file}")
            except NameError:
                # Variables not defined (shouldn't happen, but safe fallback)
                pass
        elif run:
            print(f"\nView run at: {run.url}")
        
    finally:
        finish_run(run)


if __name__ == "__main__":
    asyncio.run(main())
