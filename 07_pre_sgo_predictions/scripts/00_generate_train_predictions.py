#!/usr/bin/env python3
"""
Stage 07: SGO Training - Generate Initial Predictions
======================================================

Pre-processing script that generates initial predictions for each tribe before SGO training.
- Loads tribe seed characteristics (from Stage 06)
- Loads user characteristics and training data (from Stage 04/05)
- Generates predictions for each review using persona + user characteristics
- Outputs predictions in format expected by SGO training

Usage:
    python 07_pre_sgo_predictions/scripts/00_generate_train_predictions.py
    
    # Process all clusters (default)
    python 07_pre_sgo_predictions/scripts/00_generate_train_predictions.py
    
    # Process only specific clusters
    python 07_pre_sgo_predictions/scripts/00_generate_train_predictions.py --clusters segment_0
    python 07_pre_sgo_predictions/scripts/00_generate_train_predictions.py --clusters segment_0 segment_1 segment_2
"""

import sys
import logging
import os
from pathlib import Path
import argparse
from openai import OpenAI

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Load .env so OPENAI_API_KEY and OPENAI_BASE_URL (Bedrock) are set
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent.parent.parent / ".env", override=True)
except ImportError:
    pass

from utils.openai_client import create_openai_client
from utils.wandb_utils import init_wandb_run, load_config, get_stage_config, get_openai_config, finish_run
from prediction_lib.data_loader import DataLoader
from prediction_lib.generate_model_predictions import PipelineOrchestrator
from prediction_lib.llm_client import RateLimiter
from prediction_lib.prompt_generation import load_prompt

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Paths will be determined from config

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--clusters", nargs="+", help="Clusters to process")
    args = parser.parse_args()

    # Config - everything must come from config, no defaults
    config = load_config()
    stage_config = get_stage_config("07_pre_sgo_predictions")
    openai_config = get_openai_config()
    
    # Validate required config sections
    if not stage_config:
        logging.error("❌ Stage config not found")
        return
    
    input_artifacts = stage_config.get("input_artifacts", {})
    if not input_artifacts:
        logging.error("❌ input_artifacts not found in config")
        return
    
    output_artifacts = stage_config.get("output_artifacts", {})
    if not output_artifacts:
        logging.error("❌ output_artifacts not found in config")
        return
    
    hyperparams = stage_config.get("hyperparameters", {})
    if not hyperparams:
        logging.error("❌ hyperparameters not found in config")
        return
    
    file_patterns = stage_config.get("file_patterns", {})
    if not file_patterns:
        logging.error("❌ file_patterns not found in config")
        return
    
    dataset_pattern_mapping = stage_config.get("dataset_pattern_mapping", {})
    if not dataset_pattern_mapping:
        logging.error("❌ dataset_pattern_mapping not found in config")
        return
    
    prompt_config = stage_config.get("prompt", {})
    if not prompt_config:
        logging.error("❌ prompt configuration not found in config")
        return
    
    schema_config = stage_config.get("schema", {})
    schema_version = schema_config.get("version")
    if not schema_version:
        logging.error("❌ schema.version not found in config")
        return
    
    # Get dataset_type from config (defaults to "train" for initial predictions)
    dataset_type = stage_config.get("dataset_type", "train")
    if dataset_type not in ["train", "test"]:
        logging.error(f"❌ Invalid dataset_type in config: {dataset_type}. Must be 'train' or 'test'")
        return
    
    # Get all artifact names from config (no defaults)
    tribe_seed_artifact = input_artifacts.get("tribe_seed_characteristics")
    user_backstory_artifact = input_artifacts.get("user_backstories")
    training_data_artifact = input_artifacts.get("training_data_with_topics")
    topic_universe_artifact = input_artifacts.get("topic_universe")
    user_tribes_artifact = input_artifacts.get("user_tribes")
    category_mapping_artifact = input_artifacts.get("category_mapping")  # Required
    
    if not all([tribe_seed_artifact, user_backstory_artifact, training_data_artifact, 
                topic_universe_artifact, user_tribes_artifact, category_mapping_artifact]):
        missing = [k for k, v in {
            "tribe_seed_characteristics": tribe_seed_artifact,
            "user_backstories": user_backstory_artifact,
            "training_data_with_topics": training_data_artifact,
            "topic_universe": topic_universe_artifact,
            "user_tribes": user_tribes_artifact,
            "category_mapping": category_mapping_artifact
        }.items() if not v]
        logging.error(f"❌ Missing required artifacts in config: {missing}")
        return
    
    # Get output artifact name from config
    base_output_artifact = output_artifacts.get("initial_predictions")
    if not base_output_artifact:
        logging.error("❌ initial_predictions output artifact not found in config")
        return
    
    # Get model configuration from config (no defaults)
    model_name = hyperparams.get("model")
    theme_prediction_model = None
    # Override model names for logprobs modes if specified in config
    if scoring_mode in ["logprobs", "logprobs-without-persona-context"]:
        review_prediction_model = stage_config.get("review_prediction_model")
        theme_prediction_model = stage_config.get("theme_prediction_model")
        if review_prediction_model:
            model_name = review_prediction_model
            logging.info(f"Using {review_prediction_model} for review generation (from config.review_prediction_model)")
        if theme_prediction_model:
            logging.info(f"Using {theme_prediction_model} for theme classification (from config.theme_prediction_model)")
        if stage_config.get("logprobs_model_id"):
            theme_prediction_model = stage_config["logprobs_model_id"]
            logging.info(f"Using {theme_prediction_model} for theme classification (from config.logprobs_model_id).")
    else:
        base_url = os.environ.get("OPENAI_BASE_URL", "").strip()
        if base_url and "bedrock-mantle" in base_url and stage_config.get("bedrock_model_id"):
            model_name = stage_config["bedrock_model_id"]
    
    min_request_interval = hyperparams.get("min_request_interval")
    max_tokens = hyperparams.get("max_tokens")
    temperature = hyperparams.get("temperature")
    num_workers = hyperparams.get("num_workers")
    max_retries = hyperparams.get("max_retries", 3)
    
    if not all([model_name, min_request_interval is not None, num_workers]):
        missing = [k for k, v in {
            "model": model_name,
            "min_request_interval": min_request_interval,
            "num_workers": num_workers
        }.items() if v is None]
        logging.error(f"❌ Missing required hyperparameters in config: {missing}")
        return
    
    # Update output artifact name to include model
    model_suffix = model_name.replace("-", "_").replace(".", "_")  # e.g., "gpt_4o" or "o3"
    output_artifact = f"{base_output_artifact}_{model_suffix}"
    
    # Get prompt configuration
    prompt_dir_name = prompt_config.get("directory")
    
    # Get scoring mode from config (defaults to "confidence")
    scoring_mode = stage_config.get("scoring_mode", "confidence")
    if scoring_mode not in ["confidence", "logprobs", "logprobs-without-persona-context"]:
        logging.error(f"❌ Invalid scoring_mode in config: {scoring_mode}. Must be 'confidence', 'logprobs', or 'logprobs-without-persona-context'")
        return
    
    # Select prompt filename based on scoring mode and enhanced prompt setting
    use_enhanced = stage_config.get("use_enhanced_prompt", False)
    
    if scoring_mode in ["logprobs", "logprobs-without-persona-context"]:
        if use_enhanced:
            prompt_filename = prompt_config.get("initial_prediction_enhanced_logprobs")
            if not prompt_filename:
                logging.warning("⚠️ Enhanced logprobs prompt not found, falling back to regular logprobs prompt")
                prompt_filename = prompt_config.get("initial_prediction_logprobs")
        else:
            prompt_filename = prompt_config.get("initial_prediction_logprobs")
        if not prompt_filename:
            logging.error("❌ initial_prediction_logprobs not found in prompt config (required for logprobs modes)")
            return
    else:
        if use_enhanced:
            prompt_filename = prompt_config.get("initial_prediction_enhanced")
            if not prompt_filename:
                logging.warning("⚠️ Enhanced prompt not found, falling back to regular prompt")
                prompt_filename = prompt_config.get("initial_prediction")
        else:
            prompt_filename = prompt_config.get("initial_prediction")
    
    if not prompt_dir_name or not prompt_filename:
        logging.error("❌ Prompt configuration incomplete in config")
        return
    
    logging.info(f"Using scoring mode: {scoring_mode}")
    
    logging.info(f"Using model: {model_name}")
    logging.info(f"Rate limiting: {min_request_interval}s between requests")
    if model_name != "o3":
        logging.info(f"Max tokens: {max_tokens}, Temperature: {temperature}")
    else:
        logging.info("Model is o3 - max_tokens and temperature will not be applied")
    logging.info(f"Output artifact: {output_artifact}")


    run = init_wandb_run(
        run_name="generate_train_predictions",
        stage="07_sgo_training",
        config={
            "description": "Generate initial predictions for each tribe before SGO training",
            "tribe_seed_input": tribe_seed_artifact,
            "user_backstory_input": user_backstory_artifact,
            "training_data_input": training_data_artifact,
            "topic_universe_input": topic_universe_artifact,
            "user_tribes_input": user_tribes_artifact,
            "category_mapping_input": category_mapping_artifact,
            "output_artifact": output_artifact,
            "num_workers": num_workers,
            "model": model_name,
            "min_request_interval": min_request_interval,
            "max_tokens": max_tokens if model_name != "o3" else None,
            "temperature": temperature if model_name != "o3" else None,
            "max_retries": max_retries,
            "schema_version": schema_version,
            "dataset_type": dataset_type
        }
    )

    if run is None:
        logging.error("❌ W&B run initialization failed. Cannot proceed without W&B (local fallbacks disabled).")
        return
    
    try:
        # Init Resources
        api_key = openai_config.get("api_key") or os.environ.get("OPENAI_API_KEY")
        if not api_key:
            logging.error("❌ OPENAI_API_KEY not found in config or environment")
            return

        if scoring_mode in ["logprobs", "logprobs-without-persona-context"]:
            client = OpenAI(api_key=api_key)
        else:
            client = create_openai_client(openai_config=openai_config, timeout=120.0)
        loader = DataLoader(run, file_patterns, dataset_pattern_mapping)
        limiter = RateLimiter(min_interval=min_request_interval)
        
        # Load prompt template from config path
        stage_dir = Path(__file__).parent.parent
        prompt_dir = stage_dir / prompt_dir_name
        prompt_path = prompt_dir / prompt_filename
        
        if not prompt_path.exists():
            logging.error(f"❌ Prompt file not found: {prompt_path}")
            return
        
        prompt_template = load_prompt(prompt_dir, prompt_filename)
        logging.info(f"✅ Loaded prompt template from: {prompt_path}")
        
        # Load Data from W&B only (no local fallbacks)
        logging.info(f"Loading tribe seed characteristics from W&B: {tribe_seed_artifact}")
        tribe_seed_data = loader.load_tribe_seeds(tribe_seed_artifact)
        if not tribe_seed_data:
            logging.error("❌ Failed to load tribe seed data")
            return
        logging.info(f"Loaded seed characteristics for {len(tribe_seed_data)} tribes")
        
        logging.info(f"Loading user backstories from W&B: {user_backstory_artifact}")
        user_backstories = loader.load_user_backstories(user_backstory_artifact)
        if not user_backstories:
            logging.error("❌ Failed to load user backstories")
            return
        logging.info(f"Loaded backstories for {len(user_backstories)} users")
        
        logging.info(f"Loading training data from W&B: {training_data_artifact}")
        review_data = loader.load_review_data(training_data_artifact, dataset_type="train")
        if not review_data:
            logging.error("❌ Failed to load training data")
            return
        
        # Count reviews with empty predicted_themes in input artifact
        total_reviews = 0
        reviews_with_empty_themes = 0
        for user_id, user_data in review_data.items():
            reviews = user_data.get('reviews', [])
            total_reviews += len(reviews)
            for review in reviews:
                review_themes = review.get('themes', []) or review.get('predicted_themes', [])
                if not review_themes or (isinstance(review_themes, list) and len(review_themes) == 0):
                    reviews_with_empty_themes += 1
        
        logging.info(f"Loaded training data for {len(review_data)} users with {total_reviews} total reviews")
        if reviews_with_empty_themes > 0:
            logging.warning(f"⚠️  Found {reviews_with_empty_themes} reviews with empty predicted_themes (will be skipped during processing)")
        else:
            logging.info(f"✅ All {total_reviews} reviews have predicted_themes")
        
        logging.info(f"Loading topic universe from W&B: {topic_universe_artifact}")
        topic_universe = loader.load_topic_universe(topic_universe_artifact)
        if not topic_universe:
            logging.error("❌ Failed to load topic universe")
            return
        logging.info(f"Loaded topic universe")
        
        logging.info(f"Loading user tribes from W&B: {user_tribes_artifact}")
        user_to_tribe_map = loader.load_user_tribes(user_tribes_artifact)
        if not user_to_tribe_map:
            logging.error("❌ Failed to load user tribes")
            return
        logging.info(f"Mapped {len(user_to_tribe_map)} users to tribes")
        
        # Load category mapping (required)
        logging.info(f"Loading category mapping from W&B: {category_mapping_artifact}")
        category_mapping = loader.load_category_mapping(category_mapping_artifact)
        if category_mapping is None:
            logging.error("❌ Failed to load category mapping - REQUIRED. Cannot proceed.")
            return

        # Extract output_stage from config (if specified, otherwise defaults to "07_sgo_training")
        output_config = stage_config.get("output", {})
        output_stage = output_config.get("stage", "07_sgo_training")
        # Add output_stage, scoring_mode, and theme_prediction_model to hyperparams so prediction generator can use it
        hyperparams_with_output = hyperparams.copy()
        hyperparams_with_output["output_stage"] = output_stage
        hyperparams_with_output["scoring_mode"] = scoring_mode
        if theme_prediction_model:
            hyperparams_with_output["theme_prediction_model"] = theme_prediction_model

        # Run Pipeline
        orchestrator = PipelineOrchestrator(
            run, client, limiter, 
            prompt_template, 
            hyperparams_with_output,
            category_mapping=category_mapping
        )
        
        orchestrator.run_pipeline(
            dataset_type=dataset_type,
            tribe_seed_data=tribe_seed_data,
            user_backstories=user_backstories,
            review_data=review_data,
            topic_universe=topic_universe,
            user_to_tribe_map=user_to_tribe_map,
            output_artifact_name=output_artifact,
            target_clusters=args.clusters
        )
        
        logging.info("Initial predictions generation completed successfully!")
        
    except Exception as e:
        logging.error(f"Error in main execution: {e}", exc_info=True)
        if 'run' in locals():
            from utils.wandb_utils import log_summary
            log_summary(run, {"status": "failed", "error": str(e)})
        raise
    
    finally:
        # Only finish the run if we created it (not if it was passed from unified runner)
        # Check if this run was created by us or passed from outside
        if 'run' in locals() and run is not None:
            # Check if run was created by init_wandb_run in this script
            # If it was passed from outside (unified runner), don't finish it
            # We can detect this by checking if the run name matches what we would create
            run_name = getattr(run, 'name', '') if hasattr(run, 'name') else ''
            if run_name == "generate_train_predictions" or not run_name:
                # This is our run, finish it
                finish_run(run)
            else:
                # This run was passed from outside, don't finish it
                logging.debug(f"Skipping finish_run for passed run: {run_name}")

# DISABLED: Entry point moved to 10_running_simulations/scripts/run_pre_sgo_training.py
# This file should be imported and called from run_unified_simulations.py
# if __name__ == "__main__":
#     main()