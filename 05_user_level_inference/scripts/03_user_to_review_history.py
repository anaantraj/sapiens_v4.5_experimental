#!/usr/bin/env python3
"""
Stage 05: User Level Inference - Review History
===============================================

Compiles structured review history for each user from review data.
- Reads configuration from config.yaml
- Downloads review data artifact from W&B (train_set_v4)
- Compiles review history for each user
- Validates with schema before saving
- Logs artifact to W&B

Usage:
    python 05_user_level_inference/scripts/03_user_to_review_history.py
"""

import json
import sys
from pathlib import Path
from typing import Dict, List
import logging

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from utils.wandb_utils import (
    load_config, get_stage_config,
    init_wandb_run, finish_run, use_artifact, log_artifact,
    log_metrics, log_summary, link_to_registry, get_artifact_dir
)

# Import schema for validation
from schemas.learned_artifacts import UserReviewHistoryArtifact, ReviewHistoryItem

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def load_review_data(artifact_path: Path, file_patterns: List[str]) -> Dict:
    """
    Load review data from artifact.
    
    Args:
        artifact_path: Path to downloaded artifact directory
        file_patterns: List of filename patterns to try (in order of preference)
        
    Returns:
        Dictionary with user_id -> user_data structure
    """
    # Try each file pattern in order
    review_file = None
    for pattern in file_patterns:
        candidate_file = artifact_path / pattern
        if candidate_file.exists():
            review_file = candidate_file
            break
    
    if review_file is None:
        tried_files = [str(artifact_path / pattern) for pattern in file_patterns]
        raise FileNotFoundError(
            f"Review file not found in artifact: {artifact_path}\n"
            f"  Tried files: {tried_files}"
        )
    
    logging.info(f"Loading reviews from '{review_file}'...")
    
    try:
        with open(review_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        logging.error(f"Error decoding JSON file: {e}")
        raise
    except Exception as e:
        logging.error(f"Error reading file: {e}")
        raise


def compile_user_review_history(user_id: str, user_data: Dict) -> UserReviewHistoryArtifact:
    """
    Compile review history for a single user.
    
    Args:
        user_id: User identifier
        user_data: User data containing reviews
        
    Returns:
        UserReviewHistoryArtifact instance
    """
    reviews = user_data.get('reviews', [])
    
    review_history_items = []
    for review in reviews:
        # Extract only required fields
        review_id = review.get("review_id") or f"{user_id}_{review.get('asin', 'unknown')}_{review.get('timestamp', 'unknown')}"
        product_description = review.get("product_description", "")
        review_text = review.get("review_text", review.get("review", ""))
        
        # Only include if we have all required fields
        if review_id and product_description and review_text:
            review_item = {
                "review_id": review_id,
                "product_description": product_description,
                "review_text": review_text
            }
            review_history_items.append(ReviewHistoryItem(**review_item))
    
    # Create artifact
    user_history_data = {
        "user_id": user_id,
        "review_history": review_history_items
    }
    
    return UserReviewHistoryArtifact.from_dict(user_history_data)


def main():
    """Main execution function."""
    
    # Load configuration
    stage_config = get_stage_config("05_user_level_inference")
    
    # Get artifact names from config (REQUIRED - no hardcoded defaults)
    input_artifact_name = stage_config.get("review_history_input_artifact")
    output_artifact_name = stage_config.get("review_history_output_artifact")
    output_filename = stage_config.get("review_history_output_filename", "user_review_history.json")
    input_file_patterns = stage_config.get("review_history_input_file_patterns", [
        "train_set_reviews_with_topics.json",
        "train_set_reviews.json",
        "full_user_reviews.json"
    ])
    
    # Validate required config values
    if not input_artifact_name:
        logging.error("❌ review_history_input_artifact not found in config.yaml - REQUIRED")
        return
    if not output_artifact_name:
        logging.error("❌ review_history_output_artifact not found in config.yaml - REQUIRED")
        return
    
    # Initialize W&B run
    run = init_wandb_run(
        run_name="user_to_review_history",
        stage="05_user_level_inference",
        config={
            "description": "Compile user review histories",
            "input_artifact": input_artifact_name,
            "output_artifact": output_artifact_name,
            "schema_version": "v4"
        }
    )
    
    try:
        # Get input artifact from W&B (NO LOCAL FALLBACK)
        logging.info(f"Loading input artifact from W&B: {input_artifact_name}...")
        logging.info("  [INFO] NO LOCAL FALLBACK - downloading from W&B only")
        
        artifact_dir = use_artifact(run, input_artifact_name, artifact_type="dataset")
        
        if artifact_dir is None:
            logging.error(f"❌ Could not download reviews artifact from W&B: {input_artifact_name}")
            logging.error("   No local fallback available - artifact must be in W&B")
            return
        
        # Load review data using file patterns from config
        review_data = load_review_data(artifact_dir, input_file_patterns)
        logging.info(f"Loaded review data for {len(review_data)} users")
        
        # Compile review history for each user
        user_histories = {}
        total_reviews = 0
        
        for user_id, user_data in review_data.items():
            try:
                user_history = compile_user_review_history(user_id, user_data)
                user_histories[user_id] = user_history.to_dict()
                total_reviews += len(user_history.review_history)
            except Exception as e:
                logging.error(f"Error compiling history for user {user_id}: {e}")
                continue
        
        logging.info(f"Compiled review history for {len(user_histories)} users ({total_reviews} total reviews)")
        
        # Save to artifacts directory (local storage)
        output_dir = get_artifact_dir("05_user_level_inference", output_artifact_name)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Use output filename from config
        output_file = output_dir / output_filename
        
        logging.info(f"💾 Saving review history locally to '{output_file}'...")
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(user_histories, f, indent=2, ensure_ascii=False)
        logging.info(f"✅ Saved locally: {output_file}")
        
        # Validate all user histories
        logging.info("Validating all user histories against schema...")
        validated_count = 0
        validation_errors = []
        
        for user_id, user_data in user_histories.items():
            try:
                UserReviewHistoryArtifact.from_dict(user_data)
                validated_count += 1
            except Exception as e:
                validation_errors.append(f"User {user_id}: {e}")
        
        if validation_errors:
            logging.warning(f"Validation errors for {len(validation_errors)} users:")
            for error in validation_errors[:10]:  # Show first 10 errors
                logging.warning(f"  {error}")
        
        logging.info(f"Validated {validated_count}/{len(user_histories)} user histories")
        
        # Log artifact to W&B (also saves to W&B)
        logging.info(f"📦 Logging artifact '{output_artifact_name}' to W&B...")
        artifact = log_artifact(
            run=run,
            artifact_name=output_artifact_name,
            artifact_type="dataset",
            artifact_path=output_dir,
            metadata={
                "num_users": len(user_histories),
                "total_reviews": total_reviews,
                "avg_reviews_per_user": total_reviews / len(user_histories) if user_histories else 0,
                "schema_version": "v4",
                "schema_validated": True,
                "validation_errors": len(validation_errors),
                "input_artifact": input_artifact_name
            }
        )
        
        # Link to registry
        link_to_registry(artifact, stage="05_user_level_inference")
        
        if artifact:
            logging.info(f"✅ Artifact logged to W&B: {artifact.name}")
            logging.info(f"   📁 Local path: {output_file}")
            logging.info(f"   📦 W&B artifact: {output_artifact_name}")
        
        # Log metrics
        log_metrics(run, {
            "num_users": len(user_histories),
            "total_reviews": total_reviews,
            "avg_reviews_per_user": total_reviews / len(user_histories) if user_histories else 0,
            "validation_errors": len(validation_errors)
        })
        
        # Log summary
        log_summary(run, {
            "status": "completed",
            "users_processed": len(user_histories),
            "total_reviews": total_reviews
        })
        
        logging.info("Review history compilation completed successfully!")
        
    except Exception as e:
        logging.error(f"Error in main execution: {e}", exc_info=True)
        log_summary(run, {"status": "failed", "error": str(e)})
        raise
        
    finally:
        finish_run(run)


if __name__ == "__main__":
    main()
