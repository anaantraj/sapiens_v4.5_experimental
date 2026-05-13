#!/usr/bin/env python3
"""
Stage 06: Tribe Formation - Export Segment User Details
========================================================

Exports users with their full details grouped by segment/cluster.
- Loads user segments (from Script 01)
- Loads user review history (from Stage 05)
- Loads user backstories (from Stage 05)
- Groups users by segment
- Saves detailed JSON files per segment with all user data

Usage:
    python 06_tribe_formation/scripts/01b_export_segment_user_details.py
"""

import json
import sys
import logging
import re
from pathlib import Path
from collections import defaultdict
from typing import Dict, List

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from utils.wandb_utils import (
    load_config, get_stage_config,
    init_wandb_run, finish_run, use_artifact, log_artifact,
    log_metrics, log_summary, link_to_registry, get_artifact_dir
)

# Import schemas for validation
from schemas.learned_artifacts import (
    UserSegmentsArtifact,
    UserBackstoryArtifact
)
from schemas.learned_artifacts.segment_user_details import SegmentUserDetailsArtifact

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def main():
    """Main execution function."""
    
    # Load configuration
    stage_config = get_stage_config("06_tribe_formation")
    
    # Get artifact names from config (required, no fallbacks)
    if "segments_input_artifact_01b" not in stage_config:
        logging.error("Missing required config field: segments_input_artifact_01b")
        return
    
    segments_artifact = stage_config["segments_input_artifact_01b"]
    if "segments_filename" not in stage_config:
        logging.error("Missing required config field: segments_filename")
        return
    segments_filename = stage_config["segments_filename"]
    
    if "training_data_with_topics_artifact" not in stage_config:
        logging.error("Missing required config field: training_data_with_topics_artifact")
        return
    
    training_data_artifact = stage_config["training_data_with_topics_artifact"]
    if "training_data_filename" not in stage_config:
        logging.error("Missing required config field: training_data_filename")
        return
    training_data_filename = stage_config["training_data_filename"]
    
    if "backstory_input_artifact" not in stage_config:
        logging.error("Missing required config field: backstory_input_artifact")
        return
    
    backstory_artifact = stage_config["backstory_input_artifact"]
    if "backstory_filename" not in stage_config:
        logging.error("Missing required config field: backstory_filename")
        return
    backstory_filename = stage_config["backstory_filename"]
    
    if "segment_details_output_artifact" not in stage_config:
        logging.error("Missing required config field: segment_details_output_artifact")
        return
    
    output_artifact = stage_config["segment_details_output_artifact"]
    
    # Initialize W&B run
    run = init_wandb_run(
        run_name="export_segment_user_details",
        stage="06_tribe_formation",
        config={
            "description": "Export users with details grouped by segment",
            "segments_artifact": segments_artifact,
            "training_data_artifact": training_data_artifact,
            "backstory_artifact": backstory_artifact,
            "output_artifact": output_artifact,
            "schema_version": "v4"
        }
    )
    
    try:
        # =====================================================================
        # Step 1: Load user segments from W&B (no local fallback)
        # =====================================================================
        logging.info(f"Loading user segments from: {segments_artifact}")
        
        # Download segments artifact from W&B (required, no local fallback)
        segments_path = use_artifact(run, segments_artifact, artifact_type="dataset")
        
        if segments_path is None:
            logging.error(f"Could not download segments artifact: {segments_artifact}")
            logging.error(f"Make sure to run 06_tribe_formation/scripts/01_user_to_segments_mapper.py first and upload the artifact to W&B")
            return
        
        # Resolve path to handle any symlinks or relative paths
        # W&B may return path with :v0, :v1, etc. (invalid on Linux), need to handle this
        segments_path_str = str(segments_path)
        # If path contains :vN (invalid on Linux), try replacing with -vN
        if not Path(segments_path).exists():
            # Replace :v0, :v1, :v2, etc. with -v0, -v1, -v2, etc.
            segments_path_str = re.sub(r':(v\d+)', r'-\1', segments_path_str)
            segments_path = Path(segments_path_str)
        
        segments_path = Path(segments_path).resolve()
        logging.info(f"[W&B] Segments artifact downloaded to: {segments_path}")
        
        # Get segments file using filename from config (required, no fallback)
        segments_file = segments_path / segments_filename
        
        # Check if file exists - if not, list available files for debugging
        if not segments_file.exists():
            logging.error(f"Segments file not found in artifact: {segments_file}")
            logging.error(f"Expected file: {segments_filename}")
            
            # List available files for debugging (but don't use them)
            if segments_path.exists():
                available_files = list(segments_path.glob("*.json"))
                if available_files:
                    logging.error(f"Available JSON files in artifact:")
                    for f in available_files:
                        logging.error(f"  - {f.name}")
                else:
                    logging.error(f"No JSON files found in artifact directory: {segments_path}")
            else:
                logging.error(f"Artifact directory does not exist: {segments_path}")
            
            logging.error(f"Please update config.yaml with the correct segments_filename from the list above")
            return
        
        logging.info(f"[OK] Using segments file: {segments_file.name}")
        
        user_segments = UserSegmentsArtifact.from_file(segments_file)
        logging.info(f"Loaded segments for {len(user_segments)} users")
        
        # =====================================================================
        # Step 2: Load training data with topics from W&B (no local fallback)
        # =====================================================================
        logging.info(f"Loading training data with topics from: {training_data_artifact}")
        
        # Download training data artifact from W&B (required, no local fallback)
        training_path = use_artifact(run, training_data_artifact, artifact_type="dataset")
        
        if training_path is None:
            logging.error(f"Could not download training data artifact: {training_data_artifact}")
            logging.error(f"Make sure the artifact exists in W&B")
            return
        
        # Resolve path to handle any symlinks or relative paths
        training_path_str = str(training_path)
        if not Path(training_path).exists():
            # Replace :v0, :v1, :v2, etc. with -v0, -v1, -v2, etc.
            training_path_str = re.sub(r':(v\d+)', r'-\1', training_path_str)
            training_path = Path(training_path_str)
        
        training_path = Path(training_path).resolve()
        logging.info(f"[W&B] Training data artifact downloaded to: {training_path}")
        
        # Get training data file using filename from config (required, no fallback)
        training_file = training_path / training_data_filename
        
        # Check if file exists - if not, list available files for debugging
        if not training_file.exists():
            logging.error(f"Training data file not found in artifact: {training_file}")
            logging.error(f"Expected file: {training_data_filename}")
            
            # List available files for debugging (but don't use them)
            if training_path.exists():
                available_files = list(training_path.glob("*.json"))
                if available_files:
                    logging.error(f"Available JSON files in artifact:")
                    for f in available_files:
                        logging.error(f"  - {f.name}")
                else:
                    logging.error(f"No JSON files found in artifact directory: {training_path}")
            else:
                logging.error(f"Artifact directory does not exist: {training_path}")
            
            logging.error(f"Please update config.yaml with the correct training_data_filename from the list above")
            return
        
        logging.info(f"[OK] Using training data file: {training_file.name}")
        
        logging.info(f"Loading training data from: {training_file}")
        with open(training_file, 'r', encoding='utf-8') as f:
            training_data = json.load(f)
        
        logging.info(f"Loaded training data for {len(training_data)} users")
        
        # =====================================================================
        # Step 3: Load user backstories from W&B (optional, no local fallback)
        # =====================================================================
        user_backstories = {}
        try:
            logging.info(f"Loading user backstories from: {backstory_artifact}")
            
            # Download backstory artifact from W&B (required, no local fallback)
            backstory_path = use_artifact(run, backstory_artifact, artifact_type="dataset")
            
            if backstory_path is None:
                logging.warning(f"Could not download backstory artifact: {backstory_artifact}")
                logging.warning("Continuing without backstories")
                backstory_file = None
            else:
                # Resolve path to handle any symlinks or relative paths
                backstory_path_str = str(backstory_path)
                if not Path(backstory_path).exists():
                    # Replace :v0, :v1, :v2, etc. with -v0, -v1, -v2, etc.
                    backstory_path_str = re.sub(r':(v\d+)', r'-\1', backstory_path_str)
                    backstory_path = Path(backstory_path_str)
                
                backstory_path = Path(backstory_path).resolve()
                logging.info(f"[W&B] Backstory artifact downloaded to: {backstory_path}")
                
                # Get backstory file using filename from config (required, no fallback)
                backstory_file = backstory_path / backstory_filename
                
                if not backstory_file.exists():
                    logging.warning(f"Backstory file not found in artifact: {backstory_file}")
                    logging.warning(f"Expected file: {backstory_filename}")
                    logging.warning("Continuing without backstories")
                    backstory_file = None
                else:
                    logging.info(f"[OK] Using backstory file: {backstory_file.name}")
            
            if backstory_file and backstory_file.exists():
                user_backstories = UserBackstoryArtifact.from_file(backstory_file)
                logging.info(f"Loaded backstories for {len(user_backstories)} users")
        except Exception as e:
            logging.warning(f"Error loading backstories: {e}, continuing without them")
        
        # =====================================================================
        # Step 4: Group users by segment with full review data
        # =====================================================================
        logging.info("Grouping users by segment with full review data...")
        
        # Extract segment number from segment_id (e.g., "segment_0" -> 0)
        def get_segment_number(segment_id: str) -> int:
            try:
                return int(segment_id.replace("segment_", ""))
            except (ValueError, AttributeError) as e:
                logging.warning(f"Could not parse segment_id '{segment_id}': {e}")
                return -1
        
        # Group users by segment
        segment_users_dict = defaultdict(dict)  # segment_id -> {user_id: user_data}
        
        for user_id, segment_artifact in user_segments.items():
            segment_id = segment_artifact.segment_id
            segment_number = get_segment_number(segment_id)
            
            # Get user's training data (reviews with all fields)
            if user_id not in training_data:
                logging.warning(f"User {user_id} not found in training data, skipping")
                continue
            
            user_training_data = training_data[user_id]
            if not isinstance(user_training_data, dict):
                logging.warning(f"User {user_id} training data is not a dict, skipping")
                continue
            user_reviews = user_training_data.get('reviews', [])
            if not isinstance(user_reviews, list):
                logging.warning(f"User {user_id} reviews is not a list, skipping")
                continue
            
            # Process each review to include all fields
            processed_reviews = []
            for review in user_reviews:
                # Get themes and probabilities from Stage 04 merged data
                # Stage 04 uses: "themes" (list) and "theme_token_probabilities" (dict)
                themes_list = review.get("themes", review.get("predicted_themes", []))
                theme_probs = review.get("theme_token_probabilities", review.get("topic_probabilities", {}))
                
                # Calculate primary topic (theme with highest probability)
                primary_topic = None
                if theme_probs and isinstance(theme_probs, dict) and len(theme_probs) > 0:
                    try:
                        primary_topic = max(theme_probs.items(), key=lambda x: x[1])[0]
                    except (ValueError, TypeError) as e:
                        logging.warning(f"Error finding primary topic from theme_probs: {e}")
                        primary_topic = None
                elif themes_list and isinstance(themes_list, list) and len(themes_list) > 0:
                    primary_topic = themes_list[0]
                
                processed_review = {
                    "product_description": review.get("product_description", ""),
                    "review_text": review.get("review_text", review.get("review", "")),
                    "rating": review.get("rating"),
                    "category": review.get("category"),
                    "main_category": review.get("category"),  # For compatibility
                    "timestamp": review.get("timestamp"),
                    "predicted_themes": themes_list,  # List format (from "themes" in Stage 04)
                    "topic_probabilities": theme_probs,  # Dict format (from "theme_token_probabilities" in Stage 04)
                    "primary_topic": primary_topic,  # Calculated from highest probability theme
                    "sentiment": review.get("sentiment"),
                    "asin": review.get("asin"),
                    "cluster": segment_number  # Add cluster number to each review
                }
                # Explicitly exclude user characteristics from review objects
                # (in case they exist in source data)
                if "overall_characteristics" in processed_review:
                    processed_review.pop("overall_characteristics")
                if "category_characteristics" in processed_review:
                    processed_review.pop("category_characteristics")
                
                # User characteristics are added at user level only, not in each review
                processed_reviews.append(processed_review)
            
            # Create user data structure matching clustering_1.py format
            user_data = {
                "reviews": processed_reviews,
                "cluster": segment_number,
                "num_reviews": len(processed_reviews)
            }
            
            # Add user-level characteristics (only if they exist)
            if user_id in user_backstories:
                backstory_artifact = user_backstories[user_id]
                user_data["overall_characteristics"] = backstory_artifact.overall_characteristics.model_dump()
                # Only include category_characteristics if it exists and is not empty
                if backstory_artifact.category_characteristics:
                    user_data["category_characteristics"] = {
                        cat: char.model_dump()
                        for cat, char in backstory_artifact.category_characteristics.items()
                    }
                # Don't add empty category_characteristics
            else:
                # Don't add empty characteristics if user has no backstory
                pass
            
            segment_users_dict[segment_id][user_id] = user_data
        
        logging.info(f"Grouped users into {len(segment_users_dict)} segments")
        
        # =====================================================================
        # Step 5: Save detailed JSON files per segment (matching clustering_1.py format)
        # =====================================================================
        logging.info("Saving detailed JSON files per segment...")
        
        artifact_dir = get_artifact_dir("06_tribe_formation", output_artifact)
        artifact_dir.mkdir(parents=True, exist_ok=True)
        
        segment_stats = {}
        
        for segment_id, users_dict in segment_users_dict.items():
            # Extract segment number for filename
            segment_number = get_segment_number(segment_id)
            
            # Validate data against schema
            try:
                validated_data = SegmentUserDetailsArtifact.validate_data(users_dict)
                logging.info(f"  Validated data for segment {segment_number}: {len(validated_data)} users")
            except Exception as e:
                logging.error(f"  Schema validation FAILED for segment {segment_number}: {e}")
                logging.error(f"  This indicates a data quality issue. Continuing with unvalidated data.")
                validated_data = users_dict  # Use original if validation fails
            
            # Save per-segment file in the same format as clustering_1.py
            # Format: {user_id: {reviews: [...], overall_characteristics: {...}, ...}}
            segment_file = artifact_dir / f"details_cluster_{segment_number}.json"
            
            with open(segment_file, 'w', encoding='utf-8') as f:
                json.dump(validated_data, f, indent=4, ensure_ascii=False)
            
            total_reviews = sum(
                user.get("num_reviews", 0) 
                for user in validated_data.values() 
                if isinstance(user, dict)
            )
            segment_stats[segment_id] = {
                "num_users": len(validated_data),
                "total_reviews": total_reviews
            }
            
            logging.info(f"  Saved {segment_file.name}: {len(validated_data)} users, {total_reviews} reviews")
        
        # Save combined summary
        summary_file = artifact_dir / "segments_summary.json"
        summary_data = {
            "num_segments": len(segment_users_dict),
            "total_users": sum(stats["num_users"] for stats in segment_stats.values()),
            "total_reviews": sum(stats["total_reviews"] for stats in segment_stats.values()),
            "segment_stats": segment_stats
        }
        
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary_data, f, indent=2, ensure_ascii=False)
        
        logging.info(f"Saved summary: {summary_file}")
        
        # =====================================================================
        # Step 6: Log to W&B
        # =====================================================================
        logging.info("Logging artifact to W&B...")
        
        artifact_metadata = {
            **summary_data,
            "schema_version": "v4",
            "schema_validated": True,
            "artifact_type": "learned_artifact",
            "format": "llm_ready_json_data"
        }
        
        artifact = log_artifact(
            run=run,
            artifact_name=output_artifact,
            artifact_type="dataset",
            artifact_path=artifact_dir,
            metadata=artifact_metadata,
        )
        link_to_registry(artifact, stage="06_tribe_formation")
        
        # Log metrics
        log_metrics(run, {
            "num_segments": len(segment_users_dict),
            "total_users": summary_data["total_users"],
            "total_reviews": summary_data["total_reviews"],
            "avg_users_per_segment": summary_data["total_users"] / len(segment_users_dict) if segment_users_dict else 0,
            "avg_reviews_per_segment": summary_data["total_reviews"] / len(segment_users_dict) if segment_users_dict else 0
        })
        
        log_summary(run, {
            "status": "completed",
            "segments_exported": len(segment_users_dict),
            "users_exported": summary_data["total_users"]
        })
        
        logging.info("Export completed successfully!")
        
    except Exception as e:
        logging.error(f"Error in main execution: {e}", exc_info=True)
        log_summary(run, {"status": "failed", "error": str(e)})
        raise
    
    finally:
        finish_run(run)


if __name__ == "__main__":
    main()

