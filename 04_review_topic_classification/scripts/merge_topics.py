#!/usr/bin/env python3
"""
Stage 04: Review Topic Classification - Merge Topics
====================================================

Merges topic classifications back into the original training/test set data.
- Loads original train/test set data from Stage 02
- Loads review topic classifications from Script 01
- Matches reviews and merges topic data
- Saves merged training/test data with topics

Usage:
    python 04_review_topic_classification/scripts/merge_topics.py
"""

import json
import sys
import logging
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Set
import hashlib

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from utils.wandb_utils import (
    load_config, get_stage_config,
    init_wandb_run, finish_run, use_artifact, log_artifact,
    log_metrics, log_summary, link_to_registry, get_artifact_dir
)

# Import schemas for validation
from schemas.learned_artifacts import ReviewTopicClassificationArtifact

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def create_review_key(user_id: str, asin: Optional[str], timestamp: Optional[str], review_text: str) -> str:
    """
    Create a unique key for matching reviews.
    
    Args:
        user_id: User ID
        asin: Product ASIN
        timestamp: Review timestamp
        review_text: Review text (first 100 chars for matching)
        
    Returns:
        Unique key string
    """
    # Use first 100 chars of review text for matching
    text_hash = hashlib.md5(review_text[:100].encode()).hexdigest()[:8]
    return f"{user_id}_{asin or 'unknown'}_{timestamp or 'unknown'}_{text_hash}"


def create_review_id(review: Dict) -> str:
    """Create a unique ID for a review based on its content."""
    user_id = review.get('user_id', '')
    asin = review.get('asin', '')
    timestamp = review.get('timestamp', '')
    review_text = review.get('review_text', review.get('review', ''))
    return create_review_key(user_id, asin, timestamp, review_text)


def load_checkpoint(checkpoint_file: Path) -> Tuple[Dict, Dict, Set[str]]:
    """
    Load checkpoint data if it exists.
    
    Returns:
        Tuple of (all_merged_data, filtered_merged_data, processed_review_ids)
    """
    if not checkpoint_file.exists():
        return {}, {}, set()
    
    logging.info(f"Loading checkpoint from: {checkpoint_file}")
    try:
        with open(checkpoint_file, 'r', encoding='utf-8') as f:
            checkpoint_data = json.load(f)
        
        all_merged = checkpoint_data.get('all_merged_data', {})
        filtered_merged = checkpoint_data.get('filtered_merged_data', {})
        processed_ids = set(checkpoint_data.get('processed_review_ids', []))
        
        logging.info(f"  Loaded {len(processed_ids)} processed reviews")
        logging.info(f"  All merged: {sum(len(u.get('reviews', [])) for u in all_merged.values())} reviews")
        logging.info(f"  Filtered merged: {sum(len(u.get('reviews', [])) for u in filtered_merged.values())} reviews")
        
        return all_merged, filtered_merged, processed_ids
    except Exception as e:
        logging.warning(f"Error loading checkpoint: {e}. Starting fresh.")
        return {}, {}, set()


def save_checkpoint(
    checkpoint_file: Path,
    all_merged_data: Dict,
    filtered_merged_data: Dict,
    processed_review_ids: Set[str]
):
    """Save checkpoint data."""
    checkpoint_data = {
        'all_merged_data': all_merged_data,
        'filtered_merged_data': filtered_merged_data,
        'processed_review_ids': list(processed_review_ids)
    }
    
    try:
        with open(checkpoint_file, 'w', encoding='utf-8') as f:
            json.dump(checkpoint_data, f, indent=2, ensure_ascii=False)
        logging.debug(f"Checkpoint saved: {len(processed_review_ids)} reviews processed")
    except Exception as e:
        logging.warning(f"Error saving checkpoint: {e}")


def load_data(filepath: Path) -> Dict:
    """Load review data (train or test set)."""
    if not filepath.exists():
        error_msg = f"Data file not found: {filepath}"
        logging.error(error_msg)
        raise FileNotFoundError(error_msg)
    
    logging.info(f"Loading data from: {filepath}")
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        error_msg = f"Error decoding JSON file '{filepath}': {e}"
        logging.error(error_msg)
        raise ValueError(error_msg) from e
    except Exception as e:
        error_msg = f"Error reading file '{filepath}': {e}"
        logging.error(error_msg)
        raise RuntimeError(error_msg) from e


def load_topic_classifications(filepath: Path) -> Dict[str, ReviewTopicClassificationArtifact]:
    """Load review topic classifications from JSONL file."""
    if not filepath.exists():
        error_msg = f"Topic classifications file not found: {filepath}"
        logging.error(error_msg)
        raise FileNotFoundError(error_msg)
    
    logging.info(f"Loading topic classifications from: {filepath}")
    
    # Load JSONL file (one classification per line)
    classifications = {}
    parse_errors = 0
    
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                if not line.strip():
                    continue
                try:
                    data = json.loads(line.strip())
                    # Ensure all required fields are present
                    if 'review_id' not in data:
                        logging.warning(f"Line {line_num}: Missing review_id, skipping")
                        parse_errors += 1
                        continue
                    
                    classification = ReviewTopicClassificationArtifact.from_dict(data)
                    classifications[classification.review_id] = classification
                except Exception as e:
                    logging.warning(f"Error parsing line {line_num}: {e}")
                    parse_errors += 1
                    continue  # Skip bad lines but continue processing (data quality issue)
    
    except Exception as e:
        error_msg = f"Error reading topic classifications file '{filepath}': {e}"
        logging.error(error_msg)
        raise RuntimeError(error_msg) from e
    
    if len(classifications) == 0 and parse_errors > 0:
        error_msg = f"No valid topic classifications loaded from '{filepath}'. All {parse_errors} lines had errors."
        logging.error(error_msg)
        raise ValueError(error_msg)
    
    logging.info(f"Loaded {len(classifications)} topic classifications")
    if parse_errors > 0:
        logging.warning(f"Skipped {parse_errors} lines due to parsing errors (data quality issues)")
    
    return classifications


def merge_topics_into_data(
    review_data: Dict,
    topic_classifications: Dict[str, ReviewTopicClassificationArtifact],
    checkpoint_file: Optional[Path] = None,
    save_interval: int = 100
) -> Tuple[Dict, Dict, int]:
    """
    Merge topic classifications into review data (train or test set).
    
    Args:
        review_data: Original review data (user_id -> {reviews: [...]})
        topic_classifications: Review topic classifications (review_id -> classification)
        
    Returns:
        Tuple of:
        - all_merged_data: All reviews with all probabilities (including zeros)
        - filtered_merged_data: Only reviews with at least one theme (regardless of probability)
        - removed_count: Number of reviews removed (no themes present)
    """
    logging.info("Merging topic classifications into review data...")
    
    # Load checkpoint if exists
    processed_review_ids = set()
    if checkpoint_file and checkpoint_file.exists():
        all_merged_data, filtered_merged_data, processed_review_ids = load_checkpoint(checkpoint_file)
        logging.info(f"Resuming from checkpoint: {len(processed_review_ids)} reviews already processed")
    else:
        all_merged_data = {}  # All reviews (including all-zero probabilities)
        filtered_merged_data = {}  # Only reviews with at least one theme (regardless of probability)
    
    matched_count = 0
    unmatched_count = 0
    removed_count = 0  # Count reviews removed due to all-zero probabilities
    processed_count = len(processed_review_ids)
    total_reviews = sum(len(user_data.get('reviews', [])) for user_data in review_data.values())
    
    for user_id, user_data in review_data.items():
        all_merged_user_data = user_data.copy()
        filtered_merged_user_data = user_data.copy()
        all_merged_reviews = []
        filtered_merged_reviews = []
        
        for review in user_data.get('reviews', []):
            # Create unique review ID for checkpoint tracking
            review_id_key = create_review_id(review)
            
            # Skip if already processed (resume functionality)
            if review_id_key in processed_review_ids:
                continue
            
            merged_review = review.copy()
            
            # Try to find matching classification
            review_text = review.get('review_text', review.get('review', ''))
            asin = review.get('asin')
            timestamp = review.get('timestamp')
            
            # Strategy 1: Use review_id if it exists
            review_id = review.get('review_id')
            if review_id and review_id in topic_classifications:
                classification = topic_classifications[review_id]
                matched_count += 1
            else:
                # Strategy 2: Match by user_id + review_text + asin
                # The review_id in classifications is typically: user_id_asin_timestamp_hash
                matching_classification = None
                for classification in topic_classifications.values():
                    if (classification.user_id == user_id and
                        classification.review == review_text and
                        (not asin or not classification.asin or classification.asin == asin)):
                        matching_classification = classification
                        break
                
                if matching_classification:
                    classification = matching_classification
                    matched_count += 1
                else:
                    unmatched_count += 1
                    classification = None
            
            # Merge topic data if found
            if classification:
                # Use theme_token_probabilities (new field) - already contains only identified themes with their token probabilities
                theme_token_probs = classification.theme_token_probabilities or {}
                themes_list = classification.identified_themes or []  # Read from schema as identified_themes
                
                # Filter out themes with probability = 0, keep only non-zero probabilities
                non_zero_probabilities = {
                    theme: prob 
                    for theme, prob in theme_token_probs.items() 
                    if prob > 0.0
                }
                
                # Store theme_token_probabilities (token probabilities for identified themes)
                merged_review['theme_token_probabilities'] = theme_token_probs
                merged_review['themes'] = themes_list  # Renamed from identified_themes to themes
                
                merged_review['sentiment'] = classification.sentiment
                
                # Remove review_id field (not needed - reviews are grouped by user_id)
                if 'review_id' in merged_review:
                    del merged_review['review_id']
                
                # Add to ALL merged data (includes all reviews, even all-zero)
                all_merged_reviews.append(merged_review)
                
                # Add to FILTERED merged data only if has at least one theme
                # Filter based on number of themes, not probability values
                if len(themes_list) > 0:
                    filtered_merged_reviews.append(merged_review)
                else:
                    removed_count += 1
                    logging.debug(f"Filtering out review {review_id or 'unknown'}: no themes present")
            else:
                # No classification found - add to all_merged with defaults, but not to filtered
                # This review will be excluded from filtered file (empty themes)
                removed_count += 1
                merged_review['theme_token_probabilities'] = {}
                merged_review['themes'] = []  # Empty themes (renamed from identified_themes)
                merged_review['sentiment'] = None
                all_merged_reviews.append(merged_review)
                # Don't add to filtered_merged_reviews (empty themes)
                logging.debug(f"No classification found for review {review_id or 'unknown'} - excluding from filtered file")
            
            # Mark as processed and save checkpoint periodically
            processed_review_ids.add(review_id_key)
            processed_count += 1
            
            if checkpoint_file and processed_count % save_interval == 0:
                save_checkpoint(checkpoint_file, all_merged_data, filtered_merged_data, processed_review_ids)
                logging.info(f"Progress: {processed_count}/{total_reviews} reviews processed ({processed_count/total_reviews*100:.1f}%)")
        
        # Add user to all_merged_data if they have any reviews
        if all_merged_reviews:
            all_merged_user_data['reviews'] = all_merged_reviews
            all_merged_data[user_id] = all_merged_user_data
        
        # Add user to filtered_merged_data only if they have reviews with at least one theme
        if filtered_merged_reviews:
            filtered_merged_user_data['reviews'] = filtered_merged_reviews
            filtered_merged_data[user_id] = filtered_merged_user_data
    
    # Final checkpoint save
    if checkpoint_file:
        save_checkpoint(checkpoint_file, all_merged_data, filtered_merged_data, processed_review_ids)
    
    logging.info(f"Matched {matched_count} reviews, {unmatched_count} unmatched")
    logging.info(f"Processed {processed_count} total reviews")
    logging.info(f"Removed {removed_count} reviews from filtered dataset (no themes present)")
    logging.info(f"All reviews dataset: {sum(len(u.get('reviews', [])) for u in all_merged_data.values())} reviews")
    logging.info(f"Filtered dataset: {sum(len(u.get('reviews', [])) for u in filtered_merged_data.values())} reviews")
    return all_merged_data, filtered_merged_data, removed_count


def main():
    """Main execution function."""
    
    print("=" * 70)
    print("STAGE 04: Merge Topics with Review Data")
    print("=" * 70)
    
    # Load configuration
    stage_config = get_stage_config("04_review_topic_classification")
    
    # Get stage directory from config
    stage_directory = stage_config.get("stage_directory")
    if not stage_directory:
        raise ValueError("stage_directory must be specified in config.yaml")
    
    hyperparams = stage_config.get("hyperparameters")
    if not hyperparams:
        raise ValueError("hyperparameters must be specified in config.yaml")
    
    # Get dataset_mode from config (required)
    dataset_mode = hyperparams.get("dataset_mode")
    if not dataset_mode:
        raise ValueError("hyperparameters.dataset_mode must be specified in config.yaml")
    
    if dataset_mode not in ["train", "test"]:
        raise ValueError(f"hyperparameters.dataset_mode must be 'train' or 'test', got: {dataset_mode}")
    
    # Get input artifact names from config (required, no defaults)
    # Should be a dict with 'train' and 'test' keys, or a pattern string
    input_artifact_reviews_config = stage_config.get("input_artifact_reviews")
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
    
    output_artifact_topics = stage_config.get("output_artifact")
    if not output_artifact_topics:
        raise ValueError("output_artifact must be specified in config.yaml")
    
    # Get merged output artifact pattern from config
    merged_output_artifact_pattern = stage_config.get("merged_output_artifact_pattern")
    if not merged_output_artifact_pattern:
        raise ValueError("merged_output_artifact_pattern must be specified in config.yaml")
    
    # Format output artifact name based on dataset_mode
    merged_output_artifact = merged_output_artifact_pattern.format(dataset_mode=dataset_mode)
    
    # Get paths configuration
    paths_config = stage_config.get("paths")
    if not paths_config:
        raise ValueError("paths must be specified in config.yaml")
    
    input_review_filenames = paths_config.get("input_review_filenames")
    if not input_review_filenames:
        raise ValueError("paths.input_review_filenames must be specified in config.yaml")
    
    if dataset_mode not in input_review_filenames:
        raise ValueError(f"paths.input_review_filenames.{dataset_mode} must be specified in config.yaml")
    
    review_filenames = input_review_filenames.get(dataset_mode)
    if not review_filenames or not isinstance(review_filenames, list) or len(review_filenames) == 0:
        raise ValueError(f"paths.input_review_filenames.{dataset_mode} must be a non-empty list in config.yaml")
    
    output_filename_pattern = paths_config.get("output_filename_pattern")
    if not output_filename_pattern:
        raise ValueError("paths.output_filename_pattern must be specified in config.yaml")
    
    merged_output_filenames = paths_config.get("merged_output_filenames")
    if not merged_output_filenames:
        raise ValueError("paths.merged_output_filenames must be specified in config.yaml")
    
    intermediate_filename_pattern = merged_output_filenames.get("intermediate")
    if not intermediate_filename_pattern:
        raise ValueError("paths.merged_output_filenames.intermediate must be specified in config.yaml")
    
    filtered_filename_pattern = merged_output_filenames.get("filtered")
    if not filtered_filename_pattern:
        raise ValueError("paths.merged_output_filenames.filtered must be specified in config.yaml")
    
    # Get artifact type from config
    artifact_type = stage_config.get("artifact_type")
    if not artifact_type:
        raise ValueError("artifact_type must be specified in config.yaml")
    
    # Get job_type from config
    job_type = stage_config.get("job_type")
    if not job_type:
        raise ValueError("job_type must be specified in config.yaml")
    
    print(f"\n[Config] Dataset mode: {dataset_mode}")
    print(f"[Config] Input artifact (reviews): {input_artifact_reviews}")
    print(f"[Config] Input artifact (topics): {output_artifact_topics}")
    print(f"[Config] Output artifact: {merged_output_artifact}")
    
    # Initialize W&B run
    run = init_wandb_run(
        run_name=f"merge_topics_with_{dataset_mode}_data",
        stage=stage_directory,
        job_type=job_type
    )
    
    try:
        # =====================================================================
        # Step 1: Download review data from W&B (ONLY - no local fallback)
        # =====================================================================
        print("\n" + "-" * 70)
        print("Step 1: Download Review Data from W&B")
        print("-" * 70)
        print(f"[REQUIRED] Input artifact: {input_artifact_reviews}")
        print(f"[REQUIRED] Artifact type: {artifact_type}")
        print(f"[INFO] Downloading from W&B (NO local fallback)...")
        
        # Download review data artifact from W&B (ONLY - no local fallback)
        data_path = use_artifact(run, input_artifact_reviews, artifact_type=artifact_type)
        
        if data_path is None:
            print(f"[ERROR] ✗ Could not download review data artifact from W&B: {input_artifact_reviews}")
            print(f"[ERROR]   Make sure artifact '{input_artifact_reviews}' exists in W&B")
            print(f"[ERROR]   Make sure Stage 02 has been completed and artifact uploaded")
            print(f"[ERROR]   No local fallback available - W&B is the only source")
            return
        
        # Resolve path to handle any symlinks or relative paths
        # W&B may return path with :v0, :v1, etc. (invalid on Linux), need to handle this
        data_path_str = str(data_path)
        # If path contains :vN (invalid on Linux), try replacing with -vN
        # Handle any version number (v0, v1, v2, etc.)
        if not Path(data_path).exists():
            # Replace :v0, :v1, :v2, etc. with -v0, -v1, -v2, etc.
            data_path_str = re.sub(r':(v\d+)', r'-\1', data_path_str)
            data_path = Path(data_path_str)
        
        data_path = Path(data_path).resolve()
        print(f"[OK] ✓ Review data artifact downloaded to: {data_path}")
        
        # Use exact filenames from config (no fallback)
        data_file = None
        for filename in review_filenames:
            candidate = data_path / filename
            if candidate.exists():
                data_file = candidate
                logging.info(f"Found review data file: {filename}")
                break
        
        if data_file is None:
            error_msg = f"Review data file not found in artifact: {data_path}\n"
            error_msg += f"Expected filenames (in order): {review_filenames}\n"
            error_msg += f"Artifact path: {data_path}\n"
            error_msg += f"[DEBUG] Listing files in artifact directory:\n"
            if data_path.exists():
                for item in data_path.iterdir():
                    error_msg += f"  - {item.name} ({'file' if item.is_file() else 'dir'})\n"
            else:
                error_msg += f"  [ERROR] Artifact directory does not exist!\n"
            logging.error(error_msg)
            raise FileNotFoundError(f"Required review data file not found in W&B artifact. {error_msg}")
        
        review_data = load_data(data_file)
        original_total_reviews = sum(len(user_data.get('reviews', [])) for user_data in review_data.values())
        logging.info(f"Loaded data for {len(review_data)} users ({original_total_reviews} reviews)")
        
        # =====================================================================
        # Step 2: Download topic classifications from W&B (ONLY - no local fallback)
        # =====================================================================
        print("\n" + "-" * 70)
        print("Step 2: Download Topic Classifications from W&B")
        print("-" * 70)
        print(f"[REQUIRED] Input artifact: {output_artifact_topics}")
        print(f"[REQUIRED] Artifact type: {artifact_type}")
        print(f"[INFO] Downloading from W&B (NO local fallback)...")
        
        # Download topic classifications artifact from W&B (ONLY - no local fallback)
        topics_path = use_artifact(run, output_artifact_topics, artifact_type=artifact_type)
        
        if topics_path is None:
            print(f"[ERROR] ✗ Could not download topic classifications artifact from W&B: {output_artifact_topics}")
            print(f"[ERROR]   Make sure artifact '{output_artifact_topics}' exists in W&B")
            print(f"[ERROR]   Make sure Stage 04 Script 01 has been completed and artifact uploaded")
            print(f"[ERROR]   No local fallback available - W&B is the only source")
            return
        
        # Resolve path to handle any symlinks or relative paths
        # W&B may return path with :v0, :v1, etc. (invalid on Linux), need to handle this
        topics_path_str = str(topics_path)
        # If path contains :vN (invalid on Linux), try replacing with -vN
        # Handle any version number (v0, v1, v2, etc.)
        if not Path(topics_path).exists():
            # Replace :v0, :v1, :v2, etc. with -v0, -v1, -v2, etc.
            topics_path_str = re.sub(r':(v\d+)', r'-\1', topics_path_str)
            topics_path = Path(topics_path_str)
        
        topics_path = Path(topics_path).resolve()
        print(f"[OK] ✓ Topic classifications artifact downloaded to: {topics_path}")
        
        # Debug: Check if path exists and list contents
        if not topics_path.exists():
            print(f"[ERROR] ✗ Artifact directory does not exist: {topics_path}")
            print(f"[ERROR]   This indicates a problem with W&B artifact download")
            print(f"[ERROR]   Artifact name: {output_artifact_topics}")
            raise FileNotFoundError(f"W&B artifact directory does not exist: {topics_path}")
        
        print(f"[DEBUG] Artifact directory contents:")
        for item in topics_path.iterdir():
            print(f"  - {item.name} ({'file' if item.is_file() else 'dir'})")
        
        # Use exact filename from config (no fallback - W&B only)
        topics_filename = output_filename_pattern.format(dataset_mode=dataset_mode)
        topics_file = topics_path / topics_filename
        
        if not topics_file.exists():
            error_msg = f"Topic classifications file not found in artifact: {topics_file}\n"
            error_msg += f"Expected exact filename: {topics_filename}\n"
            error_msg += f"Artifact path: {topics_path}\n"
            error_msg += f"Make sure Stage 04 Script 01 has been completed with dataset_mode='{dataset_mode}'\n"
            logging.error(error_msg)
            raise FileNotFoundError(f"Required topic classifications file '{topics_filename}' not found in W&B artifact '{output_artifact_topics}'. {error_msg}")
        
        topic_classifications = load_topic_classifications(topics_file)
        
        # Setup checkpointing
        artifact_dir = get_artifact_dir(stage_directory, merged_output_artifact)
        artifact_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_file = artifact_dir / "merge_checkpoint.json"
        checkpoint_interval = hyperparams.get("checkpoint_interval")
        if checkpoint_interval is None:
            raise ValueError("hyperparameters.checkpoint_interval must be specified in config.yaml")
        
        # =====================================================================
        # Step 3: Merge topics into review data
        # =====================================================================
        print("\n" + "-" * 70)
        print("Step 3: Merge Topics into Review Data")
        print("-" * 70)
        all_merged_data, filtered_merged_data, removed_reviews = merge_topics_into_data(
            review_data, topic_classifications, checkpoint_file, checkpoint_interval
        )
        
        # =====================================================================
        # Step 4: Save merged data locally (both intermediate and final)
        # =====================================================================
        print("\n" + "-" * 70)
        print("Step 4: Save Merged Data Locally")
        print("-" * 70)
        print(f"[INFO] Saving output locally FIRST (before W&B upload)...")
        
        # Generate output filenames based on dataset_mode
        intermediate_filename = intermediate_filename_pattern.format(dataset_mode=dataset_mode)
        filtered_filename = filtered_filename_pattern.format(dataset_mode=dataset_mode)
        
        intermediate_file = artifact_dir / intermediate_filename
        output_file = artifact_dir / filtered_filename
        
        # Save intermediate file: ALL reviews with all probabilities (including zeros)
        print(f"[INFO] Saving intermediate file (all reviews): {intermediate_filename}")
        with open(intermediate_file, 'w', encoding='utf-8') as f:
            json.dump(all_merged_data, f, indent=2, ensure_ascii=False)
        print(f"[OK] ✓ Saved intermediate (all reviews) to: {intermediate_file}")
        
        # Save final file: Only reviews with at least one theme (regardless of probability)
        print(f"[INFO] Saving filtered file (reviews with themes): {filtered_filename}")
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(filtered_merged_data, f, indent=2, ensure_ascii=False)
        print(f"[OK] ✓ Saved filtered (reviews with themes) to: {output_file}")
        print(f"[INFO] → Proceeding to W&B upload in next step...")
        
        # Remove checkpoint file after successful completion
        if checkpoint_file.exists():
            checkpoint_file.unlink()
            logging.info(f"Removed checkpoint file: {checkpoint_file}")
        
        # Calculate statistics
        # For intermediate (all reviews)
        all_total_reviews = sum(len(user_data.get('reviews', [])) for user_data in all_merged_data.values())
        all_reviews_with_topics = sum(
            sum(1 for review in user_data.get('reviews', [])
                if review.get('themes') and len(review.get('themes', {})) > 0)
            for user_data in all_merged_data.values()
        )
        
        # For filtered (reviews with themes)
        total_reviews = sum(len(user_data.get('reviews', [])) for user_data in filtered_merged_data.values())
        reviews_with_topics = total_reviews  # All remaining reviews have at least one theme
        total_themes_with_scores = sum(
            sum(len(review.get('themes', {})) for review in user_data.get('reviews', [])
                if review.get('themes'))
            for user_data in filtered_merged_data.values()
        )
        
        # =====================================================================
        # Step 5: Upload Artifact to W&B
        # =====================================================================
        print("\n" + "-" * 70)
        print("Step 5: Upload Artifact to W&B")
        print("-" * 70)
        print(f"[INFO] ✓ Local files already saved at: {artifact_dir}")
        print(f"[INFO] → Now uploading to W&B: {merged_output_artifact}")
        print(f"[INFO]   Artifact type: {artifact_type}")
        
        artifact_uploaded = False
        wandb_success = True
        
        artifact_metadata = {
            "num_users_all": len(all_merged_data),
            "num_users_filtered": len(filtered_merged_data),
            "total_reviews_all": all_total_reviews,
            "total_reviews_filtered": total_reviews,
            "original_total_reviews": original_total_reviews,
            "removed_reviews": removed_reviews,
            "reviews_with_topics_all": all_reviews_with_topics,
            "reviews_with_topics_filtered": reviews_with_topics,
            "total_themes_with_scores": total_themes_with_scores,
            "avg_themes_per_review": total_themes_with_scores / reviews_with_topics if reviews_with_topics > 0 else 0,
            "coverage": reviews_with_topics / original_total_reviews if original_total_reviews > 0 else 0,
            "schema_version": "v4",
            "artifact_type": "merged_dataset",
            "filter_by_themes": True,  # Filter based on presence of themes, not probability values
            "remove_reviews_without_themes": True,  # Reviews with no themes are removed from filtered
            "has_intermediate_file": True  # Intermediate file with all reviews is saved
        }
        
        try:
            print(f"[INFO] Creating artifact metadata...")
            print(f"[INFO] Uploading artifact to W&B (this may take a moment)...")
            artifact = log_artifact(
                run=run,
                artifact_name=merged_output_artifact,
                artifact_type=artifact_type,
                artifact_path=artifact_dir,
                metadata=artifact_metadata,
            )
            
            print(f"[INFO] Linking artifact to registry...")
            link_to_registry(artifact, stage=stage_directory)
            artifact_uploaded = True
            print(f"[OK] ✓ Artifact successfully uploaded to W&B: {merged_output_artifact}")
            if run:
                print(f"[INFO]   View artifact in W&B run: {run.url}")
        except Exception as e:
            print(f"[ERROR] ✗ Failed to upload artifact to W&B: {e}")
            print(f"[INFO] ✓ Artifact saved locally at: {artifact_dir}")
            print(f"[INFO]   Local files are available regardless of W&B upload status")
            print(f"[INFO]   You can retry the W&B upload later if needed")
            wandb_success = False
        
        # Log metrics
        try:
            log_metrics(run, {
            "num_users_all": len(all_merged_data),
            "num_users_filtered": len(filtered_merged_data),
            "total_reviews_all": all_total_reviews,
            "total_reviews_filtered": total_reviews,
            "original_total_reviews": original_total_reviews,
            "removed_reviews": removed_reviews,
            "reviews_with_topics_all": all_reviews_with_topics,
            "reviews_with_topics_filtered": reviews_with_topics,
            "total_themes_with_scores": total_themes_with_scores,
                "avg_themes_per_review": artifact_metadata["avg_themes_per_review"],
                "coverage": artifact_metadata["coverage"]
            })
            
            log_summary(run, {
                "status": "completed",
                "users_processed_all": len(all_merged_data),
                "users_processed_filtered": len(filtered_merged_data),
                "reviews_merged_all": all_total_reviews,
                "reviews_merged_filtered": reviews_with_topics
            })
            print("[OK] Metrics logged to W&B")
        except Exception as e:
            print(f"[WARNING] Failed to log metrics to W&B: {e}")
            print("[INFO] Metrics logging failed, but local files are saved")
            wandb_success = False
        
        # =====================================================================
        # Final Summary
        # =====================================================================
        print("\n" + "=" * 70)
        print("STAGE 04 MERGE COMPLETE")
        print("=" * 70)
        print(f"\nSummary:")
        print(f"  Dataset mode: {dataset_mode}")
        print(f"  Original reviews: {original_total_reviews}")
        print(f"  Matched reviews: {len(topic_classifications)}")
        
        print(f"\n  INTERMEDIATE FILE (all reviews):")
        print(f"    File: {intermediate_filename}")
        print(f"    Users: {len(all_merged_data)}")
        print(f"    Total reviews: {all_total_reviews}")
        print(f"    Reviews with topics: {all_reviews_with_topics}")
        
        print(f"\n  FILTERED FILE (reviews with themes):")
        print(f"    File: {filtered_filename}")
        print(f"    Users: {len(filtered_merged_data)}")
        print(f"    Total reviews: {total_reviews}")
        print(f"    Removed reviews (no themes): {removed_reviews}")
        print(f"    Reviews with topics: {reviews_with_topics} ({artifact_metadata['coverage']*100:.1f}% of original)")
        print(f"    Total themes: {total_themes_with_scores}")
        print(f"    Avg themes per review: {artifact_metadata['avg_themes_per_review']:.2f}")
        
        print(f"\n" + "=" * 70)
        print("W&B Upload Status")
        print("=" * 70)
        if wandb_success and artifact_uploaded:
            print(f"[OK] ✓ W&B Upload: SUCCESS")
            print(f"  - Artifact: {merged_output_artifact}")
            print(f"  - Artifact type: {artifact_type}")
            if run:
                print(f"  - View run: {run.url}")
        else:
            print(f"[WARNING] ✗ W&B Upload: FAILED")
            if artifact_uploaded:
                print(f"  - Artifact: {merged_output_artifact} (uploaded but metrics failed)")
            else:
                print(f"  - Artifact: {merged_output_artifact} (upload failed)")
            print(f"\n[INFO] ✓ All outputs are saved locally regardless of W&B status")
            print(f"  Local files are available at: {artifact_dir}")
            print(f"  You can retry the W&B upload later if needed")
        
        print(f"\n[INFO] Local files (always saved):")
        print(f"  - Directory: {artifact_dir}")
        print(f"  - Intermediate file: {intermediate_filename}")
        print(f"    Path: {intermediate_file}")
        print(f"  - Filtered file: {filtered_filename}")
        print(f"    Path: {output_file}")
        
        logging.info("Merge completed successfully!")
        logging.info(f"  Original reviews: {original_total_reviews}")
        logging.info(f"")
        logging.info(f"  INTERMEDIATE FILE (all reviews):")
        logging.info(f"    File: {intermediate_file.name}")
        logging.info(f"    Users: {len(all_merged_data)}")
        logging.info(f"    Total reviews: {all_total_reviews}")
        logging.info(f"    Reviews with topics: {all_reviews_with_topics}")
        logging.info(f"")
        logging.info(f"  FILTERED FILE (reviews with themes):")
        logging.info(f"    File: {output_file.name}")
        logging.info(f"    Users: {len(filtered_merged_data)}")
        logging.info(f"    Total reviews: {total_reviews}")
        logging.info(f"    Removed reviews (no themes present): {removed_reviews}")
        logging.info(f"    Reviews with topics: {reviews_with_topics} ({artifact_metadata['coverage']*100:.1f}% of original)")
        logging.info(f"    Total themes: {total_themes_with_scores}")
        logging.info(f"    Avg themes per review: {artifact_metadata['avg_themes_per_review']:.2f}")
        logging.info(f"")
        logging.info(f"  Note: Intermediate file contains ALL reviews with all probabilities")
        logging.info(f"  Note: Filtered file contains only reviews with at least one theme (regardless of probability)")
        
    except Exception as e:
        logging.error(f"Error in main execution: {e}", exc_info=True)
        log_summary(run, {"status": "failed", "error": str(e)})
        raise
    
    finally:
        finish_run(run)


if __name__ == "__main__":
    main()

