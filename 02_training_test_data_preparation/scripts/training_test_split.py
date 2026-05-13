#!/usr/bin/env python3
"""
Stage 02: Training/Test Data Preparation
========================================

Creates train/test split from input dataset.
- Reads configuration from config.yaml
- Downloads input artifact from W&B
- Creates train/test splits based on config parameters
- Logs metrics during processing
- Uploads output artifacts to W&B

TEST SET = GROUND TRUTH for evaluation

Usage:
    python 02_training_test_data_preparation/scripts/training_test_split.py
"""

import json
import math
import random
import sys
from pathlib import Path

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from utils.wandb_utils import (
    load_config, get_stage_config, get_project_root,
    init_wandb_run, finish_run, use_artifact, log_artifact,
    log_metrics, log_summary, link_to_registry, get_artifact_dir,
    validate_stage_dependencies, create_comprehensive_artifact_metadata,
    get_learned_artifact_schema
)


def load_review_data(filepath: Path) -> dict:
    """Loads the aggregated user review data from the JSON file."""
    if not filepath.exists():
        print(f"[ERROR] Input file not found at '{filepath}'")
        return None
    
    print(f"Loading review data from '{filepath}'...")
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        print(f"[ERROR] Error decoding JSON file: {e}")
        return None


def main():
    """Main execution function."""
    
    # =========================================================================
    # Step 1: Load configuration from config.yaml
    # =========================================================================
    print("=" * 70)
    print("STAGE 02: Training/Test Data Preparation")
    print("=" * 70)
    
    # Load stage-specific config
    cfg = get_stage_config("02_train_test_split")
    
    # Get stage directory from config
    stage_directory = cfg.get("stage_directory")
    if not stage_directory:
        raise ValueError("stage_directory must be specified in config.yaml")
    
    # Get parameters from config (no defaults - all must be in config)
    input_artifact = cfg.get("input_artifact")
    if not input_artifact:
        raise ValueError("input_artifact must be specified in config.yaml")
    
    # Handle both old format (output_artifact_train/test) and new format (output_artifacts)
    output_artifacts = cfg.get("output_artifacts", {})
    if output_artifacts:
        output_artifact_train = output_artifacts.get("train")
        output_artifact_test = output_artifacts.get("test")
    else:
        output_artifact_train = cfg.get("output_artifact_train")
        output_artifact_test = cfg.get("output_artifact_test")
    
    if not output_artifact_train or not output_artifact_test:
        raise ValueError("output_artifacts.train and output_artifacts.test must be specified in config.yaml")
    
    # Get hyperparameters
    hyperparams = cfg.get("hyperparameters", {})
    train_ratio = hyperparams.get("train_ratio")
    test_ratio = hyperparams.get("test_ratio")
    min_reviews_for_test = hyperparams.get("min_reviews_for_test")
    users_with_2_reviews_pct = hyperparams.get("users_with_2_reviews_percent")
    users_with_1_review_pct = hyperparams.get("users_with_1_review_percent")
    
    if train_ratio is None or test_ratio is None or min_reviews_for_test is None:
        raise ValueError("hyperparameters.train_ratio, test_ratio, and min_reviews_for_test must be specified in config.yaml")
    
    # Get paths configuration
    paths_config = cfg.get("paths")
    if not paths_config:
        raise ValueError("paths must be specified in config.yaml")
    
    train_output_filename = paths_config.get("train_output_filename")
    test_output_filename = paths_config.get("test_output_filename")
    input_json_filename = paths_config.get("input_json_filename")
    
    if not train_output_filename or not test_output_filename:
        raise ValueError("paths.train_output_filename and paths.test_output_filename must be specified in config.yaml")
    
    if not input_json_filename:
        raise ValueError("paths.input_json_filename must be specified in config.yaml")
    
    # Get artifact type from config
    artifact_type = cfg.get("artifact_type")
    if not artifact_type:
        raise ValueError("artifact_type must be specified in config.yaml")
    
    # Get job_type from config
    job_type = cfg.get("job_type")
    if not job_type:
        raise ValueError("job_type must be specified in config.yaml")
    
    print(f"\n[Config] Input artifact: {input_artifact}")
    print(f"[Config] Train ratio: {train_ratio}")
    print(f"[Config] Test ratio: {test_ratio}")
    print(f"[Config] Min reviews for test extraction: {min_reviews_for_test}")
    
    # =========================================================================
    # Step 2: Initialize W&B run with config
    # =========================================================================
    run = init_wandb_run(
        run_name=f"train_test_split_{output_artifact_train}",
        stage=stage_directory,
        job_type=job_type,
        # Config from YAML is automatically loaded by init_wandb_run
    )
    
    try:
        # =====================================================================
        # Step 3: Validate stage dependencies (sequential execution)
        # =====================================================================
        print(f"\n[Step 1] Validating stage dependencies...")
        required_artifacts = [input_artifact] if input_artifact else []
        
        if not validate_stage_dependencies(run, stage_directory, required_artifacts):
            print("[ERROR] Stage 01 must be completed first!")
            print("[ERROR] Please run Stage 01 to create required artifacts.")
            return
        
        # =====================================================================
        # Step 4: Download input artifact from W&B (ONLY - no local fallback)
        # =====================================================================
        print(f"\n[Step 2] Downloading input artifact from W&B: {input_artifact}")
        
        # Download artifact from W&B (no local fallback)
        input_path = use_artifact(run, input_artifact, artifact_type=artifact_type)
        
        if input_path is None:
            print("[ERROR] Could not download input artifact from W&B")
            print(f"[ERROR] Make sure artifact '{input_artifact}' exists in W&B")
            return
        
        # Find the JSON file in the downloaded artifact (exact filename required)
        input_file = input_path / input_json_filename
        
        if not input_file.exists():
            print(f"[ERROR] Input JSON file not found in artifact: {input_file}")
            print(f"[ERROR] Expected filename: {input_json_filename}")
            print(f"[ERROR] Artifact path: {input_path}")
            return
        
        print(f"[OK] Using input file from W&B artifact: {input_file}")
        
        # =====================================================================
        # Step 4: Load and process data
        # =====================================================================
        print(f"\n[Step 2] Loading data...")
        
        all_user_data = load_review_data(input_file)
        if all_user_data is None:
            return
        
        total_users = len(all_user_data)
        total_reviews = sum(len(u.get('reviews', [])) for u in all_user_data.values())
        
        print(f"[OK] Loaded {total_users:,} users with {total_reviews:,} reviews")
        
        # Log initial metrics
        log_metrics(run, {
            "input/total_users": total_users,
            "input/total_reviews": total_reviews,
        })
        
        # =====================================================================
        # Step 5: Create train/test split (Review-level split per user)
        # =====================================================================
        print(f"\n[Step 3] Creating train/test split...")
        print(f"  Split method: Review-level split (same users in both sets)")
        print(f"  Target test ratio: {test_ratio:.0%} of reviews per user")
        print(f"  Min reviews for test extraction: {min_reviews_for_test}")
        
        test_set_data = {}
        train_set_data = {}
        
        # Statistics
        users_with_test_reviews = 0
        users_all_train = 0
        total_test_reviews = 0
        total_train_reviews = 0
        
        # Process each user: split their reviews 80/20
        for user_id, user_data in all_user_data.items():
            reviews = user_data.get('reviews', [])
            num_reviews = len(reviews)
            
            if num_reviews < min_reviews_for_test:
                # User has too few reviews - all go to train only
                train_set_data[user_id] = {'reviews': reviews}
                total_train_reviews += num_reviews
                users_all_train += 1
                continue
            
            # Calculate how many reviews to extract for test (20% of user's reviews)
            test_reviews_count = max(1, int(num_reviews * test_ratio))
            # Ensure at least 1 review for test if user has enough reviews
            if num_reviews >= min_reviews_for_test:
                test_reviews_count = max(1, test_reviews_count)
            
            # Randomly sample reviews for test set
            # Shuffle reviews to ensure random selection
            shuffled_reviews = reviews.copy()
            random.shuffle(shuffled_reviews)
            
            # Split: test gets test_ratio, train gets the rest
            test_reviews = shuffled_reviews[:test_reviews_count]
            train_reviews = shuffled_reviews[test_reviews_count:]
            
            # Add to both sets (same user in both)
            if len(test_reviews) > 0:
                test_set_data[user_id] = {'reviews': test_reviews}
                users_with_test_reviews += 1
                total_test_reviews += len(test_reviews)
            
            if len(train_reviews) > 0:
                train_set_data[user_id] = {'reviews': train_reviews}
                total_train_reviews += len(train_reviews)
        
        # Calculate final counts and ratios
        train_users_count = len(train_set_data)
        test_users_count = len(test_set_data)
        train_reviews_count = total_train_reviews
        test_reviews_count = total_test_reviews
        actual_test_ratio = total_test_reviews / total_reviews if total_reviews > 0 else 0
        actual_train_ratio = total_train_reviews / total_reviews if total_reviews > 0 else 0
        
        print(f"\n[Step 4] Split complete:")
        print(f"  Train: {train_users_count:,} users, {total_train_reviews:,} reviews ({actual_train_ratio:.1%})")
        print(f"  Test:  {test_users_count:,} users, {total_test_reviews:,} reviews ({actual_test_ratio:.1%})")
        print(f"  Users with test reviews: {users_with_test_reviews:,}")
        print(f"  Users with only train reviews: {users_all_train:,}")
        print(f"  Note: Same users appear in both train and test sets (review-level split)")
        
        # =====================================================================
        # Step 6: Save outputs locally
        # =====================================================================
        print(f"\n[Step 5] Saving outputs...")
        
        # Create artifact directories
        train_dir = get_artifact_dir(stage_directory, output_artifact_train)
        test_dir = get_artifact_dir(stage_directory, output_artifact_test)
        
        train_file = train_dir / train_output_filename
        test_file = test_dir / test_output_filename
        
        with open(train_file, 'w', encoding='utf-8') as f:
            json.dump(train_set_data, f, indent=2)
        print(f"  [OK] Saved: {train_file}")
        
        with open(test_file, 'w', encoding='utf-8') as f:
            json.dump(test_set_data, f, indent=2)
        print(f"  [OK] Saved: {test_file}")
        
        # =====================================================================
        # Step 7: Log final metrics and upload to W&B (with error handling)
        # =====================================================================
        final_metrics = {
            "output/train_users": train_users_count,
            "output/train_reviews": train_reviews_count,
            "output/train_ratio": actual_train_ratio,
            "output/test_users": test_users_count,
            "output/test_reviews": test_reviews_count,
            "output/test_ratio": actual_test_ratio,
            "output/users_with_test_reviews": users_with_test_reviews,
            "output/users_all_train": users_all_train,
        }
        
        # Try to log metrics to W&B (non-critical)
        wandb_success = True
        try:
            log_metrics(run, final_metrics)
            log_summary(run, final_metrics)
        except Exception as e:
            print(f"[WARNING] Failed to log metrics to W&B: {e}")
            print("[INFO] Metrics logging failed, but local files are saved")
            wandb_success = False
        
        # =====================================================================
        # Step 8: Upload artifacts to W&B (with error handling)
        # =====================================================================
        print(f"\n[Step 6] Uploading artifacts to W&B...")
        
        train_artifact_uploaded = False
        test_artifact_uploaded = False
        
        # Upload train artifact with comprehensive metadata
        try:
            train_metadata = create_comprehensive_artifact_metadata(
                stage=stage_directory,
                artifact_name=output_artifact_train,
                sample_size=train_users_count,
                learned_artifact_schema=get_learned_artifact_schema(stage_directory, output_artifact_train),
                additional_metadata={
                    "split": "train",
                    "num_users": train_users_count,
                    "num_reviews": train_reviews_count,
                    "ratio": actual_train_ratio,
                }
            )
            train_artifact = log_artifact(
                run=run,
                artifact_name=output_artifact_train,
                artifact_type=artifact_type,
                artifact_path=train_dir,
                metadata=train_metadata
            )
            link_to_registry(train_artifact, stage=stage_directory)
            train_artifact_uploaded = True
            print(f"  [OK] Train artifact uploaded to W&B: {output_artifact_train}")
        except Exception as e:
            print(f"  [ERROR] Failed to upload train artifact to W&B: {e}")
            print(f"  [INFO] Train artifact saved locally at: {train_file}")
            wandb_success = False
        
        # Upload test artifact (GROUND TRUTH) with comprehensive metadata
        try:
            test_metadata = create_comprehensive_artifact_metadata(
                stage=stage_directory,
                artifact_name=output_artifact_test,
                sample_size=test_users_count,
                learned_artifact_schema=get_learned_artifact_schema(stage_directory, output_artifact_test),
                additional_metadata={
                    "split": "test",
                    "is_ground_truth": True,
                    "num_users": test_users_count,
                    "num_reviews": test_reviews_count,
                    "ratio": actual_test_ratio,
                }
            )
            test_artifact = log_artifact(
                run=run,
                artifact_name=output_artifact_test,
                artifact_type=artifact_type,
                artifact_path=test_dir,
                metadata=test_metadata
            )
            link_to_registry(test_artifact, stage=stage_directory)
            test_artifact_uploaded = True
            print(f"  [OK] Test artifact uploaded to W&B: {output_artifact_test}")
        except Exception as e:
            print(f"  [ERROR] Failed to upload test artifact to W&B: {e}")
            print(f"  [INFO] Test artifact saved locally at: {test_file}")
            wandb_success = False
        
        # =====================================================================
        # Done!
        # =====================================================================
        print("\n" + "=" * 70)
        print("[OK] Stage 02 Complete!")
        print("=" * 70)
        print(f"\nOutput artifacts (LOCAL - always saved):")
        print(f"  - Train: {train_file}")
        print(f"    {train_reviews_count:,} reviews, {train_users_count:,} users")
        print(f"  - Test:  {test_file}")
        print(f"    {test_reviews_count:,} reviews, {test_users_count:,} users")
        
        if wandb_success and train_artifact_uploaded and test_artifact_uploaded:
            print(f"\n[OK] W&B Upload: SUCCESS")
            print(f"  - Train artifact: {output_artifact_train}")
            print(f"  - Test artifact: {output_artifact_test}")
            if run:
                print(f"\nView run: {run.url}")
        else:
            print(f"\n[WARNING] W&B Upload: PARTIAL/FAILED")
            if train_artifact_uploaded:
                print(f"  - Train artifact: {output_artifact_train} ✓")
            else:
                print(f"  - Train artifact: FAILED (saved locally)")
            if test_artifact_uploaded:
                print(f"  - Test artifact: {output_artifact_test} ✓")
            else:
                print(f"  - Test artifact: FAILED (saved locally)")
            print(f"\n[INFO] All outputs are saved locally regardless of W&B status")
            print(f"  Local files are available at:")
            print(f"    - {train_file}")
            print(f"    - {test_file}")
        
    finally:
        finish_run(run)


if __name__ == "__main__":
    main()
