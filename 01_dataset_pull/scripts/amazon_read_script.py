#!/usr/bin/env python3
"""
Stage 01: Dataset Pull
======================

Pull Amazon reviews data and save to artifacts folder.
- Reads configuration from config.yaml
- Processes categories specified in config
- Logs artifact to W&B: Amazon Full Data collection

Usage:
    python 01_dataset_pull/scripts/amazon_read_script.py
"""

import json
import os
import sys
from pathlib import Path

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from utils.wandb_utils import (
    load_config, get_stage_config,
    init_wandb_run, finish_run, log_artifact,
    log_metrics, log_summary, link_to_registry, get_artifact_dir
)

# Import local utilities
sys.path.insert(0, str(Path(__file__).parent.parent))
from utlis.file_io import load_existing_database
from utlis.data_loaders import sample_from_existing_data
from utlis.category_process import process_category
from utlis.config_loader import load_category_config, print_config_summary, calculate_category_stats




def main():
    """Main execution function."""
    
    # =========================================================================
    # Step 1: Load configuration from config.yaml
    # =========================================================================
    print("=" * 70)
    print("STAGE 01: Dataset Pull")
    print("=" * 70)
    
    cfg = get_stage_config("01_dataset_pull")
    config_data = load_category_config(cfg)
    
    # Validate required config fields
    if "output_artifact" not in cfg:
        raise ValueError("[ERROR] 'output_artifact' is required in config.yaml")
    output_artifact_name = cfg["output_artifact"]
    
    if "hyperparameters" not in cfg:
        raise ValueError("[ERROR] 'hyperparameters' section is required in config.yaml")
    hyperparams = cfg["hyperparameters"]
    
    if "output_filename" not in hyperparams:
        raise ValueError("[ERROR] 'hyperparameters.output_filename' is required in config.yaml")
    output_filename = hyperparams["output_filename"]
    
    # Get S3 configuration
    s3_config = hyperparams.get("s3", {})
    if s3_config.get("enabled", False):
        if "bucket" not in s3_config or not s3_config["bucket"]:
            raise ValueError("[ERROR] 'hyperparameters.s3.bucket' is required when S3 is enabled")
        if "region" not in s3_config:
            raise ValueError("[ERROR] 'hyperparameters.s3.region' is required when S3 is enabled")
        print(f"[INFO] S3 enabled: bucket={s3_config['bucket']}, region={s3_config['region']}")
    
    categories = config_data["categories"]
    category_settings = config_data["category_settings"]
    sampling_enabled = config_data["sampling_enabled"]
    sampling_config = config_data["sampling_config"]
    
    # User lists and source data dir are only needed when sampling is disabled
    if not sampling_enabled:
        if "user_lists_dir" not in hyperparams:
            raise ValueError("[ERROR] 'hyperparameters.user_lists_dir' is required in config.yaml when sampling is disabled")
        if "source_data_dir" not in hyperparams:
            raise ValueError("[ERROR] 'hyperparameters.source_data_dir' is required in config.yaml when sampling is disabled")
        user_lists_dir = hyperparams["user_lists_dir"]
        source_data_dir = hyperparams["source_data_dir"]
    else:
        # These are not used when sampling is enabled, but set to None to avoid errors
        user_lists_dir = None
        source_data_dir = None
    
    print_config_summary(config_data, output_artifact_name)
    
    artifact_dir = get_artifact_dir("01_dataset_pull", output_artifact_name)
    output_file = artifact_dir / output_filename
    
    # Only check for user_lists_dir if sampling is disabled and S3 is disabled
    if not sampling_enabled and not s3_config.get("enabled", False):
        if not os.path.exists(user_lists_dir):
            print(f"[ERROR] Directory '{user_lists_dir}' not found.")
            return
    print("\n" + "-" * 70 + "\nStep 2: Initialize W&B Run\n" + "-" * 70)
    
    run = init_wandb_run(
        run_name=f"dataset_pull_{output_artifact_name}",
        stage="01_dataset_pull",
        job_type="data_pull"
    )
    
    try:
        if sampling_enabled:
            if "source_file" not in sampling_config:
                raise ValueError("[ERROR] 'hyperparameters.sampling.source_file' is required in config.yaml when sampling is enabled")
            source_file = sampling_config["source_file"]
            
            if "total_users" not in sampling_config:
                raise ValueError("[ERROR] 'hyperparameters.sampling.total_users' is required in config.yaml when sampling is enabled")
            target_users = sampling_config["total_users"]
            
            if "total_reviews" not in sampling_config:
                raise ValueError("[ERROR] 'hyperparameters.sampling.total_reviews' is required in config.yaml when sampling is enabled")
            target_reviews = sampling_config["total_reviews"]
            
            if "maintain_category_ratios" not in sampling_config:
                raise ValueError("[ERROR] 'hyperparameters.sampling.maintain_category_ratios' is required in config.yaml when sampling is enabled")
            maintain_ratios = sampling_config["maintain_category_ratios"]
            
            if "balanced" not in sampling_config:
                raise ValueError("[ERROR] 'hyperparameters.sampling.balanced' is required in config.yaml when sampling is enabled")
            balanced = sampling_config["balanced"]
            
            if "use_all_categories" not in sampling_config:
                raise ValueError("[ERROR] 'hyperparameters.sampling.use_all_categories' is required in config.yaml when sampling is enabled")
            use_all_categories = sampling_config["use_all_categories"]
            master_db = sample_from_existing_data(
                source_file=source_file,
                target_users=target_users,
                target_reviews=target_reviews,
                enabled_categories=categories,
                maintain_ratios=maintain_ratios,
                balanced=balanced,
                use_all_categories=use_all_categories,
                s3_config=s3_config
            )
            
            total_new_reviews = sum(len(u.get('reviews', [])) for u in master_db.values())
            total_duplicates = 0
            category_stats = calculate_category_stats(master_db)
            for category, stats in category_stats.items():
                log_metrics(run, {f"category/{category}/new_reviews": stats["new_reviews"], f"category/{category}/duplicates": 0})
        else:
            print("\n" + "-" * 70 + "\nStep 3: Load Existing Data\n" + "-" * 70)
            master_db, existing_keys = load_existing_database(output_file)
            print("\n" + "-" * 70 + "\nStep 4: Process Categories\n" + "-" * 70)
            print(f"Processing {len(categories)} categories...")
            total_new_reviews = total_duplicates = 0
            category_stats = {}
            for category in categories:
                cat_settings = category_settings.get(category, {})
                new_reviews, duplicates = process_category(
                    category, master_db, existing_keys, user_lists_dir, source_data_dir,
                    max_users=cat_settings.get("max_users"), max_reviews_per_user=cat_settings.get("max_reviews_per_user"),
                    s3_config=s3_config
                )
                total_new_reviews += new_reviews
                total_duplicates += duplicates
                category_stats[category] = {"new_reviews": new_reviews, "duplicates_skipped": duplicates}
                log_metrics(run, {f"category/{category}/new_reviews": new_reviews, f"category/{category}/duplicates": duplicates})
        print("\n" + "-" * 70 + "\nStep 5: Save Database\n" + "-" * 70)
        print(f"Total Unique Users: {len(master_db)}")
        total_reviews = sum(len(u.get('reviews', [])) for u in master_db.values())
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(master_db, f, indent=2)
        print(f"[OK] Database saved to {output_file}")
        print("\n" + "-" * 70 + "\nStep 6: Log Metrics\n" + "-" * 70)
        
        log_metrics(run, {
            "total_users": len(master_db),
            "total_reviews": total_reviews,
            "new_reviews_this_run": total_new_reviews,
            "duplicates_skipped": total_duplicates,
        })
        
        log_summary(run, {
            "final_users": len(master_db),
            "final_reviews": total_reviews,
            "categories_processed": len(categories),
        })
        
        print(f"[OK] Metrics logged")
        print("\n" + "-" * 70 + "\nStep 7: Upload Artifact to W&B\n" + "-" * 70)
        
        artifact = log_artifact(
            run=run,
            artifact_name=output_artifact_name,
            artifact_type="dataset",
            artifact_path=artifact_dir,
            metadata={
                "num_users": len(master_db),
                "num_reviews": total_reviews,
                "categories": categories,
                "category_stats": category_stats,
                "columns": ["product_description", "review_text", "rating", "category", "timestamp", "asin"]
            }
        )
        
        link_to_registry(artifact, stage="01_dataset_pull")
        print("\n" + "=" * 70 + "\nSTAGE 01 COMPLETE\n" + "=" * 70)
        print(f"\nSummary:\n  Total users: {len(master_db)}\n  Total reviews: {total_reviews}\n  New reviews this run: {total_new_reviews}\n  Duplicates skipped: {total_duplicates}\n  Categories processed: {len(categories)}\n  Output: {output_file}")
        if run:
            print(f"\nView run at: {run.url}")
        
    finally:
        finish_run(run)


if __name__ == "__main__":
    main()
