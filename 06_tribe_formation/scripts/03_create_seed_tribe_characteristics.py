#!/usr/bin/env python3
"""
Stage 06: Tribe Formation - Create Seed Tribe Characteristics
============================================================

Combines micro cluster summary and detail data to create consolidated
tribe seed characteristics for each tribe.
- Loads micro cluster summaries and details from user_tribe_v4 artifact
- Combines data for each tribe
- Outputs tribe seed characteristics as learned artifact

Usage:
    python 06_tribe_formation/scripts/03_create_seed_tribe_characteristics.py
"""

import os
import json
import sys
import logging
import re
from pathlib import Path
from typing import Dict, Optional

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from utils.wandb_utils import (
    load_config, get_stage_config,
    init_wandb_run, finish_run, use_artifact, log_artifact,
    log_metrics, log_summary, link_to_registry, get_artifact_dir
)

# Import schema for validation
from schemas.learned_artifacts import TribeSeedCharacteristicsArtifact

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def combine_micro_cluster_data(summary_path: Path, details_path: Path) -> Optional[Dict]:
    """
    Combine summary and details data for a single micro cluster.
    
    Args:
        summary_path: Path to summary JSON file
        details_path: Path to details JSON file
        
    Returns:
        Combined data dictionary or None if error
    """
    # Check if files exist
    if not summary_path.exists():
        logging.warning(f"Summary file not found: {summary_path}")
        return None
    
    if not details_path.exists():
        logging.warning(f"Details file not found: {details_path}")
        return None
    
    # Read summary file
    try:
        with open(summary_path, 'r', encoding='utf-8') as f:
            summary_data = json.load(f)
    except Exception as e:
        logging.error(f"Error reading summary file {summary_path}: {e}")
        return None
    
    # Read details file
    try:
        with open(details_path, 'r', encoding='utf-8') as f:
            details_data = json.load(f)
    except Exception as e:
        logging.error(f"Error reading details file {details_path}: {e}")
        return None
    
    # Extract data from summary
    persona_name = summary_data.get("persona_name")
    qualitative_summary = summary_data.get("qualitative_summary", {})
    quantitative_summary = summary_data.get("quantitative_summary", {})
    micro_cluster_id = summary_data.get("micro_cluster_id")
    total_users_in_cluster = summary_data.get("total_users_in_cluster")
    total_reviews_from_cluster = summary_data.get("total_reviews_from_cluster")
    
    # Extract data from details
    justification = details_data.get("justification")
    member_user_characteristics = details_data.get("member_user_characteristics", [])
    key_topics = details_data.get("key_topics", [])
    members_grouped_by_user = details_data.get("members_grouped_by_user", {})
    
    # Ensure members_grouped_by_user is properly grouped by user_id
    # It should be a dict: {user_id: [list of reviews]}
    if not isinstance(members_grouped_by_user, dict):
        logging.warning(f"members_grouped_by_user is not a dict, converting...")
        members_grouped_by_user = {}
    else:
        # Verify grouping structure
        num_users_grouped = len(members_grouped_by_user)
        total_reviews_grouped = sum(len(reviews) if isinstance(reviews, list) else 0 for reviews in members_grouped_by_user.values())
        logging.debug(f"  Verified grouping: {num_users_grouped} users, {total_reviews_grouped} reviews grouped by user")
    
    # Use micro_cluster_id as tribe_id
    tribe_id = micro_cluster_id if micro_cluster_id else "unknown"
    
    # Combine into target format
    combined_data = {
        "tribe_id": tribe_id,
        "persona_name": persona_name,
        "micro_cluster_id": micro_cluster_id,
        "total_users_in_cluster": total_users_in_cluster,
        "total_reviews_from_cluster": total_reviews_from_cluster,
        "quantitative_summary": quantitative_summary,
        "qualitative_summary": qualitative_summary,
        "justification": justification,
        "key_topics": key_topics,
        "persona_summary": qualitative_summary.get("persona_summary") if isinstance(qualitative_summary, dict) else None,
        "key_motivations": qualitative_summary.get("key_motivations", []) if isinstance(qualitative_summary, dict) else [],
        "common_praises": qualitative_summary.get("common_praises", []) if isinstance(qualitative_summary, dict) else [],
        "common_criticisms": qualitative_summary.get("common_criticisms", []) if isinstance(qualitative_summary, dict) else [],
        "core_characteristics": qualitative_summary.get("core_characteristics", []) if isinstance(qualitative_summary, dict) else [],
        "potential_goals": qualitative_summary.get("potential_goals", []) if isinstance(qualitative_summary, dict) else [],
        "member_user_characteristics": member_user_characteristics,
        "members_grouped_by_user": members_grouped_by_user  # Preserve user grouping
    }
    
    return combined_data


def main():
    """Main function to process all micro clusters."""
    
    # =========================================================================
    # Step 1: Load configuration
    # =========================================================================
    print("=" * 70)
    print("STAGE 06: Tribe Formation - Create Seed Tribe Characteristics")
    print("=" * 70)
    
    config = load_config()
    stage_config = get_stage_config("06_tribe_formation")
    
    # Get artifact names from config (required, no fallbacks)
    if "tribe_seed_input_artifact" not in stage_config:
        logging.error("Missing required config field: tribe_seed_input_artifact")
        return
    
    input_artifact = stage_config["tribe_seed_input_artifact"]
    if "tribe_seed_summaries_dir" not in stage_config:
        logging.error("Missing required config field: tribe_seed_summaries_dir")
        return
    summaries_dir = stage_config["tribe_seed_summaries_dir"]
    
    if "tribe_seed_details_dir" not in stage_config:
        logging.error("Missing required config field: tribe_seed_details_dir")
        return
    details_dir = stage_config["tribe_seed_details_dir"]
    
    if "tribe_seed_output_artifact" not in stage_config:
        logging.error("Missing required config field: tribe_seed_output_artifact")
        return
    
    output_artifact = stage_config["tribe_seed_output_artifact"]
    
    print(f"\n[Config] Input artifact: {input_artifact}")
    print(f"[Config] Output artifact: {output_artifact}")
    
    # =========================================================================
    # Step 2: Initialize W&B run
    # =========================================================================
    run = init_wandb_run(
        run_name="create_seed_tribe_characteristics",
        stage="06_tribe_formation",
        job_type="data_processing",
    )
    
    try:
        # =====================================================================
        # Step 3: Load input artifact from W&B (no local fallback)
        # =====================================================================
        print(f"\n[Step 3] Loading input artifact: {input_artifact}")
        
        # Download input artifact from W&B (required, no local fallback)
        input_path = use_artifact(run, input_artifact, artifact_type="dataset")
        
        if input_path is None:
            logging.error(f"Could not download input artifact: {input_artifact}")
            logging.error(f"Make sure to run 06_tribe_formation/scripts/02_user_to_tribe_via_segments_mapper.py first and upload the artifact to W&B")
            return
        
        # Resolve path to handle any symlinks or relative paths
        # W&B may return path with :v0, :v1, etc. (invalid on Linux), need to handle this
        input_path_str = str(input_path)
        if not Path(input_path).exists():
            # Replace :v0, :v1, :v2, etc. with -v0, -v1, -v2, etc.
            input_path_str = re.sub(r':(v\d+)', r'-\1', input_path_str)
            input_path = Path(input_path_str)
        
        input_path = Path(input_path).resolve()
        logging.info(f"[W&B] Input artifact downloaded to: {input_path}")
        
        # Find summary and details directories using names from config
        summary_base = input_path / summaries_dir
        details_base = input_path / details_dir
        
        if not summary_base.exists():
            logging.error(f"Summary directory not found: {summary_base}")
            return
        
        if not details_base.exists():
            logging.error(f"Details directory not found: {details_base}")
            return
        
        logging.info(f"✅ Found summary directory: {summary_base}")
        logging.info(f"✅ Found details directory: {details_base}")
        
        # =====================================================================
        # Step 4: Process all micro clusters
        # =====================================================================
        print(f"\n[Step 4] Processing micro clusters...")
        
        # Get all cluster directories
        clusters = sorted([d for d in summary_base.iterdir() if d.is_dir() and d.name.startswith("cluster_")])
        
        if not clusters:
            logging.warning("No cluster directories found")
            return
        
        logging.info(f"Found {len(clusters)} cluster directories to process")
        
        all_tribe_characteristics = {}  # tribe_id -> TribeSeedCharacteristicsArtifact
        total_processed = 0
        total_errors = 0
        
        for cluster_dir in clusters:
            cluster_id = cluster_dir.name.replace("cluster_", "")
            logging.info(f"Processing cluster {cluster_id}...")
            
            # Get all micro cluster summary files
            summary_files = sorted([
                f for f in cluster_dir.iterdir() 
                if f.name.startswith("micro_") and f.name.endswith("_summary.json")
            ])
            
            if not summary_files:
                logging.warning(f"No summary files found in {cluster_dir}")
                continue
            
            for summary_file in summary_files:
                # Extract micro_id from filename (e.g., "micro_0_summary.json" -> "0")
                micro_id = summary_file.name.replace("micro_", "").replace("_summary.json", "")
                
                # Construct corresponding details file path
                details_dir = details_base / cluster_dir.name
                details_file = details_dir / f"micro_{micro_id}_details.json"
                
                # Combine data
                combined_data = combine_micro_cluster_data(summary_file, details_file)
                
                if combined_data is None:
                    total_errors += 1
                    continue
                
                # Validate with schema
                try:
                    tribe_artifact = TribeSeedCharacteristicsArtifact.from_dict(combined_data)
                    
                    # Verify user grouping is preserved
                    if tribe_artifact.members_grouped_by_user:
                        num_users = len(tribe_artifact.members_grouped_by_user)
                        num_reviews = sum(len(reviews) if isinstance(reviews, list) else 0 
                                        for reviews in tribe_artifact.members_grouped_by_user.values())
                        logging.debug(f"    Grouped: {num_users} users, {num_reviews} reviews")
                    
                    all_tribe_characteristics[tribe_artifact.tribe_id] = tribe_artifact
                    total_processed += 1
                    logging.info(f"  ✓ Processed: {tribe_artifact.tribe_id} ({tribe_artifact.persona_name})")
                except Exception as e:
                    logging.error(f"Validation error for tribe {combined_data.get('tribe_id', 'unknown')}: {e}")
                    total_errors += 1
                    continue
        
        # Calculate grouping statistics
        total_users_grouped = 0
        total_reviews_grouped = 0
        for tribe in all_tribe_characteristics.values():
            if tribe.members_grouped_by_user:
                total_users_grouped += len(tribe.members_grouped_by_user)
                total_reviews_grouped += sum(
                    len(reviews) if isinstance(reviews, list) else 0 
                    for reviews in tribe.members_grouped_by_user.values()
                )
        
        logging.info(f"\n[OK] Processed {total_processed} tribes, {total_errors} errors")
        logging.info(f"[OK] User grouping: {total_users_grouped} users with {total_reviews_grouped} reviews grouped by user")
        
        # =====================================================================
        # Step 5: Save output artifact
        # =====================================================================
        print(f"\n[Step 5] Saving output artifact: {output_artifact}")
        
        artifact_dir = get_artifact_dir("06_tribe_formation", output_artifact)
        artifact_dir.mkdir(parents=True, exist_ok=True)
        
        # Save as JSON with tribe_id as keys
        output_file = artifact_dir / "tribe_seed_characteristics.json"
        output_data = {
            tribe_id: tribe.to_dict()
            for tribe_id, tribe in all_tribe_characteristics.items()
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        logging.info(f"[OK] Saved {len(all_tribe_characteristics)} tribe characteristics to: {output_file}")
        
        # Also save per-cluster files for backward compatibility
        cluster_output_dir = artifact_dir / "micro_cluster_personas"
        cluster_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Group by cluster
        tribes_by_cluster = {}
        for tribe_id, tribe in all_tribe_characteristics.items():
            # Extract cluster from tribe_id (e.g., "cluster_0_micro_5" -> "cluster_0")
            if "_micro_" in tribe_id:
                cluster_name = tribe_id.split("_micro_")[0]
            elif tribe.micro_cluster_id and "_micro_" in tribe.micro_cluster_id:
                cluster_name = tribe.micro_cluster_id.split("_micro_")[0]
            else:
                cluster_name = "unknown"
            
            if cluster_name not in tribes_by_cluster:
                tribes_by_cluster[cluster_name] = []
            tribes_by_cluster[cluster_name].append(tribe)
        
        # Save per-cluster files
        for cluster_name, tribes in tribes_by_cluster.items():
            cluster_dir = cluster_output_dir / cluster_name
            cluster_dir.mkdir(parents=True, exist_ok=True)
            
            for tribe in tribes:
                # Extract micro_id from tribe_id
                if "_micro_" in tribe.tribe_id:
                    micro_id = tribe.tribe_id.split("_micro_")[-1]
                elif tribe.micro_cluster_id and "_micro_" in tribe.micro_cluster_id:
                    micro_id = tribe.micro_cluster_id.split("_micro_")[-1]
                else:
                    micro_id = "unknown"
                
                output_filename = f"micro_{micro_id}.json"
                output_path = cluster_dir / output_filename
                
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(tribe.to_dict(), f, indent=4, ensure_ascii=False)
        
        logging.info(f"[OK] Saved per-cluster files to: {cluster_output_dir}")
        
        # =====================================================================
        # Step 6: Log to W&B
        # =====================================================================
        logging.info("Logging artifact to W&B...")
        
        artifact_metadata = {
            "num_tribes": len(all_tribe_characteristics),
            "num_clusters": len(tribes_by_cluster),
            "total_processed": total_processed,
            "total_errors": total_errors,
            "schema_version": "v4",
            "schema_validated": True,
            "artifact_type": "learned_artifact"
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
            "num_tribes": len(all_tribe_characteristics),
            "num_clusters": len(tribes_by_cluster),
            "total_processed": total_processed,
            "total_errors": total_errors
        })
        
        log_summary(run, {
            "status": "completed",
            "tribes_processed": total_processed,
            "tribes_created": len(all_tribe_characteristics)
        })
        
        print("\n" + "=" * 70)
        print("[OK] Stage 06 Script 03 Complete!")
        print("=" * 70)
        print(f"\nOutput artifact: {output_artifact}")
        print(f"Total tribes: {len(all_tribe_characteristics)}")
        if run:
            print(f"\nView run: {run.url}")
        
    except Exception as e:
        logging.error(f"Error in main execution: {e}", exc_info=True)
        log_summary(run, {"status": "failed", "error": str(e)})
        raise
    
    finally:
        finish_run(run)


if __name__ == "__main__":
    main()
