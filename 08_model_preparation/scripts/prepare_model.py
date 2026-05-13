#!/usr/bin/env python3
"""
Stage 08: Model Preparation
============================

Combines micro cluster summaries and details into final SAPIENS persona models.
- Reads configuration from config.yaml
- Downloads SGO training output artifact from W&B
- Combines summary and details for each micro cluster
- Creates final persona JSON files
- Logs final model artifact to W&B

Usage:
    python 08_model_preparation/scripts/prepare_model.py
"""

import json
import sys
import logging
from pathlib import Path

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from utils.wandb_utils import (
    load_config, get_stage_config,
    init_wandb_run, finish_run, use_artifact, log_artifact,
    log_metrics, log_summary, link_to_registry, get_artifact_dir,
    validate_stage_dependencies
)

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def combine_micro_cluster_data(input_path: Path, cluster_id: str, micro_id: str, 
                                summaries_dir: str, details_dir: str, 
                                cluster_prefix: str, summary_pattern: str, details_pattern: str) -> dict:
    """
    Combine summary and details data for a single micro cluster.
    
    Args:
        input_path: Path to the SGO training output directory
        cluster_id: Cluster ID (e.g., "0", "1")
        micro_id: Micro cluster ID (e.g., "0", "1")
        summaries_dir: Directory name for summaries (from config)
        details_dir: Directory name for details (from config)
        cluster_prefix: Prefix for cluster directories (from config)
        summary_pattern: File pattern for summary files (from config)
        details_pattern: File pattern for details files (from config)
        
    Returns:
        Combined persona data dictionary, or None if files not found
    """
    # Paths within the artifact (from config)
    summary_base_dir = input_path / summaries_dir
    details_base_dir = input_path / details_dir
    
    # Format file patterns with micro_id
    summary_filename = summary_pattern.format(micro_id=micro_id)
    details_filename = details_pattern.format(micro_id=micro_id)
    
    summary_path = summary_base_dir / f"{cluster_prefix}{cluster_id}" / summary_filename
    details_path = details_base_dir / f"{cluster_prefix}{cluster_id}" / details_filename
    
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
    
    # Combine into target format
    # Note: cluster_prefix is passed but not used here since macro_cluster_id format is fixed
    combined_data = {
        "persona_name": persona_name,
        "micro_cluster_id": micro_cluster_id,
        "macro_cluster_id": f"{cluster_prefix}{cluster_id}",
        "total_users_in_cluster": total_users_in_cluster,
        "total_reviews_from_cluster": total_reviews_from_cluster,
        "quantitative_summary": quantitative_summary,
        "qualitative_summary": qualitative_summary,
        "justification": justification,
        "key_topics": key_topics,
        "persona_summary": qualitative_summary.get("persona_summary"),
        "key_motivations": qualitative_summary.get("key_motivations", []),
        "common_praises": qualitative_summary.get("common_praises", []),
        "common_criticisms": qualitative_summary.get("common_criticisms", []),
        "core_characteristics": qualitative_summary.get("core_characteristics", []),
        "potential_goals": qualitative_summary.get("potential_goals", []),
        "member_user_characteristics": member_user_characteristics
    }
    
    return combined_data


def main():
    """Main execution function."""
    
    # =========================================================================
    # Step 1: Load configuration from config.yaml
    # =========================================================================
    print("=" * 70)
    print("STAGE 08: Model Preparation")
    print("=" * 70)
    
    cfg = get_stage_config("08_model_preparation")
    
    # Get dataset type (train/test)
    dataset_type = cfg.get("dataset_type", "train")
    if dataset_type not in ["train", "test"]:
        logging.error(f"Invalid dataset_type: {dataset_type}. Must be 'train' or 'test'")
        return
    
    # Get input artifact based on dataset type (REQUIRED - no fallback)
    input_artifacts = cfg.get("input_artifacts", {})
    if dataset_type == "train":
        input_artifact = input_artifacts.get("sgo_training_results_train")
    else:
        input_artifact = input_artifacts.get("sgo_training_results_test")
    
    if not input_artifact:
        logging.error(f"Input artifact for dataset_type '{dataset_type}' not found in config.yaml")
        logging.error("Required: input_artifacts.sgo_training_results_train or input_artifacts.sgo_training_results_test")
        return
    
    # Get output artifact (REQUIRED - no fallback)
    output_artifacts = cfg.get("output_artifacts", {})
    output_artifact_name = output_artifacts.get("sapiens_model")
    if not output_artifact_name:
        logging.error("Output artifact 'sapiens_model' not found in config.yaml")
        logging.error("Required: output_artifacts.sapiens_model")
        return
    
    # Get directory structure from config
    dir_structure = cfg.get("directory_structure", {})
    summaries_dir = dir_structure.get("summaries_dir", "micro_cluster_summaries")
    details_dir = dir_structure.get("details_dir", "micro_cluster_details")
    
    # Get file patterns from config
    file_patterns = cfg.get("file_patterns", {})
    summary_pattern = file_patterns.get("summary_file", "micro_{micro_id}_summary.json")
    details_pattern = file_patterns.get("details_file", "micro_{micro_id}_details.json")
    output_pattern = file_patterns.get("output_file", "micro_{micro_id}.json")
    cluster_prefix = file_patterns.get("cluster_prefix", "cluster_")
    micro_prefix = file_patterns.get("micro_prefix", "micro_")
    
    # Get artifact types to try (from config)
    artifact_types = cfg.get("hyperparameters", {}).get("artifact_types", ["model", "dataset"])
    
    print(f"\n[Config] Dataset Type: {dataset_type}")
    print(f"[Config] Input artifact: {input_artifact}")
    print(f"[Config] Output artifact: {output_artifact_name}")
    print(f"[Config] Summaries directory: {summaries_dir}")
    print(f"[Config] Details directory: {details_dir}")
    
    # =========================================================================
    # Step 2: Initialize W&B run
    # =========================================================================
    print("\n" + "-" * 70)
    print("Step 2: Initialize W&B Run")
    print("-" * 70)
    
    run = init_wandb_run(
        run_name=f"model_preparation_{output_artifact_name}",
        stage="08_model_preparation",
        job_type="model_preparation"
    )
    
    # Validate dependencies (try first artifact type from config)
    required_artifacts = [input_artifact]
    primary_artifact_type = artifact_types[0] if artifact_types else "model"
    if not validate_stage_dependencies(run, "08_model_preparation", required_artifacts, artifact_type=primary_artifact_type):
        logging.error(f"Stage 07 (SGO Training) must be completed first! Required artifact: {input_artifact}")
        return
    
    try:
        # =====================================================================
        # Step 3: Download input artifact
        # =====================================================================
        print("\n" + "-" * 70)
        print("Step 3: Download Input Artifact")
        print("-" * 70)
        
        # Try artifact types from config (in order)
        input_path = None
        for artifact_type in artifact_types:
            logging.info(f"Trying to download artifact as type '{artifact_type}'...")
            input_path = use_artifact(run, input_artifact, artifact_type=artifact_type)
            if input_path is not None:
                logging.info(f"Successfully downloaded artifact as type '{artifact_type}'")
                break
        
        if input_path is None:
            logging.error(f"Could not download input artifact '{input_artifact}' from W&B")
            logging.error(f"Tried artifact types: {artifact_types}")
            logging.error("NO LOCAL FALLBACKS - artifact must be available in W&B")
            return
        
        print(f"[OK] Input artifact downloaded to: {input_path}")
        
        # =====================================================================
        # Step 4: Process all micro clusters
        # =====================================================================
        print("\n" + "-" * 70)
        print("Step 4: Combine Micro Cluster Data")
        print("-" * 70)
        
        # Setup output directory
        output_dir = get_artifact_dir("08_model_preparation", output_artifact_name)
        personas_dir = output_dir / "personas"
        personas_dir.mkdir(parents=True, exist_ok=True)
        
        # Find summary directory (using config path)
        summary_base = input_path / summaries_dir
        if not summary_base.exists():
            logging.error(f"Summary directory '{summaries_dir}' not found in artifact")
            logging.error(f"Searched in: {input_path}")
            logging.error("NO LOCAL FALLBACKS - artifact structure must match config.yaml")
            return
        
        # Get all cluster directories (using config prefix)
        clusters = sorted([d for d in summary_base.iterdir() 
                          if d.is_dir() and d.name.startswith(cluster_prefix)])
        
        if not clusters:
            logging.error("No cluster directories found")
            return
        
        print(f"[OK] Found {len(clusters)} clusters to process")
        
        total_processed = 0
        total_errors = 0
        cluster_stats = {}
        
        for cluster_dir in clusters:
            # Extract cluster_id using config prefix
            cluster_id = cluster_dir.name.replace(cluster_prefix, "")
            logging.info(f"Processing cluster {cluster_id}...")
            
            # Create cluster subdirectory in output (using config prefix)
            cluster_output_dir = personas_dir / f"{cluster_prefix}{cluster_id}"
            cluster_output_dir.mkdir(parents=True, exist_ok=True)
            
            # Get all micro cluster summary files (using config pattern)
            # Pattern: "micro_{micro_id}_summary.json"
            summary_suffix = summary_pattern.replace("{micro_id}", "").replace(micro_prefix, "")
            summary_files = sorted([f for f in cluster_dir.iterdir() 
                                   if f.is_file() and f.name.startswith(micro_prefix) and summary_suffix in f.name])
            
            cluster_processed = 0
            
            for summary_file in summary_files:
                # Extract micro_id from filename using config pattern
                # Pattern: "micro_{micro_id}_summary.json" -> extract micro_id
                # Remove micro_prefix and summary file extension
                base_name = summary_file.name
                if base_name.startswith(micro_prefix):
                    base_name = base_name[len(micro_prefix):]
                # Remove summary file extension (e.g., "_summary.json")
                # pattern_suffix already calculated above (e.g., "_summary")
                if base_name.endswith(summary_suffix + ".json"):
                    micro_id = base_name[:-len(summary_suffix + ".json")]
                else:
                    logging.warning(f"Could not extract micro_id from {summary_file.name} (expected suffix: {summary_suffix}.json), skipping")
                    continue
                
                # Combine data (pass config parameters)
                combined_data = combine_micro_cluster_data(
                    input_path, cluster_id, micro_id,
                    summaries_dir, details_dir,
                    cluster_prefix, summary_pattern, details_pattern
                )
                
                if combined_data is None:
                    total_errors += 1
                    continue
                
                # Save combined persona file (using config pattern)
                output_filename = output_pattern.format(micro_id=micro_id)
                output_path = cluster_output_dir / output_filename
                
                try:
                    with open(output_path, 'w', encoding='utf-8') as f:
                        json.dump(combined_data, f, indent=4, ensure_ascii=False)
                    logging.info(f"  [OK] Created: cluster_{cluster_id}/{output_filename}")
                    total_processed += 1
                    cluster_processed += 1
                except Exception as e:
                    logging.error(f"Error writing {output_path}: {e}")
                    total_errors += 1
            
            cluster_stats[f"{cluster_prefix}{cluster_id}"] = cluster_processed
        
        # =====================================================================
        # Step 5: Create model metadata
        # =====================================================================
        print("\n" + "-" * 70)
        print("Step 5: Create Model Metadata")
        print("-" * 70)
        
        model_metadata = {
            "model_name": "SAPIENS",
            "version": output_artifact_name.split("_")[-1] if "_" in output_artifact_name else "v1",
            "total_personas": total_processed,
            "total_clusters": len(clusters),
            "cluster_stats": cluster_stats,
            "input_artifact": input_artifact,
        }
        
        metadata_path = output_dir / "model_metadata.json"
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(model_metadata, f, indent=4)
        
        print(f"[OK] Model metadata saved to: {metadata_path}")
        
        # =====================================================================
        # Step 6: Log metrics
        # =====================================================================
        print("\n" + "-" * 70)
        print("Step 6: Log Metrics")
        print("-" * 70)
        
        log_metrics(run, {
            "total_personas": total_processed,
            "total_clusters": len(clusters),
            "total_errors": total_errors,
        })
        
        log_summary(run, {
            "final_personas": total_processed,
            "final_clusters": len(clusters),
            "preparation_errors": total_errors,
        })
        
        print(f"[OK] Metrics logged")
        
        # =====================================================================
        # Step 7: Upload artifact to W&B
        # =====================================================================
        print("\n" + "-" * 70)
        print("Step 7: Upload Artifact to W&B")
        print("-" * 70)
        
        artifact = log_artifact(
            run=run,
            artifact_name=output_artifact_name,
            artifact_type="model",
            artifact_path=output_dir,
            metadata={
                "total_personas": total_processed,
                "total_clusters": len(clusters),
                "cluster_stats": cluster_stats,
                "input_artifact": input_artifact,
                "model_type": "sapiens_personas",
            }
        )
        
        # Link to registry
        link_to_registry(artifact, stage="08_model_preparation")
        
        # =====================================================================
        # Final Summary
        # =====================================================================
        print("\n" + "=" * 70)
        print("STAGE 08 COMPLETE")
        print("=" * 70)
        print(f"\nSummary:")
        print(f"  Total personas created: {total_processed}")
        print(f"  Total clusters: {len(clusters)}")
        print(f"  Total errors: {total_errors}")
        print(f"  Output directory: {output_dir}")
        print(f"\nCluster breakdown:")
        for cluster, count in cluster_stats.items():
            print(f"  {cluster}: {count} personas")
        
        if run:
            print(f"\nView run at: {run.url}")
        
    finally:
        finish_run(run)


if __name__ == "__main__":
    main()
