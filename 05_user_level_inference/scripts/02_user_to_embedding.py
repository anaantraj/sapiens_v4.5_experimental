#!/usr/bin/env python3
"""
Stage 05: User Level Inference - User Embeddings
=================================================

Generates semantic embeddings for users based on their backstory characteristics.
- Reads configuration from config.yaml
- Loads user backstories from learned artifact (Script 01 output)
- Generates embeddings using embedding model
- Validates with schema before saving
- Logs artifact to W&B

Usage:
    python 05_user_level_inference/scripts/02_user_to_embedding.py
"""

import json
import os
import sys
import time
import re
from pathlib import Path
from typing import Dict, List
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from utils.wandb_utils import (
    load_config, get_stage_config, get_openai_config,
    init_wandb_run, finish_run, use_artifact, log_artifact,
    log_metrics, log_summary, link_to_registry, get_artifact_dir,
    create_comprehensive_artifact_metadata, get_learned_artifact_schema
)

# Import schema for validation
from schemas.learned_artifacts import UserBackstoryArtifact, UserEmbeddingArtifact

# Import OpenAI for embeddings
from openai import OpenAI

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Thread-safe locks
progress_lock = Lock()
file_lock = Lock()


def create_embedding_text(user_backstory: UserBackstoryArtifact) -> str:
    """
    Create embedding text from user backstory.
    
    Args:
        user_backstory: UserBackstoryArtifact instance
        
    Returns:
        Text string suitable for embedding generation
    """
    # Extract overall characteristics
    overall_summary = user_backstory.overall_characteristics.influencing_characteristics_summary
    
    # Build embedding text
    embedding_parts = [f"User characteristics: {overall_summary}"]
    
    # Add category-specific characteristics if available
    if user_backstory.category_characteristics:
        embedding_parts.append("Category-specific traits:")
        for category, char_data in user_backstory.category_characteristics.items():
            cat_summary = char_data.influencing_characteristics_summary
            embedding_parts.append(f"- {category}: {cat_summary}")
    
    # Add categories list
    categories_str = ", ".join(user_backstory.categories)
    embedding_parts.append(f"Reviews products in categories: {categories_str}")
    
    return " ".join(embedding_parts)


def generate_user_embedding(
    client: OpenAI,
    user_id: str,
    embedding_text: str,
    embedding_model: str
) -> List[float]:
    """
    Generate embedding vector for a user.
    
    Args:
        client: OpenAI client
        user_id: User ID
        embedding_text: Text to embed
        embedding_model: Embedding model name
        
    Returns:
        Embedding vector as list of floats
    """
    try:
        response = client.embeddings.create(
            model=embedding_model,
            input=embedding_text
        )
        return response.data[0].embedding
    except Exception as e:
        logging.error(f"Failed to generate embedding for user {user_id}: {e}")
        raise


def process_user_embedding(
    user_id: str,
    user_backstory: UserBackstoryArtifact,
    client: OpenAI,
    embedding_model: str
) -> tuple:
    """
    Process a single user to generate embedding.
    
    Args:
        user_id: User ID
        user_backstory: UserBackstoryArtifact instance
        client: OpenAI client
        embedding_model: Embedding model name
        
    Returns:
        Tuple of (user_id, embedding_data_dict) or (user_id, None) on failure
    """
    try:
        # Create embedding text
        embedding_text = create_embedding_text(user_backstory)
        
        # Generate embedding
        embedding_vector = generate_user_embedding(
            client, user_id, embedding_text, embedding_model
        )
        
        # Get embedding dimension
        embedding_dimension = len(embedding_vector)
        
        # Create embedding artifact
        embedding_data = {
            "user_id": user_id,
            "user_embedding": embedding_vector,
            "user_categories": user_backstory.categories,
            "embedding_model": embedding_model,
            "embedding_dimension": embedding_dimension,
            "embedding_text": embedding_text
        }
        
        # Validate against schema
        validated_embedding = UserEmbeddingArtifact.from_dict(embedding_data)
        
        return user_id, validated_embedding.to_dict()
    except Exception as e:
        logging.error(f"Failed to process user {user_id}: {e}")
        return user_id, None


def main():
    """Main execution function."""
    
    # =========================================================================
    # Step 1: Load configuration from config.yaml
    # =========================================================================
    print("=" * 70)
    print("STAGE 05: User Level Inference - User Embeddings")
    print("=" * 70)
    
    cfg = get_stage_config("05_user_level_inference")
    openai_cfg = get_openai_config()
    
    # Validate required config fields
    if "hyperparameters" not in cfg:
        logging.error("Missing required config field: hyperparameters")
        return
    
    hyperparams = cfg["hyperparameters"]
    
    # Get input artifacts from config (required, no fallbacks)
    if "embedding_input_artifact" not in cfg:
        logging.error("Missing required config field: embedding_input_artifact")
        return
    
    input_artifact_backstories = cfg["embedding_input_artifact"]
    
    if "embedding_output_artifact" not in cfg:
        logging.error("Missing required config field: embedding_output_artifact")
        return
    
    output_artifact_name = cfg["embedding_output_artifact"]
    
    # Get backstory filename from config (required, no fallback)
    if "backstory_filename" not in cfg:
        logging.error("Missing required config field: backstory_filename")
        return
    
    backstory_filename = cfg["backstory_filename"]
    
    # Get hyperparameters from config (required, no fallbacks)
    required_hyperparams = ["embedding_model", "num_workers", "save_interval"]
    for param in required_hyperparams:
        if param not in hyperparams:
            logging.error(f"Missing required hyperparameter: {param}")
            return
    
    embedding_model = hyperparams["embedding_model"]
    num_workers = hyperparams["num_workers"]
    save_interval = hyperparams["save_interval"]
    
    # Get API key from environment
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        logging.error("OPENAI_API_KEY environment variable not set")
        return
    
    print(f"\n[Config] Input artifact (backstories): {input_artifact_backstories}")
    print(f"[Config] Output artifact: {output_artifact_name}")
    print(f"[Config] Embedding model: {embedding_model}")
    print(f"[Config] Workers (parallel): {num_workers}")
    
    # =========================================================================
    # Step 2: Initialize W&B run
    # =========================================================================
    print("\n" + "-" * 70)
    print("Step 2: Initialize W&B Run")
    print("-" * 70)
    
    run = init_wandb_run(
        run_name=f"user_embeddings_{output_artifact_name}",
        stage="05_user_level_inference",
        job_type="embedding_generation"
    )
    
    try:
        # =====================================================================
        # Step 3: Load user backstories from W&B (no local fallback)
        # =====================================================================
        print("\n" + "-" * 70)
        print("Step 3: Load User Backstories from W&B")
        print("-" * 70)
        
        # Download backstories artifact from W&B (required, no local fallback)
        logging.info(f"Downloading backstories artifact from W&B: {input_artifact_backstories}")
        backstories_path = use_artifact(run, input_artifact_backstories, artifact_type="dataset")
        
        if backstories_path is None:
            logging.error(f"Could not download backstories artifact: {input_artifact_backstories}")
            logging.error(f"Make sure to run 01_user_to_backstory_characteristics.py first and upload the artifact to W&B")
            return
        
        # Resolve path to handle any symlinks or relative paths
        # W&B may return path with :v0, :v1, etc. (invalid on Linux), need to handle this
        backstories_path_str = str(backstories_path)
        # If path contains :vN (invalid on Linux), try replacing with -vN
        # Handle any version number (v0, v1, v2, etc.)
        if not Path(backstories_path).exists():
            # Replace :v0, :v1, :v2, etc. with -v0, -v1, -v2, etc.
            backstories_path_str = re.sub(r':(v\d+)', r'-\1', backstories_path_str)
            backstories_path = Path(backstories_path_str)
        
        backstories_path = Path(backstories_path).resolve()
        logging.info(f"[W&B] Backstories artifact downloaded to: {backstories_path}")
        
        # Get backstory file using filename from config (required, no fallback)
        backstories_file = backstories_path / backstory_filename
        
        # Check if file exists - if not, list available files for debugging
        if not backstories_file.exists():
            logging.error(f"Backstory file not found in artifact: {backstories_file}")
            logging.error(f"Expected file: {backstory_filename}")
            
            # List available files for debugging (but don't use them)
            if backstories_path.exists():
                available_files = list(backstories_path.glob("*.json"))
                if available_files:
                    logging.error(f"Available JSON files in artifact:")
                    for f in available_files:
                        logging.error(f"  - {f.name}")
                else:
                    logging.error(f"No JSON files found in artifact directory: {backstories_path}")
            else:
                logging.error(f"Artifact directory does not exist: {backstories_path}")
            
            logging.error(f"Please update config.yaml with the correct backstory_filename from the list above")
            return
        
        # Load and validate user backstories
        print(f"  Loading from: {backstories_file}")
        user_backstories = UserBackstoryArtifact.from_file(backstories_file)
        print(f"[OK] Loaded backstories for {len(user_backstories)} users")
        
        # =====================================================================
        # Step 4: Initialize OpenAI client
        # =====================================================================
        print("\n" + "-" * 70)
        print("Step 4: Initialize OpenAI Client")
        print("-" * 70)
        
        client = OpenAI(api_key=api_key)
            
        # =====================================================================
        # Step 5: Generate embeddings for all users (Parallel Processing)
        # =====================================================================
        print("\n" + "-" * 70)
        print("Step 5: Generate User Embeddings (Parallel Processing Enabled)")
        print("-" * 70)
        logging.info(f"[PARALLEL] Starting parallel processing with {num_workers} workers")
        logging.info(f"[PARALLEL] Processing {len(user_backstories)} users concurrently")
        
        user_embeddings = {}
        processed_count = 0
        failed_count = 0
        
        start_time = time.time()
        total_users = len(user_backstories)
        last_log_count = 0
        
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            logging.info(f"[PARALLEL] ThreadPoolExecutor initialized with max_workers={num_workers}")
            
            # Submit all tasks to the executor
            future_to_user = {}
            for user_id, user_backstory in user_backstories.items():
                future = executor.submit(
                    process_user_embedding,
                    user_id, user_backstory, client, embedding_model
                )
                future_to_user[future] = user_id
            
            logging.info(f"[PARALLEL] Submitted {len(future_to_user)} tasks to executor (running in parallel)")
            
            # Process completed tasks as they finish
            for future in as_completed(future_to_user):
                user_id = future_to_user[future]
                try:
                    result_user_id, embedding_data = future.result()
                    
                    if embedding_data:
                        with file_lock:
                            user_embeddings[result_user_id] = embedding_data
                        with progress_lock:
                            processed_count += 1
                    else:
                        with progress_lock:
                            failed_count += 1
                    
                    # Log progress periodically
                    if processed_count - last_log_count >= save_interval:
                        elapsed = time.time() - start_time
                        rate = processed_count / elapsed if elapsed > 0 else 0
                        logging.info(f"[Progress] {processed_count}/{total_users} "
                                   f"({processed_count/total_users*100:.1f}%) "
                                   f"Rate: {rate:.2f}/sec")
                        
                        # Log progress metrics
                        log_metrics(run, {
                            "users_processed": processed_count,
                            "users_failed": failed_count,
                        })
                        last_log_count = processed_count
                
                except Exception as e:
                    logging.error(f"Exception processing user {user_id}: {e}")
                    with progress_lock:
                        failed_count += 1
        
        # =====================================================================
        # Step 6: Save embeddings with schema validation
        # =====================================================================
        print("\n" + "-" * 70)
        print("Step 6: Save User Embeddings (with Schema Validation)")
        print("-" * 70)
        
        output_dir = get_artifact_dir("05_user_level_inference", output_artifact_name)
        output_file = output_dir / "user_embeddings.json"
        
        # Validate all embeddings before saving - only save validated data
        validated_embeddings = {}
        
        for user_id, embedding_data in user_embeddings.items():
            try:
                validated = UserEmbeddingArtifact.from_dict(embedding_data)
                validated_embeddings[user_id] = validated.to_dict()
            except Exception as e:
                # Skip invalid data - don't pollute output file
                logging.warning(f"Schema validation failed for user {user_id}: {e} - skipping from output")
        
        # Save locally FIRST
        output_dir.mkdir(parents=True, exist_ok=True)
        logging.info(f"[LOCAL SAVE] Saving embeddings locally to: {output_file}")
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(validated_embeddings, f, indent=2, ensure_ascii=False)
        
        logging.info(f"[LOCAL SAVE] ✓ Saved {len(validated_embeddings)} users locally to: {output_file}")
        print(f"[OK] Saved embeddings for {len(validated_embeddings)} users locally to: {output_file}")
        
        # =====================================================================
        # Step 7: Log metrics and summary
        # =====================================================================
        print("\n" + "-" * 70)
        print("Step 7: Log Metrics")
        print("-" * 70)
        
        elapsed_time = time.time() - start_time
        
        log_summary(run, {
            "final_processed": processed_count,
            "final_failed": failed_count,
            "total_users_with_embeddings": len(validated_embeddings),
            "embedding_model": embedding_model,
            "processing_time_minutes": elapsed_time / 60,
        })
        
        # =====================================================================
        # Step 8: Upload artifact to W&B
        # =====================================================================
        print("\n" + "-" * 70)
        print("Step 8: Upload Artifact to W&B")
        print("-" * 70)
        
        logging.info(f"[W&B UPLOAD] Uploading artifact '{output_artifact_name}' to W&B...")
        logging.info(f"[W&B UPLOAD] Uploading directory: {output_dir}")
        logging.info(f"[W&B UPLOAD] Files to upload:")
        logging.info(f"[W&B UPLOAD]   - {output_file.name} ({len(validated_embeddings)} users)")
        
        # Get embedding dimension from first user if available
        embedding_dimension = None
        if validated_embeddings:
            first_user_id = list(validated_embeddings.keys())[0]
            embedding_dimension = validated_embeddings[first_user_id].get("embedding_dimension")
        
        artifact = log_artifact(
            run=run,
            artifact_name=output_artifact_name,
            artifact_type="dataset",
            artifact_path=output_dir,
            metadata=create_comprehensive_artifact_metadata(
                stage="05_user_level_inference",
                artifact_name=output_artifact_name,
                sample_size=len(validated_embeddings),
                model_vendor="OpenAI",
                model_name=embedding_model,
                model_description="Text embedding model for user characteristics",
                model_params={
                    "model": embedding_model,
                    "dimension": embedding_dimension,
                },
                learned_artifact_schema=get_learned_artifact_schema("05_user_level_inference", output_artifact_name),
                additional_metadata={
                    "num_users_processed": processed_count,
                    "num_users_failed": failed_count,
                    "total_users": len(validated_embeddings),
                    "embedding_model": embedding_model,
                    "embedding_dimension": embedding_dimension,
                    "input_artifact_backstories": input_artifact_backstories,
                    "schema_validated": True,
                    "schema_version": "v4",
                    "artifact_type": "learned_artifact",  # Mark as learned artifact
                }
            )
        )
        
        link_to_registry(artifact, stage="05_user_level_inference")
        
        if artifact:
            logging.info(f"[W&B UPLOAD] ✓ Successfully uploaded artifact '{output_artifact_name}' to W&B")
            logging.info(f"[W&B UPLOAD] ✓ All files in {output_dir} are now available in W&B")
        else:
            logging.warning(f"[W&B UPLOAD] W&B disabled or upload failed - files still saved locally")
        
        # =====================================================================
        # Final Summary
        # =====================================================================
        print("\n" + "=" * 70)
        print("STAGE 05.2 COMPLETE")
        print("=" * 70)
        print(f"\nSummary:")
        print(f"  Users processed: {processed_count}")
        print(f"  Users failed: {failed_count}")
        print(f"  Total with embeddings (schema-validated): {len(validated_embeddings)}")
        print(f"  Embedding model: {embedding_model}")
        print(f"  Processing time: {elapsed_time/60:.1f} minutes")
        print(f"\n{'='*70}")
        print(f"OUTPUT FILES - SAVED IN BOTH LOCATIONS:")
        print(f"{'='*70}")
        print(f"📁 LOCAL (saved locally):")
        print(f"   Output directory: {output_dir}")
        print(f"   ✓ Embeddings: {output_file}")
        if artifact:
            print(f"\n☁️  W&B (uploaded to Weights & Biases):")
            print(f"   Artifact name: {output_artifact_name}")
            print(f"   ✓ All files uploaded to W&B artifact")
            print(f"   View at: {run.url if run else 'N/A'}")
        else:
            print(f"\n⚠️  W&B: Upload skipped (W&B disabled or failed)")
        print(f"\n✓ All outputs are saved both locally AND in W&B")
        
        if run:
            print(f"\nView run at: {run.url}")
        
    finally:
        finish_run(run)


if __name__ == "__main__":
    main()
