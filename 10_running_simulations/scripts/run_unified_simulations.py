#!/usr/bin/env python3
"""
Unified Simulation Runner
==========================

Orchestrates all simulation types:
- Persona-based simulations (using trained personas)
- Baseline simulations (history/backstory methods)
- SGO simulations (using SGO corrected predictions)

Supports:
- Multiple configurations
- Deduplication across simulation types
- Parallel execution
- W&B artifact logging

Usage:
    python 10_running_simulations/scripts/run_unified_simulations.py
"""

import os
import sys
import json
import logging
import time
import numpy as np
from pathlib import Path
from collections import defaultdict
import threading

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

try:
    from dotenv import load_dotenv
    load_dotenv(PROJECT_ROOT / ".env", override=True)
except ImportError:
    pass

from utils.wandb_utils import (
    get_stage_config, get_openai_config, load_stage_config_file, load_config,
    init_wandb_run, finish_run, log_artifact, link_to_registry, get_artifact_dir
)
from utils.openai_client import create_openai_client

# Import simulation runners (from same directory)
SCRIPT_DIR = Path(__file__).parent
sys.path.insert(0, str(SCRIPT_DIR))

# Import baseline runner
from run_baselines_integrated import run_baselines_from_simulation

# Import SGO simulation runner
from run_sgo_simulation import run_sgo_simulation

# Import pre SGO and SGO training runners
from run_pre_sgo_training import run_pre_sgo_training
from run_sgo_training import run_sgo_training

# Initialize OpenAI-compatible client for embeddings
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
if OPENAI_API_KEY:
    openai_client = create_openai_client(openai_config=get_openai_config(), timeout=120.0)
else:
    openai_client = None
    logging.warning("OPENAI_API_KEY not set - embeddings will be empty")

# Embedding configuration
_cfg_temp = get_stage_config("10_simulations")
_openai_cfg = get_openai_config()
EMBEDDING_MODEL = _cfg_temp.get("hyperparameters", {}).get("embedding_model", 
                   _openai_cfg.get("embedding_model", "text-embedding-3-small"))

# Embedding cache
EMBEDDING_CACHE = {}
EMBEDDING_CACHE_LOCK = threading.Lock()
TOPIC_EMBEDDING_CACHE = {}
rate_limit_lock = threading.Lock()
last_request_time = [0.0]
MIN_REQUEST_INTERVAL = 0.1

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def rate_limited_request():
    """Ensure we don't exceed rate limits."""
    with rate_limit_lock:
        current_time = time.time()
        time_since_last = current_time - last_request_time[0]
        if time_since_last < MIN_REQUEST_INTERVAL:
            time.sleep(MIN_REQUEST_INTERVAL - time_since_last)
        last_request_time[0] = time.time()

def get_text_embedding(text: str, cache_key: str = None) -> list:
    """Get embedding for a text using OpenAI's embedding model."""
    if not text or not text.strip() or not openai_client:
        return []
    
    # Check cache
    if cache_key:
        with EMBEDDING_CACHE_LOCK:
            if cache_key in EMBEDDING_CACHE:
                return EMBEDDING_CACHE[cache_key]
    
    try:
        rate_limited_request()
        text_clean = str(text).replace("\n", " ")[:8000]
        response = openai_client.embeddings.create(
            input=[text_clean],
            model=EMBEDDING_MODEL
        )
        embedding = response.data[0].embedding
        
        # Cache result
        if cache_key:
            with EMBEDDING_CACHE_LOCK:
                EMBEDDING_CACHE[cache_key] = embedding
        
        return embedding
    except Exception as e:
        logging.warning(f"Failed to get embedding: {e}")
        return []

def get_topic_embedding(topic_name: str) -> list:
    """Get embedding for a topic name."""
    if not topic_name:
        return []
    
    if topic_name in TOPIC_EMBEDDING_CACHE:
        return TOPIC_EMBEDDING_CACHE[topic_name]
    
    embedding = get_text_embedding(topic_name, cache_key=f"topic_{topic_name}")
    TOPIC_EMBEDDING_CACHE[topic_name] = embedding
    return embedding

def convert_topics_to_probs(topics) -> dict:
    """Convert topic list or dict to probability format."""
    if isinstance(topics, dict):
        # Normalize to ensure probabilities sum to 1
        total = sum(topics.values())
        if total > 0:
            return {k: v / total for k, v in topics.items()}
        return topics
    elif isinstance(topics, list):
        if not topics:
            return {}
        prob = 1.0 / len(topics)
        return {topic: prob for topic in topics}
    return {}

def load_json_file(filepath) -> dict:
    """Safely loads a JSON file."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        logging.error(f"Error loading {filepath}: {e}")
        return {}

def save_json_file(data, filepath):
    """Safely saves a JSON file."""
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def create_dedupe_key(result: dict, key_fields: list) -> str:
    """Create a deduplication key from result fields."""
    key_parts = []
    for field in key_fields:
        value = result.get(field, '')
        key_parts.append(str(value))
    return "_".join(key_parts)

def deduplicate_results(all_results: dict, dedupe_config: dict) -> dict:
    """
    Deduplicate results across simulation types.
    
    Args:
        all_results: Dict of {simulation_key: result_data}
        dedupe_config: Deduplication configuration
        
    Returns:
        Deduplicated results dict
    """
    if not dedupe_config.get("enabled", False):
        return all_results
    
    key_fields = dedupe_config.get("dedupe_key_fields", ["user_id", "stimulus_product_id"])
    keep_method = dedupe_config.get("keep_method", "best")
    
    # Group by dedupe key
    grouped = defaultdict(list)
    for sim_key, result in all_results.items():
        dedupe_key = create_dedupe_key(result, key_fields)
        grouped[dedupe_key].append((sim_key, result))
    
    # Deduplicate
    deduplicated = {}
    for dedupe_key, items in grouped.items():
        if len(items) == 1:
            # No duplicates
            sim_key, result = items[0]
            deduplicated[sim_key] = result
        else:
            # Multiple results for same key - choose one
            if keep_method == "first":
                sim_key, result = items[0]
                deduplicated[sim_key] = result
            elif keep_method == "last":
                sim_key, result = items[-1]
                deduplicated[sim_key] = result
            elif keep_method == "best":
                # Keep result with best overall accuracy or recall
                best_item = None
                best_score = -1
                for sim_key, result in items:
                    metrics = result.get('metrics', {})
                    # Try to get overall_accuracy or recall@max(3,k)
                    score = metrics.get('overall_accuracy', 0.0)
                    if score == 0.0:
                        score = metrics.get('recall@max(3,k)', 0.0)
                    if score > best_score:
                        best_score = score
                        best_item = (sim_key, result)
                if best_item:
                    sim_key, result = best_item
                    deduplicated[sim_key] = result
                else:
                    # Fallback to first
                    sim_key, result = items[0]
                    deduplicated[sim_key] = result
            else:
                # Unknown method - keep first
                sim_key, result = items[0]
                deduplicated[sim_key] = result
    
    logging.info(f"Deduplication: {len(all_results)} -> {len(deduplicated)} results")
    return deduplicated

def convert_pre_sgo_to_simulation_results(pre_sgo_output_dir: str) -> dict:
    """
    Convert Pre SGO Training results to unified simulation format.
    
    Args:
        pre_sgo_output_dir: Path to Pre SGO Training output directory
        
    Returns:
        Dict of {result_key: simulation_result} in unified format
    """
    output_dir = Path(pre_sgo_output_dir)
    if not output_dir.exists():
        logging.warning(f"Pre SGO output directory not found: {pre_sgo_output_dir}")
        return {}
    
    # Find all summary files (format: {micro_id}_summary_enhanced_persona_micro_cluster_accuracy.json)
    summary_files = list(output_dir.glob("*_summary_enhanced_persona_micro_cluster_accuracy.json"))
    if not summary_files:
        # Also check subdirectories (cluster folders)
        for cluster_dir in output_dir.iterdir():
            if cluster_dir.is_dir():
                summary_files.extend(cluster_dir.glob("*_summary_enhanced_persona_micro_cluster_accuracy.json"))
    
    if not summary_files:
        logging.warning(f"No Pre SGO summary files found in {pre_sgo_output_dir}")
        return {}
    
    all_simulation_results = {}
    
    for summary_file in summary_files:
        data = load_json_file(summary_file)
        if not data:
            continue
        
        # Extract cluster and micro IDs from filename or metadata
        filename = summary_file.stem
        # Format: {micro_id}_summary_enhanced_persona_micro_cluster_accuracy
        micro_id = filename.split('_summary')[0]
        cluster_id = summary_file.parent.name if summary_file.parent.name != output_dir.name else "cluster_0"
        
        # Extract metadata
        metadata = data.get('metadata', {})
        persona_name = metadata.get('persona_name', f'persona_{micro_id}')
        total_users = metadata.get('total_users_in_cluster', 0)
        
        # Get user predictions
        user_predictions = data.get('user_predictions', {})
        if not user_predictions:
            continue
        
        for user_id, reviews in user_predictions.items():
            if not isinstance(reviews, list):
                continue
            
            num_reviews = len(reviews)
            
            for review_idx, review_entry in enumerate(reviews):
                # Get prediction and actual data
                prediction = review_entry.get('prediction', {})
                actual = review_entry.get('actual', {})
                
                if not prediction or not actual:
                    continue
                
                # Extract review data
                product_description = actual.get('product_description', 'N/A')
                product_id = actual.get('asin', actual.get('product_id', 'N/A'))
                real_review = actual.get('review_text', '')
                real_rating = actual.get('rating', 3.0)
                
                synthetic_review = prediction.get('review_text', '')
                predicted_rating = prediction.get('rating', 3.0)
                predicted_sentiment = prediction.get('sentiment', 'Neutral')
                
                # Get themes - Pre SGO uses 'themes' dict in prediction, 'predicted_themes' list in actual
                real_themes = actual.get('predicted_themes', [])
                predicted_themes = prediction.get('themes', {})
                
                # Convert themes to probabilities
                real_topic_probs = convert_topics_to_probs(real_themes)
                synthetic_topic_probs = convert_topics_to_probs(predicted_themes)
                
                # Compute embeddings
                real_review_emb = get_text_embedding(real_review, f"pre_sgo_real_{user_id}_{review_idx}")
                synthetic_review_emb = get_text_embedding(synthetic_review, f"pre_sgo_syn_{user_id}_{review_idx}")
                
                # Compute topic embeddings with probabilities
                real_topic_probs_with_emb = {}
                for topic, prob in real_topic_probs.items():
                    topic_emb = get_topic_embedding(topic)
                    real_topic_probs_with_emb[topic] = {
                        'probability': prob,
                        'embedding': topic_emb
                    }
                
                synthetic_topic_probs_with_emb = {}
                for topic, prob in synthetic_topic_probs.items():
                    topic_emb = get_topic_embedding(topic)
                    synthetic_topic_probs_with_emb[topic] = {
                        'probability': prob,
                        'embedding': topic_emb
                    }
                
                # Create simulation result entry
                result_key = f"pre_sgo_{user_id}_{product_id}_{review_idx}"
                
                simulation_result = {
                    'user_id': user_id,
                    'num_reviews_of_user': num_reviews,
                    'tribe_name': persona_name,
                    'tribe_id': micro_id,
                    'num_users_in_tribe': total_users,
                    'cluster_name': cluster_id,
                    'cluster_id': cluster_id,
                    'num_tribes_in_cluster': 1,  # Will be updated during aggregation
                    'stimulus_product_id': product_id,
                    'stimulus_product_description': product_description,
                    'real_review': real_review,
                    'real_review_embedding': real_review_emb,
                    'rating': float(real_rating),
                    'real_topic_probabilities': real_topic_probs_with_emb,
                    'synthetic_review': synthetic_review,
                    'synthetic_review_embedding': synthetic_review_emb,
                    'synthetic_topic_probabilities': synthetic_topic_probs_with_emb,
                    'sentiment_predicted': predicted_sentiment,
                    'rating_predicted': float(predicted_rating),
                    'simulation_method': 'pre_sgo',  # Mark as Pre SGO simulation
                    'metrics': review_entry.get('metrics', {})
                }
                
                all_simulation_results[result_key] = simulation_result
    
    logging.info(f"✅ Converted {len(all_simulation_results)} Pre SGO Training results to simulation format")
    return all_simulation_results

def main():
    """Main unified simulation runner."""
    logging.info("=" * 70)
    logging.info("UNIFIED SIMULATION RUNNER")
    logging.info("=" * 70)
    
    # Load configuration
    cfg = get_stage_config("10_simulations")
    sim_config = cfg.get("simulation_config", {})
    dedupe_config = cfg.get("deduplication", {})
    output_artifacts = cfg.get("output_artifacts", {})
    
    # Initialize W&B run
    run = init_wandb_run(
        run_name=f"unified_simulations_{time.strftime('%Y%m%d_%H%M%S')}",
        stage="10_simulations",
        job_type="unified_simulation"
    )
    
    try:
        all_simulation_results = {}
        simulation_metadata = {
            "history": {"count": 0, "file": None},
            "backstory": {"count": 0, "file": None},
            "pre_sgo": {"status": "skipped", "output_artifact": None},
            "pre_sgo_training": {"status": "skipped", "output_artifact": None, "local_output_dir": None},
            "sgo_training": {"status": "skipped", "output_artifact": None, "local_output_dir": None},
            "post_sgo": {"count": 0, "file": None}
        }
        
        # Get hyperparameters for training steps
        hyper_config = cfg.get("hyperparameters", {})
        
        # =====================================================================
        # 0. Run Pre SGO Training (Initial Predictions)
        # =====================================================================
        pre_sgo_config = sim_config.get("pre_sgo", {})
        if pre_sgo_config.get("enabled", False):
            # Load method-specific config file
            global_config = load_config()
            method_config = load_stage_config_file("10_running_simulations", "config_pre_sgo.yaml")
            pre_sgo_config = {**global_config, **method_config}
            # Merge with any overrides from main config
            main_pre_sgo = sim_config.get("pre_sgo", {})
            if main_pre_sgo:
                pre_sgo_config.update(main_pre_sgo)
            
            logging.info("\n" + "=" * 70)
            logging.info("PRE SGO TRAINING: Initial Predictions Generation")
            logging.info("=" * 70)
            
            try:
                result = run_pre_sgo_training(
                    run=run,
                    pre_sgo_config=pre_sgo_config,  # Already extracted from sim_config
                    hyper_config=hyper_config
                )
                
                if result.get("status") == "success":
                    simulation_metadata["pre_sgo_training"]["status"] = "success"
                    simulation_metadata["pre_sgo_training"]["output_artifact"] = result.get("output_artifact")
                    simulation_metadata["pre_sgo_training"]["local_output_dir"] = result.get("local_output_dir")
                    logging.info(f"✅ Pre SGO training completed:")
                    logging.info(f"   📦 W&B Artifact: {result.get('output_artifact')}")
                    logging.info(f"   📁 Local Directory: {result.get('local_output_dir')}")
                    
                    # Convert Pre SGO Training results to unified simulation format (if enabled)
                    convert_to_sim_format = pre_sgo_config.get("convert_to_simulation_format", False)
                    if convert_to_sim_format:
                        logging.info("\n" + "=" * 70)
                        logging.info("Converting Pre SGO Training results to simulation format")
                        logging.info("=" * 70)
                        pre_sgo_results = convert_pre_sgo_to_simulation_results(result.get("local_output_dir"))
                        if pre_sgo_results:
                            all_simulation_results.update(pre_sgo_results)
                            simulation_metadata["pre_sgo"]["num_results"] = len(pre_sgo_results)
                            logging.info(f"✅ Added {len(pre_sgo_results)} Pre SGO simulation results to unified results")
                        else:
                            logging.warning("⚠️  No Pre SGO results converted (check output directory)")
                    else:
                        logging.info("⏭️  Pre SGO to simulation format conversion disabled (set convert_to_simulation_format: true to enable)")
                else:
                    simulation_metadata["pre_sgo"]["status"] = "error"
                    logging.error(f"❌ Pre SGO training failed: {result.get('error')}")
            except Exception as e:
                logging.error(f"❌ Pre SGO training failed: {e}")
                import traceback
                traceback.print_exc()
                simulation_metadata["pre_sgo"]["status"] = "error"
        else:
            logging.info("⏭️  Pre SGO training disabled (set pre_sgo.enabled: true to enable)")
        
        # =====================================================================
        # 0.5. Run SGO Training (Delta Refinement)
        # =====================================================================
        sgo_training_config = sim_config.get("sgo_training", {})
        if sgo_training_config.get("enabled", False):
            # Load method-specific config file
            global_config = load_config()
            method_config = load_stage_config_file("10_running_simulations", "config_sgo_training.yaml")
            sgo_training_config = {**global_config, **method_config}
            # Merge with any overrides from main config
            main_sgo_training = sim_config.get("sgo_training", {})
            if main_sgo_training:
                sgo_training_config.update(main_sgo_training)
            logging.info("\n" + "=" * 70)
            logging.info("SGO TRAINING: Delta Refinement")
            logging.info("=" * 70)
            
            try:
                # SGO training config is already in sim_config, pass it directly
                result = run_sgo_training(
                    run=run,
                    sgo_config=sgo_training_config,  # Already extracted from sim_config
                    hyper_config=hyper_config
                )
                
                if result.get("status") == "success":
                    simulation_metadata["sgo_training"]["status"] = "success"
                    simulation_metadata["sgo_training"]["output_artifact"] = result.get("output_artifact")
                    simulation_metadata["sgo_training"]["local_output_dir"] = result.get("local_output_dir")
                    logging.info(f"✅ SGO training completed:")
                    logging.info(f"   📦 W&B Artifact: {result.get('output_artifact')}")
                    logging.info(f"   📁 Local Directory: {result.get('local_output_dir')}")
                else:
                    simulation_metadata["sgo_training"]["status"] = "error"
                    logging.error(f"❌ SGO training failed: {result.get('error')}")
            except Exception as e:
                logging.error(f"❌ SGO training failed: {e}")
                import traceback
                traceback.print_exc()
                simulation_metadata["sgo_training"]["status"] = "error"
        else:
            logging.info("⏭️  SGO training disabled (set sgo_training.enabled: true to enable)")
        
        # =====================================================================
        # 1. Run Persona Simulations
        # =====================================================================
        persona_config = sim_config.get("persona_simulations", {})
        if persona_config.get("enabled", False):
            # Load method-specific config file
            global_config = load_config()
            method_config = load_stage_config_file("10_running_simulations", "config_persona_simulations.yaml")
            persona_config = {**global_config, **method_config}
            # Merge with any overrides from main config
            main_persona = sim_config.get("persona_simulations", {})
            if main_persona:
                persona_config.update(main_persona)
            
            logging.info("\n" + "=" * 70)
            logging.info("Running Persona Simulations")
            logging.info("=" * 70)
            
            try:
                # Run persona simulations (this will create its own W&B run)
                # We need to capture results from the output artifact
                persona_output = output_artifacts.get("persona_simulation_results", "simulation_results_v4")
                logging.info(f"Persona simulations will output to: {persona_output}")
                # Note: run_simulations.py handles its own W&B run, so we'll load results after
                logging.info("✅ Persona simulations completed (check separate W&B run)")
                simulation_metadata["persona_simulations"]["count"] = 1  # Mark as run
            except Exception as e:
                logging.error(f"❌ Persona simulations failed: {e}")
        else:
            logging.info("⏭️  Persona simulations disabled (set persona_simulations.enabled: true to enable)")
        
        # =====================================================================
        # 2. Run Baseline Simulations
        # =====================================================================
        baseline_config = sim_config.get("baseline_simulations", {})
        if baseline_config.get("enabled", False):
            # Load method-specific config file
            global_config = load_config()
            method_config = load_stage_config_file("10_running_simulations", "config_baseline_simulations.yaml")
            baseline_config = {**global_config, **method_config}
            # Merge with any overrides from main config
            main_baseline = sim_config.get("baseline_simulations", {})
            if main_baseline:
                baseline_config.update(main_baseline)
            
            logging.info("\n" + "=" * 70)
            logging.info("Running Baseline Simulations")
            logging.info("=" * 70)
            
            # Get baseline configuration
            models = baseline_config.get("models", ["o3"])
            methods = baseline_config.get("methods", ["history"])
            run_all_combinations = baseline_config.get("run_all_combinations", True)
            
            logging.info(f"Configuration:")
            logging.info(f"  Models: {models}")
            logging.info(f"  Methods: {methods}")
            logging.info(f"  Run all combinations: {run_all_combinations}")
            
            try:
                # Pass baseline config to the runner
                baseline_config_override = {
                    "models": models,
                    "methods": methods,
                    "run_all_combinations": run_all_combinations
                }
                success = run_baselines_from_simulation(baseline_config_override=baseline_config_override)
                if success:
                    logging.info("✅ Baseline simulations completed")
                    # Baseline results are saved by run_baselines.py
                    simulation_metadata["baseline_simulations"]["count"] = len(models) * len(methods) if run_all_combinations else 1
                else:
                    logging.error("❌ Baseline simulations failed")
            except Exception as e:
                logging.error(f"❌ Baseline simulations failed: {e}")
                import traceback
                traceback.print_exc()
        else:
            logging.info("⏭️  Baseline simulations disabled (set baseline_simulations.enabled: true to enable)")
        
        # =====================================================================
        # 3. Run SGO Simulations
        # =====================================================================
        sgo_config = sim_config.get("sgo_simulations", {})
        if sgo_config.get("enabled", False):
            logging.info("\n" + "=" * 70)
            logging.info("Running SGO Simulations")
            logging.info("=" * 70)
            
            # Determine which SGO artifact to use
            dataset_type = cfg.get("dataset_type", "train")
            input_artifact = sgo_config.get("input_artifact")
            
            if not input_artifact:
                # Auto-select based on dataset_type
                if dataset_type == "train":
                    input_artifact = sgo_config.get("input_artifact_train")
                else:
                    input_artifact = sgo_config.get("input_artifact_test")
            
            if not input_artifact:
                # Fallback to input_artifacts
                input_artifacts = cfg.get("input_artifacts", {})
                if dataset_type == "train":
                    input_artifact = input_artifacts.get("sgo_training_results_train")
                else:
                    input_artifact = input_artifacts.get("sgo_training_results_test")
            
            if not input_artifact:
                logging.error("❌ SGO input artifact not specified")
                logging.error("   Set sgo_simulations.input_artifact or sgo_simulations.input_artifact_train/test")
            else:
                logging.info(f"Using SGO artifact: {input_artifact} (dataset_type: {dataset_type})")
                
                # Temporarily update config for SGO runner
                original_sgo_config = cfg.get("sgo_simulation", {})
                cfg["sgo_simulation"] = {"input_artifact": input_artifact}
                
                try:
                    max_workers = cfg.get("hyperparameters", {}).get("max_workers", 10)
                    sgo_result_file = run_sgo_simulation(run, max_workers=max_workers)
                    if sgo_result_file:
                        # Load SGO results
                        sgo_results = load_json_file(sgo_result_file)
                        if sgo_results:
                            all_simulation_results.update(sgo_results)
                            simulation_metadata["sgo_simulations"]["count"] = len(sgo_results)
                            simulation_metadata["sgo_simulations"]["file"] = str(sgo_result_file)
                            logging.info(f"✅ Loaded {len(sgo_results)} SGO simulation results")
                    else:
                        logging.error("❌ SGO simulations failed")
                except Exception as e:
                    logging.error(f"❌ SGO simulations failed: {e}")
                    import traceback
                    traceback.print_exc()
                finally:
                    # Restore original config
                    if original_sgo_config:
                        cfg["sgo_simulation"] = original_sgo_config
        else:
            logging.info("⏭️  SGO simulations disabled (set sgo_simulations.enabled: true to enable)")
        
        # =====================================================================
        # 4. Deduplicate Results
        # =====================================================================
        if all_simulation_results and dedupe_config.get("enabled", False):
            logging.info("\n" + "=" * 70)
            logging.info("Deduplicating Results")
            logging.info("=" * 70)
            
            all_simulation_results = deduplicate_results(all_simulation_results, dedupe_config)
        
        # =====================================================================
        # 5. Save Unified Results
        # =====================================================================
        if all_simulation_results:
            logging.info("\n" + "=" * 70)
            logging.info("Saving Unified Results")
            logging.info("=" * 70)
            
            unified_output = output_artifacts.get("unified_simulation_results", "unified_simulation_results_v4")
            output_dir = get_artifact_dir("10_simulations", unified_output)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Save unified results
            unified_file = output_dir / "unified_simulation_results.json"
            save_json_file(all_simulation_results, unified_file)
            
            # Save metadata
            metadata_file = output_dir / "simulation_metadata.json"
            simulation_metadata["total_results"] = len(all_simulation_results)
            simulation_metadata["deduplication_enabled"] = dedupe_config.get("enabled", False)
            save_json_file(simulation_metadata, metadata_file)
            
            logging.info(f"✅ Saved {len(all_simulation_results)} unified simulation results")
            logging.info(f"   Results: {unified_file}")
            logging.info(f"   Metadata: {metadata_file}")
            
            # Log to W&B
            artifact = log_artifact(
                run=run,
                artifact_name=unified_output,
                artifact_type="result",
                artifact_path=str(output_dir),
                metadata={
                    "total_results": len(all_simulation_results),
                    "simulation_types": list(simulation_metadata.keys()),
                    "deduplication_enabled": dedupe_config.get("enabled", False),
                    "schema_version": "v4"
                }
            )
            
            if artifact:
                link_to_registry(artifact, stage="10_simulations")
                logging.info(f"✅ Logged unified artifact: {artifact.name}")
        
        # =====================================================================
        # Summary
        # =====================================================================
        logging.info("\n" + "=" * 70)
        logging.info("SIMULATION RUNNER SUMMARY")
        logging.info("=" * 70)
        
        pre_sgo_config = sim_config.get("pre_sgo", {})
        sgo_training_config = sim_config.get("sgo_training", {})
        persona_config = sim_config.get("persona_simulations", {})
        baseline_config = sim_config.get("baseline_simulations", {})
        sgo_config = sim_config.get("sgo_simulations", {})
        
        logging.info(f"Pre SGO Training: {'✅ Enabled' if pre_sgo_config.get('enabled') else '⏭️  Disabled'}")
        if pre_sgo_config.get("enabled"):
            pre_sgo_status = simulation_metadata.get("pre_sgo_training", {}).get("status", "unknown")
            pre_sgo_artifact = simulation_metadata.get("pre_sgo_training", {}).get("output_artifact", "N/A")
            pre_sgo_results_count = simulation_metadata.get("pre_sgo_training", {}).get("num_results", 0)
            logging.info(f"  - Status: {pre_sgo_status}")
            logging.info(f"  - Output: {pre_sgo_artifact}")
            if pre_sgo_results_count > 0:
                logging.info(f"  - Simulation Results: {pre_sgo_results_count}")
        
        logging.info(f"SGO Training: {'✅ Enabled' if sgo_training_config.get('enabled') else '⏭️  Disabled'}")
        if sgo_training_config.get("enabled"):
            sgo_training_status = simulation_metadata.get("sgo_training", {}).get("status", "unknown")
            sgo_training_artifact = simulation_metadata.get("sgo_training", {}).get("output_artifact", "N/A")
            logging.info(f"  - Status: {sgo_training_status}")
            logging.info(f"  - Output: {sgo_training_artifact}")
        
        logging.info(f"Persona Simulations: {'✅ Enabled' if persona_config.get('enabled') else '⏭️  Disabled'}")
        logging.info(f"Baseline Simulations: {'✅ Enabled' if baseline_config.get('enabled') else '⏭️  Disabled'}")
        if baseline_config.get("enabled"):
            models = baseline_config.get("models", [])
            methods = baseline_config.get("methods", [])
            logging.info(f"  - Models: {models}")
            logging.info(f"  - Methods: {methods}")
            logging.info(f"  - Combinations: {len(models) * len(methods) if baseline_config.get('run_all_combinations') else 1}")
        logging.info(f"SGO Simulations: {'✅ Enabled' if sgo_config.get('enabled') else '⏭️  Disabled'}")
        if sgo_config.get("enabled"):
            dataset_type = cfg.get("dataset_type", "train")
            logging.info(f"  - Dataset: {dataset_type}")
        logging.info(f"Total Unified Results: {len(all_simulation_results)}")
        logging.info(f"Deduplication: {'✅ Enabled' if dedupe_config.get('enabled') else '❌ Disabled'}")
    
    finally:
        finish_run(run)

if __name__ == "__main__":
    main()

