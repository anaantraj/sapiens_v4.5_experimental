#!/usr/bin/env python3
"""
SGO Simulation Runner
=====================

Runs SGO simulations using corrected predictions from Stage 07 (post SGO training).
- Reads SGO training results (corrected predictions) from W&B
- Formats predictions as simulation results matching the simulation schema
- Computes embeddings and metrics
- Logs results to W&B

This script uses the SGO corrected predictions directly (no LLM calls needed).
"""

import os
import sys
import json
import logging
import time
import numpy as np
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

try:
    from dotenv import load_dotenv
    load_dotenv(PROJECT_ROOT / ".env", override=True)
except ImportError:
    pass

from utils.openai_client import create_openai_client
from utils.wandb_utils import (
    get_stage_config, get_openai_config,
    init_wandb_run, finish_run, use_artifact, log_artifact,
    log_metrics, log_summary, link_to_registry, get_artifact_dir
)

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# =============================================================================
# Configuration
# =============================================================================

_cfg = get_stage_config("10_simulations")
_openai_cfg = get_openai_config()

STAGE_DIR = Path(__file__).parent.parent
EMBEDDING_MODEL = _cfg.get("hyperparameters", {}).get("embedding_model", 
                   _openai_cfg.get("embedding_model", "text-embedding-3-small"))

# Initialize OpenAI-compatible client
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    logging.error("OPENAI_API_KEY environment variable not set")
    raise ValueError("OPENAI_API_KEY environment variable not set")
openai_client = create_openai_client(openai_config=_openai_cfg, timeout=120.0)

# Rate limiting
rate_limit_lock = threading.Lock()
last_request_time = [0.0]
MIN_REQUEST_INTERVAL = 0.1

# Embedding cache
EMBEDDING_CACHE = {}
EMBEDDING_CACHE_LOCK = threading.Lock()
TOPIC_EMBEDDING_CACHE = {}

# =============================================================================
# Helper Functions
# =============================================================================

def load_json_file(filepath) -> dict:
    """Safely loads a JSON file."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        logging.error(f"File not found: {filepath}")
        return None
    except json.JSONDecodeError:
        logging.error(f"Error decoding JSON from file: {filepath}")
        return None

def convert_to_serializable(obj):
    """Recursively converts numpy types to native Python types."""
    if isinstance(obj, np.integer): return int(obj)
    if isinstance(obj, np.floating): return float(obj)
    if isinstance(obj, np.ndarray): return obj.tolist()
    if isinstance(obj, dict): return {k: convert_to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, list): return [convert_to_serializable(item) for item in obj]
    return obj

def rate_limited_request():
    """Ensure we don't exceed rate limits."""
    with rate_limit_lock:
        current_time = time.time()
        time_since_last = current_time - last_request_time[0]
        if time_since_last < MIN_REQUEST_INTERVAL:
            time.sleep(MIN_REQUEST_INTERVAL - time_since_last)
        last_request_time[0] = time.time()

# =============================================================================
# Embedding Functions
# =============================================================================

def get_text_embedding(text: str, cache_key: str = None) -> list:
    """Get embedding for a text using OpenAI's embedding model."""
    if not text or not text.strip():
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

def compute_weighted_topic_embedding(topic_probs: dict) -> list:
    """Compute weighted topic embedding from topic probabilities."""
    if not topic_probs:
        return []
    
    embeddings = []
    weights = []
    
    for topic, prob in topic_probs.items():
        if prob > 0:
            emb = get_topic_embedding(topic)
            if emb:
                embeddings.append(np.array(emb))
                weights.append(prob)
    
    if not embeddings:
        return []
    
    # Normalize weights
    weights = np.array(weights)
    weights = weights / weights.sum()
    
    # Weighted average
    weighted_emb = np.zeros_like(embeddings[0])
    for emb, w in zip(embeddings, weights):
        weighted_emb += w * emb
    
    return weighted_emb.tolist()

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

# =============================================================================
# SGO Simulation Processing
# =============================================================================

def process_sgo_prediction_file(filepath: Path, cluster_id: str, micro_id: str, 
                                topic_universe: dict, run) -> list:
    """
    Process a single SGO corrected predictions file and convert to simulation format.
    
    Args:
        filepath: Path to corrected predictions JSON file
        cluster_id: Cluster ID
        micro_id: Micro cluster ID
        topic_universe: Topic universe dict (category -> themes)
        run: W&B run for logging
        
    Returns:
        List of simulation result entries
    """
    data = load_json_file(filepath)
    if not data:
        return []
    
    # Extract metadata
    metadata = data.get('metadata', {})
    persona_name = metadata.get('persona_name', f'persona_{micro_id}')
    total_users = metadata.get('total_users_in_cluster', 0)
    total_reviews = metadata.get('total_reviews_from_cluster', 0)
    
    # Get user predictions
    user_predictions = data.get('user_predictions', {})
    if not user_predictions:
        return []
    
    simulation_results = []
    
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
            product_id = actual.get('product_id', actual.get('asin', 'N/A'))
            real_review = actual.get('review_text', '')
            real_rating = actual.get('rating', 3.0)
            
            synthetic_review = prediction.get('review_text', '')
            predicted_rating = prediction.get('rating', 3.0)
            predicted_sentiment = prediction.get('sentiment', 'Neutral')
            
            # Get themes
            real_themes = actual.get('predicted_themes', [])
            predicted_themes = prediction.get('predicted_themes', {})
            
            # Get category for theme lookup
            category = actual.get('category', '')
            
            # Convert themes to probabilities
            real_topic_probs = convert_topics_to_probs(real_themes)
            synthetic_topic_probs = convert_topics_to_probs(predicted_themes)
            
            # Compute embeddings
            real_review_emb = get_text_embedding(real_review, f"sgo_real_{user_id}_{review_idx}")
            synthetic_review_emb = get_text_embedding(synthetic_review, f"sgo_syn_{user_id}_{review_idx}")
            
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
            result_key = f"{user_id}_{product_id}_{review_idx}"
            
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
                'simulation_method': 'sgo',  # Mark as SGO simulation
                'metrics': {}  # Can be computed later if needed
            }
            
            simulation_results.append((result_key, simulation_result))
    
    return simulation_results

def run_sgo_simulation(run, max_workers: int = 10):
    """
    Main function to run SGO simulations.
    
    Args:
        run: W&B run object
        max_workers: Number of parallel workers for processing
        
    Returns:
        Path to output file or None if failed
    """
    logging.info("=" * 70)
    logging.info("SGO Simulation Runner")
    logging.info("=" * 70)
    
    # Get SGO simulation config (try both locations for compatibility)
    sgo_config = _cfg.get("sgo_simulation", {})
    if not sgo_config:
        # Try new location
        sim_config = _cfg.get("simulation_config", {})
        sgo_config = sim_config.get("sgo_simulations", {})
    
    input_artifact = sgo_config.get("input_artifact")
    
    if not input_artifact:
        # Try dataset-specific artifacts
        dataset_type = _cfg.get("dataset_type", "train")
        if dataset_type == "train":
            input_artifact = sgo_config.get("input_artifact_train")
        else:
            input_artifact = sgo_config.get("input_artifact_test")
    
    if not input_artifact:
        # Fallback to input_artifacts
        input_artifacts = _cfg.get("input_artifacts", {})
        dataset_type = _cfg.get("dataset_type", "train")
        if dataset_type == "train":
            input_artifact = input_artifacts.get("sgo_training_results_train")
        else:
            input_artifact = input_artifacts.get("sgo_training_results_test")
    
    if not input_artifact:
        logging.error("SGO input artifact not specified in config.yaml")
        logging.error("Required: sgo_simulation.input_artifact or simulation_config.sgo_simulations.input_artifact")
        logging.error("   Or set: input_artifacts.sgo_training_results_train/test")
        return None
    
    # Download SGO training results
    logging.info(f"Downloading SGO training results: {input_artifact}")
    sgo_path = use_artifact(run, input_artifact, artifact_type="model")
    
    if not sgo_path:
        # Try dataset type
        sgo_path = use_artifact(run, input_artifact, artifact_type="dataset")
    
    if not sgo_path:
        logging.error(f"Could not download SGO artifact: {input_artifact}")
        return None
    
    logging.info(f"SGO artifact downloaded to: {sgo_path}")
    
    # Download topic universe
    input_artifacts = _cfg.get("input_artifacts", {})
    topic_artifact = input_artifacts.get("topics")
    if not topic_artifact:
        logging.error("Topic universe artifact not found in config")
        return None
    
    topic_path = use_artifact(run, topic_artifact, artifact_type="dataset")
    if not topic_path:
        logging.error(f"Could not download topic universe: {topic_artifact}")
        return None
    
    # Load topic universe
    topic_file = topic_path / "topic_universe.json"
    if not topic_file.exists():
        topic_files = list(topic_path.glob("**/*topic*universe*.json"))
        if topic_files:
            topic_file = topic_files[0]
        else:
            logging.error(f"Could not find topic universe file in {topic_path}")
            return None
    
    topic_universe_data = load_json_file(topic_file)
    if not topic_universe_data:
        logging.error("Failed to load topic universe")
        return None
    
    # Extract topics by category
    topic_universe = {}
    if isinstance(topic_universe_data, dict):
        for category, topics_data in topic_universe_data.items():
            if isinstance(topics_data, list):
                topic_universe[category] = [t.get('topic_name', t) if isinstance(t, dict) else str(t) for t in topics_data]
            elif isinstance(topics_data, dict) and 'topics' in topics_data:
                topics = topics_data['topics']
                if isinstance(topics, list):
                    topic_universe[category] = [t.get('topic_name', t) if isinstance(t, dict) else str(t) for t in topics]
    
    # Find all corrected prediction files
    corrected_dir = sgo_path
    if not corrected_dir.exists():
        logging.error(f"SGO artifact path does not exist: {sgo_path}")
        return None
    
    # Look for cluster directories
    cluster_dirs = []
    if corrected_dir.is_dir():
        for item in corrected_dir.iterdir():
            if item.is_dir() and item.name.startswith('cluster_'):
                cluster_dirs.append(item)
    
    if not cluster_dirs:
        logging.error("No cluster directories found in SGO artifact")
        return None
    
    logging.info(f"Found {len(cluster_dirs)} clusters to process")
    
    # Process all files
    all_simulation_results = {}
    all_tasks = []
    
    for cluster_dir in cluster_dirs:
        cluster_id = cluster_dir.name
        for file in cluster_dir.iterdir():
            if file.is_file() and file.name.endswith('_corrected.json'):
                # Extract micro_id from filename
                micro_match = file.name.replace('_corrected.json', '').replace('micro_', '')
                micro_id = f"micro_{micro_match}" if not micro_match.startswith('micro_') else micro_match
                all_tasks.append((file, cluster_id, micro_id))
    
    logging.info(f"Processing {len(all_tasks)} SGO prediction files...")
    
    # Process in parallel
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_task = {
            executor.submit(process_sgo_prediction_file, filepath, cluster_id, micro_id, topic_universe, run): (filepath, cluster_id, micro_id)
            for filepath, cluster_id, micro_id in all_tasks
        }
        
        with tqdm(total=len(all_tasks), desc="Processing SGO predictions") as pbar:
            for future in as_completed(future_to_task):
                try:
                    results = future.result()
                    for result_key, result_data in results:
                        all_simulation_results[result_key] = result_data
                except Exception as e:
                    logging.error(f"Error processing file: {e}")
                finally:
                    pbar.update(1)
    
    # Update cluster/tribe counts
    cluster_tribe_counts = defaultdict(int)
    cluster_user_counts = defaultdict(lambda: defaultdict(int))
    
    for result in all_simulation_results.values():
        cluster_id = result['cluster_id']
        tribe_id = result['tribe_id']
        cluster_tribe_counts[cluster_id] += 1
        cluster_user_counts[cluster_id][tribe_id] = result['num_users_in_tribe']
    
    # Update counts in results
    for result in all_simulation_results.values():
        cluster_id = result['cluster_id']
        result['num_tribes_in_cluster'] = len(set(r['tribe_id'] for r in all_simulation_results.values() if r['cluster_id'] == cluster_id))
    
    # Save results
    # Save in 10_running_simulations/artifacts
    output_artifact_name = _cfg.get("output_artifacts", {}).get("sgo_simulation_results", "sgo_simulation_results_v4")
    output_dir = get_artifact_dir("10_running_simulations", output_artifact_name)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_file = output_dir / "sgo_simulation_results.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(convert_to_serializable(all_simulation_results), f, indent=2)
    
    logging.info(f"✅ Saved {len(all_simulation_results)} SGO simulation results to {output_file}")
    
    # Log to W&B
    artifact = log_artifact(
        run=run,
        artifact_name=output_artifact_name,
        artifact_type="result",
        artifact_path=str(output_dir),
        metadata={
            "simulation_method": "sgo",
            "num_results": len(all_simulation_results),
            "num_clusters": len(cluster_tribe_counts),
            "schema_version": "v4"
        }
    )
    
    if artifact:
        link_to_registry(artifact, stage="10_simulations")
        logging.info(f"✅ Logged artifact: {artifact.name}")
    
    return output_file

if __name__ == "__main__":
    run = init_wandb_run(
        run_name=f"sgo_simulation_{time.strftime('%Y%m%d_%H%M%S')}",
        stage="10_simulations",
        job_type="sgo_simulation"
    )
    
    try:
        max_workers = _cfg.get("hyperparameters", {}).get("max_workers", 10)
        result_file = run_sgo_simulation(run, max_workers=max_workers)
        if result_file:
            logging.info(f"✅ SGO simulation completed: {result_file}")
        else:
            logging.error("❌ SGO simulation failed")
    finally:
        finish_run(run)

