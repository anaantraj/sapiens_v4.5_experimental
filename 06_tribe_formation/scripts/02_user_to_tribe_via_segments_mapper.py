#!/usr/bin/env python3
"""
Stage 06: Tribe Formation - User to Tribe via Segments Mapper
==============================================================

Performs micro-clustering within each macro segment to identify granular user personas (tribes).
- Loads segment user details and user segments (embeddings) from learned artifacts
- Performs cosine similarity-based clustering within each segment
- Generates micro-cluster personas using LLM
- Outputs user tribe assignments as learned artifact

Usage:
    python 06_tribe_formation/scripts/02_user_to_tribe_via_segments_mapper.py
"""

import os
import json
import sys
import logging
import time
import re
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from openai import OpenAI, RateLimitError, APIError
from collections import defaultdict, Counter
from typing import List, Dict, Optional, Set, Tuple
from pathlib import Path
import pickle

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Load .env so OPENAI_API_KEY and OPENAI_BASE_URL (Bedrock) are set
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent.parent.parent / ".env", override=True)
except ImportError:
    pass

from utils.openai_client import create_openai_client
from utils.wandb_utils import (
    load_config, get_stage_config, get_openai_config,
    init_wandb_run, finish_run, use_artifact, log_artifact,
    log_metrics, log_summary, link_to_registry, get_artifact_dir
)

# Import schemas for validation
from schemas.learned_artifacts import (
    UserSegmentsArtifact,
    UserTribeArtifact
)

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Prompt directory
PROMPT_DIR = Path(__file__).parent.parent / "prompts"

# Optimization constants
BATCH_SIZE_SIMILARITY = 5000  # Process similarity in batches
MAX_REVIEWS_PER_CLUSTER = 2000  # Limit reviews loaded per cluster
MAX_CHARS_PER_CLUSTER = 50000  # Limit characteristics loaded per cluster
API_RATE_LIMIT_DELAY = 1  # seconds between API calls
MAX_RETRIES = 3
RETRY_DELAY = 5  # seconds


def load_prompt(prompt_filename: str) -> str:
    """Load a prompt template from file."""
    prompt_path = PROMPT_DIR / prompt_filename
    if prompt_path.exists():
        with open(prompt_path, 'r', encoding='utf-8') as f:
            return f.read().strip()
    else:
        raise FileNotFoundError(f"Prompt file not found: {prompt_path}")


def calculate_quantitative_summary(reviews_list: list) -> dict:
    """Calculates quantitative statistics from a list of review objects."""
    total_reviews = len(reviews_list)
    if total_reviews == 0:
        return {
            "average_rating": "N/A",
            "sentiment_distribution_percent": {},
        }
        
    # 1. Average Rating
    try:
        avg_rating = np.mean([r['rating'] for r in reviews_list if 'rating' in r and r['rating'] is not None])
        avg_rating_str = f"{avg_rating:.2f}"
    except Exception:
        avg_rating_str = "N/A"
        
    # 2. Sentiment Distribution
    try:
        sentiment_counts = Counter([r.get('sentiment', 'Unknown') for r in reviews_list])
        sentiment_dist = {key: round((count / total_reviews) * 100, 1) for key, count in sentiment_counts.items()}
    except Exception:
        sentiment_dist = {}

    return {
        "average_rating": avg_rating_str,
        "sentiment_distribution_percent": sentiment_dist,
    }


def summarize_micro_cluster(reviews_list: List[dict], users_char_list: List[dict], 
                           llm_model: str, client: OpenAI, prompt_template: str,
                           retries: int = MAX_RETRIES) -> dict:
    """
    Analyzes a list of reviews AND their corresponding user characteristics
    to generate a synthesized qualitative persona.
    Optimized with sampling and retry logic.
    """
    # 1. Prepare Review Data (sample intelligently)
    sample_size = min(50, len(reviews_list))
    if len(reviews_list) > sample_size:
        step = max(1, len(reviews_list) // sample_size)
        sample_reviews = [reviews_list[i] for i in range(0, len(reviews_list), step)][:sample_size]
    else:
        sample_reviews = reviews_list
    
    context_data = [
        {
            "user_id": r.get("user_id"), 
            "product_description": r.get("product_description", "")[:200],
            "review_text": r.get("review_text", "")[:500],
            "rating": r.get("rating"),
            "predicted_themes": r.get("predicted_themes", []),
            "sentiment": r.get("sentiment")
        } 
        for r in sample_reviews
    ]
    reviews_json_str = json.dumps(context_data, indent=2)

    # 2. Prepare User Characteristic Data (sample)
    sample_chars_size = min(50, len(users_char_list))
    if len(users_char_list) > sample_chars_size:
        step = max(1, len(users_char_list) // sample_chars_size)
        sample_chars = [users_char_list[i] for i in range(0, len(users_char_list), step)][:sample_chars_size]
    else:
        sample_chars = users_char_list
    users_json_str = json.dumps(sample_chars, indent=2)
    
    # 3. Format prompt with data
    prompt = prompt_template.format(
        users_json_str=users_json_str,
        reviews_json_str=reviews_json_str
    )
    
    # Retry logic with exponential backoff
    for attempt in range(retries):
        try:
            if attempt > 0:
                delay = RETRY_DELAY * (2 ** (attempt - 1))
                logging.warning(f"  Retry attempt {attempt + 1}/{retries} after {delay}s...")
                time.sleep(delay)
            
            chat_completion = client.chat.completions.create(
                messages=[
                    {"role": "user", "content": prompt}
                ],
                model=llm_model,
                response_format={"type": "json_object"},
                timeout=120
            )
            response_content = chat_completion.choices[0].message.content.strip()
            
            time.sleep(API_RATE_LIMIT_DELAY)
            return json.loads(response_content)
            
        except RateLimitError as e:
            wait_time = RETRY_DELAY * (2 ** attempt)
            logging.warning(f"  Rate limit hit. Waiting {wait_time}s...")
            time.sleep(wait_time)
            if attempt == retries - 1:
                logging.error(f"  ❌ Rate limit exceeded after {retries} attempts")
                
        except APIError as e:
            logging.error(f"  API error (attempt {attempt + 1}/{retries}): {e}")
            if attempt == retries - 1:
                logging.error(f"  ❌ API call failed after {retries} attempts")
                
        except Exception as e:
            logging.error(f"  Unexpected error (attempt {attempt + 1}/{retries}): {e}")
            if attempt == retries - 1:
                logging.error(f"  ❌ Failed after {retries} attempts: {e}")
    
    # Return error structure on failure
    return {
        "persona_name": "LLM_Error",
        "justification": f"LLM call failed after {retries} attempts",
        "key_topics": [],
        "qualitative_summary": {
            "persona_summary": "Error",
            "key_motivations": [],
            "common_praises": [],
            "common_criticisms": [],
            "core_characteristics": [],
            "potential_goals": []
        }
    }


def get_checkpoint_path(checkpoint_dir: Path, cluster_name: str) -> Path:
    """Get checkpoint file path for a cluster."""
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    return checkpoint_dir / f"{cluster_name}_checkpoint.pkl"


def load_checkpoint(checkpoint_dir: Path, cluster_name: str) -> Tuple[Set[int], int]:
    """Load processed indices and counter from checkpoint."""
    checkpoint_path = get_checkpoint_path(checkpoint_dir, cluster_name)
    if checkpoint_path.exists():
        try:
            with open(checkpoint_path, 'rb') as f:
                checkpoint_data = pickle.load(f)
                processed = checkpoint_data.get('processed_indices', set())
                counter = checkpoint_data.get('micro_cluster_counter', 0)
                logging.info(f"📂 Loaded checkpoint: {len(processed)} users already processed, counter: {counter}")
                return processed, counter
        except Exception as e:
            logging.warning(f"Could not load checkpoint: {e}")
    return set(), 0


def save_checkpoint(checkpoint_dir: Path, cluster_name: str, processed_indices: Set[int], micro_cluster_counter: int):
    """Save checkpoint with processed indices."""
    checkpoint_path = get_checkpoint_path(checkpoint_dir, cluster_name)
    try:
        checkpoint_data = {
            'processed_indices': processed_indices,
            'micro_cluster_counter': micro_cluster_counter,
            'timestamp': time.time()
        }
        with open(checkpoint_path, 'wb') as f:
            pickle.dump(checkpoint_data, f)
    except Exception as e:
        logging.warning(f"Could not save checkpoint: {e}")


def batch_cosine_similarity(seed_embedding: np.ndarray, all_embeddings: np.ndarray, 
                           batch_size: int = BATCH_SIZE_SIMILARITY) -> np.ndarray:
    """Calculate cosine similarity in batches to save memory."""
    similarities = np.zeros(all_embeddings.shape[0])
    for i in range(0, all_embeddings.shape[0], batch_size):
        end_idx = min(i + batch_size, all_embeddings.shape[0])
        batch = all_embeddings[i:end_idx]
        batch_similarities = cosine_similarity(seed_embedding, batch)[0]
        similarities[i:end_idx] = batch_similarities
    return similarities


def run_clustering_pass(embeddings: np.ndarray, metadata: List[dict], all_user_data: dict,
                       processed_indices: Set[int], cluster_name: str, segment_id: str,
                       artifact_dir: Path, checkpoint_dir: Path, client: OpenAI,
                       prompt_template: str, args: dict, start_counter: int = 0, 
                       relaxed: bool = False) -> Tuple[Set[int], int, Dict[str, dict]]:
    """
    Run a single clustering pass with given parameters.
    Returns: (processed_indices, micro_cluster_counter, user_tribes_dict)
    """
    total_items = len(embeddings)
    
    # Adjust parameters for relaxed pass
    if relaxed:
        similarity_threshold = max(0.5, args['similarity_threshold'] - 0.15)
        min_user_count = max(2, args['min_user_count'] - 2)
        max_user_count = args['max_user_count'] + 10
        min_review_count = max(2, args['min_review_count'] - 2)
        cluster_prefix = "relaxed_micro"
        logging.info(f"\n{'='*60}")
        logging.info(f"🔄 RELAXED PASS for unprocessed users in {cluster_name}")
        logging.info(f"{'='*60}")
    else:
        similarity_threshold = args['similarity_threshold']
        min_user_count = args['min_user_count']
        max_user_count = args['max_user_count']
        min_review_count = args['min_review_count']
        cluster_prefix = "micro"
    
    micro_cluster_counter = start_counter
    user_tribes_dict = {}  # Track user -> tribe assignments
    
    unprocessed_count = total_items - len(processed_indices)
    if unprocessed_count == 0:
        logging.info(f"  ℹ️  No unprocessed users. Skipping {'relaxed' if relaxed else 'main'} pass.")
        return processed_indices, micro_cluster_counter, user_tribes_dict
    
    logging.info(f"  Processing {unprocessed_count} unprocessed users")
    
    # Clustering loop
    for i in range(total_items):
        if i in processed_indices:
            continue 

        seed_embedding = embeddings[i:i+1]
        if total_items > BATCH_SIZE_SIMILARITY:
            similarities = batch_cosine_similarity(seed_embedding, embeddings, BATCH_SIZE_SIMILARITY)
        else:
            similarities = cosine_similarity(seed_embedding, embeddings)[0]
        similar_indices = np.where(similarities >= similarity_threshold)[0]
        
        new_cluster_indices = [idx for idx in similar_indices if idx not in processed_indices]
        
        if len(new_cluster_indices) > max_user_count:
            sorted_indices = sorted(new_cluster_indices, key=lambda idx: similarities[idx], reverse=True)
            new_cluster_indices = sorted_indices[:max_user_count]

        potential_member_users = [metadata[idx] for idx in new_cluster_indices]
        
        all_reviews_for_cluster = []
        all_chars_for_cluster = []
        user_ids_in_cluster = set()
        
        for user in potential_member_users:
            user_id = user.get('user_id')
            if not user_id:
                continue
                
            user_ids_in_cluster.add(user_id)
            user_data = all_user_data.get(user_id)
            
            if user_data and isinstance(user_data, dict):
                if "reviews" in user_data:
                    reviews_list = user_data["reviews"]
                    if isinstance(reviews_list, list):
                        if len(all_reviews_for_cluster) < MAX_REVIEWS_PER_CLUSTER:
                            for review in reviews_list:
                                if isinstance(review, dict):
                                    if len(all_reviews_for_cluster) >= MAX_REVIEWS_PER_CLUSTER:
                                        break
                                    review_copy = review.copy() 
                                    review_copy["user_id"] = user_id 
                                    all_reviews_for_cluster.append(review_copy)
                
                # Support both new and old field names for backward compatibility
                overall_chars_key = "overall_characteristics" if "overall_characteristics" in user_data else ("Overall_characteristics" if "Overall_characteristics" in user_data else "llm_characteristics")
                if overall_chars_key in user_data:
                    current_chars_len = sum(len(str(c.get('characteristic_summary', ''))) for c in all_chars_for_cluster)
                    if current_chars_len < MAX_CHARS_PER_CLUSTER:
                        overall_chars = user_data[overall_chars_key]
                        if isinstance(overall_chars, dict):
                            if "influencing_characteristics_summary" in overall_chars:
                                summary = overall_chars["influencing_characteristics_summary"]
                                if len(str(summary)) > 2000:
                                    summary = str(summary)[:2000] + "..."
                                all_chars_for_cluster.append({
                                    "user_id": user_id,
                                    "characteristic_summary": summary
                                })
        
        if len(user_ids_in_cluster) < min_user_count:
            processed_indices.add(i)
            continue
        
        if len(all_reviews_for_cluster) < min_review_count:
            processed_indices.add(i)
            continue

        # Found a valid cluster
        logging.info(f"  ✅ Found {'relaxed ' if relaxed else ''}micro-cluster: {len(user_ids_in_cluster)} users, {len(all_reviews_for_cluster)} reviews")
        logging.info(f"  📊 Analyzing with LLM...")
        
        llm_summary = summarize_micro_cluster(
            all_reviews_for_cluster, all_chars_for_cluster, 
            args['llm_model'], client, prompt_template
        )
        persona_name = llm_summary.get('persona_name', 'LLM_Error')
        logging.info(f"  ✅ Defined Micro-Persona: '{persona_name}'")

        quantitative_summary = calculate_quantitative_summary(all_reviews_for_cluster)

        # Create tribe ID
        micro_cluster_id_num = f"{cluster_prefix}_{micro_cluster_counter}"
        tribe_id = f"{segment_id}_{micro_cluster_id_num}"
        
        # Create tribe metadata
        tribe_metadata = {
            "total_users": len(user_ids_in_cluster),
            "total_reviews": len(all_reviews_for_cluster),
            "persona_name": persona_name,
            "relaxed_criteria": relaxed
        }
        
        # Assign users to tribe
        for user_id in user_ids_in_cluster:
            # Get similarity score for this user
            user_idx = next((idx for idx, m in enumerate(metadata) if m.get('user_id') == user_id), None)
            if user_idx is not None:
                # Clamp similarity score to [0.0, 1.0] to handle floating point precision issues
                similarity = float(similarities[user_idx])
                similarity = max(0.0, min(1.0, similarity))  # Clamp to [0.0, 1.0]
            else:
                similarity = None
            
            user_tribes_dict[user_id] = {
                "user_id": user_id,
                "segment_id": segment_id,
                "tribe_id": tribe_id,
                "tribe_name": persona_name,
                "similarity_score": similarity,
                "tribe_metadata": tribe_metadata
            }
        
        # Save summary and detail files (for backward compatibility)
        summary_subdir = artifact_dir / "micro_cluster_summaries" / cluster_name
        detail_subdir = artifact_dir / "micro_cluster_details" / cluster_name
        summary_subdir.mkdir(parents=True, exist_ok=True)
        detail_subdir.mkdir(parents=True, exist_ok=True)
        
        summary_data = {
            "persona_name": persona_name,
            "micro_cluster_id": tribe_id,
            "total_users_in_cluster": len(user_ids_in_cluster),
            "total_reviews_from_cluster": len(all_reviews_for_cluster),
            "quantitative_summary": quantitative_summary,
            "qualitative_summary": llm_summary.get("qualitative_summary"),
            "relaxed_criteria": relaxed
        }
        
        detail_data = {
            "persona_name": persona_name,
            "justification": llm_summary.get("justification"),
            "key_topics": llm_summary.get("key_topics"),
            "member_user_characteristics": all_chars_for_cluster,
            "members_grouped_by_user": {
                user_id: [r for r in all_reviews_for_cluster if r.get('user_id') == user_id]
                for user_id in user_ids_in_cluster
            },
            "relaxed_criteria": relaxed
        }
        
        summary_file = summary_subdir / f"{micro_cluster_id_num}_summary.json"
        detail_file = detail_subdir / f"{micro_cluster_id_num}_details.json"
        
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary_data, f, indent=4, ensure_ascii=False)
        
        with open(detail_file, 'w', encoding='utf-8') as f:
            json.dump(detail_data, f, indent=4, ensure_ascii=False)
        
        processed_indices.update(new_cluster_indices)
        coverage = len(processed_indices) / total_items
        logging.info(f"  📈 Coverage: {len(processed_indices)}/{total_items} users ({coverage:.1%})")
        
        micro_cluster_counter += 1
        
        if relaxed:
            if len(processed_indices) >= total_items:
                logging.info(f"  ✅ All users processed in relaxed pass")
                break
        else:
            coverage_goal = args['coverage_goal']
            min_clusters = args['min_clusters_to_find']
            if coverage >= coverage_goal and micro_cluster_counter >= min_clusters:
                logging.info(f"  ✅ Reached {coverage_goal:.0%} coverage with {micro_cluster_counter} clusters.")
                break
    
    return processed_indices, micro_cluster_counter, user_tribes_dict


def main():
    """Main execution function."""
    
    # Load configuration
    load_config()
    stage_config = get_stage_config("06_tribe_formation")
    openai_config = get_openai_config()
    
    # Get artifact names from config (required, no fallbacks)
    if "segment_details_input_artifact" not in stage_config:
        logging.error("Missing required config field: segment_details_input_artifact")
        return
    
    segment_details_artifact = stage_config["segment_details_input_artifact"]
    if "segment_details_filename_pattern" not in stage_config:
        logging.error("Missing required config field: segment_details_filename_pattern")
        return
    segment_details_filename_pattern = stage_config["segment_details_filename_pattern"]
    
    if "tribe_input_artifact" not in stage_config:
        logging.error("Missing required config field: tribe_input_artifact")
        return
    
    user_segments_artifact = stage_config["tribe_input_artifact"]
    if "tribe_segments_filename" not in stage_config:
        logging.error("Missing required config field: tribe_segments_filename")
        return
    tribe_segments_filename = stage_config["tribe_segments_filename"]
    
    if "tribe_output_artifact" not in stage_config:
        logging.error("Missing required config field: tribe_output_artifact")
        return
    
    output_artifact = stage_config["tribe_output_artifact"]
    
    # Get hyperparameters from config (required, no fallbacks)
    if "hyperparameters" not in stage_config:
        logging.error("Missing required config field: hyperparameters")
        return
    
    hyperparams = stage_config["hyperparameters"]
    
    if "tribe_hyperparameters" not in hyperparams:
        logging.error("Missing required config field: hyperparameters.tribe_hyperparameters")
        return
    
    tribe_hyperparams = hyperparams["tribe_hyperparameters"]
    
    # Validate required tribe hyperparameters
    required_tribe_params = ["llm_model", "similarity_threshold", "min_user_count", "max_user_count", 
                             "min_review_count", "coverage_goal", "min_clusters_to_find", "process_unprocessed"]
    for param in required_tribe_params:
        if param not in tribe_hyperparams:
            logging.error(f"Missing required hyperparameter: tribe_hyperparameters.{param}")
            return
    
    llm_model = tribe_hyperparams["llm_model"]
    base_url = os.environ.get("OPENAI_BASE_URL", "").strip()
    if base_url and "bedrock-mantle" in base_url and stage_config.get("bedrock_model_id"):
        llm_model = stage_config["bedrock_model_id"]

    # Build args from config (all required, no fallbacks)
    args = {
        **hyperparams,
        **tribe_hyperparams,
        "llm_model": llm_model,
        "similarity_threshold": tribe_hyperparams["similarity_threshold"],
        "min_user_count": tribe_hyperparams["min_user_count"],
        "max_user_count": tribe_hyperparams["max_user_count"],
        "min_review_count": tribe_hyperparams["min_review_count"],
        "coverage_goal": tribe_hyperparams["coverage_goal"],
        "min_clusters_to_find": tribe_hyperparams["min_clusters_to_find"],
        "process_unprocessed": tribe_hyperparams["process_unprocessed"]
    }
    
    # Initialize W&B run
    run = init_wandb_run(
        run_name="user_to_tribe_via_segments",
        stage="06_tribe_formation",
        config={
            "description": "Micro-clustering within segments to form tribes",
            "segment_details_input": segment_details_artifact,
            "user_segments_input": user_segments_artifact,
            "output_artifact": output_artifact,
            "schema_version": "v4",
            **args
        }
    )
    
    try:
        # Initialize client. Uses Bedrock when OPENAI_BASE_URL is set.
        if not (openai_config.get("api_key") or os.environ.get("OPENAI_API_KEY")):
            logging.error("❌ OPENAI_API_KEY not found in config or environment")
            return
        client = create_openai_client(openai_config=openai_config, timeout=120.0)
        
        # Load prompt template
        prompt_template = load_prompt("micro_cluster_persona_prompt.txt")
        
        # =====================================================================
        # Step 1: Load segment user details from W&B (no local fallback)
        # =====================================================================
        logging.info(f"Loading segment user details from: {segment_details_artifact}")
        
        # Download segment details artifact from W&B (required, no local fallback)
        segment_details_path = use_artifact(run, segment_details_artifact, artifact_type="dataset")
        
        if segment_details_path is None:
            logging.error(f"Could not download segment details artifact: {segment_details_artifact}")
            logging.error(f"Make sure to run 06_tribe_formation/scripts/01b_export_segment_user_details.py first and upload the artifact to W&B")
            return
        
        # Resolve path to handle any symlinks or relative paths
        # W&B may return path with :v0, :v1, etc. (invalid on Linux), need to handle this
        segment_details_path_str = str(segment_details_path)
        if not Path(segment_details_path).exists():
            # Replace :v0, :v1, :v2, etc. with -v0, -v1, -v2, etc.
            segment_details_path_str = re.sub(r':(v\d+)', r'-\1', segment_details_path_str)
            segment_details_path = Path(segment_details_path_str)
        
        segment_details_path = Path(segment_details_path).resolve()
        logging.info(f"[W&B] Segment details artifact downloaded to: {segment_details_path}")
        
        # Find all segment detail files using pattern from config
        segment_files = list(segment_details_path.glob(segment_details_filename_pattern))
        if not segment_files:
            logging.error(f"No segment detail files found in {segment_details_path}")
            logging.error(f"Expected pattern: {segment_details_filename_pattern}")
            
            # List available files for debugging
            if segment_details_path.exists():
                available_files = list(segment_details_path.glob("*.json"))
                if available_files:
                    logging.error(f"Available JSON files in artifact:")
                    for f in available_files:
                        logging.error(f"  - {f.name}")
                else:
                    logging.error(f"No JSON files found in artifact directory: {segment_details_path}")
            else:
                logging.error(f"Artifact directory does not exist: {segment_details_path}")
            
            logging.error(f"Please update config.yaml with the correct segment_details_filename_pattern from the list above")
            return
        
        logging.info(f"Found {len(segment_files)} segment files to process")
        
        # =====================================================================
        # Step 2: Load user segments from W&B (no local fallback)
        # =====================================================================
        logging.info(f"Loading user segments from: {user_segments_artifact}")
        
        # Download user segments artifact from W&B (required, no local fallback)
        user_segments_path = use_artifact(run, user_segments_artifact, artifact_type="dataset")
        
        if user_segments_path is None:
            logging.error(f"Could not download user segments artifact: {user_segments_artifact}")
            logging.error(f"Make sure to run 06_tribe_formation/scripts/01_user_to_segments_mapper.py first and upload the artifact to W&B")
            return
        
        # Resolve path to handle any symlinks or relative paths
        # W&B may return path with :v0, :v1, etc. (invalid on Linux), need to handle this
        user_segments_path_str = str(user_segments_path)
        if not Path(user_segments_path).exists():
            # Replace :v0, :v1, :v2, etc. with -v0, -v1, -v2, etc.
            user_segments_path_str = re.sub(r':(v\d+)', r'-\1', user_segments_path_str)
            user_segments_path = Path(user_segments_path_str)
        
        user_segments_path = Path(user_segments_path).resolve()
        logging.info(f"[W&B] User segments artifact downloaded to: {user_segments_path}")
        
        # Get segments file using filename from config (required, no fallback)
        segments_file = user_segments_path / tribe_segments_filename
        
        # Check if file exists - if not, list available files for debugging
        if not segments_file.exists():
            logging.error(f"Segments file not found in artifact: {segments_file}")
            logging.error(f"Expected file: {tribe_segments_filename}")
            
            # List available files for debugging (but don't use them)
            if user_segments_path.exists():
                available_files = list(user_segments_path.glob("*.json"))
                if available_files:
                    logging.error(f"Available JSON files in artifact:")
                    for f in available_files:
                        logging.error(f"  - {f.name}")
                else:
                    logging.error(f"No JSON files found in artifact directory: {user_segments_path}")
            else:
                logging.error(f"Artifact directory does not exist: {user_segments_path}")
            
            logging.error(f"Please update config.yaml with the correct tribe_segments_filename from the list above")
            return
        
        logging.info(f"[OK] Using segments file: {segments_file.name}")
        
        user_segments = UserSegmentsArtifact.from_file(segments_file)
        logging.info(f"Loaded segments for {len(user_segments)} users")
        
        # Create embeddings lookup by segment
        segment_embeddings = defaultdict(dict)  # segment_id -> {user_id: embedding}
        for user_id, segment_artifact in user_segments.items():
            segment_id = segment_artifact.segment_id
            segment_embeddings[segment_id][user_id] = segment_artifact.user_embedding
        
        # =====================================================================
        # Step 3: Process each segment
        # =====================================================================
        artifact_dir = get_artifact_dir("06_tribe_formation", output_artifact)
        artifact_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_dir = artifact_dir / "checkpoints"
        
        all_user_tribes = {}  # user_id -> UserTribeArtifact
        
        for segment_file in sorted(segment_files):
            # Extract segment number
            segment_number = segment_file.stem.replace("details_cluster_", "")
            segment_id = f"segment_{segment_number}"
            cluster_name = f"cluster_{segment_number}"
            
            logging.info(f"\n{'='*60}")
            logging.info(f"Processing Segment: {segment_id} ({cluster_name})")
            logging.info(f"{'='*60}")
            
            # Load segment user data (format: {user_id: {reviews: [...], ...}})
            try:
                with open(segment_file, 'r', encoding='utf-8') as f:
                    segment_data = json.load(f)
                logging.info(f"✅ Loaded user data for {len(segment_data)} users from segment")
            except Exception as e:
                logging.error(f"❌ Error loading segment data: {e}")
                continue
            
            # Get embeddings for this segment
            if segment_id not in segment_embeddings:
                logging.warning(f"⚠️  No embeddings found for {segment_id}, skipping")
                continue
            
            embeddings_dict = segment_embeddings[segment_id]
            
            # Match user IDs and create embeddings array
            user_ids = []
            embeddings_list = []
            metadata = []
            
            for user_id in segment_data.keys():
                if user_id in embeddings_dict:
                    user_ids.append(user_id)
                    embeddings_list.append(embeddings_dict[user_id])
                    metadata.append({'user_id': user_id})
            
            if not embeddings_list:
                logging.warning(f"⚠️  No matching embeddings found for segment {segment_id}, skipping")
                continue
            
            embeddings = np.array(embeddings_list).astype('float32')
            total_items = len(embeddings)
            
            logging.info(f"Starting micro-clustering for {total_items} users")
            logging.info(f"  Similarity threshold: {args['similarity_threshold']}")
            logging.info(f"  User range: {args['min_user_count']}-{args['max_user_count']} per cluster")
            
            # Load checkpoint
            processed_indices, micro_cluster_counter = load_checkpoint(checkpoint_dir, cluster_name)
            if processed_indices:
                logging.info(f"  Resuming: {len(processed_indices)} users already processed")
            
            # Main clustering pass
            processed_indices, micro_cluster_counter, user_tribes = run_clustering_pass(
                embeddings, metadata, segment_data, processed_indices, cluster_name, segment_id,
                artifact_dir, checkpoint_dir, client, prompt_template, args, 
                start_counter=micro_cluster_counter, relaxed=False
            )
            
            # Save checkpoint
            save_checkpoint(checkpoint_dir, cluster_name, processed_indices, micro_cluster_counter)
            
            # Relaxed pass if needed
            unprocessed_count = total_items - len(processed_indices)
            if unprocessed_count > 0 and args['process_unprocessed']:
                logging.info(f"\n  📋 {unprocessed_count} users remain unprocessed")
                processed_indices, relaxed_count, relaxed_tribes = run_clustering_pass(
                    embeddings, metadata, segment_data, processed_indices, cluster_name, segment_id,
                    artifact_dir, checkpoint_dir, client, prompt_template, args, 
                    start_counter=0, relaxed=True
                )
                user_tribes.update(relaxed_tribes)
                logging.info(f"  ✅ Relaxed pass: {relaxed_count} additional clusters")
            
            # Validate and add to all_user_tribes
            for user_id, tribe_data in user_tribes.items():
                try:
                    tribe_artifact = UserTribeArtifact.from_dict(tribe_data)
                    all_user_tribes[user_id] = tribe_artifact
                except Exception as e:
                    logging.warning(f"Validation error for user {user_id}: {e}")
            
            logging.info(f"✅ Finished processing {cluster_name}: {len(user_tribes)} users assigned to tribes")
        
        # =====================================================================
        # Step 4: Save user tribes as learned artifact
        # =====================================================================
        logging.info(f"\nSaving user tribes learned artifact...")
        
        tribes_output_file = artifact_dir / "user_tribes.json"
        tribes_output_data = {
            user_id: tribe.to_dict()
            for user_id, tribe in all_user_tribes.items()
        }
        
        with open(tribes_output_file, 'w', encoding='utf-8') as f:
            json.dump(tribes_output_data, f, indent=2, ensure_ascii=False)
        
        logging.info(f"Saved {len(all_user_tribes)} user tribe assignments")
        
        # =====================================================================
        # Step 5: Log to W&B
        # =====================================================================
        logging.info("Logging artifact to W&B...")
        
        # Calculate statistics
        tribes_by_segment = defaultdict(int)
        tribes_by_tribe = defaultdict(int)
        for tribe in all_user_tribes.values():
            tribes_by_segment[tribe.segment_id] += 1
            tribes_by_tribe[tribe.tribe_id] += 1
        
        artifact_metadata = {
            "num_users": len(all_user_tribes),
            "num_segments": len(tribes_by_segment),
            "num_tribes": len(tribes_by_tribe),
            "tribes_per_segment": {seg: count for seg, count in tribes_by_segment.items()},
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
            "num_users": len(all_user_tribes),
            "num_segments": len(tribes_by_segment),
            "num_tribes": len(tribes_by_tribe),
            "avg_users_per_tribe": len(all_user_tribes) / len(tribes_by_tribe) if tribes_by_tribe else 0
        })
        
        log_summary(run, {
            "status": "completed",
            "users_processed": len(all_user_tribes),
            "tribes_created": len(tribes_by_tribe)
        })
        
        logging.info("Tribe formation completed successfully!")
        
    except Exception as e:
        logging.error(f"Error in main execution: {e}", exc_info=True)
        log_summary(run, {"status": "failed", "error": str(e)})
        raise
    
    finally:
        finish_run(run)


if __name__ == "__main__":
    main()
