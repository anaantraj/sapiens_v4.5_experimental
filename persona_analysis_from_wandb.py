import os
import re
import random
import logging
import time
import json
from pathlib import Path
from typing import List, Optional, Dict, Any
from openai import OpenAI
from openai import RateLimitError, APIError
import wandb
import yaml

# Import W&B utilities
import sys
sys.path.append(str(Path(__file__).parent))
from utils.wandb_utils import download_artifact, load_stage_config_file

# --- Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Configuration ---
CONFIG_PATH = "10_running_simulations/config.yaml"
SUMMARY_OUTPUT_FILE = "persona_analysis.json"
CHECKPOINT_FILE = "labeling_checkpoint.json"  # For resume capability
INDIVIDUAL_DIR = "individual_persona_files"
MAX_REVIEW_CHAR_LIMIT = 300000 
MAX_REVIEWS_TO_LOAD = 10000  # Limit reviews loaded into memory
LLM_MODEL = "o3"
MAX_RETRIES = 3
RETRY_DELAY = 5  # seconds
API_RATE_LIMIT_DELAY = 1  # seconds between API calls

# Initialize OpenAI Client
try:
    api_key = os.getenv("OPENAI_API_KEY")
    if api_key is None:
        raise ValueError("OPENAI_API_KEY environment variable not set")
    logging.info("✅ OpenAI API key found")
    client = OpenAI(api_key=api_key)
except Exception as e:
    logging.error(f"❌ OpenAI API key error: {e}")
    print(f"CRITICAL ERROR: {e}")
    print("Please set the OPENAI_API_KEY environment variable.")
    exit(1)

def load_config() -> Dict[str, Any]:
    """Load configuration from 10_running_simulations/config.yaml"""
    try:
        config = load_stage_config_file("10_running_simulations", "config.yaml")
        if not config:
            raise ValueError("Config file is empty or not found")
        logging.info("✅ Loaded configuration from config.yaml")
        return config
    except Exception as e:
        logging.error(f"❌ Error loading config: {e}")
        raise

def search_artifacts_by_name(partial_name: str, artifact_type: str = "dataset", limit: int = 10) -> List[str]:
    """Search for artifacts with similar names in W&B"""
    try:
        config = load_config()
        wandb_config = config.get("wandb", {})
        entity = wandb_config.get("entity")
        project = wandb_config.get("project", "SAPIENS-FINAL")
        
        if not entity:
            entity = "pradeepbolleddu-vectorial-ai"
        
        api = wandb.Api()
        # Search for artifacts
        artifacts = api.artifacts(f"{entity}/{project}", type=artifact_type)
        
        matching = []
        for artifact in artifacts:
            if partial_name.lower() in artifact.name.lower():
                matching.append(artifact.name)
                if len(matching) >= limit:
                    break
        
        return matching
    except Exception as e:
        logging.debug(f"Could not search artifacts: {e}")
        return []

def download_tribe_seed_characteristics(config: Dict[str, Any]) -> Optional[Path]:
    """Download tribe_seed_characteristics artifact from W&B"""
    artifact_name = config.get("input_artifacts", {}).get("model")
    if not artifact_name:
        # Try alternative location in config
        artifact_name = config.get("simulation_config", {}).get("sgo_training", {}).get("input_artifacts", {}).get("tribe_seed_characteristics")
    
    # If still not found, try the known correct artifact name
    if not artifact_name:
        logging.warning("⚠️  Artifact name not found in config, using default: tribe_seed_characteristics_sampled_2kusers_10kreviews_v4:latest")
        artifact_name = "tribe_seed_characteristics_sampled_2kusers_10kreviews_v4:latest"
    
    logging.info(f"📥 Downloading artifact: {artifact_name}")
    
    # Parse artifact name to separate name and version (like use_artifact does)
    if ":" in artifact_name:
        name, version = artifact_name.rsplit(":", 1)
    else:
        name, version = artifact_name, "latest"
    
    # Try different artifact types (learned artifacts are usually "dataset")
    artifact_types = ["dataset", "model", "result"]
    
    # Also try different version formats and name variations
    versions_to_try = [version]
    if version == "latest":
        # Try without version (just the name)
        versions_to_try.append(None)
    else:
        # Try "latest" as fallback
        versions_to_try.append("latest")
    
    # If the name doesn't include "sampled_2kusers_10kreviews", try that variation too
    name_variations = [name]
    if "sampled_2kusers_10kreviews" not in name:
        # Try adding the sampled suffix
        name_variations.append(f"{name}_sampled_2kusers_10kreviews")
        # Also try replacing _v4 with _sampled_2kusers_10kreviews_v4
        if name.endswith("_v4"):
            base_name = name[:-3]  # Remove _v4
            name_variations.append(f"{base_name}_sampled_2kusers_10kreviews_v4")
    elif "sampled_2kusers_10kreviews" in name and not name.endswith("_sampled_2kusers_10kreviews_v4"):
        # Try the version without sampled suffix
        name_variations.append(name.replace("_sampled_2kusers_10kreviews", ""))
    
    for name_var in name_variations:
        for artifact_type in artifact_types:
            for ver in versions_to_try:
                if ver is None:
                    # Try to get the artifact without version (will use latest available)
                    logging.info(f"Trying name: {name_var}, type: {artifact_type} (no version specified)")
                else:
                    logging.info(f"Trying name: {name_var}, type: {artifact_type}, version: {ver}")
                
                try:
                    # Download artifact (without run tracking)
                    download_path = download_artifact(
                        artifact_name=name_var,
                        artifact_type=artifact_type,
                        version=ver if ver else "latest"
                    )
                    
                    if download_path:
                        # download_artifact returns a string path, convert to Path
                        path_obj = Path(download_path)
                        # Handle W&B path conversion (colon to dash)
                        if not path_obj.exists() and ':' in str(path_obj):
                            path_obj = Path(str(path_obj).replace(':', '-'))
                        logging.info(f"✅ Artifact downloaded to: {path_obj} (name: {name_var}, type: {artifact_type}, version: {ver or 'auto'})")
                        return path_obj
                except Exception as e:
                    logging.debug(f"Failed with name '{name_var}', type '{artifact_type}', version '{ver}': {e}")
                    continue
    
    # If all attempts failed, try to search for similar artifacts
    logging.warning(f"⚠️  Artifact not found with exact name: {name}")
    logging.info(f"🔍 Searching for similar artifacts...")
    
    similar_artifacts = []
    for artifact_type in artifact_types:
        found = search_artifacts_by_name(name, artifact_type=artifact_type, limit=5)
        similar_artifacts.extend(found)
    
    if similar_artifacts:
        logging.info(f"   Found similar artifacts:")
        for similar in set(similar_artifacts):
            logging.info(f"     - {similar}")
        logging.info(f"   Consider updating the config with one of these names")
    
    # Provide helpful error message
    logging.error(f"❌ Failed to download artifact: {artifact_name}")
    logging.error(f"   Tried artifact types: {', '.join(artifact_types)}")
    logging.error(f"   Tried versions: {', '.join(str(v) for v in versions_to_try)}")
    logging.error(f"   Please verify:")
    logging.error(f"   1. The artifact exists in W&B with name: {name}")
    logging.error(f"   2. The artifact has alias/version: {version}")
    logging.error(f"   3. You have access to the W&B project")
    logging.error(f"   4. Check W&B UI to see the exact artifact name and version")
    if similar_artifacts:
        logging.error(f"   5. Consider using one of the similar artifacts found above")
    return None

def extract_cluster_ids_from_artifact(artifact_path: Path) -> List[str]:
    """Extract cluster IDs from tribe_seed_characteristics artifact"""
    cluster_ids = []
    
    # Look for tribe_seed_characteristics.json file
    json_file = artifact_path / "tribe_seed_characteristics.json"
    
    if not json_file.exists():
        # Try to find any JSON file in the artifact
        json_files = list(artifact_path.glob("*.json"))
        if json_files:
            json_file = json_files[0]
            logging.info(f"Using JSON file: {json_file.name}")
        else:
            logging.error(f"❌ No JSON file found in artifact: {artifact_path}")
            return []
    
    try:
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Handle different data structures
        if isinstance(data, dict):
            # If it's a dict with tribe_ids as keys
            for tribe_id in data.keys():
                # Extract cluster number from tribe_id (e.g., "cluster_0_micro_5" -> "0")
                cluster_match = re.search(r"cluster_(-?\d+)", tribe_id)
                if cluster_match:
                    cluster_num = cluster_match.group(1)
                    if cluster_num not in cluster_ids:
                        cluster_ids.append(cluster_num)
                else:
                    # If no cluster pattern, use the tribe_id as is
                    if tribe_id not in cluster_ids:
                        cluster_ids.append(tribe_id)
        elif isinstance(data, list):
            # If it's a list of tribe objects
            for item in data:
                if isinstance(item, dict):
                    tribe_id = item.get("tribe_id", "")
                    cluster_match = re.search(r"cluster_(-?\d+)", tribe_id)
                    if cluster_match:
                        cluster_num = cluster_match.group(1)
                        if cluster_num not in cluster_ids:
                            cluster_ids.append(cluster_num)
        
        # Sort cluster IDs numerically
        try:
            cluster_ids = sorted(cluster_ids, key=lambda x: int(x) if x.lstrip('-').isdigit() else float('inf'))
        except:
            cluster_ids = sorted(cluster_ids)
        
        logging.info(f"✅ Extracted {len(cluster_ids)} unique cluster IDs")
        return cluster_ids
        
    except Exception as e:
        logging.error(f"❌ Error reading artifact JSON: {e}")
        return []

def get_cluster_summary_from_artifact(artifact_path: Path, cluster_id: str) -> Optional[str]:
    """Extract summary text for a specific cluster from the artifact"""
    json_file = artifact_path / "tribe_seed_characteristics.json"
    
    if not json_file.exists():
        json_files = list(artifact_path.glob("*.json"))
        if json_files:
            json_file = json_files[0]
    
    if not json_file.exists():
        return None
    
    try:
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Search for cluster data
        cluster_pattern = f"cluster_{cluster_id}"
        
        if isinstance(data, dict):
            # Find matching tribe_id
            for tribe_id, tribe_data in data.items():
                if cluster_pattern in tribe_id:
                    # Build summary from tribe data
                    summary_parts = []
                    
                    # Add persona name
                    persona_name = tribe_data.get("persona_name", "")
                    if persona_name:
                        summary_parts.append(f"Persona Name: {persona_name}")
                    
                    # Add quantitative summary
                    quant_summary = tribe_data.get("quantitative_summary", {})
                    if quant_summary:
                        summary_parts.append("\n## Quantitative Summary:")
                        if isinstance(quant_summary, dict):
                            for key, value in quant_summary.items():
                                summary_parts.append(f"{key}: {value}")
                        else:
                            summary_parts.append(str(quant_summary))
                    
                    # Add qualitative summary
                    qual_summary = tribe_data.get("qualitative_summary", {})
                    if qual_summary:
                        summary_parts.append("\n## Qualitative Summary:")
                        if isinstance(qual_summary, dict):
                            for key, value in qual_summary.items():
                                if isinstance(value, list):
                                    summary_parts.append(f"{key}: {', '.join(map(str, value))}")
                                else:
                                    summary_parts.append(f"{key}: {value}")
                        else:
                            summary_parts.append(str(qual_summary))
                    
                    # Add key topics
                    key_topics = tribe_data.get("key_topics", [])
                    if key_topics:
                        summary_parts.append(f"\n## Key Topics: {', '.join(key_topics)}")
                    
                    return "\n".join(summary_parts)
        
        elif isinstance(data, list):
            # Search in list
            for item in data:
                if isinstance(item, dict):
                    tribe_id = item.get("tribe_id", "")
                    if cluster_pattern in tribe_id:
                        # Build summary similarly
                        summary_parts = []
                        persona_name = item.get("persona_name", "")
                        if persona_name:
                            summary_parts.append(f"Persona Name: {persona_name}")
                        
                        quant_summary = item.get("quantitative_summary", {})
                        if quant_summary:
                            summary_parts.append("\n## Quantitative Summary:")
                            if isinstance(quant_summary, dict):
                                for key, value in quant_summary.items():
                                    summary_parts.append(f"{key}: {value}")
                        
                        qual_summary = item.get("qualitative_summary", {})
                        if qual_summary:
                            summary_parts.append("\n## Qualitative Summary:")
                            if isinstance(qual_summary, dict):
                                for key, value in qual_summary.items():
                                    if isinstance(value, list):
                                        summary_parts.append(f"{key}: {', '.join(map(str, value))}")
                                    else:
                                        summary_parts.append(f"{key}: {value}")
                        
                        key_topics = item.get("key_topics", [])
                        if key_topics:
                            summary_parts.append(f"\n## Key Topics: {', '.join(key_topics)}")
                        
                        return "\n".join(summary_parts)
        
        return None
        
    except Exception as e:
        logging.error(f"❌ Error extracting summary for cluster {cluster_id}: {e}")
        return None

def get_cluster_reviews_from_artifact(artifact_path: Path, cluster_id: str) -> List[str]:
    """Extract review texts for a specific cluster from the artifact"""
    json_file = artifact_path / "tribe_seed_characteristics.json"
    
    if not json_file.exists():
        json_files = list(artifact_path.glob("*.json"))
        if json_files:
            json_file = json_files[0]
    
    if not json_file.exists():
        return []
    
    try:
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        cluster_pattern = f"cluster_{cluster_id}"
        all_reviews = []
        
        if isinstance(data, dict):
            # Find matching tribe_id
            for tribe_id, tribe_data in data.items():
                if cluster_pattern in tribe_id:
                    # Extract reviews from member_user_characteristics
                    member_chars = tribe_data.get("member_user_characteristics", [])
                    for member in member_chars:
                        if isinstance(member, dict):
                            # Look for review_text or similar fields
                            review_text = member.get("review_text") or member.get("characteristic_summary", "")
                            if review_text and isinstance(review_text, str):
                                all_reviews.append(review_text.strip())
                    
                    # Also check members_grouped_by_user if available
                    members_grouped = tribe_data.get("members_grouped_by_user", {})
                    for user_id, user_data in members_grouped.items():
                        if isinstance(user_data, dict):
                            reviews = user_data.get("reviews", [])
                            if isinstance(reviews, list):
                                for review in reviews:
                                    if isinstance(review, dict):
                                        review_text = review.get("review_text", "")
                                        if review_text:
                                            all_reviews.append(review_text.strip())
        
        elif isinstance(data, list):
            # Search in list
            for item in data:
                if isinstance(item, dict):
                    tribe_id = item.get("tribe_id", "")
                    if cluster_pattern in tribe_id:
                        member_chars = item.get("member_user_characteristics", [])
                        for member in member_chars:
                            if isinstance(member, dict):
                                review_text = member.get("review_text") or member.get("characteristic_summary", "")
                                if review_text and isinstance(review_text, str):
                                    all_reviews.append(review_text.strip())
        
        return all_reviews
        
    except Exception as e:
        logging.error(f"❌ Error extracting reviews for cluster {cluster_id}: {e}")
        return []

def build_llm_prompt(summary_content, all_reviews: List[str]):
    """
    Builds a prompt with the summary + as many reviews as fit in the char limit.
    Optimized for large datasets: uses list join instead of string concatenation.
    """
    # Sample reviews intelligently if we have too many
    if len(all_reviews) > MAX_REVIEWS_TO_LOAD:
        # Use stratified sampling: take from beginning, middle, and end
        sample_size = MAX_REVIEWS_TO_LOAD
        step = len(all_reviews) // sample_size
        sampled_reviews = [all_reviews[i] for i in range(0, len(all_reviews), step)][:sample_size]
        random.shuffle(sampled_reviews)
        all_reviews = sampled_reviews
        logging.info(f"Sampled {len(all_reviews)} reviews from larger dataset")
    else:
        random.shuffle(all_reviews)
    
    # Use list join for better performance
    review_separator = "\n\n---\n\n"
    review_parts = []
    current_chars = 0
    reviews_packed = 0
    separator_len = len(review_separator)
    
    for review in all_reviews:
        review_text = str(review).strip()
        if not review_text:
            continue
        
        review_len = len(review_text)
        if (current_chars + review_len + separator_len) > MAX_REVIEW_CHAR_LIMIT:
            break
        
        review_parts.append(review_text)
        current_chars += review_len + separator_len
        reviews_packed += 1
    
    sampled_reviews_text = review_separator.join(review_parts)
    logging.info(f"Packed {reviews_packed} reviews ({current_chars:,} chars) into the prompt.")

    # --- ENHANCED PROMPT (No change here) ---
    prompt = f"""
You are an expert market researcher and data analyst. Your task is to synthesize a large batch of customer data into a single, cohesive "Macro Persona."

You are provided with two sets of data:
1.  **Quantitative Summary:** A high-level statistical overview of the cluster.
2.  **Raw Reviews:** A large, representative sample of raw customer reviews from this cluster.

Based on *both* the summary and the raw reviews, provide a deep analysis in the following structured format.

**DO NOT** add any extra text before or after this format.

## Persona Name:
[Provide a single, creative, and memorable name for this persona (e.g., "TheFrustratedPowerUser," "ThePrice-ConsciousHobbyist").]

## Core Characteristics:
[Provide a 2-3 sentence summary of *who* this person is. What is their main goal or relationship with the product?]

## Key Motivations (Why they are here):
[Use a bulleted list to describe what *drives* this persona.]
* [Motivation 1 (e.g., "To find a cheap solution for...")]
* [Motivation 2 (e.g., "To get advanced features for...")]
* [Motivation 3 (e.g., "Seeking reliable customer support for...")]

## Common Praise (What they LIKE):
[Based *only* on the raw reviews, list the top 3-5 specific things this group consistently *likes*. Use bullet points.]
* [Praise 1 (e.g., "Loves the user interface")]
* [Praise 2 (e.g., "Frequently mentions the fast load times")]

## Common Criticism (What they DISLIKE):
[Based *only* on the raw reviews, list the top 3-5 specific things this group consistently *dislikes* or finds frustrating. Use bullet points.]
* [Criticism 1 (e.g., "Confusing pricing structure")]
* [Criticism 2 (e.g., "Bugs in the mobile app")]

## Macro Understanding & Opportunity:
[Synthesize everything. What is the single biggest *story* or *opportunity* this persona represents? What product gap, market opportunity, or key frustration stands out the most? This is your key strategic insight.]

---
### 1. Quantitative Summary
{summary_content}

---
### 2. Raw Reviews (Sample)
{sampled_reviews_text}
"""
    return prompt

def get_persona_analysis_from_llm(prompt: str, retries: int = MAX_RETRIES) -> Optional[dict]:
    """
    Sends the prompt to the LLM with retry logic and rate limiting.
    Parses the multi-section response.
    """
    for attempt in range(retries):
        try:
            if attempt > 0:
                delay = RETRY_DELAY * (2 ** (attempt - 1))  # Exponential backoff
                logging.warning(f"Retry attempt {attempt + 1}/{retries} after {delay}s...")
                time.sleep(delay)
            
            logging.info(f"Sending request to OpenAI API (attempt {attempt + 1}/{retries})...")
            response = client.chat.completions.create(
                model=LLM_MODEL,
                messages=[{"role": "user", "content": prompt}],
                timeout=120  # 2 minute timeout
            )
            content = response.choices[0].message.content
            logging.info("✅ Received response from API")
            
            # Rate limiting delay
            time.sleep(API_RATE_LIMIT_DELAY)
            
            # Parse response with regex
            name_match = re.search(r"## Persona Name:\s*(.*?)(?=## |\Z)", content, re.IGNORECASE | re.DOTALL)
            chars_match = re.search(r"## Core Characteristics:\s*(.*?)(?=## |\Z)", content, re.IGNORECASE | re.DOTALL)
            motive_match = re.search(r"## Key Motivations \(Why they are here\):\s*(.*?)(?=## |\Z)", content, re.IGNORECASE | re.DOTALL)
            praise_match = re.search(r"## Common Praise \(What they LIKE\):\s*(.*?)(?=## |\Z)", content, re.IGNORECASE | re.DOTALL)
            crit_match = re.search(r"## Common Criticism \(What they DISLIKE\):\s*(.*?)(?=## |\Z)", content, re.IGNORECASE | re.DOTALL)
            macro_match = re.search(r"## Macro Understanding & Opportunity:\s*(.*?)(?=---|\Z)", content, re.IGNORECASE | re.DOTALL)

            if not name_match:
                logging.error(f"Could not parse LLM response. Response preview: {content[:200]}...")
                if attempt < retries - 1:
                    continue  # Retry
                return None

            logging.info("✅ Successfully parsed LLM response")
            
            return {
                "name": name_match.group(1).strip() if name_match else "Unknown",
                "characteristics": chars_match.group(1).strip() if chars_match else "Not found",
                "motivations": motive_match.group(1).strip() if motive_match else "Not found",
                "common_praise": praise_match.group(1).strip() if praise_match else "Not found",
                "common_criticism": crit_match.group(1).strip() if crit_match else "Not found",
                "macro_opportunity": macro_match.group(1).strip() if macro_match else "Not found"
            }
            
        except RateLimitError as e:
            wait_time = RETRY_DELAY * (2 ** attempt)
            logging.warning(f"Rate limit hit. Waiting {wait_time}s before retry...")
            time.sleep(wait_time)
            if attempt == retries - 1:
                logging.error(f"❌ Rate limit exceeded after {retries} attempts")
                return None
                
        except APIError as e:
            logging.error(f"API error (attempt {attempt + 1}/{retries}): {e}")
            if attempt == retries - 1:
                logging.error(f"❌ API call failed after {retries} attempts")
                return None
                
        except Exception as e:
            logging.error(f"Unexpected error (attempt {attempt + 1}/{retries}): {e}")
            if attempt == retries - 1:
                logging.error(f"❌ Failed after {retries} attempts: {e}")
                return None
    
    return None

def load_checkpoint() -> set:
    """Load processed cluster IDs from checkpoint file."""
    if os.path.exists(CHECKPOINT_FILE):
        try:
            with open(CHECKPOINT_FILE, 'r') as f:
                checkpoint_data = json.load(f)
                processed = set(checkpoint_data.get('processed_clusters', []))
                logging.info(f"📂 Loaded checkpoint: {len(processed)} clusters already processed")
                return processed
        except Exception as e:
            logging.warning(f"Could not load checkpoint: {e}. Starting fresh.")
    return set()

def save_checkpoint(processed_clusters: set, results: List[dict]):
    """Save checkpoint with processed clusters and current results."""
    try:
        checkpoint_data = {
            'processed_clusters': list(processed_clusters),
            'results_count': len(results),
            'last_updated': time.time()
        }
        with open(CHECKPOINT_FILE, 'w') as f:
            json.dump(checkpoint_data, f, indent=2)
    except Exception as e:
        logging.warning(f"Could not save checkpoint: {e}")

# --- Main Execution ---
if __name__ == "__main__":
    all_persona_results = []
    
    # Create output directory
    os.makedirs(INDIVIDUAL_DIR, exist_ok=True)
    
    # Load checkpoint for resume capability
    processed_clusters = load_checkpoint()
    
    # Load configuration
    logging.info("📋 Loading configuration...")
    try:
        config = load_config()
    except Exception as e:
        logging.error(f"❌ Failed to load configuration: {e}")
        exit(1)
    
    # Download tribe_seed_characteristics artifact
    logging.info("📥 Downloading tribe_seed_characteristics artifact from W&B...")
    artifact_path = download_tribe_seed_characteristics(config)
    
    if not artifact_path or not artifact_path.exists():
        logging.error("❌ Failed to download artifact. Exiting.")
        exit(1)
    
    # Extract cluster IDs from artifact
    logging.info("🔍 Extracting cluster IDs from artifact...")
    cluster_ids = extract_cluster_ids_from_artifact(artifact_path)
    
    if not cluster_ids:
        logging.error("❌ No cluster IDs found in artifact. Exiting.")
        exit(1)
    
    logging.info(f"Found {len(cluster_ids)} clusters to process")
    if processed_clusters:
        remaining = len(cluster_ids) - len(processed_clusters)
        logging.info(f"  {len(processed_clusters)} already processed, {remaining} remaining")

    for cluster_id in cluster_ids:
        # Skip if already processed (checkpoint resume)
        if cluster_id in processed_clusters:
            logging.info(f"⏭️  Skipping Cluster #{cluster_id} (already processed)")
            continue
        
        logging.info(f"\n{'='*60}")
        logging.info(f"Processing Persona Cluster #{cluster_id}")
        logging.info(f"{'='*60}")
        
        # Get summary from artifact
        summary_content = get_cluster_summary_from_artifact(artifact_path, cluster_id)
        if not summary_content:
            logging.warning(f"⚠️  No summary found for cluster {cluster_id}. Skipping.")
            continue
        
        # Get reviews from artifact
        all_reviews = get_cluster_reviews_from_artifact(artifact_path, cluster_id)
        if all_reviews:
            logging.info(f"✅ Loaded {len(all_reviews):,} reviews from artifact")
        else:
            logging.warning(f"⚠️  No reviews found for cluster {cluster_id}. Continuing with summary only.")
            all_reviews = []

        # Build prompt and get LLM analysis
        logging.info(f"📝 Building prompt for cluster {cluster_id}...")
        final_prompt = build_llm_prompt(summary_content, all_reviews)
        
        if not final_prompt:
            logging.warning(f"⚠️  Could not build prompt for cluster {cluster_id}")
            continue
        
        llm_result = get_persona_analysis_from_llm(final_prompt)
        if not llm_result:
            logging.error(f"❌ Failed to get LLM analysis for cluster {cluster_id}")
            continue
        
        llm_result['cluster_id'] = cluster_id
        
        # Save individual persona file immediately
        individual_filename = f"persona_cluster_{cluster_id}.json"
        individual_filepath = os.path.join(INDIVIDUAL_DIR, individual_filename)
        try:
            with open(individual_filepath, 'w', encoding='utf-8') as f:
                json.dump(llm_result, f, indent=4, ensure_ascii=False)
            logging.info(f"💾 Saved individual persona to {individual_filepath}")
        except Exception as e:
            logging.error(f"❌ Could not save individual file: {e}")
        
        # Add to results and update checkpoint
        all_persona_results.append(llm_result)
        processed_clusters.add(cluster_id)
        save_checkpoint(processed_clusters, all_persona_results)
        
        logging.info(f"✅ Finished processing Cluster #{cluster_id}")

    # --- Save Final Summary File ---
    logging.info(f"\n{'='*60}")
    if all_persona_results:
        logging.info(f"💾 Saving final summary...")
        try:
            with open(SUMMARY_OUTPUT_FILE, 'w', encoding='utf-8') as f:
                json.dump(all_persona_results, f, indent=4, ensure_ascii=False)
            logging.info(f"✅ ANALYSIS COMPLETE!")
            logging.info(f"{'='*60}")
            logging.info(f"📁 Summary file: {SUMMARY_OUTPUT_FILE}")
            logging.info(f"📁 Individual files: {INDIVIDUAL_DIR}/")
            logging.info(f"📊 Total personas analyzed: {len(all_persona_results)}")
            logging.info(f"📊 Clusters processed: {len(processed_clusters)}")
            
            # Clean up checkpoint file on successful completion
            if os.path.exists(CHECKPOINT_FILE):
                try:
                    os.remove(CHECKPOINT_FILE)
                    logging.info(f"🧹 Cleaned up checkpoint file")
                except Exception as e:
                    logging.warning(f"Could not remove checkpoint file: {e}")
            
            logging.info(f"{'='*60}")
        except Exception as e:
            logging.error(f"❌ Error saving final summary: {e}")
            logging.info(f"💾 Checkpoint saved. You can resume later.")
    else:
        logging.warning("❌ No persona analyses were generated. Please check for errors.")
        if processed_clusters:
            logging.info(f"💾 Checkpoint saved with {len(processed_clusters)} processed clusters.")
        exit(1)

