"""
Test Mode Predictions Module
============================

When test_mode is enabled in SGO training, this module handles:
- Loading test set reviews
- Loading memory files from each cluster
- Making predictions using persona chars, user chars, category chars, product desc, and memory
"""

import os
import json
import logging
from pathlib import Path
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Tuple, Optional

# Import project root utilities
import sys
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from utils.wandb_utils import use_artifact
from sgo_training.utils import io_utils
from sgo_training.config import settings
from sgo_training.utils import logging_utils

# Import prediction library components
prediction_lib_path = str(project_root / "07_pre_sgo_predictions" / "scripts")
if prediction_lib_path not in sys.path:
    sys.path.insert(0, prediction_lib_path)

from prediction_lib.prompt_generation import create_enhanced_prompt, load_prompt
from prediction_lib.llm_client import get_llm_prediction, RateLimiter
from prediction_lib import config as lib_config


def load_memory_file(memory_dir: str, cluster_id: str, micro_id: str) -> Dict:
    """
    Load memory file (batch_analyses) for a specific cluster and micro cluster.
    
    Args:
        memory_dir: Base directory containing memory files
        cluster_id: Cluster ID (e.g., "cluster_0")
        micro_id: Micro cluster ID (e.g., "micro_0")
        
    Returns:
        Dictionary with memory data, or empty dict if not found
    """
    memory_file = os.path.join(memory_dir, cluster_id, f"{micro_id}_batch_analyses.json")
    if os.path.exists(memory_file):
        return io_utils.load_json_file(memory_file, {})
    return {}


def format_memory_for_prompt(memory_data: Dict, persona_name: str, user_id: str) -> str:
    """
    Format memory data (batch_analyses) into a string for prompt inclusion.
    Only extracts analyses for the specific user_id.
    
    Args:
        memory_data: Dictionary with batch data (e.g., "iteration_1_batch_1": {"reviews": [...], "analyses": [...]})
        persona_name: Name of the persona (not used but kept for compatibility)
        user_id: User ID to filter analyses for (e.g., "AE32INSVRFOUVTTXAAY3E6VXFUQA")
        
    Returns:
        Formatted string with only this user's analyses, or empty string if none found
    """
    user_analyses = []
    
    # Extract analyses only for this specific user
    for batch_key, batch_data in memory_data.items():
        if not isinstance(batch_data, dict):
            continue
        
        reviews = batch_data.get('reviews', [])
        analyses = batch_data.get('analyses', [])
        
        if not isinstance(reviews, list) or not isinstance(analyses, list):
            continue
        
        # Find all review indices for this user
        # Review keys are in format: "{user_id}_review_{index}"
        for idx, review_key in enumerate(reviews):
            if isinstance(review_key, str) and review_key.startswith(f"{user_id}_review_"):
                # Found a review for this user - get corresponding analysis
                if idx < len(analyses):
                    analysis = analyses[idx]
                    if analysis:  # Only add non-empty analyses
                        user_analyses.append(str(analysis))
    
    if user_analyses:
        # Format as past learnings
        past_learnings = ' | '.join(user_analyses)
        memory_text = f"""
**Past Learnings (Memory from SGO Training):**
{past_learnings}
"""
        # Print memory for debugging
        logging_utils.log_thread_safe(f"  📝 Memory for user {user_id}: {len(user_analyses)} analyses", 'info')
        logging.debug(f"  Memory content:\n{memory_text}")
        return memory_text
    logging.debug(f"  📝 No memory found for user {user_id}")
    return ""


def create_prompt_with_memory(
    persona_context: dict,
    user_char_summary: str,
    category_char_summary: str,
    product_description: str,
    category: str,
    category_themes: list,
    prompt_template: str,
    memory_text: str = ""
) -> str:
    """
    Create enhanced prompt with memory included.
    Extends create_enhanced_prompt to include memory.
    
    Args:
        persona_context: Persona context dictionary
        user_char_summary: User characteristics summary
        category_char_summary: Category-specific characteristics summary
        product_description: Product description
        category: Category name
        category_themes: List of themes for the category
        prompt_template: Base prompt template
        memory_text: Formatted memory text to append
        
    Returns:
        Complete prompt string
    """
    # First create the base prompt
    base_prompt = create_enhanced_prompt(
        persona_context=persona_context,
        user_char_summary=user_char_summary,
        category_char_summary=category_char_summary,
        product_description=product_description,
        category=category,
        category_themes=category_themes,
        prompt_template=prompt_template
    )
    
    # Append memory if provided
    if memory_text:
        # Insert memory after persona characteristics but before the task
        # Find a good insertion point (after category section or before themes)
        if "### 3. Your Category-Specific Behavior" in base_prompt:
            # Insert after category section
            insertion_point = base_prompt.find("### 3. Your Category-Specific Behavior")
            end_of_category = base_prompt.find("\n\n", insertion_point + 100)
            if end_of_category != -1:
                base_prompt = base_prompt[:end_of_category] + memory_text + base_prompt[end_of_category:]
            else:
                # Fallback: append before themes section
                themes_pos = base_prompt.find("Possible Review Themes:")
                if themes_pos != -1:
                    base_prompt = base_prompt[:themes_pos] + memory_text + "\n" + base_prompt[themes_pos:]
                else:
                    # Last resort: append at end
                    base_prompt = base_prompt + memory_text
        else:
            # Insert before themes section
            themes_pos = base_prompt.find("Possible Review Themes:")
            if themes_pos != -1:
                base_prompt = base_prompt[:themes_pos] + memory_text + "\n" + base_prompt[themes_pos:]
            else:
                # Last resort: append at end
                base_prompt = base_prompt + memory_text
    
    return base_prompt


def process_single_test_review(args_tuple):
    """
    Process a single test review prediction with memory.
    
    Args:
        args_tuple: (review, ground_truth_review, persona_context, user_char_summary, category_char_summary,
                    main_category, category_themes, memory_text, client, rate_limiter,
                    prompt_template, model_name, max_tokens, temperature, max_retries, scoring_mode, theme_prediction_model, product_description)
    
    Returns:
        dict with prediction data or None if error
    """
    (review, ground_truth_review, persona_context, user_char_summary, category_char_summary,
     main_category, category_themes, memory_text, client, rate_limiter,
     prompt_template, model_name, max_tokens, temperature, max_retries, scoring_mode, theme_prediction_model, product_description) = args_tuple
    
    # Create prompt with memory
    prompt = create_prompt_with_memory(
        persona_context=persona_context,
        user_char_summary=user_char_summary,
        category_char_summary=category_char_summary,
        product_description=review.get('product_description', ''),
        category=main_category,
        category_themes=category_themes,
        prompt_template=prompt_template,
        memory_text=memory_text
    )
    
    # Get LLM prediction (pass scoring_mode and theme_prediction_model to get logprobs)
    prediction = get_llm_prediction(
        prompt, client, rate_limiter, category_themes, model_name, 
        max_tokens, temperature, max_retries, scoring_mode, 
        theme_prediction_model, product_description
    )
    
    if prediction is None:
        return {
            'status': 'failed',
            'error': 'LLM call failed after all retries',
            'product_description': review.get('product_description', ''),
            'category': review.get('category'),
            'asin': review.get('asin'),
            'timestamp': review.get('timestamp'),
        }
    
    # Extract ground truth data (topic_probabilities, logprobs, review_text)
    actual_review_text = ground_truth_review.get('review_text', '') if ground_truth_review else review.get('review_text', '')
    actual_topic_probabilities = ground_truth_review.get('topic_probabilities', {}) if ground_truth_review else {}
    actual_topic_logprobs = ground_truth_review.get('topic_logprobs', {}) if ground_truth_review else {}
    
    # Build prediction dict with topic_probabilities and logprobs
    prediction_dict = {
        'review_text': prediction.get('review_text', ''),
        'rating': prediction.get('rating'),
        'sentiment': prediction.get('sentiment'),
        'predicted_themes': prediction.get('predicted_themes', {})
    }
    
    # Add topic_probabilities and logprobs to prediction if available
    if prediction.get('predicted_themes'):
        # Convert predicted_themes dict to topic_probabilities format
        prediction_dict['topic_probabilities'] = prediction.get('predicted_themes', {})
    if prediction.get('theme_logprobs'):
        prediction_dict['topic_logprobs'] = prediction.get('theme_logprobs', {})
    
    # Build actual dict with topic_probabilities, logprobs, and review_text
    actual_dict = {
        'review_text': actual_review_text,
        'rating': ground_truth_review.get('rating') if ground_truth_review else review.get('rating'),
        'sentiment': ground_truth_review.get('sentiment') if ground_truth_review else review.get('sentiment'),
        'predicted_themes': ground_truth_review.get('themes', ground_truth_review.get('predicted_themes', [])) if ground_truth_review else review.get('themes', review.get('predicted_themes', []))
    }
    
    # Add topic_probabilities and logprobs to actual if available
    if actual_topic_probabilities:
        actual_dict['topic_probabilities'] = actual_topic_probabilities
    if actual_topic_logprobs:
        actual_dict['topic_logprobs'] = actual_topic_logprobs
    
    # Return format expected by SGO training
    result = {
        'status': 'success',
        'product_description': review.get('product_description', ''),
        'category': review.get('category'),
        'asin': review.get('asin'),
        'timestamp': review.get('timestamp'),
        'prediction': prediction_dict,  # Contains: review_text, rating, sentiment, predicted_themes, topic_probabilities, topic_logprobs
        'actual': actual_dict  # Contains: review_text, rating, sentiment, predicted_themes, topic_probabilities, topic_logprobs
    }
    
    return result


def load_ground_truth_file(ground_truth_path: Path, cluster_id: str, micro_id: str) -> Dict:
    """
    Load ground truth file for a specific cluster and micro cluster.
    
    Args:
        ground_truth_path: Base path to ground truth artifact (folder structure)
        cluster_id: Cluster ID (e.g., "cluster_0")
        micro_id: Micro cluster ID (e.g., "micro_0")
        
    Returns:
        Dictionary with ground truth data organized by user_id, or empty dict if not found
    """
    if not ground_truth_path or not ground_truth_path.exists():
        return {}
    
    # Try different file naming patterns
    gt_file_path = ground_truth_path / cluster_id / f"{micro_id}_details.json"
    if not gt_file_path.exists():
        gt_file_path = ground_truth_path / cluster_id / f"{micro_id}_predictions.json"
    if not gt_file_path.exists():
        # Try alternative naming patterns
        cluster_dir = ground_truth_path / cluster_id
        if cluster_dir.exists():
            for alt_file in cluster_dir.glob(f"*{micro_id}*.json"):
                if 'summary' not in alt_file.name.lower() and 'grand' not in alt_file.name.lower():
                    gt_file_path = alt_file
                    break
    
    if not gt_file_path.exists():
        return {}
    
    try:
        with open(gt_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
            # Handle ground truth structure: members_grouped_by_user
            ground_truth_by_user = {}
            if 'members_grouped_by_user' in data:
                for user_id, reviews in data['members_grouped_by_user'].items():
                    if isinstance(reviews, list):
                        ground_truth_by_user[user_id] = reviews
                    elif isinstance(reviews, dict) and 'reviews' in reviews:
                        ground_truth_by_user[user_id] = reviews['reviews']
            
            # Handle standard prediction file structure
            elif 'user_predictions' in data:
                for user_id, user_data in data['user_predictions'].items():
                    if isinstance(user_data, dict) and 'reviews' in user_data:
                        ground_truth_by_user[user_id] = user_data['reviews']
                    elif isinstance(user_data, list):
                        ground_truth_by_user[user_id] = user_data
            
            return ground_truth_by_user
    except Exception as e:
        logging_utils.log_thread_safe(f"[WARNING] Could not load ground truth from {gt_file_path}: {e}", 'warning')
        return {}


def match_ground_truth_review(prediction_review: Dict, ground_truth_reviews: List[Dict]) -> Optional[Dict]:
    """
    Match a prediction review with its corresponding ground truth review.
    Matches by asin, user_id, and product_description.
    
    Args:
        prediction_review: Review from prediction file
        ground_truth_reviews: List of ground truth reviews for the same user
        
    Returns:
        Matched ground truth review or None
    """
    pred_asin = prediction_review.get('asin')
    pred_user_id = prediction_review.get('user_id')
    pred_product_desc = prediction_review.get('product_description', '').strip()
    
    for gt_review in ground_truth_reviews:
        gt_asin = gt_review.get('asin')
        gt_user_id = gt_review.get('user_id')
        gt_product_desc = gt_review.get('product_description', '').strip()
        
        # Match by asin first (most reliable)
        if pred_asin and gt_asin and pred_asin == gt_asin:
            return gt_review
        
        # Match by user_id + product_description (fallback)
        if pred_user_id and gt_user_id and pred_user_id == gt_user_id:
            if pred_product_desc and gt_product_desc and pred_product_desc == gt_product_desc:
                return gt_review
    
    return None


def run_test_mode_predictions(
    run,
    test_data_path: str,  # Path to predictions artifact (cluster directories)
    ground_truth_path: Optional[str] = None,  # Path to ground truth artifact (folder structure)
    tribe_seed_data: Dict = None,
    user_backstories: Dict = None,
    topic_universe: Dict = None,
    category_mapping: Dict = None,
    memory_base_dir: str = "",
    output_dir: str = "",
    prompt_template: str = "",
    client = None,
    rate_limiter = None,
    model_name: str = "o3",
    max_tokens: int = 2000,
    temperature: float = 0.7,
    max_retries: int = 3,
    num_workers: int = 10,
    target_clusters: Optional[List[str]] = None,
    target_micro: Optional[str] = None,  # Optional - filter by specific micro cluster (e.g., "micro_15")
    user_to_tribe_map: Optional[Dict] = None,  # Optional - users extracted from prediction files
    scoring_mode: str = "logprobs",  # Scoring mode: "confidence", "logprobs", or "logprobs-without-persona-context"
    theme_prediction_model: Optional[str] = None  # Model for theme classification with logprobs
):
    """
    Run test mode predictions for all test reviews using memory files.
    Processes cluster by cluster, micro cluster by micro cluster from predictions artifact.
    
    Args:
        run: W&B run object
        test_data_path: Path to predictions artifact (contains cluster_* directories)
        tribe_seed_data: Tribe seed characteristics
        user_backstories: User backstories
        user_to_tribe_map: Mapping from user_id to tribe_id
        topic_universe: Topic universe dictionary
        category_mapping: Category mapping dictionary
        memory_base_dir: Base directory containing memory files
        output_dir: Output directory for predictions
        prompt_template: Prompt template string
        client: OpenAI client
        rate_limiter: Rate limiter instance
        model_name: Model name for predictions
        max_tokens: Max tokens for non-o3 models
        temperature: Temperature for non-o3 models
        max_retries: Max retries for LLM calls
        num_workers: Number of parallel workers
        target_clusters: Optional list of cluster IDs to process
        
    Returns:
        Dictionary with results organized by cluster
    """
    logging_utils.log_thread_safe("[TEST MODE] Starting test mode predictions (processing cluster by cluster)...", 'info')
    
    test_data_path = Path(test_data_path)
    ground_truth_path = Path(ground_truth_path) if ground_truth_path else None
    
    if ground_truth_path:
        logging_utils.log_thread_safe(f"[TEST MODE] Using ground truth from: {ground_truth_path}", 'info')
    else:
        logging_utils.log_thread_safe("[TEST MODE] No ground truth path provided - will use data from prediction files only", 'warning')
    
    # Build theme map from topic universe
    category_themes_map = {}
    if isinstance(topic_universe, dict):
        for category, themes_data in topic_universe.items():
            if isinstance(themes_data, dict) and 'topics' in themes_data:
                category_themes_map[category] = themes_data['topics']
            elif isinstance(themes_data, list):
                category_themes_map[category] = themes_data
    
    # Filter tribes if target_clusters provided
    if target_clusters:
        filtered_tribes = {}
        for tribe_id, tribe_data in tribe_seed_data.items():
            # Parse tribe_id to get cluster_id
            if '_' in tribe_id:
                parts = tribe_id.split('_')
                if len(parts) >= 2:
                    cluster_id = f"{parts[0]}_{parts[1]}"
                    if cluster_id in target_clusters:
                        filtered_tribes[tribe_id] = tribe_data
        tribe_seed_data = filtered_tribes
        logging_utils.log_thread_safe(f"[TEST MODE] Filtered to {len(tribe_seed_data)} tribes for target clusters", 'info')
    
    # Organize results by cluster
    clusters_dict = defaultdict(lambda: defaultdict(dict))
    all_tribe_predictions = {}
    
    # Process each tribe
    for tribe_id, tribe_seed in tribe_seed_data.items():
        logging_utils.log_thread_safe(f"[TEST MODE] Processing tribe: {tribe_id}", 'info')
        
        # Parse tribe_id to get cluster and micro IDs
        if '_' in tribe_id:
            parts = tribe_id.split('_')
            if len(parts) >= 4:
                cluster_id = f"{parts[0]}_{parts[1]}"
                micro_id = f"{parts[2]}_{parts[3]}"
            else:
                cluster_id = tribe_id
                micro_id = "micro_0"
        else:
            cluster_id = tribe_id
            micro_id = "micro_0"
        
        # Filter by micro cluster if specified
        if target_micro and micro_id != target_micro:
            continue
        
        # Load memory file for this cluster (will be filtered per user later)
        memory_data = load_memory_file(memory_base_dir, cluster_id, micro_id)
        if memory_data:
            logging_utils.log_thread_safe(f"[TEST MODE] Loaded memory data for {cluster_id}/{micro_id} ({len(memory_data)} batch entries)", 'info')
        else:
            logging_utils.log_thread_safe(f"[TEST MODE] No memory data found for {cluster_id}/{micro_id}, proceeding without memory", 'warning')
        
        # Load ground truth for this cluster/micro
        ground_truth_by_user = {}
        if ground_truth_path:
            ground_truth_by_user = load_ground_truth_file(ground_truth_path, cluster_id, micro_id)
            if ground_truth_by_user:
                logging_utils.log_thread_safe(f"[TEST MODE] Loaded ground truth for {cluster_id}/{micro_id} ({len(ground_truth_by_user)} users)", 'info')
            else:
                logging_utils.log_thread_safe(f"[TEST MODE] No ground truth found for {cluster_id}/{micro_id}", 'warning')
        
        # Get persona context
        persona_context = {
            'persona_name': tribe_seed.persona_name if hasattr(tribe_seed, 'persona_name') else f'persona_{tribe_id}',
            'qualitative_summary': tribe_seed.qualitative_summary.model_dump() if hasattr(tribe_seed.qualitative_summary, 'model_dump') else tribe_seed.qualitative_summary
        }
        
        # Load user reviews from test_micro_cluster_details artifact file for this cluster/micro
        # Try different file naming patterns: micro_*_details.json, micro_*_predictions.json
        user_predictions = {}
        pred_file_path = test_data_path / cluster_id / f"{micro_id}_details.json"
        if not pred_file_path.exists():
            # Try predictions.json pattern
            pred_file_path = test_data_path / cluster_id / f"{micro_id}_predictions.json"
        if not pred_file_path.exists():
            # Try alternative naming patterns
            for alt_file in (test_data_path / cluster_id).glob(f"*{micro_id}*.json"):
                if 'summary' not in alt_file.name.lower() and 'grand' not in alt_file.name.lower():
                    pred_file_path = alt_file
                    break
        
        if pred_file_path.exists():
            try:
                with open(pred_file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    
                    # Handle test_micro_cluster_details structure: members_grouped_by_user
                    if 'members_grouped_by_user' in data:
                        for user_id, reviews in data['members_grouped_by_user'].items():
                            if isinstance(reviews, list):
                                user_predictions[user_id] = {'reviews': reviews}
                            elif isinstance(reviews, dict) and 'reviews' in reviews:
                                user_predictions[user_id] = reviews
                    
                    # Handle standard prediction file structure
                    elif 'user_predictions' in data:
                        user_predictions = data['user_predictions']
                    elif 'predictions' in data:
                        user_predictions = data['predictions']
                    elif isinstance(data, dict):
                        for user_id, user_data in data.items():
                            if isinstance(user_data, dict):
                                if 'reviews' in user_data:
                                    user_predictions[user_id] = user_data
                                elif 'actual' in user_data or 'predicted' in user_data:
                                    if user_id not in user_predictions:
                                        user_predictions[user_id] = {'reviews': []}
                                    user_predictions[user_id]['reviews'].append(user_data)
                            elif isinstance(user_data, list):
                                user_predictions[user_id] = {'reviews': user_data}
            except Exception as e:
                logging_utils.log_thread_safe(f"[WARNING] Could not load test data from {pred_file_path}: {e}", 'warning')
        
        if not user_predictions:
            logging_utils.log_thread_safe(f"[WARNING] No user predictions loaded for {cluster_id}/{micro_id}, skipping", 'warning')
            continue
        
        # Extract users directly from the prediction file (they're already organized by micro cluster)
        # No need for user_to_tribe_map - users in this file belong to this micro cluster
        tribe_users = list(user_predictions.keys())
        if not tribe_users:
            logging_utils.log_thread_safe(f"[WARNING] No users found in predictions file for {cluster_id}/{micro_id}, skipping", 'warning')
            continue
        
        logging_utils.log_thread_safe(f"[TEST MODE] Found {len(tribe_users)} users in {cluster_id}/{micro_id} (tribe: {tribe_id})", 'info')
        
        # Prepare review tasks
        all_review_tasks = []
        user_review_indices = defaultdict(list)
        
        for user_id in tribe_users:
            if user_id not in user_predictions:
                continue
            
            # Get user context
            user_story = user_backstories.get(user_id)
            user_summary = user_story.overall_characteristics.influencing_characteristics_summary if user_story else ""
            if not user_summary:
                continue
            
            # Format memory for this specific user (filter analyses by user_id)
            persona_name = tribe_seed.persona_name if hasattr(tribe_seed, 'persona_name') else f'persona_{tribe_id}'
            user_memory_text = format_memory_for_prompt(memory_data, persona_name, user_id) if memory_data else ""
            
            # Print memory for this user
            if user_memory_text:
                logging_utils.log_thread_safe(f"  📝 User {user_id} memory context:", 'info')
                logging_utils.log_thread_safe(f"     {user_memory_text.strip()}", 'info')
            else:
                logging_utils.log_thread_safe(f"  📝 User {user_id}: No memory found", 'info')
            
            user_data = user_predictions[user_id]
            reviews = user_data.get('reviews', []) if isinstance(user_data, dict) else []
            if not reviews:
                continue
            
            # Get ground truth reviews for this user
            user_ground_truth_reviews = ground_truth_by_user.get(user_id, [])
            
            for review in reviews:
                cat = review.get('category')
                if not cat:
                    continue
                
                main_cat = lib_config.map_category_to_main_category(cat, category_mapping.get('category_to_main_mapping', {}) if isinstance(category_mapping, dict) else category_mapping) or cat
                normalized_main = main_cat.upper().strip()
                
                # Get themes for this category
                themes = category_themes_map.get(normalized_main, [])
                if not themes:
                    # Try alternative category names
                    for alt_cat in category_themes_map.keys():
                        if normalized_main in alt_cat.upper() or alt_cat.upper() in normalized_main:
                            themes = category_themes_map[alt_cat]
                            break
                
                if not themes:
                    continue
                
                # Match ground truth review for this prediction review
                ground_truth_review = match_ground_truth_review(review, user_ground_truth_reviews) if user_ground_truth_reviews else None
                
                # Get category-specific characteristics
                cat_summary = ""
                if user_story and user_story.category_characteristics:
                    c_chars = user_story.category_characteristics.get(main_cat) or user_story.category_characteristics.get(cat)
                    if c_chars:
                        cat_summary = c_chars.influencing_characteristics_summary
                
                task = (review, ground_truth_review, persona_context, user_summary, cat_summary, main_cat, themes,
                       user_memory_text, client, rate_limiter, prompt_template, model_name,
                       max_tokens, temperature, max_retries, scoring_mode, theme_prediction_model, review.get('product_description', ''))
                
                user_review_indices[user_id].append(len(all_review_tasks))
                all_review_tasks.append(task)
        
        if not all_review_tasks:
            logging_utils.log_thread_safe(f"[TEST MODE] No review tasks for tribe {tribe_id}", 'warning')
            continue
        
        logging_utils.log_thread_safe(f"[TEST MODE] Created {len(all_review_tasks)} review tasks for tribe {tribe_id}", 'info')
        
        # Execute parallel processing
        tribe_user_predictions = {}
        tribe_failed_predictions = defaultdict(list)
        
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            future_to_task = {executor.submit(process_single_test_review, task): task for task in all_review_tasks}
            
            for future in as_completed(future_to_task):
                try:
                    result = future.result()
                    if result and result.get('status') == 'success':
                        # Extract user_id and review index from task
                        task_idx = all_review_tasks.index(future_to_task[future])
                        # Find which user this belongs to
                        for user_id, indices in user_review_indices.items():
                            if task_idx in indices:
                                review_idx = indices.index(task_idx)
                                if user_id not in tribe_user_predictions:
                                    tribe_user_predictions[user_id] = []
                                tribe_user_predictions[user_id].append(result)
                                break
                    elif result and result.get('status') == 'failed':
                        # Track failed predictions
                        for user_id, indices in user_review_indices.items():
                            if any(i < len(all_review_tasks) for i in indices):
                                tribe_failed_predictions[user_id].append(result)
                                break
                except Exception as e:
                    logging_utils.log_thread_safe(f"[TEST MODE] Error processing review: {e}", 'error')
        
        # Store results
        all_tribe_predictions[tribe_id] = tribe_user_predictions
        clusters_dict[cluster_id][micro_id] = tribe_user_predictions
        
        # Save failed predictions
        if tribe_failed_predictions:
            failure_dir = os.path.join(output_dir, "_failures", cluster_id)
            os.makedirs(failure_dir, exist_ok=True)
            failure_file = os.path.join(failure_dir, f"{micro_id}_test_failed_predictions.json")
            io_utils.save_json_file(tribe_failed_predictions, failure_file)
            logging_utils.log_thread_safe(f"[TEST MODE] Saved {sum(len(v) for v in tribe_failed_predictions.values())} failed predictions", 'warning')
    
    # Save predictions by cluster
    for cluster_id, micro_dict in clusters_dict.items():
        cluster_dir = os.path.join(output_dir, cluster_id)
        os.makedirs(cluster_dir, exist_ok=True)
        
        for micro_id, predictions in micro_dict.items():
            output_file = os.path.join(cluster_dir, f"{micro_id}_test_predictions.json")
            output_data = {
                'metadata': {
                    'cluster_id': cluster_id,
                    'micro_cluster_id': micro_id,
                    'dataset_type': 'test',
                    'test_mode': True
                },
                'user_predictions': predictions
            }
            io_utils.save_json_file(output_data, output_file)
            logging_utils.log_thread_safe(f"[TEST MODE] Saved predictions to {output_file}", 'info')
    
    logging_utils.log_thread_safe("[TEST MODE] Test mode predictions completed", 'success')
    return clusters_dict

