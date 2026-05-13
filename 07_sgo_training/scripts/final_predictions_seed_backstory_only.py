#!/usr/bin/env python3
"""
Final Predictions with Seed Backstory Only
==========================================

Generates final predictions for training data using ONLY seed backstory as context.
- Uses the same pipeline as pre_sgo_predictions
- Loads refined characteristics for each tribe (persona profile)
- EXCLUDES user backstory (user characteristics from user_backstories) from context
- EXCLUDES category characteristics (category chars from user_backstories) from context
- EXCLUDES user motives (memory/batch_analyses) from context
- Includes ONLY: seed backstory (original persona seed) and refined characteristics (persona profile)

Usage:
    python 07_sgo_training/scripts/final_predictions_seed_backstory_only.py
    python 07_sgo_training/scripts/final_predictions_seed_backstory_only.py --only-cluster cluster_0
    python 07_sgo_training/scripts/final_predictions_seed_backstory_only.py --only-cluster cluster_4 --only-micro micro_15
    python 07_sgo_training/scripts/final_predictions_seed_backstory_only.py --start-from-cluster cluster_1
"""

import sys
import logging
import os
import json
from pathlib import Path
import argparse
from threading import Lock
from openai import OpenAI

# Add project root FIRST (before any imports)
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Import from project root utils FIRST (before _package_setup which might modify paths)
from utils.wandb_utils import init_wandb_run, load_config, get_stage_config, get_openai_config, finish_run

# Set up package structure for sgo_training imports
scripts_dir = Path(__file__).parent
if str(scripts_dir) not in sys.path:
    sys.path.insert(0, str(scripts_dir))
import _package_setup

# Add prediction_lib path (from 07_pre_sgo_predictions/scripts)
prediction_lib_path = str(project_root / "07_pre_sgo_predictions" / "scripts")
if prediction_lib_path not in sys.path:
    sys.path.insert(0, prediction_lib_path)
from prediction_lib.data_loader import DataLoader
from prediction_lib.generate_model_predictions import PipelineOrchestrator
from prediction_lib.llm_client import RateLimiter
from prediction_lib.prompt_generation import load_prompt, create_enhanced_prompt
from prediction_lib.single_review_prediction import process_single_review

# Import sgo_training utilities for memory loading
from sgo_training.utils import io_utils

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def load_memory_file(memory_dir: str, cluster_id: str, micro_id: str) -> dict:
    """Load memory file (batch_analyses) for a specific cluster and micro cluster.
    For test clusters, tries to load memory from training micro clusters in the same cluster.
    """
    # First, try to load memory for the specific micro cluster
    memory_file = os.path.join(memory_dir, cluster_id, f"{micro_id}_batch_analyses.json")
    if os.path.exists(memory_file):
        return io_utils.load_json_file(memory_file, {})
    
    # If not found (e.g., test cluster), try to load memory from training micro clusters in the same cluster
    cluster_dir = os.path.join(memory_dir, cluster_id)
    if os.path.exists(cluster_dir):
        # Find all training micro cluster memory files (micro_0 through micro_20 or so)
        all_memory = {}
        training_micros_loaded = []
        total_batches_loaded = 0
        for file in sorted(os.listdir(cluster_dir)):
            if file.endswith('_batch_analyses.json'):
                # Extract micro_id from filename (e.g., "micro_0_batch_analyses.json" -> "micro_0")
                file_micro_id = file.replace('_batch_analyses.json', '')
                training_memory_file = os.path.join(cluster_dir, file)
                training_memory = io_utils.load_json_file(training_memory_file, {})
                if training_memory:
                    # Merge all training memory from this cluster
                    batch_count = len(training_memory)
                    total_batches_loaded += batch_count
                    all_memory.update(training_memory)
                    training_micros_loaded.append(file_micro_id)
        
        if all_memory:
            logging.info(f"  📚 Loaded memory from {len(training_micros_loaded)} training micro clusters in {cluster_id} (test cluster {micro_id})")
            logging.info(f"     Training micros: {training_micros_loaded}")
            logging.info(f"     Total batches loaded: {total_batches_loaded} (will be filtered by user_id when formatting)")
            return all_memory
    
    return {}


def load_user_learnings_summary(user_learnings_dir: str, cluster_id: str, micro_id: str, user_id: str) -> str:
    """Load user learnings summary from tribe_user_learnings directory.
    
    Args:
        user_learnings_dir: Base directory for user learnings (e.g., "07_sgo_training/artifacts/user_learnings")
        cluster_id: Cluster ID (e.g., "cluster_4")
        micro_id: Micro cluster ID (e.g., "micro_1")
        user_id: User ID (may have _1 suffix that needs to be stripped)
    
    Returns:
        Summary string from new_characteristics.summary, or empty string if not found
    """
    if not user_learnings_dir:
        logging.debug(f"  ⚠️  user_learnings_dir not provided for user {user_id}")
        return ""
    
    # Strip _1 suffix from user_id if present (reviews have _1 suffix but files don't)
    clean_user_id = user_id
    if user_id.endswith('_1'):
        clean_user_id = user_id[:-2]
        logging.debug(f"  🔧 Stripped _1 suffix from user_id: {user_id} -> {clean_user_id}")
    
    user_learnings_file = os.path.join(user_learnings_dir, "tribe_user_learnings", cluster_id, micro_id, f"{clean_user_id}_user_learnings.json")
    
    if not os.path.exists(user_learnings_file):
        logging.debug(f"  ⚠️  User learnings file not found: {user_learnings_file}")
        # Try with original user_id in case it doesn't have suffix
        if clean_user_id != user_id:
            alt_file = os.path.join(user_learnings_dir, "tribe_user_learnings", cluster_id, micro_id, f"{user_id}_user_learnings.json")
            if os.path.exists(alt_file):
                logging.debug(f"  ✅ Found file with original user_id: {alt_file}")
                user_learnings_file = alt_file
            else:
                return ""
        else:
            return ""
    
    try:
        data = io_utils.load_json_file(user_learnings_file, {})
        if not data:
            logging.warning(f"  ⚠️  Empty data loaded from {user_learnings_file}")
            return ""
        
        new_characteristics = data.get('new_characteristics', {})
        if not new_characteristics:
            logging.warning(f"  ⚠️  No new_characteristics found in {user_learnings_file}")
            return ""
        
        summary = new_characteristics.get('summary', '')
        if not summary:
            logging.warning(f"  ⚠️  No summary found in new_characteristics for user {user_id}")
            return ""
        
        logging.debug(f"  ✅ Loaded user learnings summary for {user_id} ({len(summary)} chars)")
        return summary
    except Exception as e:
        logging.error(f"  ❌ Failed to load user learnings for {user_id}: {e}")
        import traceback
        logging.error(f"  Traceback: {traceback.format_exc()}")
        return ""


def load_refined_characteristics(refined_chars_dir: str, cluster_id: str, micro_id: str) -> dict:
    """Load refined characteristics file for a specific cluster and micro cluster.
    Expected path: Prediction_Accuracy_After_SGO/v6_updated_refined_chars/{cluster_id}/{micro_id}_updated_refined_chars.json
    """
    # Primary path: Prediction_Accuracy_After_SGO/v6_updated_refined_chars/{cluster_id}/{micro_id}_updated_refined_chars.json
    primary_file = os.path.join(refined_chars_dir, cluster_id, f"{micro_id}_updated_refined_chars.json")
    
    if os.path.exists(primary_file):
        data = io_utils.load_json_file(primary_file, {})
        # Extract qualitative_summary from metadata
        metadata = data.get('metadata', {})
        qualitative_summary = metadata.get('qualitative_summary', {})
        if qualitative_summary:
            # Return the full structure with metadata for persona_name
            return {
                'persona_name': metadata.get('persona_name', ''),
                'qualitative_summary': qualitative_summary
            }
        return data
    
    # Fallback: try other possible file patterns
    possible_files = [
        os.path.join(refined_chars_dir, cluster_id, f"{micro_id}_refined_characteristics.json"),
        os.path.join(refined_chars_dir, cluster_id, f"refined_characteristics_{micro_id}.json"),
        os.path.join(refined_chars_dir, f"{cluster_id}_{micro_id}_refined_characteristics.json"),
    ]
    
    for file_path in possible_files:
        if os.path.exists(file_path):
            return io_utils.load_json_file(file_path, {})
    
    return {}


def format_memory_for_prompt(memory_data: dict, persona_name: str, user_id: str, debug=False) -> str:
    """
    Format memory data (batch_analyses) into a string for prompt inclusion.
    First tries to extract analyses for the specific user_id.
    If no user-specific analyses are found, returns entire memory_data as context.
    
    Args:
        memory_data: Dictionary with batch data (e.g., "iteration_1_batch_1": {"reviews": [...], "analyses": [...]})
        persona_name: Name of the persona (not used but kept for compatibility)
        user_id: User ID to filter analyses for (e.g., "AE32INSVRFOUVTTXAAY3E6VXFUQA")
        debug: If True, log detailed information about memory filtering
    
    Returns:
        Formatted string with user's analyses, or entire memory if no analyses found, or empty string if no memory
    """
    user_analyses = []
    total_batches = 0
    total_reviews_in_memory = 0
    user_reviews_found = 0
    
    # Extract analyses only for this specific user
    for batch_key, batch_data in memory_data.items():
        if not isinstance(batch_data, dict):
            continue
        
        total_batches += 1
        reviews = batch_data.get('reviews', [])
        analyses = batch_data.get('analyses', [])
        
        if not isinstance(reviews, list) or not isinstance(analyses, list):
            continue
        
        total_reviews_in_memory += len(reviews)
        
        # Find all review indices for this user
        # Review keys are in format: "{user_id}_review_{index}"
        for idx, review_key in enumerate(reviews):
            if isinstance(review_key, str) and review_key.startswith(f"{user_id}_review_"):
                # Found a review for this user - get corresponding analysis
                user_reviews_found += 1
                if idx < len(analyses):
                    analysis = analyses[idx]
                    if analysis:  # Only add non-empty analyses
                        user_analyses.append(str(analysis))
    
    if debug:
        logging.info(f"  📚 Memory Filtering Stats for user {user_id}:")
        logging.info(f"     - Total batches in memory: {total_batches}")
        logging.info(f"     - Total reviews in memory: {total_reviews_in_memory}")
        logging.info(f"     - Reviews found for this user: {user_reviews_found}")
        logging.info(f"     - Analyses extracted for this user: {len(user_analyses)}")
        if user_analyses:
            total_chars = sum(len(str(a)) for a in user_analyses)
            logging.info(f"     - Total memory text length: {total_chars} characters")
            logging.info(f"     - Average analysis length: {total_chars // len(user_analyses) if user_analyses else 0} characters")
    
    if user_analyses:
        # Format as past learnings (user-specific analyses found)
        past_learnings = ' | '.join(user_analyses)
        return f"""
**Past Learnings (Memory from SGO Training):**
{past_learnings}
"""
    elif memory_data:
        # No user-specific analyses found - return entire memory_data as context
        if debug:
            logging.info(f"  ⚠️  No user-specific analyses found for user {user_id}, returning entire memory ({len(memory_data)} batches)")
        
        # Format entire memory_data as JSON
        entire_memory = json.dumps(memory_data, indent=2, ensure_ascii=False)
        return f"""
**Past Learnings (Memory from SGO Training - Entire Memory):**
{entire_memory}
"""
    return ""


def format_refined_characteristics_for_prompt(refined_chars_data: dict) -> dict:
    """
    Format refined characteristics from the updated format to prompt format.
    
    REFINED MODE STRUCTURE (from v6_updated_refined_chars_back):
    - metadata.qualitative_summary.inherent_behavioral_traits -> core_characteristics
    - metadata.qualitative_summary.latent_motivations.main -> key_motivations
    - metadata.qualitative_summary.latent_motivations.validation_triggers -> common_praises
    - metadata.qualitative_summary.latent_motivations.friction_points -> common_criticisms
    - metadata.qualitative_summary.implicit_goals -> potential_goals
    
    SEED MODE STRUCTURE (from tribe_seed):
    - qualitative_summary.core_characteristics -> core_characteristics
    - qualitative_summary.key_motivations -> key_motivations
    - qualitative_summary.common_praises -> common_praises
    - qualitative_summary.common_criticisms -> common_criticisms
    - qualitative_summary.potential_goals -> potential_goals
    
    Returns a dict with keys: persona_name, persona_summary, key_motivations, 
    common_praises, common_criticisms, core_characteristics, potential_goals
    """
    if not refined_chars_data:
        return {}
    
    # Extract persona_name - can be at top level or in metadata
    persona_name = refined_chars_data.get('persona_name', '')
    if not persona_name:
        metadata = refined_chars_data.get('metadata', {})
        persona_name = metadata.get('persona_name', '')
    
    # Extract qualitative_summary - REFINED mode has it in metadata, SEED mode has it at top level
    qualitative_summary = refined_chars_data.get('qualitative_summary', {})
    if not qualitative_summary:
        # Try metadata.qualitative_summary (REFINED mode structure)
        metadata = refined_chars_data.get('metadata', {})
        qualitative_summary = metadata.get('qualitative_summary', {})
    if not qualitative_summary:
        # Fallback: try direct access
        qualitative_summary = refined_chars_data
    
    persona_summary = qualitative_summary.get('persona_summary', '')
    
    def extract_text_from_objects(items):
        """Extract text field from list of objects, or return strings as-is."""
        if not items or not isinstance(items, list):
            return []
        
        result = []
        for item in items:
            if isinstance(item, str):
                result.append(item)
            elif isinstance(item, dict):
                # Extract 'text' field from objects (REFINED mode has objects with 'text' field)
                text = item.get('text', '')
                if text:
                    result.append(text)
        return result
    
    # REFINED MODE: Map from refined structure to prompt format
    # Check if we have refined structure (inherent_behavioral_traits exists)
    if 'inherent_behavioral_traits' in qualitative_summary:
        # REFINED MODE STRUCTURE
        inherent_traits = qualitative_summary.get('inherent_behavioral_traits', [])
        core_characteristics = extract_text_from_objects(inherent_traits)
        
        latent_motivations = qualitative_summary.get('latent_motivations', {})
        key_motivations = extract_text_from_objects(latent_motivations.get('main', []))
        
        validation_triggers = latent_motivations.get('validation_triggers', [])
        common_praises = validation_triggers if isinstance(validation_triggers, list) else []
        
        friction_points = latent_motivations.get('friction_points', [])
        common_criticisms = friction_points if isinstance(friction_points, list) else []
        
        implicit_goals = qualitative_summary.get('implicit_goals', [])
        potential_goals = extract_text_from_objects(implicit_goals)
    else:
        # SEED MODE STRUCTURE (direct access to fields)
        core_characteristics = qualitative_summary.get('core_characteristics', [])
        key_motivations = qualitative_summary.get('key_motivations', [])
        common_praises = qualitative_summary.get('common_praises', [])
        common_criticisms = qualitative_summary.get('common_criticisms', [])
        potential_goals = qualitative_summary.get('potential_goals', [])
        
        # Ensure they're lists
        core_characteristics = core_characteristics if isinstance(core_characteristics, list) else []
        key_motivations = key_motivations if isinstance(key_motivations, list) else []
        common_praises = common_praises if isinstance(common_praises, list) else []
        common_criticisms = common_criticisms if isinstance(common_criticisms, list) else []
        potential_goals = potential_goals if isinstance(potential_goals, list) else []
    
    return {
        'persona_name': persona_name,
        'persona_summary': persona_summary,
        'key_motivations': key_motivations,
        'common_praises': common_praises,
        'common_criticisms': common_criticisms,
        'core_characteristics': core_characteristics,
        'potential_goals': potential_goals
    }


def create_final_prompt_with_memory_and_refined_chars(
    refined_chars_dict: dict,
    user_char_summary: str,
    category_char_summary: str,
    product_description: str,
    category: str,
    category_themes: list,
    prompt_template: str,
    user_motives: str = "",
    scoring_mode: str = "confidence",
    seed_backstory: dict = None
) -> str:
    """
    Create enhanced prompt with ONLY seed backstory as context.
    Includes ONLY:
    - Refined characteristics (persona profile)
    - Seed backstory (original persona seed)
    NOTE: User characteristics (user backstory), category characteristics, and user motives (memory) are EXCLUDED from context
    """
    # Use refined characteristics instead of baseline persona
    persona_name = refined_chars_dict.get('persona_name', 'Unknown Persona')
    persona_summary = refined_chars_dict.get('persona_summary', 'N/A')
    key_motivations = refined_chars_dict.get('key_motivations', [])
    common_praises = refined_chars_dict.get('common_praises', [])
    common_criticisms = refined_chars_dict.get('common_criticisms', [])
    core_characteristics = refined_chars_dict.get('core_characteristics', [])
    potential_goals = refined_chars_dict.get('potential_goals', [])
    
    def format_list(items):
        if not items or not isinstance(items, list):
            return "N/A"
        return "\n- ".join([""] + items)
    
    motivations_text = format_list(key_motivations)
    praises_text = format_list(common_praises)
    criticisms_text = format_list(common_criticisms)
    characteristics_text = format_list(core_characteristics)
    goals_text = format_list(potential_goals)
    
    # Format seed backstory section
    seed_backstory_section = ""
    if seed_backstory:
        seed_name = seed_backstory.get('persona_name', 'Unknown Persona')
        seed_summary = seed_backstory.get('persona_summary', 'N/A')
        seed_motivations = format_list(seed_backstory.get('key_motivations', []))
        seed_praises = format_list(seed_backstory.get('common_praises', []))
        seed_criticisms = format_list(seed_backstory.get('common_criticisms', []))
        seed_characteristics = format_list(seed_backstory.get('core_characteristics', []))
        seed_goals = format_list(seed_backstory.get('potential_goals', []))
        
        seed_backstory_section = f"""
### 0. Original Persona Seed (Baseline)
**Your Original Persona Profile (Before Refinement):**
**Persona Name:** {seed_name}
**Summary:** {seed_summary}
**Key Motivations:**{seed_motivations}
**Common Praises:**{seed_praises}
**Common Criticisms:**{seed_criticisms}
**Core Characteristics:**{seed_characteristics}
**Potential Goals:**{seed_goals}
"""
    
    # NOTE: User motives (memory) are EXCLUDED - not added to context
    # NOTE: Category characteristics section is EXCLUDED - not added to context
    category_section = ""  # Always empty - category characteristics excluded
    
    # Combine all context sections (ONLY seed backstory - excludes user backstory, category characteristics, and user motives)
    all_context_sections = ""
    if seed_backstory_section:
        all_context_sections += seed_backstory_section
    
    # Format the prompt template
    if scoring_mode == "logprobs":
        prompt = prompt_template.format(
            persona_name=persona_name,
            persona_summary=persona_summary,
            key_motivations=motivations_text,
            common_praises=praises_text,
            common_criticisms=criticisms_text,
            core_characteristics=characteristics_text,
            potential_goals=goals_text,
            user_char_summary=user_char_summary or "N/A",
            category_section=category_section or "N/A",  # EXCLUDED: category characteristics not used
            category=category,
            product_description=product_description,
            memory_section="No past learnings available.",  # EXCLUDED: user motives (memory) not used
            all_context_sections=all_context_sections
        )
    else:
        # Confidence mode: include theme scoring instructions
        themes_list = "\n".join([f"{i+1}. {theme}" for i, theme in enumerate(category_themes)])
        
        # Generate themes JSON template
        themes_json_template = "\n".join([
            f'    "{theme}": <float 0.0-1.0>,' if i < len(category_themes) - 1 
            else f'    "{theme}": <float 0.0-1.0>'
            for i, theme in enumerate(category_themes)
        ])
        
        prompt = prompt_template.format(
            persona_name=persona_name,
            persona_summary=persona_summary,
            key_motivations=motivations_text,
            common_praises=praises_text,
            common_criticisms=criticisms_text,
            core_characteristics=characteristics_text,
            potential_goals=goals_text,
            user_char_summary=user_char_summary or "N/A",
            category_section=category_section or "N/A",  # EXCLUDED: category characteristics not used
            category=category,
            product_description=product_description,
            memory_section="No past learnings available.",  # EXCLUDED: user motives (memory) not used
            themes_list=themes_list,
            themes_json_template=themes_json_template,
            all_context_sections=all_context_sections
        )
    
    return prompt


def process_single_review_with_memory_and_refined_chars(args_tuple, debug_print_context=False):
    """
    Process a single review prediction with memory and refined characteristics.
    Modified version of process_single_review that uses memory and refined chars in prompt.
    Also calculates deltas (text_delta, theme_delta/JSD, overall_delta).
    Thread-safe with lock for embeddings_cache access.
    """
    (actual_review, refined_chars_dict, user_char_summary, category_char_summary,
     main_category, category_themes, user_motives, seed_backstory,
     client, rate_limiter, prompt_template, model_name, max_tokens, temperature, max_retries, 
     scoring_mode, theme_prediction_model, review_key, embeddings_cache, embeddings_cache_lock) = args_tuple
    
    # NOTE: user_motives (memory) is EXCLUDED - always empty
    user_motives = ""  # EXCLUDED: user motives (memory) not used as context
    
    # Log context (only for first review to avoid spam)
    if review_key.endswith('_review_0'):
        logging.info(f"  ℹ️  Using ONLY seed backstory as context (user backstory, category chars, and user motives EXCLUDED)")
    
    # Create prompt with ONLY seed backstory as context (EXCLUDES user backstory, category characteristics, and user motives):
    # - refined_chars_dict (persona profile)
    # - seed_backstory (original persona seed)
    # NOTE: user_char_summary (user backstory), category_char_summary (category chars), and user_motives (memory) are EXCLUDED - passed as empty strings
    prompt = create_final_prompt_with_memory_and_refined_chars(
        refined_chars_dict=refined_chars_dict,
        user_char_summary=user_char_summary,
        category_char_summary=category_char_summary,
        product_description=actual_review.get('product_description', ''),
        category=main_category,
        category_themes=category_themes,
        prompt_template=prompt_template,
        user_motives=user_motives or "",
        scoring_mode=scoring_mode,
        seed_backstory=seed_backstory
    )
    
    # DEBUG: Log full prompt for first review (check review_key to identify)
    if review_key.endswith('_review_0'):
        # Extract user_id from review_key (format: "{user_id}_review_0")
        user_id_from_key = review_key.rsplit('_review_', 1)[0]
        logging.info(f"\n{'='*80}")
        logging.info(f"🔍 DEBUG: Full Prompt for First Review (User: {user_id_from_key})")
        logging.info(f"{'='*80}")
        logging.info(f"Prompt length: {len(prompt)} characters")
        logging.info(f"Prompt preview (first 1000 chars):\n{prompt[:1000]}...")
        logging.info(f"{'='*80}\n")
    
    # Use the same LLM prediction logic from pre_sgo_predictions
    from prediction_lib.llm_client import get_llm_prediction
    
    product_description = actual_review.get('product_description', '')
    prediction = get_llm_prediction(
        prompt, client, rate_limiter, category_themes, model_name, 
        max_tokens, temperature, max_retries, scoring_mode, 
        theme_prediction_model, product_description
    )
    
    # Check if LLM call failed
    if prediction is None:
        return {
            'status': 'failed',
            'error': 'LLM call failed after all retries',
            'product_description': actual_review.get('product_description', ''),
            'category': actual_review.get('category'),
            'asin': actual_review.get('asin'),
            'timestamp': actual_review.get('timestamp'),
            'actual': {
                'review_text': actual_review.get('review_text', ''),
                'rating': actual_review.get('rating'),
                'sentiment': actual_review.get('sentiment'),
                'topic_probabilities': actual_review.get('topic_probabilities', {})  # Use topic_probabilities as actual (ground truth)
            }
        }
    
    # DEBUG: Log raw theme logprobs and predicted themes for first review
    if review_key.endswith('_review_0'):
        user_id_from_key = review_key.rsplit('_review_', 1)[0]
        logging.info(f"\n{'='*80}")
        logging.info(f"🔍 DEBUG: Theme Probabilities for First Review (User: {user_id_from_key})")
        logging.info(f"{'='*80}")
        
        # Log raw theme logprobs if available
        theme_logprobs = prediction.get('theme_logprobs', {})
        if theme_logprobs:
            logging.info(f"Raw Theme Logprobs:")
            for theme, logprob_data in theme_logprobs.items():
                if isinstance(logprob_data, dict):
                    yes_logprob = logprob_data.get('yes', 'N/A')
                    no_logprob = logprob_data.get('no', 'N/A')
                    yes_token = logprob_data.get('token_yes', 'N/A')
                    no_token = logprob_data.get('token_no', 'N/A')
                    logging.info(f"  {theme}:")
                    logging.info(f"    yes_logprob: {yes_logprob}, token: {yes_token}")
                    logging.info(f"    no_logprob: {no_logprob}, token: {no_token}")
                else:
                    logging.info(f"  {theme}: {logprob_data}")
        else:
            logging.warning("  ⚠️  No theme_logprobs found in prediction")
        
        # Log predicted themes
        pred_themes = prediction.get('predicted_themes', {})
        if pred_themes:
            logging.info(f"\nPredicted Themes (normalized probabilities):")
            sorted_themes = sorted(pred_themes.items(), key=lambda x: x[1], reverse=True)
            for theme, prob in sorted_themes[:10]:  # Top 10
                logging.info(f"  {theme}: {prob:.6f}")
            
            # Check if all probabilities are the same
            unique_probs = set(pred_themes.values())
            if len(unique_probs) == 1:
                logging.warning(f"  ⚠️  WARNING: All themes have the same probability ({list(unique_probs)[0]:.6f})!")
                logging.warning(f"  This suggests logprobs extraction failed and defaults were used.")
        else:
            logging.warning("  ⚠️  No predicted_themes found in prediction")
        
        logging.info(f"{'='*80}\n")
    
    # Check if logprob extraction failed - check BOTH raw logprobs and normalized probabilities
    theme_logprobs = prediction.get('theme_logprobs', {})
    pred_themes = prediction.get('predicted_themes', {})
    
    # First check: Are all raw logprobs using default -10.0? (direct indicator of failure)
    logprob_extraction_failed = False
    if isinstance(theme_logprobs, dict) and theme_logprobs:
        all_defaults = True
        for theme, logprob_data in theme_logprobs.items():
            if isinstance(logprob_data, dict):
                yes_logprob = logprob_data.get('yes', None)
                no_logprob = logprob_data.get('no', None)
                # Check if both are exactly -10.0 (default value when extraction fails)
                if yes_logprob != -10.0 or no_logprob != -10.0:
                    all_defaults = False
                    break
            else:
                all_defaults = False
                break
        
        if all_defaults:
            logging.error(f"❌ REJECTING REVIEW: All theme logprobs are default -10.0. Logprob extraction failed.")
            logging.error(f"   ASIN: {actual_review.get('asin', 'unknown')}, User: {review_key.rsplit('_review_', 1)[0] if '_review_' in review_key else 'unknown'}")
            logprob_extraction_failed = True
    
    # Second check: Are all normalized probabilities equal/similar? (indirect indicator)
    if not logprob_extraction_failed and isinstance(pred_themes, dict) and pred_themes:
        probs = list(pred_themes.values())
        unique_probs = set(probs)
        
        # Check if all probabilities are the same (or very close - within 0.001)
        if len(unique_probs) == 1:
            # All themes have exactly the same probability - logprob extraction failed
            prob_value = list(unique_probs)[0]
            logging.error(f"❌ REJECTING REVIEW: All {len(probs)} themes have same probability ({prob_value:.6f}). Logprob extraction failed.")
            logging.error(f"   ASIN: {actual_review.get('asin', 'unknown')}, User: {review_key.rsplit('_review_', 1)[0] if '_review_' in review_key else 'unknown'}")
            logprob_extraction_failed = True
        elif len(unique_probs) == 2:
            # Check if probabilities are very close (within 0.001) - might be partial failure
            sorted_probs = sorted(unique_probs)
            if sorted_probs[1] - sorted_probs[0] < 0.001:
                # Very close probabilities - likely logprob extraction issue
                logging.error(f"❌ REJECTING REVIEW: All themes have very similar probabilities (diff < 0.001). Logprob extraction likely failed.")
                logging.error(f"   ASIN: {actual_review.get('asin', 'unknown')}, User: {review_key.rsplit('_review_', 1)[0] if '_review_' in review_key else 'unknown'}")
                logging.error(f"   Probabilities: {sorted_probs}")
                logprob_extraction_failed = True
    
    # If logprob extraction failed, return failed status
    if logprob_extraction_failed:
        return {
            'status': 'failed',
            'error': 'Logprob extraction failed: LLM did not provide proper logprobs (default -10.0 used or equal probabilities)',
            'error_type': 'logprob_extraction_failed',
            'product_description': actual_review.get('product_description', ''),
            'category': actual_review.get('category'),
            'asin': actual_review.get('asin'),
            'timestamp': actual_review.get('timestamp'),
            'actual': {
                'review_text': actual_review.get('review_text', ''),
                'rating': actual_review.get('rating'),
                'sentiment': actual_review.get('sentiment'),
                'topic_probabilities': actual_review.get('topic_probabilities', {})
            },
            'predicted_themes': pred_themes,  # Include for debugging
            'theme_logprobs': theme_logprobs  # Include raw logprobs for debugging
        }
    
    # Calculate metrics
    from prediction_lib.metrics import calculate_review_metrics
    
    metrics = calculate_review_metrics(prediction, {
        'rating': actual_review.get('rating'),
        'sentiment': actual_review.get('sentiment'),
        'predicted_themes': actual_review.get('themes', actual_review.get('predicted_themes', []))
    })
    
    # Calculate deltas (text_delta, theme_delta/JSD, overall_delta)
    deltas = {}
    try:
        # Import delta calculation functions
        # calculate_text_delta not used - text_delta is hardcoded to 0
        from calculate_behaviour_loss.weighted_topic_review_metric_calculation import (
            # calculate_text_delta,  # COMMENTED OUT: not calculating text_delta
            calculate_theme_delta,
            calculate_theme_delta_logprobs
        )
        # from sgo_training.utils import embeddings  # COMMENTED OUT: not calculating text_delta, so embeddings not needed
        
        # Get scoring mode from params (passed through task tuple)
        scoring_mode = scoring_mode  # From args_tuple
        
        # Calculate text_delta (cosine similarity between embeddings)
        # COMMENTED OUT: Not calculating text_delta, hardcoding to 0
        # pred_text = prediction.get('review_text', '')
        # act_text = actual_review.get('review_text', '')
        # 
        # if pred_text and act_text:
        #     try:
        #         # Use get_or_compute_embedding like SGO training pipeline (with lock for thread-safety)
        #         pred_emb = embeddings.get_or_compute_embedding(
        #             pred_text,
        #             review_key,
        #             'pred_final_refined',
        #             client,
        #             embeddings_cache,
        #             lock=embeddings_cache_lock  # Thread-safe cache access
        #         )
        #         act_emb = embeddings.get_or_compute_embedding(
        #             act_text,
        #             review_key,
        #             'actual',
        #             client,
        #             embeddings_cache,
        #             lock=embeddings_cache_lock  # Thread-safe cache access
        #         )
        #         if pred_emb is not None and act_emb is not None:
        #             text_delta = calculate_text_delta(pred_emb, act_emb)
        #             deltas['text_delta'] = text_delta
        #     except Exception as e:
        #         logging.warning(f"Failed to calculate text_delta: {e}")
        
        # Hardcode text_delta to 0 (not calculating it)
        deltas['text_delta'] = 0.0
        
        # Calculate theme_delta (JSD for logprobs mode, 1-recall for confidence mode)
        # Use topic_probabilities as actual themes (matching calculate_jsd_pre_sgo.py logic)
        pred_themes = prediction.get('predicted_themes', {})
        
        # For actual themes: MUST use topic_probabilities (ground truth distribution)
        # NO FALLBACKS - script should fail if topic_probabilities is missing
        act_themes = actual_review.get('topic_probabilities')
        if act_themes is None:
            error_msg = f"CRITICAL ERROR: topic_probabilities is missing for review (asin: {actual_review.get('asin', 'unknown')}, user: {review_key})"
            logging.error(error_msg)
            raise ValueError(error_msg)
        
        if not isinstance(act_themes, dict):
            error_msg = f"CRITICAL ERROR: topic_probabilities must be a dict, got {type(act_themes)} for review (asin: {actual_review.get('asin', 'unknown')}, user: {review_key})"
            logging.error(error_msg)
            raise ValueError(error_msg)
        
        if len(act_themes) == 0:
            error_msg = f"CRITICAL ERROR: topic_probabilities is empty dict for review (asin: {actual_review.get('asin', 'unknown')}, user: {review_key})"
            logging.error(error_msg)
            raise ValueError(error_msg)
        
        # DO NOT normalize or modify topic_probabilities - use them AS-IS
        # They are the actual ground truth probabilities and should not be changed
        # The JSD calculation function will handle normalization internally if needed
        
        # Handle list format for predicted themes (convert to uniform distribution only if list)
        # DO NOT normalize dict format - use predicted_themes AS-IS
        if isinstance(pred_themes, list):
            if not pred_themes:
                pred_themes = {}
            else:
                pred_themes = {theme: 1.0 / len(pred_themes) for theme in pred_themes}
        
        # DO NOT normalize pred_themes if it's already a dict - use AS-IS
        # The JSD calculation function will handle normalization internally if needed
        
        if pred_themes and act_themes:
            try:
                # Get all themes for JSD calculation (if logprobs mode)
                all_themes = None
                if scoring_mode in ["logprobs", "logprobs-without-persona-context"]:
                    # For JSD, we need all possible themes - use category_themes from task
                    all_themes = category_themes  # From args_tuple
                    theme_delta = calculate_theme_delta_logprobs(pred_themes, act_themes, all_themes)
                    deltas['theme_delta'] = theme_delta
                    deltas['theme_jsd'] = theme_delta  # Explicit JSD field
                else:
                    # Confidence mode: use 1-recall
                    theme_delta = calculate_theme_delta(pred_themes, act_themes)
                    deltas['theme_delta'] = theme_delta
            except Exception as e:
                logging.warning(f"Failed to calculate theme_delta: {e}")
        
        # Calculate overall_delta (weighted combination)
        # Note: text_delta is hardcoded to 0, so overall_delta only uses theme_delta
        if 'theme_delta' in deltas:
            # Use same weights as SGO training (from config or default)
            weight_text = 0.7  # Default, can be overridden from config
            weight_theme = 0.3  # Default, can be overridden from config
            # text_delta is 0, so overall_delta = 0 * weight_text + theme_delta * weight_theme
            overall_delta = (deltas.get('text_delta', 0.0) * weight_text) + (deltas['theme_delta'] * weight_theme)
            deltas['overall_delta'] = overall_delta
            deltas['delta_weights'] = {'text': weight_text, 'theme': weight_theme}
            
            # DEBUG: Log delta calculation details for first review
            if review_key.endswith('_review_0'):
                logging.info(f"\n{'='*80}")
                logging.info(f"🔍 DEBUG: Delta Calculation for First Review")
                logging.info(f"{'='*80}")
                # logging.info(f"Text Delta: {deltas.get('text_delta', 'N/A')} (cosine distance, lower is better)")  # COMMENTED OUT: text_delta not calculated
                logging.info(f"Theme Delta (JSD): {deltas.get('theme_delta', 'N/A')} (0=identical, 1=max divergence)")
                logging.info(f"Overall Delta: {overall_delta} (weighted: text={weight_text}, theme={weight_theme})")
                if isinstance(pred_themes, dict):
                    pred_top = sorted(pred_themes.items(), key=lambda x: x[1], reverse=True)[:5]
                    logging.info(f"Predicted Themes (top 5): {[(t, f'{p:.4f}') for t, p in pred_top]}")
                else:
                    logging.info(f"Predicted Themes: {pred_themes[:5] if isinstance(pred_themes, list) else pred_themes}")
                if isinstance(act_themes, dict):
                    act_top = sorted(act_themes.items(), key=lambda x: x[1], reverse=True)[:5]
                    logging.info(f"Actual Topics (top 5): {[(t, f'{p:.4f}') for t, p in act_top]}")
                else:
                    logging.info(f"Actual Topics: {act_themes[:5] if isinstance(act_themes, list) else act_themes}")
                logging.info(f"{'='*80}\n")
    except Exception as e:
        logging.warning(f"Failed to calculate deltas: {e}")
    
    result = {
        'status': 'success',
        'product_description': actual_review.get('product_description', ''),
        'category': actual_review.get('category'),
        'asin': actual_review.get('asin'),
        'timestamp': actual_review.get('timestamp'),
        'prediction': prediction,
        'actual': {
            'review_text': actual_review.get('review_text', ''),
            'rating': actual_review.get('rating'),
            'sentiment': actual_review.get('sentiment'),
            'topic_probabilities': actual_review.get('topic_probabilities', {})  # Use topic_probabilities as actual (ground truth)
        }
    }
    
    if metrics:
        result['metrics'] = metrics
    else:
        # Default metrics
        result['metrics'] = {
            'rating_score': 0.0,
            'sentiment_score': 0.0,
            'recall@1': 0.0,
            'recall@3': 0.0,
            'recall@5': 0.0,
            'recall@k': 0.0,
            'recall@max(3,k)': 0.0,
            'recall@max(3,k)_threshold_0.8': 0.0,
            'num_themes_above_0.8': 0,
            'num_additional_themes_0.8': 0,
            'recall@max(3,k)_threshold_0.85': 0.0,
            'num_themes_above_0.85': 0,
            'num_additional_themes_0.85': 0,
            'overall_accuracy': 0.0,
            'weights_used': {'rating': 0.4, 'sentiment': 0.3, 'theme_recall': 0.3}
        }
    
    # Add deltas to result
    if deltas:
        result['deltas'] = deltas
    
    return result


def format_seed_characteristics_for_prompt(tribe_seed) -> dict:
    """
    Format seed characteristics (from tribe_seed) for prompt.
    Extracts qualitative_summary from tribe_seed and formats it for prompt use.
    """
    if not tribe_seed:
        return {}
    
    persona_name = tribe_seed.persona_name if hasattr(tribe_seed, 'persona_name') else 'Unknown Persona'
    qualitative_summary = tribe_seed.qualitative_summary if hasattr(tribe_seed, 'qualitative_summary') else None
    
    if not qualitative_summary:
        return {
            'persona_name': persona_name,
            'persona_summary': 'N/A',
            'key_motivations': [],
            'common_praises': [],
            'common_criticisms': [],
            'core_characteristics': [],
            'potential_goals': []
        }
    
    # Extract fields from qualitative_summary
    persona_summary = qualitative_summary.persona_summary if hasattr(qualitative_summary, 'persona_summary') else 'N/A'
    key_motivations = qualitative_summary.key_motivations if hasattr(qualitative_summary, 'key_motivations') else []
    common_praises = qualitative_summary.common_praises if hasattr(qualitative_summary, 'common_praises') else []
    common_criticisms = qualitative_summary.common_criticisms if hasattr(qualitative_summary, 'common_criticisms') else []
    core_characteristics = qualitative_summary.core_characteristics if hasattr(qualitative_summary, 'core_characteristics') else []
    potential_goals = qualitative_summary.potential_goals if hasattr(qualitative_summary, 'potential_goals') else []
    
    return {
        'persona_name': persona_name,
        'persona_summary': persona_summary,
        'key_motivations': key_motivations if isinstance(key_motivations, list) else [],
        'common_praises': common_praises if isinstance(common_praises, list) else [],
        'common_criticisms': common_criticisms if isinstance(common_criticisms, list) else [],
        'core_characteristics': core_characteristics if isinstance(core_characteristics, list) else [],
        'potential_goals': potential_goals if isinstance(potential_goals, list) else []
    }


class FinalPredictionsRefinedCharsOrchestrator(PipelineOrchestrator):
    """
    Extended orchestrator that adds memory and refined characteristics loading and uses them in prompts.
    """
    def __init__(self, run, client, rate_limiter, prompt_template, config_params, 
                 category_mapping=None, memory_base_dir=None, refined_chars_base_dir=None, 
                 characteristics_mode="refined", user_learnings_base_dir=None):
        super().__init__(run, client, rate_limiter, prompt_template, config_params, category_mapping)
        self.memory_base_dir = memory_base_dir
        self.refined_chars_base_dir = refined_chars_base_dir
        self.characteristics_mode = characteristics_mode  # "refined" or "seed"
        self.user_learnings_base_dir = user_learnings_base_dir
    
    def _parse_tribe_id(self, tribe_id):
        """Parse tribe_id to get cluster and micro IDs."""
        import re
        if "_micro_" in tribe_id:
            parts = tribe_id.split("_micro_")
            raw_cluster = parts[0]
            micro_id = f"micro_{parts[1]}"
            
            # Normalize segment_0 -> cluster_0
            match = re.search(r'(\d+)', raw_cluster)
            cluster_id = f"cluster_{match.group(1)}" if match else raw_cluster
            return cluster_id, micro_id
        return tribe_id, "micro_0"
    
    def _execute_parallel(self, tasks, indices_map):
        """Runs single review predictions in parallel threads with memory and refined chars.
        Thread-safe: Uses locks for results collection and embeddings cache access.
        """
        from collections import defaultdict
        from concurrent.futures import ThreadPoolExecutor, as_completed
        from threading import Lock
        
        results = defaultdict(list)
        failed_results = defaultdict(list)
        results_lock = Lock()  # Lock for thread-safe results collection
        
        with ThreadPoolExecutor(max_workers=self.params['num_workers']) as executor:
            future_map = {}
            for uid, idxs in indices_map.items():
                for i in idxs:
                    future = executor.submit(process_single_review_with_memory_and_refined_chars, tasks[i])
                    future_map[future] = (uid, i)
            
            for future in as_completed(future_map):
                uid, task_idx = future_map[future]
                try:
                    res = future.result()
                    if res:
                        with results_lock:
                            if res.get('status') == 'failed':
                                failed_results[uid].append(res)
                            else:
                                results[uid].append(res)
                except Exception as e:
                    logging.error(f"Review processing exception for {uid}: {e}")
                    actual_review = tasks[task_idx][0]
                    with results_lock:
                        failed_results[uid].append({
                            'status': 'failed',
                            'error': f'Processing exception: {str(e)}',
                            'error_type': 'exception',
                            'product_description': actual_review.get('product_description', ''),
                            'category': actual_review.get('category'),
                            'asin': actual_review.get('asin'),
                            'timestamp': actual_review.get('timestamp'),
                            'actual': {
                                'review_text': actual_review.get('review_text', ''),
                                'rating': actual_review.get('rating'),
                                'sentiment': actual_review.get('sentiment'),
                                'topic_probabilities': actual_review.get('topic_probabilities', {})  # Use topic_probabilities as actual (ground truth)
                            }
                        })
        
        return results, failed_results
    
    def _save_failed_predictions(self, base_dir, cluster_id, micro_id, tribe_id, failed_predictions, dataset_type):
        """
        Save failed predictions to a separate file.
        Failed predictions are reviews where logprob extraction failed (all themes have equal probabilities).
        """
        import json
        from pathlib import Path
        
        # Create failures directory
        failures_dir = base_dir / cluster_id / "_failures"
        failures_dir.mkdir(parents=True, exist_ok=True)
        
        # Save failed predictions
        suffix = "_test" if dataset_type == "test" else ""
        failed_file = failures_dir / f"{micro_id}{suffix}_failed_predictions.json"
        
        # Count total failed reviews
        total_failed = sum(len(reviews) for reviews in failed_predictions.values())
        
        failed_data = {
            "metadata": {
                "cluster_id": cluster_id,
                "micro_cluster_id": micro_id,
                "tribe_id": tribe_id,
                "dataset_type": dataset_type,
                "total_failed_reviews": total_failed,
                "failure_reason": "logprob_extraction_failed",
                "note": "These reviews were rejected because logprob extraction failed, resulting in equal theme probabilities"
            },
            "failed_predictions": failed_predictions
        }
        
        with open(failed_file, 'w', encoding='utf-8') as f:
            json.dump(failed_data, f, indent=2, ensure_ascii=False)
        
        logging.info(f"  ❌ Saved {total_failed} failed predictions to: {failed_file}")
        
        # Log failure reasons breakdown
        failure_reasons = {}
        for user_id, reviews in failed_predictions.items():
            for review in reviews:
                error_type = review.get('error_type', 'unknown')
                failure_reasons[error_type] = failure_reasons.get(error_type, 0) + 1
        
        if failure_reasons:
            logging.info(f"  📊 Failure breakdown:")
            for reason, count in failure_reasons.items():
                logging.info(f"     {reason}: {count}")
    
    def _filter_tribes(self, tribe_data, target_clusters):
        """Filter tribes by cluster names. Accepts both 'cluster_X' and 'segment_X' formats.
        Also supports 'cluster_X micro_Y' format for specific micro cluster filtering."""
        if not target_clusters:
            return tribe_data
        
        import re
        
        # Parse target clusters - can be "cluster_X" or "cluster_X micro_Y"
        target_clusters_raw = set()
        target_clusters_segment = set()  # For matching tribe_ids
        target_micro_map = {}  # Map cluster to micro: {"segment_0": "micro_15"}
        
        for cluster_spec in target_clusters:
            parts = cluster_spec.strip().split()
            if len(parts) == 2:
                # Format: "cluster_X micro_Y"
                cluster_name = parts[0]
                micro_name = parts[1]
                target_clusters_raw.add(cluster_name)
                # Convert to segment format
                if cluster_name.startswith("cluster_"):
                    segment_name = cluster_name.replace("cluster_", "segment_", 1)
                    target_clusters_segment.add(segment_name)
                    target_micro_map[segment_name] = micro_name
                elif cluster_name.startswith("segment_"):
                    target_clusters_segment.add(cluster_name)
                    target_micro_map[cluster_name] = micro_name
            else:
                # Format: "cluster_X" (all micro clusters in this cluster)
                cluster_name = parts[0]
                target_clusters_raw.add(cluster_name)
                if cluster_name.startswith("cluster_"):
                    segment_name = cluster_name.replace("cluster_", "segment_", 1)
                    target_clusters_segment.add(segment_name)
                elif cluster_name.startswith("segment_"):
                    target_clusters_segment.add(segment_name)
                else:
                    # Try to extract number
                    match = re.search(r'(\d+)', cluster_name)
                    if match:
                        num = match.group(1)
                        target_clusters_segment.add(f"segment_{num}")
        
        # Filter tribes that belong to target clusters
        filtered_tribe_seed_data = {}
        for tribe_id, tribe_seed in tribe_data.items():
            # Extract cluster and micro from tribe_id (e.g., "segment_0_micro_15" -> "segment_0", "micro_15")
            if "_micro_" in tribe_id:
                cluster_id = tribe_id.split("_micro_")[0]
                micro_id = f"micro_{tribe_id.split('_micro_')[1]}"
            else:
                cluster_id = tribe_id
                micro_id = None
            
            # Check if cluster matches
            if cluster_id in target_clusters_segment:
                # If specific micro cluster is specified, check it matches
                if cluster_id in target_micro_map:
                    if micro_id == target_micro_map[cluster_id]:
                        filtered_tribe_seed_data[tribe_id] = tribe_seed
                else:
                    # No specific micro specified, include all micros in this cluster
                    filtered_tribe_seed_data[tribe_id] = tribe_seed
        
        if not filtered_tribe_seed_data:
            logging.error(f"No tribes found for specified clusters: {target_clusters}")
            available = set(tid.split('_micro_')[0] if '_micro_' in tid else tid for tid in tribe_data.keys())
            logging.error(f"Available clusters: {available}")
            return {}
        
        if target_micro_map:
            logging.info(f"Filtered to {len(filtered_tribe_seed_data)} tribe(s) matching: {target_clusters}")
        else:
            logging.info(f"Filtered to {len(filtered_tribe_seed_data)} tribes in clusters: {sorted(target_clusters_raw)}")
        return filtered_tribe_seed_data
    
    def _verify_saved_files(self, artifact_dir, clusters_dict):
        """Verifies all files are saved correctly. Override to handle dataset_type-specific filenames."""
        logging.info(f"\n{'='*60}")
        logging.info(f"Verifying all predictions are saved...")
        logging.info(f"{'='*60}")
        
        # Get dataset_type (stored in self.dataset_type from run_pipeline)
        dataset_type = getattr(self, 'dataset_type', 'train')
        infix = "_test" if dataset_type == "test" else ""
        
        total_clusters = len(clusters_dict)
        total_micro_clusters = sum(len(micro_clusters) for micro_clusters in clusters_dict.values())
        total_users_saved = 0
        total_reviews_saved = 0
        
        for cluster_id, micro_clusters in clusters_dict.items():
            cluster_dir = artifact_dir / cluster_id
            cluster_total_users = 0
            cluster_total_reviews = 0
            
            for micro_id, tribe_data in micro_clusters.items():
                # Check for summary file with correct infix based on dataset_type
                summary_file = cluster_dir / f"{micro_id}{infix}_summary_enhanced_persona_micro_cluster_accuracy.json"
                if summary_file.exists():
                    try:
                        from prediction_lib import utils
                        summary_data = utils.load_json_file(summary_file)
                        if summary_data:
                            num_users = summary_data.get("metadata", {}).get("total_users_in_cluster", 0)
                            num_reviews = summary_data.get("metadata", {}).get("total_reviews_from_cluster", 0)
                            cluster_total_users += num_users
                            cluster_total_reviews += num_reviews
                    except Exception as e:
                        logging.warning(f"  ⚠️  Could not read {cluster_id}/{summary_file.name}: {e}")
                else:
                    logging.warning(f"  ⚠️  File not found: {cluster_id}/{micro_id}{infix}_summary_enhanced_persona_micro_cluster_accuracy.json")
            
            total_users_saved += cluster_total_users
            total_reviews_saved += cluster_total_reviews
            
            # Check for grand summary file with correct infix
            grand_summary_file = cluster_dir / f"grand_summary{infix}_enhanced_persona_micro_cluster.json"
            if grand_summary_file.exists():
                logging.info(f"  ✅ {cluster_id}: {len(micro_clusters)} micro-clusters, {cluster_total_users} users, {cluster_total_reviews} reviews")
            else:
                logging.warning(f"  ⚠️  {cluster_id}/grand_summary{infix}_enhanced_persona_micro_cluster.json not found")
        
        logging.info(f"\n✅ Verification complete:")
        logging.info(f"   - {total_clusters} clusters")
        logging.info(f"   - {total_micro_clusters} micro-clusters")
        logging.info(f"   - {total_users_saved} users")
        logging.info(f"   - {total_reviews_saved} reviews")
    
    def run_pipeline(self, dataset_type, tribe_seed_data, user_backstories, review_data, 
                     topic_universe, user_to_tribe_map, output_artifact_name, target_clusters=None):
        """Main execution method with memory and refined characteristics loading."""
        logging.info(f"Starting final predictions pipeline for {dataset_type.upper()} set with memory and refined characteristics")
        # Store dataset_type for use in processing
        self.dataset_type = dataset_type
        
        # 1. Setup Theme Map
        category_themes_map = self._build_theme_map(topic_universe)
        logging.info(f"Built theme map with {len(category_themes_map)} categories")
        
        # 2. Filter Tribes if target_clusters provided
        if target_clusters:
            tribe_seed_data = self._filter_tribes(tribe_seed_data, target_clusters)
        
        # 3. Prepare Output Directory
        output_stage = self.params.get("output_stage", "07_sgo_training")
        from utils.wandb_utils import get_artifact_dir
        artifact_dir = get_artifact_dir(output_stage, output_artifact_name)
        artifact_dir.mkdir(parents=True, exist_ok=True)
        
        all_tribe_predictions = {}
        from collections import defaultdict
        clusters_dict = defaultdict(lambda: defaultdict(dict))
        
        # 4. Main Loop
        for tribe_id, tribe_seed in tribe_seed_data.items():
            logging.info(f"\n{'='*60}")
            logging.info(f"Processing Tribe: {tribe_id}")
            logging.info(f"{'='*60}")
            
            # Parse tribe_id to get cluster and micro IDs
            cluster_id, micro_id = self._parse_tribe_id(tribe_id)
            
            # Load memory data for this tribe (will be filtered per user later)
            memory_data = {}
            if self.memory_base_dir:
                memory_data = load_memory_file(self.memory_base_dir, cluster_id, micro_id)
                if memory_data:
                    # Check if we loaded from the specific micro or from training micros
                    specific_memory_file = os.path.join(self.memory_base_dir, cluster_id, f"{micro_id}_batch_analyses.json")
                    if os.path.exists(specific_memory_file):
                        logging.info(f"  ✅ Loaded memory data for {cluster_id}/{micro_id}")
                    else:
                        # We loaded from training micro clusters in the same cluster
                        logging.info(f"  ✅ Loaded memory from training micro clusters in {cluster_id} (for test cluster {micro_id})")
                else:
                    logging.warning(f"  ⚠️  No memory data found for {cluster_id}/{micro_id} (checked specific micro and training micros in cluster)")
            
            # Load refined characteristics (persona profile) - EXCLUDES user backstory from context
            # Mode determines whether to use refined characteristics or seed characteristics
            if self.characteristics_mode == "refined" and self.refined_chars_base_dir:
                # Try to load refined characteristics from files
                refined_chars_data = load_refined_characteristics(self.refined_chars_base_dir, cluster_id, micro_id)
                if refined_chars_data:
                    refined_chars_dict = format_refined_characteristics_for_prompt(refined_chars_data)
                    if refined_chars_dict and refined_chars_dict.get('persona_name'):
                        logging.info(f"  ✅ Using REFINED characteristics for {cluster_id}/{micro_id}")
                    else:
                        logging.warning(f"  ⚠️  Failed to format refined characteristics, falling back to seed")
                        refined_chars_dict = format_seed_characteristics_for_prompt(tribe_seed)
                else:
                    logging.warning(f"  ⚠️  Refined characteristics file not found for {cluster_id}/{micro_id}, falling back to seed")
                    refined_chars_dict = format_seed_characteristics_for_prompt(tribe_seed)
            else:
                # Use seed characteristics (default or when mode is "seed")
                refined_chars_dict = format_seed_characteristics_for_prompt(tribe_seed)
                if refined_chars_dict and refined_chars_dict.get('persona_name'):
                    logging.info(f"  ✅ Using SEED characteristics for {cluster_id}/{micro_id} (mode: {self.characteristics_mode})")
                else:
                    logging.warning(f"  ⚠️  Failed to format seed characteristics for {cluster_id}/{micro_id}")
            
            # Load seed backstory (original persona seed) - once per tribe
            seed_backstory_dict = format_seed_characteristics_for_prompt(tribe_seed)
            
            # Log what we're using
            if refined_chars_dict:
                persona_name = refined_chars_dict.get('persona_name', 'Unknown')
                key_motivations_count = len(refined_chars_dict.get('key_motivations', []))
                core_chars_count = len(refined_chars_dict.get('core_characteristics', []))
                logging.info(f"  📋 Using persona: {persona_name} ({key_motivations_count} motivations, {core_chars_count} characteristics)")
            if seed_backstory_dict:
                seed_name = seed_backstory_dict.get('persona_name', 'Unknown')
                logging.info(f"  📋 Seed backstory loaded: {seed_name}")
            
            # Identify Users
            tribe_users = [uid for uid, tid in user_to_tribe_map.items() if tid == tribe_id]
            if not tribe_users:
                logging.warning(f"  ⚠️  No users found for tribe {tribe_id}, skipping")
                continue
            logging.info(f"  Found {len(tribe_users)} users in tribe")
            
            # Prepare Data for Workers
            all_review_tasks = []
            user_review_indices = defaultdict(list)
            task_keys_seen = set()
            
            # Initialize embeddings cache for this tribe (shared across threads)
            embeddings_cache = {}
            embeddings_cache_lock = Lock()  # Lock for thread-safe cache access
            
            # Build Tasks (same as parent class but with memory_text and refined_chars_text)
            users_with_reviews = 0
            reviews_skipped_no_data = 0
            reviews_skipped_no_themes = 0
            reviews_skipped_empty_themes = 0
            reviews_skipped_duplicate = 0
            is_first_review_in_cluster = True  # Track first review for debug logging
            
            for user_id in tribe_users:
                if user_id not in review_data:
                    continue
                
                # Load context sources (ONLY seed backstory - EXCLUDES user backstory, category characteristics, and user motives):
                # NOTE: User characteristics (user backstory), category characteristics, and user motives (memory) are EXCLUDED - set to empty strings
                user_story = user_backstories.get(user_id)
                user_chars = ""  # EXCLUDED: user backstory not used as context
                user_motives = ""  # EXCLUDED: user motives (memory) not used as context
                
                # Seed backstory (loaded once per tribe above)
                # NOTE: Category characteristics are EXCLUDED - not loaded
                
                # Log context sources (ONLY seed backstory included)
                logging.info(f"  📝 User {user_id} context sources (ONLY seed backstory - excludes user backstory, category chars, and user motives):")
                logging.info(f"     - User backstory: ❌ (EXCLUDED from context)")
                logging.info(f"     - Category characteristics: ❌ (EXCLUDED from context)")
                logging.info(f"     - User motives (memory): ❌ (EXCLUDED from context)")
                
                reviews = review_data[user_id].get('reviews', [])
                if not reviews:
                    continue
                
                users_with_reviews += 1
                
                for review_idx, review in enumerate(reviews):
                    cat = review.get('category')
                    if not cat:
                        reviews_skipped_no_data += 1
                        continue
                    
                    # CRITICAL: topic_probabilities is REQUIRED - no fallbacks
                    # Script should fail if topic_probabilities is missing
                    topic_probs = review.get('topic_probabilities')
                    if topic_probs is None:
                        error_msg = f"CRITICAL ERROR: topic_probabilities is missing for review (asin: {review.get('asin', 'unknown')}, user: {user_id}, category: {cat}, dataset_type: {self.dataset_type}). Review keys: {list(review.keys())}"
                        logging.error(error_msg)
                        raise ValueError(error_msg)
                    
                    if not isinstance(topic_probs, dict):
                        error_msg = f"CRITICAL ERROR: topic_probabilities must be a dict, got {type(topic_probs)} for review (asin: {review.get('asin', 'unknown')}, user: {user_id}, category: {cat})"
                        logging.error(error_msg)
                        raise ValueError(error_msg)
                    
                    if len(topic_probs) == 0:
                        error_msg = f"CRITICAL ERROR: topic_probabilities is empty dict for review (asin: {review.get('asin', 'unknown')}, user: {user_id}, category: {cat})"
                        logging.error(error_msg)
                        raise ValueError(error_msg)
                    
                    asin = review.get('asin', '')
                    timestamp = review.get('timestamp', '')
                    review_text = review.get('review_text', '')
                    task_key = f"{user_id}_{asin}_{timestamp}_{hash(review_text[:50])}"
                    if task_key in task_keys_seen:
                        reviews_skipped_duplicate += 1
                        continue
                    task_keys_seen.add(task_key)
                    
                    # Create review_key for embedding cache (same format as SGO training)
                    review_key = f"{user_id}_review_{review_idx}"
                    
                    from prediction_lib import config as lib_config
                    main_cat = lib_config.map_category_to_main_category(cat, self.category_mapping) or cat
                    normalized_main = self._normalize_category(main_cat)
                    
                    themes = category_themes_map.get(normalized_main, [])
                    if not themes:
                        reviews_skipped_no_themes += 1
                        continue
                    
                    # EXCLUDED: Category characteristics not used as context
                    cat_summary = ""  # Always empty - category characteristics excluded
                    
                    # DEBUG: Print full context for first review in cluster
                    if is_first_review_in_cluster and review_idx == 0:
                        logging.info(f"\n{'='*80}")
                        logging.info(f"🔍 DEBUG: First Review Context for {cluster_id}/{micro_id}")
                        logging.info(f"{'='*80}")
                        logging.info(f"User ID: {user_id}")
                        logging.info(f"Review ASIN: {review.get('asin', 'unknown')}")
                        logging.info(f"Category: {cat} -> {main_cat}")
                        logging.info(f"\n--- PERSONA CHARACTERISTICS (mode: {self.characteristics_mode}) ---")
                        logging.info(f"Persona Name: {refined_chars_dict.get('persona_name', 'N/A')}")
                        logging.info(f"Persona Summary: {refined_chars_dict.get('persona_summary', 'N/A')[:200]}...")
                        logging.info(f"Key Motivations ({len(refined_chars_dict.get('key_motivations', []))}): {refined_chars_dict.get('key_motivations', [])[:3]}")
                        logging.info(f"Core Characteristics ({len(refined_chars_dict.get('core_characteristics', []))}): {refined_chars_dict.get('core_characteristics', [])[:3]}")
                        logging.info(f"\n--- SEED BACKSTORY (Original Persona) ---")
                        if seed_backstory_dict:
                            logging.info(f"Seed Persona: {seed_backstory_dict.get('persona_name', 'N/A')}")
                            logging.info(f"Seed Summary: {seed_backstory_dict.get('persona_summary', 'N/A')[:200]}...")
                        else:
                            logging.info("Seed backstory: Not available")
                        logging.info(f"\n--- USER CHARACTERISTICS (EXCLUDED - user backstory not used) ---")
                        logging.info(f"User Chars: EXCLUDED from context")
                        logging.info(f"\n--- USER MOTIVES (EXCLUDED - memory not used) ---")
                        logging.info(f"User Motives: EXCLUDED from context")
                        logging.info(f"\n--- CATEGORY CHARACTERISTICS (EXCLUDED - category chars not used) ---")
                        logging.info(f"Category Summary: EXCLUDED from context")
                        logging.info(f"\n--- PRODUCT INFO ---")
                        logging.info(f"Product Description: {review.get('product_description', 'N/A')[:200]}...")
                        logging.info(f"Category Themes ({len(themes)}): {themes[:5]}")
                        logging.info(f"\n--- ACTUAL REVIEW (Ground Truth) ---")
                        logging.info(f"Review Text: {review.get('review_text', 'N/A')[:200]}...")
                        logging.info(f"Rating: {review.get('rating', 'N/A')}")
                        logging.info(f"Sentiment: {review.get('sentiment', 'N/A')}")
                        topic_probs = review.get('topic_probabilities', {})
                        if topic_probs:
                            top_topics = sorted(topic_probs.items(), key=lambda x: x[1], reverse=True)[:5]
                            logging.info(f"Top 5 Topic Probabilities: {[(t, f'{p:.4f}') for t, p in top_topics]}")
                        logging.info(f"{'='*80}\n")
                        is_first_review_in_cluster = False  # Mark that we've logged the first review
                    
                    # Add refined_chars_dict (replaces persona_context), review_key, embeddings_cache, and lock to task tuple
                    # Pass context sources (ONLY seed backstory - excludes user backstory, category characteristics, and user motives): user_chars (empty), cat_summary (empty), user_motives (empty), seed_backstory
                    task = (review, refined_chars_dict, user_chars, cat_summary, main_cat, themes,
                            user_motives, seed_backstory_dict,
                            self.client, self.rate_limiter, self.prompt_template,
                            self.params.get('model'),
                            self.params.get('max_tokens'),
                            self.params.get('temperature'),
                            self.params.get('max_retries', 3),
                            self.params.get('scoring_mode', 'confidence'),
                            self.params.get('theme_prediction_model'),
                            review_key, embeddings_cache, embeddings_cache_lock)
                    
                    user_review_indices[user_id].append(len(all_review_tasks))
                    all_review_tasks.append(task)
            
            if not all_review_tasks:
                logging.warning(f"  ⚠️  No review tasks created for tribe {tribe_id}")
                continue
            
            logging.info(f"  Created {len(all_review_tasks)} review tasks from {users_with_reviews} users")
            
            # Run Parallel Execution
            tribe_user_predictions, tribe_failed_predictions = self._execute_parallel(all_review_tasks, user_review_indices)
            
            # Store results
            all_tribe_predictions[tribe_id] = tribe_user_predictions
            clusters_dict[cluster_id][micro_id] = tribe_user_predictions
            
            # Collect and save deltas for all reviews
            all_reviews_deltas = []
            for user_id, user_results in tribe_user_predictions.items():
                for review_idx, result in enumerate(user_results):
                    if result.get('status') == 'success' and 'deltas' in result:
                        deltas = result.get('deltas', {})
                        # Extract review information
                        review_key = f"{user_id}_review_{review_idx}"
                        
                        # Build comprehensive delta info (matching SGO training format)
                        # Note: text_delta is hardcoded to 0, not calculated
                        delta_info = {
                            'review_key': review_key,
                            'user_id': user_id,
                            'review_idx': review_idx,
                            'text_delta': 0.0,  # Hardcoded to 0, not calculated
                            'theme_delta': deltas.get('theme_delta'),
                            'theme_jsd': deltas.get('theme_jsd'),  # Explicit JSD field for logprobs mode
                            'overall_delta': deltas.get('overall_delta'),
                            'delta_weights': deltas.get('delta_weights', {'text': 0.7, 'theme': 0.3}),
                            'prediction': {
                                'review_text': result.get('prediction', {}).get('review_text', ''),
                                'rating': result.get('prediction', {}).get('rating'),
                                'sentiment': result.get('prediction', {}).get('sentiment'),
                                'predicted_themes': result.get('prediction', {}).get('predicted_themes', {})
                            },
                            'actual': {
                                'review_text': result.get('actual', {}).get('review_text', ''),
                                'rating': result.get('actual', {}).get('rating'),
                                'sentiment': result.get('actual', {}).get('sentiment'),
                                'topic_probabilities': result.get('actual', {}).get('topic_probabilities', {})  # Use topic_probabilities as actual (ground truth)
                            },
                            'category': result.get('category'),
                            'product_description': result.get('product_description', '')
                        }
                        all_reviews_deltas.append(delta_info)
            
            # Save deltas to file (matching SGO training format)
            if all_reviews_deltas:
                import time
                delta_dir = artifact_dir / "deltas" / cluster_id
                delta_dir.mkdir(parents=True, exist_ok=True)
                delta_file = delta_dir / f"{micro_id}_all_reviews_deltas.json"
                
                # Calculate statistics before saving
                import numpy as np
                theme_deltas = [d.get('theme_delta') for d in all_reviews_deltas if d.get('theme_delta') is not None]
                theme_jsds = [d.get('theme_jsd') for d in all_reviews_deltas if d.get('theme_jsd') is not None]
                # text_deltas not calculated - hardcoded to 0
                # text_deltas = [d.get('text_delta') for d in all_reviews_deltas if d.get('text_delta') is not None]
                overall_deltas = [d.get('overall_delta') for d in all_reviews_deltas if d.get('overall_delta') is not None]
                
                # Calculate averages - prefer theme_jsd over theme_delta for JSD
                avg_theme_jsd = float(np.mean(theme_jsds)) if theme_jsds else (float(np.mean(theme_deltas)) if theme_deltas else None)
                avg_theme_delta = float(np.mean(theme_deltas)) if theme_deltas else None
                avg_text_delta = 0.0  # Hardcoded to 0, not calculated
                avg_overall_delta = float(np.mean(overall_deltas)) if overall_deltas else None
                
                # Calculate min/max for reference
                min_theme_jsd = float(np.min(theme_jsds)) if theme_jsds else (float(np.min(theme_deltas)) if theme_deltas else None)
                max_theme_jsd = float(np.max(theme_jsds)) if theme_jsds else (float(np.max(theme_deltas)) if theme_deltas else None)
                min_text_delta = 0.0  # Hardcoded to 0, not calculated
                max_text_delta = 0.0  # Hardcoded to 0, not calculated
                median_theme_jsd = float(np.median(theme_jsds)) if theme_jsds else (float(np.median(theme_deltas)) if theme_deltas else None)
                median_text_delta = 0.0  # Hardcoded to 0, not calculated
                median_overall_delta = float(np.median(overall_deltas)) if overall_deltas else None
                
                all_deltas_data = {
                    'cluster_id': cluster_id,
                    'micro_cluster_id': micro_id,
                    'tribe_id': tribe_id,
                    'total_reviews': len(all_reviews_deltas),
                    'calculation_timestamp': time.time(),
                    'deltas': all_reviews_deltas,
                    'statistics': {
                        'avg_theme_jsd': avg_theme_jsd,
                        'avg_theme_delta': avg_theme_delta,
                        'avg_text_delta': avg_text_delta,
                        'avg_overall_delta': avg_overall_delta,
                        'min_theme_jsd': min_theme_jsd,
                        'max_theme_jsd': max_theme_jsd,
                        'min_text_delta': min_text_delta,
                        'max_text_delta': max_text_delta,
                        'median_theme_jsd': median_theme_jsd,
                        'median_text_delta': median_text_delta,
                        'median_overall_delta': median_overall_delta,
                        'num_reviews_with_theme_delta': len(theme_deltas),
                        'num_reviews_with_text_delta': len(all_reviews_deltas),  # All reviews have text_delta=0
                        'num_reviews_with_overall_delta': len(overall_deltas)
                    }
                }
                
                import json
                with open(delta_file, 'w', encoding='utf-8') as f:
                    json.dump(all_deltas_data, f, indent=2, ensure_ascii=False)
                
                logging.info(f"  💾 Saved deltas for {len(all_reviews_deltas)} reviews to: {delta_file}")
                
                # Print delta statistics
                if all_reviews_deltas:
                    if theme_deltas:
                        logging.info(f"  📊 Delta Statistics:")
                        logging.info(f"     Theme Delta (JSD):   min={min(theme_deltas):.4f}, max={max(theme_deltas):.4f}, mean={avg_theme_jsd:.4f}, median={median_theme_jsd:.4f}")
                    # Text Delta not calculated - hardcoded to 0
                    # if text_deltas:
                    #     logging.info(f"     Text Delta:          min={min(text_deltas):.4f}, max={max(text_deltas):.4f}, mean={avg_text_delta:.4f}, median={median_text_delta:.4f}")
                    if overall_deltas:
                        logging.info(f"     Overall Delta:       min={min(overall_deltas):.4f}, max={max(overall_deltas):.4f}, mean={avg_overall_delta:.4f}, median={median_overall_delta:.4f}")
                    logging.info(f"     📊 Average JSD for {cluster_id}/{micro_id}: {avg_theme_jsd:.4f}")
            
            # Save failed predictions
            if tribe_failed_predictions:
                self._save_failed_predictions(artifact_dir, cluster_id, micro_id, tribe_id, tribe_failed_predictions, dataset_type)
            
            # Save Micro-Cluster Files (use refined_chars_dict for persona context)
            persona_context_for_save = {
                'persona_name': refined_chars_dict.get('persona_name', ''),
                'qualitative_summary': {
                    'persona_summary': refined_chars_dict.get('persona_summary', ''),
                    'key_motivations': refined_chars_dict.get('key_motivations', []),
                    'common_praises': refined_chars_dict.get('common_praises', []),
                    'common_criticisms': refined_chars_dict.get('common_criticisms', []),
                    'core_characteristics': refined_chars_dict.get('core_characteristics', []),
                    'potential_goals': refined_chars_dict.get('potential_goals', [])
                }
            }
            self._save_micro_cluster_data(
                artifact_dir, cluster_id, micro_id, tribe_id,
                tribe_user_predictions, persona_context_for_save, dataset_type
            )
            
            # Update Cluster Grand Summary
            self._update_grand_summary(artifact_dir, cluster_id, micro_id, dataset_type)
        
        # Verify all files are saved
        self._verify_saved_files(artifact_dir, clusters_dict)
        
        # Final Logging & Artifacts
        self._log_final_artifacts(self.run, artifact_dir, output_artifact_name, all_tribe_predictions, dataset_type)


def main():
    parser = argparse.ArgumentParser(description='Generate final predictions with memory and refined characteristics from SGO training')
    parser.add_argument('--only-cluster', type=str, help='Process only this specific cluster (e.g., cluster_0)')
    parser.add_argument('--start-from-cluster', type=str, help='Start processing from this cluster onwards')
    parser.add_argument('--only-micro', type=str, help='Process only this specific micro cluster (e.g., micro_15). Must be used with --only-cluster')
    args = parser.parse_args()
    
    # Validate micro cluster argument
    if args.only_micro and not args.only_cluster:
        logging.error("❌ --only-micro requires --only-cluster to be specified")
        return
    
    # Config - use pre_sgo_predictions config structure
    config = load_config()
    stage_config = get_stage_config("07_pre_sgo_predictions")
    openai_config = get_openai_config()
    
    logging.info(f"Using config from stage: 07_pre_sgo_predictions")
    
    if not stage_config:
        logging.error("❌ Stage config not found")
        return
    
    # Get memory and refined chars base directories from sgo_training config
    sgo_config = get_stage_config("07_sgo_training")
    memory_base_dir = None
    refined_chars_base_dir = None
    user_learnings_base_dir = None
    
    if sgo_config:
        # Try to get from artifacts path
        artifacts_dir = Path(__file__).parent.parent / "artifacts"
        memory_dir = artifacts_dir / "sgo_training_results_v6" / "_memory"
        if memory_dir.exists():
            memory_base_dir = str(memory_dir)
            logging.info(f"Using memory directory: {memory_base_dir}")
        
        # Hardcoded refined characteristics directory: v6_updated_refined_chars
        project_root = Path(__file__).parent.parent.parent
        refined_chars_dir = project_root / "v6_updated_refined_chars"
        if refined_chars_dir.exists():
            refined_chars_base_dir = str(refined_chars_dir)
            logging.info(f"Using refined characteristics directory (hardcoded): {refined_chars_base_dir}")
        else:
            error_msg = f"CRITICAL ERROR: Refined characteristics directory not found at hardcoded path: {refined_chars_dir}"
            logging.error(error_msg)
            raise ValueError(error_msg)
        
        # User learnings directory: 07_sgo_training/artifacts/user_learnings
        user_learnings_dir = artifacts_dir / "user_learnings"
        if user_learnings_dir.exists():
            user_learnings_base_dir = str(user_learnings_dir)
            logging.info(f"✅ Using user learnings directory: {user_learnings_base_dir}")
        else:
            logging.warning(f"⚠️  User learnings directory not found: {user_learnings_dir}. Will fallback to user_backstories.")
    
    # Use same config loading as pre_sgo_predictions
    input_artifacts = stage_config.get("input_artifacts", {})
    output_artifacts = stage_config.get("output_artifacts", {})
    hyperparams = stage_config.get("hyperparameters", {})
    file_patterns = stage_config.get("file_patterns", {})
    dataset_pattern_mapping = stage_config.get("dataset_pattern_mapping", {})
    prompt_config = stage_config.get("prompt", {})
    schema_config = stage_config.get("schema", {})
    
    # Get all required configs (same as pre_sgo_predictions)
    tribe_seed_artifact = input_artifacts.get("tribe_seed_characteristics")
    user_backstory_artifact = input_artifacts.get("user_backstories")
    training_data_artifact = input_artifacts.get("training_data_with_topics")
    topic_universe_artifact = input_artifacts.get("topic_universe")
    user_tribes_artifact = input_artifacts.get("user_tribes")
    category_mapping_artifact = input_artifacts.get("category_mapping")
    
    # Get scoring mode and model config
    scoring_mode = stage_config.get("scoring_mode", "confidence")
    model_name = hyperparams.get("model")
    theme_prediction_model = None
    if scoring_mode in ["logprobs", "logprobs-without-persona-context"]:
        review_prediction_model = stage_config.get("review_prediction_model")
        theme_prediction_model = stage_config.get("theme_prediction_model")
        if review_prediction_model:
            model_name = review_prediction_model
    
    # Select prompt filename based on scoring mode
    use_enhanced = stage_config.get("use_enhanced_prompt", False)
    if scoring_mode in ["logprobs", "logprobs-without-persona-context"]:
        prompt_filename = "final_prediction_prompt_logprobs.txt"
    else:
        prompt_filename = "final_prediction_prompt.txt"
    
    # Load prompt from sgo_training prompts directory
    stage_dir = Path(__file__).parent.parent
    prompt_dir = stage_dir / "prompts"
    prompt_path = prompt_dir / prompt_filename
    
    if not prompt_path.exists():
        logging.error(f"❌ Prompt file not found: {prompt_path}")
        return
    
    prompt_template = load_prompt(prompt_dir, prompt_filename)
    logging.info(f"✅ Loaded prompt template from: {prompt_path}")
    
    # Output artifact name - save to W&B with fixed name (excludes user backstory)
    output_artifact = "sgo_train_final_predictions_seed_backstory_only"
    logging.info(f"Output artifact name: {output_artifact}")
    
    # Initialize W&B run
    run = init_wandb_run(
        run_name="final_predictions_seed_backstory_only",
        stage="07_sgo_training",
        config={
            "description": "Generate final predictions with ONLY seed backstory as context (EXCLUDES user backstory, category characteristics, and user motives)",
            "memory_base_dir": memory_base_dir,
            "refined_chars_base_dir": refined_chars_base_dir,
            "output_artifact": output_artifact
        }
    )
    
    if run is None:
        logging.error("❌ W&B run initialization failed")
        return
    
    try:
        # Init Resources
        api_key = openai_config.get("api_key") or os.environ.get("OPENAI_API_KEY")
        if not api_key:
            logging.error("❌ OPENAI_API_KEY not found")
            return
        
        client = OpenAI(api_key=api_key)
        loader = DataLoader(run, file_patterns, dataset_pattern_mapping)
        limiter = RateLimiter(min_interval=hyperparams.get("min_request_interval"))
        
        # Get dataset_type from config (default to "train")
        dataset_type = sgo_config.get("dataset_type", "train") if sgo_config else "train"
        
        # Check if test data folder exists - if so, force dataset_type to "test"
        project_root = Path(__file__).parent.parent.parent
        test_data_path = project_root / "ground_truth_test_micro_cluster_details_converted"
        if test_data_path.exists() and test_data_path.is_dir():
            dataset_type = "test"
            logging.info(f"✅ Test data folder found, forcing dataset_type to 'test'")
        else:
            logging.info(f"Dataset type from config: {dataset_type}")
        
        # Load Data (same as pre_sgo_predictions)
        logging.info("Loading data from W&B...")
        tribe_seed_data = loader.load_tribe_seeds(tribe_seed_artifact)
        user_backstories = loader.load_user_backstories(user_backstory_artifact)
        
        # Load review data based on dataset_type
        if dataset_type == "test":
            # For test mode: load from local ground_truth_test_micro_cluster_details_converted folder
            
            if not test_data_path.exists():
                logging.error(f"❌ Test data directory not found: {test_data_path}")
                return
            
            if not test_data_path.is_dir():
                logging.error(f"❌ Test data path is not a directory: {test_data_path}")
                return
            
            logging.info(f"✅ Using local test data from: {test_data_path}")
            logging.info(f"📂 Test data path exists: {test_data_path.exists()}")
            logging.info(f"📂 Test data path is directory: {test_data_path.is_dir()}")
            
            # Convert test_micro_cluster_details structure to review_data format
            # Structure: cluster_X/micro_Y_details.json with members_grouped_by_user
            review_data = {}
            user_to_tribe_map = {}
            
            # Process each cluster/micro file
            for cluster_dir in sorted(test_data_path.iterdir()):
                if not cluster_dir.is_dir() or not cluster_dir.name.startswith('cluster_'):
                    continue
                
                cluster_id = cluster_dir.name
                for micro_file in sorted(cluster_dir.glob('micro_*_details.json')):
                    micro_id = micro_file.stem.replace('_details', '')
                    logging.info(f"Loading test data from {cluster_id}/{micro_id}...")
                    logging.info(f"📄 Reading file: {micro_file}")
                    
                    with open(micro_file, 'r', encoding='utf-8') as f:
                        micro_data = json.load(f)
                    
                    logging.info(f"📊 File loaded. Keys in micro_data: {list(micro_data.keys())}")
                    
                    # Extract reviews from members_grouped_by_user
                    # Structure: members_grouped_by_user[user_id] = [review1, review2, ...]
                    members_by_user = micro_data.get('members_grouped_by_user', {})
                    logging.info(f"📊 Found {len(members_by_user)} users in {cluster_id}/{micro_id}")
                    
                    for user_id, user_reviews in members_by_user.items():
                        # user_reviews is already a list of review dicts
                        if not isinstance(user_reviews, list):
                            logging.warning(f"  ⚠️  User {user_id} in {cluster_id}/{micro_id} has non-list data, skipping")
                            continue
                        
                        if user_id not in review_data:
                            review_data[user_id] = {'reviews': []}
                        
                        logging.info(f"  👤 Processing {len(user_reviews)} reviews for user {user_id}")
                        
                        # Process each review - CRITICAL: topic_probabilities is REQUIRED
                        for review_idx, review in enumerate(user_reviews):
                            # Debug: log first review structure
                            if review_idx == 0:
                                asin = review.get('asin', 'unknown')
                                logging.info(f"    📝 First review ASIN: {asin}, Keys: {list(review.keys())}")
                                if 'topic_probabilities' in review:
                                    logging.info(f"    ✅ topic_probabilities found: {type(review['topic_probabilities'])}")
                                else:
                                    logging.error(f"    ❌ topic_probabilities NOT FOUND in review keys!")
                            if not isinstance(review, dict):
                                error_msg = f"CRITICAL ERROR: Review in {cluster_id}/{micro_id} for user {user_id} is not a dict"
                                logging.error(error_msg)
                                raise ValueError(error_msg)
                            
                            # CRITICAL: topic_probabilities is REQUIRED - no fallbacks, script should fail
                            topic_probs = review.get('topic_probabilities')
                            if topic_probs is None:
                                asin = review.get('asin', 'unknown')
                                error_msg = f"CRITICAL ERROR: Review {asin} in {cluster_id}/{micro_id} for user {user_id} missing topic_probabilities. Review keys: {list(review.keys())}"
                                logging.error(error_msg)
                                raise ValueError(error_msg)
                            
                            if not isinstance(topic_probs, dict):
                                asin = review.get('asin', 'unknown')
                                error_msg = f"CRITICAL ERROR: Review {asin} in {cluster_id}/{micro_id} for user {user_id} has topic_probabilities of wrong type: {type(topic_probs)}, expected dict"
                                logging.error(error_msg)
                                raise ValueError(error_msg)
                            
                            if len(topic_probs) == 0:
                                asin = review.get('asin', 'unknown')
                                error_msg = f"CRITICAL ERROR: Review {asin} in {cluster_id}/{micro_id} for user {user_id} has empty topic_probabilities dict"
                                logging.error(error_msg)
                                raise ValueError(error_msg)
                            
                            # Add review to review_data
                            review_data[user_id]['reviews'].append(review)
                        
                        # Map user to tribe (segment_X_micro_Y format)
                        segment_id = cluster_id.replace('cluster_', 'segment_')
                        tribe_id = f"{segment_id}_micro_{micro_id.replace('micro_', '')}"
                        user_to_tribe_map[user_id] = tribe_id
            
            logging.info(f"✅ Loaded {sum(len(ud['reviews']) for ud in review_data.values())} test reviews from {len(review_data)} users")
        else:
            # Train mode: use existing loader
            review_data = loader.load_review_data(training_data_artifact, dataset_type="train")
            user_to_tribe_map = loader.load_user_tribes(user_tribes_artifact)
        
        # Load topic_universe from local file instead of W&B
        topic_universe_file = project_root / "topic_universe_seta_final.json"
        if not topic_universe_file.exists():
            logging.error(f"❌ Topic universe file not found: {topic_universe_file}")
            return
        logging.info(f"Loading topic universe from local file: {topic_universe_file}")
        topic_universe = io_utils.load_json_file(str(topic_universe_file), {})
        if not topic_universe:
            logging.error("❌ Failed to load topic universe from local file")
            return
        logging.info(f"✅ Loaded topic universe from local file with {len(topic_universe)} categories")
        category_mapping = loader.load_category_mapping(category_mapping_artifact)
        
        if not all([tribe_seed_data, user_backstories, review_data, topic_universe, category_mapping]):
            logging.error("❌ Failed to load required data")
            return
        
        # Determine target clusters from arguments (before creating orchestrator)
        target_clusters = None
        if args.only_cluster:
            if args.only_micro:
                # Format: "cluster_X micro_Y" for filtering specific tribe
                target_clusters = [f"{args.only_cluster} {args.only_micro}"]
                logging.info(f"Processing only {args.only_cluster}/{args.only_micro}")
            else:
                target_clusters = [args.only_cluster]
        elif args.start_from_cluster:
            # Filter tribes starting from this cluster
            import re
            try:
                start_num = int(re.search(r'cluster_(\d+)', args.start_from_cluster).group(1))
                # Filter tribe_seed_data to only include clusters >= start_num
                filtered_tribes = {}
                for tribe_id, tribe_seed in tribe_seed_data.items():
                    # Parse tribe_id to get cluster_id (same logic as _parse_tribe_id)
                    if "_micro_" in tribe_id:
                        parts = tribe_id.split("_micro_")
                        raw_cluster = parts[0]
                        match = re.search(r'(\d+)', raw_cluster)
                        cluster_id = f"cluster_{match.group(1)}" if match else raw_cluster
                    else:
                        cluster_id = tribe_id
                    
                    cluster_num = int(re.search(r'cluster_(\d+)', cluster_id).group(1))
                    if cluster_num >= start_num:
                        filtered_tribes[tribe_id] = tribe_seed
                tribe_seed_data = filtered_tribes
                logging.info(f"Filtered to {len(tribe_seed_data)} tribes starting from {args.start_from_cluster}")
            except (AttributeError, ValueError):
                logging.warning(f"Invalid start_from_cluster format: {args.start_from_cluster}. Ignoring.")
        
        # Prepare hyperparams
        output_config = stage_config.get("output", {})
        output_stage = output_config.get("stage", "07_sgo_training")
        hyperparams_with_output = hyperparams.copy()
        hyperparams_with_output["output_stage"] = output_stage
        hyperparams_with_output["scoring_mode"] = scoring_mode
        if theme_prediction_model:
            hyperparams_with_output["theme_prediction_model"] = theme_prediction_model
        
        # Run Pipeline with memory and refined characteristics
        # Get characteristics_mode from config
        characteristics_mode = sgo_config.get("characteristics_mode", "refined") if sgo_config else "refined"
        logging.info(f"Characteristics mode: {characteristics_mode}")
        
        orchestrator = FinalPredictionsRefinedCharsOrchestrator(
            run, client, limiter,
            prompt_template,
            hyperparams_with_output,
            category_mapping=category_mapping,
            memory_base_dir=memory_base_dir,
            refined_chars_base_dir=refined_chars_base_dir,
            characteristics_mode=characteristics_mode,
            user_learnings_base_dir=user_learnings_base_dir
        )
        
        orchestrator.run_pipeline(
            dataset_type=dataset_type,
            tribe_seed_data=tribe_seed_data,
            user_backstories=user_backstories,
            review_data=review_data,
            topic_universe=topic_universe,
            user_to_tribe_map=user_to_tribe_map,
            output_artifact_name=output_artifact,
            target_clusters=target_clusters
        )
        
        logging.info("Final predictions generation with refined characteristics completed successfully!")
        
    except Exception as e:
        logging.error(f"Error in main execution: {e}", exc_info=True)
        if 'run' in locals():
            from utils.wandb_utils import log_summary
            log_summary(run, {"status": "failed", "error": str(e)})
        raise
    
    finally:
        if 'run' in locals() and run is not None:
            finish_run(run)


if __name__ == "__main__":
    main()

