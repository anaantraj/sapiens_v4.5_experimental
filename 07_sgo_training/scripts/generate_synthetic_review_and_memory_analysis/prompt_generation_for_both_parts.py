import os
import sys
from pathlib import Path

# Set up package structure for sgo_training imports
scripts_dir = Path(__file__).parent.parent
if str(scripts_dir) not in sys.path:
    sys.path.insert(0, str(scripts_dir))
import _package_setup

from sgo_training.config import settings
from sgo_training.utils import io_utils

# --- Prompt Generation Functions ---

def generate_part_a_batch_prompt(
    micro_persona_characteristics, 
    user_characteristics_list, 
    persona_name, 
    batch_data_list, 
    persona_memory, 
    relevant_themes, 
    batch_motives=None,
    user_chars_data=None,  # For per-review user characteristics
    category_mapping=None,  # For category-specific characteristics
    scoring_mode="confidence"  # For logprobs mode, themes section will be removed
):
    """
    Generates a prompt for Part A: Correction, for a whole batch.
    Strict Mode: Raises FileNotFoundError if prompt template is missing.
    """
    # 1. Load Template (Strict) - use logprobs version if in logprobs mode
    if scoring_mode in ["logprobs", "logprobs-without-persona-context"]:
        template_name = "part_a_batch_prompt_logprobs"
    else:
        template_name = "part_a_batch_prompt"
    
    template = io_utils.load_prompt(template_name, settings.PROMPT_DIR_PART_A)
    if not template:
        raise FileNotFoundError(
            f"CRITICAL: '{template_name}.txt' not found in {settings.PROMPT_DIR_PART_A}. "
            "Aborting execution to prevent malformed prompt generation."
        )
    

    # 2. Build Context String
    profile_parts = []
    
    # Micro Persona Characteristics
    if micro_persona_characteristics:
        persona_summary = micro_persona_characteristics.get('persona_summary', '')
        if persona_summary:
            profile_parts.append(f"- **Micro Persona Characteristics:** {persona_summary}")
    
    # User Characteristics - Only add if provided (legacy support)
    # Note: Per-review user characteristics will be added in reviews section
    if user_characteristics_list:
        if isinstance(user_characteristics_list, list):
            user_chars_text = ' '.join(user_characteristics_list)
        else:
            user_chars_text = str(user_characteristics_list)
        profile_parts.append(f"- **User Characteristics (Aggregated - Legacy):** {user_chars_text}")
    
    if not profile_parts:
        # We allow this, but log it via context string. 
        # If strict data validation is needed, raise ValueError here.
        profile_parts.append("- **Context:** No profile or characteristics provided.")
        
    context_str = "\n".join(profile_parts)

    # 3. Format Reviews
    if not batch_data_list:
        raise ValueError("CRITICAL: batch_data_list is empty. Cannot generate batch prompt.")

    reviews_to_process_str = ""
    # Check if batch_data_list contains full review info (with user_id) or just data
    # If it's full review info, extract user characteristics per review
    is_full_review_info = isinstance(batch_data_list[0], dict) and 'user_id' in batch_data_list[0] if batch_data_list else False
    
    # Cache user characteristics per (user_id, category) to avoid recomputation
    # Key: (user_id, main_category), Value: formatted characteristics string
    user_chars_cache = {}
    
    def get_user_characteristics_cached(user_id, review_category):
        """Get user characteristics with caching to avoid recomputation for same user+category."""
        if user_id == 'unknown' or not user_chars_data or user_id not in user_chars_data:
            return 'None'
        
        # Map to main category for cache key
        if review_category and category_mapping:
            if isinstance(category_mapping, dict) and 'category_to_main_mapping' in category_mapping:
                main_category = category_mapping['category_to_main_mapping'].get(review_category, review_category)
            else:
                main_category = category_mapping.get(review_category, review_category)
        else:
            main_category = review_category
        
        cache_key = (user_id, main_category)
        if cache_key in user_chars_cache:
            return user_chars_cache[cache_key]
        
        # Compute user characteristics
        user_data = user_chars_data[user_id]
        characteristics = []
        
        # General characteristics
        llm_chars = user_data.get('llm_characteristics', {})
        if isinstance(llm_chars, dict):
            user_char = llm_chars.get('influencing_characteristics_summary', '')
            if user_char:
                characteristics.append(f"[General Characteristics] {user_char}")
        
        # Category-specific characteristics
        if main_category:
            category_chars = user_data.get('category_characteristics', {})
            if isinstance(category_chars, dict) and main_category in category_chars:
                cat_char = category_chars[main_category].get('influencing_characteristics_summary', '')
                if cat_char:
                    characteristics.append(f"[{main_category} Specific] {cat_char}")
        
        user_chars_text = ' '.join(characteristics) if characteristics else 'None'
        user_chars_cache[cache_key] = user_chars_text
        return user_chars_text
    
    for i, item in enumerate(batch_data_list):
        # Handle both formats: full review info or just data
        if is_full_review_info:
            # Full review info format: {user_id, review_idx, data, ...}
            user_id = item.get('user_id', 'unknown')
            data = item.get('data', item)  # Fallback to item itself if 'data' key doesn't exist
            review_category = data.get('category', 'unknown') if isinstance(data, dict) else 'unknown'
        else:
            # Legacy format: just data dict
            data = item
            user_id = 'unknown'
            review_category = data.get('category', 'unknown') if isinstance(data, dict) else 'unknown'
        
        # Get user characteristics for this specific review (with caching)
        user_chars_text = get_user_characteristics_cached(user_id, review_category)
        
        # Get the original failed prediction
        original_pred = data.get('prediction', {}) if isinstance(data, dict) else {}
        original_themes = original_pred.get('predicted_themes', {})
        
        # Format the original prediction info
        original_pred_str = f"Rating: {original_pred.get('rating', 'N/A')}, Sentiment: {original_pred.get('sentiment', 'N/A')}"
        if original_themes:
            top_3_themes = sorted(original_themes.items(), key=lambda x: x[1], reverse=True)[:3]
            themes_str = ', '.join([f"{theme} ({score:.2f})" for theme, score in top_3_themes])
            original_pred_str += f", Top 3 Themes: {themes_str}"
        
        reviews_to_process_str += (
            f"\n--- Review Index {i} ---\n"
            f"User Characteristics: {user_chars_text}\n"
            f"Product Description: \"{data.get('product_description', 'N/A') if isinstance(data, dict) else 'N/A'}\"\n"
            f"Original (Failed) Prediction: {original_pred_str}\n"
        )

    # 4. Format Motives (Analysis from Part B)
    motives_section = ""
    if batch_motives:
        motives_list = []
        for analysis in batch_motives:
            # Filter out error messages or empty analyses
            if "LLM Error" not in analysis and "No missed motives" not in analysis:
                motives_list.append(analysis)
        
        if motives_list:
            motives_section = f"\n**Recent Failure Analyses (Motives & Explanations):**\n" + "\n".join([f"- {m}" for m in motives_list])

    # 5. Format Memory (Past Learnings)
    persona_data = persona_memory.get(persona_name, {})
    if isinstance(persona_data, list):
        enrichment_log = persona_data
    else:
        enrichment_log = persona_data.get('batch_analyses', [])
        
    past_learnings = ' | '.join([str(item) for item in enrichment_log]) if enrichment_log else 'None - this is the first iteration'
    
    # 6. Format Relevant Themes (skip for logprobs mode - themes will be classified separately)
    if scoring_mode in ["logprobs", "logprobs-without-persona-context"]:
        relevant_themes_str = "N/A - Themes will be classified separately using logprobs"
    else:
        relevant_themes_str = ', '.join(relevant_themes) if relevant_themes else 'None'
    
    # 7. Fill Template
    return template.format(
        context_str=context_str,
        past_learnings=past_learnings,
        motives_section=motives_section,
        relevant_themes=relevant_themes_str,
        reviews_to_process_str=reviews_to_process_str.strip()
    )

def generate_part_a_individual_prompt_logprobs(
    micro_persona_characteristics,
    user_characteristics_text,
    persona_name,
    review_data,
    persona_memory,
    batch_motives=None,
    category_mapping=None
):
    """
    Generates an individual prompt for Part A: Correction in logprobs mode (single review, not batch).
    This is used when scoring_mode is "logprobs" or "logprobs-without-persona-context".
    
    Args:
        micro_persona_characteristics: Persona profile dict
        user_characteristics_text: User characteristics string for this specific review
        persona_name: Name of the persona
        review_data: Single review data dict (with 'prediction', 'category', 'product_description', etc.)
        persona_memory: Persona memory dict
        batch_motives: List of batch analyses (optional)
        category_mapping: Category mapping dict (optional)
    
    Returns:
        Formatted prompt string for a single review
    """
    # 1. Load Template (Strict)
    template = io_utils.load_prompt("part_a_individual_prompt_logprobs", settings.PROMPT_DIR_PART_A)
    if not template:
        raise FileNotFoundError(
            f"CRITICAL: 'part_a_individual_prompt_logprobs.txt' not found in {settings.PROMPT_DIR_PART_A}. "
            "Aborting execution to prevent malformed prompt generation."
        )
    
    # 2. Build Context String (persona profile)
    profile_parts = []
    if micro_persona_characteristics:
        persona_summary = micro_persona_characteristics.get('persona_summary', '')
        if persona_summary:
            profile_parts.append(f"- **Micro Persona Characteristics:** {persona_summary}")
    
    if not profile_parts:
        profile_parts.append("- **Context:** No profile provided.")
    context_str = "\n".join(profile_parts)
    
    # 3. Format Past Learnings
    persona_data = persona_memory.get(persona_name, {})
    if isinstance(persona_data, list):
        enrichment_log = persona_data
    else:
        enrichment_log = persona_data.get('batch_analyses', [])
    past_learnings = ' | '.join([str(item) for item in enrichment_log]) if enrichment_log else 'None - this is the first iteration'
    
    # 4. Format Motives (Analysis from Part B)
    motives_section = ""
    if batch_motives:
        motives_list = []
        for analysis in batch_motives:
            if "LLM Error" not in analysis and "No missed motives" not in analysis:
                motives_list.append(analysis)
        
        if motives_list:
            motives_section = f"\n**Recent Failure Analyses (Motives & Explanations):**\n" + "\n".join([f"- {m}" for m in motives_list])
    
    # 5. Format Original Prediction
    original_pred = review_data.get('prediction', {}) if isinstance(review_data, dict) else {}
    original_themes = original_pred.get('predicted_themes', {})
    
    original_pred_str = f"Rating: {original_pred.get('rating', 'N/A')}, Sentiment: {original_pred.get('sentiment', 'N/A')}"
    if original_themes:
        top_3_themes = sorted(original_themes.items(), key=lambda x: x[1], reverse=True)[:3]
        themes_str = ', '.join([f"{theme} ({score:.2f})" for theme, score in top_3_themes])
        original_pred_str += f", Top 3 Themes: {themes_str}"
    
    # 6. Get Product Description
    product_description = review_data.get('product_description', 'N/A') if isinstance(review_data, dict) else 'N/A'
    
    # 7. Fill Template
    return template.format(
        context_str=context_str,
        past_learnings=past_learnings,
        motives_section=motives_section,
        user_characteristics=user_characteristics_text or 'None',
        product_description=product_description,
        original_prediction=original_pred_str
    )

def generate_part_b_batch_prompt(
    micro_persona_characteristics, 
    persona_name, 
    batch_reviews_data, 
    persona_memory, 
    user_chars_data, 
    iteration_number=1
):
    """
    Generates a prompt for Part B: Analysis, for a BATCH of reviews.
    Strict Mode: Raises FileNotFoundError if prompt template is missing.
    """
    # 1. Load Template (Strict)
    template = io_utils.load_prompt("part_b_batch_prompt", settings.PROMPT_DIR_PART_B)
    if not template:
        raise FileNotFoundError(
            f"CRITICAL: 'part_b_batch_prompt.txt' not found in {settings.PROMPT_DIR_PART_B}. "
            "Aborting execution."
        )

    # 2. Build Context String
    profile_parts = []
    if micro_persona_characteristics:
        persona_summary = micro_persona_characteristics.get('persona_summary', '')
        if persona_summary:
            profile_parts.append(f"- **Micro Persona Characteristics:** {persona_summary}")
    
    if not profile_parts:
        profile_parts.append("- **Context:** No profile provided.")
    context_str = "\n".join(profile_parts)

    # 3. Build Reviews Section
    if not batch_reviews_data:
        raise ValueError("CRITICAL: batch_reviews_data is empty. Cannot generate batch prompt.")

    reviews_section = ""
    for i, review_info in enumerate(batch_reviews_data):
        review_data = review_info['data']
        user_id = review_info.get('user_id', 'unknown')
        
        user_chars_text = 'None'
        if user_chars_data and user_id in user_chars_data:
             data = user_chars_data[user_id]
             summary = data.get('llm_characteristics', {}).get('influencing_characteristics_summary', '')
             if summary: user_chars_text = f"[General] {summary}"

        # Get Themes
        predicted_themes = review_data['prediction'].get('predicted_themes', {})
        actual_themes = review_data['actual'].get('predicted_themes', [])
        failed_text = review_data['prediction'].get('review_text', 'N/A')
        
        k = len(actual_themes) if actual_themes else 0
        n = max(3, k)
        
        actual_themes_set = set(str(theme).strip().lower() for theme in actual_themes) if isinstance(actual_themes, list) else set()
        
        # Find incorrect themes in top N
        incorrect_themes = []
        if isinstance(predicted_themes, dict):
            top_n_predicted = sorted(predicted_themes.items(), key=lambda x: x[1], reverse=True)[:n]
            for theme, score in top_n_predicted:
                if str(theme).strip().lower() not in actual_themes_set:
                    incorrect_themes.append(theme)
        
        incorrect_str = ', '.join(incorrect_themes) if incorrect_themes else "None (all top predicted themes correct)"
        
        # Format Lists
        actual_str = ', '.join([str(t) for t in actual_themes]) if actual_themes else 'None'
        pred_keys = list(predicted_themes.keys()) if isinstance(predicted_themes, dict) else []
        pred_str = ', '.join([str(t) for t in pred_keys[:n]]) if pred_keys else 'None'
        
        prediction_context = ""
        if iteration_number > 1:
            prediction_context = f"**NOTE: This is a CORRECTED prediction from iteration {iteration_number - 1} that still needs improvement.**"

        reviews_section += f"""
--- Review Index {i} ---
{prediction_context}
**User Characteristics:** {user_chars_text}
**Product Description:** "{review_data.get('product_description', 'N/A')}"
**Actual Review Text:** "{review_data['actual']['review_text']}"
**Actual Themes:** {actual_str}
**Failed Prediction Text:** "{failed_text}"
**Predicted Themes (top {n}):** {pred_str}
**Incorrectly Predicted:** {incorrect_str}
"""

    # 4. Format Memory
    persona_data = persona_memory.get(persona_name, {})
    if isinstance(persona_data, list):
        enrichment_log = persona_data
    else:
        enrichment_log = persona_data.get('batch_analyses', [])
        
    past_learnings = ' | '.join([str(item) for item in enrichment_log]) if enrichment_log else 'None'

    # 5. Fill Template
    return template.format(
        context_str=context_str,
        refined_chars_section="", 
        past_learnings=past_learnings,
        reviews_section=reviews_section.strip()
    )
def generate_part_b_individual_prompt(micro_persona_characteristics, user_characteristics_list, persona_name, review_data, persona_memory):
    """Generates a prompt for Part B: Analysis, for a SINGLE review. Loads template from file."""
    # Load prompt template from file
    prompt_template = load_prompt("part_b_individual_prompt", PROMPT_DIR_PART_B)
    if not prompt_template:
        logging.error("part_b_individual_prompt.txt not found. Cannot generate prompt.")
        return ""
    
    # Handle both old format (list) and new format (dict with batch_analyses)
    persona_data = persona_memory.get(persona_name, {})
    if isinstance(persona_data, list):
        enrichment_log = persona_data
    else:
        enrichment_log = persona_data.get('batch_analyses', [])

    # Get predicted and actual themes to identify incorrect ones in top 3
    predicted_themes = review_data['prediction'].get('predicted_themes', {})
    actual_themes = review_data['actual'].get('predicted_themes', [])
    k = len(actual_themes) if actual_themes else 0
    n = max(3, k)
    
    # Convert actual themes to a set for comparison (normalize strings)
    actual_themes_set = set(str(theme).strip().lower() for theme in actual_themes) if isinstance(actual_themes, list) else set()
    
    # Find incorrectly predicted themes from TOP n
    incorrect_themes_top_n = []
    if isinstance(predicted_themes, dict):
        # Get top n predicted themes (sorted by score)
        top_n_predicted = sorted(predicted_themes.items(), key=lambda x: x[1], reverse=True)[:n]
        for theme, score in top_n_predicted:
            theme_normalized = str(theme).strip().lower()
            # Check if this theme is not in actual themes
            if theme_normalized not in actual_themes_set:
                incorrect_themes_top_n.append(theme)
    
    # Format incorrect themes string (from top n)
    if incorrect_themes_top_n:
        incorrect_themes_str = ', '.join(incorrect_themes_top_n)
    else:
        incorrect_themes_str = "None (all top n predicted themes are correct)"
    
    review_to_process_str = (
        f"Product Description: \"{review_data.get('product_description', 'N/A')}\"\n"
        f"Actual Review Text: \"{review_data['actual']['review_text']}\"\n"
        f"Incorrectly Predicted Themes (from top {n}): {incorrect_themes_str}\n"
    )
    
    # Get persona summary text
    persona_summary_text = ""
    if micro_persona_characteristics:
        persona_summary_text = micro_persona_characteristics.get('persona_summary', '')
    
    # Get actual themes as a list for better context
    actual_themes_list = actual_themes if isinstance(actual_themes, list) else []
    actual_themes_str = ', '.join([str(t) for t in actual_themes_list]) if actual_themes_list else 'None'
    
    # Get predicted themes as a list for comparison
    predicted_themes_list = list(predicted_themes.keys()) if isinstance(predicted_themes, dict) else []
    predicted_themes_str = ', '.join([str(t) for t in predicted_themes_list[:n]]) if predicted_themes_list else 'None'
    
    # Format past learnings
    past_learnings = ' | '.join([str(item) for item in enrichment_log]) if enrichment_log else 'None'
    
    # Format the prompt using the template
    prompt = prompt_template.format(
        persona_summary_text=persona_summary_text if persona_summary_text else 'N/A',
        past_learnings=past_learnings,
        review_to_process_str=review_to_process_str.strip(),
        actual_themes_str=actual_themes_str,
        predicted_themes_str=predicted_themes_str,
        incorrect_themes_str=incorrect_themes_str
    )
    
    return prompt
