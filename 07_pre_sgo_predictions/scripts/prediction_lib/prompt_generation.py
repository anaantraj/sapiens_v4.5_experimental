from pathlib import Path

def load_prompt(prompt_dir: Path, prompt_filename: str) -> str:
    """Load a prompt template from file."""
    prompt_path = prompt_dir / prompt_filename
    if prompt_path.exists():
        with open(prompt_path, 'r', encoding='utf-8') as f:
            return f.read().strip()
    else:
        raise FileNotFoundError(f"Prompt file not found: {prompt_path}")

def create_enhanced_prompt(
    persona_context: dict,
    user_char_summary: str,
    category_char_summary: str,
    product_description: str,
    category: str,
    category_themes: list,
    prompt_template: str,
    scoring_mode: str = "confidence"
) -> str:
    """
    Enhanced prompt that generates review_text, predicted_themes, sentiment, and rating.
    Uses prompt template from file.
    
    Args:
        scoring_mode: "confidence" (default) or "logprobs"
            - "confidence": Includes theme scoring instructions and JSON template
            - "logprobs": Only generates review, rating, and sentiment (no theme scores)
    """
    persona_name = persona_context.get('persona_name', 'Unknown Persona')
    qual_summary = persona_context.get('qualitative_summary', {})
    
    persona_summary = qual_summary.get('persona_summary', 'N/A')
    key_motivations = qual_summary.get('key_motivations', [])
    common_praises = qual_summary.get('common_praises', [])
    common_criticisms = qual_summary.get('common_criticisms', [])
    core_characteristics = qual_summary.get('core_characteristics', [])
    potential_goals = qual_summary.get('potential_goals', [])
    
    def format_list(items):
        if not items or not isinstance(items, list):
            return "N/A"
        return "\n- ".join([""] + items)
    
    motivations_text = format_list(key_motivations)
    praises_text = format_list(common_praises)
    criticisms_text = format_list(common_criticisms)
    characteristics_text = format_list(core_characteristics)
    goals_text = format_list(potential_goals)
    
    category_section = ""
    if category_char_summary and category_char_summary.strip():
        category_section = f"""
### 3. Your Category-Specific Behavior ({category})
**How You Specifically Behave When Reviewing {category} Products:**
{category_char_summary}
"""
    
    # Format the prompt template
    if scoring_mode == "logprobs":
        # Logprobs mode: no theme scoring instructions needed
        prompt = prompt_template.format(
            persona_name=persona_name,
            persona_summary=persona_summary,
            key_motivations=motivations_text,
            common_praises=praises_text,
            common_criticisms=criticisms_text,
            core_characteristics=characteristics_text,
            potential_goals=goals_text,
            user_char_summary=user_char_summary or "N/A",
            category_section=category_section,
            category=category,
            product_description=product_description
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
            category_section=category_section,
            category=category,
            product_description=product_description,
            themes_list=themes_list,
            themes_json_template=themes_json_template
        )
    
    return prompt

    
