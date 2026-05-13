import time
import json_repair
import re
import sys
from pathlib import Path

# Set up package structure for sgo_training imports
scripts_dir = Path(__file__).parent.parent
if str(scripts_dir) not in sys.path:
    sys.path.insert(0, str(scripts_dir))
import _package_setup

from sgo_training.config import settings
from sgo_training.utils.cost_tracker import track_api_call, extract_token_usage

def process_part_a_batch(prompt, client, batch_size, required_themes=None):
    """
    Executes Part A: Correction with retry logic
    Args:
        prompt: The prompt to send to the LLM
        client: OpenAI client
        batch_size: Expected number of corrections
        required_themes: List of theme names that MUST be present in predicted_themes (optional, for validation)
    Returns:
        List of corrections (dicts) or empty dicts for failed corrections
    """
    max_retries = getattr(settings, 'MAX_RETRIES', 3)
    retry_delay = 2  # Base delay in seconds
    
    for attempt in range(max_retries):
        try:
            time.sleep(settings.API_RATE_LIMIT_DELAY)
            
            # Enhanced system message that emphasizes ALL themes requirement
            system_message = """You are a persona-based review prediction agent. Your response must be a single, valid JSON object containing a 'corrections' array.

CRITICAL REQUIREMENTS:
1. Each correction MUST include: review_text, rating, sentiment, and predicted_themes
2. The predicted_themes object MUST contain EVERY theme from "Possible Review Themes" with confidence scores (0.0-1.0)
3. DO NOT omit any theme - missing themes will cause validation errors
4. Theme names must match EXACTLY (case-sensitive) the theme names provided in the prompt"""
            
            # Use REVIEW_PREDICTION_MODEL for Part A (review generation/correction)
            model_to_use = getattr(settings, 'REVIEW_PREDICTION_MODEL', None)
            if model_to_use is None:
                # Fallback to ANALYSIS_MODEL for backward compatibility
                model_to_use = settings.ANALYSIS_MODEL
            
            response = client.chat.completions.create(
                model=model_to_use,
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": prompt}
                ],
            )
            
            # Track API call for cost calculation
            input_tokens, output_tokens, cached_tokens = extract_token_usage(response)
            track_api_call(
                model=model_to_use,
                call_type="part_a_batch",
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cached_input_tokens=cached_tokens,
                prompt_length=len(system_message) + len(prompt),
                metadata={"batch_size": batch_size, "attempt": attempt + 1}
            )
            
            text = response.choices[0].message.content.strip()
            
            # Debug: Log raw response (first 500 chars) to see what LLM returned
            if attempt == 0:  # Only log on first attempt to avoid spam
                print(f"  - DEBUG: LLM response preview (first 500 chars): {text[:500]}...")
            
            parsed = json_repair.loads(text)
            corrections = parsed.get("corrections", [])
            
            if not isinstance(corrections, list) or len(corrections) != batch_size:
                if attempt < max_retries - 1:
                    wait_time = retry_delay * (2 ** attempt)
                    print(f"  - WARNING: Expected {batch_size} corrections, got {len(corrections) if isinstance(corrections, list) else 'non-list'}. Retrying in {wait_time}s... (attempt {attempt+1}/{max_retries})")
                    time.sleep(wait_time)
                    continue
                else:
                    print(f"  - ERROR: Expected {batch_size} corrections, got {len(corrections) if isinstance(corrections, list) else 'non-list'} after {max_retries} attempts")
                    return [{} for _ in range(batch_size)]
            
            # Debug: Log what we got from LLM
            print(f"  - DEBUG: Received {len(corrections)} corrections from LLM")
            for i, corr in enumerate(corrections[:2]):  # Show first 2
                if isinstance(corr, dict):
                    has_review_text = 'review_text' in corr
                    has_predicted_themes = 'predicted_themes' in corr
                    pred_themes_type = type(corr.get('predicted_themes', None)).__name__
                    pred_themes_len = len(corr.get('predicted_themes', {})) if isinstance(corr.get('predicted_themes'), dict) else 0
                    print(f"  - DEBUG: Correction {i}: review_text={has_review_text}, predicted_themes={has_predicted_themes} (type={pred_themes_type}, len={pred_themes_len})")
            
            # ALWAYS validate that predicted_themes exists (even if required_themes is not provided)
            validated_corrections = []
            for i, correction in enumerate(corrections):
                if not isinstance(correction, dict):
                    print(f"  - ❌ VALIDATION FAILURE: Correction {i} is not a dict: {type(correction)}")
                    print(f"  -    This means the LLM returned invalid format. Check the LLM response structure.")
                    validated_corrections.append({})
                    continue
                
                # CRITICAL: Always check for predicted_themes (required field)
                pred_themes = correction.get('predicted_themes', {})
                if not isinstance(pred_themes, dict):
                    print(f"  - ❌ VALIDATION FAILURE: Correction {i} predicted_themes is not a dict")
                    print(f"  -    Type: {type(pred_themes)}, Value: {pred_themes}")
                    print(f"  -    Correction keys: {list(correction.keys())}")
                    print(f"  -    LLM returned predicted_themes in wrong format (expected dict, got {type(pred_themes).__name__})")
                    validated_corrections.append({})
                    continue
                
                # Check if predicted_themes is empty dict
                if not pred_themes:
                    print(f"  - ❌ VALIDATION FAILURE: Correction {i} predicted_themes is empty dict!")
                    print(f"  -    Correction has keys: {list(correction.keys())}")
                    print(f"  -    LLM returned empty predicted_themes - this usually means the LLM didn't include theme predictions")
                    validated_corrections.append({})
                    continue
                
                # If required_themes provided, validate all themes are present
                if required_themes and len(required_themes) > 0:
                    missing_themes = []
                    for required_theme in required_themes:
                        # Check if exact match exists
                        if required_theme in pred_themes:
                            continue
                        
                        # Check if LLM split compound theme (e.g., "Quality, Durability and Longevity" -> "Quality" + "Durability" + "Longevity")
                        # Try to find if components of the required theme exist
                        theme_found = False
                        
                        # Extract all individual words/components from the required theme
                        # Handle both comma-separated and "and"-separated parts
                        # Example: "Quality, Durability and Longevity" -> ["Quality", "Durability", "Longevity"]
                        import re
                        # Split by comma first, then by "and" in each part
                        all_components = []
                        if ',' in required_theme:
                            comma_parts = [p.strip() for p in required_theme.split(',')]
                            for part in comma_parts:
                                if ' and ' in part.lower():
                                    # Split by "and" as well
                                    and_parts = [p.strip() for p in re.split(r'\s+and\s+', part, flags=re.IGNORECASE)]
                                    all_components.extend(and_parts)
                                else:
                                    all_components.append(part)
                        elif ' and ' in required_theme.lower():
                            all_components = [p.strip() for p in re.split(r'\s+and\s+', required_theme, flags=re.IGNORECASE)]
                        else:
                            # Single word theme
                            all_components = [required_theme.strip()]
                        
                        # Remove empty strings
                        all_components = [c for c in all_components if c]
                        
                        # Now check if we can find these components in the returned themes
                        found_components = []
                        for component in all_components:
                            component_lower = component.lower()
                            component_found = False
                            
                            # Check exact match first
                            if component in pred_themes:
                                found_components.append(component)
                                component_found = True
                            else:
                                # Check if component is a significant word in any returned theme
                                for returned_theme in pred_themes.keys():
                                    returned_lower = returned_theme.lower()
                                    # Check if component is a standalone word or significant substring
                                    # Use word boundaries or check if it's a significant part
                                    if (component_lower == returned_lower or 
                                        component_lower in returned_lower or 
                                        returned_lower in component_lower or
                                        # Check if component is a significant word (not just a substring like "and" in "band")
                                        re.search(r'\b' + re.escape(component_lower) + r'\b', returned_lower)):
                                        found_components.append(returned_theme)
                                        component_found = True
                                        break
                            
                            # Debug: Log if component not found
                            if not component_found and i < 2:  # Only log for first 2 corrections to avoid spam
                                print(f"  - DEBUG: Component '{component}' not found in returned themes: {list(pred_themes.keys())[:5]}")
                        
                        # For compound themes, be very lenient - accept if ANY component is found
                        # This handles cases where LLM splits compound themes like "Quality, Durability and Longevity" -> "Quality"
                        # The LLM may only return the most relevant component, which is acceptable
                        if len(all_components) > 0 and len(found_components) >= 1:
                            theme_found = True
                            print(f"  - ✅ Theme '{required_theme}' found as split components: {found_components} (found {len(found_components)}/{len(all_components)} components)")
                        else:
                            # Debug: Log why validation failed
                            if i < 2:  # Only log for first 2 corrections
                                print(f"  - DEBUG: Theme '{required_theme}' validation failed:")
                                print(f"  -   Components: {all_components}")
                                print(f"  -   Found: {found_components}")
                                print(f"  -   Required: at least 1, Found: {len(found_components)}")
                                print(f"  -   LLM returned themes: {list(pred_themes.keys())}")
                        
                        if not theme_found:
                            missing_themes.append(required_theme)
                    
                    if missing_themes:
                        failure_msg = f"missing_required_themes: {missing_themes}"
                        print(f"  - ❌ VALIDATION FAILURE: Correction {i} missing {len(missing_themes)}/{len(required_themes)} required themes")
                        print(f"  -    Missing themes: {missing_themes[:5]}{'...' if len(missing_themes) > 5 else ''}")
                        print(f"  -    LLM returned themes: {list(pred_themes.keys())[:5]}{'...' if len(pred_themes) > 5 else ''}")
                        print(f"  -    This means the LLM didn't include all required themes in its response")
                        # Store failure reason in the empty dict for later extraction
                        failed_correction = {}
                        failed_correction['_failure_reason'] = failure_msg
                        validated_corrections.append(failed_correction)
                        continue
                    
                    # Check if all values are valid numbers
                    invalid_scores = []
                    for theme, score in pred_themes.items():
                        try:
                            score_float = float(score)
                            if not (0.0 <= score_float <= 1.0):
                                invalid_scores.append(f"{theme}:{score}")
                        except (ValueError, TypeError):
                            invalid_scores.append(f"{theme}:{score}")
                    
                    if invalid_scores:
                        print(f"  - ❌ VALIDATION FAILURE: Correction {i} has invalid confidence scores")
                        print(f"  -    Invalid scores: {invalid_scores[:5]}{'...' if len(invalid_scores) > 5 else ''}")
                        print(f"  -    Scores must be numbers between 0.0 and 1.0")
                        validated_corrections.append({})
                        continue
                else:
                    # No required_themes - can't validate completeness, but at least predicted_themes exists
                    if required_themes is None:
                        print(f"  - ⚠️  WARNING: required_themes is None - only checking predicted_themes exists (not completeness)")
                    elif len(required_themes) == 0:
                        print(f"  - ⚠️  WARNING: required_themes is empty - only checking predicted_themes exists (not completeness)")
                
                # Validation passed
                validated_corrections.append(correction)
            
            return validated_corrections
            
        except Exception as e:
            if attempt < max_retries - 1:
                wait_time = retry_delay * (2 ** attempt)
                print(f"  - WARNING: Part A API call failed (attempt {attempt+1}/{max_retries}): {e}. Retrying in {wait_time}s...")
                time.sleep(wait_time)
                continue
            else:
                print(f"  - ERROR: Part A API call failed after {max_retries} attempts: {e}")
                import traceback
                traceback.print_exc()
                return [{} for _ in range(batch_size)]
    
    # Should never reach here, but just in case
    print(f"  - ERROR: Part A API call failed after {max_retries} attempts")
    return [{} for _ in range(batch_size)]

def process_part_b_batch(prompt, client, batch_size):
    """
    Makes one call to the OpenAI API for Part B batch and returns a list of analyses.
    Includes retry logic with exponential backoff.
    """
    max_retries = getattr(settings, 'MAX_RETRIES', 3)
    retry_delay = 2  # Base delay in seconds
    
    for attempt in range(max_retries):
        try:
            # Add delay to reduce rate limit hits
            time.sleep(settings.API_RATE_LIMIT_DELAY)
            
            # Use FEEDBACK_LOOP_MODEL for Part B (feedback loop analysis)
            model_to_use = getattr(settings, 'FEEDBACK_LOOP_MODEL', None)
            if model_to_use is None:
                # Fallback to ANALYSIS_MODEL for backward compatibility
                model_to_use = settings.ANALYSIS_MODEL
            
            system_message = "You are a deep analytical expert in consumer behavior and persona psychology. Your task is root cause analysis: identify WHY predictions fail and what SPECIFIC characteristics are missing or need refinement. Your response must be a valid JSON object with an 'analyses' array containing one analysis string per review in the format `[Characteristic 1, Characteristic 2] [Root cause explanation]`."
            
            # Check if model is gpt-5.2 (uses reasoning_effort parameter)
            if "gpt-5.2" in model_to_use.lower():
                # Use chat.completions.create with reasoning_effort for gpt-5.2
                response = client.chat.completions.create(
                    model=model_to_use,
                    messages=[
                        {"role": "system", "content": system_message},
                        {"role": "user", "content": prompt}
                    ],
                    reasoning_effort="medium"
                )
            else:
                # Use standard chat.completions.create API for other models
                response = client.chat.completions.create(
                    model=model_to_use,
                    messages=[
                        {"role": "system", "content": system_message},
                        {"role": "user", "content": prompt}
                    ],
                )
            
            # Track API call for cost calculation
            input_tokens, output_tokens, cached_tokens = extract_token_usage(response)
            track_api_call(
                model=model_to_use,
                call_type="part_b_batch",
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cached_input_tokens=cached_tokens,
                prompt_length=len(system_message) + len(prompt),
                metadata={"batch_size": batch_size, "attempt": attempt + 1, "reasoning_effort": "medium" if "gpt-5.2" in model_to_use.lower() else None}
            )
            
            llm_response_text = response.choices[0].message.content.strip()

            # Use json_repair for robustness
            parsed_response = json_repair.loads(llm_response_text)

            analyses = parsed_response.get("analyses", [])

            # Check that we got a list and the count matches
            if not isinstance(analyses, list) or len(analyses) != batch_size:
                if attempt < max_retries - 1:
                    wait_time = retry_delay * (2 ** attempt)
                    print(f"  - WARNING: Part B batch response mismatch. Expected {batch_size} items, got {len(analyses)}. Retrying in {wait_time}s... (attempt {attempt+1}/{max_retries})")
                    time.sleep(wait_time)
                    continue
                else:
                    print(f"  - WARNING: OpenAI Part B batch response mismatch after {max_retries} attempts. Expected {batch_size} items in a list, got {len(analyses)}. Using error markers.")
                    return [f"[LLM Error] [Response mismatch: expected {batch_size} analyses, got {len(analyses)}]"] * batch_size

            # Parse each analysis string to ensure proper format
            parsed_analyses = []
            for i, analysis_str in enumerate(analyses):
                if not isinstance(analysis_str, str):
                    analysis_str = str(analysis_str)
                
                # Use the same parsing logic as individual processor
                try:
                    # 1. Check for the "perfect" format: [Motive] [Explanation]
                    analysis_match = re.search(r"(\[.*?\]\s*\[.*?\])", analysis_str, re.DOTALL)
                    if analysis_match:
                        analysis_string = analysis_match.group(1).strip()
                        # Check if motives part is None, empty, or indicates no missed motives
                        motives_match = re.search(r"\[(.*?)\]", analysis_string)
                        if motives_match:
                            motives_content = motives_match.group(1).strip()
                            if not motives_content or motives_content.lower() in ['none', 'no missed motives', 'no themes missed', 'n/a', 'na']:
                                parsed_analyses.append("[No missed motives] [No additional themes or motives were missed from the actual review]")
                                continue
                        parsed_analyses.append(analysis_string)
                        continue

                    # 2. Check for the "common error" format: [Motive] Explanation
                    analysis_match_flexible = re.search(r"(\[.*?\])\s*(.*)", analysis_str, re.DOTALL)
                    if analysis_match_flexible:
                        motives_part = analysis_match_flexible.group(1).strip()
                        explanation_part = analysis_match_flexible.group(2).strip()
                        
                        if not explanation_part:
                            parsed_analyses.append(f"[LLM Error] [Found motives but no explanation: {analysis_str}]")
                            continue

                        # Check if motives part indicates no missed motives
                        motives_content_match = re.search(r"\[(.*?)\]", motives_part)
                        if motives_content_match:
                            motives_content = motives_content_match.group(1).strip()
                            if not motives_content or motives_content.lower() in ['none', 'no missed motives', 'no themes missed', 'n/a', 'na']:
                                parsed_analyses.append("[No missed motives] [No additional themes or motives were missed from the actual review]")
                                continue

                        analysis_string = f"{motives_part} [{explanation_part}]"
                        parsed_analyses.append(analysis_string)
                        continue

                    # 3. If format is unknown, return as-is with error marker
                    parsed_analyses.append(f"[LLM Error] [Output format mismatch: {analysis_str}]")
                except Exception as e:
                    parsed_analyses.append(f"[LLM Error] [Parsing failed: {str(e)}]")

            return parsed_analyses

        except Exception as e:
            if attempt < max_retries - 1:
                wait_time = retry_delay * (2 ** attempt)
                print(f"  - WARNING: Part B API call failed (attempt {attempt+1}/{max_retries}): {e}. Retrying in {wait_time}s...")
                time.sleep(wait_time)
                continue
            else:
                print(f"  - ERROR: OpenAI Part B batch API call failed after {max_retries} attempts. Error: {e}")
                # Return a default list of error messages
                return [f"[LLM Error] [API call failed after {max_retries} attempts: {e}]"] * batch_size
    
    # Should never reach here, but just in case
    print(f"  - ERROR: Part B API call failed after {max_retries} attempts")
    return [f"[LLM Error] [API call failed after {max_retries} attempts]"] * batch_size
