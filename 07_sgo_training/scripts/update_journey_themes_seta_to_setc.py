#!/usr/bin/env python3
"""
Update journey files to replace Set A themes with Set C (ground truth) themes.
Maps separate themes to combined themes where appropriate.
"""

import json
from pathlib import Path
from typing import Dict, Any, Optional
import logging
import re

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_theme_mapping() -> Dict[str, Dict[str, str]]:
    """Load theme mapping from Set A to Set C."""
    mapping_file = Path('theme_mapping_seta_to_setc.json')
    if not mapping_file.exists():
        logger.error(f"Mapping file not found: {mapping_file}")
        return {}
    
    with open(mapping_file, 'r') as f:
        return json.load(f)


def load_category_mapping() -> Dict[str, str]:
    """Load category mapping from category_mapping_to_7_main.json."""
    mapping_file = Path('category_mapping_to_7_main.json')
    if not mapping_file.exists():
        logger.warning(f"Category mapping file not found: {mapping_file}, using fallback")
        return {}
    
    try:
        with open(mapping_file, 'r') as f:
            data = json.load(f)
        return data.get('category_to_main_mapping', {})
    except Exception as e:
        logger.warning(f"Error loading category mapping: {e}, using fallback")
        return {}


def get_category_from_journey_file(journey_file: Path, category_mapping: Dict[str, str]) -> Optional[str]:
    """Extract category from journey file path or content and map to main category."""
    # Try to get from file content
    try:
        with open(journey_file, 'r') as f:
            data = json.load(f)
        
        # Get category from first review
        if data:
            first_key = next(iter(data.keys()))
            first_review = data[first_key]
            actual_review = first_review.get('actual_review_data', {})
            category = actual_review.get('category', '')
            
            if not category:
                return None
            
            # Map to main category using category_mapping
            main_category = category_mapping.get(category, category)
            
            # Normalize main category name to match topic_universe keys
            category_normalize = {
                'Appliances': 'Appliances',
                'All Beauty': 'All_Beauty',
                'Digital Music': 'Digital_Music',
                'Video Games': 'Video_Games',
                'Health & Personal Care': 'Health_and_Personal_Care',
                'Software': 'Software',
                'Clothing Shoes & Jewelry': 'Clothing_Shoes_and_Jewelry',
            }
            
            return category_normalize.get(main_category, main_category)
    except Exception as e:
        logger.debug(f"Error getting category from {journey_file}: {e}")
    
    return None


def map_themes_in_dict(themes: Dict[str, Any], category: str, theme_mapping: Dict[str, Dict[str, str]]) -> Dict[str, Any]:
    """Map themes in a dictionary from Set A to Set C."""
    if not themes or not isinstance(themes, dict):
        return themes
    
    if category not in theme_mapping:
        return themes
    
    mapping = theme_mapping[category]
    mapped_themes = {}
    seen_combined = set()
    
    # First pass: collect all themes that need to be mapped
    theme_parts = {}
    for theme, value in themes.items():
        if theme in mapping:
            new_theme = mapping[theme]
            if new_theme != theme:  # It's a mapping to combined theme
                if new_theme not in theme_parts:
                    theme_parts[new_theme] = []
                theme_parts[new_theme].append((theme, value))
            else:
                # Direct match, keep as is
                mapped_themes[theme] = value
        else:
            # Theme not in mapping - check if it's in Set C
            # Load Set C to check
            set_c_file = Path('topic_universe.json')
            if set_c_file.exists():
                with open(set_c_file, 'r') as f:
                    set_c = json.load(f)
                if category in set_c and theme in set_c[category]:
                    # Theme exists in Set C, keep it
                    mapped_themes[theme] = value
                else:
                    # Theme doesn't exist in Set C, skip it
                    logger.debug(f"Skipping theme '{theme}' not in Set C for category '{category}'")
    
    # Second pass: combine theme parts into combined themes
    for combined_theme, parts in theme_parts.items():
        if combined_theme not in seen_combined:
            # Combine probabilities (take max or sum - using max for now)
            combined_value = max(value for _, value in parts) if parts else 0.0
            mapped_themes[combined_theme] = combined_value
            seen_combined.add(combined_theme)
    
    return mapped_themes


def update_journey_file(journey_file: Path, theme_mapping: Dict[str, Dict[str, str]], category_mapping: Dict[str, str]) -> bool:
    """Update a single journey file with new theme names."""
    try:
        logger.info(f"Processing: {journey_file}")
        
        with open(journey_file, 'r', encoding='utf-8') as f:
            journey_data = json.load(f)
        
        # Get category (using category mapping)
        category = get_category_from_journey_file(journey_file, category_mapping)
        if not category:
            logger.warning(f"Could not determine category for {journey_file}, skipping")
            return False
        
        if category not in theme_mapping:
            logger.warning(f"No mapping found for category '{category}' in {journey_file}, skipping")
            return False
        
        updated_count = 0
        
        # Update each review
        for review_key, review_data in journey_data.items():
            # Update initial prediction
            initial_pred = review_data.get('initial_prediction_data', {})
            if initial_pred:
                initial_themes = initial_pred.get('predicted_themes', {})
                if initial_themes:
                    mapped_themes = map_themes_in_dict(initial_themes, category, theme_mapping)
                    if mapped_themes != initial_themes:
                        initial_pred['predicted_themes'] = mapped_themes
                        updated_count += 1
            
            # Update correction journey
            correction_journey = review_data.get('correction_journey', [])
            for correction in correction_journey:
                corrected_pred = correction.get('corrected_prediction_data', {})
                if corrected_pred:
                    corrected_themes = corrected_pred.get('predicted_themes', {})
                    if corrected_themes:
                        mapped_themes = map_themes_in_dict(corrected_themes, category, theme_mapping)
                        if mapped_themes != corrected_themes:
                            corrected_pred['predicted_themes'] = mapped_themes
                            updated_count += 1
        
        if updated_count > 0:
            # Save updated file
            with open(journey_file, 'w', encoding='utf-8') as f:
                json.dump(journey_data, f, indent=2, ensure_ascii=False)
            logger.info(f"  ✅ Updated {updated_count} predictions in {journey_file}")
            return True
        else:
            logger.info(f"  ⏭️  No updates needed for {journey_file}")
            return False
        
    except Exception as e:
        logger.error(f"Error processing {journey_file}: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main execution function."""
    # Load category mapping
    category_mapping = load_category_mapping()
    if not category_mapping:
        logger.warning("No category mapping loaded, will use fallback")
    
    # Load theme mapping
    theme_mapping = load_theme_mapping()
    if not theme_mapping:
        logger.error("Failed to load theme mapping")
        return 1
    
    # Find all journey files
    results_dir = Path('07_sgo_training/artifacts/sgo_training_results_v3_sample')
    journey_dir = results_dir / '_journey'
    
    if not journey_dir.exists():
        logger.error(f"Journey directory not found: {journey_dir}")
        return 1
    
    journey_files = list(journey_dir.glob('cluster_*/micro_*_journey.json'))
    
    if not journey_files:
        logger.warning("No journey files found")
        return 1
    
    logger.info(f"Found {len(journey_files)} journey files to process")
    
    # Process each file
    success_count = 0
    fail_count = 0
    
    for journey_file in sorted(journey_files):
        if update_journey_file(journey_file, theme_mapping, category_mapping):
            success_count += 1
        else:
            fail_count += 1
    
    logger.info("=" * 80)
    logger.info(f"Processing complete: {success_count} succeeded, {fail_count} failed")
    logger.info("=" * 80)
    
    return 0 if fail_count == 0 else 1


if __name__ == "__main__":
    import sys
    sys.exit(main())

