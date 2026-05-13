#!/usr/bin/env python3
"""
Map Predicted Themes from Set B to Set A
========================================

This script maps predicted theme names from Set B (old/incorrect) to Set A (new/correct)
in all delta files in the sgo_train_final_predictions_refined_chars folder.

Usage:
    python Metrics and analysis/scripts/map_predicted_themes_setb_to_seta.py
"""

import json
import sys
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict
from tqdm import tqdm

# Add project root to path
BASE_DIR = Path(__file__).parent.parent.parent
sys.path.insert(0, str(BASE_DIR))

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# Theme mapping from Set B to Set A for each category
THEME_MAPPINGS = {
    "Clothing_Shoes_and_Jewelry": {
        "Fit & Sizing Accuracy": "Fit & Sizing Accuracy",
        "Comfort & Wearability": "Comfort, Support & Wearability",
        "Material & Fabric Performance": "Material Quality, Durability & Construction",
        "Durability & Construction Quality": "Material Quality, Durability & Construction",
        "Style & Appearance": "Style, Color & Aesthetic Appeal",
        "Odor or Chemical Smell": "Care, Laundering & Maintenance Requirements",
        "Functionality & Extra Features": "Functional Features & Performance",
        "Value for Money": "Value, Versatility & Purchase Experience"
    },
    "Appliances": {
        "Product accuracy and authenticity": "Value for Money & Authenticity",
        "Performance, efficiency, and noise levels": "Performance & Functionality + Sensory Experience",
        "Build quality and durability": "Product Quality & Durability",
        "Installation, fit, and compatibility": "Ease of Installation, Use & Maintenance + Compatibility & Fit with Existing Appliances",
        "Value for money": "Value for Money & Authenticity",
        "Appearance and aesthetics": "Design & Ergonomics",
        "Maintenance and cleaning ease": "Ease of Installation, Use & Maintenance",
        "Customer service and delivery experience": "Customer Service, Shipping & Packaging"
    },
    "All_Beauty": {
        "Comfort & Wearability": "Sensory & Aesthetic Experience",
        "Build Quality & Durability": "Quality, Durability & Reliability",
        "Ease of Application & User Experience": "Ease of Use, Application & Maintenance",
        "Eco-Friendliness, Ingredients & Skin Sensitivity": "Ingredient Safety & Transparency + Sustainability & Eco-Friendliness",
        "Performance & Results": "Product Performance & Effectiveness",
        "Packaging & Presentation": "Customer Support, Shipping & Service",
        "Portability & Travel-Friendliness": "Ease of Use, Application & Maintenance",
        "Price & Value": "Value for Money",
        "Safety & User Protection": "Ingredient Safety & Transparency",
        "Sensory Attributes (Scent & Texture)": "Sensory & Aesthetic Experience"
    },
    "Digital_Music": {
        "Emotional and nostalgic resonance": "Emotional, nostalgic and therapeutic impact",
        "Musical performance, authenticity, and artistry": "Songwriting, musicianship and stylistic authenticity",
        "Historical significance and rarity of releases": "Authenticity, rarity and collectability + Artist background and historical context",
        "Packaging, liner notes, and artwork quality": "Packaging, liner notes and metadata accuracy",
        "Production, recording, and overall sound quality": "Sound quality and production/mastering",
        "Track selection, completeness, and bonus content": "Track selection and album curation",
        "Overall value for money": "Authenticity, rarity and collectability"
    },
    "Video_Games": {
        "Hardware quality & battery life": "Hardware Build Quality Comfort and Connectivity + Battery Life and Portability",
        "User-friendliness of setup and controls": "Gameplay Mechanics Controls and Difficulty + Accessibility & Inclusive Design",
        "Gameplay enjoyment & engagement": "Content Depth Storyline and Appropriateness",
        "Graphics and visual presentation": "Audio and Visual Quality",
        "Technical performance & stability": "Performance Stability and Bug Issues",
        "Value for money": "Pricing Value and Monetization",
        "Shipping, packaging & product condition": "Customer Service Support and Delivery"
    },
    "Health_and_Personal_Care": {
        "Performance & Effectiveness": "Effectiveness & Precision",
        "Battery Life & Charging": "Power & Battery Performance",
        "Durability & Build Quality": "Build Quality & Durability",
        "Ease of Use & Ergonomics": "Ease of Use & Convenience",
        "Safety, Ingredients & Sensory Attributes": "Safety & Health Considerations + Comfort & Sensory Experience",
        "Accessories, Attachments & Replacement Parts": "Cleaning & Maintenance",
        "Packaging & Presentation": "Environmental & Packaging Impact",
        "Value for Money": "Value for Money"
    },
    "Software": {
        "Pricing & Value": "Customer Support, Pricing & Marketing Transparency",
        "User Interface, Usability & Customization": "Feature Set, Usability & Accessibility",
        "Performance & Resource Efficiency": "Installation & Technical Performance",
        "Reliability & Stability": "Installation & Technical Performance",
        "Features & Content Availability": "Content Availability, Licensing & Online Requirements",
        "Customer Support & Documentation": "Customer Support, Pricing & Marketing Transparency",
        "Installation, Updates & Compatibility": "Installation & Technical Performance + Hardware Integration & Performance",
        "Audio & Media Quality": "Gameplay Experience & Educational Value"
    }
}

# Category name mappings (from file category names to mapping keys)
CATEGORY_MAPPINGS = {
    "AMAZON FASHION": "Clothing_Shoes_and_Jewelry",
    "Clothing_Shoes_and_Jewelry": "Clothing_Shoes_and_Jewelry",
    "Appliances": "Appliances",
    "All_Beauty": "All_Beauty",
    "Digital_Music": "Digital_Music",
    "Video_Games": "Video_Games",
    "Health_and_Personal_Care": "Health_and_Personal_Care",
    "Software": "Software"
}


def get_category_mapping_key(category: str) -> Optional[str]:
    """
    Get the mapping key for a category name.
    
    Args:
        category: Category name from the file
        
    Returns:
        Mapping key or None if not found
    """
    # Try direct match first
    if category in CATEGORY_MAPPINGS:
        return CATEGORY_MAPPINGS[category]
    
    # Try case-insensitive match
    category_lower = category.lower().replace(" ", "_")
    for key, value in CATEGORY_MAPPINGS.items():
        if key.lower().replace(" ", "_") == category_lower:
            return value
    
    # Try partial match
    for key, value in CATEGORY_MAPPINGS.items():
        if category_lower in key.lower() or key.lower() in category_lower:
            return value
    
    return None


def map_themes_setb_to_seta(
    predicted_themes: Dict[str, float],
    category: str
) -> Dict[str, float]:
    """
    Map predicted themes from Set B to Set A.
    
    Args:
        predicted_themes: Dictionary of theme names (Set B) to probabilities
        category: Category name from the review
        
    Returns:
        Dictionary of mapped theme names (Set A) to probabilities
    """
    # Get the mapping for this category
    mapping_key = get_category_mapping_key(category)
    if not mapping_key or mapping_key not in THEME_MAPPINGS:
        # No mapping found, return as-is
        logger.debug(f"No mapping found for category: {category}")
        return predicted_themes
    
    theme_mapping = THEME_MAPPINGS[mapping_key]
    
    # Create new dictionary with mapped themes
    mapped_themes = defaultdict(float)
    unmapped_themes = {}
    
    for theme_b, prob in predicted_themes.items():
        if theme_b in theme_mapping:
            # Map to Set A theme
            theme_a = theme_mapping[theme_b]
            mapped_themes[theme_a] += prob
        else:
            # Theme not in mapping - keep as-is (might already be Set A)
            unmapped_themes[theme_b] = prob
    
    # Add unmapped themes (they might already be Set A)
    for theme, prob in unmapped_themes.items():
        mapped_themes[theme] += prob
    
    return dict(mapped_themes)


def process_json_file_with_predictions(file_path: Path) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Process a JSON file that contains predictions (either delta file or summary file).
    
    Args:
        file_path: Path to the JSON file
        
    Returns:
        Tuple of (updated_data, statistics_dict)
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        logger.error(f"Error reading file {file_path}: {e}")
        return None, None
    
    # Statistics
    stats = {
        'file_path': str(file_path),
        'total_reviews': 0,
        'mapped_reviews': 0,
        'categories_found': set(),
        'themes_mapped': 0,
        'themes_kept': 0
    }
    
    # Check if it's a delta file structure
    if 'deltas' in data:
        deltas = data.get('deltas', [])
        stats['total_reviews'] = len(deltas)
        
        for review in deltas:
            category = review.get('category', '')
            if category:
                stats['categories_found'].add(category)
            
            # Map prediction.predicted_themes
            prediction = review.get('prediction', {})
            if prediction:
                predicted_themes = prediction.get('predicted_themes', {})
                if predicted_themes:
                    original_themes = set(predicted_themes.keys())
                    mapped_themes = map_themes_setb_to_seta(predicted_themes, category)
                    mapped_themes_set = set(mapped_themes.keys())
                    
                    if original_themes != mapped_themes_set:
                        stats['mapped_reviews'] += 1
                        stats['themes_mapped'] += len(original_themes - mapped_themes_set)
                        stats['themes_kept'] += len(original_themes & mapped_themes_set)
                    
                    prediction['predicted_themes'] = mapped_themes
            
            # Map actual.topic_probabilities
            actual = review.get('actual', {})
            if actual:
                topic_probabilities = actual.get('topic_probabilities', {})
                if topic_probabilities:
                    original_topics = set(topic_probabilities.keys())
                    mapped_topics = map_themes_setb_to_seta(topic_probabilities, category)
                    mapped_topics_set = set(mapped_topics.keys())
                    
                    if original_topics != mapped_topics_set:
                        stats['mapped_reviews'] += 1
                        stats['themes_mapped'] += len(original_topics - mapped_topics_set)
                        stats['themes_kept'] += len(original_topics & mapped_topics_set)
                    
                    actual['topic_probabilities'] = mapped_topics
    
    # Check if it's a summary file structure with user_predictions
    elif 'user_predictions' in data:
        user_predictions = data.get('user_predictions', {})
        
        for user_id, reviews in user_predictions.items():
            if not isinstance(reviews, list):
                continue
            
            for review in reviews:
                stats['total_reviews'] += 1
                category = review.get('category', '')
                if category:
                    stats['categories_found'].add(category)
                
                # Map prediction.predicted_themes
                prediction = review.get('prediction', {})
                if prediction:
                    predicted_themes = prediction.get('predicted_themes', {})
                    if predicted_themes:
                        original_themes = set(predicted_themes.keys())
                        mapped_themes = map_themes_setb_to_seta(predicted_themes, category)
                        mapped_themes_set = set(mapped_themes.keys())
                        
                        if original_themes != mapped_themes_set:
                            stats['mapped_reviews'] += 1
                            stats['themes_mapped'] += len(original_themes - mapped_themes_set)
                            stats['themes_kept'] += len(original_themes & mapped_themes_set)
                        
                        prediction['predicted_themes'] = mapped_themes
                
                # Map actual.topic_probabilities
                actual = review.get('actual', {})
                if actual:
                    topic_probabilities = actual.get('topic_probabilities', {})
                    if topic_probabilities:
                        original_topics = set(topic_probabilities.keys())
                        mapped_topics = map_themes_setb_to_seta(topic_probabilities, category)
                        mapped_topics_set = set(mapped_topics.keys())
                        
                        if original_topics != mapped_topics_set:
                            stats['mapped_reviews'] += 1
                            stats['themes_mapped'] += len(original_topics - mapped_topics_set)
                            stats['themes_kept'] += len(original_topics & mapped_topics_set)
                        
                        actual['topic_probabilities'] = mapped_topics
    
    stats['categories_found'] = list(stats['categories_found'])
    
    return data, stats


def process_all_files(base_dir: Path) -> Dict[str, Any]:
    """
    Process all files with predictions in the base directory.
    
    Args:
        base_dir: Path to the base directory
        
    Returns:
        Dictionary with statistics
    """
    # Find all relevant files
    # 1. Delta files
    deltas_dir = base_dir / "deltas"
    delta_files = []
    if deltas_dir.exists():
        delta_files = list(deltas_dir.rglob("micro_*_all_reviews_deltas.json"))
    
    # 2. Summary files in cluster directories
    cluster_files = []
    for cluster_dir in base_dir.glob("cluster_*"):
        if cluster_dir.is_dir():
            # Get micro cluster summary files
            cluster_files.extend(cluster_dir.glob("micro_*_test_summary_*.json"))
            # Get grand summary files (but exclude JSD/WD summary files)
            for json_file in cluster_dir.glob("*.json"):
                if "jsd" not in json_file.name.lower() and "wd" not in json_file.name.lower() and "mapping" not in json_file.name.lower():
                    cluster_files.append(json_file)
    
    # Remove duplicates
    all_files = list(set(delta_files + cluster_files))
    
    logger.info(f"Found {len(all_files)} files to process")
    logger.info(f"  - {len(delta_files)} delta files")
    logger.info(f"  - {len(cluster_files)} cluster/summary files")
    
    # Overall statistics
    overall_stats = {
        'total_files': len(all_files),
        'total_reviews': 0,
        'mapped_reviews': 0,
        'themes_mapped': 0,
        'themes_kept': 0,
        'categories_found': set(),
        'file_stats': []
    }
    
    # Process each file
    for file_path in tqdm(all_files, desc="Processing files"):
        updated_data, stats = process_json_file_with_predictions(file_path)
        
        if updated_data is None:
            continue
        
        # Save updated file
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(updated_data, f, indent=2, ensure_ascii=False)
            logger.debug(f"Updated file: {file_path}")
        except Exception as e:
            logger.error(f"Error saving file {file_path}: {e}")
            continue
        
        # Aggregate statistics
        overall_stats['total_reviews'] += stats['total_reviews']
        overall_stats['mapped_reviews'] += stats['mapped_reviews']
        overall_stats['themes_mapped'] += stats['themes_mapped']
        overall_stats['themes_kept'] += stats['themes_kept']
        overall_stats['categories_found'].update(stats['categories_found'])
        overall_stats['file_stats'].append(stats)
    
    overall_stats['categories_found'] = sorted(list(overall_stats['categories_found']))
    
    return overall_stats


def main():
    """Main execution function."""
    logger.info("=" * 80)
    logger.info("MAPPING THEMES FROM SET B TO SET A (Predictions & Actual)")
    logger.info("=" * 80)
    
    # Set paths
    base_dir = BASE_DIR / "07_sgo_training" / "artifacts" / "sgo_train_final_predictions_refined_chars"
    
    if not base_dir.exists():
        logger.error(f"Base directory not found: {base_dir}")
        return
    
    logger.info(f"Processing files in: {base_dir}")
    
    # Process all files
    stats = process_all_files(base_dir)
    
    # Save summary statistics
    output_dir = BASE_DIR / "07_sgo_training" / "artifacts" / "sgo_train_final_predictions_refined_chars"
    summary_file = output_dir / "theme_mapping_summary.json"
    
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Saved summary statistics to: {summary_file}")
    
    # Print summary
    logger.info("\n" + "=" * 80)
    logger.info("SUMMARY STATISTICS")
    logger.info("=" * 80)
    logger.info(f"Total files processed: {stats['total_files']}")
    logger.info(f"Total reviews: {stats['total_reviews']}")
    logger.info(f"Reviews with mapped themes: {stats['mapped_reviews']}")
    logger.info(f"Total themes mapped: {stats['themes_mapped']}")
    logger.info(f"Total themes kept (already Set A): {stats['themes_kept']}")
    logger.info(f"Categories found: {', '.join(stats['categories_found'])}")
    
    logger.info("\n" + "=" * 80)
    logger.info("MAPPING COMPLETE")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()

