"""Configuration loader utilities."""
from typing import Dict, Any, List
from collections import defaultdict


def load_category_config(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """
    Load and parse category configuration from config dict.
    
    Args:
        cfg: Configuration dictionary from get_stage_config()
        
    Returns:
        Dictionary with:
        - categories: List of enabled category names
        - category_settings: Dict of category-specific settings
        - sampling_enabled: Boolean indicating if sampling is enabled
        - sampling_config: Dict of sampling configuration
    """
    hyperparams = cfg.get("hyperparameters", {})
    category_config = hyperparams.get("category_config", {})
    sampling_config = hyperparams.get("sampling", {})
    
    # Get enabled categories
    categories = []
    category_settings = {}
    for cat_name, cat_settings in category_config.items():
        if cat_settings.get("enabled", True):
            categories.append(cat_name)
            category_settings[cat_name] = cat_settings
    
    # Get sampling configuration - sampling is always enabled
    sampling_enabled = sampling_config.get("enabled", True)  # Default to True since sampling is always required
    
    return {
        "categories": categories,
        "category_settings": category_settings,
        "sampling_enabled": sampling_enabled,
        "sampling_config": sampling_config
    }


def print_config_summary(config_data: Dict[str, Any], output_artifact_name: str):
    """
    Print a summary of the configuration.
    
    Args:
        config_data: Output from load_category_config()
        output_artifact_name: Name of the output artifact
    """
    categories = config_data["categories"]
    sampling_enabled = config_data["sampling_enabled"]
    sampling_config = config_data["sampling_config"]
    
    print(f"\nConfiguration Summary:")
    print(f"  Output Artifact: {output_artifact_name}")
    print(f"  Sampling Enabled: {sampling_enabled}")
    
    if sampling_enabled:
        print(f"  Sampling Config:")
        print(f"    Source File: {sampling_config.get('source_file', 'N/A')}")
        print(f"    Target Users: {sampling_config.get('total_users', 'N/A')}")
        print(f"    Target Reviews: {sampling_config.get('total_reviews', 'N/A')}")
        print(f"    Maintain Ratios: {sampling_config.get('maintain_category_ratios', 'N/A')}")
        print(f"    Balanced: {sampling_config.get('balanced', 'N/A')}")
        print(f"    Use All Categories: {sampling_config.get('use_all_categories', 'N/A')}")
    
    print(f"  Enabled Categories ({len(categories)}):")
    for cat in categories:
        print(f"    - {cat}")


def calculate_category_stats(master_db: Dict[str, Any]) -> Dict[str, Dict[str, int]]:
    """
    Calculate statistics per category from the master database.
    
    Args:
        master_db: Master database dictionary with user_id as keys
        
    Returns:
        Dictionary mapping category names to stats dict with:
        - new_reviews: Number of reviews in this category
    """
    category_stats = defaultdict(lambda: {"new_reviews": 0})
    
    for user_id, user_data in master_db.items():
        for review in user_data.get('reviews', []):
            category = review.get('category', 'Unknown')
            if not category or category == 'None':
                category = 'Unknown'
            category_stats[category]["new_reviews"] += 1
    
    return dict(category_stats)

