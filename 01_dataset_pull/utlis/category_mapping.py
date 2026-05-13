"""Category mapping utilities."""
import json
from pathlib import Path
from typing import Optional

# Map JSON category names to code category names
JSON_TO_CODE_MAPPING = {
    "Fashion": "Clothing_Shoes_and_Jewelry",
    "All Beauty": "All_Beauty",
    "Digital Music": "Digital_Music",
    "Video Games": "Video_Games",
    "Health & Personal Care": "Health_and_Personal_Care",
    "Appliances": "Appliances",
    "Software": "Software"
}

# Reverse mapping: code category names to JSON category names
CODE_TO_JSON_MAPPING = {v: k for k, v in JSON_TO_CODE_MAPPING.items()}

# Load category mapping from JSON file
def _load_category_mapping():
    """Load category mapping from category_mapping_to_7_main.json."""
    project_root = Path(__file__).parent.parent.parent
    mapping_file = project_root / "category_mapping_to_7_main.json"
    
    if not mapping_file.exists():
        # Fallback: return empty dict if file doesn't exist
        return {}
    
    try:
        with open(mapping_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            return data.get("category_to_main_mapping", {})
    except Exception as e:
        print(f"[WARNING] Failed to load category mapping from {mapping_file}: {e}")
        return {}

# Load the mapping once at module level
_CATEGORY_TO_MAIN_MAPPING = _load_category_mapping()

# Create a lookup dict with code category names
CATEGORY_MAPPING = {}
for json_cat, json_main in _CATEGORY_TO_MAIN_MAPPING.items():
    # Convert JSON main category name to code category name
    code_main = JSON_TO_CODE_MAPPING.get(json_main, json_main)
    CATEGORY_MAPPING[json_cat] = code_main


def map_category_to_main_category(category_name: str) -> Optional[str]:
    """
    Map category name to one of 7 main categories using category_mapping_to_7_main.json.
    
    Args:
        category_name: Category name from review data
        
    Returns:
        Main category name (code format) or None if no match
    """
    if not category_name:
        return None
    
    # Try exact match first
    if category_name in CATEGORY_MAPPING:
        return CATEGORY_MAPPING[category_name]
    
    # Try case-insensitive match
    for cat, main_cat in CATEGORY_MAPPING.items():
        if cat.lower() == category_name.lower():
            return main_cat
    
    # Try matching against JSON mapping directly
    if category_name in _CATEGORY_TO_MAIN_MAPPING:
        json_main = _CATEGORY_TO_MAIN_MAPPING[category_name]
        return JSON_TO_CODE_MAPPING.get(json_main, json_main)
    
    # Try case-insensitive match against JSON mapping
    for cat, json_main in _CATEGORY_TO_MAIN_MAPPING.items():
        if cat.lower() == category_name.lower():
            return JSON_TO_CODE_MAPPING.get(json_main, json_main)
    
    return None

