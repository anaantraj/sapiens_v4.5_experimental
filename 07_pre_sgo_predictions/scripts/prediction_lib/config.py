import logging
from typing import Optional

def map_category_to_main_category(category_name: str, mapping: dict) -> Optional[str]:
    """Map a category name to one of the 7 main categories."""
    if not category_name:
        return None
    
    # Exact match
    if category_name in mapping:
        return mapping[category_name]
    
    # Case-insensitive
    for cat, main_cat in mapping.items():
        if cat.lower() == category_name.lower():
            return main_cat
    
    return None