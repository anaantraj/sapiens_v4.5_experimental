import json
import logging
import numpy as np
from pathlib import Path
from typing import Optional

def load_json_file(filepath: Path) -> Optional[dict]:
    """Safely loads a JSON file."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        logging.error(f"File not found: {filepath}")
        return None
    except json.JSONDecodeError as e:
        logging.error(f"Error decoding JSON from {filepath}: {e}")
        return None

def convert_to_serializable(obj):
    """Recursively converts numpy types and Pydantic models to native Python types."""
    # Handle Pydantic models
    if hasattr(obj, 'model_dump'):
        return convert_to_serializable(obj.model_dump())
    if hasattr(obj, 'dict'):  # Fallback for older Pydantic versions
        return convert_to_serializable(obj.dict())
    
    # Handle numpy types
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    
    # Handle dicts and lists recursively
    if isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [convert_to_serializable(item) for item in obj]
    
    return obj