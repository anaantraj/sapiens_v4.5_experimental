import json
import os
import re
from pathlib import Path

def load_json_file(filename, default_value=None):
    if default_value is None: default_value = {}
    if not filename or not os.path.exists(filename): return default_value
    try:
        with open(filename, 'r', encoding='utf-8') as f: return json.load(f)
    except (json.JSONDecodeError, FileNotFoundError): return default_value

def save_json_file(data, filename):
    directory = os.path.dirname(filename)
    if directory: os.makedirs(directory, exist_ok=True)
    with open(filename, 'w', encoding='utf-8') as f: 
        json.dump(data, f, indent=4)

def load_prompt(prompt_name: str, prompt_dir: Path):
    prompt_file = prompt_dir / f"{prompt_name}.txt"
    if prompt_file.exists():
        with open(prompt_file, 'r', encoding='utf-8') as f:
            return f.read()
    return None

# Removed duplicate function - using the one below with input_dir=None default
def load_category_mapping(run, artifact_name, file_pattern):
    """
    Load category mapping from W&B artifact (no local fallbacks).
    
    Args:
        run: W&B run object (required for downloading artifact)
        artifact_name: Name of the category mapping artifact (e.g., "category_mapping_to_7_main:latest")
        file_pattern: Filename pattern within the artifact (must be provided from config)
    
    Returns:
        dict: category_to_main_mapping dictionary, or None if loading fails
    """
    import logging
    from pathlib import Path
    
    # Validate required parameters
    if run is None:
        logging.error("❌ W&B run is None - cannot download category mapping from W&B. Local fallbacks are disabled.")
        return None
    
    if not artifact_name:
        logging.error("❌ category_mapping artifact_name not specified - REQUIRED")
        return None
    
    # Import W&B utilities
    try:
        import sys
        from pathlib import Path as PathLib
        # Add project root to path for imports
        project_root = PathLib(__file__).parent.parent.parent.parent
        sys.path.insert(0, str(project_root))
        from utils.wandb_utils import use_artifact
    except ImportError as e:
        logging.error(f"❌ Could not import wandb_utils: {e}")
        return None
    
    # Download artifact from W&B (no local fallback)
    logging.info(f"[INFO] Downloading category mapping from W&B: {artifact_name} (no local fallback)...")
    artifact_path = use_artifact(run, artifact_name, artifact_type="dataset")
    
    if artifact_path is None:
        logging.error(f"❌ Could not download category mapping artifact from W&B: {artifact_name} (local fallbacks disabled)")
        return None
    
    if not artifact_path.exists():
        logging.error(f"❌ Artifact path does not exist: {artifact_path}")
        return None
    
    logging.info(f"[INFO] ✅ Category mapping artifact downloaded from W&B to: {artifact_path}")
    
    # Find the category_mapping.json file in the artifact
    if artifact_path.is_file():
        file_path = artifact_path
    elif artifact_path.is_dir():
        # Check root level for exact filename match
        file_path = artifact_path / file_pattern
        if not file_path.exists() or not file_path.is_file():
            # Debug: log what files are actually in the directory
            all_files = [f for f in artifact_path.iterdir() if f.is_file()]
            if all_files:
                logging.error(f"Files in artifact directory: {[f.name for f in all_files]}")
            logging.error(f"Looking for: {file_pattern} in {artifact_path}")
            return None
    else:
        logging.error(f"❌ Artifact path is neither file nor directory: {artifact_path}")
        return None
    
    # Load the JSON file
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            mapping_data = json.load(f)
    except Exception as e:
        logging.error(f"❌ Error loading category mapping file {file_path}: {e}")
        return None
    
    if not mapping_data:
        logging.error(f"❌ Category mapping file is empty or invalid - REQUIRED")
        return None
    
    # Extract category_to_main_mapping from the full structure
    category_to_main_mapping = mapping_data.get('category_to_main_mapping', {})
    
    if not category_to_main_mapping:
        logging.error("❌ category_to_main_mapping not found in the full mapping data - REQUIRED")
        return None
    
    logging.info(f"✅ Loaded category mapping with {len(category_to_main_mapping)} categories from W&B")
    return category_to_main_mapping

def load_micro_cluster_summary(cluster_id, micro_cluster_id, data=None):
    """
    Load micro cluster summary from input data metadata or tribe seed artifact.
    Falls back to extracting from input data if available.
    """
    # First try to extract from input data metadata (preferred)
    if data and isinstance(data, dict):
        metadata = data.get('metadata', {})
        if metadata:
            qualitative_summary = metadata.get('qualitative_summary', {})
            persona_name = metadata.get('persona_name', f'persona_for_{micro_cluster_id}')
            if qualitative_summary:
                return {
                    'qualitative_summary': qualitative_summary,
                    'persona_name': persona_name
                }
    
    # Fallback: try to load from tribe seed artifact
    from sgo_training.config import settings
    MICRO_CLUSTER_SUMMARIES_DIR = settings.PATHS.get('MICRO_CLUSTER_SUMMARIES_DIR')
    if not MICRO_CLUSTER_SUMMARIES_DIR:
        # Last resort: return empty (will use defaults)
        return {
            'qualitative_summary': {},
            'persona_name': f'persona_for_{micro_cluster_id}'
        }
    tribe_seed_path = MICRO_CLUSTER_SUMMARIES_DIR
    # Look for tribe matching this cluster and micro
    micro_num = micro_cluster_id.split('_')[1] if '_' in micro_cluster_id else "0"
    cluster_num = cluster_id.split('_')[1] if '_' in cluster_id else "0"
    tribe_id = f"cluster_{cluster_num}_micro_{micro_num}"
    tribe_file = os.path.join(tribe_seed_path, f"{tribe_id}.json")
    
    if os.path.exists(tribe_file):
        tribe_data = load_json_file(tribe_file, {})
        if tribe_data:
            return {
                'qualitative_summary': tribe_data.get('qualitative_summary', {}),
                'persona_name': tribe_data.get('persona_name', f'persona_for_{micro_cluster_id}')
            }
    
    # Last resort: return empty (will use defaults)
    return {
        'qualitative_summary': {},
        'persona_name': f'persona_for_{micro_cluster_id}'
    }

def load_user_characteristics(filepath=None):
    """Load user characteristics from learned artifact (user backstory)."""
    from sgo_training.config import settings
    import logging
    
    target_file = filepath or settings.PATHS.get('USER_CHARS_FILE')
    if not target_file or not os.path.exists(target_file):
        logging.warning(f"User backstory file not found at {target_file}, returning empty dict")
        return {}    
    try:
        # Load the JSON file directly
        user_backstories_dict = load_json_file(target_file, {})
        if not user_backstories_dict:
            return {}
        
        # Convert to expected format
        user_chars_data = {}
        for user_id, backstory in user_backstories_dict.items():
            if isinstance(backstory, dict):
                user_chars_data[user_id] = {
                    "llm_characteristics": backstory.get("overall_characteristics", {}),
                    "category_characteristics": backstory.get("category_characteristics", {})
                }
        return user_chars_data
    except Exception as e:
        logging.error(f"Error loading user characteristics: {e}")
        return {}
def discover_all_micro_cluster_files(input_dir=None):
    """
    Discovers all micro cluster files in the Prediction_Accuracy_Enhanced directory.
    Returns a list of tuples: (cluster_id, micro_cluster_id, filepath)
    """
    from sgo_training.config import settings
    from sgo_training.utils import logging_utils
    
    # Use setting if argument not provided
    directory_to_search = input_dir or settings.PATHS.get('PREDICTION_INPUT_DIR')
    files_to_process = []
    
    if not os.path.exists(directory_to_search):
        logging_utils.log_thread_safe(f"Input directory not found: {directory_to_search}", 'error')
        return files_to_process
    
    # Iterate through all cluster directories
    for cluster_dir in sorted(os.listdir(directory_to_search)):
        cluster_path = os.path.join(directory_to_search, cluster_dir)
        
        # Skip if not a directory or doesn't match cluster pattern
        if not os.path.isdir(cluster_path) or not cluster_dir.startswith('cluster_'):
            continue
        
        cluster_id = cluster_dir
        
        # Find all micro cluster summary files in this cluster directory
        for filename in sorted(os.listdir(cluster_path)):
            # Look for files matching pattern: 
            # - micro_X_summary_enhanced_persona_micro_cluster_accuracy.json
            # - micro_X_summary_enhanced_delta_corrected.json
            # Skip grand_summary files or performance files
            if filename.startswith('grand_summary') or filename.startswith('performance_by_theme'):
                continue
                
            if 'micro_' in filename and (filename.endswith('_accuracy.json') or filename.endswith('_delta_corrected.json')):
                # Extract micro cluster ID
                micro_match = re.search(r'micro_(\d+)', filename)
                if micro_match:
                    micro_cluster_id = f"micro_{micro_match.group(1)}"
                    filepath = os.path.join(cluster_path, filename)
                    files_to_process.append((cluster_id, micro_cluster_id, filepath))
    
    return files_to_process