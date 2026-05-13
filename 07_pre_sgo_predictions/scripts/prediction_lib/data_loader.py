import logging
from pathlib import Path
from utils.wandb_utils import use_artifact
from .utils import load_json_file
from schemas.learned_artifacts import TribeSeedCharacteristicsArtifact, UserBackstoryArtifact

class DataLoader:
    def __init__(self, run, file_patterns: dict, dataset_pattern_mapping: dict):
        """
        Initialize DataLoader.
        
        Args:
            run: W&B run object
            file_patterns: Dictionary of file patterns from config (e.g., {"tribe_seed_characteristics": "tribe_seed_characteristics.json"})
            dataset_pattern_mapping: Dictionary mapping dataset_type to file_patterns key (e.g., {"train": "training_data_with_topics", "test": "test_data_with_topics"})
        """
        self.run = run
        self.file_patterns = file_patterns
        self.dataset_pattern_mapping = dataset_pattern_mapping

    def _get_artifact(self, artifact_name):
        """Download artifact from W&B only (no local fallback)."""
        if self.run is None:
            logging.error(f"❌ W&B run is None - cannot download artifact from W&B. Local fallbacks are disabled.")
            return None
        
        logging.info(f"[INFO] Downloading {artifact_name} from W&B (no local fallback)...")
        path = use_artifact(self.run, artifact_name, artifact_type="dataset")
        if path is None:
            logging.error(f"❌ Could not download artifact from W&B: {artifact_name} (local fallbacks disabled)")
            return None
        
        if not path.exists():
            logging.error(f"❌ Artifact path does not exist: {path}")
            return None
        
        logging.info(f"[INFO] ✅ Artifact downloaded from W&B to: {path}")
        return path
    
    def _find_file_in_artifact(self, artifact_path, file_pattern):
        """Find a file in an artifact directory. Only checks root level, no fallbacks, no lists."""
        if artifact_path is None:
            return None
        
        if artifact_path.is_file():
            return artifact_path
        
        if artifact_path.is_dir():
            # Only check root level - exact filename match only, no fallbacks
            file_path = artifact_path / file_pattern
            if file_path.exists() and file_path.is_file():
                logging.info(f"Found file at root: {file_path}")
                return file_path
            else:
                # Debug: log what files are actually in the directory
                all_files = [f for f in artifact_path.iterdir() if f.is_file()]
                if all_files:
                    logging.error(f"Files in artifact directory: {[f.name for f in all_files]}")
                logging.error(f"Looking for: {file_pattern} in {artifact_path}")
        
        return None

    def load_review_data(self, artifact_name, dataset_type):
        """
        Loads review data from W&B artifact.
        dataset_type: 'train' or 'test'
        """
        artifact_path = self._get_artifact(artifact_name)
        if not artifact_path:
            logging.error(f"Could not load {dataset_type} data from {artifact_name}")
            return {}
        
        # Get file pattern key from config mapping - no hardcoding
        pattern_key = self.dataset_pattern_mapping.get(dataset_type)
        if not pattern_key:
            logging.error(f"dataset_pattern_mapping not found for dataset_type '{dataset_type}' in config")
            return {}
        
        # Get file pattern from config using the mapped key
        file_pattern = self.file_patterns.get(pattern_key)
        if not file_pattern:
            logging.error(f"File pattern key '{pattern_key}' (mapped from dataset_type '{dataset_type}') not found in file_patterns config")
            return {}
        
        file_path = self._find_file_in_artifact(artifact_path, file_pattern)
        if file_path:
            return load_json_file(file_path)
        
        logging.error(f"Could not find {file_pattern} in artifact {artifact_name}")
        return {}

    def load_tribe_seeds(self, artifact_name):
        """Load tribe seed characteristics from W&B artifact."""
        artifact_path = self._get_artifact(artifact_name)
        if not artifact_path:
            logging.error(f"Could not load tribe seeds from {artifact_name}")
            return {}
        
        file_pattern = self.file_patterns.get("tribe_seed_characteristics")
        if not file_pattern:
            logging.error("tribe_seed_characteristics file pattern not found in config")
            return {}
        
        file_path = self._find_file_in_artifact(artifact_path, file_pattern)
        
        if not file_path:
            logging.error(f"Tribe seed file '{file_pattern}' not found in artifact {artifact_name}")
            return {}
        
        logging.info(f"Found tribe seed file: {file_path}")
        return TribeSeedCharacteristicsArtifact.from_file(file_path)

    def load_user_backstories(self, artifact_name):
        """Load user backstories from W&B artifact."""
        artifact_path = self._get_artifact(artifact_name)
        if not artifact_path:
            logging.error(f"Could not load user backstories from {artifact_name}")
            return {}
        
        file_pattern = self.file_patterns.get("user_backstories")
        if not file_pattern:
            logging.error("user_backstories file pattern not found in config")
            return {}
        
        file_path = self._find_file_in_artifact(artifact_path, file_pattern)
        
        if not file_path:
            logging.error(f"User backstory file '{file_pattern}' not found in artifact {artifact_name}")
            return {}
        
        return UserBackstoryArtifact.from_file(file_path)

    def load_topic_universe(self, artifact_name):
        """Load topic universe from W&B artifact."""
        artifact_path = self._get_artifact(artifact_name)
        if not artifact_path:
            logging.error(f"Could not load topic universe from {artifact_name}")
            return {}
        
        file_pattern = self.file_patterns.get("topic_universe")
        if not file_pattern:
            logging.error("topic_universe file pattern not found in config")
            return {}
        file_path = self._find_file_in_artifact(artifact_path, file_pattern)
        
        if not file_path:
            logging.error(f"Topic universe file not found in artifact {artifact_name}")
            return {}
        
        return load_json_file(file_path)

    def load_user_tribes(self, artifact_name):
        """Load user tribes mapping from W&B artifact."""
        artifact_path = self._get_artifact(artifact_name)
        if not artifact_path:
            logging.error(f"Could not load user tribes from {artifact_name}")
            return {}
        
        file_pattern = self.file_patterns.get("user_tribes")
        if not file_pattern:
            logging.error("user_tribes file pattern not found in config")
            return {}
        
        file_path = self._find_file_in_artifact(artifact_path, file_pattern)
        
        if not file_path:
            logging.error(f"User tribes file '{file_pattern}' not found in artifact {artifact_name}")
            return {}
        
        data = load_json_file(file_path)
        if not data:
            return {}
        
        return {uid: d.get('tribe_id') for uid, d in data.items() if d.get('tribe_id')}
    
    def load_category_mapping(self, artifact_name):
        """Load category mapping from W&B artifact. Fails if not available (required)."""
        if not artifact_name:
            logging.error("❌ category_mapping artifact not specified in config - REQUIRED")
            return None
        
        artifact_path = self._get_artifact(artifact_name)
        if not artifact_path:
            logging.error(f"❌ Could not download category mapping artifact: {artifact_name} - REQUIRED")
            return None
        
        logging.info(f"Category mapping artifact path: {artifact_path}")
        logging.info(f"Artifact path exists: {artifact_path.exists()}")
        logging.info(f"Artifact path is_dir: {artifact_path.is_dir()}")
        
        file_pattern = self.file_patterns.get("category_mapping")
        if not file_pattern:
            logging.error("❌ category_mapping file pattern not found in config - REQUIRED")
            return None
        
        file_path = self._find_file_in_artifact(artifact_path, file_pattern)
        
        if not file_path:
            logging.error(f"❌ Category mapping file '{file_pattern}' not found in artifact {artifact_name} - REQUIRED")
            logging.error(f"   Artifact path: {artifact_path}")
            logging.error(f"   Looking for: {artifact_path / file_pattern}")
            return None
        
        mapping_data = load_json_file(file_path)
        if not mapping_data:
            logging.error(f"❌ Category mapping file is empty or invalid - REQUIRED")
            return None
        
        logging.info(f"✅ Loaded category mapping with {len(mapping_data)} categories")
        return mapping_data