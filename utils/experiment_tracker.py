"""
Experiment Tracker
==================

Tracks complete experiment metadata for W&B logging:
- Method + hyperparameters for each module
- Artifact lineage (which artifacts were used)
- Dataset information (name, version, sample size, link)

Usage:
    from utils.experiment_tracker import ExperimentTracker
    
    tracker = ExperimentTracker(run)
    tracker.log_experiment_metadata()
"""

import yaml
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime


def get_project_root() -> Path:
    """Get the project root directory."""
    return Path(__file__).parent.parent


def load_global_config() -> Dict[str, Any]:
    """Load the global config.yaml from project root."""
    config_path = get_project_root() / "config.yaml"
    if config_path.exists():
        with open(config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f) or {}
    return {}


def load_stage_config(stage_name: str) -> Dict[str, Any]:
    """
    Load stage-specific config.yaml.
    
    Args:
        stage_name: Stage folder name (e.g., "01_dataset_pull")
        
    Returns:
        Merged config (global + stage-specific)
    """
    root = get_project_root()
    
    # Load global config
    global_config = load_global_config()
    
    # Load stage config
    stage_config_path = root / stage_name / "config.yaml"
    stage_config = {}
    if stage_config_path.exists():
        with open(stage_config_path, 'r', encoding='utf-8') as f:
            stage_config = yaml.safe_load(f) or {}
    
    # Merge: stage overrides global
    merged = {**global_config, **stage_config}
    return merged


class ExperimentTracker:
    """
    Tracks experiment metadata for W&B logging.
    
    Captures:
    1. Method + hyperparameters for each module used
    2. List of artifacts from each module (lineage)
    3. Dataset name, version, sample size, link
    """
    
    # All stage folder names
    STAGES = [
        "01_dataset_pull",
        "02_training_test_data_preparation",
        "03_topic_universe",
        "04_review_topic_classification",
        "05_user_level_inference",
        "06_tribe_formation",
        "07_sgo_training",
        "08_model_preparation",
        "09_baselines",
        "10_running_simulations",
        "11_evaluation",
        "12_analysis",
    ]
    
    def __init__(self, run=None):
        """
        Initialize experiment tracker.
        
        Args:
            run: W&B run object (optional)
        """
        self.run = run
        self.global_config = load_global_config()
        self.stage_configs = {}
        self._load_all_stage_configs()
    
    def _load_all_stage_configs(self):
        """Load configs from all stages."""
        for stage in self.STAGES:
            self.stage_configs[stage] = load_stage_config(stage)
    
    def get_pipeline_config(self) -> Dict[str, Any]:
        """
        Get method + hyperparameters for all stages.
        
        Returns:
            Dict with stage -> {method, hyperparameters}
        """
        pipeline_config = {}
        
        for stage in self.STAGES:
            cfg = self.stage_configs.get(stage, {})
            pipeline_config[stage] = {
                "method": cfg.get("method", "unknown"),
                "hyperparameters": cfg.get("hyperparameters", {}),
            }
        
        return pipeline_config
    
    def get_artifact_lineage(self) -> Dict[str, Any]:
        """
        Get artifact lineage - which artifacts each stage uses/produces.
        
        Returns:
            Dict with stage -> {input_artifacts, output_artifact}
        """
        lineage = {}
        
        for stage in self.STAGES:
            cfg = self.stage_configs.get(stage, {})
            
            # Handle different config structures
            input_artifacts = []
            if "input_artifact" in cfg and cfg["input_artifact"]:
                input_artifacts.append(cfg["input_artifact"])
            if "input_artifacts" in cfg:
                if isinstance(cfg["input_artifacts"], dict):
                    input_artifacts.extend(cfg["input_artifacts"].values())
                elif isinstance(cfg["input_artifacts"], list):
                    input_artifacts.extend(cfg["input_artifacts"])
            
            output_artifacts = []
            if "output_artifact" in cfg and cfg["output_artifact"]:
                output_artifacts.append(cfg["output_artifact"])
            if "output_artifacts" in cfg:
                if isinstance(cfg["output_artifacts"], dict):
                    output_artifacts.extend(cfg["output_artifacts"].values())
                elif isinstance(cfg["output_artifacts"], list):
                    output_artifacts.extend(cfg["output_artifacts"])
            
            lineage[stage] = {
                "input_artifacts": input_artifacts,
                "output_artifacts": output_artifacts,
            }
        
        return lineage
    
    def get_dataset_info(self) -> Dict[str, Any]:
        """
        Get dataset information for experiment tracking.
        
        Returns:
            Dict with name, version, sample_size, link
        """
        dataset_config = self.global_config.get("dataset", {})
        wandb_config = self.global_config.get("wandb", {})
        
        return {
            "name": dataset_config.get("name", "Unknown Dataset"),
            "version": dataset_config.get("version", "v1"),
            "source": dataset_config.get("source", ""),
            "sample_size": dataset_config.get("sample_size", {}),
            "link": dataset_config.get(
                "link", 
                f"wandb://{wandb_config.get('entity', '')}/{wandb_config.get('project', '')}/amazon_reviews_v1"
            ),
        }
    
    def get_full_experiment_metadata(self) -> Dict[str, Any]:
        """
        Get complete experiment metadata for W&B logging.
        
        Returns:
            Dict with all experiment metadata
        """
        return {
            "experiment_timestamp": datetime.now().isoformat(),
            "project": self.global_config.get("project", {}),
            "pipeline_config": self.get_pipeline_config(),
            "artifact_lineage": self.get_artifact_lineage(),
            "dataset": self.get_dataset_info(),
            "wandb": {
                "entity": self.global_config.get("wandb", {}).get("entity"),
                "project": self.global_config.get("wandb", {}).get("project"),
            },
        }
    
    def log_experiment_metadata(self) -> Dict[str, Any]:
        """
        Log complete experiment metadata to W&B.
        
        Returns:
            The metadata dict that was logged
        """
        metadata = self.get_full_experiment_metadata()
        
        if self.run is not None:
            # Log as W&B config
            self.run.config.update({"experiment_metadata": metadata})
            print("[ExperimentTracker] Logged experiment metadata to W&B config")
        else:
            print("[ExperimentTracker] No W&B run - metadata not logged")
        
        return metadata
    
    def create_artifact_metadata(
        self,
        stage: str,
        additional_metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Create standard artifact metadata for a stage.
        
        Args:
            stage: Stage name (e.g., "01_dataset_pull")
            additional_metadata: Extra metadata to include
            
        Returns:
            Metadata dict for artifact logging
        """
        cfg = self.stage_configs.get(stage, {})
        dataset_info = self.get_dataset_info()
        lineage = self.get_artifact_lineage().get(stage, {})
        
        metadata = {
            # Method tracking
            "method": cfg.get("method", "unknown"),
            "hyperparameters": cfg.get("hyperparameters", {}),
            
            # Lineage tracking
            "derived_from": lineage.get("input_artifacts", []),
            "stage": stage,
            
            # Dataset info (all 6 required fields)
            "dataset_name": dataset_info.get("name"),
            "dataset_version": dataset_info.get("version"),
            "dataset_sample_size": dataset_info.get("sample_size", {}),  # Full dataset size
            "dataset_link": dataset_info.get("link"),
            
            # Timestamp
            "created_at": datetime.now().isoformat(),
        }
        
        # Merge additional metadata
        if additional_metadata:
            metadata.update(additional_metadata)
        
        return metadata


# =============================================================================
# Convenience Functions
# =============================================================================

def get_stage_method(stage: str) -> str:
    """Get the method name for a stage."""
    cfg = load_stage_config(stage)
    return cfg.get("method", "unknown")


def get_stage_hyperparameters(stage: str) -> Dict[str, Any]:
    """Get hyperparameters for a stage."""
    cfg = load_stage_config(stage)
    return cfg.get("hyperparameters", {})


def get_stage_artifacts(stage: str) -> Dict[str, Any]:
    """Get input/output artifacts for a stage."""
    cfg = load_stage_config(stage)
    
    input_artifacts = []
    if "input_artifact" in cfg and cfg["input_artifact"]:
        input_artifacts.append(cfg["input_artifact"])
    if "input_artifacts" in cfg:
        if isinstance(cfg["input_artifacts"], dict):
            input_artifacts.extend(cfg["input_artifacts"].values())
    
    output_artifact = cfg.get("output_artifact")
    if not output_artifact and "output_artifacts" in cfg:
        output_artifact = cfg["output_artifacts"]
    
    return {
        "input": input_artifacts,
        "output": output_artifact,
    }


def create_artifact_metadata_for_stage(
    stage: str,
    sample_size: Optional[int] = None,
    **extra_metadata
) -> Dict[str, Any]:
    """
    Create artifact metadata for a specific stage.
    
    Args:
        stage: Stage name
        sample_size: Number of samples in artifact
        **extra_metadata: Additional metadata fields
        
    Returns:
        Metadata dict ready for W&B artifact logging
    """
    tracker = ExperimentTracker()
    
    metadata = tracker.create_artifact_metadata(stage)
    
    # Add artifact-specific sample size (number of records in this artifact)
    if sample_size is not None:
        metadata["artifact_sample_size"] = sample_size
    
    metadata.update(extra_metadata)
    
    return metadata


# =============================================================================
# Example Usage
# =============================================================================

if __name__ == "__main__":
    # Demo: Print experiment metadata
    tracker = ExperimentTracker()
    
    print("=" * 70)
    print("EXPERIMENT METADATA")
    print("=" * 70)
    
    metadata = tracker.get_full_experiment_metadata()
    
    print("\n[Dataset Info]")
    print(f"  Name: {metadata['dataset']['name']}")
    print(f"  Version: {metadata['dataset']['version']}")
    print(f"  Sample Size: {metadata['dataset']['sample_size']}")
    
    print("\n[Pipeline Config]")
    for stage, cfg in metadata['pipeline_config'].items():
        print(f"  {stage}:")
        print(f"    Method: {cfg['method']}")
        if cfg['hyperparameters']:
            print(f"    Hyperparameters: {list(cfg['hyperparameters'].keys())}")
    
    print("\n[Artifact Lineage]")
    for stage, lineage in metadata['artifact_lineage'].items():
        if lineage['input_artifacts'] or lineage['output_artifacts']:
            print(f"  {stage}:")
            print(f"    Inputs: {lineage['input_artifacts']}")
            print(f"    Outputs: {lineage['output_artifacts']}")
