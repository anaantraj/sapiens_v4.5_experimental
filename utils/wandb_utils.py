"""
SAPIENS W&B Utilities
=====================
Helper functions for Weights & Biases integration.

This module provides:
- Configuration loading from config.yaml
- W&B run initialization with proper config tracking
- Artifact upload/download with lineage tracking
- Real-time metric logging

Usage:
    from utils.wandb_utils import (
        load_config, get_stage_config,
        init_wandb_run, use_artifact, log_artifact,
        log_metrics, link_to_registry
    )
"""

import os
import yaml
import wandb
from pathlib import Path
from typing import Optional, Dict, Any, List, Union
from datetime import datetime

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # python-dotenv not available, skip


# =============================================================================
# Configuration Loading
# =============================================================================

def get_project_root() -> Path:
    """Get the project root directory."""
    return Path(__file__).parent.parent


def load_config(config_path: str = "config.yaml") -> dict:
    """
    Load project configuration from YAML file.
    
    Args:
        config_path: Path to config file (relative to project root)
        
    Returns:
        Configuration dictionary
    """
    config_file = get_project_root() / config_path
    
    if not config_file.exists():
        return {}
    
    with open(config_file, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f) or {}


def load_stage_config_file(stage_folder: str, config_filename: str = "config.yaml") -> dict:
    """
    Load stage-specific config file from stage folder.
    
    Args:
        stage_folder: Stage folder name (e.g., "01_dataset_pull")
        config_filename: Config filename (default: "config.yaml", can be "config_backstory.yaml", etc.)
        
    Returns:
        Stage configuration dictionary from {stage_folder}/{config_filename}
    """
    root = get_project_root()
    
    # Try to find the stage folder
    stage_config_path = root / stage_folder / config_filename
    
    if stage_config_path.exists():
        with open(stage_config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f) or {}
    
    return {}


def get_stage_config(stage: str) -> dict:
    """
    Get configuration for a specific pipeline stage.
    
    This merges:
    1. Global config (from root config.yaml)
    2. Stage-specific config (from {stage}/config.yaml)
    
    Stage config values override global config values.
    
    Args:
        stage: Stage name (e.g., "01_dataset_pull", "10_simulations")
        
    Returns:
        Merged configuration dictionary
        
    Example:
        cfg = get_stage_config("07_sgo_training")
        method = cfg['method']  # "sapiens_goal_optimization"
        hyperparams = cfg['hyperparameters']
    """
    # Load global config
    global_config = load_config()
    
    # Normalize stage name to folder format
    # Handle both "10_simulations" and "stage_10_simulations"
    stage_folder = stage.replace("stage_", "") if stage.startswith("stage_") else stage
    
    # Try to map short names to full folder names
    stage_folder_map = {
        "01_dataset_pull": "01_dataset_pull",
        "02_train_test_split": "02_training_test_data_preparation",
        "03_topic_universe": "03_topic_universe",
        "04_review_classification": "04_review_topic_classification",
        "05_user_inference": "05_user_level_inference",
        "05_5_user_embeddings": "05_5_user_embeddings",
        "06_tribe_formation": "06_tribe_formation",
        "07_sgo_training": "07_sgo_training",
        "08_model_preparation": "08_model_preparation",
        "09_baselines": "09_baselines",
        "10_simulations": "10_running_simulations",
        "11_evaluation": "11_evaluation",
        "12_analysis": "12_analysis",
    }
    
    # Get full folder name
    full_folder = stage_folder_map.get(stage_folder, stage_folder)
    
    # Load stage-specific config
    stage_config = load_stage_config_file(full_folder)
    
    # If no stage config file found, try legacy format from global config
    if not stage_config:
        # Try different key formats in global config (backward compatibility)
        stage_key = f"stage_{stage_folder}"
        stage_config = global_config.get(stage_key, {})
    
    # Merge: global + stage (stage overrides global)
    merged_config = {**global_config, **stage_config}
    
    # Also include hyperparameters at top level for convenience
    if 'hyperparameters' in stage_config:
        merged_config.update(stage_config['hyperparameters'])
    
    return merged_config


def get_wandb_config() -> dict:
    """Get W&B configuration section."""
    config = load_config()
    return config.get("wandb", {})


def get_openai_config() -> dict:
    """Get OpenAI configuration section."""
    config = load_config()
    return config.get("openai", {})


def is_wandb_enabled() -> bool:
    """Check if W&B logging is enabled in config."""
    wandb_config = get_wandb_config()
    return wandb_config.get("enabled", True)


# =============================================================================
# W&B Run Management
# =============================================================================

def init_wandb_run(
    run_name: str,
    stage: str,
    config: Optional[Dict[str, Any]] = None,
    tags: Optional[List[str]] = None,
    job_type: Optional[str] = None,
) -> Optional[wandb.sdk.wandb_run.Run]:
    """
    Initialize a W&B run for a SAPIENS pipeline stage.
    
    Automatically loads stage config from config.yaml and merges with provided config.
    
    Args:
        run_name: Descriptive name for this run
        stage: Pipeline stage (e.g., "01_dataset_pull")
        config: Additional configuration to log (overrides YAML config)
        tags: Additional tags for the run
        job_type: Type of job (e.g., "training", "evaluation")
        
    Returns:
        wandb.Run object, or None if W&B is disabled
        
    Example:
        run = init_wandb_run(
            run_name="training_split_v1",
            stage="02_train_test_split",
            config={"custom_param": 123}
        )
    """
    if not is_wandb_enabled():
        print("[W&B] Disabled in config.yaml - running locally")
        return None
    
    project_config = load_config()
    wandb_config = project_config.get("wandb", {})
    stage_config = get_stage_config(stage)
    
    # Build full config: stage config + overrides
    full_config = {
        "stage": stage,
        "project_version": project_config.get("project", {}).get("version", "v1"),
        **stage_config,  # Stage-specific config from YAML
        **(config or {})  # User overrides
    }
    
    # Build tags
    run_tags = [stage]
    if job_type:
        run_tags.append(job_type)
    if tags:
        run_tags.extend(tags)
    
    # Stage can override W&B project (e.g. use old project where artifacts exist)
    project = stage_config.get("wandb_project") or wandb_config.get("project", "SAPIENS-FINAL")
    run = wandb.init(
        entity=wandb_config.get("entity"),
        project=project,
        name=run_name,
        config=full_config,
        tags=run_tags,
        job_type=job_type,
    )
    
    print(f"[W&B] Run initialized: {run.url}")
    
    # Upload config files to W&B for visibility and reproducibility
    if run:
        root = get_project_root()
        
        # Upload global config.yaml
        global_config_path = root / "config.yaml"
        if global_config_path.exists():
            try:
                wandb.save(str(global_config_path), base_path=str(root), policy="now")
                print(f"[W&B] Uploaded global config.yaml")
            except Exception as e:
                print(f"[W&B] Warning: Could not upload global config.yaml: {e}")
        
        # Upload stage-specific config.yaml
        stage_folder_map = {
            "01_dataset_pull": "01_dataset_pull",
            "02_train_test_split": "02_training_test_data_preparation",
            "02_training_test_data_preparation": "02_training_test_data_preparation",
            "03_topic_universe": "03_topic_universe",
            "04_review_classification": "04_review_topic_classification",
            "04_review_topic_classification": "04_review_topic_classification",
            "05_user_inference": "05_user_level_inference",
            "05_user_level_inference": "05_user_level_inference",
            "05_5_user_embeddings": "05_5_user_embeddings",
            "06_tribe_formation": "06_tribe_formation",
            "07_sgo_training": "07_sgo_training",
            "08_model_preparation": "08_model_preparation",
            "09_baselines": "09_baselines",
            "10_simulations": "10_running_simulations",
            "10_running_simulations": "10_running_simulations",
            "11_evaluation": "11_evaluation",
            "12_analysis": "12_analysis",
            "Metrics and analysis": "Metrics and analysis",
        }
        
        stage_folder = stage_folder_map.get(stage, stage)
        stage_config_path = root / stage_folder / "config.yaml"
        if stage_config_path.exists():
            try:
                wandb.save(str(stage_config_path), base_path=str(root), policy="now")
                print(f"[W&B] Uploaded stage config.yaml: {stage}")
            except Exception as e:
                print(f"[W&B] Warning: Could not upload stage config.yaml: {e}")
    
    return run


def finish_run(run: Optional[wandb.sdk.wandb_run.Run]) -> None:
    """Safely finish a W&B run."""
    if run is not None:
        run.finish()
        print("[W&B] Run finished")


# =============================================================================
# Artifact Management
# =============================================================================

def use_artifact(
    run: Optional[wandb.sdk.wandb_run.Run],
    artifact_name: str,
    artifact_type: str = "dataset",
) -> Optional[Path]:
    """
    Download and use an artifact as input, tracking lineage.
    
    This marks the artifact as an INPUT to the current run,
    enabling W&B to track data lineage.
    
    Args:
        run: Active W&B run (or None if W&B disabled)
        artifact_name: Name with version (e.g., "train_set_v1:latest")
        artifact_type: Type of artifact
        
    Returns:
        Path to downloaded artifact directory, or None if failed
        
    Example:
        train_path = use_artifact(run, "train_set_v1:latest", "dataset")
        data = json.load(open(train_path / "train_set_reviews.json"))
    """
    config = load_config()
    wandb_config = config.get("wandb", {})
    entity = wandb_config.get("entity")
    project = wandb_config.get("project", "SAPIENS-FINAL")
    # Prefer run's entity/project when run exists (respects stage-level wandb_project override)
    if run is not None:
        try:
            entity = run.entity or entity
            project = run.project or project
        except Exception:
            pass
    if not entity:
        entity = "pradeepbolleddu-vectorial-ai"
    
    # Parse artifact name and version
    if ":" in artifact_name:
        name, version = artifact_name.rsplit(":", 1)
    else:
        name, version = artifact_name, "latest"
    
    artifact_path = f"{entity}/{project}/{name}:{version}"
    
    if run is not None:
        # Use artifact through run (tracks lineage)
        try:
            artifact = run.use_artifact(artifact_path, type=artifact_type)
            download_path = artifact.download()
            print(f"[W&B] Using artifact: {artifact_path}")
            print(f"      Downloaded to: {download_path}")
            # W&B may return path with colon, but filesystem uses dash - normalize it
            path_obj = Path(download_path)
            # If path doesn't exist, try replacing colon with dash (W&B filesystem conversion)
            if not path_obj.exists() and ':' in str(path_obj):
                path_obj = Path(str(path_obj).replace(':', '-'))
            return path_obj
        except Exception as e:
            print(f"[W&B] Error using artifact {artifact_path}: {e}")
            return None
    else:
        # W&B disabled - no local fallback allowed
        print(f"[W&B] Disabled - no local artifact fallback allowed for {artifact_name}")
        return None


def create_comprehensive_artifact_metadata(
    stage: str,
    artifact_name: str,
    sample_size: Optional[int] = None,
    model_vendor: Optional[str] = None,
    model_name: Optional[str] = None,
    model_description: Optional[str] = None,
    model_params: Optional[Dict[str, Any]] = None,
    prompt_prediction: Optional[str] = None,
    prompt_context: Optional[str] = None,
    learned_artifact_schema: Optional[Dict[str, Any]] = None,
    additional_metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Create comprehensive artifact metadata following the schema from the architecture diagram.
    
    This includes:
    - Method and hyperparameters
    - Model schema (vendor, description, params)
    - Prompt information (prediction, context)
    - Learned artifact schema (structure of generated data)
    - Input artifact lineage
    - Dataset information
    
    Args:
        stage: Stage name (e.g., "01_dataset_pull")
        artifact_name: Name of the artifact
        sample_size: Number of samples/records in artifact
        model_vendor: Model vendor (e.g., "OpenAI", "Anthropic")
        model_name: Model name (e.g., "gpt-4-turbo")
        model_description: Description of what the model does
        model_params: Model parameters (temperature, max_tokens, etc.)
        prompt_prediction: Prompt used for predictions
        prompt_context: Prompt context
        learned_artifact_schema: Schema/structure of the learned artifact data
            Example: {
                "structure": "dict",
                "fields": {
                    "user_id": {"type": "string", "description": "..."},
                    "embedding": {"type": "array", "dimension": 1536, "description": "..."}
                }
            }
        additional_metadata: Any additional metadata to include
        
    Returns:
        Comprehensive metadata dictionary
    """
    # Load stage config
    stage_config = get_stage_config(stage)
    global_config = load_config()
    
    # Get method and hyperparameters
    method = stage_config.get("method", "unknown")
    hyperparameters = stage_config.get("hyperparameters", {})
    
    # Get OpenAI config for model info
    openai_cfg = get_openai_config()
    
    # Build model schema
    model_schema = {}
    if model_vendor or model_name:
        model_schema = {
            "model_vendor": model_vendor or "OpenAI",
            "model_name": model_name or openai_cfg.get("analysis_model", "gpt-4-turbo"),
            "model_description": model_description or f"Model for {stage}",
            "model_params": model_params or {
                "temperature": hyperparameters.get("temperature", 0.7),
                "max_tokens": hyperparameters.get("max_tokens", 2000),
            }
        }
    
    # Build prompt schema
    prompt_schema = {}
    if prompt_prediction or prompt_context:
        prompt_schema = {
            "prompt_prediction": prompt_prediction or "",
            "prompt_context": prompt_context or "",
        }
    
    # Get input artifacts (lineage)
    input_artifacts = []
    if "input_artifact" in stage_config:
        input_artifacts.append(stage_config["input_artifact"])
    if "input_artifacts" in stage_config:
        if isinstance(stage_config["input_artifacts"], dict):
            input_artifacts.extend(stage_config["input_artifacts"].values())
        elif isinstance(stage_config["input_artifacts"], list):
            input_artifacts.extend(stage_config["input_artifacts"])
    
    # Get dataset info
    dataset_info = global_config.get("dataset", {})
    
    # Build comprehensive metadata
    metadata = {
        # Stage info
        "stage": stage,
        "artifact_name": artifact_name,
        "created_at": datetime.now().isoformat(),
        
        # Method and hyperparameters
        "method": method,
        "hyperparameters": hyperparameters,
        
        # Model schema (from image)
        "model_schema": model_schema,
        
        # Prompt schema (from image)
        "prompt_schema": prompt_schema,
        
        # Learned artifact schema (from image)
        "learned_artifact_schema": learned_artifact_schema or {},
        
        # Artifact lineage
        "input_artifacts": input_artifacts,
        "derived_from": input_artifacts,  # Alias for compatibility
        
        # Dataset information
        "dataset_name": dataset_info.get("name", "Amazon Reviews"),
        "dataset_version": dataset_info.get("version", "v1"),
        "dataset_link": dataset_info.get("link", ""),
        
        # Sample size
        "sample_size": sample_size,
        "artifact_sample_size": sample_size,  # Alias
    }
    
    # Add additional metadata
    if additional_metadata:
        metadata.update(additional_metadata)
    
    return metadata


def get_learned_artifact_schema(stage: str, artifact_name: str) -> Dict[str, Any]:
    """
    Get the learned artifact schema for a specific stage/artifact.
    
    This defines the structure of the data produced by each stage.
    
    Args:
        stage: Stage name (e.g., "05_user_level_inference")
        artifact_name: Artifact name (e.g., "user_inference_v1")
        
    Returns:
        Schema dictionary describing the artifact structure
    """
    schemas = {
        # Stage 01: Amazon Reviews
        "amazon_reviews_v1": {
            "structure": "dict",
            "key_type": "user_id (string)",
            "value_type": "dict",
            "fields": {
                "reviews": {
                    "type": "array",
                    "description": "List of user reviews",
                    "item_fields": {
                        "review_text": {"type": "string", "description": "Review content"},
                        "rating": {"type": "integer", "range": "1-5"},
                        "asin": {"type": "string", "description": "Product ID"},
                        "category": {"type": "string", "description": "Product category"},
                        "timestamp": {"type": "string", "description": "Review timestamp"}
                    }
                }
            }
        },
        
        # Stage 02: Train/Test Sets
        "train_set_v1": {
            "structure": "dict",
            "key_type": "user_id (string)",
            "value_type": "dict",
            "fields": {
                "reviews": {
                    "type": "array",
                    "description": "Training reviews with same structure as amazon_reviews_v1",
                    "item_fields": {
                        "review_text": {"type": "string"},
                        "rating": {"type": "integer", "range": "1-5"},
                        "asin": {"type": "string"},
                        "category": {"type": "string"},
                        "timestamp": {"type": "string"}
                    }
                }
            }
        },
        "test_set_v1": {
            "structure": "dict",
            "key_type": "user_id (string)",
            "value_type": "dict",
            "fields": {
                "reviews": {
                    "type": "array",
                    "description": "Test reviews (ground truth) with same structure as amazon_reviews_v1",
                    "item_fields": {
                        "review_text": {"type": "string"},
                        "rating": {"type": "integer", "range": "1-5"},
                        "asin": {"type": "string"},
                        "category": {"type": "string"},
                        "timestamp": {"type": "string"}
                    }
                }
            }
        },
        
        # Stage 03: Topic Universe
        "topic_universe_v1": {
            "structure": "dict",
            "key_type": "category_name (string)",
            "value_type": "array",
            "description": "Topics discovered per category",
            "fields": {
                "topics": {
                    "type": "array",
                    "item_type": "string",
                    "description": "List of theme/topic names for the category"
                }
            }
        },
        
        # Stage 04: Review Topics
        "review_topics_v1": {
            "structure": "jsonl",
            "description": "JSONL file with one review per line",
            "fields": {
                "user_id": {"type": "string", "description": "User identifier"},
                "review": {"type": "string", "description": "Review text"},
                "category": {"type": "string", "description": "Product category"},
                "main_category": {"type": "string", "description": "Main category"},
                "predicted_themes": {"type": "array", "item_type": "string", "description": "List of predicted themes"},
                "theme_probabilities": {"type": "dict", "key_type": "theme_name", "value_type": "float (0.0-1.0)", "description": "Probability scores for all themes"},
                "sentiment": {"type": "string", "enum": ["Positive", "Negative", "Neutral"], "description": "Predicted sentiment"},
                "rating": {"type": "integer", "range": "1-5", "description": "Review rating"}
            }
        },
        
        # Stage 05: User Inference
        "user_inference_v1": {
            "structure": "dict",
            "key_type": "user_id (string)",
            "value_type": "dict",
            "fields": {
                "llm_characteristics": {
                    "type": "dict",
                    "description": "LLM-generated user characteristics",
                    "fields": {
                        "influencing_characteristics_summary": {
                            "type": "string",
                            "description": "Summary of what influences user's reviews"
                        },
                        "backstory": {
                            "type": "dict",
                            "fields": {
                                "persona_summary": {"type": "string", "description": "2-3 sentence summary of user"},
                                "inferred_demographics": {"type": "dict", "description": "Age range, lifestyle, occupation"},
                                "key_motivations": {"type": "array", "item_type": "string", "description": "What motivates the user"},
                                "common_praises": {"type": "array", "item_type": "string", "description": "What user typically praises"},
                                "common_criticisms": {"type": "array", "item_type": "string", "description": "What user typically criticizes"},
                                "core_characteristics": {"type": "array", "item_type": "string", "description": "Core personality traits"},
                                "potential_goals": {"type": "array", "item_type": "string", "description": "User's potential goals"}
                            }
                        }
                    }
                },
                "category_characteristics": {
                    "type": "dict",
                    "key_type": "category_name (string)",
                    "value_type": "dict",
                    "description": "Category-specific user behavior",
                    "fields": {
                        "influencing_characteristics_summary": {"type": "string", "description": "How user behaves for this category"}
                    }
                }
            }
        },
        
        # Stage 05.5: User Embeddings
        "user_embeddings_v1": {
            "structure": "dict",
            "key_type": "user_id (string)",
            "value_type": "dict",
            "fields": {
                "embedding": {
                    "type": "array",
                    "item_type": "float",
                    "dimension": 1536,
                    "description": "Vector embedding of user characteristics summary",
                    "model": "text-embedding-3-small"
                }
            }
        },
        
        # Stage 06: Tribe Formation
        "tribe_formation_v1": {
            "structure": "directory",
            "description": "Directory with cluster summaries and details",
            "fields": {
                "micro_cluster_summaries": {
                    "type": "directory",
                    "structure": "cluster_X/micro_Y_summary.json",
                    "fields": {
                        "persona_name": {"type": "string", "description": "Name of the persona"},
                        "qualitative_summary": {
                            "type": "dict",
                            "fields": {
                                "persona_summary": {"type": "string", "description": "Summary of persona"},
                                "key_motivations": {"type": "array", "item_type": "string"},
                                "common_praises": {"type": "array", "item_type": "string"},
                                "common_criticisms": {"type": "array", "item_type": "string"},
                                "core_characteristics": {"type": "array", "item_type": "string"},
                                "potential_goals": {"type": "array", "item_type": "string"}
                            }
                        },
                        "total_users_in_cluster": {"type": "integer", "description": "Number of users in this micro-cluster"}
                    }
                },
                "micro_cluster_details": {
                    "type": "directory",
                    "structure": "cluster_X/micro_Y_details.json",
                    "fields": {
                        "members_grouped_by_user": {
                            "type": "dict",
                            "key_type": "user_id",
                            "value_type": "array",
                            "description": "Reviews for each user in the micro-cluster",
                            "item_fields": {
                                "review_text": {"type": "string"},
                                "rating": {"type": "integer"},
                                "asin": {"type": "string"},
                                "category": {"type": "string"}
                            }
                        }
                    }
                }
            }
        },
        
        # Stage 07: SGO Model
        "sgo_model_v1": {
            "structure": "directory",
            "description": "Refined persona models after SGO training",
            "fields": {
                "cluster_X": {
                    "type": "directory",
                    "fields": {
                        "micro_Y_results.json": {
                            "type": "dict",
                            "fields": {
                                "cluster_id": {"type": "string"},
                                "micro_id": {"type": "string"},
                                "total_failed": {"type": "integer"},
                                "total_improved": {"type": "integer"},
                                "improvement_rate": {"type": "float"},
                                "final_characteristics": {"type": "dict", "description": "Refined persona characteristics"},
                                "journey_log": {"type": "array", "description": "Training iteration history"}
                            }
                        }
                    }
                }
            }
        },
        
        # Stage 10: Simulation Results
        "simulation_results_v1": {
            "structure": "array",
            "description": "Array of simulation result objects",
            "item_type": "dict",
            "fields": {
                "user_id": {"type": "string", "description": "User identifier"},
                "num_reviews_of_user": {"type": "integer", "description": "N of reviews of user"},
                "tribe_name": {"type": "string", "description": "Micro-cluster/persona name"},
                "tribe_id": {"type": "string", "description": "Tribe id (micro cluster)"},
                "num_users_in_tribe": {"type": "integer", "description": "No of users in tribe"},
                "cluster_name": {"type": "string", "description": "Cluster/Segment name"},
                "cluster_id": {"type": "string", "description": "Cluster/Segment id"},
                "num_tribes_in_cluster": {"type": "integer", "description": "No of tribe in cluster"},
                "stimulus_product_id": {"type": "string", "description": "Stimulus - Product ID"},
                "stimulus_product_description": {"type": "string", "description": "Stimulus - Product description"},
                "real_review": {"type": "string", "description": "Real review text"},
                "real_review_embedding": {"type": "array", "item_type": "float", "dimension": 1536, "description": "Real review embedding"},
                "rating": {"type": "integer", "range": "1-5", "description": "Actual rating"},
                "real_topic_probs": {"type": "dict", "key_type": "topic_name", "value_type": "float (0.0-1.0)", "description": "JSON - Topic probabilities for real review"},
                "real_topic_embedding": {"type": "array", "item_type": "float", "description": "Weighted topic embedding for real review (for WD metric)"},
                "synthetic_review": {"type": "string", "description": "Synthetic review text"},
                "synthetic_review_embedding": {"type": "array", "item_type": "float", "dimension": 1536, "description": "Synthetic review embedding"},
                "synthetic_topic_probs": {"type": "dict", "key_type": "topic_name", "value_type": "float (0.0-1.0)", "description": "JSON - Topic probabilities for synthetic review"},
                "synthetic_topic_embedding": {"type": "array", "item_type": "float", "description": "Weighted topic embedding for synthetic review (for WD metric)"},
                "sentiment_predicted": {"type": "string", "enum": ["Positive", "Negative", "Neutral"], "description": "Predicted sentiment"},
                "rating_predicted": {"type": "integer", "range": "1-5", "description": "Predicted rating"},
                "metrics": {"type": "dict", "description": "Accuracy and recall metrics"}
            }
        }
    }
    
    # Return schema for this artifact, or empty dict if not found
    return schemas.get(artifact_name, {})


def validate_stage_dependencies(
    run: Optional[wandb.sdk.wandb_run.Run],
    stage: str,
    required_artifacts: List[str],
    artifact_type: str = "dataset"
) -> bool:
    """
    Validate that all required artifacts from previous stages exist before running a stage.
    
    This ensures sequential execution: Stage 1 must complete before Stage 2, etc.
    
    Args:
        run: W&B run object
        stage: Current stage name (e.g., "02_training_test_data_preparation")
        required_artifacts: List of artifact names that must exist (e.g., ["amazon_reviews_v1:latest"])
        artifact_type: Type of artifacts to check
        
    Returns:
        True if all artifacts exist, False otherwise
        
    Example:
        if not validate_stage_dependencies(run, "02_train_test_split", ["amazon_reviews_v1:latest"]):
            logging.error("Stage 01 must be completed first!")
            return
    """
    if not required_artifacts:
        return True
    
    print(f"\n[Validation] Checking dependencies for {stage}...")
    print(f"  Required artifacts: {', '.join(required_artifacts)}")
    
    validation = validate_artifact_dependencies(run, required_artifacts, artifact_type)
    
    all_exist = all(validation.values())
    
    if all_exist:
        print(f"[Validation] OK All dependencies satisfied")
    else:
        missing = [art for art, exists in validation.items() if not exists]
        print(f"[Validation] X Missing dependencies:")
        for art in missing:
            print(f"    - {art}")
        print(f"\n[Validation] Please run previous stages first to create required artifacts.")
    
    return all_exist


def validate_artifact_dependencies(
    run: Optional[wandb.sdk.wandb_run.Run],
    required_artifacts: List[str],
    artifact_type: str = "dataset"
) -> Dict[str, bool]:
    """
    Validate that all required artifacts exist in W&B before running a stage.
    
    Args:
        run: W&B run object (or None if W&B disabled)
        required_artifacts: List of artifact names (e.g., ["amazon_reviews_v1:latest"])
        artifact_type: Type of artifacts to check
        
    Returns:
        Dict mapping artifact_name -> exists (bool)
        
    Example:
        validation = validate_artifact_dependencies(
            run, 
            ["amazon_reviews_v1:latest", "train_set_v1:latest"]
        )
        if not all(validation.values()):
            missing = [art for art, exists in validation.items() if not exists]
            logging.error(f"Missing artifacts: {missing}")
    """
    if not run:
        print("[W&B] Disabled - skipping artifact validation")
        return {art: True for art in required_artifacts}
    
    results = {}
    config = load_config()
    wandb_config = config.get("wandb", {})
    entity = wandb_config.get("entity")
    project = wandb_config.get("project", "SAPIENS-FINAL")
    # Prefer run's entity/project when run exists (respects stage-level wandb_project override)
    if run is not None:
        try:
            entity = run.entity or entity
            project = run.project or project
        except Exception:
            pass
    if not entity:
        entity = "pradeepbolleddu-vectorial-ai"
    
    try:
        import wandb
        api = wandb.Api()
    except Exception as e:
        print(f"[W&B] Warning: Could not initialize W&B API for validation: {e}")
        return {art: False for art in required_artifacts}
    
    for artifact_name in required_artifacts:
        # Parse artifact name and version
        if ":" in artifact_name:
            name, version = artifact_name.rsplit(":", 1)
        else:
            name, version = artifact_name, "latest"
        
        artifact_path = f"{entity}/{project}/{name}:{version}"
        
        try:
            artifact = api.artifact(artifact_path, type=artifact_type)
            results[artifact_name] = True
            print(f"[W&B] OK Artifact exists: {artifact_path}")
        except Exception as e:
            results[artifact_name] = False
            print(f"[W&B] X Artifact missing: {artifact_path}")
            print(f"      Error: {e}")
    
    return results


def download_artifact(
    artifact_name: str,
    artifact_type: str = "dataset",
    version: str = "latest",
    download_root: Optional[str] = None,
) -> Optional[str]:
    """
    Download an artifact from W&B (without run tracking).
    
    Use this for one-off downloads outside of a run context.
    For tracked downloads, use use_artifact() instead.
    
    Args:
        artifact_name: Name of the artifact
        artifact_type: Type of artifact
        version: Version to download ("latest", "v1", etc.)
        download_root: Optional local directory for download
        
    Returns:
        Path to downloaded artifact, or None if failed
    """
    config = load_config()
    wandb_config = config.get("wandb", {})
    entity = wandb_config.get("entity")
    project = wandb_config.get("project", "SAPIENS-FINAL")
    
    # Fallback: use default entity if still None
    if not entity:
        entity = "pradeepbolleddu-vectorial-ai"
    
    artifact_path = f"{entity}/{project}/{artifact_name}:{version}"
    
    try:
        api = wandb.Api()
        artifact = api.artifact(artifact_path, type=artifact_type)
        return artifact.download(root=download_root)
    except Exception as e:
        print(f"[W&B] Error downloading artifact: {e}")
        return None


def log_artifact(
    run: Optional[wandb.sdk.wandb_run.Run],
    artifact_name: str,
    artifact_type: str,
    artifact_path: Union[str, Path],
    metadata: Optional[Dict[str, Any]] = None,
    aliases: Optional[List[str]] = None,
) -> Optional[wandb.Artifact]:
    """
    Log an artifact to W&B.
    
    Args:
        run: Active W&B run (or None if W&B disabled)
        artifact_name: Name for the artifact
        artifact_type: Type of artifact ("dataset", "model", "result")
        artifact_path: Local path to artifact (file or directory)
        metadata: Optional metadata dictionary
        aliases: Optional aliases (default: ["latest"])
        
    Returns:
        wandb.Artifact object, or None if W&B disabled
        
    Example:
        artifact = log_artifact(
            run=run,
            artifact_name="train_set_v1",
            artifact_type="dataset",
            artifact_path=output_dir,
            metadata={"num_users": 5000, "split": "train"}
        )
    """
    if run is None:
        print(f"[W&B] Disabled - artifact not logged: {artifact_name}")
        return None
    
    artifact = wandb.Artifact(
        name=artifact_name,
        type=artifact_type,
        metadata=metadata or {},
    )
    
    path = Path(artifact_path)
    if path.is_dir():
        artifact.add_dir(str(path))
    elif path.is_file():
        artifact.add_file(str(path))
    else:
        raise ValueError(f"Artifact path does not exist: {artifact_path}")
    
    run.log_artifact(artifact, aliases=aliases or ["latest"])
    print(f"[W&B] Logged artifact: {artifact_name} ({artifact_type})")
    
    return artifact


def get_artifact_dir(stage: str, artifact_name: str) -> Path:
    """
    Get the local artifact directory path for a stage.
    Creates the directory if it doesn't exist.
    
    Args:
        stage: Pipeline stage (e.g., "01_dataset_pull")
        artifact_name: Name of the artifact
        
    Returns:
        Path object to the artifact directory
    """
    project_root = get_project_root()
    artifact_dir = project_root / stage / "artifacts" / artifact_name
    artifact_dir.mkdir(parents=True, exist_ok=True)
    return artifact_dir


# =============================================================================
# Metrics Logging
# =============================================================================

def log_metrics(
    run: Optional[wandb.sdk.wandb_run.Run],
    metrics: Dict[str, Any],
    step: Optional[int] = None,
    commit: bool = True,
) -> None:
    """
    Log metrics to W&B during execution.
    
    Args:
        run: Active W&B run
        metrics: Dictionary of metric name -> value
        step: Optional step number (for time series)
        commit: Whether to commit immediately (default True)
        
    Example:
        # Log during training
        for epoch in range(num_epochs):
            loss = train_step()
            log_metrics(run, {"epoch": epoch, "loss": loss, "lr": lr})
        
        # Log final metrics
        log_metrics(run, {"final_accuracy": 0.95, "total_samples": 10000})
    """
    if run is None:
        return
    
    if step is not None:
        metrics["step"] = step
    
    run.log(metrics, commit=commit)


def log_summary(
    run: Optional[wandb.sdk.wandb_run.Run],
    summary: Dict[str, Any],
) -> None:
    """
    Log summary metrics (final values) to W&B.
    
    Args:
        run: Active W&B run
        summary: Dictionary of summary metric name -> value
        
    Example:
        log_summary(run, {
            "best_accuracy": 0.95,
            "total_training_time": 3600,
            "final_loss": 0.05
        })
    """
    if run is None:
        return
    
    for key, value in summary.items():
        run.summary[key] = value


# =============================================================================
# Registry Linking
# =============================================================================

def get_collection_for_stage(stage: str) -> Dict[str, str]:
    """
    Get the W&B collection info for a pipeline stage.
    
    Args:
        stage: Pipeline stage (e.g., "01_dataset_pull")
        
    Returns:
        Dict with 'collection' name and 'type'
    """
    config = load_config()
    collections = config.get("wandb", {}).get("collections", {})
    
    if stage in collections:
        return collections[stage]
    else:
        return {"collection": "Processed Data", "type": "dataset"}


def link_to_registry(
    artifact: Optional[wandb.Artifact],
    stage: Optional[str] = None,
    collection_name: Optional[str] = None,
    registry_name: str = "retail_data",
) -> bool:
    """
    Link an artifact to a W&B Registry collection.
    
    Args:
        artifact: The artifact to link (or None)
        stage: Pipeline stage - will auto-detect collection from config
        collection_name: Override collection name (optional)
        registry_name: Name of the registry
        
    Returns:
        True if linked successfully, False otherwise
    """
    if artifact is None:
        return False
    
    config = load_config()
    wandb_config = config.get("wandb", {})
    registry_org = wandb_config.get("registry_org", wandb_config.get("entity"))
    
    # Determine collection name
    if collection_name is None:
        if stage is None:
            print("[W&B] Cannot link to registry: no stage or collection specified")
            return False
        collection_info = get_collection_for_stage(stage)
        collection_name = collection_info.get("collection", "Processed Data")
    
    registry_path = f"{registry_org}/wandb-registry-{registry_name}/{collection_name}"
    
    try:
        artifact.link(registry_path)
        print(f"[W&B] Linked to registry: {registry_path}")
        return True
    except Exception as e:
        print(f"[W&B] Could not link to registry: {e}")
        return False


# =============================================================================
# Convenience Helpers
# =============================================================================

def log_dataset_artifact(
    run: Optional[wandb.sdk.wandb_run.Run],
    name: str,
    path: Union[str, Path],
    num_samples: int,
    columns: Optional[List[str]] = None,
    **extra_metadata
) -> Optional[wandb.Artifact]:
    """Helper for logging dataset artifacts with standard metadata."""
    metadata = {
        "num_samples": num_samples,
        "columns": columns or [],
        **extra_metadata
    }
    return log_artifact(run, name, "dataset", path, metadata)


def log_model_artifact(
    run: Optional[wandb.sdk.wandb_run.Run],
    name: str,
    path: Union[str, Path],
    model_type: str,
    **extra_metadata
) -> Optional[wandb.Artifact]:
    """Helper for logging model artifacts with standard metadata."""
    metadata = {
        "model_type": model_type,
        **extra_metadata
    }
    return log_artifact(run, name, "model", path, metadata)


def log_result_artifact(
    run: Optional[wandb.sdk.wandb_run.Run],
    name: str,
    path: Union[str, Path],
    metrics: Dict[str, float],
    **extra_metadata
) -> Optional[wandb.Artifact]:
    """Helper for logging result artifacts with standard metadata."""
    metadata = {
        "metrics": metrics,
        **extra_metadata
    }
    return log_artifact(run, name, "result", path, metadata)


# =============================================================================
# Context Manager
# =============================================================================

class WandBRun:
    """
    Context manager for W&B runs with automatic cleanup.
    
    Example:
        with WandBRun("my_run", "01_dataset_pull") as run:
            # Your code here
            log_metrics(run, {"accuracy": 0.95})
        # Run automatically finished
    """
    
    def __init__(
        self,
        run_name: str,
        stage: str,
        config: Optional[Dict[str, Any]] = None,
        tags: Optional[List[str]] = None,
    ):
        self.run_name = run_name
        self.stage = stage
        self.config = config
        self.tags = tags
        self.run = None
    
    def __enter__(self) -> Optional[wandb.sdk.wandb_run.Run]:
        self.run = init_wandb_run(
            self.run_name,
            self.stage,
            self.config,
            self.tags
        )
        return self.run
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        finish_run(self.run)
        return False  # Don't suppress exceptions
