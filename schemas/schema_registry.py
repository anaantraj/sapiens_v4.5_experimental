"""
Schema Registry
===============

Central registry for all learned artifact schemas.
Maps artifact types to their Pydantic models and JSON schema files.
"""

from pathlib import Path
from typing import Dict, Type, Any, List
from pydantic import BaseModel

# Import all artifact models
from schemas.learned_artifacts.topic_universe import TopicUniverseArtifact
from schemas.learned_artifacts.review_topic_classification import ReviewTopicClassificationArtifact
from schemas.learned_artifacts.user_backstory import UserBackstoryArtifact
from schemas.learned_artifacts.user_embedding import UserEmbeddingArtifact
from schemas.learned_artifacts.user_review_history import UserReviewHistoryArtifact
from schemas.learned_artifacts.user_segments import UserSegmentsArtifact
from schemas.learned_artifacts.segment_user_details import SegmentUserDetailsArtifact
from schemas.learned_artifacts.user_tribe import UserTribeArtifact
from schemas.learned_artifacts.tribe_seed_characteristics import TribeSeedCharacteristicsArtifact
from schemas.learned_artifacts.refined_characteristics import RefinedCharacteristicsArtifact
from schemas.learned_artifacts.behavior_loss import BehaviorLossArtifact
from schemas.learned_artifacts.baseline_predictions import BaselinePredictionsArtifact
from schemas.learned_artifacts.simulation_results import SimulationResultsArtifact

SCHEMA_DIR = Path(__file__).parent
LEARNED_ARTIFACTS_DIR = SCHEMA_DIR / "learned_artifacts"

# Registry mapping artifact type names to their Pydantic models
SCHEMA_REGISTRY: Dict[str, Dict[str, Any]] = {
    "topic_universe": {
        "pydantic_model": TopicUniverseArtifact,
        "json_schema_path": LEARNED_ARTIFACTS_DIR / "topic_universe_schema.json",
        "description": "Topic universe containing discovered topics for each category",
        "stage": "03_topic_universe"
    },
    "review_topic_classification": {
        "pydantic_model": ReviewTopicClassificationArtifact,
        "json_schema_path": LEARNED_ARTIFACTS_DIR / "review_topic_classification_schema.json",
        "description": "Review topic classification with topic probabilities",
        "stage": "04_review_topic_classification"
    },
    "user_backstory": {
        "pydantic_model": UserBackstoryArtifact,
        "json_schema_path": LEARNED_ARTIFACTS_DIR / "user_backstory_schema.json",
        "description": "User backstory characteristics extracted from review history",
        "stage": "05_user_level_inference"
    },
    "user_embedding": {
        "pydantic_model": UserEmbeddingArtifact,
        "json_schema_path": LEARNED_ARTIFACTS_DIR / "user_embedding_schema.json",
        "description": "User semantic embeddings based on backstory characteristics",
        "stage": "05_user_level_inference"
    },
    "user_review_history": {
        "pydantic_model": UserReviewHistoryArtifact,
        "json_schema_path": LEARNED_ARTIFACTS_DIR / "user_review_history_schema.json",
        "description": "User review history compiled from review data",
        "stage": "05_user_level_inference"
    },
    "user_segments": {
        "pydantic_model": UserSegmentsArtifact,
        "json_schema_path": LEARNED_ARTIFACTS_DIR / "user_segments_schema.json",
        "description": "User segments (macro clusters) assignments",
        "stage": "06_tribe_formation"
    },
    "segment_user_details": {
        "pydantic_model": SegmentUserDetailsArtifact,
        "json_schema_path": LEARNED_ARTIFACTS_DIR / "segment_user_details_schema.json",
        "description": "Users with full review data grouped by segment/cluster",
        "stage": "06_tribe_formation"
    },
    "user_tribe": {
        "pydantic_model": UserTribeArtifact,
        "json_schema_path": LEARNED_ARTIFACTS_DIR / "user_tribe_schema.json",
        "description": "User tribe (micro cluster) assignments",
        "stage": "06_tribe_formation"
    },
    "tribe_seed_characteristics": {
        "pydantic_model": TribeSeedCharacteristicsArtifact,
        "json_schema_path": LEARNED_ARTIFACTS_DIR / "tribe_seed_characteristics_schema.json",
        "description": "Tribe seed characteristics (consolidated persona data)",
        "stage": "06_tribe_formation"
    },
    "refined_characteristics": {
        "pydantic_model": RefinedCharacteristicsArtifact,
        "json_schema_path": LEARNED_ARTIFACTS_DIR / "refined_characteristics_schema.json",
        "description": "Refined persona characteristics from SGO training feedback loops",
        "stage": "07_sgo_training"
    },
    "behavior_loss": {
        "pydantic_model": BehaviorLossArtifact,
        "json_schema_path": LEARNED_ARTIFACTS_DIR / "behavior_loss_schema.json",
        "description": "Behavior loss metrics (text delta, theme delta, overall loss)",
        "stage": "07_sgo_training"
    },
    "baseline_predictions": {
        "pydantic_model": BaselinePredictionsArtifact,
        "json_schema_path": LEARNED_ARTIFACTS_DIR / "baseline_predictions_schema.json",
        "description": "Baseline review predictions from history/backstory methods",
        "stage": "10_running_simulations"
    },
    "simulation_results": {
        "pydantic_model": SimulationResultsArtifact,
        "json_schema_path": LEARNED_ARTIFACTS_DIR / "simulation_results_schema.json",
        "description": "Comprehensive simulation results with extensive metadata",
        "stage": "10_running_simulations"
    },
}


def validate_artifact(artifact_type: str, data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate artifact data against its schema.
    
    Args:
        artifact_type: Type of artifact (e.g., "topic_universe")
        data: Data to validate
        
    Returns:
        Validated data as dict
        
    Raises:
        ValueError: If artifact_type is unknown
        ValidationError: If data doesn't match schema
    """
    if artifact_type not in SCHEMA_REGISTRY:
        raise ValueError(
            f"Unknown artifact type: {artifact_type}. "
            f"Available types: {list(SCHEMA_REGISTRY.keys())}"
        )
    
    model_class = SCHEMA_REGISTRY[artifact_type]["pydantic_model"]
    validated = model_class(**data)
    return validated.model_dump()


def get_schema_info(artifact_type: str) -> Dict[str, Any]:
    """Get schema information for an artifact type."""
    if artifact_type not in SCHEMA_REGISTRY:
        raise ValueError(f"Unknown artifact type: {artifact_type}")
    return SCHEMA_REGISTRY[artifact_type]


def list_available_schemas() -> List[str]:
    """List all available artifact schema types."""
    return list(SCHEMA_REGISTRY.keys())

