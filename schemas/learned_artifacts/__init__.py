"""
Learned Artifacts Schemas
=========================

Pydantic models and JSON schemas for validating learned artifacts
(intermediate outputs from pipeline stages).
"""

from schemas.learned_artifacts.topic_universe import TopicUniverseArtifact
from schemas.learned_artifacts.review_topic_classification import ReviewTopicClassificationArtifact
from schemas.learned_artifacts.user_backstory import UserBackstoryArtifact
from schemas.learned_artifacts.user_embedding import UserEmbeddingArtifact
from schemas.learned_artifacts.user_review_history import UserReviewHistoryArtifact, ReviewHistoryItem
from schemas.learned_artifacts.user_segments import UserSegmentsArtifact, ClusterMetadata
from schemas.learned_artifacts.segment_user_details import SegmentUserDetailsArtifact
from schemas.learned_artifacts.user_tribe import UserTribeArtifact, TribeMetadata
from schemas.learned_artifacts.tribe_seed_characteristics import TribeSeedCharacteristicsArtifact
from schemas.learned_artifacts.refined_characteristics import (
    RefinedCharacteristicsArtifact, RefinedCharacteristicsData, 
    RefinementMetadata, CharacteristicWithCoT, CharacteristicEvidence
)
from schemas.learned_artifacts.behavior_loss import (
    BehaviorLossArtifact, BehaviorLossItem, ReviewMetrics
)
from schemas.learned_artifacts.baseline_predictions import (
    BaselinePredictionsArtifact, BaselinePredictionItem,
    BaselinePredictionsArtifactLogprobs, BaselinePredictionItemLogprobs
)
from schemas.learned_artifacts.simulation_results import (
    SimulationResultsArtifact, SimulationResultItem, TopicProbabilityWithEmbedding
)
from schemas.learned_artifacts.initial_predictions import (
    InitialPredictionsArtifact, ReviewPrediction, PredictionOutput, 
    ActualReview, InitialPredictionsReviewMetrics, PredictionMetadata,
    InitialPredictionsArtifactLogprobs, ReviewPredictionLogprobs,
    InitialPredictionsReviewMetricsLogprobs
)

__all__ = [
    "TopicUniverseArtifact",
    "ReviewTopicClassificationArtifact",
    "UserBackstoryArtifact",
    "UserEmbeddingArtifact",
    "UserReviewHistoryArtifact",
    "ReviewHistoryItem",
    "UserSegmentsArtifact",
    "ClusterMetadata",
    "SegmentUserDetailsArtifact",
    "UserTribeArtifact",
    "TribeMetadata",
    "TribeSeedCharacteristicsArtifact",
    "RefinedCharacteristicsArtifact",
    "RefinedCharacteristicsData",
    "RefinementMetadata",
    "CharacteristicWithCoT",
    "CharacteristicEvidence",
    "BehaviorLossArtifact",
    "BehaviorLossItem",
    "ReviewMetrics",
    "BaselinePredictionsArtifact",
    "BaselinePredictionItem",
    "BaselinePredictionsArtifactLogprobs",
    "BaselinePredictionItemLogprobs",
    "SimulationResultsArtifact",
    "SimulationResultItem",
    "TopicProbabilityWithEmbedding",
    "InitialPredictionsArtifact",
    "ReviewPrediction",
    "PredictionOutput",
    "ActualReview",
    "InitialPredictionsReviewMetrics",
    "PredictionMetadata",
    "InitialPredictionsArtifactLogprobs",
    "ReviewPredictionLogprobs",
    "InitialPredictionsReviewMetricsLogprobs",
]

