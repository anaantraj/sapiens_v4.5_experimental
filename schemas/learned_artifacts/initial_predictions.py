"""
Initial Predictions Learned Artifact Schema
===========================================

Pydantic model for validating initial predictions artifacts.
Contains predictions, actuals, and metrics for each review.
"""

from typing import Optional, Dict, List, Any, Union
from pydantic import BaseModel, Field
from pathlib import Path
import json


class QuantitativeSummary(BaseModel):
    """Quantitative summary statistics."""
    average_rating: Optional[Union[str, float]] = Field(None, description="Average rating")
    sentiment_distribution_percent: Dict[str, float] = Field(default_factory=dict, description="Sentiment distribution percentages")


class QualitativeSummary(BaseModel):
    """Qualitative persona summary."""
    persona_summary: Optional[str] = Field(None, description="Persona summary")
    key_motivations: List[str] = Field(default_factory=list, description="Key motivations")
    common_praises: List[str] = Field(default_factory=list, description="Common praises")
    common_criticisms: List[str] = Field(default_factory=list, description="Common criticisms")
    core_characteristics: List[str] = Field(default_factory=list, description="Core characteristics")
    potential_goals: List[str] = Field(default_factory=list, description="Potential goals")


class PredictionMetadata(BaseModel):
    """Metadata for the micro-cluster predictions."""
    persona_name: str = Field(..., description="Persona name")
    micro_cluster_id: str = Field(..., description="Micro cluster identifier")
    total_users_in_cluster: int = Field(..., description="Total users in cluster", ge=0)
    total_reviews_from_cluster: int = Field(..., description="Total reviews from cluster", ge=0)
    quantitative_summary: QuantitativeSummary = Field(..., description="Quantitative summary")
    qualitative_summary: QualitativeSummary = Field(..., description="Qualitative summary")


class PredictionOutput(BaseModel):
    """LLM prediction output for a review."""
    review_text: str = Field(..., description="Generated review text")
    rating: float = Field(..., description="Predicted rating (1-5)", ge=1.0, le=5.0)
    sentiment: str = Field(..., description="Predicted sentiment", pattern="^(Positive|Negative|Neutral)$")
    predicted_themes: Dict[str, float] = Field(..., description="Predicted themes with scores (0.0-1.0)")


class ActualReview(BaseModel):
    """Actual/ground truth review data."""
    review_text: str = Field(..., description="Actual review text")
    rating: Optional[float] = Field(None, description="Actual rating (1-5)")
    sentiment: Optional[str] = Field(None, description="Actual sentiment")
    predicted_themes: List[str] = Field(default_factory=list, description="Actual themes (list of theme names)")


class InitialPredictionsReviewMetrics(BaseModel):
    """Metrics comparing prediction vs actual for initial predictions."""
    rating_score: float = Field(..., description="Rating score (0.0-1.0)", ge=0.0, le=1.0)
    sentiment_score: float = Field(..., description="Sentiment score (0.0-1.0)", ge=0.0, le=1.0)
    recall_at_1: float = Field(..., alias="recall@1", description="Recall@1", ge=0.0, le=1.0)
    recall_at_3: float = Field(..., alias="recall@3", description="Recall@3", ge=0.0, le=1.0)
    recall_at_5: float = Field(..., alias="recall@5", description="Recall@5", ge=0.0, le=1.0)
    recall_at_k: float = Field(..., alias="recall@k", description="Recall@k", ge=0.0, le=1.0)
    recall_at_max_3k: float = Field(..., alias="recall@max(3,k)", description="Recall@max(3,k)", ge=0.0, le=1.0)
    recall_at_max_3k_threshold_08: float = Field(..., alias="recall@max(3,k)_threshold_0.8", description="Recall@max(3,k) with 0.8 threshold", ge=0.0, le=1.0)
    num_themes_above_08: int = Field(..., alias="num_themes_above_0.8", description="Number of themes above 0.8", ge=0)
    num_additional_themes_08: int = Field(..., alias="num_additional_themes_0.8", description="Number of additional themes at 0.8 threshold", ge=0)
    recall_at_max_3k_threshold_085: float = Field(..., alias="recall@max(3,k)_threshold_0.85", description="Recall@max(3,k) with 0.85 threshold", ge=0.0, le=1.0)
    num_themes_above_085: int = Field(..., alias="num_themes_above_0.85", description="Number of themes above 0.85", ge=0)
    num_additional_themes_085: int = Field(..., alias="num_additional_themes_0.85", description="Number of additional themes at 0.85 threshold", ge=0)
    overall_accuracy: float = Field(..., description="Overall accuracy score", ge=0.0, le=1.0)
    weights_used: Dict[str, float] = Field(..., description="Weights used for overall accuracy")
    num_actual_themes: int = Field(..., description="Number of actual themes", ge=0)
    
    class Config:
        populate_by_name = True  # Allow both field name and alias


class InitialPredictionsReviewMetricsLogprobs(BaseModel):
    """Metrics comparing prediction vs actual for initial predictions (logprobs mode - without num_actual_themes)."""
    rating_score: float = Field(..., description="Rating score (0.0-1.0)", ge=0.0, le=1.0)
    sentiment_score: float = Field(..., description="Sentiment score (0.0-1.0)", ge=0.0, le=1.0)
    recall_at_1: float = Field(..., alias="recall@1", description="Recall@1", ge=0.0, le=1.0)
    recall_at_3: float = Field(..., alias="recall@3", description="Recall@3", ge=0.0, le=1.0)
    recall_at_5: float = Field(..., alias="recall@5", description="Recall@5", ge=0.0, le=1.0)
    recall_at_k: float = Field(..., alias="recall@k", description="Recall@k", ge=0.0, le=1.0)
    recall_at_max_3k: float = Field(..., alias="recall@max(3,k)", description="Recall@max(3,k)", ge=0.0, le=1.0)
    recall_at_max_3k_threshold_08: float = Field(..., alias="recall@max(3,k)_threshold_0.8", description="Recall@max(3,k) with 0.8 threshold", ge=0.0, le=1.0)
    num_themes_above_08: int = Field(..., alias="num_themes_above_0.8", description="Number of themes above 0.8", ge=0)
    num_additional_themes_08: int = Field(..., alias="num_additional_themes_0.8", description="Number of additional themes at 0.8 threshold", ge=0)
    recall_at_max_3k_threshold_085: float = Field(..., alias="recall@max(3,k)_threshold_0.85", description="Recall@max(3,k) with 0.85 threshold", ge=0.0, le=1.0)
    num_themes_above_085: int = Field(..., alias="num_themes_above_0.85", description="Number of themes above 0.85", ge=0)
    num_additional_themes_085: int = Field(..., alias="num_additional_themes_0.85", description="Number of additional themes at 0.85 threshold", ge=0)
    overall_accuracy: float = Field(..., description="Overall accuracy score", ge=0.0, le=1.0)
    weights_used: Dict[str, float] = Field(..., description="Weights used for overall accuracy")
    # Note: num_actual_themes is NOT included in logprobs mode
    
    class Config:
        populate_by_name = True  # Allow both field name and alias


class ReviewPrediction(BaseModel):
    """Single review prediction with actual and metrics."""
    product_description: str = Field(..., description="Product description")
    category: Optional[str] = Field(None, description="Product category")
    asin: Optional[str] = Field(None, description="Product ASIN")
    timestamp: Optional[Union[int, str]] = Field(None, description="Review timestamp")
    prediction: PredictionOutput = Field(..., description="LLM prediction")
    actual: ActualReview = Field(..., description="Actual/ground truth data")
    metrics: InitialPredictionsReviewMetrics = Field(..., description="Metrics comparing prediction vs actual")


class ReviewPredictionLogprobs(BaseModel):
    """Single review prediction with actual and metrics (logprobs mode - without num_actual_themes)."""
    product_description: str = Field(..., description="Product description")
    category: Optional[str] = Field(None, description="Product category")
    asin: Optional[str] = Field(None, description="Product ASIN")
    timestamp: Optional[Union[int, str]] = Field(None, description="Review timestamp")
    prediction: PredictionOutput = Field(..., description="LLM prediction")
    actual: ActualReview = Field(..., description="Actual/ground truth data")
    metrics: InitialPredictionsReviewMetricsLogprobs = Field(..., description="Metrics comparing prediction vs actual")


class AggregateMetricStats(BaseModel):
    """Aggregated metric statistics."""
    mean: float = Field(..., description="Mean value")
    std: float = Field(..., description="Standard deviation", ge=0.0)
    count: int = Field(..., description="Number of samples", ge=0)


class InitialPredictionsArtifact(BaseModel):
    """
    Schema for Initial Predictions learned artifact.
    
    Contains predictions, actuals, and metrics for each review in a micro-cluster.
    
    Example:
        {
            "metadata": {
                "persona_name": "Practical DIY Fixers",
                "micro_cluster_id": "cluster_5_micro_0",
                "total_users_in_cluster": 30,
                "total_reviews_from_cluster": 182,
                "quantitative_summary": {...},
                "qualitative_summary": {...}
            },
            "model_type_used": "enhanced_persona_micro_cluster",
            "user_predictions": {
                "user_id": [
                    {
                        "product_description": "...",
                        "category": "...",
                        "prediction": {...},
                        "actual": {...},
                        "metrics": {...}
                    }
                ]
            },
            "aggregate_scores": {
                "rating_score": [1.0, 0.25, ...],
                ...
            },
            "final_metrics": {
                "rating_score": {"mean": 0.86, "std": 0.26, "count": 182},
                ...
            }
        }
    """
    
    metadata: PredictionMetadata = Field(..., description="Micro-cluster metadata")
    model_type_used: str = Field(..., description="Model type used for predictions")
    user_predictions: Dict[str, List[ReviewPrediction]] = Field(..., description="User predictions (user_id -> list of reviews)")
    aggregate_scores: Dict[str, List[float]] = Field(..., description="Raw arrays of metric values")
    final_metrics: Dict[str, AggregateMetricStats] = Field(..., description="Aggregated statistics for each metric")
    
    @classmethod
    def from_dict(cls, data: dict) -> 'InitialPredictionsArtifact':
        """Create instance from dictionary."""
        # Convert nested structures
        if 'metadata' in data and isinstance(data['metadata'], dict):
            metadata = data['metadata']
            if 'quantitative_summary' in metadata:
                metadata['quantitative_summary'] = QuantitativeSummary(**metadata['quantitative_summary'])
            if 'qualitative_summary' in metadata:
                metadata['qualitative_summary'] = QualitativeSummary(**metadata['qualitative_summary'])
            data['metadata'] = PredictionMetadata(**metadata)
        
        # Convert user_predictions
        if 'user_predictions' in data and isinstance(data['user_predictions'], dict):
            converted_predictions = {}
            for user_id, reviews in data['user_predictions'].items():
                converted_reviews = []
                for review in reviews:
                    # Create a copy to avoid modifying original
                    review_data = review.copy()
                    # Convert nested structures
                    if 'prediction' in review_data:
                        review_data['prediction'] = PredictionOutput(**review_data['prediction'])
                    if 'actual' in review_data:
                        review_data['actual'] = ActualReview(**review_data['actual'])
                    if 'metrics' in review_data:
                        # Use populate_by_name to handle aliases
                        review_data['metrics'] = InitialPredictionsReviewMetrics(**review_data['metrics'])
                    converted_reviews.append(ReviewPrediction(**review_data))
                converted_predictions[user_id] = converted_reviews
            data['user_predictions'] = converted_predictions
        
        # Convert final_metrics
        if 'final_metrics' in data and isinstance(data['final_metrics'], dict):
            converted_metrics = {}
            for metric_name, stats in data['final_metrics'].items():
                if isinstance(stats, dict):
                    converted_metrics[metric_name] = AggregateMetricStats(**stats)
                else:
                    converted_metrics[metric_name] = stats
            data['final_metrics'] = converted_metrics
        
        return cls(**data)
    
    @classmethod
    def from_file(cls, file_path: Path) -> 'InitialPredictionsArtifact':
        """
        Load and validate initial predictions from a JSON file.
        
        Args:
            file_path: Path to the JSON file
            
        Returns:
            Validated InitialPredictionsArtifact instance
        """
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        return cls.from_dict(data)
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        result = self.model_dump(exclude_none=True, by_alias=True)
        return result
    
    def to_file(self, file_path: Path):
        """Save to JSON file."""
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
    
    class Config:
        protected_namespaces = ()  # Allow "model_type_used" field


class InitialPredictionsArtifactLogprobs(BaseModel):
    """
    Schema for Initial Predictions learned artifact (logprobs mode - without num_actual_themes).
    
    Contains predictions, actuals, and metrics for each review in a micro-cluster.
    Same structure as InitialPredictionsArtifact but uses ReviewPredictionLogprobs.
    """
    
    metadata: PredictionMetadata = Field(..., description="Micro-cluster metadata")
    model_type_used: str = Field(..., description="Model type used for predictions")
    user_predictions: Dict[str, List[ReviewPredictionLogprobs]] = Field(..., description="User predictions (user_id -> list of reviews)")
    aggregate_scores: Dict[str, List[float]] = Field(..., description="Raw arrays of metric values")
    final_metrics: Dict[str, AggregateMetricStats] = Field(..., description="Aggregated statistics for each metric")
    
    @classmethod
    def from_dict(cls, data: dict) -> 'InitialPredictionsArtifactLogprobs':
        """Create instance from dictionary."""
        # Convert nested structures
        if 'metadata' in data and isinstance(data['metadata'], dict):
            metadata = data['metadata']
            if 'quantitative_summary' in metadata:
                metadata['quantitative_summary'] = QuantitativeSummary(**metadata['quantitative_summary'])
            if 'qualitative_summary' in metadata:
                metadata['qualitative_summary'] = QualitativeSummary(**metadata['qualitative_summary'])
            data['metadata'] = PredictionMetadata(**metadata)
        
        # Convert user_predictions
        if 'user_predictions' in data and isinstance(data['user_predictions'], dict):
            converted_predictions = {}
            for user_id, reviews in data['user_predictions'].items():
                converted_reviews = []
                for review in reviews:
                    # Create a copy to avoid modifying original
                    review_data = review.copy()
                    # Convert nested structures
                    if 'prediction' in review_data:
                        review_data['prediction'] = PredictionOutput(**review_data['prediction'])
                    if 'actual' in review_data:
                        review_data['actual'] = ActualReview(**review_data['actual'])
                    if 'metrics' in review_data:
                        # Use populate_by_name to handle aliases - use Logprobs metrics class
                        review_data['metrics'] = InitialPredictionsReviewMetricsLogprobs(**review_data['metrics'])
                    converted_reviews.append(ReviewPredictionLogprobs(**review_data))
                converted_predictions[user_id] = converted_reviews
            data['user_predictions'] = converted_predictions
        
        # Convert final_metrics
        if 'final_metrics' in data and isinstance(data['final_metrics'], dict):
            converted_metrics = {}
            for metric_name, stats in data['final_metrics'].items():
                if isinstance(stats, dict):
                    converted_metrics[metric_name] = AggregateMetricStats(**stats)
                else:
                    converted_metrics[metric_name] = stats
            data['final_metrics'] = converted_metrics
        
        return cls(**data)
    
    @classmethod
    def from_file(cls, file_path: Path) -> 'InitialPredictionsArtifactLogprobs':
        """
        Load and validate initial predictions from a JSON file.
        
        Args:
            file_path: Path to the JSON file
            
        Returns:
            Validated InitialPredictionsArtifactLogprobs instance
        """
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        return cls.from_dict(data)
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        result = self.model_dump(exclude_none=True, by_alias=True)
        return result
    
    def to_file(self, file_path: Path):
        """Save to JSON file."""
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
    
    class Config:
        protected_namespaces = ()  # Allow "model_type_used" field
        json_schema_extra = {
            "example": {
                "metadata": {
                    "persona_name": "Practical DIY Fixers",
                    "micro_cluster_id": "cluster_5_micro_0",
                    "total_users_in_cluster": 30,
                    "total_reviews_from_cluster": 182
                },
                "model_type_used": "enhanced_persona_micro_cluster",
                "user_predictions": {},
                "aggregate_scores": {},
                "final_metrics": {}
            }
        }

