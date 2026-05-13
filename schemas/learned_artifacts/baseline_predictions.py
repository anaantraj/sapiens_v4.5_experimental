"""
Baseline Predictions Artifact Schema
====================================

Pydantic model for baseline predictions learned artifact.
Contains review predictions from baseline methods (history/backstory).
"""

from pydantic import BaseModel, Field, ValidationError
from typing import Dict, List, Optional, Any
from pathlib import Path
import json


class PredictionData(BaseModel):
    """Predicted review data."""
    review_text: str = Field(..., description="Predicted review text")
    rating: float = Field(..., ge=1.0, le=5.0, description="Predicted rating (1-5)")
    sentiment: str = Field(..., pattern="^(Positive|Negative|Neutral)$", description="Predicted sentiment")
    predicted_themes: Dict[str, float] = Field(
        ...,
        description="Predicted theme scores (theme name -> confidence score 0.0-1.0)"
    )


class ActualData(BaseModel):
    """Actual review data."""
    review_text: str = Field(..., description="Actual review text")
    rating: float = Field(..., ge=1.0, le=5.0, description="Actual rating")
    sentiment: str = Field(..., pattern="^(Positive|Negative|Neutral)$", description="Actual sentiment")
    themes: List[str] = Field(..., description="Actual themes from review")
    topic_probabilities: Optional[Dict[str, float]] = Field(None, description="Actual topic probabilities (for logprobs mode)")


class ActualDataLogprobs(BaseModel):
    """Actual review data for logprobs mode."""
    review_text: str = Field(..., description="Actual review text")
    rating: float = Field(..., ge=1.0, le=5.0, description="Actual rating")
    sentiment: str = Field(..., pattern="^(Positive|Negative|Neutral)$", description="Actual sentiment")
    topic_probabilities: Dict[str, float] = Field(..., description="Actual topic probabilities (required for logprobs mode)")
    themes: Optional[List[str]] = Field(None, description="Actual themes from review (optional)")


class ReviewMetrics(BaseModel):
    """Review prediction metrics."""
    overall_accuracy: float = Field(..., ge=0.0, le=1.0, description="Overall accuracy score")
    rating_score: float = Field(..., ge=0.0, le=1.0, description="Rating prediction accuracy")
    sentiment_score: float = Field(..., ge=0.0, le=1.0, description="Sentiment prediction accuracy")
    recall_max_3k: float = Field(..., alias="recall@max(3,k)", ge=0.0, le=1.0, description="Theme recall score")


class ReviewMetricsLogprobs(BaseModel):
    """Review prediction metrics for logprobs mode - only JSD."""
    jsd: float = Field(..., ge=0.0, description="Jensen-Shannon Divergence between predicted and actual theme distributions")


class BaselinePredictionItem(BaseModel):
    """Individual baseline prediction item."""
    user_id: str = Field(..., description="Unique user identifier")
    review_id: Optional[str] = Field(None, description="Unique review identifier")
    review_index: int = Field(..., ge=0, description="Index of review in user's review history")
    review_type: Optional[str] = Field(None, pattern="^(first|intermediate|last)$", description="Type of review based on position")
    product_description: str = Field(..., description="Product description")
    category: str = Field(..., description="Product category")
    method: str = Field(..., pattern="^(history|backstory)$", description="Prediction method used")
    model: str = Field(..., pattern="^(o3|claude)$", description="LLM model used")
    prediction: PredictionData = Field(..., description="Predicted review data")
    actual: ActualData = Field(..., description="Actual review data")
    metrics: ReviewMetrics = Field(..., description="Prediction metrics")
    cluster_name: Optional[str] = Field(None, description="Macro cluster name")
    micro_cluster_id: Optional[str] = Field(None, description="Micro cluster identifier")
    persona_name: Optional[str] = Field(None, description="Persona name")

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'BaselinePredictionItem':
        """Load and validate BaselinePredictionItem from a dictionary."""
        try:
            return cls(**data)
        except ValidationError as e:
            raise ValueError(f"Schema validation failed for baseline prediction item: {e}")

    def to_dict(self) -> Dict[str, Any]:
        """Convert the model instance to a dictionary."""
        return self.model_dump(exclude_none=True, by_alias=True)


class BaselinePredictionsArtifact(BaseModel):
    """
    Schema for Baseline Predictions learned artifact.
    Contains review predictions from baseline methods.
    Structure: {review_key: BaselinePredictionItem}
    """
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Dict[str, BaselinePredictionItem]:
        """
        Load and validate baseline predictions from a dictionary.
        
        Args:
            data: Dictionary mapping review_key to prediction data
            
        Returns:
            Dictionary mapping review_key to validated BaselinePredictionItem instances
        """
        validated = {}
        for review_key, item_data in data.items():
            try:
                validated[review_key] = BaselinePredictionItem.from_dict(item_data)
            except Exception as e:
                raise ValueError(f"Validation error for review {review_key}: {e}")
        
        return validated

    @classmethod
    def from_file(cls, file_path: Path) -> Dict[str, BaselinePredictionItem]:
        """
        Load and validate baseline predictions from a JSON file.
        
        Args:
            file_path: Path to the JSON file containing baseline predictions
            
        Returns:
            Dictionary mapping review_key to validated BaselinePredictionItem instances
        """
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if not isinstance(data, dict):
            raise ValueError(f"Expected dictionary in {file_path}, got {type(data)}")
        
        return cls.from_dict(data)

    @staticmethod
    def to_dict(predictions_dict: Dict[str, BaselinePredictionItem]) -> Dict[str, Dict[str, Any]]:
        """
        Convert a dictionary of BaselinePredictionItem instances to a dictionary.
        
        Args:
            predictions_dict: Dictionary mapping review_key to BaselinePredictionItem instances
            
        Returns:
            Dictionary mapping review_key to prediction data dictionaries
        """
        return {key: item.to_dict() for key, item in predictions_dict.items()}

    class Config:
        json_schema_extra = {
            "example": {
                "user_001_review_0": {
                    "user_id": "user_001",
                    "review_index": 0,
                    "review_type": "first",
                    "product_description": "High-quality headphones",
                    "category": "Electronics",
                    "method": "history",
                    "model": "o3",
                    "prediction": {
                        "review_text": "Great sound quality and comfortable fit.",
                        "rating": 4.5,
                        "sentiment": "Positive",
                        "predicted_themes": {
                            "Sound Quality": 0.9,
                            "Comfort": 0.8,
                            "Value": 0.6
                        }
                    },
                    "actual": {
                        "review_text": "Excellent sound quality, very comfortable.",
                        "rating": 5.0,
                        "sentiment": "Positive",
                        "themes": ["Sound Quality", "Comfort"]
                    },
                    "metrics": {
                        "overall_accuracy": 0.85,
                        "rating_score": 0.875,
                        "sentiment_score": 1.0,
                        "recall@max(3,k)": 1.0
                    }
                }
            }
        }


class BaselinePredictionItemLogprobs(BaseModel):
    """Individual baseline prediction item for logprobs mode - only JSD metric."""
    user_id: str = Field(..., description="Unique user identifier")
    review_id: Optional[str] = Field(None, description="Unique review identifier")
    review_index: int = Field(..., ge=0, description="Index of review in user's review history")
    review_type: Optional[str] = Field(None, pattern="^(first|intermediate|last)$", description="Type of review based on position")
    product_description: str = Field(..., description="Product description")
    category: str = Field(..., description="Product category")
    method: str = Field(..., pattern="^(history|backstory)$", description="Prediction method used")
    model: str = Field(..., pattern="^(o3|claude)$", description="LLM model used")
    prediction: PredictionData = Field(..., description="Predicted review data")
    actual: ActualDataLogprobs = Field(..., description="Actual review data")
    metrics: ReviewMetricsLogprobs = Field(..., description="Prediction metrics (only JSD)")
    cluster_name: Optional[str] = Field(None, description="Macro cluster name")
    micro_cluster_id: Optional[str] = Field(None, description="Micro cluster identifier")
    persona_name: Optional[str] = Field(None, description="Persona name")

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'BaselinePredictionItemLogprobs':
        """Load and validate BaselinePredictionItemLogprobs from a dictionary."""
        try:
            return cls(**data)
        except ValidationError as e:
            raise ValueError(f"Schema validation failed for baseline prediction item (logprobs): {e}")

    def to_dict(self) -> Dict[str, Any]:
        """Convert the model instance to a dictionary."""
        return self.model_dump(exclude_none=True, by_alias=True)


class BaselinePredictionsArtifactLogprobs(BaseModel):
    """
    Schema for Baseline Predictions learned artifact (logprobs mode).
    Contains review predictions from baseline methods with only JSD metric.
    Structure: {review_key: BaselinePredictionItemLogprobs}
    """
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Dict[str, BaselinePredictionItemLogprobs]:
        """
        Load and validate baseline predictions from a dictionary (logprobs mode).
        
        Args:
            data: Dictionary mapping review_key to prediction data
            
        Returns:
            Dictionary mapping review_key to validated BaselinePredictionItemLogprobs instances
        """
        validated = {}
        for review_key, item_data in data.items():
            try:
                validated[review_key] = BaselinePredictionItemLogprobs.from_dict(item_data)
            except Exception as e:
                raise ValueError(f"Validation error for review {review_key} (logprobs): {e}")
        
        return validated

    @classmethod
    def from_file(cls, file_path: Path) -> Dict[str, BaselinePredictionItemLogprobs]:
        """
        Load and validate baseline predictions from a JSON file (logprobs mode).
        
        Args:
            file_path: Path to the JSON file containing baseline predictions
            
        Returns:
            Dictionary mapping review_key to validated BaselinePredictionItemLogprobs instances
        """
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if not isinstance(data, dict):
            raise ValueError(f"Expected dictionary in {file_path}, got {type(data)}")
        
        return cls.from_dict(data)

    @staticmethod
    def to_dict(predictions_dict: Dict[str, BaselinePredictionItemLogprobs]) -> Dict[str, Dict[str, Any]]:
        """
        Convert a dictionary of BaselinePredictionItemLogprobs instances to a dictionary.
        
        Args:
            predictions_dict: Dictionary mapping review_key to BaselinePredictionItemLogprobs instances
            
        Returns:
            Dictionary mapping review_key to prediction data dictionaries
        """
        return {key: item.to_dict() for key, item in predictions_dict.items()}

    class Config:
        json_schema_extra = {
            "example": {
                "user_001_review_0": {
                    "user_id": "user_001",
                    "review_index": 0,
                    "review_type": "first",
                    "product_description": "High-quality headphones",
                    "category": "Electronics",
                    "method": "history",
                    "model": "o3",
                    "prediction": {
                        "review_text": "Great sound quality and comfortable fit.",
                        "rating": 4.5,
                        "sentiment": "Positive",
                        "predicted_themes": {
                            "Sound Quality": 0.9,
                            "Comfort": 0.8,
                            "Value": 0.6
                        }
                    },
                    "actual": {
                        "review_text": "Excellent sound quality, very comfortable.",
                        "rating": 5.0,
                        "sentiment": "Positive",
                        "topic_probabilities": {
                            "Sound Quality": 0.7,
                            "Comfort": 0.3
                        }
                    },
                    "metrics": {
                        "jsd": 0.1234
                    }
                }
            }
        }

