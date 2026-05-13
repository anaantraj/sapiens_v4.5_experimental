"""
Behavior Loss Artifact Schema
==============================

Pydantic model for behavior loss learned artifact from SGO training.
Contains delta metrics and review-level behavior loss calculations.
"""

from pydantic import BaseModel, Field, ValidationError
from typing import Dict, List, Optional, Any
from pathlib import Path
import json


class ReviewMetrics(BaseModel):
    """Review-level metrics."""
    rating_score: Optional[float] = Field(None, ge=0.0, le=1.0)
    sentiment_score: Optional[float] = Field(None, ge=0.0, le=1.0)
    recall_at_1: Optional[float] = Field(None, ge=0.0, le=1.0, alias="recall@1")
    recall_at_3: Optional[float] = Field(None, ge=0.0, le=1.0, alias="recall@3")
    recall_at_5: Optional[float] = Field(None, ge=0.0, le=1.0, alias="recall@5")
    recall_at_max_3k: Optional[float] = Field(None, ge=0.0, le=1.0, alias="recall@max(3,k)")
    recall_at_max_3k_threshold_0_8: Optional[float] = Field(None, ge=0.0, le=1.0, alias="recall@max(3,k)_threshold_0.8")
    recall_at_max_3k_threshold_0_85: Optional[float] = Field(None, ge=0.0, le=1.0, alias="recall@max(3,k)_threshold_0.85")
    overall_accuracy: float = Field(..., ge=0.0, le=1.0)
    num_actual_themes: Optional[int] = Field(None, ge=0)

    class Config:
        populate_by_name = True


class BehaviorLossItem(BaseModel):
    """Individual review behavior loss item."""
    user_id: str = Field(..., description="Unique user identifier")
    review_id: str = Field(..., description="Unique review identifier")
    text_delta: float = Field(..., ge=0.0, le=1.0, description="Cosine distance between predicted and actual review text embeddings")
    theme_delta: float = Field(..., ge=0.0, le=1.0, description="Theme prediction error (1 - recall@max(3,k))")
    overall_behavior_loss: float = Field(..., ge=0.0, le=1.0, description="Weighted combination of text_delta and theme_delta")
    metrics: ReviewMetrics = Field(..., description="Review-level metrics")
    iteration: Optional[int] = Field(None, ge=1, description="SGO iteration number when this loss was calculated")
    improved: Optional[bool] = Field(None, description="Whether this review improved from previous iteration")

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'BehaviorLossItem':
        """Load and validate BehaviorLossItem from a dictionary."""
        try:
            return cls(**data)
        except ValidationError as e:
            raise ValueError(f"Schema validation failed for behavior loss item: {e}")

    def to_dict(self) -> Dict[str, Any]:
        """Convert the model instance to a dictionary."""
        return self.model_dump(exclude_none=True, by_alias=True)


class BehaviorLossArtifact(BaseModel):
    """
    Schema for Behavior Loss learned artifact.
    Contains delta metrics and review-level behavior loss calculations.
    Structure: {review_id: BehaviorLossItem}
    """
    # This will be a dictionary mapping review_id to BehaviorLossItem
    # We'll use a custom validator to handle this
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Dict[str, BehaviorLossItem]:
        """
        Load and validate behavior loss from a dictionary.
        
        Args:
            data: Dictionary mapping review_id to behavior loss data
            
        Returns:
            Dictionary mapping review_id to validated BehaviorLossItem instances
        """
        validated = {}
        for review_id, item_data in data.items():
            try:
                validated[review_id] = BehaviorLossItem.from_dict(item_data)
            except Exception as e:
                raise ValueError(f"Validation error for review {review_id}: {e}")
        
        return validated

    @classmethod
    def from_file(cls, file_path: Path) -> Dict[str, BehaviorLossItem]:
        """
        Load and validate behavior loss from a JSON file.
        
        Args:
            file_path: Path to the JSON file containing behavior loss data
            
        Returns:
            Dictionary mapping review_id to validated BehaviorLossItem instances
        """
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if not isinstance(data, dict):
            raise ValueError(f"Expected dictionary in {file_path}, got {type(data)}")
        
        return cls.from_dict(data)

    @staticmethod
    def to_dict(behavior_loss_dict: Dict[str, BehaviorLossItem]) -> Dict[str, Dict[str, Any]]:
        """
        Convert a dictionary of BehaviorLossItem instances to a dictionary.
        
        Args:
            behavior_loss_dict: Dictionary mapping review_id to BehaviorLossItem instances
            
        Returns:
            Dictionary mapping review_id to dictionaries
        """
        return {
            review_id: item.to_dict()
            for review_id, item in behavior_loss_dict.items()
        }

    class Config:
        json_schema_extra = {
            "example": {
                "review_001": {
                    "user_id": "U001",
                    "review_id": "review_001",
                    "text_delta": 0.15,
                    "theme_delta": 0.23,
                    "overall_behavior_loss": 0.19,
                    "metrics": {
                        "rating_score": 0.8,
                        "sentiment_score": 0.9,
                        "recall@1": 0.5,
                        "recall@3": 0.75,
                        "recall@5": 0.82,
                        "recall@max(3,k)": 0.82,
                        "overall_accuracy": 0.84,
                        "num_actual_themes": 3
                    },
                    "iteration": 1,
                    "improved": True
                }
            }
        }

