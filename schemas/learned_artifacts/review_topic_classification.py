"""
Review Topic Classification Learned Artifact Schema
====================================================

Pydantic model for validating review topic classification artifacts.
Contains topic probabilities for each review.
"""

from typing import Dict, List, Optional, Union
from pydantic import BaseModel, Field, field_validator
from pathlib import Path
import json


class ReviewTopicClassificationArtifact(BaseModel):
    """
    Schema for Review Topic Classification learned artifact.
    
    Represents a single review with its identified themes and token probabilities.
    
    Example:
        {
            "review_id": "U001_P123",
            "user_id": "U001",
            "review": "Great product, very comfortable...",
            "category": "Clothing_Shoes_and_Jewelry",
            "identified_themes": ["Quality & Durability", "Comfort & Breathability"],
            "theme_token_probabilities": {
                "Quality & Durability": 0.6,
                "Comfort & Breathability": 0.4
            },
            "sentiment": "Positive",
            "rating": 5,
            "asin": "P123",
            "timestamp": "2024-01-15"
        }
    """
    
    review_id: str = Field(..., description="Unique identifier for the review")
    user_id: str = Field(..., description="User ID who wrote the review")
    review: str = Field(..., description="The review text", min_length=1)
    category: str = Field(..., description="Product category", min_length=1)
    identified_themes: List[str] = Field(
        default_factory=list,
        description="List of themes identified by the LLM (from identified_themes field in response)"
    )
    theme_token_probabilities: Dict[str, float] = Field(
        default_factory=dict,
        description="Dictionary mapping each identified theme to its token probability"
    )
    sentiment: str = Field(..., description="Sentiment classification")
    rating: Optional[float] = Field(None, description="Product rating (if available)")
    asin: Optional[str] = Field(None, description="Product ASIN (if available)")
    timestamp: Optional[str] = Field(None, description="Review timestamp (if available)")
    error: Optional[str] = Field(None, description="Error message if processing failed")
    
    @field_validator('timestamp', mode='before')
    @classmethod
    def convert_timestamp(cls, v: Union[int, str, None]) -> Optional[str]:
        """Convert integer timestamp to string if needed."""
        if v is None:
            return None
        if isinstance(v, int):
            return str(v)
        return v
    
    @field_validator('sentiment')
    @classmethod
    def validate_sentiment(cls, v: str) -> str:
        """Validate sentiment is one of the allowed values."""
        allowed = ["Positive", "Negative", "Neutral", "Mixed"]
        if v not in allowed:
            raise ValueError(f"Sentiment must be one of {allowed}, got: {v}")
        return v
    
    @field_validator('theme_token_probabilities')
    @classmethod
    def validate_theme_token_probabilities(cls, v: Dict[str, float]) -> Dict[str, float]:
        """Validate token probabilities are between 0.0 and 1.0."""
        if not v:
            return v
        for theme, prob in v.items():
            if not isinstance(prob, (int, float)):
                raise ValueError(f"Probability for theme '{theme}' must be a number, got: {type(prob)}")
            if prob < 0.0 or prob > 1.0:
                raise ValueError(f"Probability for theme '{theme}' must be between 0.0 and 1.0, got: {prob}")
        return v
    
    @classmethod
    def from_dict(cls, data: dict) -> 'ReviewTopicClassificationArtifact':
        """Create instance from dictionary."""
        return cls(**data)
    
    @classmethod
    def from_jsonl_file(cls, file_path: Path) -> list['ReviewTopicClassificationArtifact']:
        """
        Load and validate multiple review classifications from a JSONL file.
        
        Args:
            file_path: Path to the JSONL file
            
        Returns:
            List of validated ReviewTopicClassificationArtifact instances
        """
        results = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    results.append(cls.from_dict(data))
                except json.JSONDecodeError as e:
                    raise ValueError(f"Invalid JSON on line {line_num} of {file_path}: {e}")
                except Exception as e:
                    raise ValueError(f"Validation error on line {line_num} of {file_path}: {e}")
        return results
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return self.model_dump(exclude_none=True)
    
    class Config:
        json_schema_extra = {
            "example": {
                "review_id": "U001_P123",
                "user_id": "U001",
                "review": "Great product, very comfortable and well-made.",
                "category": "Clothing_Shoes_and_Jewelry",
                "identified_themes": ["Quality & Durability", "Comfort & Breathability"],
                "theme_token_probabilities": {
                    "Quality & Durability": 0.6,
                    "Comfort & Breathability": 0.4
                },
                "sentiment": "Positive",
                "rating": 5.0,
                "asin": "P123",
                "timestamp": "2024-01-15T10:30:00Z"
            }
        }

