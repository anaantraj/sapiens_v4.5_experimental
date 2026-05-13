"""
User Review History Learned Artifact Schema
===========================================

Pydantic model for validating user review history artifacts.
Contains review history for each user.
"""

from typing import List, Optional, Dict
from pydantic import BaseModel, Field
from pathlib import Path
import json


class ReviewHistoryItem(BaseModel):
    """Individual review item in user's review history."""
    review_id: str = Field(..., description="Unique review identifier")
    product_description: str = Field(..., description="Product description or title")
    review_text: str = Field(..., description="Review text content")


class UserReviewHistoryArtifact(BaseModel):
    """
    Schema for User Review History learned artifact.
    
    Represents a user's review history.
    
    Example:
        {
            "user_id": "U001",
            "review_history": [
                {
                    "review_id": "R001",
                    "review_text": "Great product...",
                    "rating": 5,
                    "category": "Electronics",
                    "asin": "B00123ABC",
                    "timestamp": "2024-01-15T10:30:00Z"
                }
            ]
        }
    """
    
    user_id: str = Field(..., description="Unique user identifier")
    review_history: List[ReviewHistoryItem] = Field(
        ...,
        description="List of reviews"
    )
    
    @classmethod
    def from_dict(cls, data: dict) -> 'UserReviewHistoryArtifact':
        """Create instance from dictionary."""
        # Convert review_history items to ReviewHistoryItem instances
        if 'review_history' in data and isinstance(data['review_history'], list):
            review_items = []
            for item in data['review_history']:
                if isinstance(item, dict):
                    review_items.append(ReviewHistoryItem(**item))
                else:
                    review_items.append(item)
            data['review_history'] = review_items
        
        return cls(**data)
    
    @classmethod
    def from_file(cls, file_path: Path) -> Dict[str, 'UserReviewHistoryArtifact']:
        """
        Load and validate multiple user review histories from a JSON file.
        
        Args:
            file_path: Path to the JSON file containing user review histories
            
        Returns:
            Dictionary mapping user_id to validated UserReviewHistoryArtifact instances
        """
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if not isinstance(data, dict):
            raise ValueError(f"Expected dictionary in {file_path}, got {type(data)}")
        
        validated = {}
        for user_id, user_data in data.items():
            try:
                user_data["user_id"] = user_id  # Ensure user_id is set
                validated[user_id] = cls.from_dict(user_data)
            except Exception as e:
                raise ValueError(f"Validation error for user {user_id} in {file_path}: {e}")
        
        return validated
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        result = self.model_dump(exclude_none=True)
        # Convert ReviewHistoryItem instances to dicts
        if 'review_history' in result:
            result['review_history'] = [
                item.model_dump(exclude_none=True) if isinstance(item, ReviewHistoryItem) else item
                for item in result['review_history']
            ]
        return result
    
    class Config:
        json_schema_extra = {
            "example": {
                "user_id": "U001",
                "review_history": [
                    {
                        "review_id": "R001",
                        "product_description": "Wireless Bluetooth Headphones with Noise Cancellation",
                        "review_text": "Great product, very comfortable and well-made."
                    }
                ]
            }
        }

