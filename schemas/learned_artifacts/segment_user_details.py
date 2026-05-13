"""
Segment User Details Learned Artifact Schema
============================================

Pydantic model for validating segment user details artifacts.
Contains users with full review data grouped by segment/cluster.
"""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from pathlib import Path
import json


class ReviewDetail(BaseModel):
    """Individual review with all details."""
    product_description: str = Field(..., description="Product description or title")
    review_text: str = Field(..., description="Review text content")
    rating: Optional[float] = Field(None, description="Product rating (1-5)")
    category: Optional[str] = Field(None, description="Product category")
    main_category: Optional[str] = Field(None, description="Main product category")
    timestamp: Optional[Any] = Field(None, description="Review timestamp")
    predicted_themes: List[str] = Field(default_factory=list, description="List of predicted themes/topics")
    topic_probabilities: Dict[str, float] = Field(default_factory=dict, description="Topic probabilities dictionary")
    primary_topic: Optional[str] = Field(None, description="Primary topic")
    sentiment: Optional[str] = Field(None, description="Sentiment classification")
    asin: Optional[str] = Field(None, description="Product ASIN")
    cluster: int = Field(..., description="Segment/cluster number")
    # User characteristics are NOT included in reviews - they're only at user level


class UserDetail(BaseModel):
    """User with all reviews and characteristics."""
    reviews: List[ReviewDetail] = Field(..., description="List of reviews for this user")
    overall_characteristics: Dict[str, Any] = Field(default_factory=dict, description="User-level overall characteristics")
    category_characteristics: Optional[Dict[str, Any]] = Field(None, description="User-level category-specific characteristics (only if exists)")
    cluster: int = Field(..., description="Segment/cluster number")
    num_reviews: int = Field(..., description="Number of reviews for this user", ge=0)
    
    def model_dump(self, **kwargs):
        """Custom dump to exclude None or empty category_characteristics."""
        data = super().model_dump(**kwargs)
        # Remove category_characteristics if it's None or empty
        if 'category_characteristics' in data and (data['category_characteristics'] is None or data['category_characteristics'] == {}):
            data.pop('category_characteristics', None)
        return data


class SegmentUserDetailsArtifact:
    """
    Schema for Segment User Details learned artifact.
    
    Represents users with full review data grouped by segment/cluster.
    Structure: {user_id: UserDetail}
    """
    
    @classmethod
    def from_file(cls, file_path: Path) -> Dict[str, UserDetail]:
        """
        Load and validate segment user details from a JSON file.
        
        Args:
            file_path: Path to the JSON file containing segment user details
            
        Returns:
            Dictionary mapping user_id to validated UserDetail instances
        """
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if not isinstance(data, dict):
            raise ValueError(f"Expected dictionary in {file_path}, got {type(data)}")
        
        validated = {}
        for user_id, user_data in data.items():
            try:
                # Clean reviews: remove user characteristics from individual reviews
                if 'reviews' in user_data and isinstance(user_data['reviews'], list):
                    review_details = []
                    for review in user_data['reviews']:
                        # Remove user characteristics from review (they should only be at user level)
                        review_clean = {k: v for k, v in review.items() 
                                       if k not in ['overall_characteristics', 'category_characteristics']}
                        review_details.append(ReviewDetail(**review_clean))
                    user_data['reviews'] = review_details
                
                # Clean user-level: remove empty category_characteristics
                if 'category_characteristics' in user_data and (not user_data['category_characteristics'] or user_data['category_characteristics'] == {}):
                    user_data.pop('category_characteristics', None)
                
                validated[user_id] = UserDetail(**user_data)
            except Exception as e:
                raise ValueError(f"Validation error for user {user_id} in {file_path}: {e}")
        
        return validated
    
    @classmethod
    def validate_data(cls, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate segment user details data structure.
        
        Args:
            data: Dictionary mapping user_id to user data
            
        Returns:
            Validated data dictionary (serializable to JSON)
        """
        validated = {}
        for user_id, user_data in data.items():
            try:
                # Clean reviews: remove user characteristics from individual reviews
                if 'reviews' in user_data and isinstance(user_data['reviews'], list):
                    cleaned_reviews = []
                    for review in user_data['reviews']:
                        # Remove user characteristics from review (they should only be at user level)
                        review_clean = {k: v for k, v in review.items() 
                                       if k not in ['overall_characteristics', 'category_characteristics']}
                        cleaned_reviews.append(ReviewDetail(**review_clean))
                    user_data['reviews'] = cleaned_reviews
                
                # Clean user-level: remove empty category_characteristics
                if 'category_characteristics' in user_data and (not user_data['category_characteristics'] or user_data['category_characteristics'] == {}):
                    user_data.pop('category_characteristics', None)
                
                user_detail = UserDetail(**user_data)
                # Convert to dict with proper serialization
                validated[user_id] = user_detail.model_dump(mode='json')
            except Exception as e:
                raise ValueError(f"Validation error for user {user_id}: {e}")
        
        return validated

