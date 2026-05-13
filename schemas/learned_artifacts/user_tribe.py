
"""
User Tribe Learned Artifact Schema
===================================

Pydantic model for validating user tribe (micro cluster) artifacts.
Contains tribe assignments for each user.
"""

from typing import Optional, Dict
from pydantic import BaseModel, Field
from pathlib import Path
import json


class TribeMetadata(BaseModel):
    """Metadata about a tribe/micro cluster."""
    total_users: int = Field(..., description="Total number of users in this tribe", ge=0)
    total_reviews: int = Field(..., description="Total number of reviews from this tribe", ge=0)
    persona_name: Optional[str] = Field(None, description="Persona name for this tribe")
    relaxed_criteria: Optional[bool] = Field(None, description="Whether this tribe was created with relaxed criteria")


class UserTribeArtifact(BaseModel):
    """
    Schema for User Tribe learned artifact.
    
    Represents a user's tribe (micro cluster) assignment.
    
    Example:
        {
            "user_id": "U001",
            "segment_id": "segment_0",
            "tribe_id": "cluster_0_micro_5",
            "tribe_name": "Detail-Oriented Quality Seekers",
            "similarity_score": 0.87,
            "tribe_metadata": {
                "total_users": 25,
                "total_reviews": 150,
                "persona_name": "Detail-Oriented Quality Seekers",
                "relaxed_criteria": False
            }
        }
    """
    
    user_id: str = Field(..., description="Unique user identifier")
    segment_id: str = Field(..., description="Segment (macro cluster) identifier")
    tribe_id: str = Field(..., description="Tribe (micro cluster) identifier")
    tribe_name: Optional[str] = Field(None, description="Tribe persona name")
    similarity_score: Optional[float] = Field(None, description="Cosine similarity score to tribe seed", ge=0.0, le=1.0)
    tribe_metadata: TribeMetadata = Field(..., description="Metadata about the tribe")
    
    @classmethod
    def from_dict(cls, data: dict) -> 'UserTribeArtifact':
        """Create instance from dictionary."""
        # Convert tribe_metadata to TribeMetadata instance if it's a dict
        if 'tribe_metadata' in data and isinstance(data['tribe_metadata'], dict):
            data['tribe_metadata'] = TribeMetadata(**data['tribe_metadata'])
        
        return cls(**data)
    
    @classmethod
    def from_file(cls, file_path: Path) -> Dict[str, 'UserTribeArtifact']:
        """
        Load and validate multiple user tribes from a JSON file.
        
        Args:
            file_path: Path to the JSON file containing user tribes
            
        Returns:
            Dictionary mapping user_id to validated UserTribeArtifact instances
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
        # Convert TribeMetadata instance to dict if needed
        if 'tribe_metadata' in result and isinstance(result['tribe_metadata'], TribeMetadata):
            result['tribe_metadata'] = result['tribe_metadata'].model_dump(exclude_none=True)
        return result
    
    class Config:
        json_schema_extra = {
            "example": {
                "user_id": "U001",
                "segment_id": "segment_0",
                "tribe_id": "cluster_0_micro_5",
                "tribe_name": "Detail-Oriented Quality Seekers",
                "similarity_score": 0.87,
                "tribe_metadata": {
                    "total_users": 25,
                    "total_reviews": 150,
                    "persona_name": "Detail-Oriented Quality Seekers",
                    "relaxed_criteria": False
                }
            }
        }

