"""
User Segments Learned Artifact Schema
======================================

Pydantic model for validating user segments (macro clusters) artifacts.
Contains segment assignments for each user.
"""

from typing import List, Optional, Dict
from pydantic import BaseModel, Field
from pathlib import Path
import json


class ClusterMetadata(BaseModel):
    """Metadata about a cluster/segment."""
    silhouette_score: float = Field(..., description="Silhouette score for this cluster")
    cluster_size: int = Field(..., description="Number of users in this cluster", ge=0)
    davies_bouldin_score: Optional[float] = Field(None, description="Davies-Bouldin score for clustering quality")
    balance_ratio: Optional[float] = Field(None, description="Balance ratio (min/max cluster sizes)")


class UserSegmentsArtifact(BaseModel):
    """
    Schema for User Segments learned artifact.
    
    Represents a user's segment (macro cluster) assignment.
    
    Example:
        {
            "user_id": "U001",
            "segment_id": "segment_0",
            "segment_name": "Quality-Focused Reviewers",
            "user_embedding": [0.123, -0.456, 0.789, ...],
            "cluster_metadata": {
                "silhouette_score": 0.53,
                "cluster_size": 450,
                "davies_bouldin_score": 1.2,
                "balance_ratio": 0.8
            }
        }
    """
    
    user_id: str = Field(..., description="Unique user identifier")
    segment_id: str = Field(..., description="Segment (macro cluster) identifier")
    segment_name: Optional[str] = Field(None, description="Optional segment name")
    user_embedding: List[float] = Field(..., description="User embedding vector")
    cluster_metadata: ClusterMetadata = Field(..., description="Metadata about the cluster")
    
    @classmethod
    def from_dict(cls, data: dict) -> 'UserSegmentsArtifact':
        """Create instance from dictionary."""
        # Convert cluster_metadata to ClusterMetadata instance if it's a dict
        if 'cluster_metadata' in data and isinstance(data['cluster_metadata'], dict):
            data['cluster_metadata'] = ClusterMetadata(**data['cluster_metadata'])
        
        return cls(**data)
    
    @classmethod
    def from_file(cls, file_path: Path) -> Dict[str, 'UserSegmentsArtifact']:
        """
        Load and validate multiple user segments from a JSON file.
        
        Args:
            file_path: Path to the JSON file containing user segments
            
        Returns:
            Dictionary mapping user_id to validated UserSegmentsArtifact instances
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
        # Convert ClusterMetadata instance to dict if needed
        if 'cluster_metadata' in result and isinstance(result['cluster_metadata'], ClusterMetadata):
            result['cluster_metadata'] = result['cluster_metadata'].model_dump(exclude_none=True)
        return result
    
    class Config:
        json_schema_extra = {
            "example": {
                "user_id": "U001",
                "segment_id": "segment_0",
                "segment_name": "Quality-Focused Reviewers",
                "user_embedding": [0.123, -0.456, 0.789, 0.101, -0.112],
                "cluster_metadata": {
                    "silhouette_score": 0.53,
                    "cluster_size": 450,
                    "davies_bouldin_score": 1.2,
                    "balance_ratio": 0.8
                }
            }
        }

