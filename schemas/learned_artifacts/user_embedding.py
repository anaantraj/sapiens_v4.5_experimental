"""
User Embedding Learned Artifact Schema
======================================

Pydantic model for validating user embedding artifacts.
Contains semantic embeddings for users based on their characteristics.
"""

from typing import List, Optional
from pydantic import BaseModel, Field, field_validator
from pathlib import Path
import json


class UserEmbeddingArtifact(BaseModel):
    """
    Schema for User Embedding learned artifact.
    
    Represents a semantic embedding vector for a user.
    
    Example:
        {
            "user_id": "U001",
            "user_embedding": [0.123, -0.456, 0.789, ...],
            "user_categories": ["Clothing_Shoes_and_Jewelry", "Electronics"],
            "embedding_model": "text-embedding-3-small",
            "embedding_dimension": 1536,
            "embedding_text": "A detailed reviewer who focuses on quality..."
        }
    """
    
    user_id: str = Field(..., description="Unique user identifier", min_length=1)
    user_embedding: List[float] = Field(
        ...,
        description="Semantic embedding vector for the user",
        min_length=1
    )
    user_categories: List[str] = Field(
        ...,
        description="List of categories the user has reviewed",
        min_length=1
    )
    embedding_model: str = Field(
        ...,
        description="Name of the embedding model used",
        min_length=1
    )
    embedding_dimension: int = Field(
        ...,
        description="Dimension of the embedding vector",
        ge=1
    )
    embedding_text: Optional[str] = Field(
        None,
        description="Text used to generate the embedding (optional)",
        min_length=1
    )
    
    @field_validator('user_embedding')
    @classmethod
    def validate_embedding(cls, v: List[float]) -> List[float]:
        """Validate embedding is a list of numbers."""
        if not v:
            raise ValueError("User embedding cannot be empty")
        if not all(isinstance(x, (int, float)) for x in v):
            raise ValueError("All embedding values must be numbers")
        return v
    
    @field_validator('user_categories')
    @classmethod
    def validate_categories(cls, v: List[str]) -> List[str]:
        """Validate categories list."""
        if not v:
            raise ValueError("User categories cannot be empty")
        if not all(isinstance(cat, str) and cat.strip() for cat in v):
            raise ValueError("All categories must be non-empty strings")
        return v
    
    @field_validator('embedding_dimension')
    @classmethod
    def validate_dimension(cls, v: int, info) -> int:
        """Validate embedding dimension matches actual embedding length."""
        if 'user_embedding' in info.data:
            actual_dim = len(info.data['user_embedding'])
            if v != actual_dim:
                raise ValueError(
                    f"Embedding dimension ({v}) does not match actual embedding length ({actual_dim})"
                )
        return v
    
    @classmethod
    def from_dict(cls, data: dict) -> 'UserEmbeddingArtifact':
        """Create instance from dictionary."""
        return cls(**data)
    
    @classmethod
    def from_file(cls, file_path: Path) -> dict[str, 'UserEmbeddingArtifact']:
        """
        Load and validate multiple user embeddings from a JSON file.
        
        Args:
            file_path: Path to the JSON file containing user embeddings
            
        Returns:
            Dictionary mapping user_id to validated UserEmbeddingArtifact instances
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
        return self.model_dump(exclude_none=True)
    
    class Config:
        json_schema_extra = {
            "example": {
                "user_id": "U001",
                "user_embedding": [0.123, -0.456, 0.789, 0.234, -0.567],
                "user_categories": ["Clothing_Shoes_and_Jewelry", "Electronics"],
                "embedding_model": "text-embedding-3-small",
                "embedding_dimension": 1536,
                "embedding_text": "A detailed reviewer who focuses on quality and durability. Values thorough analysis and helps others make informed decisions."
            }
        }

