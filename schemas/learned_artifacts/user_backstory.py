"""
User Backstory Learned Artifact Schema
======================================

Pydantic model for validating user backstory artifacts.
Contains user characteristics extracted from review history.
"""

from typing import Dict, List, Optional
from pydantic import BaseModel, Field, field_validator
from pathlib import Path
import json


class CategoryCharacteristics(BaseModel):
    """Category-specific user characteristics."""
    influencing_characteristics_summary: str = Field(
        ...,
        description="A 2-3 sentence paragraph synthesizing the core factors that motivate this user's reviews for this category",
        min_length=1
    )


class OverallCharacteristics(BaseModel):
    """Overall user characteristics from LLM analysis."""
    influencing_characteristics_summary: str = Field(
        ...,
        description="A 2-3 sentence paragraph synthesizing the core factors that motivate this user's reviews",
        min_length=1
    )


class UserBackstoryArtifact(BaseModel):
    """
    Schema for User Backstory learned artifact.
    
    Represents user characteristics extracted from their review history.
    
    Example:
        {
            "user_id": "U001",
            "overall_characteristics": {
                "influencing_characteristics_summary": "A detailed reviewer who focuses on quality..."
            },
            "category_characteristics": {
                "Clothing_Shoes_and_Jewelry": {
                    "influencing_characteristics_summary": "Focuses on style and fit..."
                }
            },
            "num_categories": 2,
            "categories": ["Clothing_Shoes_and_Jewelry", "Electronics"]
        }
    """
    
    user_id: Optional[str] = Field(None, description="Unique user identifier (optional, as it's the key in the JSON structure)")
    overall_characteristics: OverallCharacteristics = Field(
        ...,
        description="Overall user characteristics from LLM analysis"
    )
    category_characteristics: Optional[Dict[str, CategoryCharacteristics]] = Field(
        None,
        description="Category-specific characteristics (if user reviews multiple categories)"
    )
    num_categories: int = Field(0, description="Number of categories the user has reviewed", ge=0)
    categories: List[str] = Field(
        default_factory=list,
        description="List of categories the user has reviewed (can be empty if user only has overall characteristics)"
    )
    
    @field_validator('categories')
    @classmethod
    def validate_categories(cls, v: List[str]) -> List[str]:
        """Validate categories list - allow empty list."""
        # Filter out None values and empty strings
        filtered = [cat for cat in v if cat is not None and isinstance(cat, str) and cat.strip()]
        return filtered
    
    @classmethod
    def from_dict(cls, data: dict) -> 'UserBackstoryArtifact':
        """Create instance from dictionary."""
        # Handle case where overall_characteristics might be a dict directly
        if isinstance(data.get("overall_characteristics"), dict):
            data["overall_characteristics"] = OverallCharacteristics(**data["overall_characteristics"])
        # Backward compatibility: also check for old Overall_characteristics and llm_characteristics keys
        elif isinstance(data.get("Overall_characteristics"), dict):
            data["overall_characteristics"] = OverallCharacteristics(**data.pop("Overall_characteristics"))
        elif isinstance(data.get("llm_characteristics"), dict):
            data["overall_characteristics"] = OverallCharacteristics(**data.pop("llm_characteristics"))
        
        # Handle category_characteristics
        if data.get("category_characteristics"):
            category_chars = {}
            for cat, char_data in data["category_characteristics"].items():
                if isinstance(char_data, dict):
                    category_chars[cat] = CategoryCharacteristics(**char_data)
                else:
                    category_chars[cat] = char_data
            data["category_characteristics"] = category_chars
        
        return cls(**data)
    
    @classmethod
    def from_file(cls, file_path: Path) -> Dict[str, 'UserBackstoryArtifact']:
        """
        Load and validate multiple user backstories from a JSON file.
        
        Args:
            file_path: Path to the JSON file containing user backstories
            
        Returns:
            Dictionary mapping user_id to validated UserBackstoryArtifact instances
        """
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if not isinstance(data, dict):
            raise ValueError(f"Expected dictionary in {file_path}, got {type(data)}")
        
        validated = {}
        skipped_users = []
        for user_id, user_data in data.items():
            try:
                # Add user_id for validation (it may not be in the saved data)
                user_data_for_validation = user_data.copy()
                if "user_id" not in user_data_for_validation:
                    user_data_for_validation["user_id"] = user_id
                
                # Clean categories list: remove None values and empty strings
                if "categories" in user_data_for_validation and isinstance(user_data_for_validation["categories"], list):
                    user_data_for_validation["categories"] = [
                        cat for cat in user_data_for_validation["categories"] 
                        if cat is not None and isinstance(cat, str) and cat.strip()
                    ]
                else:
                    # If categories is missing or not a list, set to empty list
                    user_data_for_validation["categories"] = []
                
                # Update num_categories to match the cleaned categories list
                if "num_categories" in user_data_for_validation:
                    user_data_for_validation["num_categories"] = len(user_data_for_validation["categories"])
                else:
                    user_data_for_validation["num_categories"] = len(user_data_for_validation["categories"])
                
                validated[user_id] = cls.from_dict(user_data_for_validation)
            except Exception as e:
                # Skip invalid users with a warning instead of failing completely
                import logging
                logging.warning(f"Skipping user {user_id} due to validation error: {e}")
                skipped_users.append(user_id)
        
        if skipped_users:
            import logging
            logging.warning(f"Skipped {len(skipped_users)} invalid users: {skipped_users[:5]}{'...' if len(skipped_users) > 5 else ''}")
        
        return validated
    
    def to_dict(self) -> dict:
        """Convert to dictionary, excluding user_id (since it's the key in the JSON structure)."""
        data = self.model_dump(exclude_none=True)
        # Remove user_id from the dict since it's redundant (it's the key in the JSON file)
        data.pop("user_id", None)
        return data
    
    class Config:
        json_schema_extra = {
            "example": {
                "user_id": "U001",
                "overall_characteristics": {
                    "influencing_characteristics_summary": "A detailed reviewer who focuses on quality and durability. Values thorough analysis and helps others make informed decisions."
                },
                "category_characteristics": {
                    "Clothing_Shoes_and_Jewelry": {
                        "influencing_characteristics_summary": "Focuses on style, fit, and comfort when reviewing clothing items."
                    },
                    "Electronics": {
                        "influencing_characteristics_summary": "Values performance, reliability, and technical specifications."
                    }
                },
                "num_categories": 2,
                "categories": ["Clothing_Shoes_and_Jewelry", "Electronics"]
            }
        }

