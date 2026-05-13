"""
Tribe Seed Characteristics Learned Artifact Schema
===================================================

Pydantic model for validating tribe seed characteristics artifacts.
Contains consolidated persona characteristics for each tribe.
"""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from pathlib import Path
import json


class QuantitativeSummary(BaseModel):
    """Quantitative summary statistics."""
    average_rating: Optional[Any] = Field(None, description="Average rating")
    sentiment_distribution_percent: Dict[str, float] = Field(default_factory=dict, description="Sentiment distribution as percentages")


class QualitativeSummary(BaseModel):
    """Qualitative persona summary."""
    persona_summary: Optional[str] = Field(None, description="Persona summary text")
    key_motivations: List[str] = Field(default_factory=list, description="List of key motivations")
    common_praises: List[str] = Field(default_factory=list, description="List of common praises")
    common_criticisms: List[str] = Field(default_factory=list, description="List of common criticisms")
    core_characteristics: List[str] = Field(default_factory=list, description="List of core characteristics")
    potential_goals: List[str] = Field(default_factory=list, description="List of potential goals")


class MemberUserCharacteristic(BaseModel):
    """Individual member user characteristic."""
    user_id: str = Field(..., description="User identifier")
    characteristic_summary: str = Field(..., description="User's characteristic summary")


class TribeSeedCharacteristicsArtifact(BaseModel):
    """
    Schema for Tribe Seed Characteristics learned artifact.
    
    Represents consolidated persona characteristics for a tribe.
    
    Example:
        {
            "tribe_id": "cluster_0_micro_5",
            "persona_name": "Detail-Oriented Quality Seekers",
            "qualitative_summary": {
                "persona_summary": "Users who meticulously evaluate...",
                "key_motivations": ["quality assurance", "detailed analysis"],
                "common_praises": ["durability", "craftsmanship"],
                "common_criticisms": ["poor quality", "defects"]
            },
            "quantitative_summary": {
                "average_rating": "4.5",
                "sentiment_distribution_percent": {
                    "Positive": 75.0,
                    "Negative": 15.0,
                    "Neutral": 10.0
                }
            },
            "key_topics": ["Quality", "Durability", "Craftsmanship"],
            "member_user_characteristics": [...]
        }
    """
    
    tribe_id: str = Field(..., description="Tribe (micro cluster) identifier")
    persona_name: Optional[str] = Field(None, description="Persona name for this tribe")
    micro_cluster_id: Optional[str] = Field(None, description="Micro cluster identifier")
    total_users_in_cluster: Optional[int] = Field(None, description="Total number of users in this tribe", ge=0)
    total_reviews_from_cluster: Optional[int] = Field(None, description="Total number of reviews from this tribe", ge=0)
    quantitative_summary: Optional[QuantitativeSummary] = Field(None, description="Quantitative statistics")
    qualitative_summary: Optional[QualitativeSummary] = Field(None, description="Qualitative persona summary")
    justification: Optional[str] = Field(None, description="Justification for the persona name")
    key_topics: List[str] = Field(default_factory=list, description="List of key topics/themes")
    
    # Convenience fields (duplicates from qualitative_summary)
    persona_summary: Optional[str] = Field(None, description="Persona summary (convenience field)")
    key_motivations: List[str] = Field(default_factory=list, description="Key motivations (convenience field)")
    common_praises: List[str] = Field(default_factory=list, description="Common praises (convenience field)")
    common_criticisms: List[str] = Field(default_factory=list, description="Common criticisms (convenience field)")
    core_characteristics: List[str] = Field(default_factory=list, description="Core characteristics (convenience field)")
    potential_goals: List[str] = Field(default_factory=list, description="Potential goals (convenience field)")
    
    member_user_characteristics: List[MemberUserCharacteristic] = Field(default_factory=list, description="List of member user characteristics")
    members_grouped_by_user: Dict[str, List[Dict[str, Any]]] = Field(default_factory=dict, description="Reviews grouped by user_id. Format: {user_id: [list of review objects]}")
    
    @classmethod
    def from_dict(cls, data: dict) -> 'TribeSeedCharacteristicsArtifact':
        """Create instance from dictionary."""
        # Convert nested objects if they're dicts
        if 'quantitative_summary' in data and isinstance(data['quantitative_summary'], dict):
            data['quantitative_summary'] = QuantitativeSummary(**data['quantitative_summary'])
        
        if 'qualitative_summary' in data and isinstance(data['qualitative_summary'], dict):
            data['qualitative_summary'] = QualitativeSummary(**data['qualitative_summary'])
        
        # Convert member_user_characteristics
        if 'member_user_characteristics' in data and isinstance(data['member_user_characteristics'], list):
            member_chars = []
            for item in data['member_user_characteristics']:
                if isinstance(item, dict):
                    member_chars.append(MemberUserCharacteristic(**item))
                else:
                    member_chars.append(item)
            data['member_user_characteristics'] = member_chars
        
        return cls(**data)
    
    @classmethod
    def from_file(cls, file_path: Path) -> Dict[str, 'TribeSeedCharacteristicsArtifact']:
        """
        Load and validate multiple tribe seed characteristics from a JSON file.
        
        Args:
            file_path: Path to the JSON file containing tribe seed characteristics
            
        Returns:
            Dictionary mapping tribe_id to validated TribeSeedCharacteristicsArtifact instances
        """
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if not isinstance(data, dict):
            raise ValueError(f"Expected dictionary in {file_path}, got {type(data)}")
        
        validated = {}
        for tribe_id, tribe_data in data.items():
            try:
                tribe_data["tribe_id"] = tribe_id  # Ensure tribe_id is set
                validated[tribe_id] = cls.from_dict(tribe_data)
            except Exception as e:
                raise ValueError(f"Validation error for tribe {tribe_id} in {file_path}: {e}")
        
        return validated
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        result = self.model_dump(exclude_none=True, mode='json')
        # Convert nested models to dicts if needed
        if 'quantitative_summary' in result and isinstance(result['quantitative_summary'], QuantitativeSummary):
            result['quantitative_summary'] = result['quantitative_summary'].model_dump(exclude_none=True)
        if 'qualitative_summary' in result and isinstance(result['qualitative_summary'], QualitativeSummary):
            result['qualitative_summary'] = result['qualitative_summary'].model_dump(exclude_none=True)
        if 'member_user_characteristics' in result:
            result['member_user_characteristics'] = [
                item.model_dump(exclude_none=True) if isinstance(item, MemberUserCharacteristic) else item
                for item in result['member_user_characteristics']
            ]
        return result
    
    class Config:
        json_schema_extra = {
            "example": {
                "tribe_id": "cluster_0_micro_5",
                "persona_name": "Detail-Oriented Quality Seekers",
                "qualitative_summary": {
                    "persona_summary": "Users who meticulously evaluate products...",
                    "key_motivations": ["quality assurance", "detailed analysis"],
                    "common_praises": ["durability", "craftsmanship"],
                    "common_criticisms": ["poor quality", "defects"]
                },
                "quantitative_summary": {
                    "average_rating": "4.5",
                    "sentiment_distribution_percent": {
                        "Positive": 75.0,
                        "Negative": 15.0,
                        "Neutral": 10.0
                    }
                },
                "key_topics": ["Quality", "Durability", "Craftsmanship"]
            }
        }

