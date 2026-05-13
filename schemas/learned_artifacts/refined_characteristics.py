"""
Refined Characteristics Artifact Schema
=======================================

Pydantic model for refined characteristics learned artifact from SGO training.
Supports both simple string format and Chain of Thought (CoT) format with confidence scores.
"""

from pydantic import BaseModel, Field, ValidationError
from typing import Dict, List, Optional, Union, Any
from pathlib import Path
import json


class CharacteristicEvidence(BaseModel):
    """Evidence item from batch analysis."""
    analysis_index: int = Field(..., description="Index of the batch analysis")
    analysis_text: str = Field(..., description="Preview text from the analysis")


class CharacteristicWithCoT(BaseModel):
    """Characteristic with Chain of Thought reasoning."""
    characteristic: str = Field(..., description="The characteristic text")
    confidence: Optional[float] = Field(None, ge=0.0, le=1.0, description="Confidence score (0.0-1.0)")
    reasoning_trace: Optional[List[str]] = Field(None, description="Step-by-step reasoning trace")
    evidence_indices: Optional[List[int]] = Field(None, description="Indices of supporting batch analyses")
    evidence: Optional[List[CharacteristicEvidence]] = Field(None, description="Evidence details")
    support_count: Optional[int] = Field(None, ge=0, description="Number of analyses supporting this")


CharacteristicItem = Union[str, CharacteristicWithCoT]


class RefinedCharacteristicsData(BaseModel):
    """Refined characteristics data structure."""
    key_motivations: List[CharacteristicItem] = Field(default_factory=list, description="Key motivations")
    common_praises: List[CharacteristicItem] = Field(default_factory=list, description="Common praises")
    common_criticisms: List[CharacteristicItem] = Field(default_factory=list, description="Common criticisms")
    core_characteristics: List[CharacteristicItem] = Field(default_factory=list, description="Core characteristics")
    potential_goals: List[CharacteristicItem] = Field(default_factory=list, description="Potential goals")


class RefinementMetadata(BaseModel):
    """Metadata about the refinement process."""
    num_analyses: int = Field(..., ge=0, description="Number of batch analyses used")
    refinement_round: int = Field(..., ge=1, description="Iteration/round number")
    consolidation_applied: Optional[bool] = Field(None, description="Whether characteristics were consolidated")


class RefinedCharacteristicsArtifact(BaseModel):
    """
    Schema for Refined Characteristics learned artifact.
    Contains persona characteristics refined through SGO training feedback loops.
    """
    tribe_id: str = Field(..., description="Unique identifier for the tribe (micro cluster)")
    refined_characteristics: RefinedCharacteristicsData = Field(..., description="Refined characteristics")
    refinement_metadata: RefinementMetadata = Field(..., description="Refinement metadata")

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'RefinedCharacteristicsArtifact':
        """Load and validate RefinedCharacteristicsArtifact from a dictionary."""
        try:
            return cls(**data)
        except ValidationError as e:
            raise ValueError(f"Schema validation failed for refined characteristics data: {e}")

    @classmethod
    def from_file(cls, file_path: Path) -> Dict[str, 'RefinedCharacteristicsArtifact']:
        """
        Load and validate multiple refined characteristics from a JSON file.
        
        Args:
            file_path: Path to the JSON file containing refined characteristics
            
        Returns:
            Dictionary mapping tribe_id to validated RefinedCharacteristicsArtifact instances
        """
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if not isinstance(data, dict):
            raise ValueError(f"Expected dictionary in {file_path}, got {type(data)}")
        
        validated = {}
        for tribe_id, tribe_data in data.items():
            try:
                tribe_data_with_id = tribe_data.copy()
                tribe_data_with_id["tribe_id"] = tribe_id
                validated[tribe_id] = cls.from_dict(tribe_data_with_id)
            except Exception as e:
                raise ValueError(f"Validation error for tribe {tribe_id} in {file_path}: {e}")
        
        return validated

    def to_dict(self) -> Dict[str, Any]:
        """Convert the model instance to a dictionary."""
        return self.model_dump(exclude_none=True)

    class Config:
        json_schema_extra = {
            "example": {
                "tribe_id": "cluster_0_micro_5",
                "refined_characteristics": {
                    "key_motivations": [
                        {
                            "characteristic": "Values quality over price",
                            "confidence": 0.85,
                            "reasoning_trace": [
                                "Step 1: Observed in analyses #2, #5, #7",
                                "Step 2: Pattern consistent across categories"
                            ],
                            "evidence_indices": [2, 5, 7],
                            "support_count": 3
                        }
                    ],
                    "common_praises": ["Durability", "Build quality"],
                    "common_criticisms": ["Poor quality", "Defects"],
                    "core_characteristics": ["Detail-oriented", "Quality-focused"],
                    "potential_goals": ["Find long-lasting products"]
                },
                "refinement_metadata": {
                    "num_analyses": 50,
                    "refinement_round": 1,
                    "consolidation_applied": False
                }
            }
        }

