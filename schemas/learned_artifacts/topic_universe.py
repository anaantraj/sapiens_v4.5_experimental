"""
Topic Universe Learned Artifact Schema
======================================

Pydantic model for validating topic universe artifacts.
Topic universe contains discovered topics/themes for each product category.
"""

from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field, field_validator
import json
from pathlib import Path


class TopicUniverseArtifact(BaseModel):
    """
    Schema for Topic Universe learned artifact.
    
    Structure: {category_name: [list of topic strings]}
    
    Example:
        {
            "Clothing_Shoes_and_Jewelry": ["Quality", "Comfort", "Style"],
            "Electronics": ["Performance", "Battery Life"]
        }
    """
    
    topics_by_category: Dict[str, List[str]] = Field(
        ...,
        description="Dictionary mapping category names to lists of topic strings",
        min_length=1
    )
    
    metadata: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Optional metadata about the topic universe (method, model, etc.)"
    )
    
    @field_validator('topics_by_category')
    @classmethod
    def validate_topics(cls, v: Dict[str, List[str]]) -> Dict[str, List[str]]:
        """Validate that each category has at least one topic."""
        for category, topics in v.items():
            if not category or not isinstance(category, str):
                raise ValueError(f"Category must be a non-empty string, got: {category}")
            if not topics or not isinstance(topics, list):
                raise ValueError(f"Topics must be a non-empty list for category {category}")
            if not all(isinstance(topic, str) and topic.strip() for topic in topics):
                raise ValueError(f"All topics must be non-empty strings for category {category}")
        return v
    
    @classmethod
    def from_file(cls, file_path: Path) -> 'TopicUniverseArtifact':
        """
        Load and validate topic universe from a JSON file.
        
        Args:
            file_path: Path to the JSON file containing topic universe
            
        Returns:
            Validated TopicUniverseArtifact instance
        """
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Handle both merged structure and single category structure
        if isinstance(data, list):
            # Single category file - need category name from filename
            category = file_path.stem.replace('final_themes_', '').replace('_', ' ')
            return cls(topics_by_category={category: data})
        elif isinstance(data, dict):
            # Check if it's already in the right format
            if all(isinstance(v, list) for v in data.values()):
                return cls(topics_by_category=data)
            else:
                # Might have metadata mixed in
                topics = {k: v for k, v in data.items() if isinstance(v, list)}
                metadata = {k: v for k, v in data.items() if k not in topics}
                return cls(topics_by_category=topics, metadata=metadata if metadata else None)
        else:
            raise ValueError(f"Invalid topic universe format in {file_path}")
    
    @classmethod
    def merge_from_files(cls, file_paths: List[Path]) -> 'TopicUniverseArtifact':
        """
        Load and merge multiple category topic files into one artifact.
        
        Args:
            file_paths: List of paths to category-specific topic files
            
        Returns:
            Merged TopicUniverseArtifact instance
        """
        merged_topics = {}
        
        for file_path in file_paths:
            # Extract category name from filename
            # e.g., "final_themes_Clothing_Shoes_and_Jewelry.json" -> "Clothing_Shoes_and_Jewelry"
            category = file_path.stem.replace('final_themes_', '').replace('_', ' ')
            
            with open(file_path, 'r', encoding='utf-8') as f:
                topics = json.load(f)
            
            if isinstance(topics, list):
                merged_topics[category] = topics
            else:
                raise ValueError(f"Expected list of topics in {file_path}, got {type(topics)}")
        
        return cls(topics_by_category=merged_topics)
    
    def get_topics_for_category(self, category: str) -> List[str]:
        """Get topics for a specific category."""
        return self.topics_by_category.get(category, [])
    
    def get_all_categories(self) -> List[str]:
        """Get list of all categories in the topic universe."""
        return list(self.topics_by_category.keys())
    
    def get_total_topics_count(self) -> int:
        """Get total number of unique topics across all categories."""
        all_topics = set()
        for topics in self.topics_by_category.values():
            all_topics.update(topics)
        return len(all_topics)
    
    def to_merged_file(self, output_path: Path) -> None:
        """
        Save as merged structure to a single JSON file.
        
        Args:
            output_path: Path where to save the merged topic universe
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(self.topics_by_category, f, indent=2, ensure_ascii=False)
    
    class Config:
        json_schema_extra = {
            "example": {
                "topics_by_category": {
                    "Clothing_Shoes_and_Jewelry": [
                        "Quality",
                        "Comfort",
                        "Style",
                        "Durability"
                    ],
                    "Electronics": [
                        "Performance",
                        "Battery Life",
                        "Build Quality"
                    ]
                },
                "metadata": {
                    "method": "map_reduce_llm",
                    "model": "o3",
                    "total_categories": 2
                }
            }
        }

