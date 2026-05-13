# Learned Artifacts Schemas

## Overview
This directory contains JSON schemas and Pydantic models for validating learned artifacts (intermediate outputs from pipeline stages).

## Structure
```
schemas/
├── __init__.py
├── schema_registry.py              # Central registry for all schemas
├── learned_artifacts/
│   ├── __init__.py
│   ├── topic_universe_schema.json  # JSON Schema (Stage 03)
│   └── topic_universe.py          # Pydantic model (Stage 03)
└── README.md
```

## Stage 03: Topic Universe Schema

### Structure
**Merged Format (Option 2):**
```json
{
  "Clothing_Shoes_and_Jewelry": ["Quality", "Comfort", "Style", ...],
  "Electronics": ["Performance", "Battery Life", ...],
  "Appliances": ["Durability", "Efficiency", ...]
}
```

### Usage Example

#### 1. Validate when saving (in Stage 03 script):
```python
from schemas.learned_artifacts import TopicUniverseArtifact

# After discovering themes for a category
final_unique_themes = sorted(list(set(all_discovered_themes)))

# Create validated artifact (merge with existing if file exists)
output_dir = get_artifact_dir("03_topic_universe", output_artifact_name)
merged_file = output_dir / "topic_universe.json"

# Load existing if it exists, otherwise create new
if merged_file.exists():
    existing = TopicUniverseArtifact.from_file(merged_file)
    # Add new category
    existing.topics_by_category[category_name] = final_unique_themes
    validated = existing
else:
    validated = TopicUniverseArtifact(
        topics_by_category={category_name: final_unique_themes},
        metadata={
            "method": "map_reduce_llm",
            "model": model,
            "rounds_completed": rounds_completed
        }
    )

# Save merged file
validated.to_merged_file(merged_file)

# Log to W&B (already validated)
log_artifact(
    run=run,
    artifact_name=output_artifact_name,
    artifact_type="dataset",
    artifact_path=output_dir,
    metadata={
        "category": category_name,
        "num_topics": len(final_unique_themes),
        "topics": final_unique_themes,
        "method": "map_reduce_llm",
        "model": model,
        "rounds_completed": rounds_completed,
        "total_categories": len(validated.topics_by_category)
    }
)
```

#### 2. Validate when loading (in later stages):
```python
from schemas.learned_artifacts import TopicUniverseArtifact

# Load and validate
topic_universe = TopicUniverseArtifact.from_file(
    Path("artifacts/topic_universe_v1/topic_universe.json")
)

# Use validated data
topics = topic_universe.get_topics_for_category("Clothing_Shoes_and_Jewelry")
all_categories = topic_universe.get_all_categories()
```

#### 3. Using the registry:
```python
from schemas.schema_registry import validate_artifact

# Validate using registry
validated_data = validate_artifact(
    "topic_universe",
    {"topics_by_category": {"Electronics": ["Performance", "Battery"]}}
)
```

## Next Steps

1. ✅ **Stage 03 Schema Created** - Topic Universe
2. ⏳ **Stage 04** - Review Topic Classification
3. ⏳ **Stage 05** - User Backstory, Embedding, Review History
4. ⏳ **Stage 06** - User Segments, User Tribe, Tribe Seed Characteristics
5. ⏳ **Stage 07** - Refined Characteristics, Behavior Loss

## Validation Benefits

- ✅ **Early Error Detection**: Catch invalid data before saving
- ✅ **Type Safety**: Ensure correct data types
- ✅ **W&B Integration**: Proper artifact tracking
- ✅ **Pipeline Stability**: Prevent downstream failures

