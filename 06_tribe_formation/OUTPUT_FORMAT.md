# User Segments (Macro Clusters) - Output Format

## Overview

The script `01_user_to_segments_mapper.py` clusters users into macro segments (segments) using embeddings, UMAP dimensionality reduction, and KMeans clustering. It generates multiple output files containing segment assignments and clustering metrics.

## Output Files

The script generates the following files in the artifact directory:

1. **`user_segments.json`** - Main learned artifact (dictionary keyed by user_id)
2. **`segment_{N}_users.json`** - Per-segment files grouped by user (one file per cluster)
3. **`clustered_persona_data.parquet`** - Parquet file with clustering data
4. **`cluster_summary.json`** - Summary statistics
5. **`clusters_visualization.png`** - Visualization of clusters

## Main Output File: `user_segments.json`

**Structure**: Top-level dictionary where:
- **Keys** = `user_id` (string)
- **Values** = User segment data (object)

### Single User Entry Structure

```json
{
  "A1B2C3D4E5F6G7": {
    "user_id": "A1B2C3D4E5F6G7",
    "segment_id": "segment_0",
    "segment_name": null,
    "user_embedding": [0.123, -0.456, 0.789, 0.101, -0.112, ...],
    "cluster_metadata": {
      "silhouette_score": 0.53,
      "cluster_size": 450,
      "davies_bouldin_score": 1.2,
      "balance_ratio": 0.8
    }
  }
}
```

## Field Descriptions

### Required Fields

1. **`user_id`** (string, required)
   - Unique user identifier
   - Added during validation (matches the dictionary key)

2. **`segment_id`** (string, required)
   - Segment (macro cluster) identifier
   - Format: `"segment_{cluster_number}"` (e.g., `"segment_0"`, `"segment_1"`)
   - Cluster numbers start from 0

3. **`user_embedding`** (array of numbers, required)
   - User embedding vector (semantic representation)
   - Typically 1536 dimensions (OpenAI embeddings)
   - Used for clustering and similarity calculations

4. **`cluster_metadata`** (object, required)
   - Metadata about the cluster/segment this user belongs to
   - Contains:
     - **`silhouette_score`** (float, required): Silhouette score for clustering quality (higher is better, range: -1 to 1)
     - **`cluster_size`** (integer, required, minimum: 0): Number of users in this cluster
     - **`davies_bouldin_score`** (float, optional): Davies-Bouldin score (lower is better)
     - **`balance_ratio`** (float, optional): Balance ratio (min/max cluster sizes, range: 0 to 1, higher is better)

### Optional Fields

5. **`segment_name`** (string or null, optional)
   - Optional human-readable segment name
   - Initially `null` (can be set later in the pipeline)
   - Example: `"Quality-Focused Reviewers"`

## Complete File Example

```json
{
  "A1B2C3D4E5F6G7": {
    "user_id": "A1B2C3D4E5F6G7",
    "segment_id": "segment_0",
    "segment_name": null,
    "user_embedding": [0.123, -0.456, 0.789, 0.101, -0.112, 0.234, ...],
    "cluster_metadata": {
      "silhouette_score": 0.53,
      "cluster_size": 450,
      "davies_bouldin_score": 1.2,
      "balance_ratio": 0.8
    }
  },
  "B2C3D4E5F6G7H8": {
    "user_id": "B2C3D4E5F6G7H8",
    "segment_id": "segment_1",
    "segment_name": null,
    "user_embedding": [-0.234, 0.567, -0.123, 0.456, 0.789, ...],
    "cluster_metadata": {
      "silhouette_score": 0.53,
      "cluster_size": 380,
      "davies_bouldin_score": 1.2,
      "balance_ratio": 0.8
    }
  }
}
```

## Per-Segment Files: `segment_{N}_users.json`

**Format**: Dictionary keyed by `user_id`, containing user data grouped by segment.

**Example**: `segment_0_users.json`

```json
{
  "A1B2C3D4E5F6G7": {
    "user_id": "A1B2C3D4E5F6G7",
    "segment_id": "segment_0",
    "cluster": 0,
    "user_embedding": [0.123, -0.456, 0.789, ...],
    "cluster_metadata": {
      "silhouette_score": 0.53,
      "cluster_size": 450,
      "davies_bouldin_score": 1.2,
      "balance_ratio": 0.8
    }
  },
  "C3D4E5F6G7H8I9": {
    "user_id": "C3D4E5F6G7H8I9",
    "segment_id": "segment_0",
    "cluster": 0,
    "user_embedding": [0.234, -0.567, 0.890, ...],
    "cluster_metadata": {
      "silhouette_score": 0.53,
      "cluster_size": 450,
      "davies_bouldin_score": 1.2,
      "balance_ratio": 0.8
    }
  }
}
```

**Note**: All users in a segment file belong to the same cluster (same `segment_id` and `cluster` number).

## Cluster Summary: `cluster_summary.json`

**Structure**: Single JSON object with overall clustering statistics.

```json
{
  "num_users": 5000,
  "num_clusters": 6,
  "silhouette_score": 0.53,
  "davies_bouldin_score": 1.2,
  "balance_ratio": 0.8,
  "cluster_sizes": {
    "0": 450,
    "1": 380,
    "2": 520,
    "3": 410,
    "4": 480,
    "5": 2760
  },
  "grouped_by_user": true,
  "segment_files_created": 6
}
```

### Summary Fields

- **`num_users`**: Total number of users clustered
- **`num_clusters`**: Number of clusters/segments created
- **`silhouette_score`**: Overall silhouette score (quality metric)
- **`davies_bouldin_score`**: Overall Davies-Bouldin score (lower is better)
- **`balance_ratio`**: Overall balance ratio across clusters
- **`cluster_sizes`**: Dictionary mapping cluster number to user count
- **`grouped_by_user`**: Always `true` (indicates data structure)
- **`segment_files_created`**: Number of per-segment files created

## Parquet File: `clustered_persona_data.parquet`

**Format**: Pandas DataFrame saved as Parquet file.

**Columns**:
- `user_id` (string): User identifier
- `embedding` (array): Full embedding vector
- `x` (float): UMAP dimension 1 (for visualization)
- `y` (float): UMAP dimension 2 (for visualization)
- `cluster` (integer): Cluster assignment (0, 1, 2, ...)

**Use Case**: For analysis, visualization, or backward compatibility with existing tools.

## Visualization: `clusters_visualization.png`

**Format**: PNG image file showing 2D UMAP projection of user embeddings colored by cluster.

**Content**:
- Scatter plot with each point representing a user
- Points colored by cluster assignment
- Legend showing cluster labels
- Title includes K value and silhouette score

## Clustering Process

1. **Input**: User embeddings from Stage 05
2. **Dimensionality Reduction**: UMAP reduces embeddings to 2D for visualization
3. **Scaling**: StandardScaler normalizes embeddings for clustering
4. **K Evaluation**: Tests K values from `min_clusters` to 10, evaluating:
   - Silhouette score
   - Davies-Bouldin score
   - Balance ratio
5. **K Selection**: Chooses optimal K based on:
   - Target K (if meets quality thresholds)
   - Best silhouette score (if target doesn't meet thresholds)
6. **Final Clustering**: Applies KMeans with selected K
7. **Output Generation**: Creates all output files

## Quality Metrics

### Silhouette Score
- **Range**: -1 to 1
- **Higher is better**: Indicates well-separated clusters
- **Good**: > 0.4
- **Excellent**: > 0.5

### Davies-Bouldin Score
- **Range**: 0 to ∞
- **Lower is better**: Indicates compact, well-separated clusters
- **Good**: < 1.5

### Balance Ratio
- **Range**: 0 to 1
- **Higher is better**: Indicates balanced cluster sizes
- **Good**: > 0.1 (10% of largest cluster)
- **Excellent**: > 0.5 (50% of largest cluster)

## Schema Validation

The output is validated against `UserSegmentsArtifact` schema:
- **Required**: `user_id`, `segment_id`, `user_embedding`, `cluster_metadata`
- **Optional**: `segment_name`
- **Nested Required**: `cluster_metadata.silhouette_score`, `cluster_metadata.cluster_size`

## Usage Examples

### Loading Main Artifact

```python
from schemas.learned_artifacts import UserSegmentsArtifact
from pathlib import Path

# Load user segments
segments_file = Path("user_segments.json")
user_segments = UserSegmentsArtifact.from_file(segments_file)

# Access a specific user's segment
user_id = "A1B2C3D4E5F6G7"
user_segment = user_segments[user_id]
print(f"User {user_id} is in {user_segment.segment_id}")
print(f"Cluster size: {user_segment.cluster_metadata.cluster_size}")
```

### Loading Per-Segment File

```python
import json

# Load users in segment 0
with open("segment_0_users.json", 'r') as f:
    segment_0_users = json.load(f)

print(f"Segment 0 has {len(segment_0_users)} users")
for user_id, user_data in segment_0_users.items():
    print(f"  {user_id}: cluster {user_data['cluster']}")
```

### Loading Summary

```python
import json

with open("cluster_summary.json", 'r') as f:
    summary = json.load(f)

print(f"Total users: {summary['num_users']}")
print(f"Number of clusters: {summary['num_clusters']}")
print(f"Silhouette score: {summary['silhouette_score']:.4f}")
print("\nCluster sizes:")
for cluster_id, size in summary['cluster_sizes'].items():
    print(f"  Cluster {cluster_id}: {size} users")
```

## Notes

- Segment IDs are zero-indexed (`segment_0`, `segment_1`, etc.)
- All users in the same segment share the same `cluster_metadata` (cluster-level metrics)
- The `segment_name` field is initially `null` and can be populated later with human-readable names
- Embeddings are the original high-dimensional vectors (not the UMAP-reduced 2D coordinates)
- The parquet file includes UMAP coordinates (`x`, `y`) for visualization purposes

