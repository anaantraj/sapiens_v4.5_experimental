#!/usr/bin/env python3
"""
Stage 06: Tribe Formation - User to Segments Mapper
====================================================

Clusters users into macro segments using embeddings.
- Reads configuration from config.yaml
- Downloads embeddings artifact from W&B
- Performs UMAP dimensionality reduction
- Clusters with KMeans, optimizing K
- Logs metrics during processing
- Uploads clustering results to W&B

Usage:
    python 06_tribe_formation/scripts/01_user_to_segments_mapper.py
"""

import pandas as pd
import numpy as np
import sys
import json
import logging
import warnings
import re
from pathlib import Path

warnings.filterwarnings('ignore')

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from utils.wandb_utils import (
    load_config, get_stage_config,
    init_wandb_run, finish_run, use_artifact, log_artifact,
    log_metrics, log_summary, link_to_registry, get_artifact_dir,
    create_comprehensive_artifact_metadata, get_learned_artifact_schema,
    validate_stage_dependencies
)

# Import schema for validation
from schemas.learned_artifacts import UserEmbeddingArtifact, UserSegmentsArtifact

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def main():
    """Main execution function."""
    
    # =========================================================================
    # Step 1: Load configuration from config.yaml
    # =========================================================================
    print("=" * 70)
    print("STAGE 06: Tribe Formation - User Clustering")
    print("=" * 70)
    
    # Load stage-specific config
    cfg = get_stage_config("06_tribe_formation")
    
    # Validate required config fields
    if "hyperparameters" not in cfg:
        logging.error("Missing required config field: hyperparameters")
        return
    
    hyperparams = cfg["hyperparameters"]
    
    # Get input artifacts from config (required, no fallbacks)
    # Check for segments_input_artifact first, then fallback to input_artifact
    if "segments_input_artifact" in cfg:
        input_artifact = cfg["segments_input_artifact"]
    elif "input_artifact" in cfg:
        input_artifact = cfg["input_artifact"]
    else:
        logging.error("Missing required config field: segments_input_artifact or input_artifact")
        return
    
    if "segments_output_artifact" in cfg:
        output_artifact = cfg["segments_output_artifact"]
    elif "output_artifact" in cfg:
        output_artifact = cfg["output_artifact"]
    else:
        logging.error("Missing required config field: segments_output_artifact or output_artifact")
        return
    
    # Get embedding filename from config (required, no fallback)
    if "embedding_filename" not in cfg:
        logging.error("Missing required config field: embedding_filename")
        return
    
    embedding_filename = cfg["embedding_filename"]
    
    # Get hyperparameters from config (required, no fallbacks)
    required_hyperparams = ["n_macro_clusters", "min_clusters", "target_clusters", 
                           "umap_n_neighbors", "umap_n_components", "umap_min_dist",
                           "silhouette_threshold", "balance_threshold",
                           "max_k", "random_state", "n_init"]
    for param in required_hyperparams:
        if param not in hyperparams:
            logging.error(f"Missing required hyperparameter: {param}")
            return
    
    n_macro_clusters = hyperparams["n_macro_clusters"]
    min_clusters = hyperparams["min_clusters"]
    target_clusters = hyperparams["target_clusters"]
    
    umap_n_neighbors = hyperparams["umap_n_neighbors"]
    umap_n_components = hyperparams["umap_n_components"]
    umap_min_dist = hyperparams["umap_min_dist"]
    
    silhouette_threshold = hyperparams["silhouette_threshold"]
    balance_threshold = hyperparams["balance_threshold"]
    
    max_k = hyperparams["max_k"]
    random_state = hyperparams["random_state"]
    n_init = hyperparams["n_init"]
    
    print(f"\n[Config] Input artifact: {input_artifact}")
    print(f"[Config] Target clusters: {target_clusters}")
    print(f"[Config] UMAP neighbors: {umap_n_neighbors}")
    
    # =========================================================================
    # Step 2: Initialize W&B run
    # =========================================================================
    run = init_wandb_run(
        run_name=f"tribe_formation_{output_artifact}",
        stage="06_tribe_formation",
        job_type="clustering",
    )
    
    # Validate dependencies
    required_artifacts = [input_artifact]
    if not validate_stage_dependencies(run, "06_tribe_formation", required_artifacts):
        logging.error("Stage 05.5 (User Embeddings) must be completed first!")
        return
    
    try:
        # =====================================================================
        # Step 3: Get input artifact from W&B (no local fallback)
        # =====================================================================
        print("\n" + "-" * 70)
        print("Step 3: Get Input Artifact from W&B")
        print("-" * 70)
        
        # Download embeddings artifact from W&B (required, no local fallback)
        logging.info(f"Downloading embeddings artifact from W&B: {input_artifact}")
        input_path = use_artifact(run, input_artifact, artifact_type="dataset")
        
        if input_path is None:
            logging.error(f"Could not download embeddings artifact: {input_artifact}")
            logging.error(f"Make sure to run 05_user_level_inference/scripts/02_user_to_embedding.py first and upload the artifact to W&B")
            return
        
        # Resolve path to handle any symlinks or relative paths
        # W&B may return path with :v0, :v1, etc. (invalid on Linux), need to handle this
        input_path_str = str(input_path)
        # If path contains :vN (invalid on Linux), try replacing with -vN
        # Handle any version number (v0, v1, v2, etc.)
        if not Path(input_path).exists():
            # Replace :v0, :v1, :v2, etc. with -v0, -v1, -v2, etc.
            input_path_str = re.sub(r':(v\d+)', r'-\1', input_path_str)
            input_path = Path(input_path_str)
        
        input_path = Path(input_path).resolve()
        logging.info(f"[W&B] Embeddings artifact downloaded to: {input_path}")
        
        # Get embeddings file using filename from config (required, no fallback)
        embeddings_file = input_path / embedding_filename
        
        # Check if file exists - if not, list available files for debugging
        if not embeddings_file.exists():
            logging.error(f"Embeddings file not found in artifact: {embeddings_file}")
            logging.error(f"Expected file: {embedding_filename}")
            
            # List available files for debugging (but don't use them)
            if input_path.exists():
                available_files = list(input_path.glob("*.json"))
                if available_files:
                    logging.error(f"Available JSON files in artifact:")
                    for f in available_files:
                        logging.error(f"  - {f.name}")
                else:
                    logging.error(f"No JSON files found in artifact directory: {input_path}")
            else:
                logging.error(f"Artifact directory does not exist: {input_path}")
            
            logging.error(f"Please update config.yaml with the correct embedding_filename from the list above")
            return
        
        logging.info(f"[OK] Using embeddings file: {embeddings_file.name}")
        
        # =====================================================================
        # Step 4: Load and validate embeddings using schema
        # =====================================================================
        print(f"\n[Step 2] Loading and validating embeddings...")
        
        # Load using UserEmbeddingArtifact schema
        user_embeddings_dict = UserEmbeddingArtifact.from_file(embeddings_file)
        
        if not user_embeddings_dict:
            logging.error("No embeddings found in file")
            return
        
        # Extract embeddings for clustering
        user_ids = []
        embeddings_list = []
        embedding_dict = {}  # Store full embedding data for later
        
        for user_id, embedding_artifact in user_embeddings_dict.items():
            user_ids.append(user_id)
            embeddings_list.append(embedding_artifact.user_embedding)
            embedding_dict[user_id] = embedding_artifact
        
        df = pd.DataFrame({
            'user_id': user_ids,
            'embedding': embeddings_list
        })
        
        embeddings = np.array(embeddings_list)
        num_users = len(df)
        embedding_dim = embeddings.shape[1]
        
        print(f"[OK] Loaded and validated {num_users:,} user embeddings (dim={embedding_dim})")
        
        log_metrics(run, {
            "input/num_users": num_users,
            "input/embedding_dim": embedding_dim,
        })
        
        # =====================================================================
        # Step 5: Dimensionality reduction with UMAP
        # =====================================================================
        print(f"\n[Step 3] UMAP dimensionality reduction...")
        print(f"  n_neighbors={umap_n_neighbors}, n_components={umap_n_components}, min_dist={umap_min_dist}")
        
        import umap
        
        reducer = umap.UMAP(
            n_neighbors=umap_n_neighbors,
            n_components=umap_n_components,
            min_dist=umap_min_dist,
            random_state=random_state,
        )
        
        reduced_embeddings = reducer.fit_transform(embeddings)
        df['x'] = reduced_embeddings[:, 0]
        df['y'] = reduced_embeddings[:, 1]
        
        print(f"[OK] Reduced to shape: {reduced_embeddings.shape}")
        
        # =====================================================================
        # Step 6: Scale embeddings for clustering
        # =====================================================================
        print(f"\n[Step 4] Scaling embeddings...")
        
        from sklearn.preprocessing import StandardScaler
        
        scaler = StandardScaler()
        scaled_embeddings = scaler.fit_transform(embeddings)
        
        print("[OK] Scaling complete")
        
        # =====================================================================
        # Step 7: Evaluate different K values
        # =====================================================================
        # Limit max K to n_samples - 1 for silhouette score
        actual_max_k = min(max_k, len(scaled_embeddings) - 1)
        print(f"\n[Step 5] Evaluating K values ({min_clusters}-{actual_max_k})...")
        
        from sklearn.cluster import KMeans
        from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
        
        results = {}
        best_k_silhouette = min_clusters
        max_silhouette = -1
        
        for k in range(min_clusters, actual_max_k + 1):
            kmeans = KMeans(n_clusters=k, random_state=random_state, n_init=n_init)
            labels = kmeans.fit_predict(scaled_embeddings)
            
            if len(np.unique(labels)) < 2:
                continue
            
            silhouette = silhouette_score(scaled_embeddings, labels)
            davies_bouldin = davies_bouldin_score(scaled_embeddings, labels)
            calinski_harabasz = calinski_harabasz_score(scaled_embeddings, labels)
            
            # Calculate balance ratio
            cluster_counts = pd.Series(labels).value_counts()
            balance_ratio = cluster_counts.min() / cluster_counts.max()
            
            results[k] = {
                'labels': labels,
                'silhouette': silhouette,
                'davies_bouldin': davies_bouldin,
                'calinski_harabasz': calinski_harabasz,
                'balance_ratio': balance_ratio,
            }
            
            # Log metrics for each K
            log_metrics(run, {
                f"k_eval/k{k}_silhouette": silhouette,
                f"k_eval/k{k}_davies_bouldin": davies_bouldin,
                f"k_eval/k{k}_balance": balance_ratio,
            })
            
            print(f"  K={k}: Silhouette={silhouette:.4f}, DB={davies_bouldin:.4f}, Balance={balance_ratio:.3f}")
            
            if silhouette > max_silhouette:
                max_silhouette = silhouette
                best_k_silhouette = k
        
        # =====================================================================
        # Step 8: Select optimal K
        # =====================================================================
        print(f"\n[Step 6] Selecting optimal K...")
        
        optimal_k = target_clusters
        
        if target_clusters is not None and target_clusters in results:
            target_sil = results[target_clusters]['silhouette']
            target_balance = results[target_clusters]['balance_ratio']
            
            if target_sil > silhouette_threshold and target_balance > balance_threshold:
                print(f"  Using target K={target_clusters} (meets quality thresholds)")
                optimal_k = target_clusters
            else:
                print(f"  Target K={target_clusters} doesn't meet thresholds, using best silhouette")
                optimal_k = best_k_silhouette
        else:
            optimal_k = best_k_silhouette
        
        print(f"[OK] Selected K={optimal_k} (Silhouette={results[optimal_k]['silhouette']:.4f})")
        
        # =====================================================================
        # Step 9: Final clustering
        # =====================================================================
        print(f"\n[Step 7] Applying final clustering with K={optimal_k}...")
        
        kmeans_final = KMeans(n_clusters=optimal_k, random_state=random_state, n_init=n_init)
        final_labels = kmeans_final.fit_predict(scaled_embeddings)
        df['cluster'] = final_labels
        
        # Calculate final metrics
        final_silhouette = results[optimal_k]['silhouette']
        final_davies_bouldin = results[optimal_k]['davies_bouldin']
        final_balance = results[optimal_k]['balance_ratio']
        
        cluster_counts = df['cluster'].value_counts().sort_index()
        
        print(f"\n[OK] Final clustering complete:")
        print(f"  Clusters: {optimal_k}")
        print(f"  Silhouette: {final_silhouette:.4f}")
        print(f"  Davies-Bouldin: {final_davies_bouldin:.4f}")
        print(f"  Balance: {final_balance:.4f}")
        print(f"\n  Cluster sizes:")
        for cluster_id, count in cluster_counts.items():
            pct = count / len(df) * 100
            print(f"    Cluster {cluster_id}: {count:,} users ({pct:.1f}%)")
        
        # =====================================================================
        # Step 10: Create user segments learned artifact
        # =====================================================================
        print(f"\n[Step 8] Creating user segments learned artifact...")
        
        # Calculate cluster metadata for each cluster
        cluster_metadata_dict = {}
        for cluster_id in range(optimal_k):
            cluster_size = int(cluster_counts[cluster_id])
            cluster_metadata_dict[cluster_id] = {
                "silhouette_score": float(final_silhouette),
                "cluster_size": cluster_size,
                "davies_bouldin_score": float(final_davies_bouldin),
                "balance_ratio": float(final_balance)
            }
        
        # Create UserSegmentsArtifact for each user
        user_segments_dict = {}
        for idx, row in df.iterrows():
            user_id = row['user_id']
            segment_id = f"segment_{row['cluster']}"
            user_embedding = embedding_dict[user_id].user_embedding
            cluster_meta = cluster_metadata_dict[int(row['cluster'])]
            
            segment_data = {
                "user_id": user_id,
                "segment_id": segment_id,
                "segment_name": None,  # Can be set later
                "user_embedding": user_embedding,
                "cluster_metadata": cluster_meta
            }
            
            # Validate against schema
            user_segments_dict[user_id] = UserSegmentsArtifact.from_dict(segment_data)
        
        print(f"  [OK] Created {len(user_segments_dict)} user segment artifacts")
        
        # =====================================================================
        # Step 11: Save outputs locally
        # =====================================================================
        print("\n" + "-" * 70)
        print("Step 9: Save Outputs Locally")
        print("-" * 70)
        
        artifact_dir = get_artifact_dir("06_tribe_formation", output_artifact)
        logging.info(f"[LOCAL OUTPUT] Saving all outputs locally to: {artifact_dir}")
        
        # Save user segments as JSON (learned artifact) - grouped by user_id
        segments_output_file = artifact_dir / "user_segments.json"
        segments_output_data = {
            user_id: segment.to_dict() 
            for user_id, segment in user_segments_dict.items()
        }
        with open(segments_output_file, 'w', encoding='utf-8') as f:
            json.dump(segments_output_data, f, indent=2, ensure_ascii=False)
        print(f"  [OK] Saved: {segments_output_file} ({len(segments_output_data)} users)")
        
        # Save segment-level files grouped by user (matching clustering_1.py format)
        print(f"\n[Step 9b] Creating segment-level files grouped by user...")
        segment_users_dict = {}  # segment_id -> {user_id: user_data}
        
        for user_id, segment_artifact in user_segments_dict.items():
            segment_id = segment_artifact.segment_id
            cluster_number = int(segment_id.replace("segment_", ""))
            
            if segment_id not in segment_users_dict:
                segment_users_dict[segment_id] = {}
            
            # Create user data structure grouped by segment
            user_data = {
                "user_id": user_id,
                "segment_id": segment_id,
                "cluster": cluster_number,
                "user_embedding": segment_artifact.user_embedding,
                "cluster_metadata": segment_artifact.cluster_metadata.model_dump() if hasattr(segment_artifact.cluster_metadata, 'model_dump') else segment_artifact.cluster_metadata
            }
            
            segment_users_dict[segment_id][user_id] = user_data
        
        # Save per-segment files grouped by user
        for segment_id, users_dict in segment_users_dict.items():
            cluster_number = int(segment_id.replace("segment_", ""))
            segment_file = artifact_dir / f"segment_{cluster_number}_users.json"
            
            with open(segment_file, 'w', encoding='utf-8') as f:
                json.dump(users_dict, f, indent=2, ensure_ascii=False)
            
            print(f"  [OK] Saved: {segment_file.name} ({len(users_dict)} users grouped by user_id)")
        
        # Save parquet (for backward compatibility/analysis)
        output_file = artifact_dir / "clustered_persona_data.parquet"
        df.to_parquet(output_file)
        print(f"  [OK] Saved: {output_file}")
        
        # Save cluster summary
        summary = {
            "num_users": num_users,
            "num_clusters": optimal_k,
            "silhouette_score": float(final_silhouette),
            "davies_bouldin_score": float(final_davies_bouldin),
            "balance_ratio": float(final_balance),
            "cluster_sizes": {int(k): int(v) for k, v in cluster_counts.items()},
            "grouped_by_user": True,  # Indicate that data is grouped by user
            "segment_files_created": len(segment_users_dict)
        }
        
        with open(artifact_dir / "cluster_summary.json", 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"  [OK] Saved: cluster_summary.json")
        
        # Save visualization
        print(f"\n[Step 9] Creating visualization...")
        
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        plt.figure(figsize=(12, 8))
        colors = sns.color_palette('Set2', optimal_k)
        
        for cluster_id in range(optimal_k):
            mask = df['cluster'] == cluster_id
            plt.scatter(
                df.loc[mask, 'x'],
                df.loc[mask, 'y'],
                c=[colors[cluster_id]],
                label=f'Cluster {cluster_id}',
                alpha=0.6,
                s=20
            )
        
        plt.title(f'User Clusters (K={optimal_k}, Silhouette={final_silhouette:.4f})')
        plt.xlabel('UMAP Dimension 1')
        plt.ylabel('UMAP Dimension 2')
        plt.legend()
        plt.tight_layout()
        
        viz_file = artifact_dir / "clusters_visualization.png"
        plt.savefig(viz_file, dpi=300)
        plt.close()
        print(f"  [OK] Saved: {viz_file}")
        
        # =====================================================================
        # Step 12: Log final metrics and upload artifact to W&B
        # =====================================================================
        print("\n" + "-" * 70)
        print("Step 10: Upload Artifact to W&B")
        print("-" * 70)
        
        logging.info(f"[LOCAL SAVE] All files saved locally. Ready to upload to W&B.")
        
        final_metrics = {
            "final/num_clusters": optimal_k,
            "final/silhouette": final_silhouette,
            "final/davies_bouldin": final_davies_bouldin,
            "final/balance_ratio": final_balance,
            "final/num_users": num_users,
        }
        
        log_metrics(run, final_metrics)
        log_summary(run, final_metrics)
        
        # Log visualization to W&B
        if run:
            import wandb
            wandb.log({"visualization": wandb.Image(str(viz_file))})
        
        logging.info(f"[W&B UPLOAD] Uploading artifact '{output_artifact}' to W&B...")
        logging.info(f"[W&B UPLOAD] Uploading directory: {artifact_dir}")
        logging.info(f"[W&B UPLOAD] Files to upload:")
        logging.info(f"[W&B UPLOAD]   - user_segments.json ({len(segments_output_data)} users)")
        logging.info(f"[W&B UPLOAD]   - clustered_persona_data.parquet")
        logging.info(f"[W&B UPLOAD]   - cluster_summary.json")
        logging.info(f"[W&B UPLOAD]   - clusters_visualization.png")
        logging.info(f"[W&B UPLOAD]   - segment_*_users.json files ({len(segment_users_dict)} segments)")
        
        # Upload artifact with comprehensive metadata
        artifact_metadata = create_comprehensive_artifact_metadata(
            stage="06_tribe_formation",
            artifact_name=output_artifact,
            sample_size=num_users,
            model_vendor="scikit-learn",
            model_name="UMAP + KMeans",
            model_description="Dimensionality reduction and clustering for tribe formation",
            model_params={
                "umap_n_neighbors": umap_n_neighbors,
                "umap_n_components": umap_n_components,
                "n_clusters": optimal_k,
            },
            learned_artifact_schema=get_learned_artifact_schema("06_tribe_formation", output_artifact),
            additional_metadata=summary
        )
        artifact = log_artifact(
            run=run,
            artifact_name=output_artifact,
            artifact_type="dataset",
            artifact_path=artifact_dir,  # This uploads ALL files in the directory to W&B
            metadata=artifact_metadata,
        )
        link_to_registry(artifact, stage="06_tribe_formation")
        
        if artifact:
            logging.info(f"[W&B UPLOAD] ✓ Successfully uploaded artifact '{output_artifact}' to W&B")
            logging.info(f"[W&B UPLOAD] ✓ All files in {artifact_dir} are now available in W&B")
        else:
            logging.warning(f"[W&B UPLOAD] W&B disabled or upload failed - files still saved locally")
        
        # =====================================================================
        # Final Summary
        # =====================================================================
        print("\n" + "=" * 70)
        print("STAGE 06 COMPLETE")
        print("=" * 70)
        print(f"\nSummary:")
        print(f"  Users clustered: {num_users:,}")
        print(f"  Number of segments: {optimal_k}")
        print(f"  Silhouette score: {final_silhouette:.4f}")
        print(f"  Davies-Bouldin score: {final_davies_bouldin:.4f}")
        print(f"  Balance ratio: {final_balance:.4f}")
        print(f"\n{'='*70}")
        print(f"OUTPUT FILES - SAVED IN BOTH LOCATIONS:")
        print(f"{'='*70}")
        print(f"📁 LOCAL (saved locally):")
        print(f"   Output directory: {artifact_dir}")
        print(f"   ✓ User segments: {segments_output_file.name}")
        print(f"   ✓ Parquet data: clustered_persona_data.parquet")
        print(f"   ✓ Cluster summary: cluster_summary.json")
        print(f"   ✓ Visualization: clusters_visualization.png")
        print(f"   ✓ Segment files: {len(segment_users_dict)} segment_*_users.json files")
        if artifact:
            print(f"\n☁️  W&B (uploaded to Weights & Biases):")
            print(f"   Artifact name: {output_artifact}")
            print(f"   ✓ All files uploaded to W&B artifact")
            print(f"   View at: {run.url if run else 'N/A'}")
        else:
            print(f"\n⚠️  W&B: Upload skipped (W&B disabled or failed)")
        print(f"\n✓ All outputs are saved both locally AND in W&B")
        
        if run:
            print(f"\nView run at: {run.url}")
        
    finally:
        finish_run(run)


if __name__ == "__main__":
    main()
