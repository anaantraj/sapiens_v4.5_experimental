"""
Script to add topic probabilities (e^logprobs) from topic_logprobs.
Downloads train_set_topic_predictions:latest from WandB,
calculates e^logprobs for each topic (before normalization),
and saves to a new file with topic_probs_before_normalization field.
"""
import json
import math
import sys
import os
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple

# Try to import wandb utilities
try:
    script_dir = Path(__file__).parent.resolve()
    # Go up from scripts/ -> sgo_training/ -> 07_sgo_training/ -> project_root
    project_root = script_dir.parent.parent.parent.resolve()
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    # Also add current working directory
    cwd = Path.cwd().resolve()
    if str(cwd) not in sys.path:
        sys.path.insert(0, str(cwd))
    from utils.wandb_utils import download_artifact, init_wandb_run, log_artifact, finish_run
    HAS_WANDB = True
except ImportError as e:
    HAS_WANDB = False
    print(f"Warning: WandB utilities not available ({e})")


def calculate_topic_probs_from_logprobs(topic_logprobs: Dict[str, Dict[str, float]]) -> Dict[str, Dict[str, float]]:
    """
    Calculate e^logprobs for each topic from topic_logprobs.
    
    Args:
        topic_logprobs: Dictionary with topic names as keys, each containing:
            - "logprob_yes": log probability for yes
            - "logprob_no": log probability for no
    
    Returns:
        Dictionary with topic names as keys, each containing:
            - "prob_yes": probability (e^logprob_yes) - before normalization
            - "prob_no": probability (e^logprob_no) - before normalization
    """
    topic_probs = {}
    
    for topic_name, logprob_data in topic_logprobs.items():
        if not isinstance(logprob_data, dict):
            continue
        
        # Handle different key names
        yes_logprob = logprob_data.get("logprob_yes") or logprob_data.get("yes")
        no_logprob = logprob_data.get("logprob_no") or logprob_data.get("no")
        
        # Calculate e^logprob for both yes and no (before normalization)
        yes_prob = math.exp(yes_logprob) if yes_logprob is not None else None
        no_prob = math.exp(no_logprob) if no_logprob is not None else None
        
        topic_probs[topic_name] = {
            "prob_yes": yes_prob,
            "prob_no": no_prob
        }
    
    return topic_probs


def process_review(review: Dict[str, Any]) -> bool:
    """
    Process a single review to add topic_probs_before_normalization.
    
    Args:
        review: Review dictionary
    
    Returns:
        True if topic_probs were added, False otherwise
    """
    if "topic_logprobs" not in review:
        return False
    
    topic_logprobs = review["topic_logprobs"]
    if not isinstance(topic_logprobs, dict):
        return False
    
    # Calculate e^logprobs
    topic_probs = calculate_topic_probs_from_logprobs(topic_logprobs)
    
    if not topic_probs:
        return False
    
    # Add to review with descriptive name
    review["topic_probs_before_normalization"] = topic_probs
    
    return True


def process_data(data: Any) -> Tuple[int, int]:
    """
    Process data structure to add topic_probs_before_normalization.
    Handles both list and dict formats.
    
    Args:
        data: Data structure (list of reviews or dict with reviews)
    
    Returns:
        Tuple of (processed_count, total_count)
    """
    processed_count = 0
    total_count = 0
    
    # Handle list format
    if isinstance(data, list):
        for review in data:
            total_count += 1
            if isinstance(review, dict):
                if process_review(review):
                    processed_count += 1
    
    # Handle dict format (user_id -> reviews or similar)
    elif isinstance(data, dict):
        for key, value in data.items():
            if isinstance(value, list):
                # Direct list of reviews: {user_id: [...]}
                for review in value:
                    total_count += 1
                    if isinstance(review, dict):
                        if process_review(review):
                            processed_count += 1
            elif isinstance(value, dict):
                # Nested structure: {user_id: {reviews: [...]}}
                reviews = value.get("reviews", [])
                if isinstance(reviews, list):
                    for review in reviews:
                        total_count += 1
                        if isinstance(review, dict):
                            if process_review(review):
                                processed_count += 1
                elif isinstance(value, dict) and "topic_logprobs" in value:
                    # Single review dict
                    total_count += 1
                    if process_review(value):
                        processed_count += 1
    
    return processed_count, total_count


def main():
    """Main function to process topic logprobs and add probabilities."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Add topic probabilities (e^logprobs) from topic_logprobs")
    parser.add_argument(
        "--artifact",
        type=str,
        default="train_set_topic_predictions:latest",
        help="WandB artifact name (default: train_set_topic_predictions:latest)"
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default=None,
        help="Output file path (default: creates _with_topic_probs.json)"
    )
    parser.add_argument(
        "--wandb",
        action="store_true",
        help="Upload results to WandB"
    )
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("Adding Topic Probabilities from Logprobs")
    print("=" * 80)
    
    # Download artifact from WandB
    print(f"\nDownloading artifact from WandB: {args.artifact}")
    try:
        if not HAS_WANDB:
            print("Error: WandB utilities not available")
            return
        
        artifact_name = args.artifact.split(":")[0]
        artifact_version = args.artifact.split(":")[1] if ":" in args.artifact else "latest"
        
        artifact_path = download_artifact(
            artifact_name=artifact_name,
            artifact_type="dataset",
            version=artifact_version
        )
        
        if not artifact_path:
            print("Error: Failed to download artifact")
            return
        
        # Handle path - WandB may return path with :vN which is invalid on Linux
        artifact_path_str = str(artifact_path)
        if ":" in artifact_path_str:
            import re
            artifact_path_str = re.sub(r':(v\d+)', r'-\1', artifact_path_str)
        
        artifact_path = Path(artifact_path_str).resolve()
        print(f"  Artifact downloaded to: {artifact_path}")
        
        # Find the JSON file
        json_file = None
        if artifact_path.is_file() and artifact_path.suffix == ".json":
            json_file = artifact_path
        elif artifact_path.is_dir():
            # Try common filenames (prioritize processed_train_predictions.json)
            for filename in [
                "processed_train_predictions.json",
                "train_set_topic_predictions.json",
                "topic_predictions.json",
                "predictions.json",
                "reviews.json",
                "data.json"
            ]:
                candidate = artifact_path / filename
                if candidate.exists():
                    json_file = candidate
                    break
            
            # If still not found, look for any JSON file in root directory
            if not json_file:
                json_files = [f for f in artifact_path.glob("*.json") if f.is_file()]
                if json_files:
                    # Prefer files without "_with_" in name (original file)
                    original_files = [f for f in json_files if "_with_" not in f.name]
                    if original_files:
                        json_file = original_files[0]
                    else:
                        json_file = json_files[0]
        
        if not json_file or not json_file.exists():
            print(f"Error: Could not find JSON file in artifact")
            print(f"  Searched in: {artifact_path}")
            if artifact_path.is_dir():
                print(f"  Directory contents: {list(artifact_path.iterdir())[:10]}")
            return
        
        print(f"  Found JSON file: {json_file}")
        
    except Exception as e:
        print(f"Error downloading artifact: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Load data
    print(f"\nLoading data from {json_file}...")
    try:
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Count total items
        if isinstance(data, list):
            print(f"  Loaded {len(data)} reviews")
        elif isinstance(data, dict):
            total_reviews = sum(
                len(v.get("reviews", [])) if isinstance(v, dict) and "reviews" in v
                else (len(v) if isinstance(v, list) else 1)
                for v in data.values()
            )
            print(f"  Loaded data with {len(data)} users, ~{total_reviews} reviews")
    except Exception as e:
        print(f"Error loading JSON file: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Process data
    print(f"\nProcessing reviews to add topic_probs_before_normalization...")
    processed_count, total_count = process_data(data)
    
    print(f"\n{'=' * 80}")
    print(f"Processing complete!")
    print(f"  Total reviews: {total_count}")
    print(f"  Processed with topic_probs: {processed_count}")
    print(f"{'=' * 80}")
    
    # Determine output path
    if args.output_file:
        output_path = Path(args.output_file)
    else:
        output_path = json_file.parent / f"{json_file.stem}_with_topic_probs.json"
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save processed data
    print(f"\nSaving processed data to: {output_path}")
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"  ✓ Successfully saved to {output_path}")
    except Exception as e:
        print(f"  ✗ Error saving file: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Upload to WandB if requested
    if args.wandb and HAS_WANDB:
        print(f"\nUploading to WandB...")
        try:
            run = init_wandb_run(
                run_name="add_topic_probs_from_logprobs",
                stage="07_sgo_training",
                config={
                    "description": "Adding topic probabilities (e^logprobs) from topic_logprobs",
                    "input_artifact": args.artifact,
                    "processed_reviews": processed_count,
                    "total_reviews": total_count
                }
            )
            
            if run:
                output_artifact_name = f"{artifact_name}_with_topic_probs"
                artifact = log_artifact(
                    run=run,
                    artifact_name=output_artifact_name,
                    artifact_type="dataset",
                    artifact_path=output_path,
                    metadata={
                        "processed_reviews": processed_count,
                        "total_reviews": total_count,
                        "description": "Topic probabilities (e^logprobs) before normalization"
                    }
                )
                print(f"  ✓ Uploaded {output_artifact_name} to WandB")
                finish_run(run)
        except Exception as e:
            print(f"  ✗ Error uploading to WandB: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()

