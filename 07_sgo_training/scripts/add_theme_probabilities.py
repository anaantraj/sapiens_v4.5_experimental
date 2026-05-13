"""
Script to add theme probabilities (e^logprobs) to SGO training results and baseline predictions.
For each review, calculates e^logprobs for each theme's "yes" and "no" values
and adds them as theme_probs (probabilities before normalization).

Can process a single file or an entire directory recursively.
Supports multiple JSON formats:
- user_predictions format (nested under user_id -> list of reviews)
- Flat dictionary format (review_key -> review_data)
- Handles files with or without theme_logprobs
"""
import json
import math
import sys
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
import shutil

# Try to import wandb utilities
try:
    # Add project root to path - handle both script execution and module import
    script_file = Path(__file__).resolve()
    script_dir = script_file.parent
    # Go up from scripts/ -> sgo_training/ -> 07_sgo_training/ -> project root
    project_root = script_dir.parent.parent.parent.resolve()
    
    # Also try current working directory as fallback
    cwd_root = Path.cwd().resolve()
    
    for root in [project_root, cwd_root]:
        if str(root) not in sys.path:
            sys.path.insert(0, str(root))
    
    from utils.wandb_utils import log_artifact, init_wandb_run, finish_run
    HAS_WANDB = True
except ImportError as e:
    HAS_WANDB = False
    # Only print warning if wandb flag is actually used
    if "--wandb" in sys.argv or "-w" in sys.argv:
        print(f"Warning: wandb utilities not available ({e}), skipping wandb upload")


def calculate_theme_probs(theme_logprobs: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
    """
    Calculate e^logprobs for each theme.
    
    Args:
        theme_logprobs: Dictionary with theme names as keys, each containing:
            - "yes": log probability for yes
            - "no": log probability for no
            - "token_yes": token used for yes
            - "token_no": token used for no
    
    Returns:
        Dictionary with theme names as keys, each containing:
            - "yes": probability (e^logprob_yes)
            - "no": probability (e^logprob_no)
    """
    theme_probs = {}
    
    for theme_name, logprob_data in theme_logprobs.items():
        if not isinstance(logprob_data, dict):
            continue
            
        yes_logprob = logprob_data.get("yes")
        no_logprob = logprob_data.get("no")
        
        # Calculate e^logprob for both yes and no
        yes_prob = math.exp(yes_logprob) if yes_logprob is not None else None
        no_prob = math.exp(no_logprob) if no_logprob is not None else None
        
        theme_probs[theme_name] = {
            "yes": yes_prob,
            "no": no_prob
        }
    
    return theme_probs


def process_prediction_data(prediction_data: Dict[str, Any]) -> bool:
    """
    Process a prediction or corrected_prediction_data object.
    Adds theme_probs to the object if theme_logprobs exists.
    Also handles pre_sgo format where predicted_themes contains e^logprobs.
    
    Args:
        prediction_data: Dictionary containing prediction data
    
    Returns:
        True if theme_probs were added, False otherwise
    """
    added = False
    
    # Case 1: Has theme_logprobs (sgo_training format) - calculate e^logprobs
    if "theme_logprobs" in prediction_data and isinstance(prediction_data["theme_logprobs"], dict):
        theme_probs = calculate_theme_probs(prediction_data["theme_logprobs"])
        prediction_data["theme_probs"] = theme_probs
        added = True
    
    # Case 2: Has predicted_themes that are e^logprobs (pre_sgo format)
    # In pre_sgo files, predicted_themes contains e^logprobs (probabilities before normalization)
    # We need to copy them to theme_probs
    # Only process if theme_logprobs doesn't exist (to avoid double processing)
    if not added and "predicted_themes" in prediction_data and isinstance(prediction_data["predicted_themes"], dict):
        predicted_themes = prediction_data["predicted_themes"]
        
        # In pre_sgo format, predicted_themes contains e^logprob_yes for each theme
        # Copy these values to theme_probs format
        # Since we only have "yes" probabilities, we'll store them as-is
        theme_probs = {}
        for theme_name, yes_prob in predicted_themes.items():
            if yes_prob is not None and isinstance(yes_prob, (int, float)):
                theme_probs[theme_name] = {
                    "yes": float(yes_prob),
                    "no": None  # Can't determine no_prob from predicted_themes alone
                }
        
        if theme_probs:
            prediction_data["theme_probs"] = theme_probs
            added = True
    
    return added


def process_review_user_predictions_format(review: Dict[str, Any]) -> bool:
    """
    Process a single review entry in user_predictions format.
    Handles both the main prediction and correction_journey entries.
    
    Args:
        review: Dictionary containing review data
    
    Returns:
        True if any theme_probs were added, False otherwise
    """
    added = False
    # Process main prediction
    if "prediction" in review and isinstance(review["prediction"], dict):
        if process_prediction_data(review["prediction"]):
            added = True
    
    # Process correction_journey entries
    if "correction_journey" in review and isinstance(review["correction_journey"], list):
        for journey_entry in review["correction_journey"]:
            if isinstance(journey_entry, dict) and "corrected_prediction_data" in journey_entry:
                corrected_data = journey_entry["corrected_prediction_data"]
                if isinstance(corrected_data, dict):
                    if process_prediction_data(corrected_data):
                        added = True
    return added


def process_review_flat_format(review_data: Dict[str, Any]) -> bool:
    """
    Process a review in flat dictionary format (baseline format).
    
    Args:
        review_data: Dictionary containing review data with prediction key
    
    Returns:
        True if theme_probs were added, False otherwise
    """
    added = False
    if "prediction" in review_data and isinstance(review_data["prediction"], dict):
        if process_prediction_data(review_data["prediction"]):
            added = True
    return added


def has_theme_logprobs(data: Dict[str, Any]) -> bool:
    """
    Check if a JSON data structure contains theme_logprobs.
    Supports multiple formats.
    
    Args:
        data: Dictionary containing JSON data
    
    Returns:
        True if theme_logprobs are found, False otherwise
    """
    if not isinstance(data, dict):
        return False
    
    # Format 1: user_predictions structure
    if "user_predictions" in data and isinstance(data["user_predictions"], dict):
        # Check a few reviews to see if they have theme_logprobs
        for user_id, reviews in list(data["user_predictions"].items())[:3]:
            if isinstance(reviews, list):
                for review in reviews[:2]:
                    if isinstance(review, dict):
                        # Check main prediction
                        if "prediction" in review:
                            pred = review["prediction"]
                            if isinstance(pred, dict):
                                if "theme_logprobs" in pred:
                                    return True
                                # Also check for predicted_themes (pre_sgo format where predicted_themes = e^logprobs)
                                if "predicted_themes" in pred and isinstance(pred["predicted_themes"], dict):
                                    return True
                        # Check correction_journey
                        if "correction_journey" in review:
                            journey = review["correction_journey"]
                            if isinstance(journey, list):
                                for entry in journey[:1]:
                                    if isinstance(entry, dict) and "corrected_prediction_data" in entry:
                                        corrected = entry["corrected_prediction_data"]
                                        if isinstance(corrected, dict):
                                            if "theme_logprobs" in corrected:
                                                return True
                                            if "predicted_themes" in corrected and isinstance(corrected["predicted_themes"], dict):
                                                return True
    
    # Format 2: Flat dictionary format (baseline format)
    # Check first few entries
    for key, value in list(data.items())[:5]:
        if isinstance(value, dict):
            if "prediction" in value:
                pred = value["prediction"]
                if isinstance(pred, dict):
                    if "theme_logprobs" in pred:
                        return True
                    if "predicted_themes" in pred and isinstance(pred["predicted_themes"], dict):
                        return True
    
    return False


def determine_output_path(input_file: Path) -> Path:
    """
    Determine the output path based on input file location.
    Handles multiple artifact directory patterns.
    
    Args:
        input_file: Input file path
    
    Returns:
        Output file path
    """
    input_stem = input_file.stem
    input_suffix = input_file.suffix
    input_str = str(input_file)
    
    # Map input directories to output directories
    replacements = {
        "sgo_training_results_v6": "sgo_training_results_v6_with_probs",
        "pre_sgo_context_all_topics_v6_gpt_5_mini_v4": "pre_sgo_context_all_topics_v6_gpt_5_mini_v4_with_probs",
        "baseline_predictions_o3_history_logprobs_v4": "baseline_predictions_o3_history_logprobs_v4_with_probs"
    }
    
    output_str = input_str
    for old_pattern, new_pattern in replacements.items():
        if old_pattern in input_str and "_with_probs" not in input_str:
            output_str = output_str.replace(old_pattern, new_pattern)
            break
    
    if output_str == input_str:
        # Fallback: save in same directory with _with_probs suffix
        output_file = input_file.parent / f"{input_stem}_with_probs{input_suffix}"
    else:
        output_file = Path(output_str).parent / f"{input_stem}_with_probs{input_suffix}"
    
    # Create parent directory if it doesn't exist
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    return output_file


def process_json_file(input_file: Path, output_file: Optional[Path] = None, verbose: bool = True) -> Tuple[bool, Optional[str]]:
    """
    Process a JSON file and add theme_probs to all reviews.
    Supports multiple JSON formats.
    
    Args:
        input_file: Path to input JSON file
        output_file: Path to output JSON file (if None, auto-determines)
        verbose: Whether to print progress messages
    
    Returns:
        Tuple of (success: bool, status: Optional[str])
        - success: True if file was processed successfully
        - status: "processed", "skipped", or "error"
    """
    if verbose:
        print(f"Loading file: {input_file}")
    
    try:
        # Load the JSON file
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Check if already processed
        if "_with_probs" in input_file.stem:
            if verbose:
                print(f"  Skipping {input_file.name}: Already has _with_probs suffix")
            return False, "skipped"
        
        # Check if file has the structure we need
        if not has_theme_logprobs(data):
            if verbose:
                print(f"  Skipping {input_file.name}: No theme_logprobs found")
            return False, "skipped"
        
        # Process all reviews based on format
        total_reviews = 0
        processed_count = 0
        
        # Format 1: user_predictions structure
        if "user_predictions" in data and isinstance(data["user_predictions"], dict):
            for user_id, reviews in data["user_predictions"].items():
                if isinstance(reviews, list):
                    for review in reviews:
                        if isinstance(review, dict):
                            if process_review_user_predictions_format(review):
                                processed_count += 1
                            total_reviews += 1
            
            if verbose:
                print(f"  Processed {processed_count} reviews with logprobs out of {total_reviews} total reviews across {len(data['user_predictions'])} users")
        
        # Format 2: Flat dictionary format
        else:
            for key, review_data in data.items():
                if isinstance(review_data, dict):
                    if process_review_flat_format(review_data):
                        processed_count += 1
                    total_reviews += 1
            
            if verbose:
                print(f"  Processed {processed_count} reviews with logprobs out of {total_reviews} total reviews")
        
        if processed_count == 0:
            if verbose:
                print(f"  Warning: No reviews with theme_logprobs found in file")
            return False, "skipped"
        
        # Determine output file path
        if output_file is None:
            output_file = determine_output_path(input_file)
        
        # Save the processed data
        if verbose:
            print(f"  Saving to: {output_file.name}")
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4, ensure_ascii=False)
        
        if verbose:
            print(f"  ✓ Successfully processed {input_file.name}")
        return True, "processed"
        
    except Exception as e:
        if verbose:
            print(f"  ✗ Error processing {input_file.name}: {e}")
        import traceback
        if verbose:
            traceback.print_exc()
        return False, "error"


def find_json_files(directory: Path, skip_dirs: Optional[List[str]] = None) -> List[Path]:
    """
    Recursively find all JSON files in a directory.
    
    Args:
        directory: Root directory to search
        skip_dirs: List of directory names to skip (e.g., ['_cache', '_journey'])
    
    Returns:
        List of JSON file paths
    """
    if skip_dirs is None:
        skip_dirs = []
    
    json_files = []
    for path in directory.rglob("*.json"):
        # Skip files in excluded directories
        if any(skip_dir in path.parts for skip_dir in skip_dirs):
            continue
        # Skip files that already have _with_probs
        if "_with_probs" in path.stem:
            continue
        json_files.append(path)
    
    return sorted(json_files)


def process_directory(directory: Path, skip_dirs: Optional[List[str]] = None, upload_to_wandb: bool = False) -> None:
    """
    Process all JSON files in a directory recursively.
    
    Args:
        directory: Directory to process
        skip_dirs: List of directory names to skip
        upload_to_wandb: Whether to upload results to wandb
    """
    if not directory.exists():
        print(f"Error: Directory not found: {directory}")
        return
    
    if not directory.is_dir():
        print(f"Error: Path is not a directory: {directory}")
        return
    
    print(f"Searching for JSON files in: {directory}")
    
    # Find all JSON files
    json_files = find_json_files(directory, skip_dirs)
    
    if not json_files:
        print("No JSON files found to process.")
        return
    
    print(f"Found {len(json_files)} JSON files to process.\n")
    
    # Initialize wandb run if requested
    run = None
    if upload_to_wandb and HAS_WANDB:
        try:
            artifact_name = directory.name
            run = init_wandb_run(
                run_name=f"add_theme_probs_{artifact_name}",
                stage="07_sgo_training",
                config={
                    "description": "Adding theme probabilities (e^logprobs) to predictions",
                    "input_directory": str(directory),
                    "total_files": len(json_files)
                }
            )
        except Exception as e:
            print(f"Warning: Could not initialize wandb run: {e}")
    
    # Process each file
    processed_count = 0
    skipped_count = 0
    error_count = 0
    output_files = []
    
    for i, json_file in enumerate(json_files, 1):
        print(f"[{i}/{len(json_files)}] Processing: {json_file.relative_to(directory)}")
        success, status = process_json_file(json_file, verbose=False)
        
        if status == "processed":
            processed_count += 1
            # Track output file for wandb upload
            output_file = determine_output_path(json_file)
            if output_file.exists():
                output_files.append(output_file)
        elif status == "skipped":
            skipped_count += 1
        elif status == "error":
            error_count += 1
    
    print(f"\n{'='*60}")
    print(f"Processing complete!")
    print(f"  Processed: {processed_count} files")
    print(f"  Skipped: {skipped_count} files")
    print(f"  Errors: {error_count} files")
    print(f"{'='*60}")
    
    # Upload to wandb if requested
    if upload_to_wandb and HAS_WANDB and run and processed_count > 0:
        print(f"\n{'='*60}")
        print("Uploading to WandB...")
        print(f"{'='*60}")
        try:
            # Determine output directory by checking where files were saved
            if output_files:
                # Get the output directory from the first processed file
                output_file = output_files[0]
                # Find the root _with_probs directory
                parts = output_file.parts
                output_dir = None
                for i, part in enumerate(parts):
                    if "_with_probs" in part:
                        output_dir = Path(*parts[:i+1])
                        break
                
                if output_dir and output_dir.exists():
                    artifact_name = f"{directory.name}_with_probs"
                    print(f"Uploading artifact: {artifact_name}")
                    print(f"From directory: {output_dir}")
                    artifact = log_artifact(
                        run=run,
                        artifact_name=artifact_name,
                        artifact_type="dataset",
                        artifact_path=output_dir,
                        metadata={
                            "processed_files": processed_count,
                            "skipped_files": skipped_count,
                            "error_files": error_count,
                            "total_files": len(json_files),
                            "input_directory": str(directory)
                        }
                    )
                    if artifact:
                        print(f"\n✅ Successfully uploaded {artifact_name} to WandB")
                        print(f"   Artifact URL: {run.url if hasattr(run, 'url') else 'N/A'}")
                    else:
                        print(f"\n⚠️  Failed to upload {artifact_name} to WandB")
                else:
                    print(f"Warning: Could not find output directory for wandb upload")
                    if output_files:
                        print(f"  First output file: {output_files[0]}")
            else:
                print("Warning: No output files to upload")
        except Exception as e:
            print(f"❌ Error uploading to wandb: {e}")
            import traceback
            traceback.print_exc()
    
    if run:
        try:
            finish_run(run)
        except Exception as e:
            print(f"Warning: Could not finish wandb run: {e}")


def main():
    """Main function to process files or directories."""
    import sys
    
    # Default directories to process
    default_dirs = [
        Path("/media/samanvitha/Seagate HDD/vectorial_sapiens_v4/07_sgo_training/artifacts/sgo_training_results_v6"),
        Path("/media/samanvitha/Seagate HDD/vectorial_sapiens_v4/07_sgo_training/artifacts/pre_sgo_context_all_topics_v6_gpt_5_mini_v4"),
        Path("/media/samanvitha/Seagate HDD/vectorial_sapiens_v4/09_baselines/artifacts/baseline_predictions_o3_history_logprobs_v4")
    ]
    
    # Check for upload flag
    upload_to_wandb = "--wandb" in sys.argv or "-w" in sys.argv
    if upload_to_wandb:
        sys.argv = [arg for arg in sys.argv if arg not in ["--wandb", "-w"]]
    
    if len(sys.argv) > 1:
        input_path = Path(sys.argv[1])
    else:
        # Process all default directories
        print("No path specified, processing all default artifact directories...\n")
        for default_dir in default_dirs:
            if default_dir.exists():
                print(f"\n{'='*80}")
                print(f"Processing: {default_dir}")
                print(f"{'='*80}\n")
                skip_dirs = ['_cache', '_journey', '_delta', '_failures', '_memory', 'cluster_logs']
                process_directory(default_dir, skip_dirs, upload_to_wandb=upload_to_wandb)
            else:
                print(f"Skipping {default_dir}: Directory not found")
        return
    
    if not input_path.exists():
        print(f"Error: Path not found: {input_path}")
        return
    
    # Determine if it's a file or directory
    if input_path.is_file():
        # Process single file
        success, status = process_json_file(input_path)
        if not success:
            print(f"Failed to process file: {status}")
    elif input_path.is_dir():
        # Process directory
        skip_dirs = ['_cache', '_journey', '_delta', '_failures', '_memory', 'cluster_logs']
        process_directory(input_path, skip_dirs, upload_to_wandb=upload_to_wandb)
    else:
        print(f"Error: Path is neither a file nor a directory: {input_path}")


if __name__ == "__main__":
    main()
