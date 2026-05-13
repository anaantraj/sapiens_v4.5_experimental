#!/usr/bin/env python3
"""
Add Predicted Themes and Recalculate Recall
============================================

This script:
1. Loads final predictions output files
2. Loads original training data to get predicted_themes
3. Adds predicted_themes to actual section if missing
4. Recalculates recall metrics using predicted_themes (not topic_probabilities)
5. Saves updated files

Usage:
    python3 07_sgo_training/scripts/add_predicted_themes_and_recalculate_recall.py \
        --input-dir 07_sgo_training/artifacts/sgo_train_final_predictions_user_seed_memory \
        --training-data-artifact train_set_reviews_with_topics_v3:latest \
        --output-dir 07_sgo_training/artifacts/sgo_train_final_predictions_refined_chars_with_recall
"""

import sys
import json
import logging
import argparse
from pathlib import Path
from typing import Dict, List, Any, Optional
from collections import defaultdict

# Add project root to path
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Add prediction_lib to path
prediction_lib_path = str(project_root / "07_pre_sgo_predictions" / "scripts")
if prediction_lib_path not in sys.path:
    sys.path.insert(0, prediction_lib_path)

from utils.wandb_utils import use_artifact, init_wandb_run, finish_run, load_config, get_stage_config
from prediction_lib.metrics import calculate_review_metrics

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


def load_training_data_asin_lookup(training_data_artifact: str, run) -> Dict[str, Dict[str, Any]]:
    """
    Load training data and create ASIN lookup for predicted_themes.
    
    Returns:
        Dictionary mapping ASIN -> review data with predicted_themes
    """
    logging.info(f"Loading training data from artifact: {training_data_artifact}")
    
    # Download artifact
    training_data_path = use_artifact(run, training_data_artifact, artifact_type="dataset")
    if not training_data_path:
        raise ValueError(f"Could not download artifact: {training_data_artifact}")
    
    training_data_path = Path(training_data_path)
    
    # Find the JSON file
    json_files = list(training_data_path.glob("*.json"))
    if not json_files:
        raise ValueError(f"No JSON files found in {training_data_path}")
    
    training_file = json_files[0]
    logging.info(f"Loading training data from: {training_file}")
    
    with open(training_file, 'r', encoding='utf-8') as f:
        training_data = json.load(f)
    
    # Create ASIN lookup
    asin_lookup = {}
    total_reviews = 0
    reviews_with_themes = 0
    
    if isinstance(training_data, dict):
        # Check if it's user-based structure (user_id -> dict with 'reviews' key)
        # This is the standard format: {user_id: {'reviews': [...]}}
        for user_id, user_data in training_data.items():
            if not isinstance(user_data, dict):
                continue
            
            # Get reviews list
            reviews = user_data.get('reviews', [])
            if not isinstance(reviews, list):
                # If user_data itself is a list, use it
                if isinstance(user_data, list):
                    reviews = user_data
                else:
                    continue
            
            for review in reviews:
                if not isinstance(review, dict):
                    continue
                
                total_reviews += 1
                asin = review.get('asin') or review.get('product_asin')
                if not asin:
                    continue
                
                # Get predicted_themes from review
                # Can be list or dict format
                predicted_themes = review.get('predicted_themes', [])
                if not predicted_themes:
                    predicted_themes = review.get('themes', [])
                
                # Handle dict format (convert to list of theme names)
                if isinstance(predicted_themes, dict):
                    # Extract theme names where value is truthy or > threshold
                    predicted_themes = [theme for theme, val in predicted_themes.items() if val]
                
                if predicted_themes and len(predicted_themes) > 0:
                    asin_lookup[str(asin).upper()] = {
                        'predicted_themes': predicted_themes if isinstance(predicted_themes, list) else list(predicted_themes),
                        'review': review
                    }
                    reviews_with_themes += 1
    
    logging.info(f"Loaded {len(asin_lookup)} reviews with predicted_themes from {total_reviews} total reviews")
    return asin_lookup


def process_summary_file(
    file_path: Path,
    asin_lookup: Dict[str, Dict[str, Any]],
    output_path: Optional[Path] = None
) -> tuple:
    """
    Process a single summary file:
    1. Add predicted_themes to actual section if missing
    2. Recalculate recall metrics using predicted_themes
    
    Returns:
        (num_updated, num_skipped, num_errors)
    """
    logging.info(f"Processing file: {file_path}")
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        logging.error(f"Error loading file {file_path}: {e}")
        return (0, 0, 1)
    
    num_updated = 0
    num_skipped = 0
    num_errors = 0
    
    if 'user_predictions' not in data:
        logging.warning(f"File {file_path} does not have 'user_predictions' key")
        return (0, 0, 0)
    
    # Process each user's reviews
    for user_id, user_reviews in data['user_predictions'].items():
        if not isinstance(user_reviews, list):
            continue
        
        for review in user_reviews:
            try:
                actual = review.get('actual', {})
                if not actual:
                    continue
                
                asin = review.get('asin') or actual.get('asin')
                if not asin:
                    num_skipped += 1
                    continue
                
                asin_key = str(asin).upper()
                
                # Check if predicted_themes is missing or empty
                current_predicted_themes = actual.get('predicted_themes', [])
                if not current_predicted_themes or (isinstance(current_predicted_themes, list) and len(current_predicted_themes) == 0):
                    # Try to get from training data
                    if asin_key in asin_lookup:
                        predicted_themes = asin_lookup[asin_key]['predicted_themes']
                        actual['predicted_themes'] = predicted_themes
                        num_updated += 1
                    else:
                        num_skipped += 1
                        continue
                else:
                    # Already has predicted_themes, skip
                    continue
                
                # Recalculate recall metrics using predicted_themes
                prediction = review.get('prediction', {})
                if prediction:
                    # Calculate metrics with predicted_themes as actual
                    metrics = calculate_review_metrics(
                        prediction,
                        {
                            'rating': actual.get('rating'),
                            'sentiment': actual.get('sentiment'),
                            'predicted_themes': actual.get('predicted_themes', [])
                        }
                    )
                    
                    if metrics:
                        review['metrics'] = metrics
                        logging.debug(f"Recalculated metrics for review {asin}")
                
            except Exception as e:
                logging.error(f"Error processing review in {file_path}: {e}")
                num_errors += 1
                continue
    
    # Save updated file
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        logging.info(f"Saved updated file to: {output_path}")
    
    return (num_updated, num_skipped, num_errors)


def main():
    parser = argparse.ArgumentParser(description='Add predicted_themes and recalculate recall')
    parser.add_argument('--input-dir', type=str, required=True, help='Input directory with summary files')
    parser.add_argument('--training-data-artifact', type=str, required=True, help='Training data artifact name (e.g., train_set_reviews_with_topics_v3:latest)')
    parser.add_argument('--output-dir', type=str, help='Output directory (default: input_dir with _with_recall suffix)')
    parser.add_argument('--cluster', type=str, help='Process only specific cluster (e.g., cluster_4)')
    parser.add_argument('--micro', type=str, help='Process only specific micro cluster (e.g., micro_15)')
    
    args = parser.parse_args()
    
    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        logging.error(f"Input directory does not exist: {input_dir}")
        return
    
    # Determine output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = Path(str(input_dir) + "_with_recall")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    logging.info(f"Output directory: {output_dir}")
    
    # Initialize W&B run
    run = init_wandb_run(
        run_name="add_predicted_themes_recalculate_recall",
        stage="07_sgo_training",
        config={
            "input_dir": str(input_dir),
            "training_data_artifact": args.training_data_artifact,
            "output_dir": str(output_dir)
        }
    )
    
    try:
        # Load training data
        asin_lookup = load_training_data_asin_lookup(args.training_data_artifact, run)
        
        # Find all summary files
        pattern = "*_summary_enhanced_persona_micro_cluster_accuracy.json"
        if args.cluster:
            cluster_dir = input_dir / args.cluster
            if cluster_dir.exists():
                if args.micro:
                    files = list(cluster_dir.glob(f"{args.micro}{pattern}"))
                else:
                    files = list(cluster_dir.glob(pattern))
            else:
                logging.error(f"Cluster directory not found: {cluster_dir}")
                return
        else:
            files = list(input_dir.rglob(pattern))
        
        if not files:
            logging.warning(f"No summary files found matching pattern: {pattern}")
            return
        
        logging.info(f"Found {len(files)} summary files to process")
        
        # Process each file
        total_updated = 0
        total_skipped = 0
        total_errors = 0
        
        for file_path in files:
            # Determine output path
            rel_path = file_path.relative_to(input_dir)
            output_path = output_dir / rel_path
            
            num_updated, num_skipped, num_errors = process_summary_file(
                file_path, asin_lookup, output_path
            )
            
            total_updated += num_updated
            total_skipped += num_skipped
            total_errors += num_errors
        
        logging.info(f"\n{'='*60}")
        logging.info(f"Processing complete!")
        logging.info(f"  Total files processed: {len(files)}")
        logging.info(f"  Reviews updated: {total_updated}")
        logging.info(f"  Reviews skipped: {total_skipped}")
        logging.info(f"  Errors: {total_errors}")
        logging.info(f"  Output directory: {output_dir}")
        logging.info(f"{'='*60}")
        
    except Exception as e:
        logging.error(f"Error in main execution: {e}", exc_info=True)
        raise
    finally:
        if run:
            finish_run(run)


if __name__ == "__main__":
    main()

