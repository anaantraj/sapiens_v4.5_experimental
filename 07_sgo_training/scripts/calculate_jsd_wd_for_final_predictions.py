#!/usr/bin/env python3
"""
Calculate JSD and WD metrics for final prediction artifacts:
- sgo_train_final_predictions_seed_backstory_only
- sgo_train_final_predictions_user_seed_backstory

Usage:
    python 07_sgo_training/scripts/calculate_jsd_wd_for_final_predictions.py
    python 07_sgo_training/scripts/calculate_jsd_wd_for_final_predictions.py --only-cluster cluster_4
"""

import sys
import json
import logging
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict
from scipy.stats import entropy
from scipy.stats import wasserstein_distance
import argparse
from tqdm import tqdm

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

EPSILON = 1e-10


def compute_jsd(P: np.ndarray, Q: np.ndarray, epsilon: float = EPSILON) -> float:
    """Compute Jensen-Shannon Divergence between two probability distributions."""
    P = P + epsilon
    Q = Q + epsilon
    P = P / P.sum()
    Q = Q / Q.sum()
    M = 0.5 * (P + Q)
    kl_pm = entropy(P, M, base=2)
    kl_qm = entropy(Q, M, base=2)
    jsd = 0.5 * kl_pm + 0.5 * kl_qm
    return float(jsd)


def normalize_distribution(theme_dict: Dict[str, float]) -> Dict[str, float]:
    """Normalize a theme probability distribution to sum to 1.0."""
    if not theme_dict:
        return {}
    total = sum(theme_dict.values())
    if total == 0:
        return {}
    return {theme: prob / total for theme, prob in theme_dict.items()}


def align_distributions(
    actual_themes: Dict[str, float],
    predicted_themes: Dict[str, float]
) -> Tuple[np.ndarray, np.ndarray]:
    """Align two theme distributions to the same set of themes and normalize."""
    all_themes = sorted(set(actual_themes.keys()) | set(predicted_themes.keys()))
    if not all_themes:
        return np.array([]), np.array([])
    
    actual_array = np.array([actual_themes.get(theme, 0.0) for theme in all_themes])
    predicted_array = np.array([predicted_themes.get(theme, 0.0) for theme in all_themes])
    
    actual_sum = actual_array.sum()
    predicted_sum = predicted_array.sum()
    
    if actual_sum > EPSILON:
        actual_array = actual_array / actual_sum
    else:
        actual_array = np.zeros_like(actual_array)
    
    if predicted_sum > EPSILON:
        predicted_array = predicted_array / predicted_sum
    else:
        predicted_array = np.zeros_like(predicted_array)
    
    return actual_array, predicted_array


def calculate_wasserstein_1_distance(
    actual_array: np.ndarray,
    predicted_array: np.ndarray
) -> float:
    """Calculate Wasserstein-1 distance (Earth Mover's Distance) between two distributions."""
    if len(actual_array) == 0 or len(predicted_array) == 0:
        return 0.0
    
    # For discrete distributions, use scipy's wasserstein_distance
    # Create positions (0, 1, 2, ...) for each theme
    positions = np.arange(len(actual_array))
    
    # Calculate Wasserstein-1 distance
    wd = wasserstein_distance(positions, positions, actual_array, predicted_array)
    
    return float(wd)


def process_review(review: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Process a single review and calculate JSD and WD."""
    # Get actual (ground truth) themes - use topic_probabilities
    actual_data = review.get('actual', {})
    actual_themes = actual_data.get('topic_probabilities', {})
    
    # If not found, try at top level (for deltas files)
    if not actual_themes:
        actual_themes = review.get('topic_probabilities', {})
    
    if not actual_themes:
        return None
    
    # Get predicted themes
    prediction = review.get('prediction', {})
    predicted_themes = prediction.get('predicted_themes', {})
    
    if not predicted_themes:
        return None
    
    # Normalize distributions
    actual_themes = normalize_distribution(actual_themes)
    predicted_themes = normalize_distribution(predicted_themes)
    
    if not actual_themes or not predicted_themes:
        return None
    
    # Align distributions
    actual_array, predicted_array = align_distributions(actual_themes, predicted_themes)
    
    if len(actual_array) == 0:
        return None
    
    # Calculate JSD
    jsd = compute_jsd(actual_array, predicted_array)
    
    # Calculate WD
    wd = calculate_wasserstein_1_distance(actual_array, predicted_array)
    
    # Build result
    result = {
        'asin': review.get('asin'),
        'user_id': review.get('user_id'),
        'review_text': review.get('actual', {}).get('review_text', ''),
        'category': review.get('category'),
        'jsd': jsd,
        'wd': wd,
        'actual_themes': actual_themes,
        'predicted_themes': predicted_themes
    }
    
    return result


def process_artifact_directory(artifact_path: Path, filter_cluster: str = None, 
                               max_reviews: int = None) -> List[Dict[str, Any]]:
    """Process all JSON files in an artifact directory."""
    all_results = []
    
    # First, try to process deltas files (they have topic_probabilities)
    deltas_files = []
    deltas_dir = artifact_path / "deltas"
    if deltas_dir.exists():
        for json_file in deltas_dir.rglob("*_all_reviews_deltas.json"):
            if filter_cluster and filter_cluster not in str(json_file):
                continue
            deltas_files.append(json_file)
    
    if deltas_files:
        logger.info(f"Found {len(deltas_files)} deltas files to process")
        for json_file in tqdm(deltas_files, desc="Processing deltas files"):
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                cluster_id = data.get('cluster_id')
                micro_id = data.get('micro_cluster_id')
                
                # Process deltas (each delta has prediction and actual with topic_probabilities)
                deltas = data.get('deltas', [])
                for delta in deltas:
                    if max_reviews and len(all_results) >= max_reviews:
                        break
                    
                    # Create review-like structure from delta
                    review = {
                        'prediction': delta.get('prediction', {}),
                        'actual': delta.get('actual', {}),
                        'category': delta.get('category'),
                        'user_id': delta.get('user_id')
                    }
                    
                    result = process_review(review)
                    if result:
                        result['cluster_id'] = cluster_id
                        result['micro_id'] = micro_id
                        result['user_id'] = delta.get('user_id')
                        result['category'] = delta.get('category')
                        result['product_description'] = delta.get('product_description', '')
                        all_results.append(result)
                
                if max_reviews and len(all_results) >= max_reviews:
                    break
                    
            except Exception as e:
                logger.warning(f"Error processing deltas file {json_file}: {e}")
                continue
    
    # If no deltas files or not enough results, try summary files
    if not all_results or (max_reviews and len(all_results) < max_reviews):
        json_files = []
        for json_file in artifact_path.rglob("*.json"):
            # Only process summary files (not cache, deltas, etc.)
            if "_summary" in str(json_file) and "_test" in str(json_file):
                if filter_cluster and filter_cluster not in str(json_file):
                    continue
                if "deltas" in str(json_file):
                    continue
                json_files.append(json_file)
        
        if json_files:
            logger.info(f"Found {len(json_files)} summary JSON files to process")
            
            for json_file in tqdm(json_files, desc="Processing summary files"):
                try:
                    with open(json_file, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    
                    # Extract cluster and micro info from path
                    cluster_id = None
                    micro_id = None
                    path_parts = json_file.parts
                    for part in path_parts:
                        if part.startswith('cluster_'):
                            cluster_id = part
                        elif part.startswith('micro_'):
                            micro_id = part.replace('_test_summary_enhanced_persona_micro_cluster_accuracy.json', '')
                            if not micro_id.startswith('micro_'):
                                # Try to extract from filename
                                if '_test_summary' in json_file.name:
                                    micro_id = json_file.name.split('_test_summary')[0]
                    
                    # Process user predictions
                    user_predictions = data.get('user_predictions', {})
                    for user_id, user_data in user_predictions.items():
                        reviews = user_data.get('reviews', [])
                        for review in reviews:
                            if max_reviews and len(all_results) >= max_reviews:
                                break
                            
                            result = process_review(review)
                            if result:
                                result['cluster_id'] = cluster_id
                                result['micro_id'] = micro_id
                                result['user_id'] = user_id
                                all_results.append(result)
                        
                        if max_reviews and len(all_results) >= max_reviews:
                            break
                    
                    if max_reviews and len(all_results) >= max_reviews:
                        break
                        
                except Exception as e:
                    logger.warning(f"Error processing {json_file}: {e}")
                    continue
    
    return all_results


def main():
    parser = argparse.ArgumentParser(description='Calculate JSD and WD for final prediction artifacts')
    parser.add_argument('--only-cluster', type=str, help='Process only this specific cluster (e.g., cluster_4)')
    parser.add_argument('--max-reviews', type=int, help='Maximum number of reviews to process per artifact')
    args = parser.parse_args()
    
    # Define artifacts to process
    artifacts = [
        {
            'name': 'sgo_train_final_predictions_seed_backstory_only',
            'path': project_root / '07_sgo_training' / 'artifacts' / 'sgo_train_final_predictions_seed_backstory_only'
        },
        {
            'name': 'sgo_train_final_predictions_user_seed_backstory',
            'path': project_root / '07_sgo_training' / 'artifacts' / 'sgo_train_final_predictions_user_seed_backstory'
        }
    ]
    
    # Create output directory
    output_dir = project_root / '07_sgo_training' / 'artifacts' / 'jsd_wd_analysis'
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory: {output_dir}")
    
    # Process each artifact
    all_results = {}
    for artifact in artifacts:
        logger.info(f"\n{'='*80}")
        logger.info(f"Processing {artifact['name']}")
        logger.info(f"{'='*80}")
        
        if not artifact['path'].exists():
            logger.error(f"Artifact directory not found: {artifact['path']}")
            continue
        
        results = process_artifact_directory(
            artifact_path=artifact['path'],
            filter_cluster=args.only_cluster,
            max_reviews=args.max_reviews
        )
        
        if not results:
            logger.warning(f"No results for {artifact['name']}")
            continue
        
        all_results[artifact['name']] = results
        
        # Save results
        output_file = output_dir / f"{artifact['name']}_jsd_wd_results.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        logger.info(f"✅ Saved results to: {output_file}")
        
        # Calculate and log statistics
        jsd_values = [r.get('jsd') for r in results if r.get('jsd') is not None]
        wd_values = [r.get('wd') for r in results if r.get('wd') is not None]
        
        if jsd_values:
            logger.info(f"  JSD: mean={np.mean(jsd_values):.6f}, median={np.median(jsd_values):.6f}, "
                       f"min={np.min(jsd_values):.6f}, max={np.max(jsd_values):.6f}, n={len(jsd_values)}")
        
        if wd_values:
            logger.info(f"  WD:  mean={np.mean(wd_values):.6f}, median={np.median(wd_values):.6f}, "
                       f"min={np.min(wd_values):.6f}, max={np.max(wd_values):.6f}, n={len(wd_values)}")
    
    # Summary
    logger.info(f"\n{'='*80}")
    logger.info("SUMMARY")
    logger.info(f"{'='*80}")
    for artifact_name, results in all_results.items():
        jsd_values = [r.get('jsd') for r in results if r.get('jsd') is not None]
        wd_values = [r.get('wd') for r in results if r.get('wd') is not None]
        logger.info(f"{artifact_name}:")
        if jsd_values:
            logger.info(f"  JSD: {np.mean(jsd_values):.6f} (n={len(jsd_values)})")
        if wd_values:
            logger.info(f"  WD:  {np.mean(wd_values):.6f} (n={len(wd_values)})")
    
    logger.info(f"\n✅ JSD and WD calculation completed!")
    logger.info(f"Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
