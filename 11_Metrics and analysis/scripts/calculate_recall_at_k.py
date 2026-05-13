#!/usr/bin/env python3
"""
Calculate Recall@1, Recall@3, Recall@5, Recall@k, and Recall@2k for Pre-SGO, Post-SGO, Baseline, Train Deltas, and Test Deltas Predictions
===========================================================================

This script:
1. Loads train_set_topic_predictions_with_topic_probs.json to get ground truth (for pre-SGO, post-SGO, baseline)
2. For deltas files (train and test): Uses ground truth directly from actual.topic_probabilities in the deltas files
3. Uses configurable threshold on topic_probabilities (BEFORE normalization) to determine ground truth themes
4. Matches reviews by ASIN (or review_text for baseline)
5. Calculates Recall@1, @3, @5, @k, and @2k (where k = number of ground truth themes)
6. Saves results and generates visualizations

Configuration:
    - Threshold is set in config.yaml under recall_at_k.ground_truth_threshold (default: 0.5)
    - max_reviews is set in config.yaml under recall_at_k.max_reviews
    - train_deltas_path is set in config.yaml under recall_at_k.train_deltas_path (optional, defaults to train deltas directory)
    - test_deltas_path is set in config.yaml under recall_at_k.test_deltas_path (optional, defaults to test deltas directory)

Usage:
    python Metrics and analysis/scripts/calculate_recall_at_k.py
"""

import json
import numpy as np
import sys
import logging
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

# Add project root to path
BASE_DIR = Path(__file__).parent.parent.parent
sys.path.insert(0, str(BASE_DIR))

from utils.wandb_utils import (
    init_wandb_run,
    get_stage_config,
    use_artifact,
    finish_run,
    load_config,
    log_artifact,
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def get_ground_truth_themes(topic_probs_before_normalization: Dict[str, Dict[str, float]], threshold: float = 0.5) -> set:
    """
    Extract ground truth themes from topic_probs_before_normalization.
    
    Args:
        topic_probs_before_normalization: Dictionary of theme -> {"prob_yes": float, "prob_no": float}
        threshold: Minimum prob_yes to include theme (default: 0.5)
        
    Returns:
        Set of theme names that are in ground truth (prob_yes >= threshold)
    """
    if not topic_probs_before_normalization:
        return set()
    
    if isinstance(topic_probs_before_normalization, list):
        # If it's a list, convert to set
        return set(topic_probs_before_normalization)
    
    # Extract prob_yes from each theme and check if >= threshold
    ground_truth_themes = set()
    for theme, probs in topic_probs_before_normalization.items():
        if isinstance(probs, dict):
            prob_yes = probs.get('prob_yes')
            # Handle None/null values - skip if prob_yes is None
            if prob_yes is not None and isinstance(prob_yes, (int, float)) and prob_yes >= threshold:
                ground_truth_themes.add(theme)
        elif isinstance(probs, (int, float)):
            # Fallback: if it's a direct probability value
            if probs >= threshold:
                ground_truth_themes.add(theme)
    
    return ground_truth_themes


def get_top_k_predicted(predicted_themes: Dict[str, float], k: int) -> List[str]:
    """
    Get top-k predicted themes sorted by probability.
    
    Args:
        predicted_themes: Dictionary of theme -> probability
        k: Number of top themes to return
        
    Returns:
        List of top-k theme names (sorted by probability, descending)
    """
    if not predicted_themes:
        return []
    
    if isinstance(predicted_themes, list):
        # If it's a list, return first k
        return predicted_themes[:k]
    
    # Sort by probability (descending) and return top-k
    sorted_themes = sorted(
        predicted_themes.items(),
        key=lambda x: x[1],
        reverse=True
    )
    return [theme for theme, _ in sorted_themes[:k]]


def calculate_recall_at_k(
    predicted_themes: Dict[str, float],
    ground_truth_themes: set,
    k: int
) -> float:
    """
    Calculate Recall@k.
    
    Recall@k = |top_k_predicted ∩ ground_truth| / |ground_truth|
    
    Args:
        predicted_themes: Dictionary of theme -> probability (predictions)
        ground_truth_themes: Set of ground truth theme names
        k: Number of top predictions to consider
        
    Returns:
        Recall@k value (0.0 to 1.0)
    """
    if not ground_truth_themes:
        return 0.0
    
    top_k_predicted = set(get_top_k_predicted(predicted_themes, k))
    intersection = top_k_predicted & ground_truth_themes
    
    recall = len(intersection) / len(ground_truth_themes)
    return recall


def load_train_set_ground_truth(train_set_path: Path, threshold: float = 0.5) -> Tuple[Dict[str, set], Dict[str, set], Dict[tuple, set], Dict[str, Dict[str, Any]]]:
    """
    Load train_set_topic_predictions_with_topic_probs.json and build ground truth lookup.
    
    Args:
        train_set_path: Path to train_set_topic_predictions_with_topic_probs.json
        threshold: Ground truth threshold for topic probabilities (before normalization)
        
    Returns:
        Tuple of (lookup_by_asin, lookup_by_text, lookup_by_baseline_key, full_data_by_asin)
        - lookup_by_asin: Dictionary mapping ASIN -> set of ground truth themes
        - lookup_by_text: Dictionary mapping review_text -> set of ground truth themes
        - lookup_by_baseline_key: Dictionary mapping (user_id, review_index, product_description, review_text) -> set of ground truth themes
        - full_data_by_asin: Dictionary mapping ASIN -> full review data
    """
    logger.info(f"Loading ground truth from: {train_set_path}")
    
    lookup_by_asin = {}
    lookup_by_text = {}
    lookup_by_baseline_key = {}  # (user_id, review_index, product_description, review_text) -> ground_truth
    full_data_by_asin = {}
    
    try:
        with open(train_set_path, 'r', encoding='utf-8') as f:
            train_data = json.load(f)
        
        logger.info(f"Loaded {len(train_data)} reviews from train set")
        
        for review in tqdm(train_data, desc="Building ground truth lookup"):
            asin = review.get('asin', '').strip().upper()
            review_text = review.get('review_text', '').strip()
            product_description = review.get('product_description', '').strip()
            review_id = review.get('review_id', '')
            
            # Use topic_probs_before_normalization instead of topic_probabilities
            topic_probs_before_normalization = review.get('topic_probs_before_normalization', {})
            
            if not topic_probs_before_normalization:
                continue
            
            # Extract ground truth themes using prob_yes from topic_probs_before_normalization
            ground_truth_themes = get_ground_truth_themes(topic_probs_before_normalization, threshold=threshold)
            
            if not ground_truth_themes:
                continue
            
            # Store by ASIN
            if asin:
                lookup_by_asin[asin] = ground_truth_themes
                full_data_by_asin[asin] = review
            
            # Store by review_text (for baseline matching fallback)
            if review_text:
                lookup_by_text[review_text] = ground_truth_themes
            
            # Store by baseline key: (user_id, review_index, product_description, review_text)
            # Extract user_id and review_index from review_id (format: "USERID_x" or "USERID_review_X")
            user_id = None
            review_index = None
            
            if review_id:
                # Try to extract user_id and review_index from review_id
                # Format might be "USERID_x" or "USERID_review_X" or just "USERID"
                parts = review_id.split('_')
                if len(parts) >= 1:
                    user_id = parts[0]
                if len(parts) >= 2:
                    # Try to extract review index from last part
                    try:
                        review_index = int(parts[-1]) if parts[-1].isdigit() else None
                    except:
                        pass
            
            # Also try to extract from review_id pattern like "USERID_review_X"
            if not review_index and review_id:
                if '_review_' in review_id:
                    parts = review_id.split('_review_')
                    if len(parts) == 2:
                        user_id = parts[0]
                        try:
                            review_index = int(parts[1])
                        except:
                            pass
            
            # Create baseline key if we have enough info
            if user_id and review_index is not None and product_description and review_text:
                baseline_key = (user_id, review_index, product_description, review_text)
                lookup_by_baseline_key[baseline_key] = ground_truth_themes
            elif product_description and review_text:
                # Fallback: use (None, None, product_description, review_text)
                baseline_key = (None, None, product_description, review_text)
                if baseline_key not in lookup_by_baseline_key:  # Don't overwrite if already exists
                    lookup_by_baseline_key[baseline_key] = ground_truth_themes
        
        logger.info(f"Built ground truth lookup: {len(lookup_by_asin)} by ASIN, {len(lookup_by_text)} by review_text, {len(lookup_by_baseline_key)} by baseline key")
        
    except Exception as e:
        logger.error(f"Error loading train set: {e}")
        raise
    
    return lookup_by_asin, lookup_by_text, lookup_by_baseline_key, full_data_by_asin


def process_review_with_ground_truth(
    review: Dict[str, Any],
    predicted_themes: Dict[str, float],
    lookup_by_asin: Dict[str, set],
    lookup_by_text: Dict[str, set],
    lookup_by_baseline_key: Optional[Dict[tuple, set]] = None,
    asin: Optional[str] = None,
    review_text: Optional[str] = None,
    is_baseline: bool = False,
    user_id: Optional[str] = None,
    review_index: Optional[int] = None,
    product_description: Optional[str] = None
) -> Optional[Dict[str, Any]]:
    """
    Process a review and calculate Recall@1, @3, @5, Recall@k, and Recall@2k.
    
    Args:
        review: Review dictionary
        predicted_themes: Dictionary of predicted theme -> probability
        lookup_by_asin: Ground truth lookup by ASIN
        lookup_by_text: Ground truth lookup by review_text
        lookup_by_baseline_key: Ground truth lookup by baseline key (user_id, review_index, product_description, review_text)
        asin: ASIN of the review (if available)
        review_text: Review text (if available)
        is_baseline: Whether this is a baseline review (needs stricter matching)
        user_id: User ID (for baseline mattching)
        review_index: Review index (for baseline matching)
        product_description: Product description (for baseline matching)
        
    Returns:
        Dictionary with recall metrics and metadata, or None if processing failed
    """
    if not predicted_themes:
        return None
    
    ground_truth_themes = None
    
    if is_baseline and lookup_by_baseline_key:
        # For baseline, try to match by (user_id, review_index, product_description, review_text)
        if user_id and review_index is not None and product_description and review_text:
            baseline_key = (user_id, review_index, product_description.strip(), review_text.strip())
            ground_truth_themes = lookup_by_baseline_key.get(baseline_key)
        
        # Fallback: try (None, None, product_description, review_text)
        if not ground_truth_themes and product_description and review_text:
            baseline_key = (None, None, product_description.strip(), review_text.strip())
            ground_truth_themes = lookup_by_baseline_key.get(baseline_key)
        
        # Final fallback: try by review_text only
        if not ground_truth_themes and review_text:
            review_text_key = review_text.strip()
            ground_truth_themes = lookup_by_text.get(review_text_key)
    else:
        # For pre-SGO/post-SGO, try to match by ASIN first
        if asin:
            asin_key = str(asin).strip().upper()
            ground_truth_themes = lookup_by_asin.get(asin_key)
        
        # If not found by ASIN, try review_text
        if not ground_truth_themes and review_text:
            review_text_key = review_text.strip()
            ground_truth_themes = lookup_by_text.get(review_text_key)
    
    if not ground_truth_themes:
        return None
    
    # Calculate Recall@1, @3, @5
    recall_at_1 = calculate_recall_at_k(predicted_themes, ground_truth_themes, 1)
    recall_at_3 = calculate_recall_at_k(predicted_themes, ground_truth_themes, 3)
    recall_at_5 = calculate_recall_at_k(predicted_themes, ground_truth_themes, 5)
    
    # Calculate Recall@k where k = number of ground truth themes
    k = len(ground_truth_themes)
    recall_at_k = calculate_recall_at_k(predicted_themes, ground_truth_themes, k) if k > 0 else 0.0
    
    # Calculate Recall@2k where 2k = 2 * number of ground truth themes
    recall_at_2k = calculate_recall_at_k(predicted_themes, ground_truth_themes, 2 * k) if k > 0 else 0.0
    
    # Calculate Recall@max(3, k) where we take max of 3 and k
    max_3_k = max(3, k) if k > 0 else 3
    recall_at_max_3_k = calculate_recall_at_k(predicted_themes, ground_truth_themes, max_3_k) if k > 0 else 0.0
    
    return {
        'review_text': review_text or review.get('actual', {}).get('review_text', ''),
        'user_id': user_id or review.get('user_id', ''),
        'asin': asin or review.get('asin') or review.get('actual', {}).get('asin', ''),
        'recall_at_1': recall_at_1,
        'recall_at_3': recall_at_3,
        'recall_at_5': recall_at_5,
        'recall_at_k': recall_at_k,
        'recall_at_2k': recall_at_2k,
        'recall_at_max_3_k': recall_at_max_3_k,
        'k': k,
        'max_3_k': max_3_k,
        'num_ground_truth_themes': len(ground_truth_themes),
        'ground_truth_themes': list(ground_truth_themes),
        'top_5_predicted': get_top_k_predicted(predicted_themes, 5),
        'tribe_id': review.get('tribe_id'),
        'cluster_id': review.get('cluster_id'),
        'micro_id': review.get('micro_id'),
        'tribe_name': review.get('tribe_name'),
    }


def process_pre_sgo_artifact(
    artifact_path: Path,
    lookup_by_asin: Dict[str, set],
    lookup_by_text: Dict[str, set],
    lookup_by_baseline_key: Optional[Dict[tuple, set]] = None,
    max_reviews: Optional[int] = None
) -> List[Dict[str, Any]]:
    """
    Process pre-SGO artifact and calculate recall metrics.
    
    Args:
        artifact_path: Path to pre-SGO artifact directory
        lookup_by_asin: Ground truth lookup by ASIN
        lookup_by_text: Ground truth lookup by review_text
        max_reviews: Maximum number of reviews to process
        
    Returns:
        List of recall metrics for each review
    """
    results = []
    
    # Find all JSON files
    json_files = list(artifact_path.rglob("*.json"))
    json_files = [f for f in json_files if "grand_summary" not in f.name.lower() and "_cache" not in str(f) and "_with_probs" not in str(f)]
    
    logger.info(f"Found {len(json_files)} JSON files to process")
    
    processed_reviews = set()  # Track processed reviews to avoid duplicates
    
    for json_file in tqdm(json_files, desc="Processing pre-SGO files"):
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except Exception as e:
            logger.warning(f"Failed to load {json_file}: {e}")
            continue
        
        # Extract tribe information from file path
        file_path_str = str(json_file)
        cluster_id = None
        micro_id = None
        tribe_id = None
        
        cluster_match = re.search(r'cluster_(\d+)', file_path_str)
        micro_match = re.search(r'micro_(\d+)', file_path_str)
        
        if cluster_match:
            cluster_id = f"cluster_{cluster_match.group(1)}"
        if micro_match:
            micro_id = f"micro_{micro_match.group(1)}"
        
        if cluster_id and micro_id:
            tribe_id = f"{cluster_id}/{micro_id}"
        
        # Skip if data is not a dict (e.g., if it's a list)
        if not isinstance(data, dict):
            continue
        
        # Extract persona_name (tribe name) from data
        persona_name = data.get('persona_name', None)
        if not persona_name:
            metadata = data.get('metadata', {})
            persona_name = metadata.get('persona_name', None)
        
        # Skip files without user_predictions
        if 'user_predictions' not in data:
            continue
        
        user_predictions = data.get('user_predictions', {})
        
        for user_id, reviews in user_predictions.items():
            if not isinstance(reviews, list):
                continue
            
            for review in reviews:
                if max_reviews and len(results) >= max_reviews:
                    break
                
                actual = review.get('actual', {})
                review_text = actual.get('review_text', '').strip()
                asin = actual.get('asin') or review.get('asin', '')
                
                # Deduplication: use (review_text_hash, asin, user_id) as unique key
                review_key_uniq = (hash(review_text[:200]) if review_text else 0, asin, user_id)
                if review_key_uniq in processed_reviews:
                    continue
                processed_reviews.add(review_key_uniq)
                
                # Get predictions
                prediction = review.get('prediction', {})
                predicted_themes = prediction.get('predicted_themes', {})
                
                if not predicted_themes:
                    continue
                
                # Add tribe information to review
                review['tribe_id'] = tribe_id
                review['cluster_id'] = cluster_id
                review['micro_id'] = micro_id
                review['tribe_name'] = persona_name
                
                # Process review
                result = process_review_with_ground_truth(
                    review, predicted_themes, lookup_by_asin, lookup_by_text,
                    asin=asin, review_text=review_text
                )
                
                if result:
                    results.append(result)
        
        if max_reviews and len(results) >= max_reviews:
            break
    
    logger.info(f"Processed {len(results)} pre-SGO reviews")
    
    return results


def process_post_sgo_artifact(
    artifact_path: Path,
    lookup_by_asin: Dict[str, set],
    lookup_by_text: Dict[str, set],
    lookup_by_baseline_key: Optional[Dict[tuple, set]] = None,
    max_reviews: Optional[int] = None
) -> List[Dict[str, Any]]:
    """
    Process post-SGO artifact and calculate recall metrics.
    
    Args:
        artifact_path: Path to post-SGO artifact directory
        lookup_by_asin: Ground truth lookup by ASIN
        lookup_by_text: Ground truth lookup by review_text
        max_reviews: Maximum number of reviews to process
        
    Returns:
        List of recall metrics for each review
    """
    results = []
    
    # Find all JSON files
    json_files = list(artifact_path.rglob("*.json"))
    json_files = [f for f in json_files if "grand_summary" not in f.name.lower() and "_cache" not in str(f) and "_with_probs" not in str(f)]
    
    logger.info(f"Found {len(json_files)} JSON files to process")
    
    processed_reviews = set()  # Track processed reviews to avoid duplicates
    
    for json_file in tqdm(json_files, desc="Processing post-SGO files"):
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except Exception as e:
            logger.warning(f"Failed to load {json_file}: {e}")
            continue
        
        # Extract tribe information from file path
        file_path_str = str(json_file)
        cluster_id = None
        micro_id = None
        tribe_id = None
        
        cluster_match = re.search(r'cluster_(\d+)', file_path_str)
        micro_match = re.search(r'micro_(\d+)', file_path_str)
        
        if cluster_match:
            cluster_id = f"cluster_{cluster_match.group(1)}"
        if micro_match:
            micro_id = f"micro_{micro_match.group(1)}"
        
        if cluster_id and micro_id:
            tribe_id = f"{cluster_id}/{micro_id}"
        
        # Skip if data is not a dict (e.g., if it's a list)
        if not isinstance(data, dict):
            continue
        
        # Extract persona_name (tribe name) from data
        persona_name = data.get('persona_name', None)
        if not persona_name:
            metadata = data.get('metadata', {})
            persona_name = metadata.get('persona_name', None)
        
        # Skip files without user_predictions
        if 'user_predictions' not in data:
            continue
        
        user_predictions = data.get('user_predictions', {})
        
        for user_id, reviews in user_predictions.items():
            if not isinstance(reviews, list):
                continue
            
            for review in reviews:
                if max_reviews and len(results) >= max_reviews:
                    break
                
                actual = review.get('actual', {})
                review_text = actual.get('review_text', '').strip()
                asin = actual.get('asin') or review.get('asin', '')
                
                # Deduplication: use (review_text_hash, asin, user_id) as unique key
                review_key_uniq = (hash(review_text[:200]) if review_text else 0, asin, user_id)
                if review_key_uniq in processed_reviews:
                    continue
                processed_reviews.add(review_key_uniq)
                
                # Get predictions
                prediction = review.get('prediction', {})
                predicted_themes = prediction.get('predicted_themes', {})
                
                if not predicted_themes:
                    continue
                
                # Add tribe information to review
                review['tribe_id'] = tribe_id
                review['cluster_id'] = cluster_id
                review['micro_id'] = micro_id
                review['tribe_name'] = persona_name
                
                # Process review
                result = process_review_with_ground_truth(
                    review, predicted_themes, lookup_by_asin, lookup_by_text,
                    lookup_by_baseline_key=lookup_by_baseline_key,
                    asin=asin, review_text=review_text, is_baseline=False
                )
                
                if result:
                    results.append(result)
        
        if max_reviews and len(results) >= max_reviews:
            break
    
    logger.info(f"Processed {len(results)} post-SGO reviews")
    
    return results


def process_deltas_artifact(
    artifact_path: Path,
    threshold: float = 0.5,
    max_reviews: Optional[int] = None,
    lookup_by_asin: Optional[Dict[str, set]] = None,
    lookup_by_text: Optional[Dict[str, set]] = None
) -> List[Dict[str, Any]]:
    """
    Process deltas artifact files and calculate recall metrics.
    Deltas files contain predictions and may contain ground truth (actual.topic_probabilities).
    If topic_probabilities is empty, falls back to matching against train set ground truth lookup.
    
    Args:
        artifact_path: Path to deltas artifact directory
        threshold: Ground truth threshold for topic_probabilities (default: 0.5)
        max_reviews: Maximum number of reviews to process
        lookup_by_asin: Optional train set ground truth lookup by ASIN (for fallback)
        lookup_by_text: Optional train set ground truth lookup by review_text (for fallback)
        
    Returns:
        List of recall metrics for each review
    """
    results = []
    
    # Find all JSON files in deltas directory
    if artifact_path.is_file():
        json_files = [artifact_path]
    else:
        json_files = list(artifact_path.rglob("*_deltas.json"))
    
    logger.info(f"Found {len(json_files)} deltas JSON files to process")
    
    processed_reviews = set()  # Track processed reviews to avoid duplicates
    
    for json_file in tqdm(json_files, desc="Processing deltas files"):
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except Exception as e:
            logger.warning(f"Failed to load {json_file}: {e}")
            continue
        
        # Extract tribe information from file path and data
        file_path_str = str(json_file)
        cluster_id = data.get('cluster_id')
        micro_cluster_id = data.get('micro_cluster_id')
        tribe_id = data.get('tribe_id')
        
        # Fallback: extract from file path if not in data
        if not cluster_id:
            cluster_match = re.search(r'cluster_(\d+)', file_path_str)
            if cluster_match:
                cluster_id = f"cluster_{cluster_match.group(1)}"
        
        if not micro_cluster_id:
            micro_match = re.search(r'micro_(\d+)', file_path_str)
            if micro_match:
                micro_cluster_id = f"micro_{micro_match.group(1)}"
        
        if not tribe_id and cluster_id and micro_cluster_id:
            tribe_id = f"{cluster_id}/{micro_cluster_id}"
        
        # Skip if data is not a dict or doesn't have deltas
        if not isinstance(data, dict) or 'deltas' not in data:
            continue
        
        deltas = data.get('deltas', [])
        if not isinstance(deltas, list):
            continue
        
        for delta in deltas:
            if max_reviews and len(results) >= max_reviews:
                break
            
            if not isinstance(delta, dict):
                continue
            
            # Get review information
            review_key = delta.get('review_key', '')
            user_id = delta.get('user_id', '')
            review_idx = delta.get('review_idx')
            review_text = delta.get('actual', {}).get('review_text', '').strip()
            product_description = delta.get('product_description', '').strip()
            
            # Deduplication: use review_key as unique identifier
            if review_key in processed_reviews:
                continue
            processed_reviews.add(review_key)
            
            # Get predictions
            prediction = delta.get('prediction', {})
            predicted_themes = prediction.get('predicted_themes', {})
            
            if not predicted_themes:
                continue
            
            # Get ground truth from actual.topic_probabilities
            actual = delta.get('actual', {})
            topic_probabilities = actual.get('topic_probabilities', {})
            
            ground_truth_themes = None
            
            # First try: use topic_probabilities from deltas file if available
            if topic_probabilities and isinstance(topic_probabilities, dict) and len(topic_probabilities) > 0:
                ground_truth_themes = get_ground_truth_themes(topic_probabilities, threshold=threshold)
            
            # Fallback: if topic_probabilities is empty, try to match against train set ground truth
            if not ground_truth_themes and (lookup_by_asin or lookup_by_text):
                # Try to get ASIN from delta (may be at top level or in actual)
                asin = delta.get('asin') or actual.get('asin')
                if asin and lookup_by_asin:
                    asin_key = str(asin).strip().upper()
                    ground_truth_themes = lookup_by_asin.get(asin_key)
                
                # If not found by ASIN, try review_text
                if not ground_truth_themes and review_text and lookup_by_text:
                    review_text_key = review_text.strip()
                    ground_truth_themes = lookup_by_text.get(review_text_key)
                
                # Log when using fallback (only for first few to avoid spam)
                if ground_truth_themes and len(results) < 5:
                    logger.debug(f"Using train set lookup for review {review_key} (topic_probabilities was empty)")
            
            if not ground_truth_themes:
                continue
            
            # Calculate Recall@1, @3, @5
            recall_at_1 = calculate_recall_at_k(predicted_themes, ground_truth_themes, 1)
            recall_at_3 = calculate_recall_at_k(predicted_themes, ground_truth_themes, 3)
            recall_at_5 = calculate_recall_at_k(predicted_themes, ground_truth_themes, 5)
            
            # Calculate Recall@k where k = number of ground truth themes
            k = len(ground_truth_themes)
            recall_at_k = calculate_recall_at_k(predicted_themes, ground_truth_themes, k) if k > 0 else 0.0
            
            # Calculate Recall@2k where 2k = 2 * number of ground truth themes
            recall_at_2k = calculate_recall_at_k(predicted_themes, ground_truth_themes, 2 * k) if k > 0 else 0.0
            
            # Calculate Recall@max(3, k) where we take max of 3 and k
            max_3_k = max(3, k) if k > 0 else 3
            recall_at_max_3_k = calculate_recall_at_k(predicted_themes, ground_truth_themes, max_3_k) if k > 0 else 0.0
            
            result = {
                'review_text': review_text,
                'user_id': user_id,
                'review_key': review_key,
                'review_idx': review_idx,
                'asin': None,  # Deltas files may not have ASIN
                'recall_at_1': recall_at_1,
                'recall_at_3': recall_at_3,
                'recall_at_5': recall_at_5,
                'recall_at_k': recall_at_k,
                'recall_at_2k': recall_at_2k,
                'recall_at_max_3_k': recall_at_max_3_k,
                'k': k,
                'max_3_k': max_3_k,
                'num_ground_truth_themes': len(ground_truth_themes),
                'ground_truth_themes': list(ground_truth_themes),
                'top_5_predicted': get_top_k_predicted(predicted_themes, 5),
                'tribe_id': tribe_id,
                'cluster_id': cluster_id,
                'micro_id': micro_cluster_id,
                'tribe_name': None,  # Deltas files may not have persona_name
                'product_description': product_description,
            }
            
            results.append(result)
        
        if max_reviews and len(results) >= max_reviews:
            break
    
    logger.info(f"Processed {len(results)} reviews from deltas files")
    
    return results


def process_baseline_artifact(
    artifact_path: Path,
    lookup_by_asin: Dict[str, set],
    lookup_by_text: Dict[str, set],
    lookup_by_baseline_key: Dict[tuple, set],
    max_reviews: Optional[int] = None
) -> List[Dict[str, Any]]:
    """
    Process baseline artifact and calculate recall metrics.
    For baseline, we match by review_text, user_id, and other details (no ASIN).
    
    Args:
        artifact_path: Path to baseline artifact file or directory
        lookup_by_asin: Ground truth lookup by ASIN (may not be used for baseline)
        lookup_by_text: Ground truth lookup by review_text
        max_reviews: Maximum number of reviews to process
        
    Returns:
        List of recall metrics for each review
    """
    results = []
    
    # Find JSON files
    if artifact_path.is_file():
        json_files = [artifact_path]
    else:
        json_files = list(artifact_path.rglob("*.json"))
    
    logger.info(f"Found {len(json_files)} baseline JSON files to process")
    
    processed_reviews = set()  # Track processed reviews to avoid duplicates
    
    for json_file in tqdm(json_files, desc="Processing baseline files"):
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except Exception as e:
            logger.warning(f"Failed to load {json_file}: {e}")
            continue
        
        # Baseline format: flat dict with review keys
        if isinstance(data, dict):
            for review_key, review in data.items():
                if max_reviews and len(results) >= max_reviews:
                    break
                
                if not isinstance(review, dict):
                    continue
                
                actual = review.get('actual', {})
                review_text = actual.get('review_text', '').strip()
                user_id = review.get('user_id', '')
                review_index = review.get('review_index')
                product_description = review.get('product_description', '').strip()
                
                # Extract user_id from key if not present
                if not user_id:
                    parts = review_key.rsplit('_review_', 1)
                    if len(parts) == 2:
                        user_id = parts[0]
                        try:
                            review_index = int(parts[1])
                        except:
                            pass
                
                # Deduplication: use (review_text_hash, user_id, review_index) as unique key (no ASIN for baseline)
                review_key_uniq = (hash(review_text[:200]) if review_text else 0, user_id, review_index)
                if review_key_uniq in processed_reviews:
                    continue
                processed_reviews.add(review_key_uniq)
                
                # Get predictions
                prediction = review.get('prediction', {})
                predicted_themes = prediction.get('predicted_themes', {})
                
                if not predicted_themes:
                    continue
                
                # For baseline, match by (user_id, review_index, product_description, review_text)
                # Check all fields match before setting ground truth
                result = process_review_with_ground_truth(
                    review, predicted_themes, lookup_by_asin, lookup_by_text,
                    lookup_by_baseline_key=lookup_by_baseline_key,
                    asin=None, review_text=review_text,
                    is_baseline=True,
                    user_id=user_id,
                    review_index=review_index,
                    product_description=product_description
                )
                
                if result:
                    results.append(result)
        
        if max_reviews and len(results) >= max_reviews:
            break
    
    logger.info(f"Processed {len(results)} baseline reviews")
    
    return results


def group_results_by_tribe(results: List[Dict[str, Any]], min_reviews: int = 5) -> Dict[str, Dict[str, Any]]:
    """
    Group recall results by tribe.
    
    Args:
        results: List of recall result dictionaries
        min_reviews: Minimum number of reviews per tribe to include
        
    Returns:
        Dictionary mapping tribe_id -> tribe statistics
    """
    tribe_results = defaultdict(list)
    tribe_names = {}
    
    for result in results:
        tribe_id = result.get('tribe_id')
        if not tribe_id:
            continue
        
        tribe_results[tribe_id].append(result)
        # Store tribe name if available
        if 'tribe_name' in result and result['tribe_name']:
            tribe_names[tribe_id] = result['tribe_name']
    
    # Calculate statistics per tribe
    tribe_stats = {}
    for tribe_id, tribe_reviews in tribe_results.items():
        if len(tribe_reviews) < min_reviews:
            continue
        
        # Extract recall values
        r1 = [r['recall_at_1'] for r in tribe_reviews]
        r3 = [r['recall_at_3'] for r in tribe_reviews]
        r5 = [r['recall_at_5'] for r in tribe_reviews]
        rk = [r['recall_at_k'] for r in tribe_reviews]
        r2k = [r['recall_at_2k'] for r in tribe_reviews]
        rmax3k = [r['recall_at_max_3_k'] for r in tribe_reviews]
        
        # Extract cluster and micro IDs
        cluster_id = tribe_reviews[0].get('cluster_id', '')
        micro_id = tribe_reviews[0].get('micro_id', '')
        tribe_name = tribe_names.get(tribe_id, tribe_id)
        
        tribe_stats[tribe_id] = {
            'tribe_id': tribe_id,
            'tribe_name': tribe_name,
            'cluster_id': cluster_id,
            'micro_id': micro_id,
            'num_reviews': len(tribe_reviews),
            'recall_at_1_mean': np.mean(r1),
            'recall_at_3_mean': np.mean(r3),
            'recall_at_5_mean': np.mean(r5),
            'recall_at_k_mean': np.mean(rk),
            'recall_at_2k_mean': np.mean(r2k),
            'recall_at_max_3_k_mean': np.mean(rmax3k),
            'recall_at_1_std': np.std(r1),
            'recall_at_3_std': np.std(r3),
            'recall_at_5_std': np.std(r5),
            'recall_at_k_std': np.std(rk),
            'recall_at_2k_std': np.std(r2k),
            'recall_at_max_3_k_std': np.std(rmax3k),
        }
    
    return tribe_stats


def create_tribe_wise_visualizations(
    pre_sgo_results: List[Dict[str, Any]],
    post_sgo_results: List[Dict[str, Any]],
    output_dir: Path
):
    """
    Create tribe-wise recall metric visualizations.
    
    Args:
        pre_sgo_results: List of pre-SGO recall results
        post_sgo_results: List of post-SGO recall results
        output_dir: Output directory for graphs
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Group results by tribe
    pre_sgo_tribes = group_results_by_tribe(pre_sgo_results, min_reviews=5)
    post_sgo_tribes = group_results_by_tribe(post_sgo_results, min_reviews=5)
    
    if not pre_sgo_tribes and not post_sgo_tribes:
        logger.warning("No tribe-level data to visualize")
        return
    
    # Get common tribes (tribes present in both)
    common_tribes = set(pre_sgo_tribes.keys()) & set(post_sgo_tribes.keys())
    
    if not common_tribes:
        logger.warning("No common tribes between Pre-SGO and Post-SGO")
        return
    
    # Sort tribes by cluster and micro ID for consistent ordering
    sorted_tribes = sorted(common_tribes, key=lambda t: (
        pre_sgo_tribes.get(t, {}).get('cluster_id', ''),
        pre_sgo_tribes.get(t, {}).get('micro_id', '')
    ))
    
    # Limit to top 20 tribes by review count for readability
    tribe_review_counts = [(t, pre_sgo_tribes[t]['num_reviews'] + post_sgo_tribes[t]['num_reviews']) 
                           for t in sorted_tribes]
    tribe_review_counts.sort(key=lambda x: x[1], reverse=True)
    top_tribes = [t[0] for t in tribe_review_counts[:20]]
    
    # Create tribe names for display
    tribe_display_names = {}
    for tribe_id in top_tribes:
        pre_name = pre_sgo_tribes[tribe_id].get('tribe_name', tribe_id)
        post_name = post_sgo_tribes[tribe_id].get('tribe_name', tribe_id)
        # Use the first non-empty name
        display_name = pre_name if pre_name and pre_name != tribe_id else post_name
        if not display_name or display_name == tribe_id:
            display_name = tribe_id.replace('/', ' - ')
        tribe_display_names[tribe_id] = display_name
    
    # Graph 1: Tribe-wise comparison for all recall metrics (Pre-SGO vs Post-SGO)
    metrics = [
        ('Recall@1', 'recall_at_1_mean'),
        ('Recall@3', 'recall_at_3_mean'),
        ('Recall@5', 'recall_at_5_mean'),
        ('Recall@k', 'recall_at_k_mean'),
        ('Recall@2k', 'recall_at_2k_mean'),
        ('Recall@max(3,k)', 'recall_at_max_3_k_mean'),
    ]
    
    for metric_name, metric_key in metrics:
        fig, ax = plt.subplots(figsize=(20, 10))
        
        pre_values = [pre_sgo_tribes[t][metric_key] for t in top_tribes]
        post_values = [post_sgo_tribes[t][metric_key] for t in top_tribes]
        tribe_labels = [tribe_display_names[t] for t in top_tribes]
        
        x = np.arange(len(top_tribes))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, pre_values, width, label='Pre-SGO', alpha=0.8, color='#4A7FB5', edgecolor='black', linewidth=1)
        bars2 = ax.bar(x + width/2, post_values, width, label='Post-SGO', alpha=0.8, color='#51CF66', edgecolor='black', linewidth=1)
        
        # Add value labels on bars
        for bars, values in [(bars1, pre_values), (bars2, post_values)]:
            for bar, val in zip(bars, values):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                       f'{val:.3f}',
                       ha='center', va='bottom', fontsize=8, fontweight='bold')
        
        ax.set_xlabel('Tribe', fontsize=12, fontweight='bold')
        ax.set_ylabel(f'Mean {metric_name}', fontsize=12, fontweight='bold')
        ax.set_title(f'{metric_name} by Tribe: Pre-SGO vs Post-SGO', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(tribe_labels, rotation=45, ha='right', fontsize=9)
        ax.legend(fontsize=11)
        ax.grid(axis='y', alpha=0.3)
        
        # Calculate and display overall averages
        pre_avg = np.mean(pre_values)
        post_avg = np.mean(post_values)
        ax.text(0.02, 0.98, f'Pre-SGO Avg: {pre_avg:.4f} | Post-SGO Avg: {post_avg:.4f}',
               transform=ax.transAxes, fontsize=10, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        safe_metric_name = metric_name.lower().replace('@', '_at_').replace('(', '').replace(')', '').replace(',', '')
        plt.savefig(output_dir / f'recall_tribe_wise_{safe_metric_name}.png', bbox_inches='tight')
        plt.close()
    
    # Graph 2: All metrics comparison for top tribes (single graph)
    fig, ax = plt.subplots(figsize=(24, 12))
    
    # Select top 10 tribes for this comprehensive view
    top_10_tribes = top_tribes[:10]
    
    x = np.arange(len(top_10_tribes))
    width = 0.12
    
    metric_keys = [
        ('recall_at_1_mean', 'Recall@1', '#4A7FB5'),
        ('recall_at_3_mean', 'Recall@3', '#51CF66'),
        ('recall_at_5_mean', 'Recall@5', '#FF6B6B'),
        ('recall_at_k_mean', 'Recall@k', '#F59E0B'),
        ('recall_at_2k_mean', 'Recall@2k', '#8B5CF6'),
        ('recall_at_max_3_k_mean', 'Recall@max(3,k)', '#10B981'),
    ]
    
    # Pre-SGO bars
    for idx, (metric_key, label, color) in enumerate(metric_keys):
        offset = (idx - 2.5) * width
        values = [pre_sgo_tribes[t][metric_key] for t in top_10_tribes]
        bars = ax.bar(x + offset, values, width, label=f'Pre-SGO {label}', alpha=0.7, color=color, edgecolor='black', linewidth=0.5)
    
    # Post-SGO bars (slightly offset and with pattern)
    for idx, (metric_key, label, color) in enumerate(metric_keys):
        offset = (idx - 2.5) * width + width/2
        values = [post_sgo_tribes[t][metric_key] for t in top_10_tribes]
        bars = ax.bar(x + offset, values, width, label=f'Post-SGO {label}', alpha=0.9, color=color, 
                     edgecolor='black', linewidth=0.5, hatch='///')
    
    ax.set_xlabel('Tribe', fontsize=12, fontweight='bold')
    ax.set_ylabel('Mean Recall', fontsize=12, fontweight='bold')
    ax.set_title('All Recall Metrics by Tribe: Pre-SGO vs Post-SGO (Top 10 Tribes)', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([tribe_display_names[t] for t in top_10_tribes], rotation=45, ha='right', fontsize=9)
    ax.legend(fontsize=8, ncol=2, loc='upper left')
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'recall_tribe_wise_all_metrics.png', bbox_inches='tight')
    plt.close()
    
    logger.info(f"Created tribe-wise visualizations for {len(top_tribes)} tribes")


def create_visualizations(
    pre_sgo_results: List[Dict[str, Any]],
    baseline_history_results: List[Dict[str, Any]],
    baseline_backstory_results: List[Dict[str, Any]],
    post_sgo_results: List[Dict[str, Any]],
    output_dir: Path
):
    """
    Create visualization graphs for Recall@k metrics.
    
    Args:
        pre_sgo_results: List of pre-SGO recall results
        baseline_history_results: List of baseline history recall results
        baseline_backstory_results: List of baseline backstory recall results
        post_sgo_results: List of post-SGO recall results
        output_dir: Output directory for graphs
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Saving graphs to: {output_dir}")
    
    # Set style
    sns.set_style("whitegrid")
    plt.rcParams['figure.dpi'] = 300
    plt.rcParams['savefig.dpi'] = 300
    
    # Extract recall values
    pre_sgo_r1 = [r['recall_at_1'] for r in pre_sgo_results]
    pre_sgo_r3 = [r['recall_at_3'] for r in pre_sgo_results]
    pre_sgo_r5 = [r['recall_at_5'] for r in pre_sgo_results]
    pre_sgo_rk = [r['recall_at_k'] for r in pre_sgo_results]
    pre_sgo_r2k = [r['recall_at_2k'] for r in pre_sgo_results]
    pre_sgo_rmax3k = [r['recall_at_max_3_k'] for r in pre_sgo_results]
    
    baseline_history_r1 = [r['recall_at_1'] for r in baseline_history_results] if baseline_history_results else []
    baseline_history_r3 = [r['recall_at_3'] for r in baseline_history_results] if baseline_history_results else []
    baseline_history_r5 = [r['recall_at_5'] for r in baseline_history_results] if baseline_history_results else []
    baseline_history_rk = [r['recall_at_k'] for r in baseline_history_results] if baseline_history_results else []
    baseline_history_r2k = [r['recall_at_2k'] for r in baseline_history_results] if baseline_history_results else []
    baseline_history_rmax3k = [r['recall_at_max_3_k'] for r in baseline_history_results] if baseline_history_results else []
    
    baseline_backstory_r1 = [r['recall_at_1'] for r in baseline_backstory_results] if baseline_backstory_results else []
    baseline_backstory_r3 = [r['recall_at_3'] for r in baseline_backstory_results] if baseline_backstory_results else []
    baseline_backstory_r5 = [r['recall_at_5'] for r in baseline_backstory_results] if baseline_backstory_results else []
    baseline_backstory_rk = [r['recall_at_k'] for r in baseline_backstory_results] if baseline_backstory_results else []
    baseline_backstory_r2k = [r['recall_at_2k'] for r in baseline_backstory_results] if baseline_backstory_results else []
    baseline_backstory_rmax3k = [r['recall_at_max_3_k'] for r in baseline_backstory_results] if baseline_backstory_results else []
    
    post_sgo_r1 = [r['recall_at_1'] for r in post_sgo_results] if post_sgo_results else []
    post_sgo_r3 = [r['recall_at_3'] for r in post_sgo_results] if post_sgo_results else []
    post_sgo_r5 = [r['recall_at_5'] for r in post_sgo_results] if post_sgo_results else []
    post_sgo_rk = [r['recall_at_k'] for r in post_sgo_results] if post_sgo_results else []
    post_sgo_r2k = [r['recall_at_2k'] for r in post_sgo_results] if post_sgo_results else []
    post_sgo_rmax3k = [r['recall_at_max_3_k'] for r in post_sgo_results] if post_sgo_results else []
    
    # Graph 1: Comparison Bar Chart
    fig, ax = plt.subplots(figsize=(16, 8))
    
    methods = []
    recall_at_1_means = []
    recall_at_3_means = []
    recall_at_5_means = []
    recall_at_k_means = []
    recall_at_2k_means = []
    recall_at_max_3_k_means = []
    
    if pre_sgo_results:
        methods.append('Pre-SGO')
        recall_at_1_means.append(np.mean(pre_sgo_r1))
        recall_at_3_means.append(np.mean(pre_sgo_r3))
        recall_at_5_means.append(np.mean(pre_sgo_r5))
        recall_at_k_means.append(np.mean(pre_sgo_rk))
        recall_at_2k_means.append(np.mean(pre_sgo_r2k))
        recall_at_max_3_k_means.append(np.mean(pre_sgo_rmax3k))
    
    if baseline_history_results:
        methods.append('Baseline (History)')
        recall_at_1_means.append(np.mean(baseline_history_r1))
        recall_at_3_means.append(np.mean(baseline_history_r3))
        recall_at_5_means.append(np.mean(baseline_history_r5))
        recall_at_k_means.append(np.mean(baseline_history_rk))
        recall_at_2k_means.append(np.mean(baseline_history_r2k))
        recall_at_max_3_k_means.append(np.mean(baseline_history_rmax3k))
    
    if baseline_backstory_results:
        methods.append('Baseline (Backstory)')
        recall_at_1_means.append(np.mean(baseline_backstory_r1))
        recall_at_3_means.append(np.mean(baseline_backstory_r3))
        recall_at_5_means.append(np.mean(baseline_backstory_r5))
        recall_at_k_means.append(np.mean(baseline_backstory_rk))
        recall_at_2k_means.append(np.mean(baseline_backstory_r2k))
        recall_at_max_3_k_means.append(np.mean(baseline_backstory_rmax3k))
    
    if post_sgo_results:
        methods.append('Post-SGO')
        recall_at_1_means.append(np.mean(post_sgo_r1))
        recall_at_3_means.append(np.mean(post_sgo_r3))
        recall_at_5_means.append(np.mean(post_sgo_r5))
        recall_at_k_means.append(np.mean(post_sgo_rk))
        recall_at_2k_means.append(np.mean(post_sgo_r2k))
        recall_at_max_3_k_means.append(np.mean(post_sgo_rmax3k))
    
    if not methods:
        logger.warning("No results to visualize")
        return
    
    x = np.arange(len(methods))
    width = 0.12
    
    ax.bar(x - 2.5*width, recall_at_1_means, width, label='Recall@1', alpha=0.8, color='#4A7FB5')
    ax.bar(x - 1.5*width, recall_at_3_means, width, label='Recall@3', alpha=0.8, color='#51CF66')
    ax.bar(x - 0.5*width, recall_at_5_means, width, label='Recall@5', alpha=0.8, color='#FF6B6B')
    ax.bar(x + 0.5*width, recall_at_k_means, width, label='Recall@k', alpha=0.8, color='#F59E0B')
    ax.bar(x + 1.5*width, recall_at_2k_means, width, label='Recall@2k', alpha=0.8, color='#8B5CF6')
    ax.bar(x + 2.5*width, recall_at_max_3_k_means, width, label='Recall@max(3,k)', alpha=0.8, color='#10B981')
    
    ax.set_xlabel('Method', fontsize=12, fontweight='bold')
    ax.set_ylabel('Mean Recall', fontsize=12, fontweight='bold')
    ax.set_title('Recall@1, @3, @5, @k, @2k, @max(3,k) Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(methods)
    ax.legend(fontsize=9)
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for i, (r1, r3, r5, rk, r2k, rmax3k) in enumerate(zip(recall_at_1_means, recall_at_3_means, recall_at_5_means, recall_at_k_means, recall_at_2k_means, recall_at_max_3_k_means)):
        ax.text(i - 2.5*width, r1 + 0.01, f'{r1:.3f}', ha='center', va='bottom', fontsize=6)
        ax.text(i - 1.5*width, r3 + 0.01, f'{r3:.3f}', ha='center', va='bottom', fontsize=6)
        ax.text(i - 0.5*width, r5 + 0.01, f'{r5:.3f}', ha='center', va='bottom', fontsize=6)
        ax.text(i + 0.5*width, rk + 0.01, f'{rk:.3f}', ha='center', va='bottom', fontsize=6)
        ax.text(i + 1.5*width, r2k + 0.01, f'{r2k:.3f}', ha='center', va='bottom', fontsize=6)
        ax.text(i + 2.5*width, rmax3k + 0.01, f'{rmax3k:.3f}', ha='center', va='bottom', fontsize=6)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'recall_comparison.png', bbox_inches='tight')
    plt.close()
    
    # Graph 2: Distribution Histograms
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    axes = axes.flatten()
    
    metrics = [
        ('Recall@1', [pre_sgo_r1, baseline_history_r1, baseline_backstory_r1, post_sgo_r1], ['Pre-SGO', 'Baseline (History)', 'Baseline (Backstory)', 'Post-SGO']),
        ('Recall@3', [pre_sgo_r3, baseline_history_r3, baseline_backstory_r3, post_sgo_r3], ['Pre-SGO', 'Baseline (History)', 'Baseline (Backstory)', 'Post-SGO']),
        ('Recall@5', [pre_sgo_r5, baseline_history_r5, baseline_backstory_r5, post_sgo_r5], ['Pre-SGO', 'Baseline (History)', 'Baseline (Backstory)', 'Post-SGO']),
        ('Recall@k', [pre_sgo_rk, baseline_history_rk, baseline_backstory_rk, post_sgo_rk], ['Pre-SGO', 'Baseline (History)', 'Baseline (Backstory)', 'Post-SGO']),
        ('Recall@2k', [pre_sgo_r2k, baseline_history_r2k, baseline_backstory_r2k, post_sgo_r2k], ['Pre-SGO', 'Baseline (History)', 'Baseline (Backstory)', 'Post-SGO']),
        ('Recall@max(3,k)', [pre_sgo_rmax3k, baseline_history_rmax3k, baseline_backstory_rmax3k, post_sgo_rmax3k], ['Pre-SGO', 'Baseline (History)', 'Baseline (Backstory)', 'Post-SGO']),
    ]
    
    colors = ['#4A7FB5', '#FF6B6B', '#9B59B6', '#51CF66']
    
    for idx, (metric_name, value_lists, labels) in enumerate(metrics):
        ax = axes[idx]
        
        for val_list, label, color in zip(value_lists, labels, colors):
            if val_list:
                ax.hist(val_list, bins=20, alpha=0.6, label=label, color=color, edgecolor='black')
                if val_list:
                    ax.axvline(np.mean(val_list), color=color, linestyle='--', linewidth=2,
                              label=f'{label} Mean: {np.mean(val_list):.3f}')
        
        ax.set_xlabel(metric_name, fontsize=11, fontweight='bold')
        ax.set_ylabel('Frequency', fontsize=11, fontweight='bold')
        ax.set_title(f'{metric_name} Distribution', fontsize=12, fontweight='bold')
        ax.legend(fontsize=8)
        ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'recall_distributions.png', bbox_inches='tight')
    plt.close()
    
    # Graph 3: All Recall Metrics Pre-SGO vs Post-SGO Comparison (Single Graph)
    if pre_sgo_results and post_sgo_results:
        logger.info("Creating Pre-SGO vs Post-SGO comparison graph for all recall metrics...")
        
        metrics_data = [
            ('Recall@1', pre_sgo_r1, post_sgo_r1, '#4A7FB5'),
            ('Recall@3', pre_sgo_r3, post_sgo_r3, '#51CF66'),
            ('Recall@5', pre_sgo_r5, post_sgo_r5, '#FF6B6B'),
            ('Recall@k', pre_sgo_rk, post_sgo_rk, '#F59E0B'),
            ('Recall@2k', pre_sgo_r2k, post_sgo_r2k, '#8B5CF6'),
            ('Recall@max(3,k)', pre_sgo_rmax3k, post_sgo_rmax3k, '#10B981'),
        ]
        
        # Calculate means
        pre_means = []
        post_means = []
        metric_names = []
        colors = []
        
        for metric_name, pre_values, post_values, color in metrics_data:
            if not pre_values or not post_values:
                continue
            metric_names.append(metric_name)
            pre_means.append(np.mean(pre_values))
            post_means.append(np.mean(post_values))
            colors.append(color)
        
        if metric_names:
            fig, ax = plt.subplots(figsize=(16, 8))
            
            x = np.arange(len(metric_names))
            width = 0.35
            
            # Create grouped bars
            bars1 = ax.bar(x - width/2, pre_means, width, label='Pre-SGO', alpha=0.8, color='#4A7FB5', edgecolor='black', linewidth=1.5)
            bars2 = ax.bar(x + width/2, post_means, width, label='Post-SGO', alpha=0.8, color='#51CF66', edgecolor='black', linewidth=1.5)
            
            # Add value labels on bars
            for bars, means in [(bars1, pre_means), (bars2, post_means)]:
                for bar, mean_val in zip(bars, means):
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                           f'{mean_val:.4f}',
                           ha='center', va='bottom', fontsize=10, fontweight='bold')
            
            ax.set_xlabel('Recall Metric', fontsize=12, fontweight='bold')
            ax.set_ylabel('Mean Recall', fontsize=12, fontweight='bold')
            ax.set_title('All Recall Metrics: Pre-SGO vs Post-SGO Comparison', fontsize=14, fontweight='bold')
            ax.set_xticks(x)
            ax.set_xticklabels(metric_names, rotation=0, ha='center')
            ax.legend(fontsize=11, loc='upper left')
            ax.set_ylim(0, max(max(pre_means), max(post_means)) * 1.15)
            ax.grid(axis='y', alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(output_dir / 'recall_all_metrics_pre_vs_post.png', bbox_inches='tight')
            plt.close()
            
            logger.info("✓ Created Pre-SGO vs Post-SGO comparison graph for all recall metrics")
    
    logger.info("✓ Created visualization graphs")


def main():
    """Main function to calculate Recall@k metrics."""
    logger.info("="*80)
    logger.info("Calculating Recall@1, @3, @5, @k using train_set ground truth")
    logger.info("="*80)
    
    # Initialize WandB (optional - can be None to use local files only)
    run = None
    try:
        run = init_wandb_run(
            run_name="calculate_recall_at_k",
            stage="Metrics and analysis",
            config={"description": "Calculate Recall@1, @3, @5, @k using train_set ground truth"},
            job_type="recall_calculation"
        )
    except Exception as e:
        logger.warning(f"WandB initialization failed, using local files only: {e}")
        run = None
    
    # Load config
    config = load_config(BASE_DIR / "Metrics and analysis" / "config.yaml")
    
    # Get recall_at_k config
    recall_config = config.get('recall_at_k', {})
    ground_truth_threshold = recall_config.get('ground_truth_threshold', 0.5)
    max_reviews = recall_config.get('max_reviews') or config.get('processing', {}).get('max_reviews')
    
    logger.info(f"Using ground truth threshold: {ground_truth_threshold}")
    
    # Load train set ground truth
    logger.info("\n" + "="*80)
    logger.info("STEP 0: Loading Train Set Ground Truth")
    logger.info("="*80)
    
    train_set_path = BASE_DIR / "artifacts" / "train_set_topic_predictions-v8" / "processed_train_predictions_with_topic_probs.json"
    if not train_set_path.exists():
        logger.error(f"Train set not found: {train_set_path}")
        return
    
    lookup_by_asin, lookup_by_text, lookup_by_baseline_key, full_data_by_asin = load_train_set_ground_truth(train_set_path, threshold=ground_truth_threshold)
    
    if not lookup_by_asin and not lookup_by_text:
        logger.error("Failed to build ground truth lookup")
        return
    
    # Get artifact paths
    pre_sgo_artifact_name = config.get('input_artifacts', {}).get('pre_sgo_context')
    baseline_history_artifact_name = config.get('input_artifacts', {}).get('user_history')
    baseline_backstory_artifact_name = config.get('input_artifacts', {}).get('user_backstory')
    post_sgo_artifact_name = config.get('input_artifacts', {}).get('post_sgo_context')
    
    # Process pre-SGO (get from WandB)
    pre_sgo_results = []
    if pre_sgo_artifact_name:
        logger.info("\n" + "="*80)
        logger.info("STEP 1: Processing Pre-SGO Artifact (from WandB)")
        logger.info("="*80)
        
        # Try local first
        pre_sgo_artifact_path = BASE_DIR / "07_sgo_training" / "artifacts" / pre_sgo_artifact_name.split(':')[0]
        if not pre_sgo_artifact_path.exists():
            pre_sgo_artifact_path = BASE_DIR / "06_pre_sgo" / "artifacts" / pre_sgo_artifact_name.split(':')[0]
        
        # If not found locally, download from WandB
        if not pre_sgo_artifact_path.exists():
            if run:
                logger.info("Local pre-SGO artifact not found, downloading from WandB...")
                pre_sgo_artifact_path = use_artifact(run, pre_sgo_artifact_name, artifact_type="dataset")
            else:
                logger.warning(f"Local pre-SGO artifact not found and WandB not available: {pre_sgo_artifact_name}")
                pre_sgo_artifact_path = None
        
        if pre_sgo_artifact_path and pre_sgo_artifact_path.exists():
            pre_sgo_results = process_pre_sgo_artifact(pre_sgo_artifact_path, lookup_by_asin, lookup_by_text, lookup_by_baseline_key, max_reviews)
        else:
            logger.warning(f"Pre-SGO artifact not found: {pre_sgo_artifact_name}")
    
    # Process baseline history (get from LOCAL only, no WandB)
    baseline_history_results = []
    if baseline_history_artifact_name:
        logger.info("\n" + "="*80)
        logger.info("STEP 2: Processing Baseline History Artifact (from LOCAL)")
        logger.info("="*80)
        
        baseline_history_artifact_path = BASE_DIR / "09_baselines" / "artifacts" / "baseline_predictions_o3_history_logprobs_v4" / "baseline_predictions_o3_history_logprobs.json"
        if not baseline_history_artifact_path.exists():
            baseline_history_artifact_path = BASE_DIR / "09_baselines" / "artifacts" / "baseline_predictions_v4" / "baseline_predictions_o3_history.json"
        
        if not baseline_history_artifact_path.exists():
            logger.warning(f"Local baseline history artifact not found: {baseline_history_artifact_path}")
            logger.warning("Baseline artifacts must be loaded from local files. Skipping...")
            baseline_history_artifact_path = None
        
        if baseline_history_artifact_path and baseline_history_artifact_path.exists():
            baseline_history_results = process_baseline_artifact(baseline_history_artifact_path, lookup_by_asin, lookup_by_text, lookup_by_baseline_key, max_reviews)
        else:
            logger.warning(f"Baseline history artifact not found: {baseline_history_artifact_path}")
    
    # Process baseline backstory (get from LOCAL only, no WandB)
    baseline_backstory_results = []
    if baseline_backstory_artifact_name:
        logger.info("\n" + "="*80)
        logger.info("STEP 3: Processing Baseline Backstory Artifact (from LOCAL)")
        logger.info("="*80)
        
        baseline_backstory_artifact_path = BASE_DIR / "09_baselines" / "artifacts" / "baseline_predictions_o3_backstory_logprobs_v4" / "baseline_predictions_o3_backstory_logprobs.json"
        if not baseline_backstory_artifact_path.exists():
            baseline_backstory_artifact_path = BASE_DIR / "09_baselines" / "artifacts" / "baseline_predictions_v4" / "baseline_predictions_o3_backstory.json"
        
        if not baseline_backstory_artifact_path.exists():
            logger.warning(f"Local baseline backstory artifact not found: {baseline_backstory_artifact_path}")
            logger.warning("Baseline artifacts must be loaded from local files. Skipping...")
            baseline_backstory_artifact_path = None
        
        if baseline_backstory_artifact_path and baseline_backstory_artifact_path.exists():
            baseline_backstory_results = process_baseline_artifact(baseline_backstory_artifact_path, lookup_by_asin, lookup_by_text, lookup_by_baseline_key, max_reviews)
        else:
            logger.warning(f"Baseline backstory artifact not found: {baseline_backstory_artifact_path}")
    
    # Process post-SGO (get from WandB)
    post_sgo_results = []
    if post_sgo_artifact_name:
        logger.info("\n" + "="*80)
        logger.info("STEP 4: Processing Post-SGO Artifact (from WandB)")
        logger.info("="*80)
        
        # Try local first
        post_sgo_artifact_path = BASE_DIR / "07_sgo_training" / "artifacts" / post_sgo_artifact_name.split(':')[0]
        
        # If not found locally, download from WandB
        if not post_sgo_artifact_path.exists():
            if run:
                logger.info("Local post-SGO artifact not found, downloading from WandB...")
                post_sgo_artifact_path = use_artifact(run, post_sgo_artifact_name, artifact_type="dataset")
            else:
                logger.warning(f"Local post-SGO artifact not found and WandB not available: {post_sgo_artifact_name}")
                post_sgo_artifact_path = None
        
        if post_sgo_artifact_path and post_sgo_artifact_path.exists():
            post_sgo_results = process_post_sgo_artifact(post_sgo_artifact_path, lookup_by_asin, lookup_by_text, lookup_by_baseline_key, max_reviews)
        else:
            logger.warning(f"Post-SGO artifact not found: {post_sgo_artifact_name}")
    
    # Process train deltas files (train post-SGO with ground truth already in file)
    train_deltas_results = []
    train_deltas_path = recall_config.get('train_deltas_path')
    if train_deltas_path:
        logger.info("\n" + "="*80)
        logger.info("STEP 5: Processing Train Deltas Files (Train Post-SGO)")
        logger.info("="*80)
        
        train_deltas_artifact_path = BASE_DIR / train_deltas_path
        if not train_deltas_artifact_path.exists():
            # Try as absolute path
            train_deltas_artifact_path = Path(train_deltas_path)
        
        if train_deltas_artifact_path.exists():
            train_deltas_results = process_deltas_artifact(
                train_deltas_artifact_path, 
                threshold=ground_truth_threshold, 
                max_reviews=max_reviews,
                lookup_by_asin=lookup_by_asin,
                lookup_by_text=lookup_by_text
            )
        else:
            logger.warning(f"Train deltas artifact path not found: {train_deltas_path}")
    else:
        # Try default path if not in config
        default_train_deltas_path = BASE_DIR / "07_sgo_training" / "artifacts" / "sgo_train_final_predictions_refined_chars" / "deltas"
        if default_train_deltas_path.exists():
            logger.info("\n" + "="*80)
            logger.info("STEP 5: Processing Train Deltas Files (Train Post-SGO) - Using Default Path")
            logger.info("="*80)
            train_deltas_results = process_deltas_artifact(
                default_train_deltas_path, 
                threshold=ground_truth_threshold, 
                max_reviews=max_reviews,
                lookup_by_asin=lookup_by_asin,
                lookup_by_text=lookup_by_text
            )
    
    # Process custom prediction files (files with user_predictions structure)
    custom_prediction_results = []
    custom_prediction_path = recall_config.get('custom_prediction_path')
    if custom_prediction_path:
        logger.info("\n" + "="*80)
        logger.info("STEP 5.5: Processing Custom Prediction Files")
        logger.info("="*80)
        
        custom_prediction_artifact_path = BASE_DIR / custom_prediction_path
        if not custom_prediction_artifact_path.exists():
            # Try as absolute path
            custom_prediction_artifact_path = Path(custom_prediction_path)
        
        if custom_prediction_artifact_path.exists():
            # Use process_post_sgo_artifact since it handles user_predictions structure
            custom_prediction_results = process_post_sgo_artifact(
                custom_prediction_artifact_path, 
                lookup_by_asin, 
                lookup_by_text, 
                lookup_by_baseline_key, 
                max_reviews
            )
        else:
            logger.warning(f"Custom prediction artifact path not found: {custom_prediction_path}")
    
    # Process test deltas files (test post-SGO with ground truth already in file)
    test_deltas_results = []
    test_deltas_path = recall_config.get('test_deltas_path')
    if test_deltas_path:
        logger.info("\n" + "="*80)
        logger.info("STEP 6: Processing Test Deltas Files (Test Post-SGO)")
        logger.info("="*80)
        
        test_deltas_artifact_path = BASE_DIR / test_deltas_path
        if not test_deltas_artifact_path.exists():
            # Try as absolute path
            test_deltas_artifact_path = Path(test_deltas_path)
        
        if test_deltas_artifact_path.exists():
            test_deltas_results = process_deltas_artifact(
                test_deltas_artifact_path, 
                threshold=ground_truth_threshold, 
                max_reviews=max_reviews,
                lookup_by_asin=lookup_by_asin,
                lookup_by_text=lookup_by_text
            )
        else:
            logger.warning(f"Test deltas artifact path not found: {test_deltas_path}")
    else:
        # Try default paths if not in config
        default_test_deltas_paths = [
            BASE_DIR / "07_sgo_training" / "artifacts" / "sgo_train_final_predictions_refined_chars_worst" / "deltas",
            BASE_DIR / "07_sgo_training" / "artifacts" / "sgo_train_final_predictions_refined_chars_gone" / "deltas",
        ]
        for default_test_deltas_path in default_test_deltas_paths:
            if default_test_deltas_path.exists():
                logger.info("\n" + "="*80)
                logger.info(f"STEP 6: Processing Test Deltas Files (Test Post-SGO) - Using Default Path: {default_test_deltas_path}")
                logger.info("="*80)
                test_deltas_results = process_deltas_artifact(
                    default_test_deltas_path, 
                    threshold=ground_truth_threshold, 
                    max_reviews=max_reviews,
                    lookup_by_asin=lookup_by_asin,
                    lookup_by_text=lookup_by_text
                )
                break  # Process first found path, or remove break to process all
    
    # Calculate summary statistics
    logger.info("\n" + "="*80)
    logger.info("SUMMARY STATISTICS")
    logger.info("="*80)
    
    def print_stats(name, results):
        if not results:
            return
        r1 = [r['recall_at_1'] for r in results]
        r3 = [r['recall_at_3'] for r in results]
        r5 = [r['recall_at_5'] for r in results]
        rk = [r['recall_at_k'] for r in results]
        r2k = [r['recall_at_2k'] for r in results]
        rmax3k = [r['recall_at_max_3_k'] for r in results]
        
        logger.info(f"\n{name} Results ({len(results)} reviews):")
        logger.info(f"  Recall@1: Mean={np.mean(r1):.4f}, Median={np.median(r1):.4f}, Std={np.std(r1):.4f}")
        logger.info(f"  Recall@3: Mean={np.mean(r3):.4f}, Median={np.median(r3):.4f}, Std={np.std(r3):.4f}")
        logger.info(f"  Recall@5: Mean={np.mean(r5):.4f}, Median={np.median(r5):.4f}, Std={np.std(r5):.4f}")
        logger.info(f"  Recall@k: Mean={np.mean(rk):.4f}, Median={np.median(rk):.4f}, Std={np.std(rk):.4f}")
        logger.info(f"  Recall@2k: Mean={np.mean(r2k):.4f}, Median={np.median(r2k):.4f}, Std={np.std(r2k):.4f}")
        logger.info(f"  Recall@max(3,k): Mean={np.mean(rmax3k):.4f}, Median={np.median(rmax3k):.4f}, Std={np.std(rmax3k):.4f}")
    
    print_stats("Pre-SGO", pre_sgo_results)
    print_stats("Baseline (History)", baseline_history_results)
    print_stats("Baseline (Backstory)", baseline_backstory_results)
    print_stats("Post-SGO", post_sgo_results)
    print_stats("Custom Predictions", custom_prediction_results)
    print_stats("Train Deltas (Train Post-SGO)", train_deltas_results)
    print_stats("Test Deltas (Test Post-SGO)", test_deltas_results)
    
    # Save results
    output_dir = BASE_DIR / "Metrics and analysis" / "artifacts" / "recall_at_k"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save results
    if pre_sgo_results:
        with open(output_dir / "pre_sgo_recall_results.json", 'w') as f:
            json.dump(pre_sgo_results, f, indent=2, ensure_ascii=False)
    
    if baseline_history_results:
        with open(output_dir / "baseline_history_recall_results.json", 'w') as f:
            json.dump(baseline_history_results, f, indent=2, ensure_ascii=False)
    
    if baseline_backstory_results:
        with open(output_dir / "baseline_backstory_recall_results.json", 'w') as f:
            json.dump(baseline_backstory_results, f, indent=2, ensure_ascii=False)
    
    if post_sgo_results:
        with open(output_dir / "post_sgo_recall_results.json", 'w') as f:
            json.dump(post_sgo_results, f, indent=2, ensure_ascii=False)
    
    if train_deltas_results:
        with open(output_dir / "train_deltas_recall_results.json", 'w') as f:
            json.dump(train_deltas_results, f, indent=2, ensure_ascii=False)
    
    if test_deltas_results:
        with open(output_dir / "test_deltas_recall_results.json", 'w') as f:
            json.dump(test_deltas_results, f, indent=2, ensure_ascii=False)
    
    if custom_prediction_results:
        with open(output_dir / "custom_prediction_recall_results.json", 'w') as f:
            json.dump(custom_prediction_results, f, indent=2, ensure_ascii=False)
    
    # Save summary
    summary = {}
    
    def add_summary(name, results):
        if not results:
            return
        r1 = [r['recall_at_1'] for r in results]
        r3 = [r['recall_at_3'] for r in results]
        r5 = [r['recall_at_5'] for r in results]
        rk = [r['recall_at_k'] for r in results]
        r2k = [r['recall_at_2k'] for r in results]
        rmax3k = [r['recall_at_max_3_k'] for r in results]
        
        summary[name] = {
            'num_reviews': len(results),
            'recall_at_1': {'mean': float(np.mean(r1)), 'median': float(np.median(r1)), 'std': float(np.std(r1))},
            'recall_at_3': {'mean': float(np.mean(r3)), 'median': float(np.median(r3)), 'std': float(np.std(r3))},
            'recall_at_5': {'mean': float(np.mean(r5)), 'median': float(np.median(r5)), 'std': float(np.std(r5))},
            'recall_at_k': {'mean': float(np.mean(rk)), 'median': float(np.median(rk)), 'std': float(np.std(rk))},
            'recall_at_2k': {'mean': float(np.mean(r2k)), 'median': float(np.median(r2k)), 'std': float(np.std(r2k))},
            'recall_at_max_3_k': {'mean': float(np.mean(rmax3k)), 'median': float(np.median(rmax3k)), 'std': float(np.std(rmax3k))},
        }
    
    add_summary('pre_sgo', pre_sgo_results)
    add_summary('baseline_history', baseline_history_results)
    add_summary('baseline_backstory', baseline_backstory_results)
    add_summary('post_sgo', post_sgo_results)
    add_summary('custom_predictions', custom_prediction_results)
    add_summary('train_deltas_post_sgo', train_deltas_results)
    add_summary('test_deltas_post_sgo', test_deltas_results)
    
    with open(output_dir / "recall_summary.json", 'w') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    logger.info(f"\n✓ Saved results to: {output_dir}")
    
    # Create visualizations
    # Use test_deltas_results as post_sgo_results for comparison if available, otherwise use train_deltas_results or post_sgo_results
    post_sgo_for_viz = test_deltas_results if test_deltas_results else (train_deltas_results if train_deltas_results else post_sgo_results)
    create_visualizations(pre_sgo_results, baseline_history_results, baseline_backstory_results, 
                         post_sgo_for_viz, output_dir)
    
    # Create tribe-wise visualizations
    # Use test_deltas_results for comparison if available, otherwise use train_deltas_results or post_sgo_results
    post_sgo_for_tribe_viz = test_deltas_results if test_deltas_results else (train_deltas_results if train_deltas_results else post_sgo_results)
    if pre_sgo_results and post_sgo_for_tribe_viz:
        logger.info("Creating tribe-wise recall visualizations...")
        create_tribe_wise_visualizations(pre_sgo_results, post_sgo_for_tribe_viz, output_dir)
        logger.info("✓ Created tribe-wise visualization graphs")
    
    # Upload to WandB if run is available
    if run:
        logger.info("\nUploading results to WandB...")
        artifact_type = config.get("artifact_type", "result")
        log_artifact(
            run=run,
            artifact_name=config.get("output_artifacts", {}).get("recall_at_k", "recall_at_k_results"),
            artifact_type=artifact_type,
            artifact_path=str(output_dir),
            metadata={"description": "Recall@1, @3, @5, @k metrics using train_set ground truth"}
        )
    
    logger.info("\n" + "="*80)
    logger.info("✓ Recall calculation complete!")
    logger.info("="*80)
    
    if run:
        finish_run(run)


if __name__ == "__main__":
    main()
