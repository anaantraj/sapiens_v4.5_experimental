#!/usr/bin/env python3
"""
Shared utilities for fuzzy topic matching
"""

import numpy as np
from typing import Dict, List, Optional
from difflib import SequenceMatcher
import os
import yaml
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent.parent
import sys
sys.path.insert(0, str(BASE_DIR))

try:
    from dotenv import load_dotenv
    load_dotenv(BASE_DIR / ".env", override=True)
except ImportError:
    pass

from utils.openai_client import create_openai_client
from utils.wandb_utils import get_openai_config

# Load config
with open(BASE_DIR / 'Metrics and analysis/config.yaml', 'r') as f:
    config = yaml.safe_load(f)

FUZZY_MATCH_THRESHOLD = config['metrics'].get('fuzzy_match_threshold', 0.8)
EMBEDDING_SIMILARITY_THRESHOLD = config['metrics'].get('embedding_similarity_threshold', 0.8)

# OpenAI setup
openai_cfg = get_openai_config()
openai_api_key = os.environ.get('OPENAI_API_KEY') or openai_cfg.get('api_key')

openai_client = None
if openai_api_key:
    try:
        openai_client = create_openai_client(openai_config=openai_cfg, timeout=120.0)
    except:
        pass

EMBEDDING_MODEL = config.get('openai', {}).get('embedding_model')


def normalize_topic_name(topic: str) -> str:
    """Normalize topic name for consistent matching (case-insensitive, strip whitespace)."""
    return topic.strip().lower()


def string_similarity(s1: str, s2: str) -> float:
    """Calculate string similarity using SequenceMatcher (0-1, where 1 is identical)."""
    if not s1 or not s2:
        return 0.0
    s1_norm = normalize_topic_name(s1)
    s2_norm = normalize_topic_name(s2)
    
    # Exact match
    if s1_norm == s2_norm:
        return 1.0
    
    # Check if one contains the other (high similarity)
    if s1_norm in s2_norm or s2_norm in s1_norm:
        return 0.85
    
    # SequenceMatcher similarity
    similarity = SequenceMatcher(None, s1_norm, s2_norm).ratio()
    
    # Word overlap (Jaccard similarity)
    words1 = set(s1_norm.split())
    words2 = set(s2_norm.split())
    if words1 and words2:
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        jaccard = intersection / union if union > 0 else 0.0
        # Combine both metrics
        similarity = max(similarity, jaccard * 0.9)
    
    return similarity


def find_fuzzy_match(theme: str, topic_list: List[str], threshold: float = None) -> Optional[str]:
    """Find best matching topic using fuzzy string matching."""
    if threshold is None:
        threshold = FUZZY_MATCH_THRESHOLD
    
    best_match = None
    best_similarity = 0.0
    
    for topic in topic_list:
        similarity = string_similarity(theme, topic)
        if similarity > best_similarity:
            best_similarity = similarity
            best_match = topic
    
    if best_similarity >= threshold:
        return best_match
    return None


def get_topic_embedding(topic_name: str, cache: Dict[str, List[float]] = None) -> Optional[List[float]]:
    """Get embedding for a topic with caching."""
    if cache is None:
        cache = {}
    if topic_name in cache:
        return cache[topic_name]
    
    if not openai_client:
        return None
    if not EMBEDDING_MODEL:
        return None
    
    try:
        response = openai_client.embeddings.create(
            input=[str(topic_name)],
            model=EMBEDDING_MODEL
        )
        embedding = response.data[0].embedding
        cache[topic_name] = embedding
        return embedding
    except Exception as e:
        return None


def find_embedding_match(theme: str, topic_list: List[str], threshold: float = None, 
                        embedding_cache: Dict[str, List[float]] = None) -> Optional[str]:
    """Find best matching topic using embedding similarity."""
    if threshold is None:
        threshold = EMBEDDING_SIMILARITY_THRESHOLD
    
    if not openai_client:
        return None
    
    if embedding_cache is None:
        embedding_cache = {}
    
    theme_emb = get_topic_embedding(theme, embedding_cache)
    if not theme_emb:
        return None
    
    best_match = None
    best_similarity = 0.0
    
    for topic in topic_list:
        topic_emb = get_topic_embedding(topic, embedding_cache)
        if not topic_emb:
            continue
        
        # Cosine similarity
        arr1 = np.array(theme_emb)
        arr2 = np.array(topic_emb)
        norm1 = np.linalg.norm(arr1)
        norm2 = np.linalg.norm(arr2)
        
        if norm1 > 0 and norm2 > 0:
            similarity = np.dot(arr1, arr2) / (norm1 * norm2)
            if similarity > best_similarity:
                best_similarity = similarity
                best_match = topic
    
    if best_similarity >= threshold:
        return best_match
    return None


def normalize_distribution_with_fuzzy(theme_dict: Dict[str, float], topic_list: List[str], 
                                      topic_to_index: Dict[str, int],
                                      embedding_cache: Dict[str, List[float]] = None) -> np.ndarray:
    """Convert theme dict to normalized probability distribution with fuzzy matching."""
    normalized_topic_to_index = {normalize_topic_name(t): idx for t, idx in topic_to_index.items()}
    
    dist = np.zeros(len(topic_list))
    matched_count = 0
    EPSILON = 1e-10
    
    for theme, score in theme_dict.items():
        normalized_theme = normalize_topic_name(theme)
        
        # Strategy 1: Exact match (normalized)
        if normalized_theme in normalized_topic_to_index:
            idx = normalized_topic_to_index[normalized_theme]
            dist[idx] += float(score)
            matched_count += 1
            continue
        
        # Strategy 2: Fuzzy string matching
        fuzzy_match = find_fuzzy_match(theme, topic_list, FUZZY_MATCH_THRESHOLD)
        if fuzzy_match:
            normalized_fuzzy = normalize_topic_name(fuzzy_match)
            if normalized_fuzzy in normalized_topic_to_index:
                idx = normalized_topic_to_index[normalized_fuzzy]
                dist[idx] += float(score)
                matched_count += 1
                continue
        
        # Strategy 3: Embedding similarity (if available)
        if openai_client:
            emb_match = find_embedding_match(theme, topic_list, EMBEDDING_SIMILARITY_THRESHOLD, embedding_cache)
            if emb_match:
                normalized_emb = normalize_topic_name(emb_match)
                if normalized_emb in normalized_topic_to_index:
                    idx = normalized_topic_to_index[normalized_emb]
                    dist[idx] += float(score)
                    matched_count += 1
                    continue
    
    total = dist.sum()
    if total > EPSILON:
        dist = dist / total
    elif matched_count == 0:
        dist = np.ones(len(topic_list)) / len(topic_list)
    else:
        dist = np.ones(len(topic_list)) / len(topic_list)
    
    return dist
