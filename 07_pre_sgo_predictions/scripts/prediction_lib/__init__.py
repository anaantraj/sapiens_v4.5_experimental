"""
Prediction Library Package

This package contains modules for generating initial predictions for product reviews
using LLM models with persona-based context.

Main Components:
- PipelineOrchestrator: Main orchestrator for generating predictions
- DataLoader: Handles loading data from W&B artifacts
- Single review processing, LLM client, metrics, and utilities
"""

# Main classes and functions for easier imports
from .generate_model_predictions import PipelineOrchestrator
from .data_loader import DataLoader
from .llm_client import RateLimiter, get_llm_prediction
from .single_review_prediction import process_single_review
from .prompt_generation import load_prompt, create_enhanced_prompt
from .metrics import calculate_review_metrics, aggregate_metrics
from .config import map_category_to_main_category
from .utils import load_json_file, convert_to_serializable

__all__ = [
    # Main orchestrator
    'PipelineOrchestrator',
    # Data loading
    'DataLoader',
    # LLM client
    'RateLimiter',
    'get_llm_prediction',
    # Review processing
    'process_single_review',
    # Prompt utilities
    'load_prompt',
    'create_enhanced_prompt',
    # Metrics
    'calculate_review_metrics',
    'aggregate_metrics',
    # Config utilities
    'map_category_to_main_category',
    # General utilities
    'load_json_file',
    'convert_to_serializable',
]

