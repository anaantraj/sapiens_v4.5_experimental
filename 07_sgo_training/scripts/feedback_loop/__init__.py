"""
Feedback Loop Module - CTO Behavior v1

Implements the iterative feedback loop for SGO training.
Main component: CTO_Behavior_v1 - Corrects predictions based on delta feedback.
"""

from .batch_correction_pipeline import process_single_batch, get_user_characteristics_for_review, map_category_to_main_category

__all__ = [
    'process_single_batch',
    'get_user_characteristics_for_review',
    'map_category_to_main_category',
]




