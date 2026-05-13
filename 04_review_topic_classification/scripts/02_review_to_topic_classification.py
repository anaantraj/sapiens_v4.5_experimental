"""
Review to Topic Classification
==============================

Stage 04 - Script 2

Purpose:
    Classify each review to its assigned topic(s) based on 
    the topic probabilities from script 01.

Input:
    - User review topic probabilities (from script 01)
    - Topic universe (from Stage 03)

Output:
    - review_topic_classification artifact
    - Table with columns: review_id, assigned_topics, confidence_scores

W&B Collection: Processed Data
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from utils.wandb_utils import init_wandb_run, log_artifact, link_to_registry

# TODO: Implement review to topic classification logic
# This script takes the topic probabilities and assigns topics to reviews

def main():
    """Main function to classify reviews to topics."""
    
    # Initialize W&B run
    run = init_wandb_run(
        run_name="review_to_topic_classification",
        stage="04_review_topic_classification",
        config={"description": "Classify reviews to topics"}
    )
    
    try:
        # TODO: Load topic probabilities from script 01 output
        # TODO: Apply classification logic (threshold-based or argmax)
        # TODO: Save results to artifacts folder
        
        print("[TODO] Implement review to topic classification")
        print("Expected output: review_id -> [assigned_topics] with confidence scores")
        
        # Example output structure:
        # {
        #     "review_id": "R001",
        #     "assigned_topics": [3, 7, 12],
        #     "confidence_scores": [0.85, 0.72, 0.61]
        # }
        
    finally:
        run.finish()


if __name__ == "__main__":
    main()
