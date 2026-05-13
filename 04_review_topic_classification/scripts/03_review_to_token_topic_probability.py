"""
Review to Token Topic Probability
=================================

Stage 04 - Script 3

Purpose:
    Compute token-level topic probabilities for each review.
    This enables fine-grained analysis of which parts of a review
    relate to which topics.

Input:
    - Reviews with topic assignments (from script 02)
    - Topic universe (from Stage 03)

Output:
    - Token-level topic probability mappings
    - Table with columns: review_id, token, token_position, topic_probabilities

W&B Collection: Processed Data
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from utils.wandb_utils import init_wandb_run, log_artifact, link_to_registry

# TODO: Implement token-level topic probability computation

def main():
    """Main function to compute token-level topic probabilities."""
    
    # Initialize W&B run
    run = init_wandb_run(
        run_name="review_to_token_topic_probability",
        stage="04_review_topic_classification",
        config={"description": "Token-level topic probabilities"}
    )
    
    try:
        # TODO: Load classified reviews from script 02
        # TODO: Tokenize reviews
        # TODO: Compute topic probability for each token
        # TODO: Save results to artifacts folder
        
        print("[TODO] Implement token-level topic probability computation")
        print("Expected output: For each token in a review, probability distribution over topics")
        
        # Example output structure:
        # {
        #     "review_id": "R001",
        #     "tokens": [
        #         {"token": "great", "position": 0, "topic_probs": [0.1, 0.8, 0.05, ...]},
        #         {"token": "battery", "position": 1, "topic_probs": [0.9, 0.02, 0.01, ...]},
        #     ]
        # }
        
    finally:
        run.finish()


if __name__ == "__main__":
    main()
