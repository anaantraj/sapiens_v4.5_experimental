#!/usr/bin/env python3
"""
SGO Training Runner
=====================

Runs the SGO training (delta refinement) step.
This is called from run_unified_simulations.py to perform delta refinement
on initial predictions.

Usage:
    Called from run_unified_simulations.py (not meant to be run directly)
"""

import sys
import logging
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Add 07_sgo_training to path
SGO_TRAINING_DIR = PROJECT_ROOT / "07_sgo_training"
sys.path.insert(0, str(SGO_TRAINING_DIR / "scripts"))

from utils.wandb_utils import get_stage_config, init_wandb_run, finish_run

# Import the main function from SGO training
import importlib.util

def load_sgo_training_main():
    """Dynamically load the main function from 07_sgo_training."""
    sgo_script = SGO_TRAINING_DIR / "scripts" / "main.py"
    spec = importlib.util.spec_from_file_location("sgo_training_main", sgo_script)
    module = importlib.util.module_from_spec(spec)
    sys.modules["sgo_training_main"] = module
    spec.loader.exec_module(module)
    return module.main

def run_sgo_training(run=None, sgo_config=None, hyper_config=None):
    """
    Run SGO training (delta refinement).
    
    Args:
        run: W&B run object (optional, will create if None)
        sgo_config: Configuration dict for SGO training (from 10_running_simulations config)
        hyper_config: Hyperparameters dict (from 10_running_simulations config)
        
    Returns:
        dict: Results with status and output artifact name
    """
    logging.info("=" * 70)
    logging.info("SGO TRAINING: Delta Refinement")
    logging.info("=" * 70)
    
    # Load configuration from 10_running_simulations config
    if sgo_config is None:
        # Load from 10_running_simulations config
        sim_config = get_stage_config("10_simulations")
        sgo_config = sim_config.get("simulation_config", {}).get("sgo_training", {})
        
        if not sgo_config:
            logging.error("❌ sgo_training configuration not found in 10_running_simulations/config.yaml")
            return {"status": "error", "error": "Configuration not found"}
    
    # Check for test_mode flag first
    test_mode = sgo_config.get("test_mode", False)
    
    # Get dataset type from main config (10_running_simulations)
    # If test_mode is enabled, force dataset_type to "test"
    sim_config = get_stage_config("10_simulations")
    if test_mode:
        dataset_type = "test"
        logging.info("🧪 TEST MODE ENABLED - Setting dataset_type to 'test' and using post-SGO predictions with memory")
    else:
        dataset_type = sim_config.get("dataset_type", "train")
    logging.info(f"Dataset type: {dataset_type}")
    
    # Merge hyperparameters: sgo_config hyperparameters override global hyperparameters
    sgo_hyperparams = sgo_config.get("hyperparameters", {})
    if hyper_config:
        # Merge: sgo hyperparameters take precedence
        merged_hyperparams = {**hyper_config, **sgo_hyperparams}
    else:
        merged_hyperparams = sgo_hyperparams
    
    # Update sgo_config with merged hyperparameters
    sgo_config["hyperparameters"] = merged_hyperparams
    sgo_config["dataset_type"] = dataset_type
    sgo_config["test_mode"] = test_mode
    
    # Get parameters for main() function
    max_workers = merged_hyperparams.get("max_workers", hyper_config.get("max_workers", 4) if hyper_config else 4)
    force_reprocess = merged_hyperparams.get("force_reprocess", hyper_config.get("force_reprocess", False) if hyper_config else False)
    only_cluster = merged_hyperparams.get("only_cluster", hyper_config.get("only_cluster", None) if hyper_config else None)
    start_from_cluster = merged_hyperparams.get("start_from_cluster", hyper_config.get("start_from_cluster", None) if hyper_config else None)
    
    # Temporarily patch get_stage_config to return our config when "07_post_sgo_predictions" is requested
    # This allows the stage-specific script to use config from 10_running_simulations
    from utils import wandb_utils
    original_get_stage_config = wandb_utils.get_stage_config
    
    def patched_get_stage_config(stage_name):
        if stage_name == "07_post_sgo_predictions":
            return sgo_config
        return original_get_stage_config(stage_name)
    
    wandb_utils.get_stage_config = patched_get_stage_config
    
    # Initialize W&B run if not provided
    if run is None:
        run = init_wandb_run(
            run_name=f"sgo_training_{dataset_type}_{Path(__file__).stem}",
            stage="10_simulations",
            job_type="sgo_training"
        )
        should_finish = True
    else:
        should_finish = False
    
    try:
        # Load the main function from SGO training
        sgo_training_main = load_sgo_training_main()
        
        # Get output artifact name from config (before calling main)
        output_artifacts = sgo_config.get("output_artifacts", {})
        output_artifact = output_artifacts.get("sgo_training_results", "sgo_training_results_v4")
        
        # Get local output path (where files will be saved)
        # Read output stage from config (defaults to 07_post_sgo_predictions if not set)
        output_stage = sgo_config.get("output", {}).get("stage", "07_post_sgo_predictions")
        from utils.wandb_utils import get_artifact_dir
        local_output_dir = get_artifact_dir(output_stage, output_artifact)
        logging.info(f"📁 Local output directory: {local_output_dir}")
        logging.info(f"   Files will be saved to: {local_output_dir}")
        logging.info(f"   Output stage (from config): {output_stage}")
        
        logging.info(f"Starting SGO training (delta refinement)...")
        logging.info(f"  Max workers: {max_workers}")
        logging.info(f"  Force reprocess: {force_reprocess}")
        logging.info(f"  Only cluster: {only_cluster}")
        logging.info(f"  Start from cluster: {start_from_cluster}")
        
        # Call the main function with parameters
        # The main function signature: main(force_reprocess=True, max_workers=4, only_cluster=None, start_from_cluster=None)
        sgo_training_main(
            force_reprocess=force_reprocess,
            max_workers=max_workers,
            only_cluster=only_cluster,
            start_from_cluster=start_from_cluster
        )
        
        logging.info("✅ SGO training (delta refinement) completed successfully")
        logging.info(f"📁 Output saved locally to: {local_output_dir}")
        logging.info(f"📦 Artifact logged to W&B as: {output_artifact}")
        
        return {
            "status": "success",
            "output_artifact": output_artifact,
            "local_output_dir": str(local_output_dir),
            "dataset_type": dataset_type
        }
            
    except Exception as e:
        logging.error(f"❌ SGO training failed: {e}")
        import traceback
        traceback.print_exc()
        return {
            "status": "error",
            "error": str(e)
        }
    finally:
        # Restore original get_stage_config
        wandb_utils.get_stage_config = original_get_stage_config
        if should_finish and run:
            finish_run(run)

if __name__ == "__main__":
    # This should not be run directly - use run_unified_simulations.py
    logging.warning("This script should be called from run_unified_simulations.py")
    logging.info("Running standalone for testing purposes...")
    
    run_sgo_training()

