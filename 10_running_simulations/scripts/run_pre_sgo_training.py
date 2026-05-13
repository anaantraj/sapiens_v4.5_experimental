#!/usr/bin/env python3
"""
Pre SGO Training Runner
========================

Runs the initial predictions generation (pre SGO training step).
This is called from run_unified_simulations.py to generate initial predictions
before SGO training (delta refinement).

Usage:
    Called from run_unified_simulations.py (not meant to be run directly)
"""

import sys
import logging
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Add 07_pre_sgo_predictions to path
PRE_SGO_DIR = PROJECT_ROOT / "07_pre_sgo_predictions"
sys.path.insert(0, str(PRE_SGO_DIR / "scripts"))

from utils.wandb_utils import get_stage_config, init_wandb_run, finish_run

# Import the main function from pre SGO predictions
# We'll import it dynamically to avoid path issues
import importlib.util

def load_pre_sgo_main():
    """Dynamically load the main function from 07_pre_sgo_predictions."""
    pre_sgo_script = PRE_SGO_DIR / "scripts" / "00_generate_train_predictions.py"
    spec = importlib.util.spec_from_file_location("pre_sgo_main", pre_sgo_script)
    module = importlib.util.module_from_spec(spec)
    sys.modules["pre_sgo_main"] = module
    spec.loader.exec_module(module)
    return module.main

def run_pre_sgo_training(run=None, pre_sgo_config=None, hyper_config=None):
    """
    Run pre SGO training (initial predictions generation).
    
    Args:
        run: W&B run object (optional, will create if None)
        pre_sgo_config: Configuration dict for pre SGO training (from 10_running_simulations config)
        hyper_config: Hyperparameters dict (from 10_running_simulations config)
        
    Returns:
        dict: Results with status and output artifact name
    """
    logging.info("=" * 70)
    logging.info("PRE SGO TRAINING: Initial Predictions Generation")
    logging.info("=" * 70)
    
    # Load configuration from 10_running_simulations config
    if pre_sgo_config is None:
        # Load from 10_running_simulations config
        sim_config = get_stage_config("10_simulations")
        pre_sgo_config = sim_config.get("simulation_config", {}).get("pre_sgo_training", {})
        
        if not pre_sgo_config:
            logging.error("❌ pre_sgo_training configuration not found in 10_running_simulations/config.yaml")
            return {"status": "error", "error": "Configuration not found"}
    
    # Get dataset type from main config (10_running_simulations)
    sim_config = get_stage_config("10_simulations")
    dataset_type = sim_config.get("dataset_type", "train")
    logging.info(f"Dataset type: {dataset_type}")
    
    # Merge hyperparameters: pre_sgo_config hyperparameters override global hyperparameters
    pre_sgo_hyperparams = pre_sgo_config.get("hyperparameters", {})
    if hyper_config:
        # Merge: pre_sgo hyperparameters take precedence
        merged_hyperparams = {**hyper_config, **pre_sgo_hyperparams}
    else:
        merged_hyperparams = pre_sgo_hyperparams
    
    # Update pre_sgo_config with merged hyperparameters
    pre_sgo_config["hyperparameters"] = merged_hyperparams
    pre_sgo_config["dataset_type"] = dataset_type
    
    # Temporarily patch get_stage_config to return our config when "07_sgo_training" is requested
    # This allows the stage-specific script to use config from 10_running_simulations
    from utils import wandb_utils
    original_get_stage_config = wandb_utils.get_stage_config
    
    def patched_get_stage_config(stage_name):
        if stage_name == "07_sgo_training":
            return pre_sgo_config
        return original_get_stage_config(stage_name)
    
    wandb_utils.get_stage_config = patched_get_stage_config
    
    # Initialize W&B run if not provided
    if run is None:
        run = init_wandb_run(
            run_name=f"pre_sgo_training_{dataset_type}_{Path(__file__).stem}",
            stage="10_simulations",
            job_type="pre_sgo_training"
        )
        should_finish = True
    else:
        should_finish = False
        # Patch init_wandb_run to return the passed run when called from 00_generate_train_predictions.py
        # This ensures 00_generate_train_predictions.py uses the unified run instead of creating its own
        original_init_wandb_run = wandb_utils.init_wandb_run
        unified_run = run  # Capture the run in closure
        
        def patched_init_wandb_run(*args, **kwargs):
            # If called from within our context, return the unified run
            # Check if this is being called from 00_generate_train_predictions.py context
            import inspect
            frame = inspect.currentframe()
            try:
                # Check the call stack to see if we're being called from 00_generate_train_predictions
                caller_frame = frame.f_back
                if caller_frame:
                    caller_file = caller_frame.f_globals.get('__file__', '')
                    if '00_generate_train_predictions' in caller_file or '01_generate_test_predictions' in caller_file:
                        logging.debug(f"Returning unified run for {caller_file}")
                        return unified_run
            finally:
                del frame
            # Otherwise, use original function
            return original_init_wandb_run(*args, **kwargs)
        
        # Store original for restoration
        wandb_utils.original_init_wandb_run = original_init_wandb_run
        wandb_utils.init_wandb_run = patched_init_wandb_run
    
    try:
        # Load the main function from pre SGO predictions
        pre_sgo_main = load_pre_sgo_main()
        
        # The main function expects argparse args, but we'll call it directly
        # We need to modify it to accept config directly, or we can use a wrapper
        
        # For now, we'll call it with default args and let it load its own config
        # The pre_sgo_main function loads config from 07_sgo_training stage
        # We need to ensure it uses the right config
        
        logging.info("Starting initial predictions generation...")
        
        # Call the main function
        # Note: The original main() uses argparse, so we need to handle this
        # We'll temporarily override sys.argv to avoid argparse issues
        import argparse
        
        # Get target clusters from config (optional) - should be in hyperparameters
        hyperparams = pre_sgo_config.get("hyperparameters", {})
        target_clusters = hyperparams.get("target_clusters", None)
        
        # Fallback to global only_cluster if target_clusters not specified
        if target_clusters is None and hyper_config:
            only_cluster = hyper_config.get("only_cluster", None)
            if only_cluster:
                target_clusters = [only_cluster]
                logging.info(f"Using only_cluster from global hyperparameters: {only_cluster}")
        
        # Validate and normalize target_clusters
        if target_clusters is not None:
            # Handle different input formats
            if isinstance(target_clusters, list):
                # Filter out None values and ensure all are strings
                target_clusters = [str(c) for c in target_clusters if c is not None and str(c).strip()]
            elif isinstance(target_clusters, str):
                target_clusters = [target_clusters.strip()] if target_clusters.strip() else None
            else:
                # If it's a complex object (like nested dict), log warning and skip
                logging.warning(f"⚠️  target_clusters has unexpected type: {type(target_clusters)}. Value: {target_clusters}")
                logging.warning(f"   Skipping target_clusters filter - will process all clusters")
                target_clusters = None
        
        # Temporarily override sys.argv to avoid argparse issues
        original_argv = sys.argv
        sys.argv = ["00_generate_train_predictions.py"]
        if target_clusters and len(target_clusters) > 0:
            # Convert cluster_0 format to segment_0 format if needed (or keep as-is)
            normalized_clusters = []
            for cluster in target_clusters:
                if cluster.startswith("cluster_"):
                    # Convert cluster_0 -> segment_0 for compatibility
                    normalized_clusters.append(cluster.replace("cluster_", "segment_", 1))
                else:
                    normalized_clusters.append(cluster)
            sys.argv.extend(["--clusters"] + normalized_clusters)
        
        try:
            # Get output artifact name from config (before calling main)
            output_artifacts = pre_sgo_config.get("output_artifacts", {})
            base_output = output_artifacts.get("initial_predictions", "initial_predictions_v4")
            
            # Add model suffix if specified
            hyperparams = pre_sgo_config.get("hyperparameters", {})
            model_name = hyperparams.get("model", "o3")
            model_suffix = model_name.replace("-", "_").replace(".", "_")
            output_artifact = f"{base_output}_{model_suffix}"
            
            # Get output stage from config (defaults to "10_running_simulations" for unified runner)
            output_config = pre_sgo_config.get("output", {})
            output_stage = output_config.get("stage", "10_running_simulations")
            
            # Get local output path (where files will be saved)
            from utils.wandb_utils import get_artifact_dir, get_project_root
            local_output_dir = get_artifact_dir(output_stage, output_artifact)
            logging.info(f"📁 Local output directory: {local_output_dir}")
            logging.info(f"   Files will be saved to: {local_output_dir}")
            logging.info(f"   Output stage: {output_stage}")
            
            # Add output_stage to hyperparams so prediction generator can use it
            # This will be passed as config_params to the PipelineOrchestrator
            hyperparams["output_stage"] = output_stage
            
            # Call main() - it will use the patched config from 10_running_simulations
            pre_sgo_main()
            logging.info("✅ Pre SGO training (initial predictions) completed successfully")
            logging.info(f"📁 Output saved locally to: {local_output_dir}")
            logging.info(f"📦 Artifact logged to W&B as: {output_artifact}")
            
            return {
                "status": "success",
                "output_artifact": output_artifact,
                "local_output_dir": str(local_output_dir),
                "dataset_type": dataset_type
            }
        finally:
            sys.argv = original_argv
            # Restore original functions
            wandb_utils.get_stage_config = original_get_stage_config
            if hasattr(wandb_utils, 'original_init_wandb_run'):
                wandb_utils.init_wandb_run = wandb_utils.original_init_wandb_run
                delattr(wandb_utils, 'original_init_wandb_run')
            
    except Exception as e:
        logging.error(f"❌ Pre SGO training failed: {e}")
        import traceback
        traceback.print_exc()
        return {
            "status": "error",
            "error": str(e)
        }
    finally:
        if should_finish and run:
            finish_run(run)

if __name__ == "__main__":
    # This should not be run directly - use run_unified_simulations.py
    logging.warning("This script should be called from run_unified_simulations.py")
    logging.info("Running standalone for testing purposes...")
    
    run_pre_sgo_training()

