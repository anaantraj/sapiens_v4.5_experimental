#!/usr/bin/env python3
"""
Integrated Baseline Runner for Run Simulation
==============================================

This script integrates baseline running into the run simulation pipeline.
It calls the baseline runner from 09_baselines which uses its own config
and reads artifacts from modules.
"""

import os
import sys
import logging
import subprocess
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from utils.wandb_utils import get_stage_config

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def run_baselines_from_simulation(baseline_config_override=None):
    """
    Run baselines as part of simulation pipeline.
    
    Args:
        baseline_config_override: Optional dict with models, methods, run_all_combinations
                                 to override 09_baselines config
    """
    logging.info("=" * 70)
    logging.info("Running Baselines (Integrated from Stage 09)")
    logging.info("=" * 70)
    
    # Get baseline config from 09_baselines
    baseline_cfg = get_stage_config("09_baselines")
    
    # Override with simulation config if provided
    if baseline_config_override:
        logging.info("Using baseline config from simulation config:")
        logging.info(f"  Models: {baseline_config_override.get('models', [])}")
        logging.info(f"  Methods: {baseline_config_override.get('methods', [])}")
        logging.info(f"  Run all combinations: {baseline_config_override.get('run_all_combinations', True)}")
    
    # Get the baseline runner script
    project_root = Path(__file__).parent.parent.parent
    baseline_script = project_root / "09_baselines" / "scripts" / "run_baselines.py"
    
    if not baseline_script.exists():
        logging.error(f"❌ Baseline script not found: {baseline_script}")
        return False
    
    # Build command to run baseline script
    cmd = [sys.executable, str(baseline_script)]
    
    # Use override config if provided, otherwise use 09_baselines config
    if baseline_config_override:
        models = baseline_config_override.get("models", [])
        methods = baseline_config_override.get("methods", [])
        run_all_combinations = baseline_config_override.get("run_all_combinations", True)
    else:
        baseline_config = baseline_cfg.get("baseline_config", {})
        models = baseline_config.get("models", ["o3", "claude"])
        methods = baseline_config.get("methods", ["history", "backstory"])
        run_all_combinations = baseline_config.get("run_all_combinations", True)
    
    # Add command-line arguments based on configuration
    if not run_all_combinations:
        # Run only first model + first method
        if models:
            cmd.extend(["--model", models[0]])
        if methods:
            cmd.extend(["--method", methods[0]])
    else:
        # Run all combinations - script will handle it
        # But we can still specify models/methods if only subset wanted
        if len(models) == 1:
            cmd.extend(["--model", models[0]])
        if len(methods) == 1:
            cmd.extend(["--method", methods[0]])
    
    logging.info(f"Running baseline script: {' '.join(cmd)}")
    
    try:
        # Set environment variable to save artifacts in stage 10
        env = os.environ.copy()
        env["SAVE_TO_STAGE_10"] = "true"
        
        # Run the baseline script
        result = subprocess.run(
            cmd,
            cwd=str(project_root),
            check=True,
            capture_output=False,
            env=env
        )
        
        if result.returncode == 0:
            logging.info("✅ Baselines completed successfully")
            return True
        else:
            logging.error(f"❌ Baselines failed with return code {result.returncode}")
            return False
    
    except subprocess.CalledProcessError as e:
        logging.error(f"❌ Error running baselines: {e}")
        return False
    except Exception as e:
        logging.error(f"❌ Unexpected error: {e}")
        return False

if __name__ == "__main__":
    success = run_baselines_from_simulation()
    sys.exit(0 if success else 1)


