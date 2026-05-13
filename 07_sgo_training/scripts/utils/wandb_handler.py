"""
W&B Handler Module
Wrapper for wandb_utils functions to provide a clean interface for the modularized code.
"""
import sys
import logging
from pathlib import Path

# Calculate project root (for error messages and fallback path setup)
# Path calculation: scripts/utils/wandb_handler.py -> scripts -> 07_post_sgo_predictions -> project_root
project_root = Path(__file__).parent.parent.parent.parent.resolve()

# Note: Project root should already be in sys.path from main.py
# This is just a fallback in case wandb_handler is imported directly
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Import from project-level utils (lazy import to avoid circular dependencies)
_wandb_utils = None

def _get_wandb_utils():
    """Lazy import of wandb_utils to avoid import errors at module load time."""
    global _wandb_utils, project_root
    if _wandb_utils is None:
        # Ensure project root is in sys.path
        if str(project_root) not in sys.path:
            sys.path.insert(0, str(project_root))
        
        # Check if utils directory exists
        utils_dir = project_root / "utils"
        if not utils_dir.exists():
            raise ImportError(
                f"utils directory not found at {utils_dir}. "
                f"Project root: {project_root}. "
                f"Make sure you're running from the correct directory."
            )
        
        try:
            # Ensure project root is first in sys.path to prioritize it over local utils
            project_root_str = str(project_root)
            scripts_path = str(Path(__file__).parent.parent.resolve())
            
            # Remove both from path temporarily
            project_root_in_path = project_root_str in sys.path
            scripts_in_path = scripts_path in sys.path
            
            if project_root_in_path:
                sys.path.remove(project_root_str)
            if scripts_in_path:
                sys.path.remove(scripts_path)
            
            # Clear any cached utils modules to force re-import
            modules_to_remove = [k for k in sys.modules.keys() if k.startswith('utils')]
            for mod_name in modules_to_remove:
                del sys.modules[mod_name]
            
            # Add project root FIRST, then scripts (so project root utils takes precedence)
            sys.path.insert(0, project_root_str)
            if scripts_in_path:
                sys.path.insert(1, scripts_path)  # Add scripts back but after project root
            
            try:
                # Now import should get project root utils
                import utils as project_utils
                
                # Verify we got the right one by checking if it has the expected functions
                utils_file = getattr(project_utils, '__file__', '')
                if not hasattr(project_utils, 'load_config') or '07_post_sgo_predictions' in str(utils_file):
                    raise ImportError(
                        f"Imported wrong utils module. "
                        f"Module file: {utils_file}. "
                        f"Expected project root utils. "
                        f"Project root: {project_root_str}"
                    )
                
                # Get functions from project root utils (exported in utils/__init__.py)
                load_config = project_utils.load_config
                get_stage_config = project_utils.get_stage_config
                get_openai_config = project_utils.get_openai_config
                _init_wandb_run = project_utils.init_wandb_run
                _finish_run = project_utils.finish_run
                use_artifact = project_utils.use_artifact
                _log_artifact = project_utils.log_artifact
                log_metrics = project_utils.log_metrics
                log_summary = project_utils.log_summary
                link_to_registry = project_utils.link_to_registry
                get_artifact_dir = project_utils.get_artifact_dir
                
                # create_comprehensive_artifact_metadata is not in __init__.py, access via wandb_utils submodule
                if hasattr(project_utils, 'wandb_utils'):
                    create_comprehensive_artifact_metadata = project_utils.wandb_utils.create_comprehensive_artifact_metadata
                else:
                    # Fallback: import the module directly using importlib
                    import importlib
                    wandb_utils_module = importlib.import_module('utils.wandb_utils')
                    create_comprehensive_artifact_metadata = wandb_utils_module.create_comprehensive_artifact_metadata
            finally:
                # Restore original path order
                if project_root_str in sys.path:
                    sys.path.remove(project_root_str)
                if scripts_path in sys.path:
                    sys.path.remove(scripts_path)
                
                # Restore in original order
                if project_root_in_path:
                    sys.path.insert(0, project_root_str)
                if scripts_in_path:
                    sys.path.insert(0 if not project_root_in_path else 1, scripts_path)
            _wandb_utils = {
                'load_config': load_config,
                'get_stage_config': get_stage_config,
                'get_openai_config': get_openai_config,
                'init_wandb_run': _init_wandb_run,
                'finish_run': _finish_run,
                'use_artifact': use_artifact,
                'log_artifact': _log_artifact,
                'log_metrics': log_metrics,
                'log_summary': log_summary,
                'link_to_registry': link_to_registry,
                'get_artifact_dir': get_artifact_dir,
                'create_comprehensive_artifact_metadata': create_comprehensive_artifact_metadata
            }
        except ImportError as e:
            error_msg = (
                f"Failed to import utils.wandb_utils: {e}\n"
                f"Project root: {project_root}\n"
                f"Utils dir exists: {utils_dir.exists()}\n"
                f"Utils dir: {utils_dir}\n"
                f"PYTHONPATH (first 5): {sys.path[:5]}\n"
                f"Make sure project root ({project_root}) is in PYTHONPATH or accessible."
            )
            logging.error(error_msg)
            raise ImportError(error_msg)
    return _wandb_utils


def init_wandb_run(run_name=None, stage=None, job_type=None, config=None):
    """
    Initialize a W&B run.
    
    Args:
        run_name: Optional run name (defaults to stage-based name)
        stage: Stage name (e.g., "07_sgo_training") - REQUIRED, no default
        job_type: Job type identifier
        config: Optional config dict
    
    Returns:
        W&B run object
    """
    utils = _get_wandb_utils()
    
    if not stage:
        raise ValueError("'stage' parameter is REQUIRED for init_wandb_run - must be provided, no default")
    
    if not run_name:
        run_name = "post_sgo_delta_refinement"
    
    if not job_type:
        job_type = "delta_refinement"
    
    if not config:
        config = utils['get_stage_config'](stage)
    
    return utils['init_wandb_run'](
        run_name=run_name,
        stage=stage,
        job_type=job_type,
        config=config
    )


def finish_run(run):
    """
    Finish a W&B run.
    
    Args:
        run: W&B run object
    """
    if run:
        utils = _get_wandb_utils()
        utils['finish_run'](run)


def log_output_artifact(run, output_base_dir, artifact_name=None, description=None):
    """
    Log output artifact to W&B.
    
    Args:
        run: W&B run object
        output_base_dir: Base directory containing output files
        artifact_name: Optional artifact name (defaults from config)
        description: Optional description
    
    Returns:
        Artifact object if successful, None otherwise
    """
    if not run or not output_base_dir:
        logging.warning("Cannot log artifact: run or output_base_dir is None")
        return None
    
    try:
        # Suppress W&B's verbose logging during artifact upload to avoid spam
        import wandb
        import logging
        
        # Try to suppress W&B's internal logging using Python's logging module
        # (wandb.run.settings.log_level doesn't exist in newer wandb versions)
        wandb_logger = logging.getLogger('wandb')
        original_level = wandb_logger.level if wandb_logger else None
        
        # Temporarily suppress W&B's internal logging (only show errors)
        if wandb_logger:
            wandb_logger.setLevel(logging.ERROR)
        
        utils = _get_wandb_utils()
        
        # Get stage config for artifact name and output stage (from config, REQUIRED)
        stage_config = None
        output_stage = None
        try:
            stage_config = utils['get_stage_config']("07_sgo_training")
            # Get output stage from config - REQUIRED, no fallback
            output_config = stage_config.get("output", {})
            output_stage = output_config.get("stage")
            if not output_stage:
                raise ValueError("'output.stage' is REQUIRED in config.yaml but not found. Please add it to the config file.")
        except ValueError:
            # Re-raise ValueError (config missing)
            raise
        except Exception as e:
            # Only catch non-ValueError exceptions (config loading failed)
            logging.error(f"Could not load stage config: {e}")
            raise ValueError(f"Failed to load config: {e}. 'output.stage' must be specified in config.yaml.")
        
        if not artifact_name and stage_config:
            artifact_name = stage_config.get("output_artifacts", {}).get(
                "sgo_training_results",
                stage_config.get("output_artifacts", {}).get(
                    "sgo_training_test_results",
                    None  # No hardcoded fallback - should be in config
                )
            )
            if not artifact_name:
                logging.warning("Artifact name not found in config.yaml output_artifacts")
        
        if not description:
            description = "SGO training results with corrected predictions"
        
        # Use output_stage from config (REQUIRED - should be set by now)
        if not output_stage:
            raise ValueError("output_stage is None - this should not happen if config is loaded correctly")
        
        artifact_metadata = utils['create_comprehensive_artifact_metadata'](
            stage=output_stage,  # From config.yaml output.stage
            artifact_name=artifact_name,
            additional_metadata={"description": description} if description else None
        )
        
        print(f"[W&B] Uploading artifact '{artifact_name}' (this may take a moment for large directories)...")
        artifact = utils['log_artifact'](
            run=run,
            artifact_name=artifact_name,
            artifact_type="dataset",
            artifact_path=output_base_dir,
            metadata=artifact_metadata
        )
        
        # Restore original log level
        if wandb_logger and original_level is not None:
            wandb_logger.setLevel(original_level)
        
        if artifact:
            utils['link_to_registry'](run, artifact_name, "Sapiens Model")
            print(f"[W&B] ✓ Artifact logged successfully: {artifact_name}")
        
        return artifact
    except Exception as e:
        logging.error(f"Error logging artifact to W&B: {e}")
        import traceback
        logging.error(traceback.format_exc())
        return None

