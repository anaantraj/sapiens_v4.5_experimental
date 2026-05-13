"""
Package setup helper - makes sgo_training imports work
This should be imported at the top of files that need sgo_training.* imports
"""
import sys
import types
import importlib
import importlib.util
from pathlib import Path

# Get the scripts directory
scripts_dir = Path(__file__).parent

# Add to path if not already there
if str(scripts_dir) not in sys.path:
    sys.path.insert(0, str(scripts_dir))

# Set up sgo_training package alias if not already set up
if 'sgo_training' not in sys.modules:
    sgo_training_module = types.ModuleType('sgo_training')
    sys.modules['sgo_training'] = sgo_training_module
    
    # Create submodule placeholders first
    sgo_training_module.config = types.ModuleType('sgo_training.config')
    sgo_training_module.utils = types.ModuleType('sgo_training.utils')
    sgo_training_module.llm = types.ModuleType('sgo_training.llm')
    
    # Register them in sys.modules
    sys.modules['sgo_training.config'] = sgo_training_module.config
    sys.modules['sgo_training.utils'] = sgo_training_module.utils
    sys.modules['sgo_training.llm'] = sgo_training_module.llm
    
    # Import llm.llm_client FIRST (before feedback_loop, because batch_correction_pipeline imports it)
    # This ensures the actual module is available when batch_correction_pipeline imports llm_client
    try:
        llm_client_module = importlib.import_module('llm.llm_client')
        # Register it immediately
        sys.modules['sgo_training.llm.llm_client'] = llm_client_module
    except Exception as e:
        # If import fails, create placeholder as fallback
        sys.modules['sgo_training.llm.llm_client'] = types.ModuleType('sgo_training.llm.llm_client')
    
    # Import feedback_loop and sgo_training modules
    # This ensures the actual modules are available when needed
    try:
        batch_proc_module = importlib.import_module('feedback_loop.batch_correction_pipeline')
        cluster_proc_module = importlib.import_module('sgo_training.sgo_training_pipeline_cluster_level')
        # Register them immediately
        sys.modules['sgo_training.feedback_loop.batch_correction_pipeline'] = batch_proc_module
        sys.modules['sgo_training.sgo_training.sgo_training_pipeline_cluster_level'] = cluster_proc_module
        # Also attach to sgo_training_module for direct access
        if not hasattr(sgo_training_module, 'sgo_training'):
            sgo_training_module.sgo_training = types.ModuleType('sgo_training.sgo_training')
        sgo_training_module.sgo_training.sgo_training_pipeline_cluster_level = cluster_proc_module
        # Import and register sgo_training subpackage
        try:
            sgo_training_package = importlib.import_module('sgo_training.sgo_training')
            sys.modules['sgo_training.sgo_training'] = sgo_training_package
            if not hasattr(sgo_training_module, 'sgo_training'):
                sgo_training_module.sgo_training = sgo_training_package
            else:
                # Merge the package into existing module
                for attr in dir(sgo_training_package):
                    if not attr.startswith('_'):
                        setattr(sgo_training_module.sgo_training, attr, getattr(sgo_training_package, attr))
        except Exception as e2:
            # If package import fails, ensure placeholder exists
            if not hasattr(sgo_training_module, 'sgo_training'):
                sgo_training_module.sgo_training = types.ModuleType('sgo_training.sgo_training')
    except Exception as e:
        # If import fails, create placeholders as fallback
        sys.modules['sgo_training.feedback_loop.batch_correction_pipeline'] = types.ModuleType('sgo_training.feedback_loop.batch_correction_pipeline')
        sys.modules['sgo_training.sgo_training.sgo_training_pipeline_cluster_level'] = types.ModuleType('sgo_training.sgo_training.sgo_training_pipeline_cluster_level')
    
    # logic submodules (commonly imported)
    sys.modules['sgo_training.logic.metrics'] = types.ModuleType('sgo_training.logic.metrics')
    sys.modules['sgo_training.logic.embeddings'] = types.ModuleType('sgo_training.logic.embeddings')
    sys.modules['sgo_training.logic.prompts'] = types.ModuleType('sgo_training.logic.prompts')
    
    # Don't create placeholder modules for utils submodules - they will be loaded via relative imports
    # Creating placeholders prevents the actual modules from loading when utils/__init__.py does "from . import ..."
    
    # llm submodules (commonly imported)
    sys.modules['sgo_training.llm.llm_client'] = types.ModuleType('sgo_training.llm.llm_client')
    
    # Now import the actual modules from scripts directory
    # Temporarily ensure scripts_dir is first in path to prioritize local modules
    scripts_in_path = str(scripts_dir) in sys.path
    if scripts_in_path:
        sys.path.remove(str(scripts_dir))
    sys.path.insert(0, str(scripts_dir))  # Put scripts first
    
    try:
        # Import local modules (from scripts directory) and attach submodules
        # Do this carefully to avoid circular imports
        
        # 1. Import config first (usually has no dependencies on other sgo_training modules)
        config_module = importlib.import_module('config')
        sgo_training_module.config = config_module
        sys.modules['sgo_training.config'] = config_module
        
        # 2. Import utils (may import wandb_handler which needs project root utils)
        # Use explicit file-based import to avoid conflicts with project root utils module
        utils_init_file = scripts_dir / 'utils' / '__init__.py'
        if utils_init_file.exists():
            # Create a proper module name that matches the package structure
            utils_module_name = 'sgo_training.utils'
            utils_spec = importlib.util.spec_from_file_location(
                utils_module_name,
                utils_init_file,
                submodule_search_locations=[str(scripts_dir / 'utils')]
            )
            if utils_spec and utils_spec.loader:
                utils_module = importlib.util.module_from_spec(utils_spec)
                # Set __package__ and __name__ to allow relative imports to work
                utils_module.__package__ = 'sgo_training.utils'
                utils_module.__name__ = utils_module_name
                # Set __path__ to the utils directory for submodule discovery
                utils_module.__path__ = [str(scripts_dir / 'utils')]
                # Register the module in sys.modules BEFORE executing so relative imports work
                sys.modules[utils_module_name] = utils_module
                # Ensure scripts_dir/utils is in sys.path so submodules can be found
                utils_dir = str(scripts_dir / 'utils')
                if utils_dir not in sys.path:
                    sys.path.insert(0, utils_dir)
                # Now execute the __init__.py - relative imports should work
                # The submodules will be loaded naturally when utils/__init__.py does "from . import ..."
                utils_spec.loader.exec_module(utils_module)
                # After execution, ensure submodules are properly registered
                # This ensures they're accessible via sgo_training.utils.wandb_handler
                if hasattr(utils_module, 'wandb_handler'):
                    wandb_handler_module = utils_module.wandb_handler
                    wandb_handler_full_name = 'sgo_training.utils.wandb_handler'
                    if wandb_handler_full_name not in sys.modules:
                        sys.modules[wandb_handler_full_name] = wandb_handler_module
            else:
                # Fallback: try importing with explicit package context
                utils_module = importlib.import_module('utils', package=str(scripts_dir))
        else:
            # Fallback to regular import if file doesn't exist
            utils_module = importlib.import_module('utils')
        
        sgo_training_module.utils = utils_module
        sys.modules['sgo_training.utils'] = utils_module
        
        # Replace placeholders with actual submodules
        if hasattr(utils_module, 'io_utils'):
            sys.modules['sgo_training.utils.io_utils'] = utils_module.io_utils
        if hasattr(utils_module, 'logging_utils'):
            sys.modules['sgo_training.utils.logging_utils'] = utils_module.logging_utils
        if hasattr(utils_module, 'wandb_handler'):
            sys.modules['sgo_training.utils.wandb_handler'] = utils_module.wandb_handler
        
        # 3. Import llm package (llm_client was already imported above)
        llm_module = importlib.import_module('llm')
        sgo_training_module.llm = llm_module
        sys.modules['sgo_training.llm'] = llm_module
        
        # Get the already-imported llm_client module (it was imported before placeholders were created)
        llm_client_module = sys.modules.get('sgo_training.llm.llm_client')
        
        # If we got a placeholder, replace it with actual module
        if llm_client_module and not hasattr(llm_client_module, 'process_part_b_batch'):
            # Replace placeholder with actual module
            actual_llm_client = importlib.import_module('llm.llm_client')
            sys.modules['sgo_training.llm.llm_client'] = actual_llm_client
            llm_client_module = actual_llm_client
        
        # Ensure llm_client is attached to parent module
        if llm_client_module:
            sgo_training_module.llm.llm_client = llm_client_module
            llm_module.llm_client = llm_client_module
    finally:
        # Restore original path order if needed
        if str(scripts_dir) in sys.path:
            sys.path.remove(str(scripts_dir))
        if scripts_in_path:
            sys.path.insert(0, str(scripts_dir))

