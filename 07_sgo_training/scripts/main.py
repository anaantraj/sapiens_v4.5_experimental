import argparse
import os
import re
import sys
import openai
import traceback
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

# CRITICAL: Add project root to sys.path FIRST, before any other imports
# This ensures utils.wandb_utils can be found
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Load .env so OPENAI_API_KEY and OPENAI_BASE_URL (Bedrock) are set
try:
    from dotenv import load_dotenv
    load_dotenv(project_root / ".env", override=True)
except ImportError:
    pass

# Set up package structure for sgo_training imports
import _package_setup

# Modular Imports (now these will work)
from utils.openai_client import create_openai_client
from sgo_training.config import settings
from sgo_training.utils import io_utils, logging_utils, wandb_handler
# Import from sgo_training subpackage
# The sgo_training directory has __init__.py that exports these functions
try:
    from sgo_training.sgo_training import (
        process_micro_cluster_file,
        generate_cluster_aggregate_summary
    )
except ImportError:
    # Fallback: import directly from the module file
    import importlib.util
    import os
    scripts_dir = Path(__file__).parent
    module_path = scripts_dir / "sgo_training" / "sgo_training_pipeline_cluster_level.py"
    spec = importlib.util.spec_from_file_location("sgo_training_pipeline_cluster_level", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    process_micro_cluster_file = module.process_micro_cluster_file
    generate_cluster_aggregate_summary = module.generate_cluster_aggregate_summary
# --- Test Mode Function ---

def _run_test_mode_predictions(
    run, stage_config, input_artifacts, output_artifacts, output_artifact, openai_config,
    force_reprocess, max_workers, only_cluster, start_from_cluster, only_micro=None
):
    """
    Run test mode predictions using memory files from SGO training.
    """
    import openai
    from pathlib import Path
    
    # Add prediction_lib to path
    prediction_lib_path = str(project_root / "07_pre_sgo_predictions" / "scripts")
    if prediction_lib_path not in sys.path:
        sys.path.insert(0, prediction_lib_path)
    
    # Import test mode predictions from local scripts (07_sgo_training/scripts)
    from test_mode_predictions import run_test_mode_predictions
    from prediction_lib.prompt_generation import load_prompt as load_prompt_template
    from prediction_lib.data_loader import DataLoader
    from prediction_lib.llm_client import RateLimiter
    
    scoring_mode = stage_config.get("scoring_mode", "logprobs")
    use_bedrock = (
        scoring_mode not in ["logprobs", "logprobs-without-persona-context"]
        and "bedrock-mantle" in (os.getenv("OPENAI_BASE_URL") or "")
        and bool(stage_config.get("bedrock_model_id"))
    )

    # Set API Key
    if openai_config and openai_config.get('api_key'):
        api_key = openai_config['api_key']
    else:
        import os
        api_key = os.getenv("OPENAI_API_KEY")
    
    if not api_key:
        logging_utils.log_thread_safe("[ERROR] OPENAI_API_KEY not found", 'error')
        return

    if use_bedrock:
        client = create_openai_client(openai_config=openai_config, timeout=120.0)
    else:
        client = openai.OpenAI(api_key=api_key)
    
    # Get output directory
    output_stage = stage_config.get("output", {}).get("stage", "07_sgo_training")
    from utils.wandb_utils import get_artifact_dir
    output_dir = get_artifact_dir(output_stage, output_artifact)
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Download test_micro_cluster_details artifact (will be processed cluster by cluster in test_mode_predictions)
    # Try test_micro_cluster_details first, then ground_truth_test_micro_cluster_details, then fallback to test_data_with_topics
    test_data_artifact = (
        input_artifacts.get("test_micro_cluster_details") or 
        input_artifacts.get("ground_truth_test_micro_cluster_details") or 
        input_artifacts.get("test_data_with_topics")
    )
    if not test_data_artifact:
        logging_utils.log_thread_safe("[ERROR] test_micro_cluster_details, ground_truth_test_micro_cluster_details, or test_data_with_topics not found in config.yaml (required for test_mode)", 'error')
        return
    
    logging_utils.log_thread_safe(f"[TEST MODE] Downloading test data artifact from W&B: {test_data_artifact}", 'info')
    from utils.wandb_utils import use_artifact
    
    # Download artifact - will be processed cluster by cluster, micro cluster by micro cluster
    test_data_path = use_artifact(run, test_data_artifact, artifact_type="dataset")
    if not test_data_path:
        logging_utils.log_thread_safe(f"[ERROR] Could not download artifact from W&B: {test_data_artifact}", 'error')
        return
    
    test_data_path = Path(test_data_path)
    cluster_dirs = [d for d in test_data_path.iterdir() if d.is_dir() and d.name.startswith('cluster_')]
    if not cluster_dirs:
        logging_utils.log_thread_safe("[ERROR] No cluster directories found in test data artifact", 'error')
        return
    
    logging_utils.log_thread_safe(f"[TEST MODE] Found {len(cluster_dirs)} cluster directories in test data artifact - will process cluster by cluster", 'info')
    
    # Download ground truth artifact (for train mode: ground_truth_train_micro_cluster_details_converted)
    # This has folder structure similar to predictions: cluster_X/micro_Y_details.json
    ground_truth_artifact = input_artifacts.get("ground_truth_train_micro_cluster_details_converted")
    ground_truth_path = None
    if ground_truth_artifact:
        logging_utils.log_thread_safe(f"[TEST MODE] Downloading ground truth artifact from W&B: {ground_truth_artifact}", 'info')
        ground_truth_path = use_artifact(run, ground_truth_artifact, artifact_type="dataset")
        if ground_truth_path:
            ground_truth_path = Path(ground_truth_path)
            logging_utils.log_thread_safe(f"[TEST MODE] Downloaded ground truth to: {ground_truth_path}", 'info')
        else:
            logging_utils.log_thread_safe(f"[WARNING] Could not download ground truth artifact: {ground_truth_artifact}", 'warning')
    else:
        logging_utils.log_thread_safe("[TEST MODE] No ground_truth_train_micro_cluster_details_converted found in config - proceeding without ground truth", 'warning')
    
    # Download sgo_training_results_for_main artifact for memory, user chars, category chars, tribe chars (from config)
    sgo_results_artifact = input_artifacts.get("sgo_training_results_for_main")
    if not sgo_results_artifact:
        logging_utils.log_thread_safe(f"[ERROR] sgo_training_results_for_main not found in input_artifacts config (required for test_mode)", 'error')
        return
    
    logging_utils.log_thread_safe(f"[TEST MODE] Downloading {sgo_results_artifact} from W&B for memory and characteristics...", 'info')
    sgo_results_path = use_artifact(run, sgo_results_artifact, artifact_type="dataset")
    if not sgo_results_path:
        logging_utils.log_thread_safe(f"[ERROR] Could not download {sgo_results_artifact} from W&B", 'error')
        return
    
    sgo_results_path = Path(sgo_results_path)
    logging_utils.log_thread_safe(f"[TEST MODE] Downloaded {sgo_results_artifact} to: {sgo_results_path}", 'info')
    
    # Set memory_base_dir to _memory folder in sgo_training_results
    memory_base_dir = str(sgo_results_path / "_memory")
    if not Path(memory_base_dir).exists():
        logging_utils.log_thread_safe(f"[WARNING] Memory directory not found in {sgo_results_artifact}: {memory_base_dir}", 'warning')
        memory_base_dir = None
    else:
        logging_utils.log_thread_safe(f"[TEST MODE] Using memory directory: {memory_base_dir}", 'info')
    
    # Initialize DataLoader for other artifacts (tribe_seeds, user_backstories, etc.)
    file_patterns = stage_config.get("file_patterns", {})
    dataset_pattern_mapping = {}  # Not needed for test mode, but DataLoader expects it
    loader = DataLoader(run, file_patterns, dataset_pattern_mapping)
    
    # Load tribe seed characteristics
    tribe_seed_artifact = input_artifacts.get("tribe_seed_characteristics")
    if not tribe_seed_artifact:
        logging_utils.log_thread_safe("[ERROR] tribe_seed_characteristics not found in config.yaml", 'error')
        return
    
    tribe_seed_data = loader.load_tribe_seeds(tribe_seed_artifact)
    if not tribe_seed_data:
        logging_utils.log_thread_safe("[ERROR] Failed to load tribe seed data", 'error')
        return
    logging_utils.log_thread_safe(f"[TEST MODE] Loaded {len(tribe_seed_data)} tribes", 'info')
    
    # Load user backstories
    user_backstory_artifact = input_artifacts.get("user_backstories")
    if not user_backstory_artifact:
        logging_utils.log_thread_safe("[ERROR] user_backstories not found in config.yaml", 'error')
        return
    
    user_backstories = loader.load_user_backstories(user_backstory_artifact)
    if not user_backstories:
        logging_utils.log_thread_safe("[ERROR] Failed to load user backstories", 'error')
        return
    logging_utils.log_thread_safe(f"[TEST MODE] Loaded backstories for {len(user_backstories)} users", 'info')
    
    # No need for user_tribes - users are already organized by micro cluster in prediction files
    # We'll extract users directly from each micro cluster file and match to persona chars
    
    # Load topic universe
    topic_universe_artifact = input_artifacts.get("topic_universe")
    if not topic_universe_artifact:
        logging_utils.log_thread_safe("[ERROR] topic_universe not found in config.yaml", 'error')
        return
    
    topic_universe = loader.load_topic_universe(topic_universe_artifact)
    if not topic_universe:
        logging_utils.log_thread_safe("[ERROR] Failed to load topic universe", 'error')
        return
    
    # Load category mapping
    category_mapping_artifact = input_artifacts.get("category_mapping")
    if not category_mapping_artifact:
        logging_utils.log_thread_safe("[ERROR] category_mapping not found in config.yaml", 'error')
        return
    
    category_mapping = io_utils.load_category_mapping(run, category_mapping_artifact, file_patterns.get("category_mapping", "category_mapping.json"))
    if not category_mapping:
        logging_utils.log_thread_safe("[ERROR] Failed to load category mapping", 'error')
        return
    
    # Memory base directory is already set from sgo_training_results_V3 above
    # If not set, try fallback to explicitly specified memory artifact
    if not memory_base_dir:
        memory_artifact = input_artifacts.get("sgo_training_results_memory")
        if memory_artifact:
            logging_utils.log_thread_safe(f"[TEST MODE] Loading memory from specified artifact: {memory_artifact}", 'info')
            try:
                from utils.wandb_utils import use_artifact
                # Try as dataset type first (since we uploaded it as dataset)
                memory_artifact_path = use_artifact(run, memory_artifact, artifact_type="dataset")
                if not memory_artifact_path:
                    # Fallback to model type
                    memory_artifact_path = use_artifact(run, memory_artifact, artifact_type="model")
                
                if memory_artifact_path:
                    potential_memory_dir = os.path.join(str(memory_artifact_path), "_memory")
                    if os.path.exists(potential_memory_dir):
                        memory_base_dir = potential_memory_dir
                        logging_utils.log_thread_safe(f"[TEST MODE] Found memory directory from v4 artifact: {memory_base_dir}", 'info')
                    else:
                        logging_utils.log_thread_safe(f"[TEST MODE] Memory directory not found in artifact at: {potential_memory_dir}", 'warning')
            except Exception as e:
                logging_utils.log_thread_safe(f"[TEST MODE] Could not load memory from specified artifact: {e}", 'warning')
    
    # If not found, try to find memory in the output directory structure
    if not memory_base_dir:
        potential_memory_dir = os.path.join(output_dir, "_memory")
        if os.path.exists(potential_memory_dir):
            memory_base_dir = potential_memory_dir
            logging_utils.log_thread_safe(f"[TEST MODE] Found memory directory: {memory_base_dir}", 'info')
    
    # If still not found, try to load from output artifact (fallback)
    if not memory_base_dir:
        logging_utils.log_thread_safe("[TEST MODE] Memory directory not found in output. Looking for previous SGO training results in W&B...", 'info')
        try:
            from utils.wandb_utils import use_artifact
            # Use the same artifact name to get previous training results
            previous_training_path = use_artifact(run, output_artifact, artifact_type="model")
            if previous_training_path:
                potential_memory_dir = os.path.join(str(previous_training_path), "_memory")
                if os.path.exists(potential_memory_dir):
                    memory_base_dir = potential_memory_dir
                    logging_utils.log_thread_safe(f"[TEST MODE] Found memory directory from W&B artifact: {memory_base_dir}", 'info')
        except Exception as e:
            logging_utils.log_thread_safe(f"[TEST MODE] Could not load memory from W&B: {e}", 'warning')
    
    if not memory_base_dir:
        logging_utils.log_thread_safe("[TEST MODE] Memory directory not found. Proceeding without memory (predictions will use base persona only)", 'warning')
    
    # Load prompt template from config
    prompt_config = stage_config.get("prompt", {})
    prompt_filename = prompt_config.get("test_prediction", "test_prediction_with_memory_prompt.txt")
    
    # Resolve prompt path from config (directory is relative to stage directory)
    prompt_dir_name = prompt_config.get("directory", "prompts")
    stage_dir = Path(__file__).parent.parent  # 07_sgo_training directory
    prompt_dir = stage_dir / prompt_dir_name
    prompt_path = prompt_dir / prompt_filename
    
    if not prompt_path.exists():
        logging_utils.log_thread_safe(f"[ERROR] Prompt file not found: {prompt_path}", 'error')
        return
    
    prompt_template = load_prompt_template(prompt_dir, prompt_filename)
    logging_utils.log_thread_safe(f"[TEST MODE] Loaded prompt template from: {prompt_path}", 'info')
    
    # Get hyperparameters
    hyperparams = stage_config.get("hyperparameters", {})
    model_name = hyperparams.get("analysis_model", "o3")
    max_tokens = hyperparams.get("max_tokens", 2000)
    temperature = hyperparams.get("temperature", 0.7)
    max_retries = hyperparams.get("max_retries", 3)
    min_request_interval = hyperparams.get("min_request_interval", 0.1)
    num_workers = hyperparams.get("num_workers", max_workers)
    default_micro_id = hyperparams.get("default_micro_id", "micro_0")
    no_memory_message = hyperparams.get("no_memory_message", "None - this is the first prediction for this persona")
    
    # Get scoring mode and theme prediction model from config (prefer logprobs_model_id when set)
    theme_prediction_model = stage_config.get("logprobs_model_id") or stage_config.get("theme_prediction_model", "gpt-4o-mini")
    if use_bedrock:
        model_name = stage_config.get("bedrock_model_id", model_name)
        # Theme model (logprobs_model_id) must support API logprobs via Chat Completions.
    
    rate_limiter = RateLimiter(min_interval=min_request_interval)
    
    # Get file patterns
    file_patterns = stage_config.get("file_patterns", {})
    
    # Determine target clusters and micro clusters
    target_clusters = None
    target_micro = None
    if only_cluster:
        target_clusters = [only_cluster]
        if only_micro:
            target_micro = only_micro
            logging_utils.log_thread_safe(f"[TEST MODE] Filtering to {only_cluster}/{only_micro}", 'info')
    
    # Run test mode predictions (from local test_mode_predictions.py)
    # Will load test data cluster by cluster from predictions artifact
    # Users are extracted directly from each micro cluster file - no user_tribes mapping needed
    run_test_mode_predictions(
        run=run,
        test_data_path=str(test_data_path),  # Pass path - will load cluster by cluster
        ground_truth_path=str(ground_truth_path) if ground_truth_path else None,  # Pass ground truth path (folder structure)
        tribe_seed_data=tribe_seed_data,
        user_backstories=user_backstories,
        topic_universe=topic_universe,
        category_mapping=category_mapping,
        memory_base_dir=memory_base_dir if memory_base_dir else "",
        output_dir=str(output_dir),
        prompt_template=prompt_template,
        client=client,
        rate_limiter=rate_limiter,
        model_name=model_name,
        max_tokens=max_tokens,
        temperature=temperature,
        max_retries=max_retries,
        num_workers=num_workers,
        target_clusters=target_clusters,
        target_micro=target_micro,  # Pass micro cluster filter
        user_to_tribe_map=None,  # Not needed - users extracted from prediction files
        scoring_mode=scoring_mode,  # Pass scoring mode to get logprobs
        theme_prediction_model=theme_prediction_model  # Pass theme prediction model
    )
    
    # Log output artifact with correct name for test mode
    test_output_artifact = output_artifacts.get("sgo_training_test_results", "sgo_training_test_sampled_2k_users_10k_reviews")
    wandb_handler.log_output_artifact(
        run, 
        str(output_dir),
        artifact_name=test_output_artifact,
        description="Test mode predictions with v4 memory context"
    )
    logging_utils.log_thread_safe("[TEST MODE] Test mode predictions completed", 'success')

# --- Worker Function ---

def process_single_file_wrapper(args_tuple):
    """
    Wrapper function for parallel processing.
    Creates its own OpenAI client for thread safety.
    Args:
        args_tuple: (cluster_id, micro_cluster_id, filepath, category_mapping, user_chars)
    """
    cluster_id, micro_cluster_id, filepath, category_mapping, user_chars = args_tuple
    
    try:
        logging_utils.log_thread_safe(f"[START] Starting processing: {cluster_id} / {micro_cluster_id}", 'info')
        
        # Create a new client for this thread (thread-safe).
        # Logprob modes stay on the existing direct path; confidence mode can use Bedrock.
        use_bedrock = (
            getattr(settings, 'SCORING_MODE', 'confidence') not in ["logprobs", "logprobs-without-persona-context"]
            and "bedrock-mantle" in (os.getenv("OPENAI_BASE_URL") or "")
        )
        if use_bedrock:
            client = create_openai_client(api_key=settings.OPENAI_API_KEY, timeout=120.0)
        else:
            client = openai.OpenAI(api_key=settings.OPENAI_API_KEY)
        
        # Process the file using the cluster processor pipeline
        result = process_micro_cluster_file(
            filepath, 
            client, 
            user_chars, 
            category_mapping
        )
        
        if result:
            logging_utils.log_thread_safe(f"[SUCCESS] Completed: {cluster_id} / {micro_cluster_id}", 'success')
            return {'status': 'success', 'cluster_id': cluster_id, 'micro_cluster_id': micro_cluster_id, 'filepath': filepath}
        else:
            logging_utils.log_thread_safe(f"[WARNING] No result returned: {cluster_id} / {micro_cluster_id}", 'warning')
            return {'status': 'no_result', 'cluster_id': cluster_id, 'micro_cluster_id': micro_cluster_id, 'filepath': filepath}
    
    except Exception as e:
        error_msg = f"[ERROR] Error processing {cluster_id} / {micro_cluster_id}: {str(e)}"
        logging_utils.log_thread_safe(error_msg, 'error')
        logging_utils.log_thread_safe(traceback.format_exc(), 'error')
        return {'status': 'error', 'cluster_id': cluster_id, 'micro_cluster_id': micro_cluster_id, 'filepath': filepath, 'error': str(e)}

# --- Main Orchestrator ---

def main(force_reprocess=True, max_workers=4, only_cluster=None, start_from_cluster=None, only_micro=None):
    """
    Main function to process all micro clusters in parallel.
    """
    # 1. Load configuration
    import sys
    from pathlib import Path
    project_root = Path(__file__).parent.parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    
    # Import from project root utils (not local utils package)
    # Clear any cached utils modules to ensure we get project root utils
    scripts_path = str(Path(__file__).parent.resolve())
    modules_to_remove = [k for k in sys.modules.keys() if k == 'utils' or k.startswith('utils.')]
    for mod_name in modules_to_remove:
        if '07_post_sgo_predictions' in str(getattr(sys.modules[mod_name], '__file__', '')):
            del sys.modules[mod_name]
    
    # Ensure project root is first in path
    if str(project_root) in sys.path:
        sys.path.remove(str(project_root))
    sys.path.insert(0, str(project_root))
    
    # Now import from project root utils
    from utils.wandb_utils import (
        load_config, get_stage_config, get_openai_config,
        use_artifact, get_artifact_dir
    )
    
    config = load_config()
    stage_config = get_stage_config("07_sgo_training")
    openai_config = get_openai_config()
    
    # Check for test_mode
    test_mode = stage_config.get("test_mode", False)
    
    # 2. Initialize W&B run
    if test_mode:
        run_name = "sgo_test_mode_predictions_v4"
        job_type = "test_mode_predictions"
    else:
        run_name = "sgo_delta_refinement"
        job_type = "delta_refinement"
    
    run = wandb_handler.init_wandb_run(
        run_name=run_name,
        stage="07_sgo_training",
        job_type=job_type,
        config=stage_config
    )
    
    # 3. Load artifacts from W&B ONLY (no local fallbacks)
    # All artifacts must be specified in config.yaml
    input_artifacts = stage_config.get("input_artifacts", {})
    if not input_artifacts:
        logging_utils.log_thread_safe("[ERROR] No input_artifacts found in config.yaml", 'error')
        wandb_handler.finish_run(run)
        return
    
    # Output artifact
    output_artifacts = stage_config.get("output_artifacts", {})
    output_artifact = output_artifacts.get("sgo_training_results")
    if not output_artifact:
        logging_utils.log_thread_safe("[ERROR] sgo_training_results not found in output_artifacts in config.yaml", 'error')
        wandb_handler.finish_run(run)
        return
    
    # If test_mode is enabled, handle test predictions separately
    if test_mode:
        logging_utils.log_thread_safe("[TEST MODE] Test mode enabled - running test predictions with memory", 'info')
        _run_test_mode_predictions(
            run, stage_config, input_artifacts, output_artifacts, output_artifact, openai_config, 
            force_reprocess, max_workers, only_cluster, start_from_cluster, only_micro
        )
        wandb_handler.finish_run(run)
        return
    
    # Load input predictions from W&B ONLY (no local fallbacks)
    # Use dataset_type to determine which artifact to load
    dataset_type = stage_config.get("dataset_type", "train")  # Default to "train" if not specified
    input_artifacts = stage_config.get("input_artifacts", {})
    
    if dataset_type == "train":
        initial_predictions_artifact = input_artifacts.get("initial_predictions_train")
        if not initial_predictions_artifact:
            logging_utils.log_thread_safe("[ERROR] initial_predictions_train not found in config.yaml", 'error')
            wandb_handler.finish_run(run)
            return
    else:  # test
        initial_predictions_artifact = input_artifacts.get("initial_predictions_test")
        if not initial_predictions_artifact:
            logging_utils.log_thread_safe("[ERROR] initial_predictions_test not found in config.yaml", 'error')
            wandb_handler.finish_run(run)
            return
    
    # Check if using local path (starts with / or ./) or W&B artifact
    if initial_predictions_artifact.startswith('/') or initial_predictions_artifact.startswith('./'):
        # Use local path
        prediction_path = Path(initial_predictions_artifact).resolve()
        if not prediction_path.exists():
            logging_utils.log_thread_safe(f"[ERROR] Local path does not exist: {prediction_path}", 'error')
            wandb_handler.finish_run(run)
            return
        logging_utils.log_thread_safe(f"[INFO] Using local {dataset_type} predictions: {prediction_path}", 'info')
    else:
        # Download from W&B
        logging_utils.log_thread_safe(f"[INFO] Loading {dataset_type} predictions from W&B: {initial_predictions_artifact}", 'info')
        prediction_path = use_artifact(run, initial_predictions_artifact, artifact_type="dataset")
        if not prediction_path:
            logging_utils.log_thread_safe(f"[ERROR] Could not load {dataset_type} predictions artifact from W&B: {initial_predictions_artifact}", 'error')
            wandb_handler.finish_run(run)
            return
        logging_utils.log_thread_safe(f"[INFO] Using W&B {dataset_type} predictions: {prediction_path}", 'info')
    
    settings.PATHS['PREDICTION_INPUT_DIR'] = str(prediction_path)
    
    # User backstory - W&B ONLY (no local fallback)
    user_backstory_artifact = input_artifacts.get("user_backstories")
    if not user_backstory_artifact:
        logging_utils.log_thread_safe("[ERROR] user_backstories not found in config.yaml", 'error')
        wandb_handler.finish_run(run)
        return
    
    logging_utils.log_thread_safe(f"[INFO] Loading user backstory from W&B: {user_backstory_artifact}", 'info')
    backstory_path = use_artifact(run, user_backstory_artifact, artifact_type="dataset")
    if not backstory_path:
        logging_utils.log_thread_safe(f"[ERROR] Could not load user backstory artifact from W&B: {user_backstory_artifact}", 'error')
        wandb_handler.finish_run(run)
        return
    
    # Find the user characteristics file in the artifact
    user_chars_file_pattern = stage_config.get("file_patterns", {}).get("user_backstories", "user_overall_characteristics.json")
    if backstory_path.is_file():
        settings.PATHS['USER_CHARS_FILE'] = str(backstory_path)
    elif backstory_path.is_dir():
        user_chars_file = backstory_path / user_chars_file_pattern
        if user_chars_file.exists():
            settings.PATHS['USER_CHARS_FILE'] = str(user_chars_file)
        else:
            logging_utils.log_thread_safe(f"[ERROR] User backstory file '{user_chars_file_pattern}' not found in artifact: {backstory_path}", 'error')
            wandb_handler.finish_run(run)
            return
    else:
        logging_utils.log_thread_safe(f"[ERROR] User backstory artifact path is invalid: {backstory_path}", 'error')
        wandb_handler.finish_run(run)
        return
    
    logging_utils.log_thread_safe(f"[INFO] Using W&B user backstory: {settings.PATHS['USER_CHARS_FILE']}", 'info')
    
    # Topic universe - W&B ONLY (no local fallback)
    topic_universe_artifact = input_artifacts.get("topic_universe")
    if not topic_universe_artifact:
        logging_utils.log_thread_safe("[ERROR] topic_universe not found in config.yaml", 'error')
        wandb_handler.finish_run(run)
        return
    
    logging_utils.log_thread_safe(f"[INFO] Loading topic universe from W&B: {topic_universe_artifact}", 'info')
    topic_path = use_artifact(run, topic_universe_artifact, artifact_type="dataset")
    if not topic_path:
        logging_utils.log_thread_safe(f"[ERROR] Could not load topic universe artifact from W&B: {topic_universe_artifact}", 'error')
        wandb_handler.finish_run(run)
        return
    
    # Find the topic universe file in the artifact
    topic_file_pattern = stage_config.get("file_patterns", {}).get("topic_universe", "topic_universe.json")
    if topic_path.is_file():
        settings.PATHS['THEMES_FILE'] = str(topic_path)
    elif topic_path.is_dir():
        themes_file = topic_path / topic_file_pattern
        if not themes_file.exists():
            # Try alternative filename
            themes_file = topic_path / "topic_universe_final.json"
        if themes_file.exists():
            settings.PATHS['THEMES_FILE'] = str(themes_file)
        else:
            logging_utils.log_thread_safe(f"[ERROR] Topic universe file '{topic_file_pattern}' not found in artifact: {topic_path}", 'error')
            wandb_handler.finish_run(run)
            return
    else:
        logging_utils.log_thread_safe(f"[ERROR] Topic universe artifact path is invalid: {topic_path}", 'error')
        wandb_handler.finish_run(run)
        return
    
    logging_utils.log_thread_safe(f"[INFO] Using W&B topic universe: {settings.PATHS['THEMES_FILE']}", 'info')
    
    # Memory will be generated fresh during this run and saved to OUTPUT_DIRS['memory']
    # No need to load existing memory - everything will be regenerated
    logging_utils.log_thread_safe("[INFO] Memory will be generated fresh during this run", 'info')
    
    # Tribe seed (for micro cluster summaries) - W&B ONLY (no local fallback)
    tribe_seed_artifact = input_artifacts.get("tribe_seed_characteristics")
    if not tribe_seed_artifact:
        logging_utils.log_thread_safe("[WARNING] tribe_seed_characteristics not found in config.yaml, will extract from prediction metadata", 'warning')
        settings.PATHS['MICRO_CLUSTER_SUMMARIES_DIR'] = None
    else:
        logging_utils.log_thread_safe(f"[INFO] Loading tribe seed from W&B: {tribe_seed_artifact}", 'info')
        tribe_seed_path = use_artifact(run, tribe_seed_artifact, artifact_type="dataset")
        if not tribe_seed_path:
            logging_utils.log_thread_safe(f"[WARNING] Could not load tribe seed artifact from W&B: {tribe_seed_artifact}, will extract from prediction metadata", 'warning')
            settings.PATHS['MICRO_CLUSTER_SUMMARIES_DIR'] = None
        else:
            settings.PATHS['MICRO_CLUSTER_SUMMARIES_DIR'] = str(tribe_seed_path)
            logging_utils.log_thread_safe(f"[INFO] Using W&B tribe seed: {settings.PATHS['MICRO_CLUSTER_SUMMARIES_DIR']}", 'info')
    
    # Output directory (read from config - REQUIRED, no hardcoded default)
    output_config = stage_config.get("output", {})
    output_stage = output_config.get("stage")
    if not output_stage:
        logging_utils.log_thread_safe("[ERROR] 'output.stage' not found in config.yaml - REQUIRED", 'error')
        wandb_handler.finish_run(run)
        return
    settings.PATHS['OUTPUT_BASE_DIR'] = str(get_artifact_dir(output_stage, output_artifact))
    Path(settings.PATHS['OUTPUT_BASE_DIR']).mkdir(parents=True, exist_ok=True)
    logging_utils.log_thread_safe(f"[INFO] Output directory: {settings.PATHS['OUTPUT_BASE_DIR']}", 'info')
    
    # Update OUTPUT_DIRS to use the new OUTPUT_BASE_DIR
    # All memory will be generated fresh and saved to output directory
    settings.PATHS['OUTPUT_DIRS'] = {
        "corrected": settings.PATHS['OUTPUT_BASE_DIR'],
        "cache": os.path.join(settings.PATHS['OUTPUT_BASE_DIR'], "_cache"),
        "memory": os.path.join(settings.PATHS['OUTPUT_BASE_DIR'], "_memory"),  # New memory generated here
        "journey": os.path.join(settings.PATHS['OUTPUT_BASE_DIR'], "_journey"),
        "delta": os.path.join(settings.PATHS['OUTPUT_BASE_DIR'], "_delta"),
        "failures": os.path.join(settings.PATHS['OUTPUT_BASE_DIR'], "_failures")
    }
    
    # Create output directories
    for dir_path in settings.PATHS['OUTPUT_DIRS'].values():
        os.makedirs(dir_path, exist_ok=True)

    # 3. Update settings from config.yaml (all parameters come from config, not hardcoded)
    settings.update_from_config(stage_config)

    if (
        getattr(settings, 'SCORING_MODE', 'confidence') not in ["logprobs", "logprobs-without-persona-context"]
        and "bedrock-mantle" in (os.getenv("OPENAI_BASE_URL") or "")
        and stage_config.get("bedrock_model_id")
    ):
        bedrock_model_id = stage_config["bedrock_model_id"]
        settings.ANALYSIS_MODEL = bedrock_model_id
        settings.REVIEW_PREDICTION_MODEL = bedrock_model_id
        settings.FEEDBACK_LOOP_MODEL = bedrock_model_id
    
    # Set API Key from config or environment
    if openai_config and openai_config.get('api_key'):
        settings.OPENAI_API_KEY = openai_config['api_key']
    elif not settings.OPENAI_API_KEY:
        settings.OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    
    if not settings.OPENAI_API_KEY:
        logging_utils.log_thread_safe("OPENAI_API_KEY is not set in config or environment variables.", 'error')
        logging_utils.log_thread_safe("Run: export OPENAI_API_KEY='your-key-here'", 'error')
        wandb_handler.finish_run(run)
        return
    
    logging_utils.log_thread_safe(f"[INFO] Configuration loaded from config.yaml:", 'info')
    logging_utils.log_thread_safe(f"  - Dataset Type: {dataset_type}", 'info')
    logging_utils.log_thread_safe(f"  - Model: {settings.ANALYSIS_MODEL}, Embedding: {settings.EMBEDDING_MODEL}", 'info')
    logging_utils.log_thread_safe(f"  - Iterations: {settings.NUM_ITERATIONS}, Batch Size: {settings.BATCH_SIZE}", 'info')
    logging_utils.log_thread_safe(f"  - Delta Threshold: {settings.DELTA_THRESHOLD}", 'info')

    # 4. Load Shared Resources
    logging_utils.log_thread_safe("[INFO] Loading shared resources...", 'info')
    
    # Load user characteristics (Used for persona context)
    user_chars_data = io_utils.load_user_characteristics(settings.PATHS['USER_CHARS_FILE'])
    if not user_chars_data:
        logging_utils.log_thread_safe(f"[WARNING] Could not load user characteristics from {settings.PATHS['USER_CHARS_FILE']}. Continuing without them.", 'warning')
        user_chars_data = {} # Handle gracefully
    else:
        logging_utils.log_thread_safe(f"[INFO] Loaded user characteristics for {len(user_chars_data)} users from {settings.PATHS['USER_CHARS_FILE']}", 'info')
        if len(user_chars_data) > 0:
            sample_user_id = list(user_chars_data.keys())[0]
            sample_user_data = user_chars_data[sample_user_id]
            logging_utils.log_thread_safe(f"[INFO] Sample user data structure - keys: {list(sample_user_data.keys()) if isinstance(sample_user_data, dict) else 'N/A'}", 'info')

    # Load category mapping - W&B ONLY (REQUIRED, no local fallback)
    category_mapping_artifact = input_artifacts.get("category_mapping")
    if not category_mapping_artifact:
        logging_utils.log_thread_safe("[ERROR] category_mapping not found in config.yaml - REQUIRED", 'error')
        wandb_handler.finish_run(run)
        return
    
    logging_utils.log_thread_safe(f"[INFO] Loading category mapping from W&B: {category_mapping_artifact}", 'info')
    category_file_pattern = stage_config.get("file_patterns", {}).get("category_mapping", "category_mapping.json")
    category_mapping = io_utils.load_category_mapping(run, category_mapping_artifact, category_file_pattern)
    if not category_mapping:
        logging_utils.log_thread_safe(f"[ERROR] Could not load category mapping from W&B: {category_mapping_artifact} - REQUIRED", 'error')
        wandb_handler.finish_run(run)
        return
    
    logging_utils.log_thread_safe(f"[INFO] Loaded category mapping from W&B: {len(category_mapping)} categories", 'info')
    
    # 5. Discover Files
    logging_utils.log_thread_safe(f"[INFO] Discovering files in {settings.PATHS['PREDICTION_INPUT_DIR']}...", 'info')
    all_files = io_utils.discover_all_micro_cluster_files()
    
    if not all_files:
        logging_utils.log_thread_safe(f"[ERROR] No micro cluster files found in {settings.PATHS['PREDICTION_INPUT_DIR']}", 'error')
        wandb_handler.finish_run(run)
        return
    
    logging_utils.log_thread_safe(f"[INFO] Found {len(all_files)} total files.", 'info')

    # 6. Apply Filters (Cluster / Start From / Reprocess)
    files_to_process = []
    
    # Filter: specific cluster
    if only_cluster:
        all_files = [f for f in all_files if f[0] == only_cluster]
        logging_utils.log_thread_safe(f"[INFO] Filtered to {len(all_files)} files for {only_cluster}", 'info')
    
    # Filter: specific micro cluster (requires only_cluster to be set)
    if only_micro:
        if not only_cluster:
            logging_utils.log_thread_safe("[ERROR] --only-micro requires --only-cluster to be specified", 'error')
            wandb_handler.finish_run(run)
            return
        all_files = [f for f in all_files if f[1] == only_micro]
        logging_utils.log_thread_safe(f"[INFO] Filtered to {len(all_files)} files for {only_cluster}/{only_micro}", 'info')

    # Filter: start from cluster
    if start_from_cluster:
        try:
            start_num = int(re.search(r'cluster_(\d+)', start_from_cluster).group(1))
            filtered = []
            for item in all_files:
                curr_num = int(re.search(r'cluster_(\d+)', item[0]).group(1))
                if curr_num >= start_num:
                    filtered.append(item)
            all_files = filtered
            logging_utils.log_thread_safe(f"[INFO] Filtered to {len(all_files)} files starting from {start_from_cluster}", 'info')
        except (AttributeError, ValueError):
            logging_utils.log_thread_safe(f"[WARNING] Invalid start_from_cluster format: {start_from_cluster}. Ignoring.", 'warning')

    # Filter: Force Reprocess
    if not force_reprocess:
        for cluster_id, micro_id, filepath in all_files:
            file_basename = os.path.basename(filepath).replace('.json', '')
            corrected_path = os.path.join(settings.PATHS['OUTPUT_DIRS']['corrected'], cluster_id, f"{file_basename}_corrected.json")
            
            if os.path.exists(corrected_path):
                # logging_utils.log_thread_safe(f"[SKIP] {cluster_id}/{micro_id} exists.", 'info')
                continue
            files_to_process.append((cluster_id, micro_id, filepath))
        
        logging_utils.log_thread_safe(f"[INFO] {len(files_to_process)} files remain after checking existing outputs.", 'info')
    else:
        files_to_process = all_files

    # 7. Execute Parallel Processing
    if not files_to_process:
        logging_utils.log_thread_safe("[INFO] No files to process.", 'info')
    else:
        logging_utils.log_thread_safe(f"[START] Starting parallel processing with {max_workers} workers...", 'info')
        
        # Prepare arguments tuple for wrapper
        worker_args = [
            (cid, mid, fpath, category_mapping, user_chars_data)
            for cid, mid, fpath in files_to_process
        ]

        results = {'success': 0, 'error': 0, 'no_result': 0}

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_file = {executor.submit(process_single_file_wrapper, args): args for args in worker_args}
            
            for future in as_completed(future_to_file):
                try:
                    res = future.result()
                    status = res.get('status', 'unknown')
                    results[status] = results.get(status, 0) + 1
                except Exception as e:
                    logging_utils.log_thread_safe(f"[CRITICAL ERROR] Thread died: {e}", 'error')
                    results['error'] += 1

        logging_utils.log_thread_safe(f"\n{'='*60}", 'info')
        logging_utils.log_thread_safe(f"[SUMMARY] Success: {results['success']} | Error: {results['error']}", 'info')
        logging_utils.log_thread_safe(f"{'='*60}\n", 'info')

    # 8. Generate Aggregate Summaries
    logging_utils.log_thread_safe("[AGGREGATE] Generating cluster-level summaries...", 'info')
    
    # Identify all clusters relevant to this run (either processed or requested)
    clusters_to_summarize = set()
    
    # Add clusters we just processed
    for item in files_to_process:
        clusters_to_summarize.add(item[0])
        
    # Add clusters if we are in "only_cluster" mode (even if no files processed this run, summarize existing)
    if only_cluster:
        clusters_to_summarize.add(only_cluster)
        
    # Or just scan output dir to find all clusters
    if not only_cluster and not files_to_process:
         if os.path.exists(settings.PATHS['OUTPUT_DIRS']['corrected']):
             for d in os.listdir(settings.PATHS['OUTPUT_DIRS']['corrected']):
                 if d.startswith('cluster_'):
                     clusters_to_summarize.add(d)

    for cluster_id in sorted(clusters_to_summarize):
        generate_cluster_aggregate_summary(cluster_id)

    # 8.5. Generate Cost Report (automatically tracks all LLM calls)
    try:
        from sgo_training.utils.cost_tracker import save_cost_report, print_cost_summary, get_detailed_report
        cost_report_file = os.path.join(settings.PATHS['OUTPUT_BASE_DIR'], "cost_report.json")
        report = save_cost_report(cost_report_file)
        
        if report and report.get('total_calls', 0) > 0:
            grand_total = report.get('grand_total_cost', 0.0)
            logging_utils.log_thread_safe(f"[COST] Generated cost report: ${grand_total:.4f} total across {report['total_calls']} API calls", 'info')
            print_cost_summary()
        else:
            logging_utils.log_thread_safe("[COST] No cost data tracked (no LLM calls made or tracking not enabled)", 'info')
    except Exception as e:
        logging_utils.log_thread_safe(f"[COST] Error generating cost report: {e}", 'warning')
        import traceback
        traceback.print_exc()

    # 9. Log Artifacts & Finish
    # Log main SGO training results artifact
    wandb_handler.log_output_artifact(run, settings.PATHS['OUTPUT_BASE_DIR'])
    
    # Log initial_deltas as a separate artifact
    if settings.PATHS['OUTPUT_DIRS'] and settings.PATHS['OUTPUT_DIRS'].get('delta'):
        delta_dir = settings.PATHS['OUTPUT_DIRS']['delta']
        if os.path.exists(delta_dir) and os.listdir(delta_dir):  # Check if directory exists and is not empty
            # Get initial_deltas artifact name from config
            try:
                # stage_config is already loaded at the top of main()
                initial_deltas_artifact_name = stage_config.get("output_artifacts", {}).get("initial_deltas")
                
                if initial_deltas_artifact_name:
                    logging_utils.log_thread_safe(f"[W&B] Logging initial_deltas artifact: {initial_deltas_artifact_name}", 'info')
                    wandb_handler.log_output_artifact(
                        run, 
                        delta_dir,
                        artifact_name=initial_deltas_artifact_name,
                        description="Initial predictions with deltas (JSD for themes, cosine for text) - pre-iteration delta calculations"
                    )
                else:
                    logging_utils.log_thread_safe("[W&B] WARNING: initial_deltas artifact name not found in config.yaml", 'warning')
            except Exception as e:
                logging_utils.log_thread_safe(f"[W&B] WARNING: Could not log initial_deltas artifact: {e}", 'warning')
                import traceback
                traceback.print_exc()
        else:
            logging_utils.log_thread_safe(f"[W&B] WARNING: Delta directory does not exist or is empty: {delta_dir}", 'warning')
    
    wandb_handler.finish_run(run)
    logging_utils.log_thread_safe("[DONE] Run complete.", 'success')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process micro clusters with delta refinement or test mode predictions')
    parser.add_argument('--max-workers', type=int, default=4, help='Number of parallel workers')
    parser.add_argument('--force-reprocess', action='store_true', help='Reprocess already completed files')
    parser.add_argument('--only-cluster', type=str, help='Process only this specific cluster (e.g., cluster_0)')
    parser.add_argument('--only-micro', type=str, help='Process only this specific micro cluster (e.g., micro_15). Requires --only-cluster to be specified.')
    parser.add_argument('--start-from-cluster', type=str, help='Start processing from this cluster onwards')
    
    args = parser.parse_args()
    
    # Validate that only_micro requires only_cluster
    if args.only_micro and not args.only_cluster:
        print("ERROR: --only-micro requires --only-cluster to be specified")
        sys.exit(1)
    
    main(
        force_reprocess=args.force_reprocess,
        max_workers=args.max_workers,
        only_cluster=args.only_cluster,
        start_from_cluster=args.start_from_cluster,
        only_micro=args.only_micro
    )