import os
from pathlib import Path

# Base Paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
BASE_DIR = Path(__file__).parent.parent

# NOTE: All configuration parameters are loaded from config.yaml via main.py
# These are defaults that will be overridden by config.yaml values
# Model Config (defaults - will be overridden by config.yaml)
EMBEDDING_MODEL = 'text-embedding-3-small'
ANALYSIS_MODEL = 'o3'
# OPENAI_API_KEY and OPENAI_BASE_URL (e.g. Bedrock) loaded from config or environment
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", None)
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", None)

# Tuning (defaults - will be overridden by config.yaml)
NUM_ITERATIONS = 5
BATCH_SIZE = 5
DELTA_THRESHOLD = 0.3 
WEIGHT_TEXT_DELTA = 0.7
WEIGHT_THEME_DELTA = 0.3

# Parallel Config (defaults - will be overridden by config.yaml)
PARALLEL_BATCHES = True
MAX_PARALLEL_BATCHES = 3
PARALLEL_EMBEDDINGS = True
MAX_PARALLEL_EMBEDDINGS = 10

# API Config (defaults - will be overridden by config.yaml)
API_RATE_LIMIT_DELAY = 0.5
MIN_REQUEST_INTERVAL = 0.1

# LLM Retry Config (defaults - will be overridden by config.yaml)
MAX_RETRIES = 3

# Logprobs Config (defaults - will be overridden by config.yaml)
TOP_LOGPROBS = 20  # Number of top logprobs to return for theme classification (higher = better chance of finding both yes/no tokens)

# Scoring Mode Config (defaults - will be overridden by config.yaml)
# These MUST be set in config.yaml - no hardcoded defaults
SCORING_MODE = None  # Options: "confidence", "logprobs", "logprobs-without-persona-context"
REVIEW_PREDICTION_MODEL = None  # Model for generating review text (Step 1 in logprobs mode, Part A in confidence mode) - MUST be set in config.yaml
THEME_PREDICTION_MODEL = None  # Model for theme classification with logprobs (Step 2 in logprobs mode) - MUST be set in config.yaml
FEEDBACK_LOOP_MODEL = None  # Model for Part B feedback loop analysis - MUST be set in config.yaml

# Prompt Directories (relative to BASE_DIR.parent, which is 07_sgo_training)
PROMPT_DIR_PART_A = BASE_DIR.parent / "prompts"  # 07_sgo_training/prompts
PROMPT_DIR_PART_B = BASE_DIR.parent / "prompts"  # 07_sgo_training/prompts

# Dynamic Paths (To be set at runtime from W&B artifacts - NO LOCAL FALLBACKS)
PATHS = {
    "PREDICTION_INPUT_DIR": None,
    "USER_CHARS_FILE": None,
    "MICRO_CLUSTER_SUMMARIES_DIR": None,
    "THEMES_FILE": None,
    "OUTPUT_BASE_DIR": None,
    "OUTPUT_DIRS": {
        "corrected": None,
        "cache": None,
        "memory": None,
        "journey": None,
        "delta": None,
        "failures": None
    }
}

def update_from_config(stage_config):
    """
    Update settings from stage_config (loaded from config.yaml).
    This ensures all parameters come from config.yaml, not hardcoded values.
    """
    global EMBEDDING_MODEL, ANALYSIS_MODEL, NUM_ITERATIONS, BATCH_SIZE, DELTA_THRESHOLD
    global WEIGHT_TEXT_DELTA, WEIGHT_THEME_DELTA, PARALLEL_BATCHES, MAX_PARALLEL_BATCHES
    global PARALLEL_EMBEDDINGS, MAX_PARALLEL_EMBEDDINGS, API_RATE_LIMIT_DELAY, MIN_REQUEST_INTERVAL
    global MAX_RETRIES, PROMPT_DIR_PART_A, PROMPT_DIR_PART_B
    global SCORING_MODE, REVIEW_PREDICTION_MODEL, THEME_PREDICTION_MODEL, FEEDBACK_LOOP_MODEL
    global TOP_LOGPROBS
    
    hyperparams = stage_config.get("hyperparameters", {})
    
    # Model config
    EMBEDDING_MODEL = hyperparams.get("embedding_model", EMBEDDING_MODEL)
    ANALYSIS_MODEL = hyperparams.get("analysis_model", ANALYSIS_MODEL)
    
    # Scoring mode config (from top-level config, not hyperparameters)
    # These MUST be provided in config.yaml - no fallback defaults
    SCORING_MODE = stage_config.get("scoring_mode")
    if SCORING_MODE is None:
        raise ValueError("scoring_mode must be set in config.yaml")
    
    # Model configuration (all modes)
    REVIEW_PREDICTION_MODEL = stage_config.get("review_prediction_model")
    # Prefer logprobs_model_id for theme classification when set (must support API logprobs)
    THEME_PREDICTION_MODEL = stage_config.get("logprobs_model_id") or stage_config.get("theme_prediction_model")
    FEEDBACK_LOOP_MODEL = stage_config.get("feedback_loop_model")
    
    # Validation: review_prediction_model is required for all modes
    if REVIEW_PREDICTION_MODEL is None:
        raise ValueError("review_prediction_model must be set in config.yaml")
    
    # Validation: theme_prediction_model or logprobs_model_id required for logprobs modes
    if SCORING_MODE in ["logprobs", "logprobs-without-persona-context"]:
        if THEME_PREDICTION_MODEL is None:
            raise ValueError("theme_prediction_model or logprobs_model_id must be set in config.yaml when using logprobs mode")
    
    # Validation: feedback_loop_model is required for Part B (all modes)
    if FEEDBACK_LOOP_MODEL is None:
        raise ValueError("feedback_loop_model must be set in config.yaml")
    
    # Delta refinement parameters
    NUM_ITERATIONS = hyperparams.get("num_iterations", NUM_ITERATIONS)
    BATCH_SIZE = hyperparams.get("batch_size", BATCH_SIZE)
    DELTA_THRESHOLD = hyperparams.get("delta_threshold", DELTA_THRESHOLD)
    WEIGHT_TEXT_DELTA = hyperparams.get("weight_text_delta", WEIGHT_TEXT_DELTA)
    WEIGHT_THEME_DELTA = hyperparams.get("weight_theme_delta", WEIGHT_THEME_DELTA)
    
    # Parallel processing
    PARALLEL_BATCHES = hyperparams.get("parallel_batches", PARALLEL_BATCHES)
    MAX_PARALLEL_BATCHES = hyperparams.get("max_parallel_batches", MAX_PARALLEL_BATCHES)
    PARALLEL_EMBEDDINGS = hyperparams.get("parallel_embeddings", PARALLEL_EMBEDDINGS)
    MAX_PARALLEL_EMBEDDINGS = hyperparams.get("max_parallel_embeddings", MAX_PARALLEL_EMBEDDINGS)
    
    # API rate limiting
    API_RATE_LIMIT_DELAY = hyperparams.get("api_rate_limit_delay", API_RATE_LIMIT_DELAY)
    MIN_REQUEST_INTERVAL = hyperparams.get("min_request_interval", MIN_REQUEST_INTERVAL)
    
    # LLM retry configuration
    MAX_RETRIES = hyperparams.get("max_retries", 3)
    
    # Logprobs Config
    TOP_LOGPROBS = hyperparams.get("top_logprobs", TOP_LOGPROBS)
    
    # Prompt directories (from config.yaml)
    prompt_config = stage_config.get("prompt", {})
    prompt_dir_name = prompt_config.get("directory", "prompts")
    # Prompt directory is relative to BASE_DIR.parent (07_sgo_training directory, not scripts subdirectory)
    PROMPT_DIR_PART_A = BASE_DIR.parent / prompt_dir_name
    PROMPT_DIR_PART_B = BASE_DIR.parent / prompt_dir_name
