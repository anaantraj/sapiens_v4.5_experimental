import os
import sys
import logging
from threading import Lock

# Global dictionary to store cluster loggers
cluster_loggers = {}
cluster_loggers_lock = Lock()

class ClusterFileHandler(logging.Handler):
    """Custom logging handler that writes to cluster-specific files"""
    def __init__(self, cluster_id, base_log_dir):
        super().__init__()
        self.cluster_id = cluster_id
        
        # Ensure the directory exists
        self.log_dir = os.path.join(base_log_dir, "cluster_logs")
        os.makedirs(self.log_dir, exist_ok=True)
        
        self.log_file = os.path.join(self.log_dir, f"{cluster_id}_terminal_output.txt")
        self.file_handle = open(self.log_file, 'a', encoding='utf-8')
    
    def emit(self, record):
        # Filter out W&B's verbose artifact upload messages to prevent log spam
        if hasattr(record, 'name') and 'wandb' in record.name.lower():
            # Skip W&B internal logging (artifact uploads, etc.)
            if 'uploading artifact' in record.getMessage().lower() or 'updating run metadata' in record.getMessage().lower():
                return
        try:
            msg = self.format(record)
            self.file_handle.write(msg + '\n')
            self.file_handle.flush()
        except Exception:
            self.handleError(record)
    
    def close(self):
        if self.file_handle:
            self.file_handle.close()
        super().close()

def get_cluster_logger(cluster_id, base_output_dir):
    """Get or create a logger for a specific cluster"""
    with cluster_loggers_lock:
        if cluster_id not in cluster_loggers:
            logger = logging.getLogger(f"cluster_{cluster_id}")
            logger.setLevel(logging.INFO)
            logger.handlers = []  # Clear existing handlers
            logger.propagate = False  # Don't propagate to root logger to avoid capturing W&B logs
            
            # Add file handler for this cluster
            file_handler = ClusterFileHandler(cluster_id, base_output_dir)
            file_handler.setLevel(logging.INFO)
            # We use a simple message format because the timestamp comes from the wrapper
            formatter = logging.Formatter('%(asctime)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
            
            cluster_loggers[cluster_id] = logger
        
        return cluster_loggers[cluster_id]

class TeeOutput:
    """
    Context Manager to tee (split) output to both console (stdout) and a log file.
    Usage:
        with TeeOutput(cluster_id, base_output_dir):
            print("This goes to both!")
    """
    def __init__(self, cluster_id, base_output_dir):
        self.cluster_id = cluster_id
        self.base_output_dir = base_output_dir
        self.cluster_logger = get_cluster_logger(cluster_id, base_output_dir)
        self.original_stdout = sys.stdout
        self.original_stderr = sys.stderr
        self.log_buffer = []
    
    def write(self, text):
        # Write to original stdout (console)
        self.original_stdout.write(text)
        self.original_stdout.flush()
        
        # Filter out W&B's verbose artifact upload messages to prevent log spam
        if text and ('uploading artifact' in text.lower() or 'updating run metadata' in text.lower()):
            # Skip W&B internal upload messages
            return
        
        # Buffer for logging (handle multi-line output or partial writes)
        if text:
            self.log_buffer.append(text)
            # If we have a complete line (ends with newline), log it
            if text.endswith('\n'):
                line = ''.join(self.log_buffer).rstrip()
                if line:
                    self.cluster_logger.info(line)
                self.log_buffer = []
    
    def flush(self):
        self.original_stdout.flush()
        # Flush any remaining buffer
        if self.log_buffer:
            line = ''.join(self.log_buffer).rstrip()
            if line:
                self.cluster_logger.info(line)
            self.log_buffer = []
    
    def __enter__(self):
        sys.stdout = self
        sys.stderr = self
        # Log start marker
        self.cluster_logger.info("=" * 80)
        self.cluster_logger.info(f"Starting processing for cluster: {self.cluster_id}")
        self.cluster_logger.info("=" * 80)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        # Flush any remaining buffer
        self.flush()
        
        # Log end marker
        self.cluster_logger.info("=" * 80)
        self.cluster_logger.info(f"Completed processing for cluster: {self.cluster_id}")
        self.cluster_logger.info("=" * 80)
        
        # Restore original streams
        sys.stdout = self.original_stdout
        sys.stderr = self.original_stderr
        
        # Log exception if one occurred
        if exc_type:
            self.cluster_logger.error(f"Process failed with error: {exc_val}")
            import traceback
            self.cluster_logger.error("".join(traceback.format_tb(exc_tb)))

def log_thread_safe(message, level='info'):
    """
    Simple helper for logging from threads where context manager might not apply directly.
    In a threaded environment, standard print() inside TeeOutput works fine.
    """
    if level == 'error':
        print(f"ERROR: {message}")
    else:
        print(message)