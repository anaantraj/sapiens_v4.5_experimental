import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import sys
from pathlib import Path

# Set up package structure for sgo_training imports
scripts_dir = Path(__file__).parent.parent
if str(scripts_dir) not in sys.path:
    sys.path.insert(0, str(scripts_dir))
import _package_setup

from sgo_training.config import settings

def get_or_compute_embedding(text, item_id, embedding_type, client, cache, lock=None):
    """
    Computes an embedding for a text string, or retrieves it from cache if available.
    Thread-safe if lock is provided.
    """
    key = f"{item_id}_{embedding_type}"
    
    # Thread-safe cache read
    if lock:
        with lock:
            if key in cache:
                return cache[key]
    else:
        if key in cache:
            return cache[key]
    
    # sanitize text
    text = str(text).replace("\n", " ")
    
    # Add small delay to reduce rate limit hits
    time.sleep(settings.MIN_REQUEST_INTERVAL)
    
    try:
        response = client.embeddings.create(input=[text], model=settings.EMBEDDING_MODEL)
        embedding = response.data[0].embedding
        
        # Thread-safe cache write
        if lock:
            with lock:
                cache[key] = embedding
        else:
            cache[key] = embedding
        
        return embedding
    except Exception as e:
        print(f"  - ERROR computing embedding for {item_id}_{embedding_type}: {e}")
        return None
def compute_embeddings_parallel(texts_and_keys, client, cache, max_workers=None, lock=None):
    """
    Compute multiple embeddings in parallel.
    Args:
        texts_and_keys: List of tuples (text, item_id, embedding_type)
        client: OpenAI client
        cache: Embedding cache dict (updated in-place, thread-safe if lock provided)
        max_workers: Max parallel workers (default: settings.MAX_PARALLEL_EMBEDDINGS)
        lock: Optional lock for thread-safe cache access
    Returns:
        Dict mapping (item_id, embedding_type) -> embedding
    """
    # Use default from settings if not provided
    max_workers = max_workers or settings.MAX_PARALLEL_EMBEDDINGS

    if not settings.PARALLEL_EMBEDDINGS or len(texts_and_keys) <= 1:
        # Sequential fallback
        results = {}
        for text, item_id, embedding_type in texts_and_keys:
            key = f"{item_id}_{embedding_type}"
            # Thread-safe cache check
            if lock:
                with lock:
                    if key in cache:
                        results[key] = cache[key]
                        continue
            else:
                if key in cache:
                    results[key] = cache[key]
                    continue
            
            embedding = get_or_compute_embedding(text, item_id, embedding_type, client, cache, lock)
            if embedding:
                results[key] = embedding
        return results
    
    def compute_single_embedding(args):
        text, item_id, embedding_type = args
        key = f"{item_id}_{embedding_type}"
        
        # Thread-safe cache check inside worker
        if lock:
            with lock:
                if key in cache:
                    return (key, cache[key])
        else:
            if key in cache:
                return (key, cache[key])
            
        try:
            # We call the helper which handles the API call and sleep
            # get_or_compute_embedding now handles thread-safe cache updates
            embedding = get_or_compute_embedding(text, item_id, embedding_type, client, cache, lock)
            return (key, embedding)
        except Exception as e:
            print(f"  - ERROR computing embedding for {item_id}_{embedding_type}: {e}")
            return (key, None)

    actual_workers = min(max_workers, len(texts_and_keys))
    results = {}
    
    with ThreadPoolExecutor(max_workers=actual_workers) as executor:
        futures = {executor.submit(compute_single_embedding, args): args for args in texts_and_keys}
        
        for future in as_completed(futures):
            try:
                # The worker returns a tuple (key, embedding) or (key, None)
                res = future.result()
                if res:
                    key, embedding = res
                    if embedding is not None:
                        results[key] = embedding
                        # Cache is already updated by get_or_compute_embedding with lock protection
                        # No need to update again here
            except Exception as e:
                args = futures[future]
                print(f"  - ERROR in parallel embedding computation for {args[1]}_{args[2]}: {e}")

    return results