"""Category processing utilities."""
from pathlib import Path
from typing import Optional, Dict, Any
from .file_io import read_jsonl
from .data_loaders import load_category_requirements


def scan_reviews_for_category(category_name: str, target_users: dict, source_data_dir: str,
                              global_existing_keys: set, s3_config: Optional[Dict[str, Any]] = None,
                              target_review_count: Optional[int] = None):
    """
    Scan reviews file and extract matching reviews.
    Supports both local files and S3.
    
    Args:
        category_name: Category name
        target_users: Dict of user_id -> review_count needed
        source_data_dir: Source data directory
        global_existing_keys: Set of existing (user_id, asin) keys
        s3_config: Optional S3 configuration
        target_review_count: Optional target number of reviews to collect (for validation)
    """
    reviews_buffer, needed_asins, local_batch_keys = [], set(), set()
    
    # Check if we should use S3
    s3_client = None
    if s3_config and s3_config.get("enabled", False):
        try:
            import boto3
            if "region" not in s3_config:
                raise ValueError("[ERROR] 'hyperparameters.s3.region' is required when S3 is enabled")
            region = s3_config["region"]
            s3_client = boto3.client('s3', region_name=region)
        except ImportError:
            pass
    
    if s3_config and s3_config.get("enabled", False):
        # S3 is enabled - this function should not be called when using sampling mode
        # If sampling is disabled but S3 is enabled, raise an error
        raise RuntimeError(
            "[ERROR] S3 is enabled but sampling is disabled. "
            "When S3 is enabled, you must use sampling mode (set sampling.enabled=true in config)."
        )
    
    # Read from local file system
    review_path = Path(source_data_dir) / f"{category_name}.jsonl"
    if not review_path.exists():
        review_path = Path(source_data_dir) / f"{category_name}.jsonl.gz"
    if not review_path.exists():
        return None, set(), 0, 0
    
    reviews_found = duplicates_skipped = 0
    # Calculate total reviews needed
    total_reviews_needed = sum(target_users.values()) if target_users else (target_review_count or 0)
    
    for r in read_jsonl(review_path):
        # Stop if we've collected enough reviews
        if target_review_count and reviews_found >= target_review_count:
            break
        if not target_users and not target_review_count:
            break
        
        user_id = r.get('reviewerID') or r.get('user_id')
        asin = r.get('parent_asin') or r.get('asin')
        
        if user_id in target_users:
            unique_key = (user_id, asin)
            if unique_key in global_existing_keys or unique_key in local_batch_keys:
                duplicates_skipped += 1
                continue
            r['dataset_category'] = category_name
            reviews_buffer.append(r)
            local_batch_keys.add(unique_key)
            if 'parent_asin' in r:
                needed_asins.add(r['parent_asin'])
            if 'asin' in r:
                needed_asins.add(r['asin'])
            target_users[user_id] -= 1
            reviews_found += 1
            if target_users[user_id] <= 0:
                del target_users[user_id]
        elif target_review_count and reviews_found < target_review_count:
            # If we have a target count and haven't reached it, consider any user
            unique_key = (user_id, asin)
            if unique_key in global_existing_keys or unique_key in local_batch_keys:
                duplicates_skipped += 1
                continue
            r['dataset_category'] = category_name
            reviews_buffer.append(r)
            local_batch_keys.add(unique_key)
            if 'parent_asin' in r:
                needed_asins.add(r['parent_asin'])
            if 'asin' in r:
                needed_asins.add(r['asin'])
            reviews_found += 1
    
    return reviews_buffer, needed_asins, reviews_found, duplicates_skipped


def load_metadata(category_name: str, needed_asins: set, source_data_dir: str,
                  s3_config: Optional[Dict[str, Any]] = None):
    """
    Load metadata for needed ASINs.
    Supports both local files and S3.
    """
    meta_lookup = {}
    
    # Check if we should use S3
    s3_client = None
    if s3_config and s3_config.get("enabled", False):
        try:
            import boto3
            if "region" not in s3_config:
                raise ValueError("[ERROR] 'hyperparameters.s3.region' is required when S3 is enabled")
            region = s3_config["region"]
            s3_client = boto3.client('s3', region_name=region)
        except ImportError:
            pass
    
    if s3_config and s3_config.get("enabled", False):
        # S3 is enabled - this function should not be called when using sampling mode
        # If sampling is disabled but S3 is enabled, raise an error
        raise RuntimeError(
            "[ERROR] S3 is enabled but sampling is disabled. "
            "When S3 is enabled, you must use sampling mode (set sampling.enabled=true in config)."
        )
    
    # Read from local file system
    meta_path = Path(source_data_dir) / f"meta_{category_name}.jsonl"
    if not meta_path.exists():
        meta_path = Path(source_data_dir) / f"meta_{category_name}.jsonl.gz"
    
    if meta_path.exists():
        for m in read_jsonl(meta_path):
            p_asin, asin = m.get('parent_asin'), m.get('asin')
            if (p_asin and p_asin in needed_asins) or (asin and asin in needed_asins):
                desc_raw = m.get('description', [])
                desc_text = " ".join(desc_raw) if isinstance(desc_raw, list) else str(desc_raw)
                data = {
                    "title": m.get('title', 'N/A'),
                    "main_category": m.get('main_category', category_name),
                    "raw_description": desc_text
                }
                if p_asin:
                    meta_lookup[p_asin] = data
                if asin:
                    meta_lookup[asin] = data
    else:
        print("   [WARNING] Meta file not found.")
    
    return meta_lookup


def merge_reviews_into_database(reviews_buffer: list, meta_lookup: dict, category_name: str,
                                master_db: dict, global_existing_keys: set):
    """
    Merge reviews into master database.
    Skips reviews that don't have a valid product description.
    
    Returns:
        Tuple of (reviews_added, reviews_skipped_no_description)
    """
    reviews_added = 0
    reviews_skipped_no_description = 0
    
    for r in reviews_buffer:
        user_id = r.get('reviewerID') or r.get('user_id')
        asin = r.get('parent_asin') or r.get('asin')
        
        meta = meta_lookup.get(asin, {})
        final_description = meta.get("raw_description", "")
        if not final_description or final_description == "[]" or final_description.strip() == "":
            final_description = meta.get("title", "")
        
        # Skip review if no valid product description exists
        if not final_description or final_description.strip() == "" or final_description == "N/A":
            reviews_skipped_no_description += 1
            continue
        
        if user_id not in master_db:
            master_db[user_id] = {'reviews': []}
        
        master_db[user_id]['reviews'].append({
            "product_description": final_description,
            "review_text": r.get('text', ''),
            "rating": r.get('rating'),
            "category": meta.get("main_category", category_name),
            "timestamp": r.get('timestamp'),
            "asin": asin
        })
        global_existing_keys.add((user_id, asin))
        reviews_added += 1
    
    return reviews_added, reviews_skipped_no_description


def process_category(category_name: str, master_db: dict, global_existing_keys: set,
                     user_lists_dir: str, source_data_dir: str,
                     max_users: int = None, max_reviews_per_user: int = None,
                     s3_config: Optional[Dict[str, Any]] = None):
    """
    Process a single category.
    Supports both local files and S3.
    """
    print(f"\n{'='*50}")
    print(f"PROCESSING: {category_name}")
    print(f"{'='*50}")
    
    if max_users is not None:
        print(f"   Max users limit: {max_users}")
    if max_reviews_per_user is not None:
        print(f"   Max reviews per user limit: {max_reviews_per_user}")
    
    target_users = load_category_requirements(
        category_name, user_lists_dir, max_users, max_reviews_per_user, s3_config
    )
    if not target_users:
        print("   Skipping (no targets found).")
        return 0, 0
    
    print(f"   Target users: {len(target_users)}")
    
    # Check if using S3 or local
    if s3_config and s3_config.get("enabled", False):
        print(f"   Scanning reviews from S3...")
    else:
        review_path = Path(source_data_dir) / f"{category_name}.jsonl"
        if not review_path.exists():
            review_path = Path(source_data_dir) / f"{category_name}.jsonl.gz"
        if not review_path.exists():
            print(f"[ERROR] Review file not found for {category_name}")
            return 0, 0
        print(f"   Scanning reviews in {review_path.name}...")
    
    # Calculate target review count
    target_review_count = sum(target_users.values()) if target_users else None
    
    reviews_buffer, needed_asins, reviews_found, duplicates_skipped = scan_reviews_for_category(
        category_name, target_users, source_data_dir, global_existing_keys, s3_config, target_review_count
    )
    
    if reviews_buffer is None:
        return 0, 0
    
    print(f"   [OK] Extracted {reviews_found} unique new reviews.")
    if duplicates_skipped > 0:
        print(f"   [SKIP] Skipped {duplicates_skipped} duplicates (already present).")
    
    meta_lookup = load_metadata(category_name, needed_asins, source_data_dir, s3_config)
    print("   Merging into master database...")
    reviews_added, reviews_skipped_no_description = merge_reviews_into_database(
        reviews_buffer, meta_lookup, category_name, master_db, global_existing_keys
    )
    
    # If we skipped reviews due to missing descriptions and haven't reached target, continue scanning
    if reviews_skipped_no_description > 0:
        print(f"   [SKIP] Skipped {reviews_skipped_no_description} reviews without product description.")
    
    # If we need more reviews and some were skipped, try to get more
    if target_review_count and reviews_added < target_review_count and reviews_skipped_no_description > 0:
        remaining_needed = target_review_count - reviews_added
        print(f"   [INFO] Need {remaining_needed} more reviews. Continuing to scan...")
        
        # Continue scanning for more reviews
        additional_buffer, additional_asins, additional_found, additional_duplicates = scan_reviews_for_category(
            category_name, {}, source_data_dir, global_existing_keys, s3_config, remaining_needed
        )
        
        if additional_buffer:
            # Load additional metadata if needed
            new_asins = additional_asins - needed_asins
            if new_asins:
                additional_meta = load_metadata(category_name, new_asins, source_data_dir, s3_config)
                meta_lookup.update(additional_meta)
            
            # Merge additional reviews
            additional_added, additional_skipped = merge_reviews_into_database(
                additional_buffer, meta_lookup, category_name, master_db, global_existing_keys
            )
            reviews_added += additional_added
            reviews_skipped_no_description += additional_skipped
            duplicates_skipped += additional_duplicates
            
            if additional_skipped > 0:
                print(f"   [SKIP] Skipped {additional_skipped} additional reviews without product description.")
    
    return reviews_added, duplicates_skipped
