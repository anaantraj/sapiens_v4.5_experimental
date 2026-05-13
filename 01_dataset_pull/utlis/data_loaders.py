"""Data loading utilities."""
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Optional, Dict, Any
from .category_mapping import map_category_to_main_category


def _get_s3_client(s3_config):
    """Get S3 client. Returns None if boto3 is not available or S3 is disabled."""
    if not s3_config or not s3_config.get("enabled", False):
        return None
    
    try:
        import boto3
        if "region" not in s3_config:
            raise ValueError("[ERROR] 'hyperparameters.s3.region' is required when S3 is enabled")
        region = s3_config["region"]
        return boto3.client('s3', region_name=region)
    except ImportError:
        print("   [WARNING] boto3 not installed. Install with: pip install boto3")
        return None
    except Exception as e:
        print(f"   [ERROR] Failed to create S3 client: {e}")
        return None


def _read_text_file_from_s3(s3_client, bucket: str, key: str) -> str:
    """Read text file from S3."""
    try:
        response = s3_client.get_object(Bucket=bucket, Key=key)
        content = response['Body'].read()
        if isinstance(content, bytes):
            content = content.decode('utf-8')
        return content
    except s3_client.exceptions.NoSuchKey:
        print(f"   [ERROR] S3 key not found: s3://{bucket}/{key}")
        return ""
    except Exception as e:
        print(f"   [ERROR] Error reading from S3 s3://{bucket}/{key}: {e}")
        return ""


def load_category_requirements(category_name: str, user_lists_dir: str, 
                                max_users: int = None, max_reviews_per_user: int = None,
                                s3_config: Optional[Dict[str, Any]] = None):
    """
    Load user requirements from text file.
    Supports both local files and S3.
    
    Args:
        category_name: Category name
        user_lists_dir: Local directory or S3 prefix for user lists
        max_users: Maximum number of users to load
        max_reviews_per_user: Maximum reviews per user
        s3_config: Optional S3 configuration dict
    """
    txt_filename = f"{category_name}_selected_users.txt"
    targets = {}
    
    # Check if we should use S3
    s3_client = _get_s3_client(s3_config)
    
    if s3_config and s3_config.get("enabled", False):
        # S3 is enabled - MUST read from S3 only
        if not s3_client:
            raise RuntimeError("[ERROR] S3 is enabled but failed to create S3 client. Install boto3: pip install boto3")
        
        bucket = s3_config.get("bucket")
        if not bucket:
            raise ValueError("[ERROR] S3 bucket is required when S3 is enabled")
        
        # Construct S3 key - use S3-specific prefix if provided, otherwise use user_lists_dir
        # Note: user_lists_s3_prefix is optional when sampling is enabled
        user_lists_s3_prefix = s3_config.get("user_lists_s3_prefix")
        if not user_lists_s3_prefix:
            # If not provided and we're in sampling mode, this shouldn't be called
            # But if we're here, use user_lists_dir as fallback (shouldn't happen in sampling mode)
            user_lists_s3_prefix = user_lists_dir
        s3_key = f"{user_lists_s3_prefix}/{txt_filename}".lstrip("/")
        
        content = _read_text_file_from_s3(s3_client, bucket, s3_key)
        if not content:
            raise FileNotFoundError(f"[ERROR] User list file not found in S3: s3://{bucket}/{s3_key}")
        
        lines = content.splitlines()
    else:
        # S3 is disabled - read from local file system
        filepath = Path(user_lists_dir) / txt_filename
        
        if not filepath.exists():
            print(f"   [WARNING] User list file not found: {filepath}")
            return {}
        
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                lines = f.readlines()
        except Exception as e:
            print(f"   [ERROR] Error reading {filepath}: {e}")
            return {}
    
    # Process lines
    try:
        for line in lines:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            if line.startswith('='):
                continue
            parts = line.split('\t')
            if len(parts) >= 2:
                try:
                    user_id = parts[0].strip()
                    review_count = int(parts[1].strip())
                    
                    if max_reviews_per_user is not None:
                        review_count = min(review_count, max_reviews_per_user)
                    
                    if review_count > 0:
                        targets[user_id] = review_count
                except ValueError: 
                    pass
        
        if max_users is not None and len(targets) > max_users:
            targets = dict(list(targets.items())[:max_users])
    except Exception as e:
        print(f"   [ERROR] Error processing user list: {e}")
        return {}
    
    return targets


def sample_from_existing_data(source_file: str, target_users: int, target_reviews: int,
                              enabled_categories: list, maintain_ratios: bool = True,
                              balanced: bool = True, use_all_categories: bool = True,
                              s3_config: Optional[Dict[str, Any]] = None):
    """Sample users and reviews from existing full_user_reviews.json."""
    print(f"\n{'='*70}")
    print("SAMPLING FROM EXISTING DATA")
    print(f"{'='*70}")
    print(f"Source file: {source_file}")
    print(f"Target users: {target_users}")
    print(f"Target reviews: {target_reviews}")
    print(f"Use all categories: {use_all_categories}")
    if not use_all_categories:
        print(f"Enabled categories: {enabled_categories}")
    
    # Check if we should use S3
    s3_client = _get_s3_client(s3_config)
    
    if s3_config and s3_config.get("enabled", False):
        # S3 is enabled - MUST read from S3 only
        if not s3_client:
            raise RuntimeError("[ERROR] S3 is enabled but failed to create S3 client. Install boto3: pip install boto3")
        
        bucket = s3_config.get("bucket")
        if not bucket:
            raise ValueError("[ERROR] S3 bucket is required when S3 is enabled")
        
        # Construct S3 key - use sampling_source_s3_prefix if provided
        if "sampling_source_s3_prefix" not in s3_config:
            raise ValueError("[ERROR] 'hyperparameters.s3.sampling_source_s3_prefix' is required when S3 is enabled and sampling is enabled")
        sampling_source_s3_prefix = s3_config["sampling_source_s3_prefix"]
        s3_key = f"{sampling_source_s3_prefix}/{source_file}".lstrip("/")
        
        print(f"\nLoading data from s3://{bucket}/{s3_key}...")
        try:
            response = s3_client.get_object(Bucket=bucket, Key=s3_key)
            content = response['Body'].read()
            if isinstance(content, bytes):
                content = content.decode('utf-8')
            full_data = json.loads(content)
        except s3_client.exceptions.NoSuchKey:
            raise FileNotFoundError(f"[ERROR] Source file not found in S3: s3://{bucket}/{s3_key}")
        except Exception as e:
            raise FileNotFoundError(f"[ERROR] Error reading from S3 s3://{bucket}/{s3_key}: {e}")
    else:
        # S3 is disabled - read from local file system
        source_path = Path(source_file)
        if not source_path.exists():
            raise FileNotFoundError(f"[ERROR] Source file not found: {source_file}")
        
        print(f"\nLoading data from {source_path}...")
        with open(source_path, 'r', encoding='utf-8') as f:
            full_data = json.load(f)
    
    print(f"Loaded {len(full_data)} users from source file")
    
    category_reviews, user_review_counts = defaultdict(list), defaultdict(int)
    print("\nCollecting reviews by category...")
    for user_id, user_data in full_data.items():
        for review in user_data.get('reviews', []):
            review_category = review.get('category', 'Unknown')
            if not review_category or review_category == 'None':
                review_category = 'Unknown'
            if not use_all_categories:
                main_category = map_category_to_main_category(review_category)
                if main_category and main_category in enabled_categories:
                    category_reviews[main_category].append((user_id, review))
                    user_review_counts[user_id] += 1
            else:
                category_reviews[review_category].append((user_id, review))
                user_review_counts[user_id] += 1
    
    print("\nOriginal category distribution:")
    total_reviews = sum(len(reviews) for reviews in category_reviews.values())
    category_ratios = {}
    for cat, reviews in category_reviews.items():
        ratio = len(reviews) / total_reviews if total_reviews > 0 else 0
        category_ratios[cat] = ratio
        print(f"  {cat}: {len(reviews)} reviews ({ratio*100:.2f}%)")
    
    if use_all_categories:
        all_categories = sorted(category_reviews.keys())
    else:
        all_categories = enabled_categories
    
    if maintain_ratios:
        target_by_category = {cat: int(target_reviews * category_ratios.get(cat, 0)) for cat in all_categories}
    else:
        reviews_per_category = target_reviews // len(all_categories) if all_categories else 0
        target_by_category = {cat: reviews_per_category for cat in all_categories}
    total_allocated = sum(target_by_category.values())
    if total_allocated < target_reviews:
        remainder = target_reviews - total_allocated
        sorted_cats = sorted(target_by_category.items(), key=lambda x: -x[1])
        for i in range(min(remainder, len(sorted_cats))):
            target_by_category[sorted_cats[i][0]] += 1
    
    print(f"\nTarget distribution ({len(all_categories)} categories):")
    for cat, target_count in sorted(target_by_category.items(), key=lambda x: -x[1]):
        if target_count > 0:
            print(f"  {cat}: {target_count} reviews")
    
    def _has_valid_product_description(review):
        """Check if review has a valid product description."""
        product_desc = review.get('product_description', '')
        if not product_desc or product_desc.strip() == "" or product_desc == "N/A":
            return False
        return True
    
    sampled_data, sampled_user_ids = {}, set()
    reviews_skipped_no_description = 0
    print("\nSampling reviews (filtering out reviews without product descriptions)...")
    categories_with_reviews = [cat for cat in all_categories if cat in category_reviews and len(category_reviews[cat]) > 0]
    
    if not categories_with_reviews:
        print("  [ERROR] No reviews found for any category!")
        return {}
    
    # First pass: Filter and sample valid reviews
    for category in categories_with_reviews:
        target_count = target_by_category[category]
        available_reviews = category_reviews[category]
        
        # Filter out reviews without valid product descriptions
        valid_reviews = [(uid, rev) for uid, rev in available_reviews if _has_valid_product_description(rev)]
        skipped_count = len(available_reviews) - len(valid_reviews)
        reviews_skipped_no_description += skipped_count
        
        if skipped_count > 0:
            print(f"  [SKIP] {category}: Skipped {skipped_count} reviews without product description")
        
        if len(valid_reviews) < target_count:
            print(f"  [WARNING] {category}: Only {len(valid_reviews)} reviews with valid descriptions available, requested {target_count}")
            target_count = len(valid_reviews)
        
        if balanced:
            user_review_map = defaultdict(list)
            for user_id, review in valid_reviews:
                user_review_map[user_id].append(review)
            users_with_reviews = list(user_review_map.keys())
            random.shuffle(users_with_reviews)
            sampled_count = 0
            for user_id in users_with_reviews:
                if sampled_count >= target_count:
                    break
                remaining = target_count - sampled_count
                reviews_to_take = min(len(user_review_map[user_id]), remaining)
                if user_id not in sampled_data:
                    sampled_data[user_id] = {'reviews': []}
                    sampled_user_ids.add(user_id)
                sampled_reviews = random.sample(user_review_map[user_id], reviews_to_take)
                sampled_data[user_id]['reviews'].extend(sampled_reviews)
                sampled_count += reviews_to_take
        else:
            sampled_reviews_data = random.sample(valid_reviews, target_count)
            for user_id, review in sampled_reviews_data:
                if user_id not in sampled_data:
                    sampled_data[user_id] = {'reviews': []}
                    sampled_user_ids.add(user_id)
                sampled_data[user_id]['reviews'].append(review)
        
        # Count how many were skipped
        skipped = len(available_reviews) - len(valid_reviews)
        if skipped > 0:
            reviews_skipped_no_description += skipped
            print(f"  [SKIP] {category}: Skipped {skipped} reviews without product description")
        if use_all_categories:
            cat_count = sum(1 for u_data in sampled_data.values() for r in u_data['reviews'] if (r.get('category') or 'Unknown') == category)
        else:
            cat_count = sum(1 for u_data in sampled_data.values() for r in u_data['reviews'] if map_category_to_main_category(r.get('category', '')) == category)
        print(f"  {category}: Sampled {cat_count} reviews")
    
    if len(sampled_user_ids) > target_users:
        print(f"\nLimiting users from {len(sampled_user_ids)} to {target_users}...")
        user_review_counts_sampled = {uid: len(data['reviews']) for uid, data in sampled_data.items()}
        users_to_keep = set(uid for uid, _ in sorted(user_review_counts_sampled.items(), key=lambda x: -x[1])[:target_users])
        sampled_data = {uid: data for uid, data in sampled_data.items() if uid in users_to_keep}
        sampled_user_ids = users_to_keep
    total_sampled_reviews = sum(len(u['reviews']) for u in sampled_data.values())
    print(f"\n{'='*70}\nSAMPLING COMPLETE\n{'='*70}")
    print(f"Sampled users: {len(sampled_data)}\nSampled reviews: {total_sampled_reviews}")
    if reviews_skipped_no_description > 0:
        print(f"Reviews skipped (no product description): {reviews_skipped_no_description}")
    
    # Final validation: Remove any reviews that somehow don't have product_description
    final_sampled_data = {}
    final_skipped = 0
    for user_id, user_data in sampled_data.items():
        valid_reviews = [r for r in user_data['reviews'] if _has_valid_product_description(r)]
        if valid_reviews:
            final_sampled_data[user_id] = {'reviews': valid_reviews}
        final_skipped += len(user_data['reviews']) - len(valid_reviews)
    
    if final_skipped > 0:
        print(f"[WARNING] Removed {final_skipped} additional reviews without product description during final validation")
    
    sampled_category_counts = defaultdict(int)
    for user_data in final_sampled_data.values():
        for review in user_data['reviews']:
            cat = review.get('category', 'Unknown')
            sampled_category_counts[cat if cat and cat != 'None' else 'Unknown'] += 1
    print("\nSampled category distribution:")
    for cat, count in sorted(sampled_category_counts.items(), key=lambda x: -x[1]):
        pct = (count / total_sampled_reviews * 100) if total_sampled_reviews > 0 else 0
        print(f"  {cat}: {count} reviews ({pct:.2f}%)")
    
    return final_sampled_data
