"""File I/O utilities for dataset pull."""
import json
import gzip
import io
from pathlib import Path
from typing import Optional, Iterator, Dict, Any


def _get_s3_client(s3_config: Optional[Dict[str, Any]] = None):
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


def _read_from_s3(s3_client, bucket: str, key: str, is_gzipped: bool = False) -> Iterator[Dict]:
    """Read JSONL file from S3."""
    try:
        response = s3_client.get_object(Bucket=bucket, Key=key)
        content = response['Body'].read()
        
        if is_gzipped:
            content = gzip.decompress(content)
        
        # Decode bytes to string
        if isinstance(content, bytes):
            content = content.decode('utf-8')
        
        for line in content.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue
    except s3_client.exceptions.NoSuchKey:
        print(f"   [ERROR] S3 key not found: s3://{bucket}/{key}")
        return
    except Exception as e:
        print(f"   [ERROR] Error reading from S3 s3://{bucket}/{key}: {e}")
        return


def read_jsonl(file_path, s3_config: Optional[Dict[str, Any]] = None):
    """
    Robust JSONL reader supporting both .jsonl and .jsonl.gz files.
    Supports both local files and S3.
    
    Args:
        file_path: Local file path or S3 key (if using S3)
        s3_config: Optional S3 configuration dict with keys: enabled, bucket, region, prefix
    """
    # Check if we should use S3
    s3_client = _get_s3_client(s3_config)
    
    if s3_config and s3_config.get("enabled", False):
        # S3 is enabled - MUST read from S3 only
        if not s3_client:
            raise RuntimeError("[ERROR] S3 is enabled but failed to create S3 client. Install boto3: pip install boto3")
        
        bucket = s3_config.get("bucket")
        if not bucket:
            raise ValueError("[ERROR] S3 bucket is required when S3 is enabled")
        
        # Construct S3 key - file_path should already be the full S3 key
        s3_key = str(file_path).lstrip("/")
        
        # Check if file is gzipped
        is_gzipped = s3_key.endswith('.gz')
        
        yield from _read_from_s3(s3_client, bucket, s3_key, is_gzipped)
        return
    
    # S3 is disabled - read from local file system
    file_path = str(file_path)
    if file_path.endswith('.gz'):
        open_func = gzip.open
        mode = 'rt'
    else:
        open_func = open
        mode = 'r'
    
    try:
        with open_func(file_path, mode, encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    continue
    except FileNotFoundError:
        print(f"   [ERROR] File not found: {file_path}")
        return
    except Exception as e:
        print(f"   [ERROR] Error reading {file_path}: {e}")
        return


def load_existing_database(output_file: Path, s3_config: Optional[Dict[str, Any]] = None):
    """
    Load existing database and create duplicate check set.
    Supports both local files and S3.
    
    Args:
        output_file: Local file path
        s3_config: Optional S3 configuration (not used for output file, only for reference)
    """
    if output_file.exists():
        print(f"[INFO] Loading existing database from {output_file}...")
        try:
            with open(output_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                existing_keys = set()
                for uid, user_data in data.items():
                    for review in user_data.get('reviews', []):
                        if 'asin' in review:
                            existing_keys.add((uid, review['asin']))
                print(f"       Loaded {len(data)} users with {len(existing_keys)} total reviews.")
                return data, existing_keys
        except json.JSONDecodeError as e:
            raise ValueError(
                f"Failed to parse existing database file '{output_file}'. "
                f"The file appears to be corrupted or invalid JSON. Error: {e}"
            ) from e
    else:
        print("   No existing database found. Starting fresh.")
        return {}, set()
