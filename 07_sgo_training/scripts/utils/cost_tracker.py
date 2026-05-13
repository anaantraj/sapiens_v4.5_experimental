"""
Cost Tracking Utility for LLM API Calls

Tracks all LLM API calls and calculates costs based on model pricing.
Provides detailed cost breakdown by model, call type, and usage.
"""
import os
import json
import time
from typing import Dict, List, Optional
from threading import Lock
from collections import defaultdict

# Model pricing per million tokens (from user specification)
MODEL_PRICING = {
    "o3": {
        "input": 2.00,      # $2.00 per million input tokens
        "cached_input": 0.50,  # $0.50 per million cached input tokens
        "output": 8.00      # $8.00 per million output tokens
    },
    "gpt-5.2": {
        "input": 1.75,      # $1.75 per million input tokens
        "cached_input": 0.175,  # $0.175 per million cached input tokens
        "output": 14.00     # $14.00 per million output tokens
    },
    "gpt-4o": {
        "input": 2.50,      # $2.50 per million input tokens
        "cached_input": 1.25,  # $1.25 per million cached input tokens
        "output": 10.00     # $10.00 per million output tokens
    },
    "gpt-4o-mini": {
        "input": 0.15,      # Standard pricing (approximate)
        "cached_input": 0.075,
        "output": 0.60
    },
    "gpt-5-mini": {
        "input": 0.15,      # Standard pricing (approximate)
        "cached_input": 0.075,
        "output": 0.60
    }
}

# Thread-safe cost tracker
_cost_tracker_lock = Lock()
_cost_data = {
    "calls": [],  # List of all API calls with details
    "summary": defaultdict(lambda: {
        "total_calls": 0,
        "total_input_tokens": 0,
        "total_output_tokens": 0,
        "total_cached_input_tokens": 0,
        "total_cost": 0.0,
        "by_call_type": defaultdict(lambda: {
            "calls": 0,
            "input_tokens": 0,
            "output_tokens": 0,
            "cached_input_tokens": 0,
            "cost": 0.0
        })
    })
}


def normalize_model_name(model: str) -> str:
    """
    Normalize model name to match pricing keys.
    Handles variations like "gpt-5.2", "gpt-5-mini", etc.
    """
    model_lower = model.lower()
    
    # Check for exact matches first
    if model_lower in MODEL_PRICING:
        return model_lower
    
    # Check for partial matches
    if "o3" in model_lower:
        return "o3"
    elif "gpt-5.2" in model_lower:
        return "gpt-5.2"
    elif "gpt-4o" in model_lower and "mini" not in model_lower:
        return "gpt-4o"
    elif "gpt-4o-mini" in model_lower:
        return "gpt-4o-mini"
    elif "gpt-5-mini" in model_lower:
        return "gpt-5-mini"
    
    # Default fallback
    return model_lower


def get_model_pricing(model: str) -> Optional[Dict]:
    """Get pricing for a model, returns None if not found."""
    normalized = normalize_model_name(model)
    return MODEL_PRICING.get(normalized)


def calculate_cost(
    model: str,
    input_tokens: int,
    output_tokens: int,
    cached_input_tokens: int = 0
) -> float:
    """
    Calculate cost for an API call.
    
    Args:
        model: Model name
        input_tokens: Number of input tokens
        output_tokens: Number of output tokens
        cached_input_tokens: Number of cached input tokens (if any)
    
    Returns:
        Total cost in USD
    """
    pricing = get_model_pricing(model)
    if not pricing:
        # Unknown model - return 0 or log warning
        return 0.0
    
    # Convert tokens to millions
    input_cost = (input_tokens / 1_000_000) * pricing["input"]
    cached_cost = (cached_input_tokens / 1_000_000) * pricing["cached_input"]
    output_cost = (output_tokens / 1_000_000) * pricing["output"]
    
    return input_cost + cached_cost + output_cost


def track_api_call(
    model: str,
    call_type: str,
    input_tokens: int,
    output_tokens: int,
    cached_input_tokens: int = 0,
    prompt_length: Optional[int] = None,
    response_length: Optional[int] = None,
    metadata: Optional[Dict] = None
):
    """
    Track an API call for cost calculation.
    
    Args:
        model: Model name used
        call_type: Type of call (e.g., "part_a_batch", "part_b_batch", "theme_logprobs", "review_generation")
        input_tokens: Number of input tokens
        output_tokens: Number of output tokens
        cached_input_tokens: Number of cached input tokens (if any)
        prompt_length: Optional prompt character length (for reference)
        response_length: Optional response character length (for reference)
        metadata: Optional additional metadata (batch_size, review_key, etc.)
    """
    with _cost_tracker_lock:
        normalized_model = normalize_model_name(model)
        cost = calculate_cost(model, input_tokens, output_tokens, cached_input_tokens)
        
        # Record individual call
        call_record = {
            "timestamp": time.time(),
            "model": model,
            "normalized_model": normalized_model,
            "call_type": call_type,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "cached_input_tokens": cached_input_tokens,
            "cost": cost,
            "prompt_length": prompt_length,
            "response_length": response_length,
            "metadata": metadata or {}
        }
        _cost_data["calls"].append(call_record)
        
        # Update summary
        summary = _cost_data["summary"][normalized_model]
        summary["total_calls"] += 1
        summary["total_input_tokens"] += input_tokens
        summary["total_output_tokens"] += output_tokens
        summary["total_cached_input_tokens"] += cached_input_tokens
        summary["total_cost"] += cost
        
        # Update by call type
        by_type = summary["by_call_type"][call_type]
        by_type["calls"] += 1
        by_type["input_tokens"] += input_tokens
        by_type["output_tokens"] += output_tokens
        by_type["cached_input_tokens"] += cached_input_tokens
        by_type["cost"] += cost


def extract_token_usage(response) -> tuple:
    """
    Extract token usage from OpenAI API response.
    Returns (input_tokens, output_tokens, cached_input_tokens)
    
    Handles various response formats and missing usage data gracefully.
    """
    input_tokens = 0
    output_tokens = 0
    cached_input_tokens = 0
    
    try:
        if hasattr(response, 'usage') and response.usage:
            usage = response.usage
            # Try different attribute names for input tokens
            if hasattr(usage, 'prompt_tokens'):
                input_tokens = usage.prompt_tokens or 0
            elif hasattr(usage, 'input_tokens'):
                input_tokens = usage.input_tokens or 0
            
            # Try different attribute names for output tokens
            if hasattr(usage, 'completion_tokens'):
                output_tokens = usage.completion_tokens or 0
            elif hasattr(usage, 'output_tokens'):
                output_tokens = usage.output_tokens or 0
            
            # Try different attribute names for cached tokens
            if hasattr(usage, 'cached_tokens'):
                cached_input_tokens = usage.cached_tokens or 0
            elif hasattr(usage, 'cache_read_tokens'):
                cached_input_tokens = usage.cache_read_tokens or 0
            elif hasattr(usage, 'cached_input_tokens'):
                cached_input_tokens = usage.cached_input_tokens or 0
    except Exception:
        # If anything fails, return zeros (cost will be 0, but call is still tracked)
        pass
    
    return input_tokens, output_tokens, cached_input_tokens


def get_cost_summary() -> Dict:
    """Get current cost summary."""
    with _cost_tracker_lock:
        return {
            "summary": dict(_cost_data["summary"]),
            "total_calls": len(_cost_data["calls"]),
            "grand_total_cost": sum(s["total_cost"] for s in _cost_data["summary"].values())
        }


def get_detailed_report() -> Dict:
    """Get detailed cost report with all calls."""
    with _cost_tracker_lock:
        summary = dict(_cost_data["summary"])
        for model, data in summary.items():
            summary[model]["by_call_type"] = dict(data["by_call_type"])
        
        return {
            "summary": summary,
            "all_calls": _cost_data["calls"].copy(),
            "total_calls": len(_cost_data["calls"]),
            "grand_total_cost": sum(s["total_cost"] for s in _cost_data["summary"].values()),
            "report_timestamp": time.time()
        }


def save_cost_report(output_file: str):
    """Save detailed cost report to JSON file."""
    report = get_detailed_report()
    os.makedirs(os.path.dirname(output_file) if os.path.dirname(output_file) else ".", exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(report, f, indent=2)
    return report


def reset_cost_tracker():
    """Reset all cost tracking data (useful for testing or new runs)."""
    with _cost_tracker_lock:
        _cost_data["calls"] = []
        _cost_data["summary"] = defaultdict(lambda: {
            "total_calls": 0,
            "total_input_tokens": 0,
            "total_output_tokens": 0,
            "total_cached_input_tokens": 0,
            "total_cost": 0.0,
            "by_call_type": defaultdict(lambda: {
                "calls": 0,
                "input_tokens": 0,
                "output_tokens": 0,
                "cached_input_tokens": 0,
                "cost": 0.0
            })
        })


def print_cost_summary():
    """Print a formatted cost summary to console."""
    summary = get_cost_summary()
    
    print("\n" + "="*80)
    print("LLM API COST SUMMARY")
    print("="*80)
    
    if summary["total_calls"] == 0:
        print("No API calls tracked yet.")
        return
    
    grand_total = summary["grand_total_cost"]
    
    for model, data in sorted(summary["summary"].items()):
        print(f"\n📊 Model: {model}")
        print(f"   Total Calls: {data['total_calls']}")
        print(f"   Input Tokens: {data['total_input_tokens']:,} ({data['total_input_tokens']/1_000_000:.3f}M)")
        print(f"   Output Tokens: {data['total_output_tokens']:,} ({data['total_output_tokens']/1_000_000:.3f}M)")
        if data['total_cached_input_tokens'] > 0:
            print(f"   Cached Input Tokens: {data['total_cached_input_tokens']:,} ({data['total_cached_input_tokens']/1_000_000:.3f}M)")
        print(f"   Total Cost: ${data['total_cost']:.4f}")
        
        # Breakdown by call type
        if data['by_call_type']:
            print(f"   Breakdown by Call Type:")
            for call_type, type_data in sorted(data['by_call_type'].items()):
                print(f"      - {call_type}:")
                print(f"          Calls: {type_data['calls']}")
                print(f"          Input: {type_data['input_tokens']:,} tokens")
                print(f"          Output: {type_data['output_tokens']:,} tokens")
                if type_data['cached_input_tokens'] > 0:
                    print(f"          Cached: {type_data['cached_input_tokens']:,} tokens")
                print(f"          Cost: ${type_data['cost']:.4f}")
    
    print(f"\n{'='*80}")
    print(f"GRAND TOTAL COST: ${grand_total:.4f}")
    print(f"Total API Calls: {summary['total_calls']}")
    print("="*80 + "\n")

