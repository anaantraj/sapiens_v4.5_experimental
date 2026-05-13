#!/usr/bin/env python3
"""
Stage 04: Review Topic Classification - Analyze Empty Themes
============================================================

Analyze review topic classifications to count reviews with empty predicted_themes per category.

This script analyzes the output from Stage 04 (review topic classification) to identify
reviews that have empty predicted_themes, which can indicate classification issues.

Usage:
    # Using local files
    python 04_review_topic_classification/scripts/05_analyze_empty_themes.py --input review_topics_train.jsonl --output empty_themes_analysis.txt
    
    # Using W&B artifacts (default from config)
    python 04_review_topic_classification/scripts/05_analyze_empty_themes.py
    
    # Using specific W&B artifacts
    python 04_review_topic_classification/scripts/05_analyze_empty_themes.py --input-artifact review_topics_classified_v4:latest --output-artifact empty_themes_analysis_v4
"""

import json
import sys
import argparse
from collections import defaultdict
from pathlib import Path
from typing import Optional, Dict, Any

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from utils.wandb_utils import (
    load_config, get_stage_config,
    init_wandb_run, finish_run, use_artifact, log_artifact,
    log_metrics, log_summary
)


def analyze_empty_themes(input_file_path: Path, output_file_path: Optional[Path] = None):
    """
    Analyze the predictions file and count empty themes per category.
    
    Args:
        input_file_path: Path to review_topics JSONL file
        output_file_path: Optional path to write output file
        
    Returns:
        Dictionary with analysis results
    """
    
    category_stats = defaultdict(lambda: {'total': 0, 'empty': 0, 'with_themes': 0})
    
    try:
        with open(input_file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                
                try:
                    result = json.loads(line)
                    # Try different field names for category
                    main_category = (
                        result.get('main_category') or 
                        result.get('category') or 
                        result.get('review_category') or 
                        'Unknown'
                    )
                    
                    # Try different field names for predicted themes
                    predicted_themes = (
                        result.get('predicted_themes') or
                        result.get('themes') or
                        list(result.get('topic_probabilities', {}).keys()) or
                        []
                    )
                    
                    category_stats[main_category]['total'] += 1
                    
                    if not predicted_themes or len(predicted_themes) == 0:
                        category_stats[main_category]['empty'] += 1
                    else:
                        category_stats[main_category]['with_themes'] += 1
                        
                except json.JSONDecodeError as e:
                    print(f"Error parsing line {line_num}: {e}")
                    continue
    
    except FileNotFoundError:
        print(f"Error: File '{input_file_path}' not found.")
        return None
    except Exception as e:
        print(f"Error reading file: {e}")
        return None
    
    # Prepare results
    total_all = 0
    empty_all = 0
    with_themes_all = 0
    
    results = []
    for category in sorted(category_stats.keys()):
        stats = category_stats[category]
        total = stats['total']
        empty = stats['empty']
        with_themes = stats['with_themes']
        empty_pct = (empty / total * 100) if total > 0 else 0
        
        results.append({
            'category': category,
            'total': total,
            'empty': empty,
            'with_themes': with_themes,
            'empty_pct': empty_pct
        })
        
        total_all += total
        empty_all += empty
        with_themes_all += with_themes
    
    empty_pct_all = (empty_all / total_all * 100) if total_all > 0 else 0
    
    # Build output text
    output_lines = []
    output_lines.append("=" * 80)
    output_lines.append("REVIEWS WITH EMPTY PREDICTED_THEMES BY CATEGORY")
    output_lines.append("=" * 80)
    output_lines.append(f"\n{'Category':<30} {'Total':<10} {'Empty Themes':<15} {'With Themes':<15} {'Empty %':<10}")
    output_lines.append("-" * 80)
    
    for result in results:
        output_lines.append(
            f"{result['category']:<30} {result['total']:<10} {result['empty']:<15} "
            f"{result['with_themes']:<15} {result['empty_pct']:<10.2f}%"
        )
    
    output_lines.append("-" * 80)
    output_lines.append(
        f"{'TOTAL':<30} {total_all:<10} {empty_all:<15} {with_themes_all:<15} {empty_pct_all:<10.2f}%"
    )
    output_lines.append("=" * 80)
    
    # Summary
    output_lines.append(f"\nSUMMARY:")
    output_lines.append(f"  Total reviews analyzed: {total_all:,}")
    output_lines.append(f"  Reviews with empty themes: {empty_all:,} ({empty_pct_all:.2f}%)")
    output_lines.append(f"  Reviews with themes: {with_themes_all:,} ({100-empty_pct_all:.2f}%)")
    
    # Show categories with highest empty percentage
    output_lines.append(f"\nCategories with highest empty theme percentage:")
    sorted_by_pct = sorted(
        [(cat, stats['empty'], stats['total'], (stats['empty']/stats['total']*100) if stats['total'] > 0 else 0) 
         for cat, stats in category_stats.items()],
        key=lambda x: x[3],
        reverse=True
    )
    
    for cat, empty, total, pct in sorted_by_pct[:5]:
        output_lines.append(f"  {cat}: {empty}/{total} ({pct:.2f}%)")
    
    output_text = "\n".join(output_lines)
    
    # Print to console
    print(output_text)
    
    # Write to file if specified
    if output_file_path:
        output_file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file_path, 'w', encoding='utf-8') as f:
            f.write(output_text)
        print(f"\nResults written to: {output_file_path}")
    
    # Return structured results for JSON output
    return {
        'summary': {
            'total_reviews': total_all,
            'empty_themes': empty_all,
            'with_themes': with_themes_all,
            'empty_pct': empty_pct_all
        },
        'by_category': results,
        'top_empty_categories': [
            {'category': cat, 'empty': empty, 'total': total, 'empty_pct': pct}
            for cat, empty, total, pct in sorted_by_pct[:5]
        ]
    }


def main():
    parser = argparse.ArgumentParser(
        description="Analyze review topic classifications to count reviews with empty predicted_themes per category."
    )
    
    # Input options (mutually exclusive)
    input_group = parser.add_mutually_exclusive_group(required=False)
    input_group.add_argument(
        '--input',
        type=str,
        help='Local path to review_topics JSONL file'
    )
    input_group.add_argument(
        '--input-artifact',
        type=str,
        help='W&B artifact name (e.g., "review_topics_classified_v4:latest"). If not specified, uses config default.'
    )
    
    # Output options
    parser.add_argument(
        '--output',
        type=str,
        help='Local path to write output text file'
    )
    parser.add_argument(
        '--output-artifact',
        type=str,
        help='W&B artifact name for output (e.g., "empty_themes_analysis_v4"). If not specified, uses config default.'
    )
    parser.add_argument(
        '--output-json',
        type=str,
        help='Local path to write output JSON file'
    )
    
    # W&B options
    parser.add_argument(
        '--wandb-disabled',
        action='store_true',
        help='Disable W&B logging'
    )
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config()
    stage_config = get_stage_config(config, "04_review_topic_classification")
    
    # Initialize W&B run if enabled
    run = None
    if not args.wandb_disabled:
        try:
            run = init_wandb_run(
                stage="04_review_topic_classification",
                job_type="analysis",
                config=stage_config
            )
        except Exception as e:
            print(f"[WARNING] Could not initialize W&B: {e}")
            print("Continuing without W&B...")
            run = None
    
    # Resolve input file path
    input_file_path = None
    
    if args.input:
        input_file_path = Path(args.input)
    elif args.input_artifact:
        if run:
            artifact_path = use_artifact(run, args.input_artifact, "dataset")
            if not artifact_path:
                print(f"Error: Could not download artifact '{args.input_artifact}'")
                return
            # Look for review_topics JSONL files in artifact directory
            jsonl_files = list(artifact_path.glob("review_topics_*.jsonl"))
            if not jsonl_files:
                jsonl_files = list(artifact_path.glob("*.jsonl"))
            if jsonl_files:
                input_file_path = jsonl_files[0]
                print(f"Using file: {input_file_path}")
            else:
                print(f"Error: Could not find review_topics JSONL file in artifact '{args.input_artifact}'")
                return
        else:
            print("Error: W&B is disabled. Cannot use --input-artifact without W&B.")
            return
    else:
        # Use default from config
        default_artifact = stage_config.get("output_artifact", "review_topics_classified_v4:latest")
        if run:
            artifact_path = use_artifact(run, default_artifact, "dataset")
            if not artifact_path:
                print(f"Error: Could not download artifact '{default_artifact}'")
                return
            # Look for review_topics JSONL files
            jsonl_files = list(artifact_path.glob("review_topics_*.jsonl"))
            if not jsonl_files:
                jsonl_files = list(artifact_path.glob("*.jsonl"))
            if jsonl_files:
                input_file_path = jsonl_files[0]
                print(f"Using default artifact '{default_artifact}' with file: {input_file_path}")
            else:
                print(f"Error: Could not find review_topics JSONL file in artifact '{default_artifact}'")
                return
        else:
            print("Error: No input specified and W&B is disabled. Use --input or --input-artifact.")
            return
    
    if not input_file_path or not input_file_path.exists():
        print(f"Error: Input file not found: {input_file_path}")
        return
    
    # Resolve output file path
    output_file_path = None
    if args.output:
        output_file_path = Path(args.output)
    
    # Run analysis
    results = analyze_empty_themes(input_file_path, output_file_path)
    
    if results is None:
        return
    
    # Write JSON output if requested
    if args.output_json:
        json_path = Path(args.output_json)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2)
        print(f"JSON results written to: {json_path}")
    
    # Log to W&B if enabled
    output_artifact_name = args.output_artifact or stage_config.get("empty_themes_analysis_output_artifact")
    if run and output_artifact_name:
        # Create output directory for artifact
        output_dir = Path("empty_themes_analysis_output")
        output_dir.mkdir(exist_ok=True)
        
        # Write text output
        text_file = output_dir / "empty_themes_analysis.txt"
        if output_file_path:
            import shutil
            shutil.copy(output_file_path, text_file)
        else:
            with open(text_file, 'w', encoding='utf-8') as f:
                f.write("\n".join([
                    "=" * 80,
                    "REVIEWS WITH EMPTY PREDICTED_THEMES BY CATEGORY",
                    "=" * 80,
                    ""
                ]))
                for result in results['by_category']:
                    f.write(
                        f"{result['category']:<30} {result['total']:<10} "
                        f"{result['empty']:<15} {result['with_themes']:<15} "
                        f"{result['empty_pct']:<10.2f}%\n"
                    )
        
        # Write JSON output
        json_file = output_dir / "empty_themes_analysis.json"
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2)
        
        # Log artifact
        log_artifact(
            run=run,
            artifact_name=output_artifact_name,
            artifact_type="result",
            artifact_path=output_dir,
            metadata={
                'total_reviews': results['summary']['total_reviews'],
                'empty_themes': results['summary']['empty_themes'],
                'empty_pct': results['summary']['empty_pct']
            }
        )
        
        # Log metrics
        log_metrics(run, {
            'total_reviews': results['summary']['total_reviews'],
            'empty_themes': results['summary']['empty_themes'],
            'empty_pct': results['summary']['empty_pct']
        })
        
        print(f"\n[W&B] Logged artifact: {output_artifact_name}")
    
    # Finish W&B run
    if run:
        finish_run(run)


if __name__ == "__main__":
    main()

