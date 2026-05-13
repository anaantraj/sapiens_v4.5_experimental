#!/usr/bin/env python3
"""
Stage 03: Topic Universe - Shrink/Consolidate Topics
=====================================================

Consolidates discovered topics using LLM to create a refined, high-level topic list.
- Reads configuration from config.yaml
- Loads topic universe from Stage 03.1 (merged structure)
- Processes each category separately
- Uses LLM to consolidate topics per category
- Updates merged topic universe with shrunk topics
- Validates with schema before saving
- Logs artifact to W&B

Usage:
    # Process specific category
    python 03_topic_universe/scripts/02_shrink_topics.py --category "Clothing_Shoes_and_Jewelry"
    
    # Process all categories in topic universe
    python 03_topic_universe/scripts/02_shrink_topics.py --all-categories
"""

import json
import os
import re
import sys
import asyncio
import tiktoken
import argparse
from pathlib import Path

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Load .env so OPENAI_API_KEY and OPENAI_BASE_URL (Bedrock) are set
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent.parent.parent / ".env", override=True)
except ImportError:
    pass

from utils.wandb_utils import (
    load_config, get_stage_config, get_openai_config,
    init_wandb_run, finish_run, use_artifact, log_artifact,
    log_metrics, log_summary, link_to_registry, get_artifact_dir
)
from utils.openai_client import create_async_openai_client

# Import schema for validation
from schemas.learned_artifacts import TopicUniverseArtifact

# Use the appropriate tokenizer for the model
encoding = tiktoken.get_encoding("cl100k_base")

# Prompt directory
PROMPT_DIR = Path(__file__).parent.parent / "prompts"


def load_prompt(prompt_filename: str) -> str:
    """Load a prompt template from file."""
    prompt_path = PROMPT_DIR / prompt_filename
    if prompt_path.exists():
        with open(prompt_path, 'r', encoding='utf-8') as f:
            return f.read().strip()
    else:
        raise FileNotFoundError(f"Prompt file not found: {prompt_path}")


async def consolidate_themes_with_llm(
    themes: list,
    category: str,
    client,
    model: str,
    max_tokens: int,
    prompt_config: dict
) -> list:
    """
    Uses an LLM (via AWS Bedrock client) to consolidate themes into a final list.
    """
    if not themes:
        return []

    themes_list_str = "\n".join([f"- {theme}" for theme in themes])
    user_prompt_template = load_prompt("theme_consolidation_user_prompt.txt")
    prompt_text = user_prompt_template.format(
        category=category,
        themes_list=themes_list_str,
        shrink_min_themes=prompt_config.get("shrink_min_themes", 4),
        shrink_max_themes=prompt_config.get("shrink_max_themes", 20)
    )

    payload_tokens = len(encoding.encode(prompt_text))
    if payload_tokens > max_tokens:
        print(f"[WARNING] Payload size ({payload_tokens}) exceeds max tokens ({max_tokens}). The LLM may truncate input.")
    print(f"  Total tokens for this LLM call: {payload_tokens}")

    try:
        print(f"  Calling LLM (Bedrock) to consolidate themes for '{category}'...")
        response = await client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt_text}],
            response_format={"type": "json_object"},
            max_tokens=min(4096, max(256, max_tokens - payload_tokens)),
        )
        if not response.choices or not response.choices[0].message:
            print("  [ERROR] LLM response missing choices/message.")
            return []
        response_content = (response.choices[0].message.content or "{}").strip()
        # Fix common Bedrock JSON glitches (same as 01 script)
        if response_content.startswith('{"{'):
            response_content = "{" + response_content[3:]
        response_content = re.sub(r'^\s*\{\s*(?=\{\s*"final_themes")', '', response_content)
        response_json = json.loads(response_content)
        final_themes = response_json.get("final_themes", [])
        if not isinstance(final_themes, list):
            print("  [WARNING] LLM returned an invalid final themes format.")
            return []
        return final_themes
    except json.JSONDecodeError as e:
        print(f"  [ERROR] LLM did not return valid JSON: {e}")
        return []
    except Exception as e:
        print(f"  [ERROR] API call failed: {e}")
        return []


async def shrink_category_topics(
    category: str,
    initial_themes: list,
    client,
    model: str,
    max_tokens: int,
    prompt_config: dict
) -> list:
    """
    Shrink/consolidate topics for a single category.
    """
    print(f"\n  Processing category: {category}")
    print(f"  Initial themes count: {len(initial_themes)}")

    if not initial_themes:
        print(f"  [SKIP] No themes to consolidate for {category}")
        return []

    consolidated = await consolidate_themes_with_llm(
        initial_themes,
        category,
        client,
        model,
        max_tokens,
        prompt_config
    )
    
    print(f"  Consolidated themes count: {len(consolidated)}")
    print(f"  Reduction: {len(initial_themes)} → {len(consolidated)} ({len(initial_themes) - len(consolidated)} removed)")
    
    return consolidated


async def main():
    """Main execution function."""
    
    # =========================================================================
    # Step 1: Parse arguments and load configuration
    # =========================================================================
    parser = argparse.ArgumentParser(description="Shrink/consolidate topic universe")
    parser.add_argument(
        "--category",
        type=str,
        help="Specific category to process (e.g., 'Clothing_Shoes_and_Jewelry')"
    )
    parser.add_argument(
        "--all-categories",
        action="store_true",
        help="Process all categories in the topic universe"
    )
    parser.add_argument(
        "--input-artifact",
        type=str,
        help="Input artifact name (default: from config)"
    )
    parser.add_argument(
        "--output-artifact",
        type=str,
        help="Output artifact name (default: from config)"
    )
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("STAGE 03: Topic Universe - Shrink Topics")
    print("=" * 70)
    
    cfg = get_stage_config("03_topic_universe")
    openai_cfg = get_openai_config()
    hyperparams = cfg.get("hyperparameters", {})
    
    # Input artifact should be the OUTPUT from 01_create_topic_universe_mapreduce.py
    # Default to shrink_input_artifact from config, or fallback to output_artifact
    input_artifact_name = args.input_artifact or cfg.get("shrink_input_artifact") or f"{cfg.get('output_artifact', 'topic_universe_v4')}:latest"
    output_artifact_name = args.output_artifact or cfg.get("shrink_output_artifact", "topic_universe_final_sampled_500users_v1")
    model = hyperparams.get("model", openai_cfg.get("analysis_model", "o3"))
    # When using Bedrock, use config bedrock_model_id if set; otherwise use model
    base_url = os.environ.get("OPENAI_BASE_URL", "").strip()
    if base_url and "bedrock-mantle" in base_url and cfg.get("bedrock_model_id"):
        model = cfg.get("bedrock_model_id")
    max_tokens = hyperparams.get("max_tokens", 200000)

    # Get prompt configuration (with defaults)
    prompt_config = hyperparams.get("prompt_config", {})
    if "shrink_min_themes" not in prompt_config:
        prompt_config["shrink_min_themes"] = 4
    if "shrink_max_themes" not in prompt_config:
        prompt_config["shrink_max_themes"] = 20

    # Create Bedrock (or configured endpoint) client
    if not os.environ.get("OPENAI_API_KEY"):
        print("[ERROR] OPENAI_API_KEY environment variable not set (set in .env or export)")
        return
    llm_client = create_async_openai_client()

    print(f"\n[Config] Input artifact: {input_artifact_name}")
    print(f"[Config] Output artifact: {output_artifact_name}")
    print(f"[Config] Model: {model}")
    print(f"[Config] Max tokens: {max_tokens}")
    
    # =========================================================================
    # Step 2: Initialize W&B run
    # =========================================================================
    print("\n" + "-" * 70)
    print("Step 2: Initialize W&B Run")
    print("-" * 70)
    
    run = init_wandb_run(
        run_name=f"shrink_topics_{output_artifact_name}",
        stage="03_topic_universe",
        job_type="topic_consolidation"
    )
    
    try:
        # =====================================================================
        # Step 3: Load input topic universe
        # =====================================================================
        print("\n" + "-" * 70)
        print("Step 3: Load Input Topic Universe")
        print("-" * 70)
        
        # PRIORITY: Use local artifact first (most recent), then fallback to W&B
        project_root = Path(__file__).parent.parent.parent
        artifact_base_name = input_artifact_name.split(":")[0]  # Remove version
        local_artifact_dir = get_artifact_dir("03_topic_universe", artifact_base_name)
        local_merged_file = local_artifact_dir / "topic_universe.json"
        
        merged_file = None
        input_path = None
        
        if local_merged_file.exists():
            print(f"[INFO] Using LOCAL artifact (most recent): {local_merged_file}")
            input_path = local_artifact_dir
            merged_file = local_merged_file
        else:
            # Fallback to W&B artifact
            print(f"  Attempting to load artifact from W&B: {input_artifact_name}")
            input_path = use_artifact(run, input_artifact_name, artifact_type="dataset")
            
            if input_path is not None:
                # Check for merged file first (new format)
                merged_file = input_path / "topic_universe.json"
                if not merged_file.exists():
                    # Try legacy format - find any final_themes_*.json files
                    legacy_files = list(input_path.glob("final_themes_*.json"))
                    if legacy_files:
                        print(f"[INFO] Found legacy format, merging...")
                        # Load and merge legacy files
                        topic_universe = TopicUniverseArtifact.merge_from_files(legacy_files)
                        merged_file = input_path / "topic_universe.json"
                        topic_universe.to_merged_file(merged_file)
                        print(f"[OK] Created merged file from legacy format")
                    else:
                        print(f"[ERROR] Artifact downloaded but no topic_universe.json or final_themes_*.json files found")
                        print(f"  Artifact path: {input_path}")
                        print(f"  Files in artifact: {list(input_path.glob('*'))}")
                        return
                else:
                    print(f"[OK] Found merged topic_universe.json in artifact")
        
        # Fallback to local artifacts directory if W&B download failed
        if merged_file is None or not merged_file.exists():
            print(f"[INFO] Could not download from W&B, trying local artifacts...")
            artifact_base_name = input_artifact_name.split(":")[0]  # Remove version
            local_artifact_dir = get_artifact_dir("03_topic_universe", artifact_base_name)
            merged_file = local_artifact_dir / "topic_universe.json"
            
            if merged_file.exists():
                input_path = local_artifact_dir
                print(f"[OK] Found local artifact at: {merged_file}")
            else:
                print(f"[ERROR] Could not find topic universe artifact")
                print(f"  Tried W&B artifact: {input_artifact_name}")
                print(f"  Tried local path: {merged_file}")
                print(f"\n  Make sure to run 01_create_topic_universe_mapreduce.py first!")
                return
        
        # Load topic universe with schema validation
        print(f"  Loading from: {merged_file}")
        topic_universe = TopicUniverseArtifact.from_file(merged_file)
        print(f"[OK] Loaded topic universe with {len(topic_universe.topics_by_category)} categories")
        
        # =====================================================================
        # Step 4: Determine which categories to process
        # =====================================================================
        print("\n" + "-" * 70)
        print("Step 4: Determine Categories to Process")
        print("-" * 70)
        
        all_categories = list(topic_universe.topics_by_category.keys())
        hyperparams = cfg.get("hyperparameters", {})
        use_parallel = hyperparams.get("parallel", False)
        
        if args.all_categories:
            categories_to_process = all_categories
            print(f"  Processing ALL categories: {categories_to_process}")
        elif args.category:
            if args.category not in all_categories:
                print(f"[ERROR] Category '{args.category}' not found in topic universe")
                print(f"  Available categories: {all_categories}")
                return
            categories_to_process = [args.category]
            print(f"  Processing single category: {args.category}")
        else:
            # Default: process ALL categories
            categories_to_process = all_categories
            print(f"  Processing ALL categories (default): {categories_to_process}")
        
        print(f"  Parallel processing: {use_parallel}")
        
        # =====================================================================
        # Step 5: Shrink topics for each category (parallel or sequential)
        # =====================================================================
        print("\n" + "-" * 70)
        print(f"Step 5: Shrink Topics ({'PARALLEL' if use_parallel else 'SEQUENTIAL'})")
        print("-" * 70)
        
        # Create tasks for all categories
        tasks = []
        
        for category in categories_to_process:
            initial_themes = topic_universe.get_topics_for_category(category)
            if initial_themes:
                task = shrink_category_topics(
                    category,
                    initial_themes,
                    llm_client,
                    model,
                    max_tokens,
                    prompt_config
                )
                tasks.append((category, task))
        
        # Execute in parallel or sequential
        if use_parallel and len(tasks) > 1:
            print(f"[INFO] Processing {len(tasks)} categories in PARALLEL...")
            # Execute all tasks in parallel
            results = await asyncio.gather(*[task for _, task in tasks])
            # Map results back to categories
            consolidated_by_category = {
                category: result 
                for (category, _), result in zip(tasks, results)
            }
        else:
            print(f"[INFO] Processing {len(tasks)} categories SEQUENTIALLY...")
            consolidated_by_category = {}
            for category, task in tasks:
                result = await task
                consolidated_by_category[category] = result
        
        # Merge results
        shrunk_topics = {}
        category_stats = {}
        
        for category in categories_to_process:
            initial_themes = topic_universe.get_topics_for_category(category)
            if not initial_themes:
                continue
                
            consolidated = consolidated_by_category.get(category, [])
            
            if consolidated:
                shrunk_topics[category] = consolidated
                category_stats[category] = {
                    "initial_count": len(initial_themes),
                    "shrunk_count": len(consolidated),
                    "reduction": len(initial_themes) - len(consolidated)
                }
                
                # Log progress
                log_metrics(run, {
                    f"category/{category}/initial_themes": len(initial_themes),
                    f"category/{category}/shrunk_themes": len(consolidated),
                    f"category/{category}/reduction": len(initial_themes) - len(consolidated),
                })
            else:
                print(f"  [WARNING] No consolidated themes for {category}, keeping original")
                shrunk_topics[category] = initial_themes
        
        # =====================================================================
        # Step 6: Update topic universe with shrunk topics
        # =====================================================================
        print("\n" + "-" * 70)
        print("Step 6: Update Topic Universe (with Schema Validation)")
        print("-" * 70)
        
        # Update the validated artifact
        topic_universe.topics_by_category.update(shrunk_topics)
        
        # Validate the updated artifact
        # (Pydantic validation happens automatically)
        print(f"[OK] Schema validation passed")
        print(f"  Total categories: {len(topic_universe.topics_by_category)}")
        for category, stats in category_stats.items():
            print(f"  {category}: {stats['initial_count']} → {stats['shrunk_count']} themes")
        
        # =====================================================================
        # Step 7: Save shrunk topic universe
        # =====================================================================
        print("\n" + "-" * 70)
        print("Step 7: Save Shrunk Topic Universe")
        print("-" * 70)
        
        output_dir = get_artifact_dir("03_topic_universe", output_artifact_name)
        output_file = output_dir / "topic_universe.json"
        
        topic_universe.to_merged_file(output_file)
        print(f"[OK] Saved shrunk topic universe to: {output_file}")
        
        # Also save individual category files (for future use)
        print(f"\n  Saving individual category files...")
        for category_name, themes in shrunk_topics.items():
            category_file = output_dir / f"final_themes_shrunk_{category_name.replace(' ', '_')}.json"
            with open(category_file, 'w', encoding='utf-8') as f:
                json.dump(themes, f, indent=2, ensure_ascii=False)
            print(f"    Saved: {category_file.name} ({len(themes)} themes)")
        print(f"[OK] Saved {len(shrunk_topics)} individual category files")
        
        # =====================================================================
        # Step 8: Log metrics and summary
        # =====================================================================
        print("\n" + "-" * 70)
        print("Step 8: Log Metrics")
        print("-" * 70)
        
        total_initial = sum(stats["initial_count"] for stats in category_stats.values())
        total_shrunk = sum(stats["shrunk_count"] for stats in category_stats.values())
        total_reduction = total_initial - total_shrunk
        
        log_summary(run, {
            "categories_processed": len(categories_to_process),
            "total_initial_themes": total_initial,
            "total_shrunk_themes": total_shrunk,
            "total_reduction": total_reduction,
            "avg_reduction_per_category": total_reduction / len(categories_to_process) if categories_to_process else 0,
        })
        
        # =====================================================================
        # Step 9: Upload artifact to W&B
        # =====================================================================
        print("\n" + "-" * 70)
        print("Step 9: Upload Artifact to W&B")
        print("-" * 70)
        
        artifact = log_artifact(
            run=run,
            artifact_name=output_artifact_name,
            artifact_type="dataset",
            artifact_path=output_dir,
            metadata={
                "categories_processed": categories_to_process,
                "total_categories": len(topic_universe.topics_by_category),
                "category_stats": category_stats,
                "total_initial_themes": total_initial,
                "total_shrunk_themes": total_shrunk,
                "method": "llm_consolidation",
                "model": model,
                "input_artifact": input_artifact_name,
                "schema_validated": True,
                "schema_version": "v4"
            }
        )
        
        link_to_registry(artifact, stage="03_topic_universe")
        print(f"[OK] Artifact uploaded to W&B: {output_artifact_name}")
        
        # =====================================================================
        # Final Summary
        # =====================================================================
        print("\n" + "=" * 70)
        print("STAGE 03.2 COMPLETE")
        print("=" * 70)
        print(f"\nSummary:")
        print(f"  Categories processed: {len(categories_to_process)}")
        print(f"  Total initial themes: {total_initial}")
        print(f"  Total shrunk themes: {total_shrunk}")
        print(f"  Total reduction: {total_reduction}")
        print(f"  Output: {output_file}")
        
        if run:
            print(f"\nView run at: {run.url}")
        
    finally:
        finish_run(run)


if __name__ == "__main__":
    asyncio.run(main())
