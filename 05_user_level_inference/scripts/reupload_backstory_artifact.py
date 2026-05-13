#!/usr/bin/env python3
"""
Re-upload the user backstory artifact to W&B with the correct file.
This fixes the issue where the artifact was uploaded empty.
"""

import sys
from pathlib import Path

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from utils.wandb_utils import (
    get_stage_config, init_wandb_run, finish_run,
    log_artifact, link_to_registry, get_artifact_dir
)

def main():
    """Re-upload the backstory artifact."""
    print("=" * 70)
    print("Re-uploading User Backstory Artifact to W&B")
    print("=" * 70)
    
    cfg = get_stage_config("05_user_level_inference")
    output_artifact_name = cfg.get("output_artifact", "user_backstory_sampled_500users_v4")
    
    print(f"\n[Config] Artifact name: {output_artifact_name}")
    
    # Initialize W&B run
    run = init_wandb_run(
        run_name=f"reupload_backstory_{output_artifact_name}",
        stage="05_user_level_inference",
        job_type="artifact_reupload"
    )
    
    try:
        # Get the artifact directory
        output_dir = get_artifact_dir("05_user_level_inference", output_artifact_name)
        output_file = output_dir / "user_overall_characteristics.json"
        
        if not output_file.exists():
            print(f"[ERROR] File not found: {output_file}")
            return
        
        file_size = output_file.stat().st_size
        print(f"\n[OK] Found file: {output_file}")
        print(f"  Size: {file_size:,} bytes ({file_size/1024:.1f} KB)")
        
        # Verify directory has the file
        files_in_dir = list(output_dir.glob("*"))
        print(f"\n[INFO] Files in directory: {[f.name for f in files_in_dir]}")
        
        # Re-upload artifact
        print("\n" + "-" * 70)
        print("Uploading Artifact to W&B")
        print("-" * 70)
        
        artifact = log_artifact(
            run=run,
            artifact_name=output_artifact_name,
            artifact_type="dataset",
            artifact_path=output_dir,
            metadata={
                "reuploaded": True,
                "file_size_bytes": file_size,
                "note": "Re-uploaded to fix empty artifact issue"
            }
        )
        
        if artifact:
            link_to_registry(artifact, stage="05_user_level_inference")
            print(f"\n[OK] Artifact re-uploaded successfully!")
            print(f"  Artifact: {artifact.name}")
            print(f"  View at: {run.url if run else 'N/A'}")
        else:
            print("\n[ERROR] Failed to upload artifact")
    
    finally:
        finish_run(run)

if __name__ == "__main__":
    main()



