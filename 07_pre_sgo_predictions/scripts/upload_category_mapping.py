import sys
from pathlib import Path

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from utils.wandb_utils import init_wandb_run, finish_run, log_artifact, link_to_registry

def main():
    project_root = Path(__file__).parent.parent.parent
    mapping_file = project_root / "category_mapping_to_7_main.json"
    
    if not mapping_file.exists():
        print(f"❌ File not found: {mapping_file}")
        return
    
    print(f"Uploading category mapping to W&B...")
    print(f"File: {mapping_file}")
    
    # Initialize W&B run
    run = init_wandb_run(
        run_name="upload_category_mapping",
        stage="07_sgo_training",
        config={
            "description": "Upload category mapping to 7 main categories",
            "artifact_name": "category_mapping_to_7_main"
        }
    )
    
    try:
        # Create a temporary directory with the file
        import tempfile
        import shutil
        import json
        
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Read the FULL mapping file (keep entire structure)
            with open(mapping_file, 'r', encoding='utf-8') as f:
                full_mapping_data = json.load(f)
            
            # Save the FULL structure as category_mapping.json
            output_file = temp_path / "category_mapping.json"
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(full_mapping_data, f, indent=2, ensure_ascii=False)
            
            category_mapping = full_mapping_data.get("category_to_main_mapping", {})
            main_categories = full_mapping_data.get("main_categories", {})
            print(f"  Prepared full mapping:")
            print(f"    - {len(category_mapping)} category mappings")
            print(f"    - {len(main_categories)} main categories: {list(main_categories.keys())}")
            
            # Upload artifact
            artifact = log_artifact(
                run=run,
                artifact_name="category_mapping_to_7_main",
                artifact_type="dataset",
                artifact_path=temp_path,
                metadata={
                    "description": "Full category mapping to 7 main categories (includes main_categories structure)",
                    "source_file": str(mapping_file),
                    "num_category_mappings": len(category_mapping),
                    "num_main_categories": len(main_categories),
                    "main_categories": list(main_categories.keys()),
                    "schema_version": "v4"
                }
            )
            
            if artifact:
                link_to_registry(artifact, stage="07_sgo_training")
                print(f"✅ Successfully uploaded category_mapping_to_7_main to W&B")
                print(f"   View at: {run.url}")
            else:
                print(f"❌ Failed to upload artifact")
    
    except Exception as e:
        print(f"❌ Error uploading artifact: {e}")
        raise
    
    finally:
        finish_run(run)

if __name__ == "__main__":
    main()

