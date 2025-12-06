import os
from huggingface_hub import snapshot_download
from pathlib import Path

def download_dataset():
    # Get the directory where this script is located
    current_dir = Path(__file__).parent
    destination_folder = current_dir / "fineweb_edu_sample"

    print(f"Target folder: {destination_folder}")

    # We use "HuggingFaceFW/fineweb-edu" (sample-10B)
    # This is a modern, high-quality web text dataset.
    # It is stored in Parquet format.
    # We use the "sample-10BT" subset which is roughly 10 billion tokens.
    # We only download the first shard to keep it tiny (~400MB).
    
    print("Downloading HuggingFaceFW/fineweb-edu (sample-10BT) directly to disk...")
    
    try:
        # snapshot_download downloads files directly to the local_dir.
        # It handles resume (if interrupted) and parallel downloads.
        # allow_patterns ensures we only get the first shard of the 10BT subset.
        snapshot_download(
            repo_id="HuggingFaceFW/fineweb-edu", 
            repo_type="dataset",
            local_dir=destination_folder,
            allow_patterns="sample/10BT/000_00000.parquet"
        )
        
        print(f"Dataset downloaded successfully to {destination_folder}")

    except Exception as e:
        print(f"Error downloading dataset: {e}")

if __name__ == "__main__":
    download_dataset()
