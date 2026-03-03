import os
import sys

try:
    from datasets import load_dataset
except ImportError:
    print("Error: 'datasets' package is not installed. Please install it using 'pip install datasets'.")
    sys.exit(1)

# Default to saving dataset locally in the workspace
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "dataset", "QEvasion")
DATASET_NAME = "ailsntua/QEvasion"

def check_and_download():
    print(f"Checking dataset: {DATASET_NAME}...")
    
    if os.path.exists(DATA_DIR) and os.listdir(DATA_DIR):
        print(f"[OK] Dataset is already downloaded at {DATA_DIR}.")
        return

    print(f"Dataset not found at {DATA_DIR}. Downloading...")
    try:
        # Load dataset from Hugging Face
        dataset = load_dataset(DATASET_NAME)
        print("Download complete. Saving to disk...")
        dataset.save_to_disk(DATA_DIR)
        print(f"[SUCCESS] Dataset saved to {DATA_DIR}.")
    except Exception as e:
        print(f"[ERROR] Failed to download dataset: {e}")
        sys.exit(1)

if __name__ == "__main__":
    check_and_download()
