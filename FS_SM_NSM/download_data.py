import os
import zipfile

def download_dataset():
    """
    Downloads and extracts the forest-fire-smoke-and-non-fire-image-dataset from Kaggle.
    Requires kaggle API to be installed and configured with kaggle.json.
    """
    dataset_slug = "amerzishminha/forest-fire-smoke-and-non-fire-image-dataset"
    zip_name = "forest-fire-smoke-and-non-fire-image-dataset.zip"
    
    print(f"[INFO] Downloading dataset: {dataset_slug}...")
    
    # Try to use the kaggle API
    try:
        import kaggle
        kaggle.api.authenticate()
        kaggle.api.dataset_download_files(dataset_slug, path=".", unzip=True)
        print("[INFO] Dataset downloaded and extracted successfully. ✅")
    except ImportError:
        print("[ERROR] Kaggle API not found. Please install it with: pip install kaggle")
    except Exception as e:
        print(f"[ERROR] Failed to download dataset: {e}")
        print("\nManual download link: https://www.kaggle.com/datasets/amerzishminha/forest-fire-smoke-and-non-fire-image-dataset")

if __name__ == "__main__":
    download_dataset()
