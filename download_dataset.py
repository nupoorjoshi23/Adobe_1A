# download_dataset.py
from huggingface_hub import hf_hub_download
import os

# --- Configuration ---
# The name of the dataset repository on Hugging Face
REPO_ID = "ds4sd/DocLayNet"

# The specific file we want to download from that repository
FILENAME_TO_DOWNLOAD = "doclaynet_core.zip"

# Where to save the file locally
# This will save it inside your 'data/raw/doclaynet' folder
LOCAL_SAVE_DIR = os.path.join("data", "raw", "doclaynet")

# --- Main Download Logic ---
def main():
    print(f"Starting download of '{FILENAME_TO_DOWNLOAD}' from '{REPO_ID}'...")
    print(f"This is a large file (~5.6 GB) and may take a long time.")
    print("The download is resumable if it gets interrupted.")

    # Ensure the save directory exists
    os.makedirs(LOCAL_SAVE_DIR, exist_ok=True)

    try:
        # This is the core function that handles the download
        downloaded_file_path = hf_hub_download(
            repo_id=REPO_ID,
            filename=FILENAME_TO_DOWNLOAD,
            repo_type="dataset",  # Specify that we are downloading from a dataset
            local_dir=LOCAL_SAVE_DIR,
            local_dir_use_symlinks=False # Important for Windows
        )

        print("\n✅ Download complete!")
        print(f"File saved to: {downloaded_file_path}")
        print("\nNext steps:")
        print(f"1. Unzip the file '{os.path.basename(downloaded_file_path)}' inside the '{LOCAL_SAVE_DIR}' folder.")
        print(f"2. Run the data preparation script: python src\\prepare_data.py")

    except Exception as e:
        print(f"\n❌ An error occurred during download: {e}")
        print("Please check your internet connection and try again.")

if __name__ == "__main__":
    main()