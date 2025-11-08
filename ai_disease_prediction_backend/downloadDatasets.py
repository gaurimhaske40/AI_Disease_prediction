import kagglehub
import os
import shutil

def download_to_custom_path(dataset_name, target_dir):
    # Download using kagglehub
    download_path = kagglehub.dataset_download(dataset_name,force_download=True)

    # Ensure the target directory exists
    os.makedirs(target_dir, exist_ok=True)

    # Move contents from download_path to target_dir
    for item in os.listdir(download_path):
        source_path = os.path.join(download_path, item)
        target_path = os.path.join(target_dir, item)
        
        # Move files and folders
        shutil.move(source_path, target_path)

    print(f"Moved dataset '{dataset_name}' to: {target_dir}")

# Example usage:
download_to_custom_path("uciml/pima-indians-diabetes-database", "./datasets")
download_to_custom_path("mansoordaku/ckdisease", "./datasets")
download_to_custom_path("rishidamarla/heart-disease-prediction", "./datasets")
