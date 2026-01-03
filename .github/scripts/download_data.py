#!/usr/bin/env python3
"""Download training data from Kaggle if not present."""

import kagglehub
import shutil
import os
import glob
import sys

def download_training_data():
    """Download and prepare training data."""
    data_dir = "Data"
    training_file = os.path.join(data_dir, "customer_churn_dataset-training-master.csv")
    
    # Check if file already exists
    if os.path.exists(training_file):
        print(f"✓ Training data already exists at {training_file}")
        return True
    
    try:
        # Download dataset from kagglehub
        print("Downloading dataset from Kaggle...")
        path = kagglehub.dataset_download("muhammadshahidazeem/customer-churn-dataset")
        print(f"Dataset downloaded to: {path}")
        
        # Find and copy the training file
        training_files = glob.glob(os.path.join(path, "*training*.csv"))
        if training_files:
            src = training_files[0]
            os.makedirs(data_dir, exist_ok=True)
            shutil.copy(src, training_file)
            print(f"✓ Copied training data to {training_file}")
            return True
        else:
            print("❌ No training file found in downloaded dataset")
            return False
            
    except Exception as e:
        print(f"❌ Failed to download dataset: {e}")
        print("Please add Data/customer_churn_dataset-training-master.csv to your repository")
        return False

if __name__ == "__main__":
    success = download_training_data()
    sys.exit(0 if success else 1)
