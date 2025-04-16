path = "/home/sn666/dissertation/benchmarking/PATHS/models_with_transcriptomics/"

# recursively iterate through all files in the directory and delete "model.pt", "train_stats.pkl", and "wandb_id"
import os
import glob
import shutil
import re

def delete_files(path):
    # get all files in the directory
    for root, dirs, files in os.walk(path):
        for file in files:
            # check if the file is "model.pt", "train_stats.pkl", or "wandb_id"
            if file == "model.pt" or file == "train_stats.pkl" or file == "wandb_id":
                # delete the file
                os.remove(os.path.join(root, file))
                print(f"Deleted {os.path.join(root, file)}")

delete_files(path)
