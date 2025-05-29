path = "/auto/archive/tcga/sn666/trained_models/PATHS/ablations/enrichment_strategy/kirc"

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
            if file == "model.pt" or file == "train_stats.pkl" or file == "wandb_id" or file == "test_gt_hazards.pkl" or file == "test_output.pkl" or file == "test_slide_ids.txt":
                # delete the file
                os.remove(os.path.join(root, file))
                print(f"Deleted {os.path.join(root, file)}")

def rename_folders(path, before="kirp", after="luad"):
    # recurisvely iterate through all folders in the directory
    # if any folder starts with kirp, change it luad
    for root, dirs, files in os.walk(path):
        for dir in dirs:
            if dir.startswith(before):
                new_dir = dir.replace(before, after)
                os.rename(os.path.join(root, dir), os.path.join(root, new_dir))
                print(f"Renamed {os.path.join(root, dir)} to {os.path.join(root, new_dir)}")

delete_files(path)
# rename_folders(path, before="kirp", after="kirc")
