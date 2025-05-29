import os
import pandas as pd
import random

# Directories
# wsi_dir = "/auto/archive/tcga/tcga/wsi/ucec"
wsi_dir = "/auto/archive/tcga/tcga/wsi/ucec_zzb20_uni"
split_dir = "/home/sn666/dissertation/benchmarking/PATHS/data/splits/survival/tcga_ucec"

# Ensure the output directory exists
os.makedirs(split_dir, exist_ok=True)

# Load unique slide IDs
full_file_list = os.listdir(wsi_dir)

# only keep slide IDs that have _10 in them if _1, _2, _5 are present for that slide
slide_ids = ["-".join(f.split("-")[:3]) for f in full_file_list]
# remove anything that doesn't contain TCGA
slide_ids = [s for s in slide_ids if "TCGA" in s]
# remove TCGA-B5-A0K2
slide_ids = [s for s in slide_ids if "TCGA-B5-A0K2" not in s]
# remove TCGA-BG-A0M9
slide_ids = [s for s in slide_ids if "TCGA-BG-A0M9" not in s]
# remove TCGA-BS-A0U5
slide_ids = [s for s in slide_ids if "TCGA-BS-A0U5" not in s]
# remove TCGA-BG-A0W2
slide_ids = [s for s in slide_ids if "TCGA-BG-A0W2" not in s]
# remove TCGA-DF-A2KR
slide_ids = [s for s in slide_ids if "TCGA-DF-A2KR" not in s]

unique_slides = sorted(set(slide_ids))

# Shuffle reproducibly
random.seed(42)
random.shuffle(unique_slides)

# Create 5-fold splits
num_folds = 5
folds = []

for i in range(num_folds):
    test_idx = i
    val_idx = (i + 1) % num_folds
    train_idx = [j for j in range(num_folds) if j not in [test_idx, val_idx]]

    fold_size = len(unique_slides) // num_folds
    remainder = len(unique_slides) % num_folds

    # Assign indices accounting for uneven splits
    indices = []
    start = 0
    for f in range(num_folds):
        extra = 1 if f < remainder else 0
        end = start + fold_size + extra
        indices.append(unique_slides[start:end])
        start = end

    test_set = indices[test_idx]
    val_set = indices[val_idx]
    train_set = sum([indices[j] for j in train_idx], [])

    folds.append({
        "train": train_set,
        "val": val_set,
        "test": test_set
    })

# Write to CSV
for i, fold in enumerate(folds):
    train_list = fold["train"]
    val_list = fold["val"]
    max_len = max(len(train_list), len(val_list))

    # pad
    train_list += [""] * (max_len - len(train_list))
    val_list += [""] * (max_len - len(val_list))

    df = pd.DataFrame({
        "train": train_list,
        "val": val_list
    })

    df.to_csv(os.path.join(split_dir, f"splits_{i}.csv"), index_label="")