#!/bin/bash

set -e  # Exit on error
set -m  # Job control enabled

# === Config ===
MODEL_PATH_WITH="/home/sn666/dissertation/benchmarking/PATHS/models_with_transcriptomics/brca_paths_0"
# MODEL_PATH_WITH="/home/sn666/dissertation/benchmarking/PATHS/backup/brca_paths_0"
MODEL_PATH_WITHOUT="/home/sn666/dissertation/benchmarking/PATHS/models/brca_paths_0"
CONFIG_PATH="/home/sn666/dissertation/config/train_config/stnet_pancancer_highest_mag.yaml"
TRANSCRIPTOMICS_CHECKPOINT="/auto/archive/tcga/sn666/trained_models/hist_to_transcriptomics/stnet_pancancer_highest_mag/epoch=5-step=17874.ckpt"
WSI_DIR="/auto/archive/tcga/tcga/wsi/brca"
CHOSEN_GENE_LIST="/home/sn666/dissertation/config/visualisation_genes/breast.txt"
TEST_LIST="/home/sn666/dissertation/benchmarking/PATHS/models/brca_paths_0/test_slide_ids.txt"
OUT_PATH="/home/sn666/dissertation/benchmarking/PATHS/visualisations_brca"

MAX_JOBS=2

# === Handle Ctrl+C ===
cleanup() {
    echo "Caught SIGINT. Killing all subprocesses..."
    pkill -P $$ || true  # kill all child processes
    exit 1
}
trap cleanup SIGINT

# === Slide processing function ===
process_slide() {
    base_filename="$1"
    SVS_FILE="$WSI_DIR/${base_filename}.svs"

    echo "[$base_filename] WITH transcriptomics..."
    python heatmap_visualise_transcriptomics.py \
        -m "$MODEL_PATH_WITH" \
        -s "$SVS_FILE" \
        -c "$CONFIG_PATH" \
        -g "$CHOSEN_GENE_LIST" \
        -o "$OUT_PATH"

    echo "[$base_filename] WITHOUT transcriptomics..."
    python heatmap_visualise_transcriptomics.py \
        -m "$MODEL_PATH_WITHOUT" \
        -s "$SVS_FILE" \
        -c "$CONFIG_PATH" \
        -g "$CHOSEN_GENE_LIST" \
        -t "$TRANSCRIPTOMICS_CHECKPOINT" \
        -o "$OUT_PATH"
}

export -f process_slide
export MODEL_PATH_WITH MODEL_PATH_WITHOUT CONFIG_PATH TRANSCRIPTOMICS_CHECKPOINT WSI_DIR CHOSEN_GENE_LIST OUT_PATH

# === Parallel execution with Ctrl+C safety ===
cat "$TEST_LIST" | xargs -n 1 -P "$MAX_JOBS" -I {} bash -c 'process_slide "$@"' _ {}