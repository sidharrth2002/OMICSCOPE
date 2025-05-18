#!/bin/bash

MODEL_PATH_WITH="/home/sn666/dissertation/benchmarking/PATHS/models_with_transcriptomics/luad_paths_3"
MODEL_PATH_WITHOUT="/home/sn666/dissertation/benchmarking/PATHS/models/luad_paths_3"
CONFIG_PATH="/home/sn666/dissertation/config/train_config/stnet_pancancer_highest_mag.yaml"
TRANSCRIPTOMICS_CHECKPOINT="/auto/archive/tcga/sn666/trained_models/hist_to_transcriptomics/stnet_pancancer_highest_mag/epoch=5-step=17874.ckpt"
WSI_DIR="/auto/archive/tcga/tcga/wsi/luad"
CHOSEN_GENE_LIST="/home/sn666/dissertation/config/visualisation_genes/lung.txt"

MAX_JOBS=2
CURRENT_JOBS=0

process_slide() {
    base_filename="$1"
    SVS_FILE="$WSI_DIR/${base_filename}.svs"

    echo "Processing $base_filename WITH transcriptomics..."
    python heatmap_visualise_transcriptomics.py \
        -m "$MODEL_PATH_WITH" \
        -s "$SVS_FILE" \
        -c "$CONFIG_PATH" \
        -g "$CHOSEN_GENE_LIST"

    echo "Processing $base_filename WITHOUT transcriptomics..."
    python heatmap_visualise_transcriptomics.py \
        -m "$MODEL_PATH_WITHOUT" \
        -s "$SVS_FILE" \
        -c "$CONFIG_PATH" \
        -g "$CHOSEN_GENE_LIST" \
        -t "$TRANSCRIPTOMICS_CHECKPOINT"
}

export -f process_slide
export MODEL_PATH_WITH MODEL_PATH_WITHOUT CONFIG_PATH TRANSCRIPTOMICS_CHECKPOINT WSI_DIR CHOSEN_GENE_LIST

cat /home/sn666/dissertation/benchmarking/PATHS/models/luad_paths_3/test_slide_ids.txt | xargs -n 1 -P "$MAX_JOBS" -I {} bash -c 'process_slide "$@"' _ {}