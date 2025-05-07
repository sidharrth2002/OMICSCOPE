#!/bin/bash
set -m

# filepath: /home/sn666/dissertation/benchmarking/PATHS/run_paths.sh

# Generate a unique log file for each Python script invocation
generate_log_file() {
    local script_name=$(basename "$1" .py)
    local timestamp=$(date "+%Y%m%d-%H%M%S")
    echo "/home/sn666/dissertation/benchmarking/PATHS/logs/${script_name}_${timestamp}.log"
}

conda activate paths2

export HF_TOKEN=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
export WANDB_API_KEY=946cc78a61a4fa0593d2f22c5a24854cc5713201

cd dissertation/benchmarking/PATHS

# =================================================================================================
# KIRC
# =================================================================================================

LOG_FILE=$(generate_log_file train)
echo "Logging to $LOG_FILE"
python train.py -m models/kirc_paths_0 | tee -a $LOG_FILE

LOG_FILE=$(generate_log_file train_with_transcriptomics)
echo "Logging to $LOG_FILE"
python train.py -m models_with_transcriptomics/kirc_paths_0 | tee -a $LOG_FILE

LOG_FILE=$(generate_log_file train)
echo "Logging to $LOG_FILE"
python train.py -m models/kirc_paths_1 | tee -a $LOG_FILE

LOG_FILE=$(generate_log_file train_with_transcriptomics)
echo "Logging to $LOG_FILE"
python train.py -m models_with_transcriptomics/kirc_paths_1 | tee -a $LOG_FILE

LOG_FILE=$(generate_log_file train)
echo "Logging to $LOG_FILE"
python train.py -m models/kirc_paths_2 | tee -a $LOG_FILE

LOG_FILE=$(generate_log_file train_with_transcriptomics)
echo "Logging to $LOG_FILE"
python train.py -m models_with_transcriptomics/kirc_paths_2 | tee -a $LOG_FILE

LOG_FILE=$(generate_log_file train)
echo "Logging to $LOG_FILE"
python train.py -m models/kirc_paths_3 | tee -a $LOG_FILE

LOG_FILE=$(generate_log_file train_with_transcriptomics)
echo "Logging to $LOG_FILE"
python train.py -m models_with_transcriptomics/kirc_paths_3 | tee -a $LOG_FILE

LOG_FILE=$(generate_log_file train)
echo "Logging to $LOG_FILE"
python train.py -m models/kirc_paths_4 | tee -a $LOG_FILE

LOG_FILE=$(generate_log_file train_with_transcriptomics)
echo "Logging to $LOG_FILE"
python train.py -m models_with_transcriptomics/kirc_paths_4 | tee -a $LOG_FILE
