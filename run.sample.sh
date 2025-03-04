#!/bin/bash
set -m

# Generate a unique log file for each Python script invocation
generate_log_file() {
    local script_name=$(basename "$1" .py)
    local timestamp=$(date "+%Y%m%d-%H%M%S")
    echo "/home/sn666/dissertation/benchmarking/PATHS/logs/${script_name}_${timestamp}.log"
}

# conda activate paths2

export HF_TOKEN=<>
export WANDB_API_KEY=<>

cd dissertation/benchmarking/PATHS

LOG_FILE=$(generate_log_file train)
echo "Logging to $LOG_FILE"
python train.py -m models/kirp_paths_0 | tee -a $LOG_FILE

LOG_FILE=$(generate_log_file train_with_transcriptomics)
echo "Logging to $LOG_FILE"
python train.py -m models_with_transcriptomics/kirp_paths_0 | tee -a $LOG_FILE

LOG_FILE=$(generate_log_file train)
echo "Logging to $LOG_FILE"
python train.py -m models/kirp_paths_1 | tee -a $LOG_FILE

LOG_FILE=$(generate_log_file train_with_transcriptomics)
echo "Logging to $LOG_FILE"
python train.py -m models_with_transcriptomics/kirp_paths_1 | tee -a $LOG_FILE

LOG_FILE=$(generate_log_file train)
echo "Logging to $LOG_FILE"
python train.py -m models/kirp_paths_2 | tee -a $LOG_FILE

LOG_FILE=$(generate_log_file train_with_transcriptomics)
echo "Logging to $LOG_FILE"
python train.py -m models_with_transcriptomics/kirp_paths_2 | tee -a $LOG_FILE

LOG_FILE=$(generate_log_file train)
echo "Logging to $LOG_FILE"
python train.py -m models/kirp_paths_3 | tee -a $LOG_FILE

LOG_FILE=$(generate_log_file train_with_transcriptomics)
echo "Logging to $LOG_FILE"
python train.py -m models_with_transcriptomics/kirp_paths_3 | tee -a $LOG_FILE

LOG_FILE=$(generate_log_file train)
echo "Logging to $LOG_FILE"
python train.py -m models/kirp_paths_4 | tee -a $LOG_FILE

LOG_FILE=$(generate_log_file train_with_transcriptomics)
echo "Logging to $LOG_FILE"
python train.py -m models_with_transcriptomics/kirp_paths_4 | tee -a $LOG_FILE

# LOG_FILE=$(generate_log_file train)
# echo "Logging to $LOG_FILE"
# python train.py -m models/brca_paths_1 | tee -a $LOG_FILE

# LOG_FILE=$(generate_log_file train_with_transcriptomics)
# echo "Logging to $LOG_FILE"
# python train.py -m models_with_transcriptomics/brca_paths_1 | tee -a $LOG_FILE

# LOG_FILE=$(generate_log_file train)
# echo "Logging to $LOG_FILE"
# python train.py -m models/coadread_paths_1 | tee -a $LOG_FILE

# LOG_FILE=$(generate_log_file train_with_transcriptomics)
# echo "Logging to $LOG_FILE"
# python train.py -m models_with_transcriptomics/coadread_paths_1 | tee -a $LOG_FILE

# LOG_FILE=$(generate_log_file train)
# echo "Logging to $LOG_FILE"
# python train.py -m models/kirc_paths_0 | tee -a $LOG_FILE

# LOG_FILE=$(generate_log_file train_with_transcriptomics)
# echo "Logging to $LOG_FILE"
# python train.py -m models_with_transcriptomics/kirc_paths_0 | tee -a $LOG_FILE

# LOG_FILE=$(generate_log_file train)
# echo "Logging to $LOG_FILE"
# python train.py -m models/kirc_paths_2 | tee -a $LOG_FILE

# LOG_FILE=$(generate_log_file train_with_transcriptomics)
# echo "Logging to $LOG_FILE"
# python train.py -m models_with_transcriptomics/kirc_paths_2 | tee -a $LOG_FILE

# LOG_FILE=$(generate_log_file train)
# echo "Logging to $LOG_FILE"
# python train.py -m models/kirc_paths_3 | tee -a $LOG_FILE

# LOG_FILE=$(generate_log_file train_with_transcriptomics)
# echo "Logging to $LOG_FILE"
# python train.py -m models_with_transcriptomics/kirc_paths_3 | tee -a $LOG_FILE

# LOG_FILE=$(generate_log_file train)
# echo "Logging to $LOG_FILE"
# python train.py -m models/kirc_paths_4 | tee -a $LOG_FILE

# LOG_FILE=$(generate_log_file train_with_transcriptomics)
# echo "Logging to $LOG_FILE"
# python train.py -m models_with_transcriptomics/kirc_paths_4 | tee -a $LOG_FILE

# LOG_FILE=$(generate_log_file train)
# echo "Logging to $LOG_FILE"
# python train.py -m models/coadread_paths_0 | tee -a $LOG_FILE

# LOG_FILE=$(generate_log_file train_with_transcriptomics)
# echo "Logging to $LOG_FILE"
# python train.py -m models_with_transcriptomics/coadread_paths_0 | tee -a $LOG_FILE

# LOG_FILE=$(generate_log_file train)
# echo "Logging to $LOG_FILE"
# python train.py -m models/coadread_paths_2 | tee -a $LOG_FILE

# LOG_FILE=$(generate_log_file train_with_transcriptomics)
# echo "Logging to $LOG_FILE"
# python train.py -m models_with_transcriptomics/coadread_paths_2 | tee -a $LOG_FILE

# LOG_FILE=$(generate_log_file train)
# echo "Logging to $LOG_FILE"
# python train.py -m models/coadread_paths_3 | tee -a $LOG_FILE

# LOG_FILE=$(generate_log_file train_with_transcriptomics)
# echo "Logging to $LOG_FILE"
# python train.py -m models_with_transcriptomics/coadread_paths_3 | tee -a $LOG_FILE

# LOG_FILE=$(generate_log_file train)
# echo "Logging to $LOG_FILE"
# python train.py -m models/coadread_paths_4 | tee -a $LOG_FILE

# LOG_FILE=$(generate_log_file train_with_transcriptomics)
# echo "Logging to $LOG_FILE"
# python train.py -m models_with_transcriptomics/coadread_paths_4 | tee -a $LOG_FILE

# LOG_FILE=$(generate_log_file train)
# echo "Logging to $LOG_FILE"
# python train.py -m models/luad_paths_1 | tee -a $LOG_FILE

# LOG_FILE=$(generate_log_file train_with_transcriptomics)
# echo "Logging to $LOG_FILE"
# python train.py -m models_with_transcriptomics/luad_paths_1 | tee -a $LOG_FILE