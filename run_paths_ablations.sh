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
export WANDB_API_KEY=c736d78c6173f0a4bc5b28939e77095015857ad0

cd dissertation/benchmarking/PATHS

LOG_FILE=$(generate_log_file ablations)
echo "Logging to $LOG_FILE"
python train.py -m ablations/leaf_frac_0.2/kirp_paths_2 | tee -a $LOG_FILE

LOG_FILE=$(generate_log_file ablations)
echo "Logging to $LOG_FILE"
python train.py -m ablations/leaf_frac_0.4/kirp_paths_2 | tee -a $LOG_FILE

LOG_FILE=$(generate_log_file ablations)
echo "Logging to $LOG_FILE"
python train.py -m ablations/leaf_frac_0.6/kirp_paths_2 | tee -a $LOG_FILE

LOG_FILE=$(generate_log_file ablations)
echo "Logging to $LOG_FILE"
python train.py -m ablations/leaf_frac_0.8/kirp_paths_2 | tee -a $LOG_FILE

LOG_FILE=$(generate_log_file ablations)
echo "Logging to $LOG_FILE"
python train.py -m ablations/leaf_frac_1.0/kirp_paths_2 | tee -a $LOG_FILE

# LOG_FILE=$(generate_log_file ablations)
# echo "Logging to $LOG_FILE"
# python train.py -m ablations/leaf_frac_0.2/kirp_paths_3 | tee -a $LOG_FILE

# LOG_FILE=$(generate_log_file ablations)
# echo "Logging to $LOG_FILE"
# python train.py -m ablations/leaf_frac_0.4/kirp_paths_3 | tee -a $LOG_FILE

# LOG_FILE=$(generate_log_file ablations)
# echo "Logging to $LOG_FILE"
# python train.py -m ablations/leaf_frac_0.6/kirp_paths_3 | tee -a $LOG_FILE

# LOG_FILE=$(generate_log_file ablations)
# echo "Logging to $LOG_FILE"
# python train.py -m ablations/leaf_frac_0.8/kirp_paths_3 | tee -a $LOG_FILE

# LOG_FILE=$(generate_log_file ablations)
# echo "Logging to $LOG_FILE"
# python train.py -m ablations/leaf_frac_1.0/kirp_paths_3 | tee -a $LOG_FILE

# LOG_FILE=$(generate_log_file ablations)
# echo "Logging to $LOG_FILE"
# python train.py -m ablations/leaf_frac_0.2/kirp_paths_4 | tee -a $LOG_FILE

# LOG_FILE=$(generate_log_file ablations)
# echo "Logging to $LOG_FILE"
# python train.py -m ablations/leaf_frac_0.4/kirp_paths_4 | tee -a $LOG_FILE

# LOG_FILE=$(generate_log_file ablations)
# echo "Logging to $LOG_FILE"
# python train.py -m ablations/leaf_frac_0.6/kirp_paths_4 | tee -a $LOG_FILE

# LOG_FILE=$(generate_log_file ablations)
# echo "Logging to $LOG_FILE"
# python train.py -m ablations/leaf_frac_0.8/kirp_paths_4 | tee -a $LOG_FILE

# LOG_FILE=$(generate_log_file ablations)
# echo "Logging to $LOG_FILE"
# python train.py -m ablations/leaf_frac_1.0/kirp_paths_4 | tee -a $LOG_FILE