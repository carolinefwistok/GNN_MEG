#!/bin/bash
#SBATCH --job-name=test_gnn
#SBATCH --output=Output/test_gnn_%j.out
#SBATCH --error=Output/test_gnn_%j.err
#SBATCH --partition=compute-p1
#SBATCH --time=00:20:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=1G
#SBATCH --account=Education-ME-MSc-TM

module load 2023r1
module load miniconda3/4.12.0

source $(conda info --base)/etc/profile.d/conda.sh
conda activate meg_gnn_env

SAVED_MODEL_NAME="TONIC"
CROSS_DATASET_NAME="BURST"

echo "Testing model for $SAVED_MODEL_NAME on dataset $CROSS_DATASET_NAME"

# Configuring paths
MODEL_NAME="gnn_scout_58_MN_grouped_${SAVED_MODEL_NAME}"
TEST_DATASET_DIR="/scratch/cwitstok/Data/processed/scout_58_MN/processed_${CROSS_DATASET_NAME}/"
OUTPUT_DIR="/scratch/cwitstok/Output/gnn_scout_58_MN_grouped_${SAVED_MODEL_NAME}/cross_performance_${CROSS_DATASET_NAME}/"

srun python test_saved_model.py \
    --model_name "$MODEL_NAME" \
    --test_dataset_dir "$TEST_DATASET_DIR" \
    --output_dir "$OUTPUT_DIR"