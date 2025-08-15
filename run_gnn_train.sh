#!/bin/bash
#SBATCH --job-name=train_gnn
#SBATCH --output=Output/train_gnn_%A_%a.out
#SBATCH --error=Output/train_gnn_%A_%a.err
#SBATCH --partition=compute-p1
#SBATCH --time=03:00:00
#SBATCH --ntasks=5
#SBATCH --cpus-per-task=10
#SBATCH --mem-per-cpu=3G
#SBATCH --account=Education-ME-MSc-TM
#--array=0-10

module load 2023r1
module load miniconda3/4.12.0

source $(conda info --base)/etc/profile.d/conda.sh
conda activate meg_gnn_env

#analysis_names=("Full" "TONIC" "BURST" "Canada" "Nijmegen" "Delta" "Theta" "Alpha" "Beta" "Gamma")
#ANALYSIS_NAME=${analysis_names[$SLURM_ARRAY_TASK_ID]}
ANALYSIS_NAME="Beta"

echo "Starting training for: $ANALYSIS_NAME"

PROCESSED_DIR="/scratch/cwitstok/Data/processed/scout_58_MN/processed_${ANALYSIS_NAME}"
MODEL_NAME="gnn_scout_58_MN_grouped_${ANALYSIS_NAME}"

echo "Running Python script for: $ANALYSIS_NAME"

srun python train_from_processed.py \
    --processed_dir "$PROCESSED_DIR" \
    --model_name "$MODEL_NAME" \
    --input_type "scout"

echo "Finished training for: $ANALYSIS_NAME"