#!/bin/bash
#SBATCH --job-name=gnn_explain
#SBATCH --output=Output/gnn_explain_%A_%a.out
#SBATCH --error=Output/gnn_explain_%A_%a.err
#SBATCH --partition=compute-p1
#SBATCH --time=04:00:00
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=10
#SBATCH --mem-per-cpu=3G
#SBATCH --account=Education-ME-MSc-TM
#SBATCH --array=0-2

module load 2023r1
module load miniconda3/4.12.0

source $(conda info --base)/etc/profile.d/conda.sh
conda activate meg_gnn_env

analysis_names=("Full" "TONIC" "BURST")
ANALYSIS_NAME=${analysis_names[$SLURM_ARRAY_TASK_ID]}

PROCESSED_DIR="/scratch/cwitstok/Data/processed/scout_58_MN/processed_${ANALYSIS_NAME}"
MODEL_NAME="gnn_scout_58_MN_grouped_${ANALYSIS_NAME}"
MODEL_PATH="Output/${MODEL_NAME}/best_model${MODEL_NAME}.pt"
BEST_RESULT_PATH="Output/${MODEL_NAME}/best_result${MODEL_NAME}.pt"
OUTPUT_EXCEL="Output/${MODEL_NAME}/explain_gnn_${ANALYSIS_NAME}.xlsx"

echo "Explaining model for: $ANALYSIS_NAME"

srun python model_explainability.py \
    --processed_dir "$PROCESSED_DIR" \
    --model_path "$MODEL_PATH" \
    --best_result_path "$BEST_RESULT_PATH" \
    --output_excel "$OUTPUT_EXCEL" \
    --files PTN13_TONIC PTN14_BURST PTN14_TONIC PTN15_BURST PTN15_TONIC\
    --input_type "scout" \
    --n_min 6

echo "Finished explanation for: $ANALYSIS_NAME"