#!/bin/bash
#SBATCH --job-name=gnnjob
#SBATCH --output=Output/gnnjob_%j.out
#SBATCH --error=Output/gnnjob_%j.err
#SBATCH --partition=compute-p1
#SBATCH --time=06:00:00
#SBATCH --ntasks=5
#SBATCH --cpus-per-task=10
#SBATCH --mem-per-cpu=3G
#SBATCH --account=Education-ME-MSc-TM
#--array=0-10

module load 2023r1
module load miniconda3/4.12.0

source $(conda info --base)/etc/profile.d/conda.sh
conda activate meg_gnn_env

#analysis_names=("Full" "TONIC" "BURST" "Canada" "Nijmegen" "Delta" "Theta" "Theta_Alpha" "Alpha" "Beta" "Gamma")
#export ANALYSIS_NAME=${analysis_names[$SLURM_ARRAY_TASK_ID]}
export ANALYSIS_NAME="Low"

SCOUT_VERSION="Scout_58_MN"

echo "Getting scout data from: $SCOUT_VERSION"

PROCESSED_DIR="/scratch/cwitstok/Data/processed/scout_58_MN"

echo "Starting graph generation for: $ANALYSIS_NAME"

srun python generate_graphs.py --processed_dir "$PROCESSED_DIR" --scout_version "$SCOUT_VERSION"

echo "Finished graph generation for: $ANALYSIS_NAME"