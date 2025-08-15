#!/bin/bash
#SBATCH --job-name=gnn_plot
#SBATCH --output=Output/gnn_plot_%A_%a.out
#SBATCH --error=Output/gnn_plot_%A_%a.err
#SBATCH --partition=compute-p1
#SBATCH --time=00:10:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=3
#SBATCH --mem-per-cpu=3G
#SBATCH --account=Education-ME-MSc-TM

module load 2023r1
module load miniconda3/4.12.0

source $(conda info --base)/etc/profile.d/conda.sh
conda activate meg_gnn_env

srun python roc_plots.py