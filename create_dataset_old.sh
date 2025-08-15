#!/bin/bash
#SBATCH --job-name=gnnjob
#SBATCH --output=Output/gnnjob_%j.out
#SBATCH --error=Output/gnnjob_%j.err
#SBATCH --partition=compute-p1
#SBATCH --time=04:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=25
#SBATCH --mem-per-cpu=3G
#SBATCH --account=Education-ME-MSc-TM

module load 2023r1
module load miniconda3/4.12.0

source $(conda info --base)/etc/profile.d/conda.sh
conda activate meg_gnn_env

srun python create_dataset_old.py

