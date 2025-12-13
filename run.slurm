#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --partition=cosbi
#SBATCH --account=cosbi
#SBATCH --gres=gpu:tesla_t4:1 
#SBATCH --time=72:00:00
#SBATCH --mem=40G
#SBATCH --output=./logs/comp_%A.log
#SBATCH --error=logs/comp_%j.err

# Initialise environment and modules
CONDA_BASE=$(conda info --base)
source ${CONDA_BASE}/bin/activate comp
export LD_LIBRARY_PATH=${CONDA_BASE}/lib

nvidia-smi

python -u "main.py" --name "uni_rna"