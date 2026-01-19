#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=kutem_gpu
#SBATCH --account=kutem
#SBATCH --gres=gpu:tesla_a100:1
#SBATCH --time=24:00:00
#SBATCH --mem=64G
#SBATCH --output=logs/get_embeddings_%j.out
#SBATCH --error=logs/get_embeddings_%j.err

# Initialise environment and modules
CONDA_BASE=$(conda info --base)
source ${CONDA_BASE}/bin/activate comp
export LD_LIBRARY_PATH=${CONDA_BASE}/lib

python -u -m utils.calculate_embeddings fungal_expression RNA --max-len 1000 --device cuda
python -u -m utils.calculate_embeddings fungal_expression DNA --max-len 1000 --device cuda
python -u -m utils.calculate_embeddings fungal_expression Protein --max-len 1000 --device cuda

python -u -m utils.calculate_embeddings cov_vaccine_degradation RNA --max-len 1000 --device cuda
python -u -m utils.calculate_embeddings cov_vaccine_degradation DNA --max-len 1000 --device cuda
python -u -m utils.calculate_embeddings cov_vaccine_degradation Protein --max-len 1000 --device cuda