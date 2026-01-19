#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=kutem_gpu
#SBATCH --account=kutem
#SBATCH --gres=gpu:tesla_a100:1
#SBATCH --time=24:00:00
#SBATCH --mem=64G
#SBATCH --output=logs/run_unimodal_%j.out
#SBATCH --error=logs/run_unimodal_%j.err

# Initialise environment and modules
CONDA_BASE=$(conda info --base)
source ${CONDA_BASE}/bin/activate comp
export LD_LIBRARY_PATH=${CONDA_BASE}/lib

nvidia-smi

python -u "run_unimodal.py" --name "uni_rna" --dataset "fungal_expression" --max-len 1000
python -u "run_unimodal.py" --name "uni_dna" --dataset "fungal_expression" --max-len 1000
python -u "run_unimodal.py" --name "uni_protein" --dataset "fungal_expression" --max-len 1000
