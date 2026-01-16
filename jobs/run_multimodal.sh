#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=ai
#SBATCH --account=ai
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --mem=128G
#SBATCH --output=logs/run_multimodal_%j.out
#SBATCH --error=logs/run_multimodal_%j.err

# Initialise environment and modules
CONDA_BASE=$(conda info --base)
source ${CONDA_BASE}/bin/activate comp541
export LD_LIBRARY_PATH=${CONDA_BASE}/lib

nvidia-smi
#python -c "import torch; print(torch.cuda.is_available())"

python -u "run_multimodal.py" --name "fusion_concat" --dataset "fungal_expression" --max-len 1000
python -u "run_multimodal.py" --name "fusion_mil" --dataset "fungal_expression" --max-len 1000
python -u "run_multimodal.py" --name "fusion_xattn" --dataset "fungal_expression" --max-len 1000