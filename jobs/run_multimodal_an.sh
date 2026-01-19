#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=kutem_gpu
#SBATCH --account=kutem
#SBATCH --gres=gpu:tesla_a100:1
#SBATCH --time=24:00:00
#SBATCH --mem=64G
#SBATCH --output=logs/run_multimodal_%j.out
#SBATCH --error=logs/run_multimodal_%j.err

# Initialise environment and modules
CONDA_BASE=$(conda info --base)
source ${CONDA_BASE}/bin/activate comp
export LD_LIBRARY_PATH=${CONDA_BASE}/lib

nvidia-smi
#python -c "import torch; print(torch.cuda.is_available())"

python -u "run_multimodal.py" --name "fusion_concat" --dataset "fungal_expression" --max-len 1000
python -u "run_multimodal.py" --name "fusion_mil" --dataset "fungal_expression" --max-len 1000
python -u "run_multimodal.py" --name "fusion_xattn" --dataset "fungal_expression" --max-len 1000