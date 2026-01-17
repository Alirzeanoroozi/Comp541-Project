#!/bin/bash
#SBATCH --job-name=get_embeddings
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=12:00:00
#SBATCH --partition=ai
#SBATCH --account=ai
#SBATCH --qos=ai
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=logs/get_embeddings_%j.out
#SBATCH --error=logs/get_embeddings_%j.err

set -euo pipefail

export NATSORT_LOCALE_LIBRARY=stdlib
# run from the directory where sbatch was called
cd "${SLURM_SUBMIT_DIR}" || exit 1

set +u
CONDA_BASE="$(conda info --base)"
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate comp541
set -u

export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"
mkdir -p data/embeddings logs

python -u -m utils.calculate_embeddings fungal_expression RNA --max-len 1000 --device cuda
python -u -m utils.calculate_embeddings fungal_expression DNA --max-len 1000 --device cuda
python -u -m utils.calculate_embeddings fungal_expression Protein --max-len 1000 --device cuda

python -u -m utils.calculate_embeddings cov_vaccine_degradation RNA --max-len 1000 --device cuda
python -u -m utils.calculate_embeddings cov_vaccine_degradation DNA --max-len 1000 --device cuda
python -u -m utils.calculate_embeddings cov_vaccine_degradation Protein --max-len 1000 --device cuda