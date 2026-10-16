#!/usr/bin/env bash
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --account=htbase
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --no-requeue
set -Eeuo pipefail
campaign_root=${1:?campaign directory}
mode=${2:?smoke or train}
module --force purge
module load EasyBuild/2024a Python
ulimit -s unlimited
export PYTHONPATH="${campaign_root}/source/src:${campaign_root}/mace_source"
export PYTHONDONTWRITEBYTECODE=1 TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
cd "$campaign_root"
nvidia-smi
exec python3 -B -u run.py "$mode" "${SLURM_ARRAY_TASK_ID:?task index}"
