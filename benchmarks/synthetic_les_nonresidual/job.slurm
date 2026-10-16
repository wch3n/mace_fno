#!/usr/bin/env bash
#SBATCH --job-name=synthetic-les-nonres
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --account=htbase
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --time=48:00:00
#SBATCH --no-requeue
set -Eeuo pipefail
SYNTHETIC_LES_ROOT=${1:?Pass campaign directory}
SYNTHETIC_LES_MODE=${2:-run}
module --force purge
module load EasyBuild/2024a Python/3.12.3-GCCcore-13.3.0 CUDA/12.6.0
export LD_LIBRARY_PATH="$EBROOTPYTHON/lib:${LD_LIBRARY_PATH:-}"
ulimit -s unlimited
export PYTHONPATH="$SYNTHETIC_LES_ROOT/source"
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
cd "$SYNTHETIC_LES_ROOT"
nvidia-smi
exec python3 -B -u campaign.py "$SYNTHETIC_LES_MODE" "${SLURM_ARRAY_TASK_ID:?Array index required}"
