#!/usr/bin/env bash
#SBATCH --account=htbase
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --time=2-00:00:00
#SBATCH --no-requeue
set -Eeuo pipefail
LES_MD_ROOT=${1:?campaign root}
LES_MD_MODE=${2:?check or run}
module --force purge
module load EasyBuild/2024a Python/3.12.3-GCCcore-13.3.0 CUDA/12.6.0
export LD_LIBRARY_PATH="$EBROOTPYTHON/lib:${LD_LIBRARY_PATH:-}"
ulimit -s unlimited
export PYTHONPATH="$LES_MD_ROOT/source"
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
cd "$LES_MD_ROOT"
exec python3 -B -u campaign.py "$LES_MD_MODE" "${SLURM_ARRAY_TASK_ID:?array index required}"
