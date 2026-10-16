#!/usr/bin/env bash
#SBATCH --account=htbase
#SBATCH --partition=debug-gpu
#SBATCH --qos=debug-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --no-requeue
set -Eeuo pipefail
LINEAR_WET_ROOT=${1:?campaign root}
LINEAR_WET_MODE=${2:?ready or worker}
module --force purge
module load EasyBuild/2024a Python
ulimit -s unlimited
export PYTHONPATH="$LINEAR_WET_ROOT/source/src:$LINEAR_WET_ROOT/mace_source"
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
cd "$LINEAR_WET_ROOT"
if [[ "$LINEAR_WET_MODE" == ready ]]; then
    exec python3 -B -u campaign.py ready "$LINEAR_WET_ROOT"
fi
exec python3 -B -u campaign.py worker "$LINEAR_WET_ROOT" --index "${3:?task index}"
