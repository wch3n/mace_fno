#!/usr/bin/env bash
#SBATCH --account=htbase
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --time=48:00:00
#SBATCH --no-requeue
set -Eeuo pipefail
LINEAR_GPU_ROOT=${1:?campaign directory}
LINEAR_GPU_MODE=${2:?smoke or train}
module --force purge
module load EasyBuild/2024a Python
ulimit -s unlimited
export PYTHONPATH="${LINEAR_GPU_ROOT}/source/src:${LINEAR_GPU_ROOT}/mace_source"
export PYTHONDONTWRITEBYTECODE=1 TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
cd "$LINEAR_GPU_ROOT"
if [[ "$LINEAR_GPU_MODE" == smoke ]]; then
    exec python3 -B -u smoke.py
fi
exec python3 -B -u run.py train "${SLURM_ARRAY_TASK_ID:?task index}"
