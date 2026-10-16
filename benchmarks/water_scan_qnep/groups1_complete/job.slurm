#!/usr/bin/env bash
#SBATCH --account=htbase
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --time=48:00:00
#SBATCH --no-requeue
set -Eeuo pipefail
WATER_CAMPAIGN=$1
module --force purge
module load EasyBuild/2024a Python
ulimit -s unlimited
export PYTHONPATH="$WATER_CAMPAIGN/source/src"
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
cd "$WATER_CAMPAIGN"
if [ "$2" = smoke ]; then
    exec python3 -B -u smoke.py
fi
exec python3 -B -u run.py "$SLURM_ARRAY_TASK_ID"
