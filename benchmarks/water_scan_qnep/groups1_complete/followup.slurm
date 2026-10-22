#!/usr/bin/env bash
#SBATCH --account=htbase
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --time=02:00:00
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
if [ "$2" = measurements ]; then
    export PYTHONPATH="$WATER_CAMPAIGN/source/src:$WATER_CAMPAIGN/followup/native_sources"
    exec python3 -B -u followup/followup.py "$WATER_CAMPAIGN" measurements
fi
exec python3 -B -u followup/followup.py "$WATER_CAMPAIGN" checks --index "$SLURM_ARRAY_TASK_ID"
