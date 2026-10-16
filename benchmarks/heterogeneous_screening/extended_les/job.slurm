#!/usr/bin/env bash
#SBATCH --account=htbase
#SBATCH --partition=debug-gpu
#SBATCH --qos=debug-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --no-requeue
set -Eeuo pipefail
EXTLES_ROOT=${1:?campaign directory}
EXTLES_MODE=${2:?smoke, train or collect}
module --force purge
module load EasyBuild/2024a Python/3.12.3-GCCcore-13.3.0 CUDA/12.6.0
# The default Python module differs between login and compute nodes. Match the
# venv's interpreter and shared library explicitly, ahead of ~/.local/lib.
export LD_LIBRARY_PATH="$EBROOTPYTHON/lib:${LD_LIBRARY_PATH:-}"
ulimit -s unlimited
unset PYTHONPATH
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
cd "$EXTLES_ROOT"
if [[ "$EXTLES_MODE" == collect ]]; then
    exec "$EXTLES_ROOT/venv/bin/python" -B -u collect.py
fi
exec "$EXTLES_ROOT/venv/bin/python" -B -u -X faulthandler run.py "$EXTLES_MODE" "${SLURM_ARRAY_TASK_ID:-0}"
