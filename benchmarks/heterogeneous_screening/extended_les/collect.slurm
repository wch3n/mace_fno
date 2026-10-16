#!/usr/bin/env bash
#SBATCH --account=htbase
#SBATCH --partition=batch
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:20:00
#SBATCH --no-requeue
set -Eeuo pipefail
EXTLES_ROOT=${1:?campaign directory}
module --force purge
module load EasyBuild/2024a Python/3.12.3-GCCcore-13.3.0
export LD_LIBRARY_PATH="$EBROOTPYTHON/lib:${LD_LIBRARY_PATH:-}"
unset PYTHONPATH
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1
cd "$EXTLES_ROOT"
exec "$EXTLES_ROOT/venv/bin/python" -B -u collect.py
