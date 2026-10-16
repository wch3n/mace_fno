#!/usr/bin/env bash
#SBATCH --job-name=hs-response-interim-probe
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
EXTLES_PROBE_ROOT=${1:?interim probe directory}
module --force purge
module load EasyBuild/2024a Python/3.12.3-GCCcore-13.3.0 CUDA/12.6.0
export LD_LIBRARY_PATH="$EBROOTPYTHON/lib:${LD_LIBRARY_PATH:-}"
ulimit -s unlimited
unset PYTHONPATH
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
cd "$EXTLES_PROBE_ROOT"
exec "$EXTLES_PROBE_ROOT/venv/bin/python" -B -u -X faulthandler interim_probe.py
