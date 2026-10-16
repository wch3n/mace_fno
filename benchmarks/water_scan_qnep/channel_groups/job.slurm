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
CAMPAIGN_ROOT=${1:?Pass the campaign directory}
module --force purge
module load EasyBuild/2024a Python
ulimit -s unlimited
export PYTHONPATH="${CAMPAIGN_ROOT}/source/src"
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
cd "$CAMPAIGN_ROOT"
nvidia-smi
if [ "${2:-train}" = smoke ]; then
    exec python3 -B -u smoke.py
fi
python3 -B -u run.py 0
exec python3 -B -u collect.py
