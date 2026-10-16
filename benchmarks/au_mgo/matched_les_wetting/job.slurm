#!/usr/bin/env bash
#SBATCH --job-name=au-les-sv-wetting
#SBATCH --partition=debug-gpu
#SBATCH --qos=debug-gpu
#SBATCH --account=htbase
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --no-requeue
set -Eeuo pipefail
AU_LES_WETTING_ROOT=${1:?Pass campaign directory}
module --force purge
module load EasyBuild/2024a Python/3.12.3-GCCcore-13.3.0 CUDA/12.6.0
export LD_LIBRARY_PATH="$EBROOTPYTHON/lib:${LD_LIBRARY_PATH:-}"
ulimit -s unlimited
export PYTHONPATH="$AU_LES_WETTING_ROOT/source"
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
cd "$AU_LES_WETTING_ROOT"
nvidia-smi
exec python3 -B -u campaign.py run "$AU_LES_WETTING_ROOT"
