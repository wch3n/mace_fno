#!/usr/bin/env bash
#SBATCH --account=htbase
#SBATCH --partition=debug-gpu
#SBATCH --qos=debug-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:4
#SBATCH --mem=160G
#SBATCH --time=01:00:00
#SBATCH --no-requeue
set -Eeuo pipefail
AU_EVAL_ROOT=$1
module --force purge
module load EasyBuild/2024a Python
ulimit -s unlimited
export PYTHONPATH="$AU_EVAL_ROOT/source/src:$AU_EVAL_ROOT/mace_source"
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
cd "$AU_EVAL_ROOT"
exec python3 -B -u run.py
