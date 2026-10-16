#!/usr/bin/env bash
#SBATCH --job-name=les-nonres-fig4
#SBATCH --account=htbase
#SBATCH --partition=batch
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
set -Eeuo pipefail
LES_MD_ROOT=${1:?campaign root}
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
cd "$LES_MD_ROOT"
exec /gpfs/home/acad/ucl-modl/wchen/.local/bin/python3 -B -u update_figure.py
