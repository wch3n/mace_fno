#!/usr/bin/env bash
#SBATCH --job-name=synthetic-les-paired
#SBATCH --partition=batch
#SBATCH --qos=normal
#SBATCH --account=htbase
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:15:00
set -Eeuo pipefail
SYNTHETIC_LES_ROOT=${1:?Pass campaign directory}
module --force purge
module load EasyBuild/2024a Python/3.12.3-GCCcore-13.3.0
export LD_LIBRARY_PATH="$EBROOTPYTHON/lib:${LD_LIBRARY_PATH:-}"
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
cd "$SYNTHETIC_LES_ROOT"
exec python3 -B -u collect.py
