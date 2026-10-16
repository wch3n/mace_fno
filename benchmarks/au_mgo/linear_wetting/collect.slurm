#!/usr/bin/env bash
#SBATCH --account=htbase
#SBATCH --partition=batch
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:15:00
set -Eeuo pipefail
LINEAR_WET_ROOT=${1:?campaign root}
module --force purge
module load EasyBuild/2024a Python
export PYTHONDONTWRITEBYTECODE=1
exec python3 -B "$LINEAR_WET_ROOT/collect.py" "$LINEAR_WET_ROOT"
