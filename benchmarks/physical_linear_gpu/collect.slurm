#!/usr/bin/env bash
#SBATCH --account=htbase
#SBATCH --partition=batch
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:15:00
#SBATCH --no-requeue
set -Eeuo pipefail
LINEAR_COLLECT_ROOT=${1:?campaign directory}
module --force purge
module load EasyBuild/2024a Python
exec python3 -B -u "$LINEAR_COLLECT_ROOT/collect.py"
