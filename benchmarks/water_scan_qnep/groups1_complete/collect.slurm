#!/usr/bin/env bash
#SBATCH --account=htbase
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:10:00
#SBATCH --no-requeue
set -Eeuo pipefail
cd "$1"
exec python3 -B collect.py
