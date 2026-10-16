#!/usr/bin/env bash
#SBATCH --job-name=hetscreen-probe
#SBATCH --partition=shared
#SBATCH --qos=shared
#SBATCH --account=htbase
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=40
#SBATCH --mem=64G
#SBATCH --time=03:00:00
# Probe trial of the heterogeneous-screening benchmark (protocol Sec. 11a).
# Submit with --output outside the checkout, e.g.
#   sbatch --output=/path/to/probe_%j.log probe_trial.slurm OUTDIR
set -euo pipefail
OUT=${1:?output directory}
module --force purge
module load EasyBuild/2024a Python
cd "${SLURM_SUBMIT_DIR}"
python3 -u probe_trial.py \
    --templates "${HOME}/mace_fno_runs/yukawa_L32_md2800/data_k0p05/kappa0p05_train.xyz" \
    --frames 40 --probe-frames 2 --pairs 2 --nproc "${SLURM_CPUS_PER_TASK}" \
    --out "${OUT}/probe_trial.json"
