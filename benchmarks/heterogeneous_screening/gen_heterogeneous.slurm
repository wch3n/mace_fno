#!/usr/bin/env bash
#SBATCH --job-name=hetscreen-gen
#SBATCH --partition=shared
#SBATCH --qos=shared
#SBATCH --account=htbase
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=80
#SBATCH --mem=96G
#SBATCH --time=08:00:00
# Generate and label the heterogeneous-screening dataset, e.g.
#   sbatch --output=/path/to/gen_%j.log gen_heterogeneous.slurm OUTDIR
# STAGE=md|label|all (default all). Templates: the kappa = 0.05 2800 K Yukawa MD data.
set -euo pipefail
OUT=${1:?output directory}
STAGE=${STAGE:-all}
TEMPLATES=${TEMPLATES:-${HOME}/mace_fno_runs/yukawa_L32_md2800/data_k0p05}
module --force purge
module load EasyBuild/2024a Python
cd "${SLURM_SUBMIT_DIR}"
python3 -u gen_heterogeneous.py --stage "${STAGE}" --templates "${TEMPLATES}" \
    --out "${OUT}" --nproc "${SLURM_CPUS_PER_TASK}"
