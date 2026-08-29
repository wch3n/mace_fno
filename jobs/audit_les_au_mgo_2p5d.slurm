#!/usr/bin/env bash
#SBATCH --job-name=audit-au-mgo-2p5d
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --partition=debug-gpu
#SBATCH --account=htbase
#SBATCH --time=00:20:00
#SBATCH --mem=32G
#SBATCH --output=logs/audit-les-au-mgo-2p5d-%j.out

set -Eeuo pipefail

module --force purge
module load EasyBuild/2024a Python
ulimit -s unlimited

PROJECT_ROOT=/gpfs/home/acad/ucl-modl/wchen/mxene_proj/mace_fno
: "${CHECKPOINT:?Set CHECKPOINT to the trained 2.5D residual checkpoint}"
AUDIT_SAMPLES=${AUDIT_SAMPLES:-32}
AUDIT_OUTPUT=${AUDIT_OUTPUT:-${CHECKPOINT%.pt}_audit.json}

cd "${PROJECT_ROOT}"
mkdir -p logs "$(dirname "${AUDIT_OUTPUT}")"

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="${PROJECT_ROOT}/src"
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export MPLCONFIGDIR="${TMPDIR:-/tmp}/mace-fno-audit-matplotlib-${SLURM_JOB_ID}"
mkdir -p "${MPLCONFIGDIR}"

python3 -u examples/audit_au_mgo_2p5d.py \
    --checkpoint "${CHECKPOINT}" \
    --samples "${AUDIT_SAMPLES}" \
    --fd-components 12 \
    --device cuda \
    --output "${AUDIT_OUTPUT}" \
    --strict
