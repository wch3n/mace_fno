#!/usr/bin/env bash
#SBATCH --job-name=ti2co2-audit
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --partition=debug-gpu
#SBATCH --account=htbase
#SBATCH --time=00:30:00
#SBATCH --mem=48G
# Scheduler output is supplied by submit.sh and remains outside the checkout.

set -Eeuo pipefail

module --force purge
module load EasyBuild/2024a Python
ulimit -s unlimited

PROJECT_ROOT=${PROJECT_ROOT:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)}
: "${CHECKPOINT:?Set CHECKPOINT to the completed FNO checkpoint}"
AUDIT_SAMPLES=${AUDIT_SAMPLES:-16}
AUDIT_OUTPUT=${AUDIT_OUTPUT:-${CHECKPOINT%.pt}_audit.json}
JOB_WORK_ROOT=${JOB_WORK_ROOT:-$(dirname "${AUDIT_OUTPUT}")/work}

mkdir -p "$(dirname "${AUDIT_OUTPUT}")" "${JOB_WORK_ROOT}"
test -s "${CHECKPOINT}"
cd "${JOB_WORK_ROOT}"

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="${PROJECT_ROOT}/src"
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export MPLCONFIGDIR="${TMPDIR:-/tmp}/mace-fno-audit-${SLURM_JOB_ID}"
mkdir -p "${MPLCONFIGDIR}"

python3 -u -m mace_fno.cli.audit_2p5d \
    --checkpoint "${CHECKPOINT}" \
    --samples "${AUDIT_SAMPLES}" \
    --fd-step 1.0e-4 \
    --fd-components 12 \
    --device cuda \
    --output "${AUDIT_OUTPUT}" \
    --strict

test -s "${AUDIT_OUTPUT}"
