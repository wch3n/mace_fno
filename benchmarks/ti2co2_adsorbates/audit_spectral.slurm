#!/usr/bin/env bash
#SBATCH --job-name=ti2co2-spectrum
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
SPECTRUM_OUTPUT=${SPECTRUM_OUTPUT:-${CHECKPOINT%.pt}_spectral_response.json}
JOB_WORK_ROOT=${JOB_WORK_ROOT:-$(dirname "${SPECTRUM_OUTPUT}")/work}

mkdir -p "$(dirname "${SPECTRUM_OUTPUT}")" "${JOB_WORK_ROOT}"
test -s "${CHECKPOINT}"
cd "${JOB_WORK_ROOT}"

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="${PROJECT_ROOT}/src"
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export MPLCONFIGDIR="${TMPDIR:-/tmp}/mace-fno-spectrum-${SLURM_JOB_ID}"
mkdir -p "${MPLCONFIGDIR}"

python3 -u -m mace_fno.cli.audit_spectral \
    --checkpoint "${CHECKPOINT}" \
    --samples 4 \
    --max-mode 2 \
    --relative-amplitude 0.05 \
    --field-batch-size 32 \
    --device cuda \
    --output "${SPECTRUM_OUTPUT}"

test -s "${SPECTRUM_OUTPUT}"
