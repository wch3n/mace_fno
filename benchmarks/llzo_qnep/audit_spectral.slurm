#!/usr/bin/env bash
#SBATCH --job-name=llzo-spectrum
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --partition=debug-gpu
#SBATCH --account=htbase
#SBATCH --time=00:30:00
#SBATCH --mem=48G
# Scheduler output is supplied by submit.sh so it stays outside the checkout.

set -Eeuo pipefail

module --force purge
module load EasyBuild/2024a Python
ulimit -s unlimited

PROJECT_ROOT=${PROJECT_ROOT:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)}
CHECKPOINT=${CHECKPOINT:?Set CHECKPOINT to the trained MACE+FNO checkpoint}
SPECTRUM_SAMPLES=${SPECTRUM_SAMPLES:-4}
SPECTRUM_MAX_MODE=${SPECTRUM_MAX_MODE:-2}
SPECTRUM_FIT_SHELLS=${SPECTRUM_FIT_SHELLS:-3}
SPECTRUM_RELATIVE_AMPLITUDE=${SPECTRUM_RELATIVE_AMPLITUDE:-0.05}
SPECTRUM_BATCH_SIZE=${SPECTRUM_BATCH_SIZE:-32}
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

python3 -u -m mace_fno.cli.audit_spectral \
    --checkpoint "${CHECKPOINT}" \
    --samples "${SPECTRUM_SAMPLES}" \
    --max-mode "${SPECTRUM_MAX_MODE}" \
    --fit-shells "${SPECTRUM_FIT_SHELLS}" \
    --relative-amplitude "${SPECTRUM_RELATIVE_AMPLITUDE}" \
    --field-batch-size "${SPECTRUM_BATCH_SIZE}" \
    --device cuda \
    --output "${SPECTRUM_OUTPUT}"

test -s "${SPECTRUM_OUTPUT}"
