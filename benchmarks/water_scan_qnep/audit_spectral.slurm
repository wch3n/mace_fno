#!/usr/bin/env bash
#SBATCH --job-name=water-scan-spectrum
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --partition=debug-gpu
#SBATCH --account=htbase
#SBATCH --time=00:30:00
#SBATCH --mem=48G
#SBATCH --output=logs/water-scan-spectrum-%j.out

set -Eeuo pipefail

module --force purge
module load EasyBuild/2024a Python
ulimit -s unlimited

PROJECT_ROOT=${PROJECT_ROOT:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)}
source "${PROJECT_ROOT}/benchmarks/runtime_paths.sh"
FNO_SEED=${FNO_SEED:-17}
MODEL_DTYPE=${MODEL_DTYPE:-float64}
CHECKPOINT=${CHECKPOINT:-${MACE_FNO_WORK_ROOT}/water_scan_qnep/fno_3d_eqgino/water_scan_fno_3d_seed${FNO_SEED}_${MODEL_DTYPE}.pt}
SPECTRUM_SAMPLES=${SPECTRUM_SAMPLES:-4}
SPECTRUM_MAX_MODE=${SPECTRUM_MAX_MODE:-2}
SPECTRUM_RELATIVE_AMPLITUDE=${SPECTRUM_RELATIVE_AMPLITUDE:-0.05}
SPECTRUM_BATCH_SIZE=${SPECTRUM_BATCH_SIZE:-32}
SPECTRUM_OUTPUT=${SPECTRUM_OUTPUT:-${CHECKPOINT%.pt}_spectral_response.json}

cd "${PROJECT_ROOT}"
mkdir -p logs "$(dirname "${SPECTRUM_OUTPUT}")"
test -s "${CHECKPOINT}"

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="${PROJECT_ROOT}/src"
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export MPLCONFIGDIR="${TMPDIR:-/tmp}/mace-fno-spectrum-${SLURM_JOB_ID}"
mkdir -p "${MPLCONFIGDIR}"

python3 -u -m mace_fno.cli.audit_spectral \
    --checkpoint "${CHECKPOINT}" \
    --samples "${SPECTRUM_SAMPLES}" \
    --max-mode "${SPECTRUM_MAX_MODE}" \
    --relative-amplitude "${SPECTRUM_RELATIVE_AMPLITUDE}" \
    --field-batch-size "${SPECTRUM_BATCH_SIZE}" \
    --device cuda \
    --output "${SPECTRUM_OUTPUT}"
