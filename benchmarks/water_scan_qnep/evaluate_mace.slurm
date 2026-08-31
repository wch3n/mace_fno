#!/usr/bin/env bash
#SBATCH --job-name=water-scan-eval
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --partition=debug-gpu
#SBATCH --account=htbase
#SBATCH --time=02:00:00
#SBATCH --mem=48G
# Scheduler output is supplied by submit.sh so it stays outside the checkout.

set -Eeuo pipefail

module --force purge
module load EasyBuild/2024a Python
ulimit -s unlimited

PROJECT_ROOT=${PROJECT_ROOT:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)}
source "${PROJECT_ROOT}/benchmarks/runtime_paths.sh"
DATA_ROOT=${DATA_ROOT:-${PROJECT_ROOT}/data/water_scan_qnep/prepared}
MACE_MODEL=${MACE_MODEL:-${MACE_FNO_WORK_ROOT}/water_scan_qnep/mace_r4p5_nl0_e100_s2e50/models/water-SCAN-r4p5-nl0_stagetwo.model}
RESULT_PATH=${RESULT_PATH:-${MACE_FNO_WORK_ROOT}/water_scan_qnep/mace_r4p5_nl0_e100_s2e50/baseline.json}
JOB_WORK_ROOT=${JOB_WORK_ROOT:-$(dirname "${RESULT_PATH}")/work}

mkdir -p "$(dirname "${RESULT_PATH}")" "${JOB_WORK_ROOT}"
test -s "${MACE_MODEL}"
cd "${JOB_WORK_ROOT}"

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="${PROJECT_ROOT}/src"
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export MPLCONFIGDIR="${TMPDIR:-/tmp}/mace-fno-matplotlib-${SLURM_JOB_ID}"
mkdir -p "${MPLCONFIGDIR}"

# The official Zenodo validation set is the held-out benchmark test set here.
python3 -u -m mace_fno.cli.evaluate_mace \
    --model "${MACE_MODEL}" \
    --train-file "${DATA_ROOT}/train.xyz" \
    --test-file "${DATA_ROOT}/test.xyz" \
    --energy-key energy \
    --forces-key forces \
    --device cuda \
    --dtype float64 \
    --output "${RESULT_PATH}"

test -s "${RESULT_PATH}"
