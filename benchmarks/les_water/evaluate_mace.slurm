#!/usr/bin/env bash
#SBATCH --job-name=les-water-mace-eval
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --partition=debug-gpu
#SBATCH --account=htbase
#SBATCH --time=00:30:00
#SBATCH --mem=48G
# Scheduler output is supplied by the launcher and remains outside the checkout.

set -Eeuo pipefail

module --force purge
module load EasyBuild/2024a Python
ulimit -s unlimited

PROJECT_ROOT=${PROJECT_ROOT:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)}
source "${PROJECT_ROOT}/benchmarks/runtime_paths.sh"
SOURCE_ROOT=${SOURCE_ROOT:-${MACE_FNO_WORK_ROOT}/les_water/source}
TRAIN_FILE=${TRAIN_FILE:-${SOURCE_ROOT}/data/train-H2O_RPBE-D3.xyz}
TEST_FILE=${TEST_FILE:-${SOURCE_ROOT}/data/test-H2O_RPBE-D3.xyz}
MACE_MODEL=${MACE_MODEL:-${SOURCE_ROOT}/pretrained/H20_stagetwo.model}
RESULT_PATH=${RESULT_PATH:?Set RESULT_PATH for the JSON report}
JOB_WORK_ROOT=${JOB_WORK_ROOT:-$(dirname "${RESULT_PATH}")/work}

mkdir -p "$(dirname "${RESULT_PATH}")" "${JOB_WORK_ROOT}"
for required in "${TRAIN_FILE}" "${TEST_FILE}" "${MACE_MODEL}"; do
    test -s "${required}"
done
cd "${JOB_WORK_ROOT}"

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="${PROJECT_ROOT}/src"
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export MPLCONFIGDIR="${TMPDIR:-/tmp}/mace-fno-matplotlib-${SLURM_JOB_ID}"
mkdir -p "${MPLCONFIGDIR}"

python3 -u -m mace_fno.cli.evaluate_mace \
    --model "${MACE_MODEL}" \
    --train-file "${TRAIN_FILE}" \
    --test-file "${TEST_FILE}" \
    --energy-key energy \
    --forces-key forces \
    --device cuda \
    --dtype float64 \
    --output "${RESULT_PATH}"

test -s "${RESULT_PATH}"
