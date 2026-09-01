#!/usr/bin/env bash
#SBATCH --job-name=ti2co2-eval
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --partition=debug-gpu
#SBATCH --account=htbase
#SBATCH --time=01:00:00
#SBATCH --mem=48G
# Scheduler output is supplied by submit.sh and remains outside the checkout.

set -Eeuo pipefail

module --force purge
module load EasyBuild/2024a Python
ulimit -s unlimited

PROJECT_ROOT=${PROJECT_ROOT:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)}
source "${PROJECT_ROOT}/benchmarks/runtime_paths.sh"
: "${MACE_MODEL:?Set MACE_MODEL to the frozen checkpoint}"
: "${RESULT_PATH:?Set RESULT_PATH to an external JSON file}"
DATA_ROOT=${DATA_ROOT:-${MACE_FNO_WORK_ROOT}/ti2co2_adsorbates/data/prepared}
JOB_WORK_ROOT=${JOB_WORK_ROOT:-$(dirname "${RESULT_PATH}")/work}

mkdir -p "$(dirname "${RESULT_PATH}")" "${JOB_WORK_ROOT}"
test -s "${MACE_MODEL}"
for split in train test; do
    test -s "${DATA_ROOT}/${split}.xyz"
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
    --train-file "${DATA_ROOT}/train.xyz" \
    --test-file "${DATA_ROOT}/test.xyz" \
    --energy-key energy_dft \
    --forces-key forces_dft \
    --device cuda \
    --dtype float64 \
    --output "${RESULT_PATH}"

test -s "${RESULT_PATH}"
