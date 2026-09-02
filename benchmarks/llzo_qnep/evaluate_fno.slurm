#!/usr/bin/env bash
#SBATCH --job-name=llzo-fno-eval
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --partition=debug-gpu
#SBATCH --account=htbase
#SBATCH --time=01:00:00
#SBATCH --mem=48G
# Scheduler output is supplied by submit.sh so it stays outside the checkout.

set -Eeuo pipefail

module --force purge
module load EasyBuild/2024a Python
ulimit -s unlimited

PROJECT_ROOT=${PROJECT_ROOT:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)}
CHECKPOINT=${CHECKPOINT:?Set CHECKPOINT to the trained MACE+FNO checkpoint}
RESULT_PATH=${RESULT_PATH:-${CHECKPOINT%.pt}_evaluation.json}
MODEL_DTYPE=${MODEL_DTYPE:-float64}
EVALUATION_BATCH_SIZE=${EVALUATION_BATCH_SIZE:-8}
JOB_WORK_ROOT=${JOB_WORK_ROOT:-$(dirname "${RESULT_PATH}")/work}

mkdir -p "$(dirname "${RESULT_PATH}")" "${JOB_WORK_ROOT}"
test -s "${CHECKPOINT}"
cd "${JOB_WORK_ROOT}"

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="${PROJECT_ROOT}/src"
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1

python3 -u -m mace_fno.cli.evaluate_residual \
    --checkpoint "${CHECKPOINT}" \
    --splits validation test \
    --batch-size "${EVALUATION_BATCH_SIZE}" \
    --device cuda \
    --dtype "${MODEL_DTYPE}" \
    --output "${RESULT_PATH}"

test -s "${RESULT_PATH}"
