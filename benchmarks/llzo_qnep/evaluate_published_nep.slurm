#!/usr/bin/env bash
#SBATCH --job-name=llzo-nep-qnep
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=batch
#SBATCH --account=htbase
#SBATCH --time=01:00:00
#SBATCH --mem=16G
# Scheduler output is supplied by submit.sh so it stays outside the checkout.

set -Eeuo pipefail

module --force purge
module load EasyBuild/2024a Python

PROJECT_ROOT=${PROJECT_ROOT:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)}
SOURCE_DATA_ROOT=${SOURCE_DATA_ROOT:?Set SOURCE_DATA_ROOT to the LLZO data directory}
DATA_ROOT=${DATA_ROOT:-${SOURCE_DATA_ROOT}/prepared}
RESULT_PATH=${RESULT_PATH:?Set RESULT_PATH for the JSON report}
JOB_WORK_ROOT=${JOB_WORK_ROOT:-$(dirname "${RESULT_PATH}")/work}

mkdir -p "$(dirname "${RESULT_PATH}")" "${JOB_WORK_ROOT}"
cd "${JOB_WORK_ROOT}"

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="${PROJECT_ROOT}/src"

python3 -c 'from calorine.calculators import CPUNEP; print("calorine CPUNEP available")'
python3 -u "${PROJECT_ROOT}/benchmarks/llzo_qnep/evaluate_published_nep.py" \
    --test-file "${DATA_ROOT}/test.xyz" \
    --model "NEP=${SOURCE_DATA_ROOT}/models/nep-LiLaZrO-PBEsol.txt" \
    --model "qNEP-mode1=${SOURCE_DATA_ROOT}/models/qnep-mode1-LiLaZrO-PBEsol.txt" \
    --model "qNEP-mode2=${SOURCE_DATA_ROOT}/models/qnep-mode2-LiLaZrO-PBEsol.txt" \
    --output "${RESULT_PATH}"

test -s "${RESULT_PATH}"
