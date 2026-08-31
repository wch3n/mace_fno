#!/usr/bin/env bash
#SBATCH --job-name=au-mgo-eval
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --partition=debug-gpu
#SBATCH --account=htbase
#SBATCH --time=02:00:00
#SBATCH --mem=48G
#SBATCH --output=logs/au-mgo-eval-%j.out

set -Eeuo pipefail

module --force purge
module load EasyBuild/2024a Python
ulimit -s unlimited

PROJECT_ROOT=${PROJECT_ROOT:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)}
source "${PROJECT_ROOT}/benchmarks/runtime_paths.sh"
DATA_ROOT=${PROJECT_ROOT}/data/les_au_mgo
MACE_MODEL=${MACE_MODEL:-${MACE_FNO_WORK_ROOT}/les_au_mgo/pretrained/Au2-MgO_r5.5_nl0_stagetwo.model}
RESULT_PATH=${RESULT_PATH:-${MACE_FNO_WORK_ROOT}/les_au_mgo/Au2-MgO_r5.5_nl0_baseline.json}

cd "${PROJECT_ROOT}"
mkdir -p "$(dirname "${RESULT_PATH}")" logs

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="${PROJECT_ROOT}/src"
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export MPLCONFIGDIR="${TMPDIR:-/tmp}/mace-fno-matplotlib-${SLURM_JOB_ID}"
mkdir -p "${MPLCONFIGDIR}"

python3 -u -m mace_fno.cli.evaluate_mace \
    --model "${MACE_MODEL}" \
    --train-file "${DATA_ROOT}/train-Au-MgO-Al.xyz" \
    --test-file "${DATA_ROOT}/test-Au-MgO-Al.xyz" \
    --energy-key energy \
    --forces-key forces \
    --device cuda \
    --dtype float64 \
    --output "${RESULT_PATH}"
