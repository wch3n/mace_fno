#!/usr/bin/env bash
#SBATCH --job-name=ti-mace-eval
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --partition=debug-gpu
#SBATCH --account=htbase
#SBATCH --time=02:00:00
#SBATCH --mem=48G
#SBATCH --output=logs/ti-mace-eval-%j.out

set -Eeuo pipefail

module --force purge
module load EasyBuild/2024a Python
ulimit -s unlimited

PROJECT_ROOT=/gpfs/home/acad/ucl-modl/wchen/mxene_proj/mace_fno
TI_ROOT=/gpfs/scratch/acad/htbase/wchen/mxene/mlip/pbe0-rvv10_8/Ti
MODEL_NAME=${MODEL_NAME:?submit with MODEL_NAME=<checkpoint basename>}
MODEL_PATH=${TI_ROOT}/models/${MODEL_NAME}.model
RESULT_PATH=${PROJECT_ROOT}/artifacts/${MODEL_NAME}_1331_baseline.json

cd "${PROJECT_ROOT}"
mkdir -p artifacts logs

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="${PROJECT_ROOT}/src"
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export MPLCONFIGDIR="${TMPDIR:-/tmp}/mace-fno-matplotlib-${SLURM_JOB_ID}"
mkdir -p "${MPLCONFIGDIR}"

HEAD_ARGUMENT=()
if [[ "${MODEL_NAME}" == *multi* ]]; then
    HEAD_ARGUMENT=(--head Default)
fi

python3 -u examples/evaluate_mace_baseline.py \
    --model "${MODEL_PATH}" \
    --train-file "${TI_ROOT}/pbe0-rvv10_1331_train.xyz" \
    --test-file "${TI_ROOT}/pbe0-rvv10_1331_test.xyz" \
    --energy-key energy_dft \
    --forces-key forces_dft \
    --device cuda \
    --dtype float64 \
    --output "${RESULT_PATH}" \
    "${HEAD_ARGUMENT[@]}"
