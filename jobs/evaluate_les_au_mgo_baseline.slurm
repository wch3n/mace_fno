#!/usr/bin/env bash
#SBATCH --job-name=les-au-mgo-eval
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --partition=debug-gpu
#SBATCH --account=htbase
#SBATCH --time=02:00:00
#SBATCH --mem=48G
#SBATCH --output=logs/les-au-mgo-eval-%j.out

set -Eeuo pipefail

module --force purge
module load EasyBuild/2024a Python
ulimit -s unlimited

PROJECT_ROOT=/gpfs/home/acad/ucl-modl/wchen/mxene_proj/mace_fno
DATA_ROOT=${PROJECT_ROOT}/data/les_au_mgo
MACE_MODEL=${MACE_MODEL:-${PROJECT_ROOT}/artifacts/les_au_mgo/pretrained/Au2-MgO_r5.5_nl0_stagetwo.model}
RESULT_PATH=${RESULT_PATH:-${PROJECT_ROOT}/artifacts/les_au_mgo/Au2-MgO_r5.5_nl0_baseline.json}

cd "${PROJECT_ROOT}"
mkdir -p artifacts/les_au_mgo logs

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="${PROJECT_ROOT}/src"
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export MPLCONFIGDIR="${TMPDIR:-/tmp}/mace-fno-matplotlib-${SLURM_JOB_ID}"
mkdir -p "${MPLCONFIGDIR}"

python3 -u examples/evaluate_mace_baseline.py \
    --model "${MACE_MODEL}" \
    --train-file "${DATA_ROOT}/train-Au-MgO-Al.xyz" \
    --test-file "${DATA_ROOT}/test-Au-MgO-Al.xyz" \
    --energy-key energy \
    --forces-key forces \
    --device cuda \
    --dtype float64 \
    --output "${RESULT_PATH}"
