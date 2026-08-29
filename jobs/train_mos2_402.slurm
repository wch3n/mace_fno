#!/usr/bin/env bash
#SBATCH --job-name=fno-402-tuned
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --partition=debug-gpu
#SBATCH --account=htbase
#SBATCH --time=02:00:00
#SBATCH --mem=64G
#SBATCH --output=logs/fno-402-tuned-%j.out

set -Eeuo pipefail

module --force purge
module load EasyBuild/2024a Python
ulimit -s unlimited

PROJECT_ROOT=${PROJECT_ROOT:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}
: "${MACE_MODEL:?Set MACE_MODEL to the frozen MoS2 checkpoint}"
: "${TRAIN_FILE:?Set TRAIN_FILE to the labeled MoS2 XYZ dataset}"
NUM_ATOMS=${NUM_ATOMS:-402}
CHECKPOINT=${CHECKPOINT:-artifacts/mos2_402_tuned.pt}
CACHE_ROOT=${CACHE_ROOT:-artifacts/cache/mos2_402}
BATCH_SIZE=${BATCH_SIZE:-2}
EVALUATION_BATCH_SIZE=${EVALUATION_BATCH_SIZE:-4}
ACCUMULATION_STEPS=${ACCUMULATION_STEPS:-2}
EVALUATION_SCOPE=${EVALUATION_SCOPE:-validation-test}
MODEL_DTYPE=${MODEL_DTYPE:-float64}

cd "${PROJECT_ROOT}"
mkdir -p artifacts "${CACHE_ROOT}" logs

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="${PROJECT_ROOT}/src"
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export MPLCONFIGDIR="${TMPDIR:-/tmp}/mace-fno-matplotlib-${SLURM_JOB_ID}"
mkdir -p "${MPLCONFIGDIR}"

if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi
fi

python3 -u examples/train_mace_residual.py \
    --mace-model "${MACE_MODEL}" \
    --train-file "${TRAIN_FILE}" \
    --train-cache "${CACHE_ROOT}/train-${NUM_ATOMS}-${MODEL_DTYPE}.pt" \
    --energy-key energy_dft \
    --forces-key forces_dft \
    --num-atoms "${NUM_ATOMS}" \
    --allow-periodic-z \
    --steps 500 \
    --grid 32 \
    --modes 4 \
    --channels 2 \
    --source-hidden-channels 32 \
    --fno-hidden-channels 16 \
    --fno-layers 2 \
    --architecture linear \
    --learning-rate 3e-4 \
    --energy-weight 1 \
    --force-weight 4 \
    --energy-scale 0.001 \
    --force-scale 0.01 \
    --validation-fraction 0.25 \
    --eval-interval 100 \
    --batch-size "${BATCH_SIZE}" \
    --evaluation-batch-size "${EVALUATION_BATCH_SIZE}" \
    --accumulation-steps "${ACCUMULATION_STEPS}" \
    --evaluation-scope "${EVALUATION_SCOPE}" \
    --dtype "${MODEL_DTYPE}" \
    --device cuda \
    --checkpoint "${CHECKPOINT}"
