#!/usr/bin/env bash
#SBATCH --job-name=les-water-3d
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --partition=debug-gpu
#SBATCH --account=htbase
#SBATCH --time=02:00:00
#SBATCH --mem=64G
#SBATCH --output=logs/les-water-fno-3d-%j.out

set -Eeuo pipefail

module --force purge
module load EasyBuild/2024a Python
ulimit -s unlimited

PROJECT_ROOT=/gpfs/home/acad/ucl-modl/wchen/mxene_proj/mace_fno
DATA_ROOT=${PROJECT_ROOT}/data/les_water
MACE_MODEL=${PROJECT_ROOT}/artifacts/les_water/pretrained/H20_stagetwo.model
VALIDATION_INDICES=${VALIDATION_INDICES:-${DATA_ROOT}/validation_indices_seed17.txt}
CACHE_ROOT=${CACHE_ROOT:-artifacts/les_water/cache}
MODEL_DTYPE=${MODEL_DTYPE:-float64}
FNO_SEED=${FNO_SEED:-17}
GRID=${GRID:-24}
Z_GRID=${Z_GRID:-24}
MODES=${MODES:-4}
Z_MODES=${Z_MODES:-4}
CHANNELS=${CHANNELS:-4}
FNO_HIDDEN_CHANNELS=${FNO_HIDDEN_CHANNELS:-16}
FNO_LAYERS=${FNO_LAYERS:-2}
SPECTRAL_SYMMETRY=${SPECTRAL_SYMMETRY:-none}
SPECTRAL_GROUPS=${SPECTRAL_GROUPS:-1}
VOLUME_INTERLACING=${VOLUME_INTERLACING:-1}
BATCH_SIZE=${BATCH_SIZE:-2}
EVALUATION_BATCH_SIZE=${EVALUATION_BATCH_SIZE:-8}
ACCUMULATION_STEPS=${ACCUMULATION_STEPS:-1}
STEPS=${STEPS:-1500}
EVAL_INTERVAL=${EVAL_INTERVAL:-250}
LEARNING_RATE=${LEARNING_RATE:-3e-4}
OUTPUT_INITIALIZATION_SCALE=${OUTPUT_INITIALIZATION_SCALE:-0}
OUTPUT_WARMUP_STEPS=${OUTPUT_WARMUP_STEPS:-0}
OUTPUT_WARMUP_LEARNING_RATE=${OUTPUT_WARMUP_LEARNING_RATE:-0}
ENERGY_SCALE=${ENERGY_SCALE:-0.000291}
FORCE_SCALE=${FORCE_SCALE:-0.026590}
FORCE_WEIGHT=${FORCE_WEIGHT:-1}
LR_SCHEDULER=${LR_SCHEDULER:-plateau}
LR_DECAY_FACTOR=${LR_DECAY_FACTOR:-0.5}
LR_PATIENCE_EVALS=${LR_PATIENCE_EVALS:-4}
MINIMUM_LEARNING_RATE=${MINIMUM_LEARNING_RATE:-1e-6}
EARLY_STOPPING_PATIENCE_EVALS=${EARLY_STOPPING_PATIENCE_EVALS:-0}
CHECKPOINT=${CHECKPOINT:-artifacts/les_water/les_water_fno_3d_seed${FNO_SEED}_${MODEL_DTYPE}.pt}

cd "${PROJECT_ROOT}"
mkdir -p artifacts/les_water "${CACHE_ROOT}" logs

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

# The published 50-configuration test file is never used for gradients or
# model selection. A deterministic 5% validation split is taken only from the
# 604 published training configurations.
python3 -u examples/train_mace_residual.py \
    --mace-model "${MACE_MODEL}" \
    --train-file "${DATA_ROOT}/train-H2O_RPBE-D3.xyz" \
    --validation-indices-file "${VALIDATION_INDICES}" \
    --test-file "${DATA_ROOT}/test-H2O_RPBE-D3.xyz" \
    --train-cache "${CACHE_ROOT}/train-${MODEL_DTYPE}.pt" \
    --test-cache "${CACHE_ROOT}/test-${MODEL_DTYPE}.pt" \
    --energy-key energy \
    --forces-key forces \
    --spatial-scheme 3d \
    --grid "${GRID}" \
    --z-grid "${Z_GRID}" \
    --modes "${MODES}" \
    --z-modes "${Z_MODES}" \
    --channels "${CHANNELS}" \
    --source-hidden-channels 64 \
    --fno-hidden-channels "${FNO_HIDDEN_CHANNELS}" \
    --fno-layers "${FNO_LAYERS}" \
    --architecture nonlinear \
    --spectral-symmetry "${SPECTRAL_SYMMETRY}" \
    --spectral-groups "${SPECTRAL_GROUPS}" \
    --volume-interlacing "${VOLUME_INTERLACING}" \
    --steps "${STEPS}" \
    --learning-rate "${LEARNING_RATE}" \
    --output-initialization-scale "${OUTPUT_INITIALIZATION_SCALE}" \
    --output-warmup-steps "${OUTPUT_WARMUP_STEPS}" \
    --output-warmup-learning-rate "${OUTPUT_WARMUP_LEARNING_RATE}" \
    --lr-scheduler "${LR_SCHEDULER}" \
    --lr-decay-factor "${LR_DECAY_FACTOR}" \
    --lr-patience-evals "${LR_PATIENCE_EVALS}" \
    --minimum-learning-rate "${MINIMUM_LEARNING_RATE}" \
    --early-stopping-patience-evals "${EARLY_STOPPING_PATIENCE_EVALS}" \
    --energy-weight 1 \
    --force-weight "${FORCE_WEIGHT}" \
    --energy-scale "${ENERGY_SCALE}" \
    --force-scale "${FORCE_SCALE}" \
    --validation-fraction 0.05 \
    --eval-interval "${EVAL_INTERVAL}" \
    --batch-size "${BATCH_SIZE}" \
    --evaluation-batch-size "${EVALUATION_BATCH_SIZE}" \
    --accumulation-steps "${ACCUMULATION_STEPS}" \
    --evaluation-scope validation-test \
    --seed "${FNO_SEED}" \
    --dtype "${MODEL_DTYPE}" \
    --device cuda \
    --checkpoint "${CHECKPOINT}"
