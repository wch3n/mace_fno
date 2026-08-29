#!/usr/bin/env bash
#SBATCH --job-name=les-au-mgo-fno
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --partition=debug-gpu
#SBATCH --account=htbase
#SBATCH --time=02:00:00
#SBATCH --mem=64G
#SBATCH --output=logs/les-au-mgo-fno-%j.out

set -Eeuo pipefail

module --force purge
module load EasyBuild/2024a Python
ulimit -s unlimited

PROJECT_ROOT=${PROJECT_ROOT:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}
DATA_ROOT=${PROJECT_ROOT}/data/les_au_mgo
MACE_MODEL=${PROJECT_ROOT}/artifacts/les_au_mgo/pretrained/Au2-MgO_r5.5_nl0_stagetwo.model
VALIDATION_INDICES=${PROJECT_ROOT}/artifacts/les_au_mgo/mace_r5.5_nl1_l0/Au2-MgO_valid_indices_123.txt
CHECKPOINT=${CHECKPOINT:-artifacts/les_au_mgo/les_au_mgo_fno_seed17.pt}
CACHE_ROOT=${CACHE_ROOT:-artifacts/les_au_mgo/cache}
FORCE_WEIGHT=${FORCE_WEIGHT:-1}
LEARNING_RATE=${LEARNING_RATE:-3e-4}
FNO_SEED=${FNO_SEED:-17}
BATCH_SIZE=${BATCH_SIZE:-4}
EVALUATION_BATCH_SIZE=${EVALUATION_BATCH_SIZE:-8}
ACCUMULATION_STEPS=${ACCUMULATION_STEPS:-1}
EVALUATION_SCOPE=${EVALUATION_SCOPE:-validation-test}
MODEL_DTYPE=${MODEL_DTYPE:-float64}
STEPS=${STEPS:-1500}
EVAL_INTERVAL=${EVAL_INTERVAL:-250}
LR_SCHEDULER=${LR_SCHEDULER:-none}
LR_DECAY_FACTOR=${LR_DECAY_FACTOR:-0.5}
LR_PATIENCE_EVALS=${LR_PATIENCE_EVALS:-4}
MINIMUM_LEARNING_RATE=${MINIMUM_LEARNING_RATE:-1e-6}
EARLY_STOPPING_PATIENCE_EVALS=${EARLY_STOPPING_PATIENCE_EVALS:-0}

cd "${PROJECT_ROOT}"
mkdir -p artifacts/les_au_mgo "${CACHE_ROOT}" logs

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

# Loss scales are the frozen model's training-set RMSEs, rounded from
# 2.260 meV/atom and 56.280 meV/Angstrom. The official test set remains
# evaluation-only, and validation uses the MACE seed-123 indices.
python3 -u examples/train_mace_residual.py \
    --mace-model "${MACE_MODEL}" \
    --train-file "${DATA_ROOT}/train-Au-MgO-Al.xyz" \
    --validation-indices-file "${VALIDATION_INDICES}" \
    --test-file "${DATA_ROOT}/test-Au-MgO-Al.xyz" \
    --train-cache "${CACHE_ROOT}/train-${MODEL_DTYPE}.pt" \
    --test-cache "${CACHE_ROOT}/test-${MODEL_DTYPE}.pt" \
    --energy-key energy \
    --forces-key forces \
    --allow-periodic-z \
    --steps "${STEPS}" \
    --grid 24 \
    --modes 4 \
    --channels 4 \
    --source-hidden-channels 64 \
    --fno-hidden-channels 16 \
    --fno-layers 2 \
    --architecture nonlinear \
    --learning-rate "${LEARNING_RATE}" \
    --lr-scheduler "${LR_SCHEDULER}" \
    --lr-decay-factor "${LR_DECAY_FACTOR}" \
    --lr-patience-evals "${LR_PATIENCE_EVALS}" \
    --minimum-learning-rate "${MINIMUM_LEARNING_RATE}" \
    --early-stopping-patience-evals "${EARLY_STOPPING_PATIENCE_EVALS}" \
    --energy-weight 1 \
    --force-weight "${FORCE_WEIGHT}" \
    --energy-scale 0.002260 \
    --force-scale 0.056280 \
    --validation-fraction 0.05 \
    --eval-interval "${EVAL_INTERVAL}" \
    --batch-size "${BATCH_SIZE}" \
    --evaluation-batch-size "${EVALUATION_BATCH_SIZE}" \
    --accumulation-steps "${ACCUMULATION_STEPS}" \
    --evaluation-scope "${EVALUATION_SCOPE}" \
    --seed "${FNO_SEED}" \
    --dtype "${MODEL_DTYPE}" \
    --device cuda \
    --checkpoint "${CHECKPOINT}"
