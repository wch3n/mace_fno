#!/usr/bin/env bash
#SBATCH --job-name=llzo-fno
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --partition=debug-gpu
#SBATCH --account=htbase
#SBATCH --time=02:00:00
#SBATCH --mem=64G
# Scheduler output is supplied by submit.sh so it stays outside the checkout.

set -Eeuo pipefail

module --force purge
module load EasyBuild/2024a Python
ulimit -s unlimited

PROJECT_ROOT=${PROJECT_ROOT:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)}
source "${PROJECT_ROOT}/benchmarks/runtime_paths.sh"
DATA_ROOT=${DATA_ROOT:-${PROJECT_ROOT}/data/llzo_qnep/prepared}
RUN_ROOT=${RUN_ROOT:-${MACE_FNO_WORK_ROOT}/llzo_qnep/fno_3d}
MACE_MODEL=${MACE_MODEL:?Set MACE_MODEL to the one-interaction checkpoint}
CACHE_ROOT=${CACHE_ROOT:-${RUN_ROOT}/cache}
MODEL_DTYPE=${MODEL_DTYPE:-float64}
FNO_SEED=${FNO_SEED:-17}
GRID=${GRID:-24}
MODES=${MODES:-4}
CHANNELS=${CHANNELS:-4}
FNO_HIDDEN_CHANNELS=${FNO_HIDDEN_CHANNELS:-16}
FNO_LAYERS=${FNO_LAYERS:-2}
VOLUME_INTERLACING=${VOLUME_INTERLACING:-1}
INTERLACING_TRAINING=${INTERLACING_TRAINING:-full}
SPECTRAL_SYMMETRY=${SPECTRAL_SYMMETRY:-none}
SPECTRAL_GROUPS=${SPECTRAL_GROUPS:-1}
METRIC_HIDDEN_CHANNELS=${METRIC_HIDDEN_CHANNELS:-16}
BATCH_SIZE=${BATCH_SIZE:-4}
EVALUATION_BATCH_SIZE=${EVALUATION_BATCH_SIZE:-8}
ACCUMULATION_STEPS=${ACCUMULATION_STEPS:-1}
STEPS=${STEPS:-20000}
EVAL_INTERVAL=${EVAL_INTERVAL:-500}
SPECTRAL_DIAGNOSTIC_SAMPLES=${SPECTRAL_DIAGNOSTIC_SAMPLES:-4}
SPECTRAL_DIAGNOSTIC_MAX_MODE=${SPECTRAL_DIAGNOSTIC_MAX_MODE:-1}
SPECTRAL_DIAGNOSTIC_FIT_SHELLS=${SPECTRAL_DIAGNOSTIC_FIT_SHELLS:-3}
SPECTRAL_DIAGNOSTIC_RELATIVE_AMPLITUDE=${SPECTRAL_DIAGNOSTIC_RELATIVE_AMPLITUDE:-0.05}
SPECTRAL_DIAGNOSTIC_FIELD_BATCH_SIZE=${SPECTRAL_DIAGNOSTIC_FIELD_BATCH_SIZE:-32}
SPECTRAL_DIAGNOSTIC_DEPTH=${SPECTRAL_DIAGNOSTIC_DEPTH:-fast}
LEARNING_RATE=${LEARNING_RATE:-3e-4}
OUTPUT_INITIALIZATION_SCALE=${OUTPUT_INITIALIZATION_SCALE:-0.1}
ENERGY_SCALE=${ENERGY_SCALE:-0.01}
FORCE_SCALE=${FORCE_SCALE:-0.10}
FORCE_WEIGHT=${FORCE_WEIGHT:-1}
CHECKPOINT=${CHECKPOINT:-${RUN_ROOT}/llzo_fno_3d_seed${FNO_SEED}_${MODEL_DTYPE}.pt}
SPECTRAL_DIAGNOSTIC_OUTPUT=${SPECTRAL_DIAGNOSTIC_OUTPUT:-${CHECKPOINT%.pt}_spectral_training.json}
JOB_WORK_ROOT=${JOB_WORK_ROOT:-${RUN_ROOT}/work}

mkdir -p "${RUN_ROOT}" "${CACHE_ROOT}" "${JOB_WORK_ROOT}"
test -s "${MACE_MODEL}"
for split in train validation test; do
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

if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi
fi

spectral_diagnostic_args=()
if (( SPECTRAL_DIAGNOSTIC_SAMPLES > 0 )); then
    spectral_diagnostic_args+=(
        --spectral-diagnostic-samples "${SPECTRAL_DIAGNOSTIC_SAMPLES}"
        --spectral-diagnostic-max-mode "${SPECTRAL_DIAGNOSTIC_MAX_MODE}"
        --spectral-diagnostic-fit-shells "${SPECTRAL_DIAGNOSTIC_FIT_SHELLS}"
        --spectral-diagnostic-relative-amplitude "${SPECTRAL_DIAGNOSTIC_RELATIVE_AMPLITUDE}"
        --spectral-diagnostic-field-batch-size "${SPECTRAL_DIAGNOSTIC_FIELD_BATCH_SIZE}"
        --spectral-diagnostic-depth "${SPECTRAL_DIAGNOSTIC_DEPTH}"
        --spectral-diagnostic-output "${SPECTRAL_DIAGNOSTIC_OUTPUT}"
    )
fi

# LLZO contains variable orthorhombic, tetragonal, and cubic cells. The
# anisotropic mode supplies log(volume length) and the normalized lattice
# metric. The optional cubic-adaptive operator has an EqGINO core that is exact
# for cubic cells and a gated conventional branch for noncubic response.
python3 -u -m mace_fno.cli.train_residual \
    --mace-model "${MACE_MODEL}" \
    --train-file "${DATA_ROOT}/train.xyz" \
    --validation-file "${DATA_ROOT}/validation.xyz" \
    --test-file "${DATA_ROOT}/test.xyz" \
    --train-cache "${CACHE_ROOT}/train-${MODEL_DTYPE}.pt" \
    --validation-cache "${CACHE_ROOT}/validation-${MODEL_DTYPE}.pt" \
    --test-cache "${CACHE_ROOT}/test-${MODEL_DTYPE}.pt" \
    --energy-key energy \
    --forces-key forces \
    --num-atoms 192 \
    --spatial-scheme 3d \
    --cell-mode anisotropic \
    --grid "${GRID}" \
    --z-grid "${GRID}" \
    --modes "${MODES}" \
    --z-modes "${MODES}" \
    --channels "${CHANNELS}" \
    --source-hidden-channels 64 \
    --fno-hidden-channels "${FNO_HIDDEN_CHANNELS}" \
    --fno-layers "${FNO_LAYERS}" \
    --architecture nonlinear \
    --spectral-symmetry "${SPECTRAL_SYMMETRY}" \
    --spectral-groups "${SPECTRAL_GROUPS}" \
    --metric-hidden-channels "${METRIC_HIDDEN_CHANNELS}" \
    --volume-interlacing "${VOLUME_INTERLACING}" \
    --interlacing-training "${INTERLACING_TRAINING}" \
    --steps "${STEPS}" \
    --learning-rate "${LEARNING_RATE}" \
    --output-initialization-scale "${OUTPUT_INITIALIZATION_SCALE}" \
    --lr-scheduler plateau \
    --lr-decay-factor 0.5 \
    --lr-patience-evals 4 \
    --minimum-learning-rate 1e-6 \
    --early-stopping-patience-evals 0 \
    --energy-weight 1 \
    --force-weight "${FORCE_WEIGHT}" \
    --energy-scale "${ENERGY_SCALE}" \
    --force-scale "${FORCE_SCALE}" \
    --validation-fraction 0.10 \
    --eval-interval "${EVAL_INTERVAL}" \
    "${spectral_diagnostic_args[@]}" \
    --batch-size "${BATCH_SIZE}" \
    --evaluation-batch-size "${EVALUATION_BATCH_SIZE}" \
    --accumulation-steps "${ACCUMULATION_STEPS}" \
    --evaluation-scope validation-test \
    --seed "${FNO_SEED}" \
    --dtype "${MODEL_DTYPE}" \
    --device cuda \
    --checkpoint "${CHECKPOINT}"

test -s "${CHECKPOINT}"
