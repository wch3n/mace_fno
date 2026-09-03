#!/usr/bin/env bash
#SBATCH --job-name=les-water-fno
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --partition=debug-gpu
#SBATCH --account=htbase
#SBATCH --time=02:00:00
#SBATCH --mem=64G
# Scheduler output is supplied by the launcher and remains outside the checkout.

set -Eeuo pipefail

module --force purge
module load EasyBuild/2024a Python
ulimit -s unlimited

PROJECT_ROOT=${PROJECT_ROOT:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)}
source "${PROJECT_ROOT}/benchmarks/runtime_paths.sh"
SOURCE_ROOT=${SOURCE_ROOT:-${MACE_FNO_WORK_ROOT}/les_water/source}
TRAIN_FILE=${TRAIN_FILE:-${SOURCE_ROOT}/data/train-H2O_RPBE-D3.xyz}
TEST_FILE=${TEST_FILE:-${SOURCE_ROOT}/data/test-H2O_RPBE-D3.xyz}
VALIDATION_INDICES=${VALIDATION_INDICES:-${PROJECT_ROOT}/benchmarks/les_water/validation_indices_seed17.txt}
MACE_MODEL=${MACE_MODEL:-${SOURCE_ROOT}/pretrained/H20_stagetwo.model}
RUN_ROOT=${RUN_ROOT:-${MACE_FNO_WORK_ROOT}/les_water/fno_metric_eqgino}
CACHE_ROOT=${CACHE_ROOT:-${RUN_ROOT}/cache}
FNO_CONFIG=${FNO_CONFIG:-${PROJECT_ROOT}/benchmarks/les_water/train_fno_3d.yaml}
MODEL_DTYPE=${MODEL_DTYPE:-float64}
FNO_SEED=${FNO_SEED:-17}
CHECKPOINT=${CHECKPOINT:-${RUN_ROOT}/les_water_fno_3d_seed${FNO_SEED}_${MODEL_DTYPE}.pt}
JOB_WORK_ROOT=${JOB_WORK_ROOT:-${RUN_ROOT}/work}

mkdir -p "${RUN_ROOT}" "${CACHE_ROOT}" "${JOB_WORK_ROOT}"
for required in \
    "${FNO_CONFIG}" "${TRAIN_FILE}" "${TEST_FILE}" \
    "${VALIDATION_INDICES}" "${MACE_MODEL}"; do
    test -s "${required}"
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

# The LES dataset has one fixed 12.429-A cubic cell. Metric-aware EqGINO
# therefore provides the appropriate isotropic reciprocal-space contraction.
train_args=(
    --config "${FNO_CONFIG}"
    --mace-model "${MACE_MODEL}"
    --train-file "${TRAIN_FILE}"
    --validation-indices-file "${VALIDATION_INDICES}"
    --test-file "${TEST_FILE}"
    --train-cache "${CACHE_ROOT}/train-${MODEL_DTYPE}.pt"
    --validation-cache "${CACHE_ROOT}/validation-${MODEL_DTYPE}.pt"
    --test-cache "${CACHE_ROOT}/test-${MODEL_DTYPE}.pt"
    --seed "${FNO_SEED}"
    --dtype "${MODEL_DTYPE}"
    --checkpoint "${CHECKPOINT}"
)

for override in \
    STEPS:--steps GRID:--grid MODES:--modes CHANNELS:--channels \
    SOURCE_HIDDEN_CHANNELS:--source-hidden-channels \
    FNO_HIDDEN_CHANNELS:--fno-hidden-channels FNO_LAYERS:--fno-layers \
    SPECTRAL_GROUPS:--spectral-groups \
    METRIC_HIDDEN_CHANNELS:--metric-hidden-channels \
    VOLUME_INTERLACING:--volume-interlacing \
    INTERLACING_TRAINING:--interlacing-training \
    MACE_TRAINING:--mace-training \
    LEARNING_RATE:--learning-rate \
    MACE_LEARNING_RATE:--mace-learning-rate \
    MACE_WARMUP_STEPS:--mace-warmup-steps \
    OUTPUT_INITIALIZATION_SCALE:--output-initialization-scale \
    ENERGY_SCALE:--energy-scale FORCE_SCALE:--force-scale \
    ENERGY_WEIGHT:--energy-weight FORCE_WEIGHT:--force-weight \
    EVAL_INTERVAL:--eval-interval \
    SPECTRAL_DIAGNOSTIC_SAMPLES:--spectral-diagnostic-samples \
    BATCH_SIZE:--batch-size EVALUATION_BATCH_SIZE:--evaluation-batch-size \
    ACCUMULATION_STEPS:--accumulation-steps; do
    mace_fno_append_env_option train_args "${override%%:*}" "${override#*:}"
done

if [[ -v GRID ]]; then
    train_args+=(--z-grid "${GRID}")
fi
if [[ -v MODES ]]; then
    train_args+=(--z-modes "${MODES}")
fi

python3 -u -m mace_fno.cli.train_residual "${train_args[@]}"

test -s "${CHECKPOINT}"
