#!/usr/bin/env bash
#SBATCH --job-name=water-scan-fno
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --partition=debug-gpu
#SBATCH --account=htbase
#SBATCH --time=2:00:00
#SBATCH --mem=64G
# Scheduler output is supplied by submit.sh so it stays outside the checkout.

set -Eeuo pipefail

module --force purge
module load EasyBuild/2024a Python
ulimit -s unlimited

PROJECT_ROOT=${PROJECT_ROOT:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)}
source "${PROJECT_ROOT}/benchmarks/runtime_paths.sh"
DATA_ROOT=${DATA_ROOT:-${PROJECT_ROOT}/data/water_scan_qnep/prepared}
RUN_ROOT=${RUN_ROOT:-${MACE_FNO_WORK_ROOT}/water_scan_qnep/fno_3d_metric_eqgino}
MACE_MODEL=${MACE_MODEL:-${MACE_FNO_WORK_ROOT}/water_scan_qnep/mace_r4p5_nl0_e100_s2e50/models/water-SCAN-r4p5-nl0_stagetwo.model}
CACHE_ROOT=${CACHE_ROOT:-${RUN_ROOT}/cache}
FNO_CONFIG=${FNO_CONFIG:-${PROJECT_ROOT}/benchmarks/water_scan_qnep/train_fno_3d.yaml}
MODEL_DTYPE=${MODEL_DTYPE:-float32}
FNO_SEED=${FNO_SEED:-17}
CHECKPOINT=${CHECKPOINT:-${RUN_ROOT}/water_scan_fno_3d_seed${FNO_SEED}_${MODEL_DTYPE}.pt}
JOB_WORK_ROOT=${JOB_WORK_ROOT:-${RUN_ROOT}/work}

mkdir -p "${RUN_ROOT}" "${CACHE_ROOT}" "${JOB_WORK_ROOT}"
test -s "${FNO_CONFIG}"
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

# The cells are aligned cubes with varying lengths. Isotropic conditioning
# supplies log(cell length) to the nonlinear metric-aware EqGINO operator.
# Interlacing is intentionally disabled by default for throughput.
train_args=(
    --config "${FNO_CONFIG}"
    --mace-model "${MACE_MODEL}"
    --train-file "${DATA_ROOT}/train.xyz"
    --validation-file "${DATA_ROOT}/validation.xyz"
    --test-file "${DATA_ROOT}/test.xyz"
    --train-cache "${CACHE_ROOT}/train-${MODEL_DTYPE}.pt"
    --validation-cache "${CACHE_ROOT}/validation-${MODEL_DTYPE}.pt"
    --test-cache "${CACHE_ROOT}/test-${MODEL_DTYPE}.pt"
    --seed "${FNO_SEED}"
    --dtype "${MODEL_DTYPE}"
    --checkpoint "${CHECKPOINT}"
)

for override in \
    STEPS:--steps GRID:--grid MODES:--modes CHANNELS:--channels \
    FNO_HIDDEN_CHANNELS:--fno-hidden-channels FNO_LAYERS:--fno-layers \
    SPECTRAL_SYMMETRY:--spectral-symmetry \
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
    FORCE_WEIGHT:--force-weight EVAL_INTERVAL:--eval-interval \
    SPECTRAL_DIAGNOSTIC_SAMPLES:--spectral-diagnostic-samples \
    SPECTRAL_DIAGNOSTIC_MAX_MODE:--spectral-diagnostic-max-mode \
    SPECTRAL_DIAGNOSTIC_FIT_SHELLS:--spectral-diagnostic-fit-shells \
    SPECTRAL_DIAGNOSTIC_RELATIVE_AMPLITUDE:--spectral-diagnostic-relative-amplitude \
    SPECTRAL_DIAGNOSTIC_FIELD_BATCH_SIZE:--spectral-diagnostic-field-batch-size \
    SPECTRAL_DIAGNOSTIC_DEPTH:--spectral-diagnostic-depth \
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
