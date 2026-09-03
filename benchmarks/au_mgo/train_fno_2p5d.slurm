#!/usr/bin/env bash
#SBATCH --job-name=au-mgo-fno-2p5d
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --partition=debug-gpu
#SBATCH --account=htbase
#SBATCH --time=02:00:00
#SBATCH --mem=64G
#SBATCH --output=/tmp/au-mgo-fno-2p5d-%j.out

set -Eeuo pipefail

module --force purge
module load EasyBuild/2024a Python
ulimit -s unlimited

PROJECT_ROOT=${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-}}
if [[ -z "${PROJECT_ROOT}" || ! -f "${PROJECT_ROOT}/benchmarks/runtime_paths.sh" ]]; then
    echo "Submit from the repository root or export PROJECT_ROOT." >&2
    exit 2
fi
source "${PROJECT_ROOT}/benchmarks/runtime_paths.sh"
DATA_ROOT=${DATA_ROOT:-${PROJECT_ROOT}/data/les_au_mgo}
MACE_MODEL=${MACE_MODEL:-${MACE_FNO_WORK_ROOT}/les_au_mgo/pretrained/Au2-MgO_r5.5_nl0_stagetwo.model}
VALIDATION_INDICES=${VALIDATION_INDICES:-${PROJECT_ROOT}/benchmarks/au_mgo/validation_indices_seed123.txt}
CACHE_ROOT=${CACHE_ROOT:-${MACE_FNO_WORK_ROOT}/les_au_mgo/cache}
FNO_CONFIG=${FNO_CONFIG:-${PROJECT_ROOT}/benchmarks/au_mgo/train_fno_2p5d.yaml}
FNO_SEED=${FNO_SEED:-17}
MODEL_DTYPE=${MODEL_DTYPE:-float64}
CHECKPOINT=${CHECKPOINT:-${MACE_FNO_WORK_ROOT}/les_au_mgo/les_au_mgo_fno_2p5d_seed${FNO_SEED}_${MODEL_DTYPE}.pt}
JOB_WORK_ROOT=${JOB_WORK_ROOT:-$(dirname "${CHECKPOINT}")/work}

mkdir -p "$(dirname "${CHECKPOINT}")" "${CACHE_ROOT}" "${JOB_WORK_ROOT}"
test -s "${FNO_CONFIG}"
test -s "${MACE_MODEL}"
test -s "${DATA_ROOT}/train-Au-MgO-Al.xyz"
test -s "${DATA_ROOT}/test-Au-MgO-Al.xyz"
test -s "${VALIDATION_INDICES}"
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

# This is matched to the established 2D seed-17/29 benchmark except for the
# explicit finite z mesh and z mixer. Mean centering is the physical default:
# it makes the residual exactly invariant to a rigid translation of the whole
# slab along z. Cell centering remains available as an explicit diagnostic.
# A 22 A window with 16 layers contains all atoms with either convention.
train_args=(
    --config "${FNO_CONFIG}"
    --mace-model "${MACE_MODEL}"
    --train-file "${DATA_ROOT}/train-Au-MgO-Al.xyz"
    --validation-indices-file "${VALIDATION_INDICES}"
    --test-file "${DATA_ROOT}/test-Au-MgO-Al.xyz"
    --train-cache "${CACHE_ROOT}/train-${MODEL_DTYPE}.pt"
    --test-cache "${CACHE_ROOT}/test-${MODEL_DTYPE}.pt"
    --seed "${FNO_SEED}"
    --dtype "${MODEL_DTYPE}"
    --checkpoint "${CHECKPOINT}"
)

for override in \
    STEPS:--steps GRID:--grid MODES:--modes Z_GRID:--z-grid \
    Z_EXTENT:--z-extent Z_CENTER:--z-center \
    LATERAL_INTERLACING:--lateral-interlacing \
    PLANAR_SYMMETRY:--planar-symmetry Z_KERNEL_SIZE:--z-kernel-size \
    Z_MIXING:--z-mixing CHANNELS:--channels \
    SOURCE_HIDDEN_CHANNELS:--source-hidden-channels \
    FNO_HIDDEN_CHANNELS:--fno-hidden-channels FNO_LAYERS:--fno-layers \
    MACE_TRAINING:--mace-training \
    LEARNING_RATE:--learning-rate LR_SCHEDULER:--lr-scheduler \
    MACE_LEARNING_RATE:--mace-learning-rate \
    MACE_WARMUP_STEPS:--mace-warmup-steps \
    LR_DECAY_FACTOR:--lr-decay-factor LR_PATIENCE_EVALS:--lr-patience-evals \
    MINIMUM_LEARNING_RATE:--minimum-learning-rate \
    EARLY_STOPPING_PATIENCE_EVALS:--early-stopping-patience-evals \
    FORCE_WEIGHT:--force-weight EVAL_INTERVAL:--eval-interval \
    SPECTRAL_DIAGNOSTIC_SAMPLES:--spectral-diagnostic-samples \
    SPECTRAL_DIAGNOSTIC_MAX_MODE:--spectral-diagnostic-max-mode \
    SPECTRAL_DIAGNOSTIC_FIT_SHELLS:--spectral-diagnostic-fit-shells \
    SPECTRAL_DIAGNOSTIC_Z_PROFILES:--spectral-diagnostic-z-profiles \
    SPECTRAL_DIAGNOSTIC_FIELD_BATCH_SIZE:--spectral-diagnostic-field-batch-size \
    SPECTRAL_DIAGNOSTIC_DEPTH:--spectral-diagnostic-depth \
    BATCH_SIZE:--batch-size EVALUATION_BATCH_SIZE:--evaluation-batch-size \
    ACCUMULATION_STEPS:--accumulation-steps \
    EVALUATION_SCOPE:--evaluation-scope; do
    mace_fno_append_env_option train_args "${override%%:*}" "${override#*:}"
done

python3 -u -m mace_fno.cli.train_residual "${train_args[@]}"
