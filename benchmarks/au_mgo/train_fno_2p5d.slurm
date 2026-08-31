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
FORCE_WEIGHT=${FORCE_WEIGHT:-1}
LEARNING_RATE=${LEARNING_RATE:-3e-4}
FNO_SEED=${FNO_SEED:-17}
BATCH_SIZE=${BATCH_SIZE:-4}
EVALUATION_BATCH_SIZE=${EVALUATION_BATCH_SIZE:-8}
ACCUMULATION_STEPS=${ACCUMULATION_STEPS:-1}
EVALUATION_SCOPE=${EVALUATION_SCOPE:-validation-test}
MODEL_DTYPE=${MODEL_DTYPE:-float64}
Z_GRID=${Z_GRID:-16}
GRID=${GRID:-24}
MODES=${MODES:-4}
CHANNELS=${CHANNELS:-4}
SOURCE_HIDDEN_CHANNELS=${SOURCE_HIDDEN_CHANNELS:-64}
FNO_HIDDEN_CHANNELS=${FNO_HIDDEN_CHANNELS:-16}
FNO_LAYERS=${FNO_LAYERS:-2}
Z_EXTENT=${Z_EXTENT:-22.0}
Z_CENTER=${Z_CENTER:-mean}
LATERAL_INTERLACING=${LATERAL_INTERLACING:-1}
PLANAR_SYMMETRY=${PLANAR_SYMMETRY:-none}
Z_KERNEL_SIZE=${Z_KERNEL_SIZE:-3}
Z_MIXING=${Z_MIXING:-global}
STEPS=${STEPS:-1500}
EVAL_INTERVAL=${EVAL_INTERVAL:-250}
SPECTRAL_DIAGNOSTIC_SAMPLES=${SPECTRAL_DIAGNOSTIC_SAMPLES:-0}
SPECTRAL_DIAGNOSTIC_MAX_MODE=${SPECTRAL_DIAGNOSTIC_MAX_MODE:-1}
SPECTRAL_DIAGNOSTIC_FIT_SHELLS=${SPECTRAL_DIAGNOSTIC_FIT_SHELLS:-2}
SPECTRAL_DIAGNOSTIC_Z_PROFILES=${SPECTRAL_DIAGNOSTIC_Z_PROFILES:-3}
SPECTRAL_DIAGNOSTIC_FIELD_BATCH_SIZE=${SPECTRAL_DIAGNOSTIC_FIELD_BATCH_SIZE:-32}
SPECTRAL_DIAGNOSTIC_DEPTH=${SPECTRAL_DIAGNOSTIC_DEPTH:-fast}
LR_SCHEDULER=${LR_SCHEDULER:-none}
LR_DECAY_FACTOR=${LR_DECAY_FACTOR:-0.5}
LR_PATIENCE_EVALS=${LR_PATIENCE_EVALS:-4}
MINIMUM_LEARNING_RATE=${MINIMUM_LEARNING_RATE:-1e-6}
EARLY_STOPPING_PATIENCE_EVALS=${EARLY_STOPPING_PATIENCE_EVALS:-0}
INTERLACING_SUFFIX=""
if (( LATERAL_INTERLACING > 1 )); then
    INTERLACING_SUFFIX="_i${LATERAL_INTERLACING}"
fi
SYMMETRY_SUFFIX=""
if [[ "${PLANAR_SYMMETRY}" != "none" ]]; then
    SYMMETRY_SUFFIX="_${PLANAR_SYMMETRY}"
fi
CHECKPOINT=${CHECKPOINT:-${MACE_FNO_WORK_ROOT}/les_au_mgo/les_au_mgo_fno_2p5d_${Z_MIXING}_${Z_CENTER}${INTERLACING_SUFFIX}${SYMMETRY_SUFFIX}_g${GRID}_m${MODES}_z${Z_GRID}_c${CHANNELS}_h${FNO_HIDDEN_CHANNELS}_l${FNO_LAYERS}_seed${FNO_SEED}_${MODEL_DTYPE}.pt}

cd "${PROJECT_ROOT}"
mkdir -p "$(dirname "${CHECKPOINT}")" "${CACHE_ROOT}"

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="${PROJECT_ROOT}/src"
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export MPLCONFIGDIR="${TMPDIR:-/tmp}/mace-fno-matplotlib-${SLURM_JOB_ID}"
mkdir -p "${MPLCONFIGDIR}"

SPECTRAL_DIAGNOSTIC_ARGS=()
if (( SPECTRAL_DIAGNOSTIC_SAMPLES > 0 )); then
    if (( LATERAL_INTERLACING != 1 )); then
        echo "The 2.5D spectral diagnostic requires LATERAL_INTERLACING=1" >&2
        exit 2
    fi
    SPECTRAL_DIAGNOSTIC_OUTPUT=${SPECTRAL_DIAGNOSTIC_OUTPUT:-${CHECKPOINT%.pt}_spectral_training.json}
    SPECTRAL_DIAGNOSTIC_ARGS=(
        --spectral-diagnostic-samples "${SPECTRAL_DIAGNOSTIC_SAMPLES}"
        --spectral-diagnostic-max-mode "${SPECTRAL_DIAGNOSTIC_MAX_MODE}"
        --spectral-diagnostic-fit-shells "${SPECTRAL_DIAGNOSTIC_FIT_SHELLS}"
        --spectral-diagnostic-z-profiles "${SPECTRAL_DIAGNOSTIC_Z_PROFILES}"
        --spectral-diagnostic-field-batch-size "${SPECTRAL_DIAGNOSTIC_FIELD_BATCH_SIZE}"
        --spectral-diagnostic-depth "${SPECTRAL_DIAGNOSTIC_DEPTH}"
        --spectral-diagnostic-output "${SPECTRAL_DIAGNOSTIC_OUTPUT}"
    )
fi

if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi
fi

# This is matched to the established 2D seed-17/29 benchmark except for the
# explicit finite z mesh and z mixer. Mean centring is the physical default:
# it makes the residual exactly invariant to a rigid translation of the whole
# slab along z. Cell centring remains available as an explicit diagnostic.
# A 22 A window with 16 layers contains all atoms with either convention.
python3 -u -m mace_fno.cli.train_residual \
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
    --grid "${GRID}" \
    --modes "${MODES}" \
    --z-grid "${Z_GRID}" \
    --z-extent "${Z_EXTENT}" \
    --z-center "${Z_CENTER}" \
    --lateral-interlacing "${LATERAL_INTERLACING}" \
    --planar-symmetry "${PLANAR_SYMMETRY}" \
    --z-kernel-size "${Z_KERNEL_SIZE}" \
    --z-mixing "${Z_MIXING}" \
    --channels "${CHANNELS}" \
    --source-hidden-channels "${SOURCE_HIDDEN_CHANNELS}" \
    --fno-hidden-channels "${FNO_HIDDEN_CHANNELS}" \
    --fno-layers "${FNO_LAYERS}" \
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
    "${SPECTRAL_DIAGNOSTIC_ARGS[@]}" \
    --batch-size "${BATCH_SIZE}" \
    --evaluation-batch-size "${EVALUATION_BATCH_SIZE}" \
    --accumulation-steps "${ACCUMULATION_STEPS}" \
    --evaluation-scope "${EVALUATION_SCOPE}" \
    --seed "${FNO_SEED}" \
    --dtype "${MODEL_DTYPE}" \
    --device cuda \
    --checkpoint "${CHECKPOINT}"
