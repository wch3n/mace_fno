#!/usr/bin/env bash
#SBATCH --job-name=ti2co2-fno
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --partition=debug-gpu
#SBATCH --account=htbase
#SBATCH --time=02:00:00
#SBATCH --mem=64G
# Scheduler output is supplied by submit.sh and remains outside the checkout.

set -Eeuo pipefail

module --force purge
module load EasyBuild/2024a Python
ulimit -s unlimited

PROJECT_ROOT=${PROJECT_ROOT:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)}
source "${PROJECT_ROOT}/benchmarks/runtime_paths.sh"
: "${RUN_ROOT:?Set RUN_ROOT to an external FNO run directory}"
: "${MACE_MODEL:?Set MACE_MODEL to the one-interaction checkpoint}"
: "${BASELINE_JSON:?Set BASELINE_JSON to its completed baseline evaluation}"
DATA_ROOT=${DATA_ROOT:-${MACE_FNO_WORK_ROOT}/ti2co2_adsorbates/data/prepared}
CACHE_ROOT=${CACHE_ROOT:-${RUN_ROOT}/cache}
JOB_WORK_ROOT=${JOB_WORK_ROOT:-${RUN_ROOT}/work}
MODEL_DTYPE=${MODEL_DTYPE:-float64}
FNO_SEED=${FNO_SEED:-17}
GRID=${GRID:-24}
MODES=${MODES:-4}
Z_GRID=${Z_GRID:-16}
Z_EXTENT=${Z_EXTENT:-16.0}
CHANNELS=${CHANNELS:-4}
SOURCE_HIDDEN_CHANNELS=${SOURCE_HIDDEN_CHANNELS:-64}
FNO_HIDDEN_CHANNELS=${FNO_HIDDEN_CHANNELS:-16}
FNO_LAYERS=${FNO_LAYERS:-2}
BATCH_SIZE=${BATCH_SIZE:-4}
EVALUATION_BATCH_SIZE=${EVALUATION_BATCH_SIZE:-8}
ACCUMULATION_STEPS=${ACCUMULATION_STEPS:-1}
STEPS=${STEPS:-20000}
EVAL_INTERVAL=${EVAL_INTERVAL:-500}
LEARNING_RATE=${LEARNING_RATE:-3e-4}
OUTPUT_INITIALIZATION_SCALE=${OUTPUT_INITIALIZATION_SCALE:-0.1}
FORCE_WEIGHT=${FORCE_WEIGHT:-1}
SPECTRAL_DIAGNOSTIC_SAMPLES=${SPECTRAL_DIAGNOSTIC_SAMPLES:-4}
CHECKPOINT=${CHECKPOINT:-${RUN_ROOT}/ti2co2_fno_2d_seed${FNO_SEED}_${MODEL_DTYPE}.pt}
SPECTRAL_DIAGNOSTIC_OUTPUT=${SPECTRAL_DIAGNOSTIC_OUTPUT:-${CHECKPOINT%.pt}_spectral_training.json}

mkdir -p "${RUN_ROOT}" "${CACHE_ROOT}" "${JOB_WORK_ROOT}"
test -s "${MACE_MODEL}"
test -s "${BASELINE_JSON}"
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

# Normalize the residual loss by the frozen baseline's training RMSE. The JSON
# is produced by evaluate_mace.slurm and therefore cannot use validation/test
# information to set these scales.
scale_values=$(python3 -c 'import json,sys; d=json.load(open(sys.argv[1])); print(d["train"]["overall"]["energy_eV_per_atom"]["rmse"], d["train"]["overall"]["forces_eV_per_A"]["rmse"])' "${BASELINE_JSON}")
read -r ENERGY_SCALE FORCE_SCALE <<<"${scale_values}"
python3 -c 'import sys; e,f=map(float,sys.argv[1:]); assert e>0 and f>0, (e,f)' "${ENERGY_SCALE}" "${FORCE_SCALE}"
printf 'Residual scales from frozen-MACE training errors: E=%s eV/atom F=%s eV/A\n' \
    "${ENERGY_SCALE}" "${FORCE_SCALE}"

spectral_args=()
if (( SPECTRAL_DIAGNOSTIC_SAMPLES > 0 )); then
    spectral_args+=(
        --spectral-diagnostic-samples "${SPECTRAL_DIAGNOSTIC_SAMPLES}"
        --spectral-diagnostic-max-mode 2
        --spectral-diagnostic-fit-shells 3
        --spectral-diagnostic-relative-amplitude 0.05
        --spectral-diagnostic-field-batch-size 32
        --spectral-diagnostic-z-profiles 3
        --spectral-diagnostic-depth fast
        --spectral-diagnostic-output "${SPECTRAL_DIAGNOSTIC_OUTPUT}"
    )
fi

# Manuscript name: 2D FNO. Configuration value 2.5d denotes the implementation
# with an in-plane FFT and an explicit finite, nonperiodic z mesh.
python3 -u -m mace_fno.cli.train_residual \
    --mace-model "${MACE_MODEL}" \
    --train-file "${DATA_ROOT}/train.xyz" \
    --validation-file "${DATA_ROOT}/validation.xyz" \
    --test-file "${DATA_ROOT}/test.xyz" \
    --train-cache "${CACHE_ROOT}/train-${MODEL_DTYPE}.pt" \
    --validation-cache "${CACHE_ROOT}/validation-${MODEL_DTYPE}.pt" \
    --test-cache "${CACHE_ROOT}/test-${MODEL_DTYPE}.pt" \
    --energy-key energy_dft \
    --forces-key forces_dft \
    --allow-periodic-z \
    --spatial-scheme 2.5d \
    --grid "${GRID}" \
    --modes "${MODES}" \
    --z-grid "${Z_GRID}" \
    --z-extent "${Z_EXTENT}" \
    --z-center mean \
    --z-mixing global \
    --z-kernel-size 3 \
    --lateral-interlacing 1 \
    --planar-symmetry none \
    --channels "${CHANNELS}" \
    --source-hidden-channels "${SOURCE_HIDDEN_CHANNELS}" \
    --fno-hidden-channels "${FNO_HIDDEN_CHANNELS}" \
    --fno-layers "${FNO_LAYERS}" \
    --architecture nonlinear \
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
    --eval-interval "${EVAL_INTERVAL}" \
    "${spectral_args[@]}" \
    --batch-size "${BATCH_SIZE}" \
    --evaluation-batch-size "${EVALUATION_BATCH_SIZE}" \
    --accumulation-steps "${ACCUMULATION_STEPS}" \
    --evaluation-scope validation-test \
    --seed "${FNO_SEED}" \
    --dtype "${MODEL_DTYPE}" \
    --device cuda \
    --checkpoint "${CHECKPOINT}"

test -s "${CHECKPOINT}"
