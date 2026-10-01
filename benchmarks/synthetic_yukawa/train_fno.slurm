#!/usr/bin/env bash
#SBATCH --job-name=synthetic-fno
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH --account=htbase
#SBATCH --time=2-00:00:00
#SBATCH --mem=64G
# Train a frozen or joint MACE1-FNO on a trained MACE1 backbone, e.g.
#   sbatch --export=ALL,KAPPA=0.05,MODE=frozen benchmarks/synthetic_yukawa/train_fno.slurm
#   sbatch --export=ALL,KAPPA=0.5,MODE=joint,DATASET=md benchmarks/synthetic_yukawa/train_fno.slurm
# MODE=linear trains the linear control (frozen backbone, one linear spectral
# operator in place of the Fourier layers).
# The default backbone is the MACE1 run of train_mace.slurm with the same
# DATASET, KAPPA and MACE_SEED; set MACE_MODEL to use another one.
# Frozen runs (100k steps) take about 36 h on one A100. Joint runs (200k steps)
# reach 125-155k steps in 48 h: resubmit with RESUME=1 to continue from the
# saved training state.
# Architecture variants of Appendix B.2 use the overrides CHANNELS, FNO_LAYERS,
# MODES and GRID (MODES and GRID also set the z mesh), e.g. CHANNELS=16.
# Pass --output to keep the scheduler log outside the checkout.

set -Eeuo pipefail

module --force purge
module load EasyBuild/2024a Python CUDA
ulimit -s unlimited

PROJECT_ROOT=${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}
source "${PROJECT_ROOT}/benchmarks/runtime_paths.sh"
HERE=${PROJECT_ROOT}/benchmarks/synthetic_yukawa
DATA_ROOT=${DATA_ROOT:-${PROJECT_ROOT}/data/synthetic_yukawa}
DATASET=${DATASET:-static}
MODE=${MODE:-frozen}
KAPPA=${KAPPA:?Set KAPPA}
FNO_SEED=${FNO_SEED:-17}
MACE_SEED=${MACE_SEED:-17}
RESUME=${RESUME:-0}
tag=${KAPPA/./p}

case "${MODE}" in
    frozen|joint|linear) FNO_CONFIG=${FNO_CONFIG:-${HERE}/train_fno_${MODE}.yaml} ;;
    *) echo "MODE must be frozen, joint or linear" >&2; exit 2 ;;
esac
case "${KAPPA}" in
    0.05) energy_scale=0.65 ;;
    0.2)  energy_scale=0.50 ;;
    0.5)  energy_scale=0.40 ;;
    *) echo "KAPPA must be 0.05, 0.2 or 0.5" >&2; exit 2 ;;
esac
case "${DATASET}" in
    static)
        data=${DATA_ROOT}/static
        train=${data}/kappa${tag}_train.xyz
        mace_name=mace1_kappa${tag}
        spectral_samples=${SPECTRAL_DIAGNOSTIC_SAMPLES:-8}
        ;;
    md)
        [[ "${KAPPA}" == 0.05 || "${KAPPA}" == 0.5 ]] || { echo "md data exist for kappa 0.05 and 0.5" >&2; exit 2; }
        data=${DATA_ROOT}/md2800/data_k${tag}
        train=${data}/kappa${tag}_train_contact.xyz
        mace_name=mace1c_md2800_kappa${tag}
        spectral_samples=${SPECTRAL_DIAGNOSTIC_SAMPLES:-0}
        ;;
    *) echo "DATASET must be static or md" >&2; exit 2 ;;
esac
valid=${data}/kappa${tag}_valid.xyz
test_file=${data}/kappa${tag}_test.xyz
MACE_MODEL=${MACE_MODEL:-${MACE_FNO_WORK_ROOT}/synthetic_yukawa/${DATASET}/kappa_${tag}/mace1_seed${MACE_SEED}/models/${mace_name}_stagetwo.model}
RUN_ROOT=${RUN_ROOT:-${MACE_FNO_WORK_ROOT}/synthetic_yukawa/${DATASET}/kappa_${tag}/fno_${MODE}_seed${FNO_SEED}}
CHECKPOINT=${RUN_ROOT}/yukawa_fno_${MODE}_seed${FNO_SEED}_float64.pt
for f in "${FNO_CONFIG}" "${MACE_MODEL}" "${train}" "${valid}" "${test_file}"; do test -s "${f}"; done
mkdir -p "${RUN_ROOT}/cache"
cd "${RUN_ROOT}"

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="${PROJECT_ROOT}/src"
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1

train_args=(
    --config "${FNO_CONFIG}"
    --mace-model "${MACE_MODEL}"
    --train-file "${train}"
    --validation-file "${valid}"
    --test-file "${test_file}"
    --train-cache "${RUN_ROOT}/cache/train-float64.pt"
    --validation-cache "${RUN_ROOT}/cache/valid-float64.pt"
    --test-cache "${RUN_ROOT}/cache/test-float64.pt"
    --seed "${FNO_SEED}"
    --checkpoint "${CHECKPOINT}"
    --energy-scale "${energy_scale}"
    --spectral-diagnostic-samples "${spectral_samples}"
)
case "${RESUME}" in
    0|false|no) ;;
    1|true|yes) train_args+=(--resume "${CHECKPOINT%.pt}.last.pt") ;;
    *) echo "RESUME must be 0 or 1" >&2; exit 2 ;;
esac
for override in STEPS:--steps CHANNELS:--channels FNO_LAYERS:--fno-layers \
    MODES:--modes GRID:--grid; do
    mace_fno_append_env_option train_args "${override%%:*}" "${override#*:}"
done
if [[ -v GRID ]]; then train_args+=(--z-grid "${GRID}"); fi
if [[ -v MODES ]]; then train_args+=(--z-modes "${MODES}"); fi

python3 -u -m mace_fno.cli.train_residual "${train_args[@]}"
test -s "${CHECKPOINT}"
