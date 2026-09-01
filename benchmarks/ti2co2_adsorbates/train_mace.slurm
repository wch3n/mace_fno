#!/usr/bin/env bash
#SBATCH --job-name=ti2co2-mace
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
: "${RUN_ROOT:?Set RUN_ROOT to an external MACE run directory}"
: "${NUM_INTERACTIONS:?Set NUM_INTERACTIONS to 1 or 2}"
if [[ "${NUM_INTERACTIONS}" != "1" && "${NUM_INTERACTIONS}" != "2" ]]; then
    printf 'NUM_INTERACTIONS must be 1 or 2; got %s\n' "${NUM_INTERACTIONS}" >&2
    exit 2
fi

DATA_ROOT=${DATA_ROOT:-${MACE_FNO_WORK_ROOT}/ti2co2_adsorbates/data/prepared}
MODEL_TAG=${MODEL_TAG:-ti2co2-r4p5-ni${NUM_INTERACTIONS}}
MODEL_DIR=${MODEL_DIR:-${RUN_ROOT}/models}
JOB_WORK_ROOT=${JOB_WORK_ROOT:-${RUN_ROOT}/work}
MACE_SEED=${MACE_SEED:-17}
BATCH_SIZE=${BATCH_SIZE:-16}
MAX_EPOCHS=${MAX_EPOCHS:-600}
START_STAGE_TWO=${START_STAGE_TWO:-400}
RESTART_LATEST=${RESTART_LATEST:-0}
ENABLE_CUEQ=${ENABLE_CUEQ:-1}

restart_args=()
case "${RESTART_LATEST}" in
    0|false|FALSE|no|NO) ;;
    1|true|TRUE|yes|YES) restart_args+=(--restart_latest) ;;
    *)
        printf 'RESTART_LATEST must be boolean; got %s\n' "${RESTART_LATEST}" >&2
        exit 2
        ;;
esac
cueq_args=()
case "${ENABLE_CUEQ}" in
    0|false|FALSE|no|NO) ;;
    1|true|TRUE|yes|YES) cueq_args+=(--enable_cueq=true) ;;
    *)
        printf 'ENABLE_CUEQ must be boolean; got %s\n' "${ENABLE_CUEQ}" >&2
        exit 2
        ;;
esac

mkdir -p "${MODEL_DIR}" "${RUN_ROOT}/checkpoints" "${RUN_ROOT}/results" \
    "${RUN_ROOT}/mace_logs" "${JOB_WORK_ROOT}"
for split in train validation test; do
    test -s "${DATA_ROOT}/${split}.xyz"
done
cd "${JOB_WORK_ROOT}"

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export MPLCONFIGDIR="${TMPDIR:-/tmp}/mace-fno-matplotlib-${SLURM_JOB_ID}"
mkdir -p "${MPLCONFIGDIR}"

if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi
fi

# Both controls use exactly the same scalar hidden width, cutoff, data split,
# loss, and optimization schedule. Only the number of local interactions differs.
mace_run_train \
    --name="${MODEL_TAG}" \
    --work_dir="${RUN_ROOT}" \
    --log_dir="${RUN_ROOT}/mace_logs" \
    --model_dir="${MODEL_DIR}" \
    --checkpoints_dir="${RUN_ROOT}/checkpoints" \
    --results_dir="${RUN_ROOT}/results" \
    --train_file="${DATA_ROOT}/train.xyz" \
    --valid_file="${DATA_ROOT}/validation.xyz" \
    --test_file="${DATA_ROOT}/test.xyz" \
    --energy_key=energy_dft \
    --forces_key=forces_dft \
    --model=MACE \
    --hidden_irreps=128x0e \
    --r_max=4.5 \
    --num_interactions="${NUM_INTERACTIONS}" \
    --correlation=3 \
    --energy_weight=1 \
    --forces_weight=100 \
    --swa_energy_weight=1000 \
    --swa_forces_weight=100 \
    --lr=0.005 \
    --swa_lr=0.0005 \
    --batch_size="${BATCH_SIZE}" \
    --num_workers=0 \
    --max_num_epochs="${MAX_EPOCHS}" \
    --stage_two \
    --start_stage_two="${START_STAGE_TWO}" \
    --scheduler_patience=30 \
    --patience=200 \
    --ema \
    --ema_decay=0.995 \
    --amsgrad \
    "${cueq_args[@]}" \
    "${restart_args[@]}" \
    --save_cpu \
    --default_dtype=float64 \
    --seed="${MACE_SEED}" \
    --launcher=none \
    --device=cuda

MODEL_PATH="${MODEL_DIR}/${MODEL_TAG}_stagetwo.model"
test -s "${MODEL_PATH}"
sha256sum "${MODEL_PATH}"
