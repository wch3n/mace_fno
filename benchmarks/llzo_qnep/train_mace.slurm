#!/usr/bin/env bash
#SBATCH --job-name=llzo-mace
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
NUM_INTERACTIONS=${NUM_INTERACTIONS:-1}
if (( NUM_INTERACTIONS < 1 )); then
    printf 'NUM_INTERACTIONS must be positive; got %s\n' "${NUM_INTERACTIONS}" >&2
    exit 2
fi
NL_TAG=$((NUM_INTERACTIONS - 1))
RUN_ROOT=${RUN_ROOT:-${MACE_FNO_WORK_ROOT}/llzo_qnep/mace_r4p5_nl${NL_TAG}}
MODEL_DIR=${MODEL_DIR:-${RUN_ROOT}/models}
JOB_WORK_ROOT=${JOB_WORK_ROOT:-${RUN_ROOT}/work}
MACE_SEED=${MACE_SEED:-17}
BATCH_SIZE=${BATCH_SIZE:-4}
MAX_EPOCHS=${MAX_EPOCHS:-100}
START_STAGE_TWO=${START_STAGE_TWO:-50}
RESTART_LATEST=${RESTART_LATEST:-0}

restart_args=()
case "${RESTART_LATEST}" in
    0|false|FALSE|no|NO) ;;
    1|true|TRUE|yes|YES) restart_args+=(--restart_latest) ;;
    *)
        printf 'RESTART_LATEST must be 0/1, true/false, or yes/no; got: %s\n' \
            "${RESTART_LATEST}" >&2
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

# NL_TAG follows the LES convention: nl0 has one message-passing interaction,
# while nl1 has two. Both use the same 4.5 A local cutoff and training protocol.
mace_run_train \
    --name="llzo-PBEsol-r4p5-nl${NL_TAG}" \
    --work_dir="${RUN_ROOT}" \
    --log_dir="${RUN_ROOT}/mace_logs" \
    --model_dir="${MODEL_DIR}" \
    --checkpoints_dir="${RUN_ROOT}/checkpoints" \
    --results_dir="${RUN_ROOT}/results" \
    --train_file="${DATA_ROOT}/train.xyz" \
    --valid_file="${DATA_ROOT}/validation.xyz" \
    --test_file="${DATA_ROOT}/test.xyz" \
    --energy_key="energy" \
    --forces_key="forces" \
    --E0s="average" \
    --model="MACE" \
    --hidden_irreps="128x0e" \
    --r_max=4.5 \
    --num_interactions="${NUM_INTERACTIONS}" \
    --batch_size="${BATCH_SIZE}" \
    --max_num_epochs="${MAX_EPOCHS}" \
    --stage_two \
    --start_stage_two="${START_STAGE_TWO}" \
    --ema \
    --ema_decay=0.99 \
    --amsgrad \
    "${restart_args[@]}" \
    --save_cpu \
    --default_dtype=float64 \
    --seed="${MACE_SEED}" \
    --launcher=none \
    --device=cuda

MODEL_PATH="${MODEL_DIR}/llzo-PBEsol-r4p5-nl${NL_TAG}_stagetwo.model"
test -s "${MODEL_PATH}"
sha256sum "${MODEL_PATH}"
