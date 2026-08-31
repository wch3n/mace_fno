#!/usr/bin/env bash
#SBATCH --job-name=water-scan-mace
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --partition=debug-gpu
#SBATCH --account=htbase
#SBATCH --time=2:00:00
#SBATCH --mem=64G
#SBATCH --output=logs/water-scan-mace-%j.out

set -Eeuo pipefail

module --force purge
module load EasyBuild/2024a Python
ulimit -s unlimited

PROJECT_ROOT=${PROJECT_ROOT:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)}
source "${PROJECT_ROOT}/benchmarks/runtime_paths.sh"
DATA_ROOT=${DATA_ROOT:-${PROJECT_ROOT}/data/water_scan_qnep/prepared}
RUN_ROOT=${RUN_ROOT:-${MACE_FNO_WORK_ROOT}/water_scan_qnep/mace_r4p5_nl0_e100_s2e50}
MODEL_DIR=${MODEL_DIR:-${RUN_ROOT}/models}
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

cd "${PROJECT_ROOT}"
mkdir -p "${MODEL_DIR}" "${RUN_ROOT}/checkpoints" "${RUN_ROOT}/results" \
    "${RUN_ROOT}/mace_logs" logs

for split in train validation test; do
    test -s "${DATA_ROOT}/${split}.xyz"
done

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export MPLCONFIGDIR="${TMPDIR:-/tmp}/mace-fno-matplotlib-${SLURM_JOB_ID}"
mkdir -p "${MPLCONFIGDIR}"

if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi
fi

# The deliberately local baseline matches the liquid-water LES comparison:
# a 4.5 A cutoff and one message-passing interaction (nl-0 in LES naming).
mace_run_train \
    --name="water-SCAN-r4p5-nl0" \
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
    --num_interactions=1 \
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

MODEL_PATH="${MODEL_DIR}/water-SCAN-r4p5-nl0_stagetwo.model"
test -s "${MODEL_PATH}"
sha256sum "${MODEL_PATH}"
