#!/usr/bin/env bash
#SBATCH --job-name=au-mgo-mace
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH --account=htbase
#SBATCH --time=48:00:00
#SBATCH --mem=64G
#SBATCH --output=logs/au-mgo-mace-%j.out

set -Eeuo pipefail

module --force purge
module load EasyBuild/2024a Python
ulimit -s unlimited

PROJECT_ROOT=${PROJECT_ROOT:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)}
source "${PROJECT_ROOT}/benchmarks/runtime_paths.sh"
DATA_ROOT=${PROJECT_ROOT}/data/les_au_mgo
RUN_ROOT=${RUN_ROOT:-${MACE_FNO_WORK_ROOT}/les_au_mgo/mace_r5.5_nl1_l0}
MODEL_DIR=${RUN_ROOT}/models

cd "${PROJECT_ROOT}"
mkdir -p "${MODEL_DIR}" "${RUN_ROOT}/checkpoints" "${RUN_ROOT}/results" \
    "${RUN_ROOT}/mace_logs" logs

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export MPLCONFIGDIR="${TMPDIR:-/tmp}/mace-fno-matplotlib-${SLURM_JOB_ID}"
mkdir -p "${MPLCONFIGDIR}"

if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi
fi

# This reproduces the ordinary-MACE setup in
# les_fit/MLIPs/MACE-LES/Au-MgO-Al/mace-r5.5-nl-1-l-0/fit.sh using the
# currently installed MACE CLI. The official test split is evaluation-only.
mace_run_train \
    --name="Au2-MgO" \
    --work_dir="${RUN_ROOT}" \
    --log_dir="${RUN_ROOT}/mace_logs" \
    --model_dir="${MODEL_DIR}" \
    --checkpoints_dir="${RUN_ROOT}/checkpoints" \
    --results_dir="${RUN_ROOT}/results" \
    --train_file="${DATA_ROOT}/train-Au-MgO-Al.xyz" \
    --valid_fraction=0.05 \
    --test_file="${DATA_ROOT}/test-Au-MgO-Al.xyz" \
    --energy_key="energy" \
    --forces_key="forces" \
    --E0s="average" \
    --model="MACE" \
    --hidden_irreps="128x0e" \
    --r_max=5.5 \
    --num_interactions=2 \
    --batch_size=10 \
    --max_num_epochs=1000 \
    --stage_two \
    --start_stage_two=500 \
    --ema \
    --ema_decay=0.99 \
    --amsgrad \
    --restart_latest \
    --save_cpu \
    --default_dtype=float64 \
    --launcher=none \
    --device=cuda

MODEL_PATH=${MODEL_DIR}/Au2-MgO_stagetwo.model
test -s "${MODEL_PATH}"
sha256sum "${MODEL_PATH}"
