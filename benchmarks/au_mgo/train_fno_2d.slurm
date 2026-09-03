#!/usr/bin/env bash
#SBATCH --job-name=au-mgo-fno-2d
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --partition=debug-gpu
#SBATCH --account=htbase
#SBATCH --time=02:00:00
#SBATCH --mem=64G
#SBATCH --output=/tmp/au-mgo-fno-2d-%j.out

set -Eeuo pipefail

module --force purge
module load EasyBuild/2024a Python
ulimit -s unlimited

PROJECT_ROOT=${PROJECT_ROOT:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)}
source "${PROJECT_ROOT}/benchmarks/runtime_paths.sh"
DATA_ROOT=${PROJECT_ROOT}/data/les_au_mgo
MACE_MODEL=${MACE_MODEL:-${MACE_FNO_WORK_ROOT}/les_au_mgo/pretrained/Au2-MgO_r5.5_nl0_stagetwo.model}
VALIDATION_INDICES=${VALIDATION_INDICES:-${PROJECT_ROOT}/benchmarks/au_mgo/validation_indices_seed123.txt}
CACHE_ROOT=${CACHE_ROOT:-${MACE_FNO_WORK_ROOT}/les_au_mgo/cache}
FNO_CONFIG=${FNO_CONFIG:-${PROJECT_ROOT}/benchmarks/au_mgo/train_fno_2d.yaml}
FNO_SEED=${FNO_SEED:-17}
MODEL_DTYPE=${MODEL_DTYPE:-float64}
CHECKPOINT=${CHECKPOINT:-${MACE_FNO_WORK_ROOT}/les_au_mgo/les_au_mgo_fno_2d_seed${FNO_SEED}_${MODEL_DTYPE}.pt}

cd "${PROJECT_ROOT}"
mkdir -p "$(dirname "${CHECKPOINT}")" "${CACHE_ROOT}"
test -s "${FNO_CONFIG}"

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

# Loss scales are the frozen model's training-set RMSEs, rounded from
# 2.260 meV/atom and 56.280 meV/Angstrom. The official test set remains
# evaluation-only, and validation uses the MACE seed-123 indices.
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
    STEPS:--steps GRID:--grid MODES:--modes CHANNELS:--channels \
    SOURCE_HIDDEN_CHANNELS:--source-hidden-channels \
    FNO_HIDDEN_CHANNELS:--fno-hidden-channels FNO_LAYERS:--fno-layers \
    LEARNING_RATE:--learning-rate LR_SCHEDULER:--lr-scheduler \
    LR_DECAY_FACTOR:--lr-decay-factor LR_PATIENCE_EVALS:--lr-patience-evals \
    MINIMUM_LEARNING_RATE:--minimum-learning-rate \
    EARLY_STOPPING_PATIENCE_EVALS:--early-stopping-patience-evals \
    FORCE_WEIGHT:--force-weight EVAL_INTERVAL:--eval-interval \
    BATCH_SIZE:--batch-size EVALUATION_BATCH_SIZE:--evaluation-batch-size \
    ACCUMULATION_STEPS:--accumulation-steps \
    EVALUATION_SCOPE:--evaluation-scope; do
    mace_fno_append_env_option train_args "${override%%:*}" "${override#*:}"
done

python3 -u -m mace_fno.cli.train_residual "${train_args[@]}"
