#!/usr/bin/env bash
#SBATCH --job-name=ti-fno-1331
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --partition=debug-gpu
#SBATCH --account=htbase
#SBATCH --time=02:00:00
#SBATCH --mem=64G
#SBATCH --output=logs/ti-fno-1331-%j.out

set -Eeuo pipefail

module --force purge
module load EasyBuild/2024a Python
ulimit -s unlimited

PROJECT_ROOT=/gpfs/home/acad/ucl-modl/wchen/mxene_proj/mace_fno
TI_ROOT=/gpfs/scratch/acad/htbase/wchen/mxene/mlip/pbe0-rvv10_8/Ti
MACE_MODEL=${TI_ROOT}/models/ft-omat_0-01_stagetwo.model
CACHE_ROOT=${CACHE_ROOT:-artifacts/cache/ti_1331}
FORCE_WEIGHT=${FORCE_WEIGHT:-1}
LEARNING_RATE=${LEARNING_RATE:-3e-4}
FNO_SEED=${FNO_SEED:-17}
BATCH_SIZE=${BATCH_SIZE:-4}
EVALUATION_BATCH_SIZE=${EVALUATION_BATCH_SIZE:-8}
ACCUMULATION_STEPS=${ACCUMULATION_STEPS:-1}
EVALUATION_SCOPE=${EVALUATION_SCOPE:-validation-test}
MODEL_DTYPE=${MODEL_DTYPE:-float64}
Z_GRID=${Z_GRID:-0}
# The selected 1331 cell family has max |z-mean(z)| ~= 6.86 A. Twenty
# angstrom leaves at least one cubic-assignment support layer at Z_GRID=12.
Z_EXTENT=${Z_EXTENT:-20.0}
Z_KERNEL_SIZE=${Z_KERNEL_SIZE:-3}
Z_MIXING=${Z_MIXING:-local}

SPATIAL_ARGUMENTS=()
if (( Z_GRID > 0 )); then
    SPATIAL_ARGUMENTS+=(
        --z-grid "${Z_GRID}"
        --z-extent "${Z_EXTENT}"
        --z-center mean
        --z-kernel-size "${Z_KERNEL_SIZE}"
        --z-mixing "${Z_MIXING}"
    )
    DEFAULT_CHECKPOINT=artifacts/ti_1331_fno_2p5d_seed17.pt
else
    DEFAULT_CHECKPOINT=artifacts/ti_1331_fno_seed17.pt
fi
CHECKPOINT=${CHECKPOINT:-${DEFAULT_CHECKPOINT}}

cd "${PROJECT_ROOT}"
mkdir -p artifacts "${CACHE_ROOT}" logs

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

python3 -u examples/train_mace_residual.py \
    --mace-model "${MACE_MODEL}" \
    --train-file "${TI_ROOT}/pbe0-rvv10_1331_train.xyz" \
    --test-file "${TI_ROOT}/pbe0-rvv10_1331_test.xyz" \
    --train-cache "${CACHE_ROOT}/train-${MODEL_DTYPE}.pt" \
    --test-cache "${CACHE_ROOT}/test-${MODEL_DTYPE}.pt" \
    --energy-key energy_dft \
    --forces-key forces_dft \
    --allow-periodic-z \
    --skip-cell-mismatch \
    --steps 1500 \
    --grid 24 \
    --modes 4 \
    "${SPATIAL_ARGUMENTS[@]}" \
    --channels 4 \
    --source-hidden-channels 64 \
    --fno-hidden-channels 16 \
    --fno-layers 2 \
    --architecture nonlinear \
    --learning-rate "${LEARNING_RATE}" \
    --energy-weight 1 \
    --force-weight "${FORCE_WEIGHT}" \
    --energy-scale 0.003 \
    --force-scale 0.06 \
    --validation-fraction 0.10 \
    --eval-interval 250 \
    --batch-size "${BATCH_SIZE}" \
    --evaluation-batch-size "${EVALUATION_BATCH_SIZE}" \
    --accumulation-steps "${ACCUMULATION_STEPS}" \
    --evaluation-scope "${EVALUATION_SCOPE}" \
    --seed "${FNO_SEED}" \
    --dtype "${MODEL_DTYPE}" \
    --device cuda \
    --checkpoint "${CHECKPOINT}"
