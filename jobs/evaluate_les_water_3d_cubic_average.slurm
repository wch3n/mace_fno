#!/usr/bin/env bash
#SBATCH --job-name=water-oh48
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --partition=debug-gpu
#SBATCH --account=htbase
#SBATCH --time=00:30:00
#SBATCH --mem=48G
#SBATCH --output=logs/eval-les-water-oh48-%j.out

set -Eeuo pipefail

module --force purge
module load EasyBuild/2024a Python
ulimit -s unlimited

PROJECT_ROOT=/gpfs/home/acad/ucl-modl/wchen/mxene_proj/mace_fno
CHECKPOINT=${CHECKPOINT:-artifacts/les_water/les_water_fno_3d_seed17_float64.pt}
OUTPUT=${OUTPUT:-artifacts/les_water/les_water_fno_3d_seed17_cubic_average.json}

cd "${PROJECT_ROOT}"
mkdir -p logs "$(dirname "${OUTPUT}")"

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="${PROJECT_ROOT}/src"
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export MPLCONFIGDIR="${TMPDIR:-/tmp}/mace-fno-oh48-matplotlib-${SLURM_JOB_ID}"
mkdir -p "${MPLCONFIGDIR}"

python3 -u examples/evaluate_les_water_3d_cubic_average.py \
    --checkpoint "${CHECKPOINT}" \
    --device cuda \
    --output "${OUTPUT}" \
    --strict
