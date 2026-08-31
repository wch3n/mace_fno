#!/usr/bin/env bash
#SBATCH --job-name=au-mgo-wetting
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --partition=debug-gpu
#SBATCH --account=htbase
#SBATCH --time=02:00:00
#SBATCH --mem=48G

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

MODEL_KIND=${MODEL_KIND:?set MODEL_KIND to mace-fno or mace}
MODEL=${MODEL:?set MODEL to the checkpoint or model path}
RUN_NAME=${RUN_NAME:?set a unique RUN_NAME}
STRUCTURES_DIR=${STRUCTURES_DIR:-${MACE_FNO_WORK_ROOT}/les_au_mgo/wetting_switch/inputs}
OUTPUT_ROOT=${OUTPUT_ROOT:-${MACE_FNO_WORK_ROOT}/les_au_mgo/wetting_switch/results}
OUTPUT_DIR=${OUTPUT_DIR:-${OUTPUT_ROOT}/${RUN_NAME}}
DTYPE=${DTYPE:-float64}
FMAX=${FMAX:-0.01}
STEPS=${STEPS:-500}

cd "${PROJECT_ROOT}"
mkdir -p "${OUTPUT_DIR}"

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="${PROJECT_ROOT}/src"
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export MPLCONFIGDIR="${TMPDIR:-/tmp}/mace-fno-matplotlib-${SLURM_JOB_ID}"
mkdir -p "${MPLCONFIGDIR}"

arguments=(
    --structures-dir "${STRUCTURES_DIR}"
    --model "${MODEL}"
    --model-kind "${MODEL_KIND}"
    --device cuda
    --dtype "${DTYPE}"
    --fmax "${FMAX}"
    --steps "${STEPS}"
    --output-dir "${OUTPUT_DIR}"
)
if [[ -n "${MACE_HEAD:-}" ]]; then
    arguments+=(--mace-head "${MACE_HEAD}")
fi

python3 -u -m mace_fno.cli.evaluate_adsorption_switch "${arguments[@]}"
