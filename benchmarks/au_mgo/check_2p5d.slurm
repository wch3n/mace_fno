#!/usr/bin/env bash
#SBATCH --job-name=check-au-mgo-2p5d
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --partition=debug-gpu
#SBATCH --account=htbase
#SBATCH --time=00:20:00
#SBATCH --mem=32G
#SBATCH --output=/tmp/check-au-mgo-2p5d-%j.out

set -Eeuo pipefail

module --force purge
module load EasyBuild/2024a Python
ulimit -s unlimited

PROJECT_ROOT=${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-}}
if [[ -z "${PROJECT_ROOT}" || ! -f "${PROJECT_ROOT}/benchmarks/runtime_paths.sh" ]]; then
    echo "Submit from the repository root or export PROJECT_ROOT." >&2
    exit 2
fi
: "${CHECKPOINT:?Set CHECKPOINT to the trained 2.5D residual checkpoint}"
CHECK_SAMPLES=${CHECK_SAMPLES:-32}
CHECK_OUTPUT=${CHECK_OUTPUT:-${CHECKPOINT%.pt}_check.json}
JOB_WORK_ROOT=${JOB_WORK_ROOT:-$(dirname "${CHECK_OUTPUT}")/work}

mkdir -p "$(dirname "${CHECK_OUTPUT}")" "${JOB_WORK_ROOT}"
test -s "${CHECKPOINT}"
cd "${JOB_WORK_ROOT}"

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="${PROJECT_ROOT}/src"
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export MPLCONFIGDIR="${TMPDIR:-/tmp}/mace-fno-check-matplotlib-${SLURM_JOB_ID}"
mkdir -p "${MPLCONFIGDIR}"

python3 -u -m mace_fno.cli.check_2p5d \
    --checkpoint "${CHECKPOINT}" \
    --samples "${CHECK_SAMPLES}" \
    --fd-components 12 \
    --device cuda \
    --output "${CHECK_OUTPUT}" \
    --strict
