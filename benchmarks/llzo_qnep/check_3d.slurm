#!/usr/bin/env bash
#SBATCH --job-name=llzo-check
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --partition=debug-gpu
#SBATCH --account=htbase
#SBATCH --time=00:30:00
#SBATCH --mem=48G
# Scheduler output is supplied by submit.sh so it stays outside the checkout.

set -Eeuo pipefail

module --force purge
module load EasyBuild/2024a Python
ulimit -s unlimited

PROJECT_ROOT=${PROJECT_ROOT:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)}
CHECKPOINT=${CHECKPOINT:?Set CHECKPOINT to the trained MACE+FNO checkpoint}
CHECK_SAMPLES=${CHECK_SAMPLES:-8}
CHECK_FD_STEP=${CHECK_FD_STEP:-1.0e-3}
CHECK_FD_COMPONENTS=${CHECK_FD_COMPONENTS:-12}
CHECK_STRICT=${CHECK_STRICT:-1}
CHECK_OUTPUT=${CHECK_OUTPUT:-${CHECKPOINT%.pt}_check.json}
JOB_WORK_ROOT=${JOB_WORK_ROOT:-$(dirname "${CHECK_OUTPUT}")/work}

strict_args=()
case "${CHECK_STRICT}" in
    0|false|FALSE|no|NO) ;;
    1|true|TRUE|yes|YES) strict_args+=(--strict) ;;
    *)
        printf 'CHECK_STRICT must be 0/1, true/false, or yes/no; got: %s\n' \
            "${CHECK_STRICT}" >&2
        exit 2
        ;;
esac

mkdir -p "$(dirname "${CHECK_OUTPUT}")" "${JOB_WORK_ROOT}"
test -s "${CHECKPOINT}"
cd "${JOB_WORK_ROOT}"

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="${PROJECT_ROOT}/src"
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1

python3 -u -m mace_fno.cli.check_3d \
    --checkpoint "${CHECKPOINT}" \
    --samples "${CHECK_SAMPLES}" \
    --fd-step "${CHECK_FD_STEP}" \
    --fd-components "${CHECK_FD_COMPONENTS}" \
    --device cuda \
    --output "${CHECK_OUTPUT}" \
    "${strict_args[@]}"

test -s "${CHECK_OUTPUT}"
