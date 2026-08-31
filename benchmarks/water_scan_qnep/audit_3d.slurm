#!/usr/bin/env bash
#SBATCH --job-name=audit-water-scan
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --partition=debug-gpu
#SBATCH --account=htbase
#SBATCH --time=00:30:00
#SBATCH --mem=48G
#SBATCH --output=logs/audit-water-scan-%j.out

set -Eeuo pipefail

module --force purge
module load EasyBuild/2024a Python
ulimit -s unlimited

PROJECT_ROOT=${PROJECT_ROOT:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)}
source "${PROJECT_ROOT}/benchmarks/runtime_paths.sh"
FNO_SEED=${FNO_SEED:-17}
MODEL_DTYPE=${MODEL_DTYPE:-float32}
CHECKPOINT=${CHECKPOINT:-${MACE_FNO_WORK_ROOT}/water_scan_qnep/fno_3d_eqgino/water_scan_fno_3d_seed${FNO_SEED}_${MODEL_DTYPE}.pt}
AUDIT_SAMPLES=${AUDIT_SAMPLES:-8}
AUDIT_FD_STEP=${AUDIT_FD_STEP:-1.0e-4}
AUDIT_FD_COMPONENTS=${AUDIT_FD_COMPONENTS:-12}
AUDIT_STRICT=${AUDIT_STRICT:-1}
AUDIT_OUTPUT=${AUDIT_OUTPUT:-${CHECKPOINT%.pt}_audit.json}

strict_args=()
case "${AUDIT_STRICT}" in
    0|false|FALSE|no|NO) ;;
    1|true|TRUE|yes|YES) strict_args+=(--strict) ;;
    *)
        printf 'AUDIT_STRICT must be 0/1, true/false, or yes/no; got: %s\n' \
            "${AUDIT_STRICT}" >&2
        exit 2
        ;;
esac

cd "${PROJECT_ROOT}"
mkdir -p logs "$(dirname "${AUDIT_OUTPUT}")"
test -s "${CHECKPOINT}"

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="${PROJECT_ROOT}/src"
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export MPLCONFIGDIR="${TMPDIR:-/tmp}/mace-fno-audit-${SLURM_JOB_ID}"
mkdir -p "${MPLCONFIGDIR}"

python3 -u -m mace_fno.cli.audit_3d \
    --checkpoint "${CHECKPOINT}" \
    --samples "${AUDIT_SAMPLES}" \
    --fd-step "${AUDIT_FD_STEP}" \
    --fd-components "${AUDIT_FD_COMPONENTS}" \
    --device cuda \
    --output "${AUDIT_OUTPUT}" \
    "${strict_args[@]}"
