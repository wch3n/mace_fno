#!/usr/bin/env bash
# Prepare Water-SCAN data and submit an externally isolated validation chain.

set -Eeuo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)}
source "${PROJECT_ROOT}/benchmarks/runtime_paths.sh"

RUN_ID=${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}
EXPERIMENT_ROOT=${EXPERIMENT_ROOT:-${MACE_FNO_WORK_ROOT}/water_scan_qnep/runs/${RUN_ID}}
SOURCE_DATA_ROOT=${SOURCE_DATA_ROOT:-${MACE_FNO_WORK_ROOT}/water_scan_qnep/data}
DATA_ROOT=${DATA_ROOT:-${SOURCE_DATA_ROOT}/prepared}
MACE_RUN_ROOT=${MACE_RUN_ROOT:-${EXPERIMENT_ROOT}/mace}
FNO_RUN_ROOT=${FNO_RUN_ROOT:-${EXPERIMENT_ROOT}/fno}
JOB_WORK_ROOT=${JOB_WORK_ROOT:-${EXPERIMENT_ROOT}/work}
LOG_ROOT=${LOG_ROOT:-${EXPERIMENT_ROOT}/logs}
FNO_SEED=${FNO_SEED:-17}
MODEL_DTYPE=${MODEL_DTYPE:-float32}
CHECKPOINT=${CHECKPOINT:-${FNO_RUN_ROOT}/water_scan_fno_3d_seed${FNO_SEED}_${MODEL_DTYPE}.pt}

mkdir -p "${EXPERIMENT_ROOT}" "${JOB_WORK_ROOT}" "${LOG_ROOT}"
cd "${JOB_WORK_ROOT}"
python3 "${PROJECT_ROOT}/benchmarks/water_scan_qnep/prepare_dataset.py" \
    --data-root "${SOURCE_DATA_ROOT}" \
    --download

common_sbatch=(
    --parsable
    --chdir="${JOB_WORK_ROOT}"
    --output="${LOG_ROOT}/%x-%j.out"
)

if [[ -n "${PRETRAINED_MACE_MODEL:-}" ]]; then
    MACE_MODEL=${PRETRAINED_MACE_MODEL}
    test -s "${MACE_MODEL}"
    mace_job="reused ${MACE_MODEL}"
    dependency=()
else
    MACE_MODEL=${MACE_RUN_ROOT}/models/water-SCAN-r4p5-nl0_stagetwo.model
    mace_job=$(sbatch "${common_sbatch[@]}" \
        --export="ALL,PROJECT_ROOT=${PROJECT_ROOT},DATA_ROOT=${DATA_ROOT},RUN_ROOT=${MACE_RUN_ROOT},JOB_WORK_ROOT=${JOB_WORK_ROOT}" \
        "${PROJECT_ROOT}/benchmarks/water_scan_qnep/train_mace.slurm")
    dependency=(--dependency="afterok:${mace_job}")
fi

baseline_job=$(sbatch "${common_sbatch[@]}" "${dependency[@]}" \
    --export="ALL,PROJECT_ROOT=${PROJECT_ROOT},DATA_ROOT=${DATA_ROOT},MACE_MODEL=${MACE_MODEL},RESULT_PATH=${EXPERIMENT_ROOT}/baseline.json,JOB_WORK_ROOT=${JOB_WORK_ROOT}" \
    "${PROJECT_ROOT}/benchmarks/water_scan_qnep/evaluate_mace.slurm")
fno_job=$(sbatch "${common_sbatch[@]}" "${dependency[@]}" \
    --export="ALL,PROJECT_ROOT=${PROJECT_ROOT},DATA_ROOT=${DATA_ROOT},RUN_ROOT=${FNO_RUN_ROOT},MACE_MODEL=${MACE_MODEL},CHECKPOINT=${CHECKPOINT},JOB_WORK_ROOT=${JOB_WORK_ROOT},FNO_SEED=${FNO_SEED},MODEL_DTYPE=${MODEL_DTYPE}" \
    "${PROJECT_ROOT}/benchmarks/water_scan_qnep/train_fno_3d.slurm")
audit_job=$(sbatch "${common_sbatch[@]}" --dependency="afterok:${fno_job}" \
    --export="ALL,PROJECT_ROOT=${PROJECT_ROOT},CHECKPOINT=${CHECKPOINT},JOB_WORK_ROOT=${JOB_WORK_ROOT},FNO_SEED=${FNO_SEED},MODEL_DTYPE=${MODEL_DTYPE}" \
    "${PROJECT_ROOT}/benchmarks/water_scan_qnep/audit_3d.slurm")
spectral_job=$(sbatch "${common_sbatch[@]}" --dependency="afterok:${fno_job}" \
    --export="ALL,PROJECT_ROOT=${PROJECT_ROOT},CHECKPOINT=${CHECKPOINT},JOB_WORK_ROOT=${JOB_WORK_ROOT},FNO_SEED=${FNO_SEED},MODEL_DTYPE=${MODEL_DTYPE}" \
    "${PROJECT_ROOT}/benchmarks/water_scan_qnep/audit_spectral.slurm")

printf 'Experiment root:    %s\n' "${EXPERIMENT_ROOT}"
printf 'MACE training:      %s\n' "${mace_job}"
printf 'MACE validation:    %s\n' "${baseline_job}"
printf 'MACE+FNO training:  %s\n' "${fno_job}"
printf 'MACE+FNO audit:     %s\n' "${audit_job}"
printf 'Spectral audit:     %s\n' "${spectral_job}"
