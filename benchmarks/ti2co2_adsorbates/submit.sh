#!/usr/bin/env bash
# Prepare the fixed-cell dataset and submit the controlled MACE/FNO comparison.

set -Eeuo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)}
source "${PROJECT_ROOT}/benchmarks/runtime_paths.sh"

SOURCE_ROOT=${SOURCE_ROOT:-/gpfs/home/acad/ucl-modl/wchen/scratch/mxene/mlip/pbe0-rvv10_8/Ti}
DATA_ROOT=${DATA_ROOT:-${MACE_FNO_WORK_ROOT}/ti2co2_adsorbates/data/prepared}
RUN_ID=${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}
EXPERIMENT_ROOT=${EXPERIMENT_ROOT:-${MACE_FNO_WORK_ROOT}/ti2co2_adsorbates/runs/${RUN_ID}}
LOG_ROOT=${LOG_ROOT:-${EXPERIMENT_ROOT}/logs}
JOB_WORK_ROOT=${JOB_WORK_ROOT:-${EXPERIMENT_ROOT}/work}
MACE_SEED=${MACE_SEED:-17}
FNO_SEED=${FNO_SEED:-17}
MODEL_DTYPE=${MODEL_DTYPE:-float64}

mkdir -p "${EXPERIMENT_ROOT}" "${LOG_ROOT}" "${JOB_WORK_ROOT}"
python3 "${PROJECT_ROOT}/benchmarks/ti2co2_adsorbates/prepare_dataset.py" \
    --source-root "${SOURCE_ROOT}" \
    --output-root "${DATA_ROOT}" \
    --seed 17 \
    --validation-fraction 0.05 \
    --test-fraction 0.10

common_sbatch=(
    --parsable
    --chdir="${JOB_WORK_ROOT}"
    --output="${LOG_ROOT}/%x-%j.out"
)

declare -A mace_jobs eval_jobs models baselines
for interactions in 1 2; do
    tag="ni${interactions}"
    run_root="${EXPERIMENT_ROOT}/mace_${tag}"
    model_tag="ti2co2-r4p5-${tag}"
    models[${tag}]="${run_root}/models/${model_tag}_stagetwo.model"
    baselines[${tag}]="${run_root}/baseline.json"
    mace_jobs[${tag}]=$(sbatch "${common_sbatch[@]}" \
        --job-name="ti-mace-${tag}" \
        --export="ALL,PROJECT_ROOT=${PROJECT_ROOT},DATA_ROOT=${DATA_ROOT},RUN_ROOT=${run_root},JOB_WORK_ROOT=${JOB_WORK_ROOT},NUM_INTERACTIONS=${interactions},MODEL_TAG=${model_tag},MACE_SEED=${MACE_SEED}" \
        "${PROJECT_ROOT}/benchmarks/ti2co2_adsorbates/train_mace.slurm")
    eval_jobs[${tag}]=$(sbatch "${common_sbatch[@]}" \
        --job-name="ti-eval-${tag}" \
        --dependency="afterok:${mace_jobs[${tag}]}" \
        --export="ALL,PROJECT_ROOT=${PROJECT_ROOT},DATA_ROOT=${DATA_ROOT},MACE_MODEL=${models[${tag}]},RESULT_PATH=${baselines[${tag}]},JOB_WORK_ROOT=${JOB_WORK_ROOT}" \
        "${PROJECT_ROOT}/benchmarks/ti2co2_adsorbates/evaluate_mace.slurm")
done

FNO_RUN_ROOT=${EXPERIMENT_ROOT}/fno_on_ni1
CHECKPOINT=${FNO_RUN_ROOT}/ti2co2_fno_2d_seed${FNO_SEED}_${MODEL_DTYPE}.pt
fno_job=$(sbatch "${common_sbatch[@]}" \
    --job-name=ti-fno-ni1 \
    --dependency="afterok:${eval_jobs[ni1]}" \
    --export="ALL,PROJECT_ROOT=${PROJECT_ROOT},DATA_ROOT=${DATA_ROOT},RUN_ROOT=${FNO_RUN_ROOT},MACE_MODEL=${models[ni1]},BASELINE_JSON=${baselines[ni1]},CHECKPOINT=${CHECKPOINT},JOB_WORK_ROOT=${JOB_WORK_ROOT},FNO_SEED=${FNO_SEED},MODEL_DTYPE=${MODEL_DTYPE}" \
    "${PROJECT_ROOT}/benchmarks/ti2co2_adsorbates/train_fno_2d.slurm")
audit_job=$(sbatch "${common_sbatch[@]}" \
    --job-name=ti-audit-fno \
    --dependency="afterok:${fno_job}" \
    --export="ALL,PROJECT_ROOT=${PROJECT_ROOT},CHECKPOINT=${CHECKPOINT},AUDIT_OUTPUT=${FNO_RUN_ROOT}/audit.json,JOB_WORK_ROOT=${JOB_WORK_ROOT}" \
    "${PROJECT_ROOT}/benchmarks/ti2co2_adsorbates/audit_fno_2d.slurm")
spectral_job=$(sbatch "${common_sbatch[@]}" \
    --job-name=ti-spectrum-fno \
    --dependency="afterok:${fno_job}" \
    --export="ALL,PROJECT_ROOT=${PROJECT_ROOT},CHECKPOINT=${CHECKPOINT},SPECTRUM_OUTPUT=${FNO_RUN_ROOT}/spectral_response.json,JOB_WORK_ROOT=${JOB_WORK_ROOT}" \
    "${PROJECT_ROOT}/benchmarks/ti2co2_adsorbates/audit_spectral.slurm")

printf 'Experiment root:       %s\n' "${EXPERIMENT_ROOT}"
printf 'One-interaction MACE:  %s; evaluation %s\n' "${mace_jobs[ni1]}" "${eval_jobs[ni1]}"
printf 'Two-interaction MACE:  %s; evaluation %s\n' "${mace_jobs[ni2]}" "${eval_jobs[ni2]}"
printf 'FNO on one-interaction MACE: %s\n' "${fno_job}"
printf 'FNO force audit:       %s\n' "${audit_job}"
printf 'FNO spectral audit:    %s\n' "${spectral_job}"
