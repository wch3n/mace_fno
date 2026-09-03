#!/usr/bin/env bash
# Submit matched metric-aware EqGINO optimization pilots on LES liquid water.

set -Eeuo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)}
source "${PROJECT_ROOT}/benchmarks/runtime_paths.sh"

RUN_ID=${RUN_ID:-metric-eqgino-pilot-$(date -u +%Y%m%dT%H%M%SZ)}
EXPERIMENT_ROOT=${EXPERIMENT_ROOT:-${MACE_FNO_WORK_ROOT}/les_water/runs/${RUN_ID}}
SOURCE_ROOT=${SOURCE_ROOT:-${MACE_FNO_WORK_ROOT}/les_water/source}
TRAIN_FILE=${TRAIN_FILE:-${SOURCE_ROOT}/data/train-H2O_RPBE-D3.xyz}
TEST_FILE=${TEST_FILE:-${SOURCE_ROOT}/data/test-H2O_RPBE-D3.xyz}
MACE_MODEL=${MACE_MODEL:-${SOURCE_ROOT}/pretrained/H20_stagetwo.model}
VALIDATION_INDICES=${VALIDATION_INDICES:-${PROJECT_ROOT}/benchmarks/les_water/validation_indices_seed17.txt}
FNO_CONFIG=${FNO_CONFIG:-${PROJECT_ROOT}/benchmarks/les_water/train_fno_3d.yaml}
GPU_PARTITION=${GPU_PARTITION:-debug-gpu}
MODEL_DTYPE=${MODEL_DTYPE:-float64}
FNO_SEED=${FNO_SEED:-17}
STEPS=${STEPS:-8000}
LOG_ROOT=${LOG_ROOT:-${EXPERIMENT_ROOT}/logs}
REPORT_ROOT=${REPORT_ROOT:-${EXPERIMENT_ROOT}/reports}
JOB_WORK_ROOT=${JOB_WORK_ROOT:-${EXPERIMENT_ROOT}/work}

for required in \
    "${TRAIN_FILE}" "${TEST_FILE}" "${MACE_MODEL}" \
    "${VALIDATION_INDICES}" "${FNO_CONFIG}"; do
    test -s "${required}"
done
mkdir -p "${EXPERIMENT_ROOT}" "${LOG_ROOT}" "${REPORT_ROOT}" "${JOB_WORK_ROOT}"

common_sbatch=(
    --parsable
    --chdir="${JOB_WORK_ROOT}"
    --output="${LOG_ROOT}/%x-%j.out"
)
gpu_sbatch=("${common_sbatch[@]}" --partition="${GPU_PARTITION}")

baseline_result=${REPORT_ROOT}/frozen-mace.json
baseline_job=$(sbatch "${gpu_sbatch[@]}" \
    --export="ALL,PROJECT_ROOT=${PROJECT_ROOT},SOURCE_ROOT=${SOURCE_ROOT},TRAIN_FILE=${TRAIN_FILE},TEST_FILE=${TEST_FILE},MACE_MODEL=${MACE_MODEL},RESULT_PATH=${baseline_result},JOB_WORK_ROOT=${JOB_WORK_ROOT}/baseline" \
    "${PROJECT_ROOT}/benchmarks/les_water/evaluate_mace.slurm")

# The first row is a migration control for the older batch-two fit. The next
# two isolate gradient-noise reduction and force-prioritized model selection.
variants=(b2-fw1 b8-fw1 b8-fw4)
batch_sizes=(2 8 8)
force_weights=(1 1 4)
evaluation_jobs=()

for index in "${!variants[@]}"; do
    variant=${variants[index]}
    batch_size=${batch_sizes[index]}
    force_weight=${force_weights[index]}
    run_root=${EXPERIMENT_ROOT}/${variant}
    checkpoint=${run_root}/les_water_${variant}_seed${FNO_SEED}_${MODEL_DTYPE}.pt
    evaluation=${REPORT_ROOT}/${variant}.json

    training_job=$(sbatch "${gpu_sbatch[@]}" \
        --export="ALL,PROJECT_ROOT=${PROJECT_ROOT},SOURCE_ROOT=${SOURCE_ROOT},TRAIN_FILE=${TRAIN_FILE},TEST_FILE=${TEST_FILE},VALIDATION_INDICES=${VALIDATION_INDICES},MACE_MODEL=${MACE_MODEL},RUN_ROOT=${run_root},CACHE_ROOT=${run_root}/cache,CHECKPOINT=${checkpoint},JOB_WORK_ROOT=${JOB_WORK_ROOT}/${variant},FNO_CONFIG=${FNO_CONFIG},FNO_SEED=${FNO_SEED},MODEL_DTYPE=${MODEL_DTYPE},STEPS=${STEPS},BATCH_SIZE=${batch_size},FORCE_WEIGHT=${force_weight}" \
        "${PROJECT_ROOT}/benchmarks/les_water/train_fno_3d.slurm")
    evaluation_job=$(sbatch "${gpu_sbatch[@]}" --dependency="afterok:${training_job}" \
        --export="ALL,PROJECT_ROOT=${PROJECT_ROOT},CHECKPOINT=${checkpoint},RESULT_PATH=${evaluation},MODEL_DTYPE=${MODEL_DTYPE},JOB_WORK_ROOT=${JOB_WORK_ROOT}/${variant}-evaluation" \
        "${PROJECT_ROOT}/benchmarks/les_water/evaluate_fno.slurm")
    evaluation_jobs+=("${evaluation_job}")
    printf '%-10s train/evaluate: %s %s\n' "${variant}" "${training_job}" "${evaluation_job}"
done

dependencies=("${baseline_job}" "${evaluation_jobs[@]}")
dependency_list=$(IFS=:; printf '%s' "${dependencies[*]}")
pilot_variants=$(IFS=:; printf '%s' "${variants[*]}")
summary_job=$(sbatch "${common_sbatch[@]}" --dependency="afterok:${dependency_list}" \
    --export="ALL,PROJECT_ROOT=${PROJECT_ROOT},BASELINE_RESULT=${baseline_result},REPORT_ROOT=${REPORT_ROOT},PILOT_VARIANTS=${pilot_variants},JOB_WORK_ROOT=${JOB_WORK_ROOT}/summary" \
    "${PROJECT_ROOT}/benchmarks/les_water/summarize_pilot.slurm")

printf 'Experiment root:   %s\n' "${EXPERIMENT_ROOT}"
printf 'Frozen MACE eval:  %s\n' "${baseline_job}"
printf 'Pilot summary:     %s\n' "${summary_job}"
printf 'Summary report:    %s\n' "${REPORT_ROOT}/pilot-summary.md"
