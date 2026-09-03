#!/usr/bin/env bash
# Prepare LLZO data and submit an externally isolated benchmark chain.

set -Eeuo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)}
source "${PROJECT_ROOT}/benchmarks/runtime_paths.sh"

RUN_ID=${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}
EXPERIMENT_ROOT=${EXPERIMENT_ROOT:-${MACE_FNO_WORK_ROOT}/llzo_qnep/runs/${RUN_ID}}
SOURCE_DATA_ROOT=${SOURCE_DATA_ROOT:-${MACE_FNO_WORK_ROOT}/llzo_qnep/data}
SPLIT_METHOD=${SPLIT_METHOD:-frame-stratified}
BLOCK_SIZE=${BLOCK_SIZE:-20}
PREPARED_NAME=${PREPARED_NAME:-prepared}
DATA_ROOT=${DATA_ROOT:-${SOURCE_DATA_ROOT}/${PREPARED_NAME}}
MACE_ONE_ROOT=${MACE_ONE_ROOT:-${EXPERIMENT_ROOT}/mace-nl0}
MACE_TWO_ROOT=${MACE_TWO_ROOT:-${EXPERIMENT_ROOT}/mace-nl1}
FNO_RUN_ROOT=${FNO_RUN_ROOT:-${EXPERIMENT_ROOT}/fno}
JOB_WORK_ROOT=${JOB_WORK_ROOT:-${EXPERIMENT_ROOT}/work}
LOG_ROOT=${LOG_ROOT:-${EXPERIMENT_ROOT}/logs}
REPORT_ROOT=${REPORT_ROOT:-${EXPERIMENT_ROOT}/reports}
FNO_SEED=${FNO_SEED:-17}
MODEL_DTYPE=${MODEL_DTYPE:-float64}
CHECKPOINT=${CHECKPOINT:-${FNO_RUN_ROOT}/llzo_fno_3d_seed${FNO_SEED}_${MODEL_DTYPE}.pt}
FNO_CONFIG=${FNO_CONFIG:-${PROJECT_ROOT}/benchmarks/llzo_qnep/train_fno_3d.yaml}
MACE_ONE_RESULT=${MACE_ONE_RESULT:-${REPORT_ROOT}/mace-nl0.json}
MACE_TWO_RESULT=${MACE_TWO_RESULT:-${REPORT_ROOT}/mace-nl1.json}
FNO_RESULT=${FNO_RESULT:-${REPORT_ROOT}/mace-fno.json}
AUDIT_OUTPUT=${AUDIT_OUTPUT:-${REPORT_ROOT}/mace-fno-audit.json}
SPECTRUM_OUTPUT=${SPECTRUM_OUTPUT:-${REPORT_ROOT}/mace-fno-spectral.json}
PUBLISHED_RESULT=${PUBLISHED_RESULT:-${REPORT_ROOT}/published-nep-qnep.json}
SUMMARY_JSON=${SUMMARY_JSON:-${REPORT_ROOT}/summary.json}
SUMMARY_MARKDOWN=${SUMMARY_MARKDOWN:-${REPORT_ROOT}/summary.md}
RUN_PUBLISHED_MODELS=${RUN_PUBLISHED_MODELS:-0}
PREPARE_ONLY=${PREPARE_ONLY:-0}

case "${RUN_PUBLISHED_MODELS}" in
    0|false|FALSE|no|NO) run_published=0 ;;
    1|true|TRUE|yes|YES) run_published=1 ;;
    *)
        printf 'RUN_PUBLISHED_MODELS must be 0/1, true/false, or yes/no; got %s\n' \
            "${RUN_PUBLISHED_MODELS}" >&2
        exit 2
        ;;
esac

mkdir -p "${EXPERIMENT_ROOT}" "${JOB_WORK_ROOT}" "${LOG_ROOT}" "${REPORT_ROOT}"
test -s "${FNO_CONFIG}"
prepare_args=(
    --data-root "${SOURCE_DATA_ROOT}"
    --download
    --split-method "${SPLIT_METHOD}"
    --block-size "${BLOCK_SIZE}"
    --prepared-name "${PREPARED_NAME}"
)
if (( run_published )); then
    prepare_args+=(--download-published-models)
fi
python3 "${PROJECT_ROOT}/benchmarks/llzo_qnep/prepare_dataset.py" \
    "${prepare_args[@]}"

case "${PREPARE_ONLY}" in
    1|true|TRUE|yes|YES)
        printf 'Prepared LLZO data at %s; no jobs submitted.\n' "${DATA_ROOT}"
        exit 0
        ;;
    0|false|FALSE|no|NO) ;;
    *)
        printf 'PREPARE_ONLY must be 0/1, true/false, or yes/no; got %s\n' \
            "${PREPARE_ONLY}" >&2
        exit 2
        ;;
esac

common_sbatch=(
    --parsable
    --chdir="${JOB_WORK_ROOT}"
    --output="${LOG_ROOT}/%x-%j.out"
)

if [[ -n "${PRETRAINED_MACE_NL0:-}" ]]; then
    MACE_ONE_MODEL=${PRETRAINED_MACE_NL0}
    test -s "${MACE_ONE_MODEL}"
    mace_one_job="reused ${MACE_ONE_MODEL}"
    mace_one_dependency=()
else
    MACE_ONE_MODEL=${MACE_ONE_ROOT}/models/llzo-PBEsol-r4p5-nl0_stagetwo.model
    mace_one_job=$(sbatch "${common_sbatch[@]}" \
        --export="ALL,PROJECT_ROOT=${PROJECT_ROOT},DATA_ROOT=${DATA_ROOT},RUN_ROOT=${MACE_ONE_ROOT},JOB_WORK_ROOT=${JOB_WORK_ROOT}/mace-nl0,NUM_INTERACTIONS=1" \
        "${PROJECT_ROOT}/benchmarks/llzo_qnep/train_mace.slurm")
    mace_one_dependency=(--dependency="afterok:${mace_one_job}")
fi

if [[ -n "${PRETRAINED_MACE_NL1:-}" ]]; then
    MACE_TWO_MODEL=${PRETRAINED_MACE_NL1}
    test -s "${MACE_TWO_MODEL}"
    mace_two_job="reused ${MACE_TWO_MODEL}"
    mace_two_dependency=()
else
    MACE_TWO_MODEL=${MACE_TWO_ROOT}/models/llzo-PBEsol-r4p5-nl1_stagetwo.model
    mace_two_job=$(sbatch "${common_sbatch[@]}" \
        --export="ALL,PROJECT_ROOT=${PROJECT_ROOT},DATA_ROOT=${DATA_ROOT},RUN_ROOT=${MACE_TWO_ROOT},JOB_WORK_ROOT=${JOB_WORK_ROOT}/mace-nl1,NUM_INTERACTIONS=2" \
        "${PROJECT_ROOT}/benchmarks/llzo_qnep/train_mace.slurm")
    mace_two_dependency=(--dependency="afterok:${mace_two_job}")
fi

baseline_one_job=$(sbatch "${common_sbatch[@]}" "${mace_one_dependency[@]}" \
    --export="ALL,PROJECT_ROOT=${PROJECT_ROOT},DATA_ROOT=${DATA_ROOT},MACE_MODEL=${MACE_ONE_MODEL},RESULT_PATH=${MACE_ONE_RESULT},JOB_WORK_ROOT=${JOB_WORK_ROOT}/eval-nl0" \
    "${PROJECT_ROOT}/benchmarks/llzo_qnep/evaluate_mace.slurm")
baseline_two_job=$(sbatch "${common_sbatch[@]}" "${mace_two_dependency[@]}" \
    --export="ALL,PROJECT_ROOT=${PROJECT_ROOT},DATA_ROOT=${DATA_ROOT},MACE_MODEL=${MACE_TWO_MODEL},RESULT_PATH=${MACE_TWO_RESULT},JOB_WORK_ROOT=${JOB_WORK_ROOT}/eval-nl1" \
    "${PROJECT_ROOT}/benchmarks/llzo_qnep/evaluate_mace.slurm")

fno_job=$(sbatch "${common_sbatch[@]}" "${mace_one_dependency[@]}" \
    --export="ALL,PROJECT_ROOT=${PROJECT_ROOT},DATA_ROOT=${DATA_ROOT},RUN_ROOT=${FNO_RUN_ROOT},MACE_MODEL=${MACE_ONE_MODEL},CHECKPOINT=${CHECKPOINT},JOB_WORK_ROOT=${JOB_WORK_ROOT}/fno,FNO_CONFIG=${FNO_CONFIG},FNO_SEED=${FNO_SEED},MODEL_DTYPE=${MODEL_DTYPE}" \
    "${PROJECT_ROOT}/benchmarks/llzo_qnep/train_fno_3d.slurm")
fno_eval_job=$(sbatch "${common_sbatch[@]}" --dependency="afterok:${fno_job}" \
    --export="ALL,PROJECT_ROOT=${PROJECT_ROOT},CHECKPOINT=${CHECKPOINT},RESULT_PATH=${FNO_RESULT},JOB_WORK_ROOT=${JOB_WORK_ROOT}/fno-eval,MODEL_DTYPE=${MODEL_DTYPE}" \
    "${PROJECT_ROOT}/benchmarks/llzo_qnep/evaluate_fno.slurm")
audit_job=$(sbatch "${common_sbatch[@]}" --dependency="afterok:${fno_job}" \
    --export="ALL,PROJECT_ROOT=${PROJECT_ROOT},CHECKPOINT=${CHECKPOINT},AUDIT_OUTPUT=${AUDIT_OUTPUT},JOB_WORK_ROOT=${JOB_WORK_ROOT}/audit" \
    "${PROJECT_ROOT}/benchmarks/llzo_qnep/audit_3d.slurm")
spectral_job=$(sbatch "${common_sbatch[@]}" --dependency="afterok:${fno_job}" \
    --export="ALL,PROJECT_ROOT=${PROJECT_ROOT},CHECKPOINT=${CHECKPOINT},SPECTRUM_OUTPUT=${SPECTRUM_OUTPUT},JOB_WORK_ROOT=${JOB_WORK_ROOT}/spectrum" \
    "${PROJECT_ROOT}/benchmarks/llzo_qnep/audit_spectral.slurm")

summary_dependencies=(
    "${baseline_one_job}"
    "${baseline_two_job}"
    "${fno_eval_job}"
    "${audit_job}"
    "${spectral_job}"
)
published_job="not requested"
published_export=""
if (( run_published )); then
    published_job=$(sbatch "${common_sbatch[@]}" \
        --export="ALL,PROJECT_ROOT=${PROJECT_ROOT},SOURCE_DATA_ROOT=${SOURCE_DATA_ROOT},DATA_ROOT=${DATA_ROOT},RESULT_PATH=${PUBLISHED_RESULT},JOB_WORK_ROOT=${JOB_WORK_ROOT}/published" \
        "${PROJECT_ROOT}/benchmarks/llzo_qnep/evaluate_published_nep.slurm")
    summary_dependencies+=("${published_job}")
    published_export=",PUBLISHED_RESULT=${PUBLISHED_RESULT}"
fi

dependency_list=$(IFS=:; printf '%s' "${summary_dependencies[*]}")
summary_job=$(sbatch "${common_sbatch[@]}" --dependency="afterok:${dependency_list}" \
    --export="ALL,PROJECT_ROOT=${PROJECT_ROOT},MANIFEST=${DATA_ROOT}/split_manifest.json,MACE_ONE_RESULT=${MACE_ONE_RESULT},MACE_TWO_RESULT=${MACE_TWO_RESULT},FNO_RESULT=${FNO_RESULT},AUDIT_OUTPUT=${AUDIT_OUTPUT},SPECTRUM_OUTPUT=${SPECTRUM_OUTPUT},SUMMARY_JSON=${SUMMARY_JSON},SUMMARY_MARKDOWN=${SUMMARY_MARKDOWN},JOB_WORK_ROOT=${JOB_WORK_ROOT}/summary${published_export}" \
    "${PROJECT_ROOT}/benchmarks/llzo_qnep/summarize.slurm")

printf 'Experiment root:       %s\n' "${EXPERIMENT_ROOT}"
printf 'FNO configuration:     %s\n' "${FNO_CONFIG}"
printf 'MACE nl0 training:     %s\n' "${mace_one_job}"
printf 'MACE nl1 training:     %s\n' "${mace_two_job}"
printf 'MACE nl0 evaluation:   %s\n' "${baseline_one_job}"
printf 'MACE nl1 evaluation:   %s\n' "${baseline_two_job}"
printf 'MACE+FNO training:     %s\n' "${fno_job}"
printf 'MACE+FNO evaluation:   %s\n' "${fno_eval_job}"
printf 'MACE+FNO audit:        %s\n' "${audit_job}"
printf 'Spectral audit:        %s\n' "${spectral_job}"
printf 'Published NEP/QNEP:    %s\n' "${published_job}"
printf 'Summary:               %s\n' "${summary_job}"
printf 'Final report:          %s\n' "${SUMMARY_MARKDOWN}"
