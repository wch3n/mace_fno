#!/usr/bin/env bash
# Submit the blocked-split, symmetry-controlled three-seed LLZO follow-up.

set -Eeuo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)}
source "${PROJECT_ROOT}/benchmarks/runtime_paths.sh"

RUN_ID=${RUN_ID:-llzo-block20-sym-$(date -u +%Y%m%dT%H%M%SZ)}
EXPERIMENT_ROOT=${EXPERIMENT_ROOT:-${MACE_FNO_WORK_ROOT}/llzo_qnep/runs/${RUN_ID}}
SOURCE_DATA_ROOT=${SOURCE_DATA_ROOT:-${MACE_FNO_WORK_ROOT}/llzo_qnep/data}
PREPARED_NAME=${PREPARED_NAME:-prepared-block20}
DATA_ROOT=${DATA_ROOT:-${SOURCE_DATA_ROOT}/${PREPARED_NAME}}
BLOCK_SIZE=${BLOCK_SIZE:-20}
SPLIT_SEED=${SPLIT_SEED:-17}
FNO_SEEDS=${FNO_SEEDS:-"17 29 41"}
MODEL_DTYPE=${MODEL_DTYPE:-float64}
GPU_PARTITION=${GPU_PARTITION:-debug-gpu}
MACE_ONE_ROOT=${MACE_ONE_ROOT:-${EXPERIMENT_ROOT}/mace-nl0}
MACE_TWO_ROOT=${MACE_TWO_ROOT:-${EXPERIMENT_ROOT}/mace-nl1}
JOB_WORK_ROOT=${JOB_WORK_ROOT:-${EXPERIMENT_ROOT}/work}
LOG_ROOT=${LOG_ROOT:-${EXPERIMENT_ROOT}/logs}
REPORT_ROOT=${REPORT_ROOT:-${EXPERIMENT_ROOT}/reports}
SUMMARY_JSON=${SUMMARY_JSON:-${REPORT_ROOT}/summary-multiseed.json}
SUMMARY_MARKDOWN=${SUMMARY_MARKDOWN:-${REPORT_ROOT}/summary-multiseed.md}

read -r -a seed_values <<< "${FNO_SEEDS}"
if (( ${#seed_values[@]} < 3 )); then
    printf 'FNO_SEEDS must contain at least three distinct integer seeds\n' >&2
    exit 2
fi
declare -A seen_seeds=()
for seed in "${seed_values[@]}"; do
    [[ "${seed}" =~ ^[0-9]+$ ]] || {
        printf 'Invalid FNO seed: %s\n' "${seed}" >&2
        exit 2
    }
    [[ -z "${seen_seeds[${seed}]:-}" ]] || {
        printf 'Duplicate FNO seed: %s\n' "${seed}" >&2
        exit 2
    }
    seen_seeds[${seed}]=1
done
FNO_SEEDS_CSV=$(IFS=,; printf '%s' "${seed_values[*]}")

mkdir -p "${EXPERIMENT_ROOT}" "${JOB_WORK_ROOT}" "${LOG_ROOT}" "${REPORT_ROOT}"
python3 "${PROJECT_ROOT}/benchmarks/llzo_qnep/prepare_dataset.py" \
    --data-root "${SOURCE_DATA_ROOT}" \
    --download \
    --split-method source-blocked \
    --block-size "${BLOCK_SIZE}" \
    --prepared-name "${PREPARED_NAME}" \
    --seed "${SPLIT_SEED}"

common_sbatch=(
    --parsable
    --chdir="${JOB_WORK_ROOT}"
    --output="${LOG_ROOT}/%x-%j.out"
)
gpu_sbatch=("${common_sbatch[@]}" --partition="${GPU_PARTITION}")

MACE_ONE_MODEL=${MACE_ONE_ROOT}/models/llzo-PBEsol-r4p5-nl0_stagetwo.model
MACE_TWO_MODEL=${MACE_TWO_ROOT}/models/llzo-PBEsol-r4p5-nl1_stagetwo.model
MACE_ONE_RESULT=${REPORT_ROOT}/mace-nl0.json
MACE_TWO_RESULT=${REPORT_ROOT}/mace-nl1.json

mace_one_job=$(sbatch "${gpu_sbatch[@]}" \
    --export="ALL,PROJECT_ROOT=${PROJECT_ROOT},DATA_ROOT=${DATA_ROOT},RUN_ROOT=${MACE_ONE_ROOT},JOB_WORK_ROOT=${JOB_WORK_ROOT}/mace-nl0,NUM_INTERACTIONS=1,MACE_SEED=${SPLIT_SEED}" \
    "${PROJECT_ROOT}/benchmarks/llzo_qnep/train_mace.slurm")
mace_two_job=$(sbatch "${gpu_sbatch[@]}" \
    --export="ALL,PROJECT_ROOT=${PROJECT_ROOT},DATA_ROOT=${DATA_ROOT},RUN_ROOT=${MACE_TWO_ROOT},JOB_WORK_ROOT=${JOB_WORK_ROOT}/mace-nl1,NUM_INTERACTIONS=2,MACE_SEED=${SPLIT_SEED}" \
    "${PROJECT_ROOT}/benchmarks/llzo_qnep/train_mace.slurm")

baseline_one_job=$(sbatch "${gpu_sbatch[@]}" --dependency="afterok:${mace_one_job}" \
    --export="ALL,PROJECT_ROOT=${PROJECT_ROOT},DATA_ROOT=${DATA_ROOT},MACE_MODEL=${MACE_ONE_MODEL},RESULT_PATH=${MACE_ONE_RESULT},JOB_WORK_ROOT=${JOB_WORK_ROOT}/eval-nl0" \
    "${PROJECT_ROOT}/benchmarks/llzo_qnep/evaluate_mace.slurm")
baseline_two_job=$(sbatch "${gpu_sbatch[@]}" --dependency="afterok:${mace_two_job}" \
    --export="ALL,PROJECT_ROOT=${PROJECT_ROOT},DATA_ROOT=${DATA_ROOT},MACE_MODEL=${MACE_TWO_MODEL},RESULT_PATH=${MACE_TWO_RESULT},JOB_WORK_ROOT=${JOB_WORK_ROOT}/eval-nl1" \
    "${PROJECT_ROOT}/benchmarks/llzo_qnep/evaluate_mace.slurm")

summary_dependencies=("${baseline_one_job}" "${baseline_two_job}")
for seed in "${seed_values[@]}"; do
    fno_root="${EXPERIMENT_ROOT}/fno/seed${seed}"
    checkpoint="${fno_root}/llzo_fno_3d_seed${seed}_${MODEL_DTYPE}.pt"
    fno_result="${REPORT_ROOT}/mace-fno-seed${seed}.json"
    audit_result="${REPORT_ROOT}/mace-fno-audit-seed${seed}.json"
    spectral_result="${REPORT_ROOT}/mace-fno-spectral-seed${seed}.json"

    fno_job=$(sbatch "${gpu_sbatch[@]}" --dependency="afterok:${mace_one_job}" \
        --export="ALL,PROJECT_ROOT=${PROJECT_ROOT},DATA_ROOT=${DATA_ROOT},RUN_ROOT=${fno_root},CACHE_ROOT=${fno_root}/cache,MACE_MODEL=${MACE_ONE_MODEL},CHECKPOINT=${checkpoint},JOB_WORK_ROOT=${JOB_WORK_ROOT}/fno-seed${seed},FNO_SEED=${seed},MODEL_DTYPE=${MODEL_DTYPE},VOLUME_INTERLACING=2,INTERLACING_TRAINING=random,SPECTRAL_SYMMETRY=cubic_adaptive,SPECTRAL_GROUPS=1" \
        "${PROJECT_ROOT}/benchmarks/llzo_qnep/train_fno_3d.slurm")
    fno_eval_job=$(sbatch "${gpu_sbatch[@]}" --dependency="afterok:${fno_job}" \
        --export="ALL,PROJECT_ROOT=${PROJECT_ROOT},CHECKPOINT=${checkpoint},RESULT_PATH=${fno_result},JOB_WORK_ROOT=${JOB_WORK_ROOT}/fno-eval-seed${seed},MODEL_DTYPE=${MODEL_DTYPE}" \
        "${PROJECT_ROOT}/benchmarks/llzo_qnep/evaluate_fno.slurm")
    audit_job=$(sbatch "${gpu_sbatch[@]}" --dependency="afterok:${fno_job}" \
        --export="ALL,PROJECT_ROOT=${PROJECT_ROOT},CHECKPOINT=${checkpoint},AUDIT_OUTPUT=${audit_result},JOB_WORK_ROOT=${JOB_WORK_ROOT}/audit-seed${seed}" \
        "${PROJECT_ROOT}/benchmarks/llzo_qnep/audit_3d.slurm")
    spectral_job=$(sbatch "${gpu_sbatch[@]}" --dependency="afterok:${fno_job}" \
        --export="ALL,PROJECT_ROOT=${PROJECT_ROOT},CHECKPOINT=${checkpoint},SPECTRUM_OUTPUT=${spectral_result},JOB_WORK_ROOT=${JOB_WORK_ROOT}/spectrum-seed${seed}" \
        "${PROJECT_ROOT}/benchmarks/llzo_qnep/audit_spectral.slurm")
    summary_dependencies+=("${fno_eval_job}" "${audit_job}" "${spectral_job}")

    printf 'FNO seed %-4s training/eval/audit/spectral: %s %s %s %s\n' \
        "${seed}" "${fno_job}" "${fno_eval_job}" "${audit_job}" "${spectral_job}"
done

dependency_list=$(IFS=:; printf '%s' "${summary_dependencies[*]}")
summary_job=$(sbatch "${common_sbatch[@]}" --dependency="afterok:${dependency_list}" \
    --export="ALL,PROJECT_ROOT=${PROJECT_ROOT},MANIFEST=${DATA_ROOT}/split_manifest.json,MACE_ONE_RESULT=${MACE_ONE_RESULT},MACE_TWO_RESULT=${MACE_TWO_RESULT},FNO_SEEDS_CSV=${FNO_SEEDS_CSV},REPORT_ROOT=${REPORT_ROOT},SUMMARY_JSON=${SUMMARY_JSON},SUMMARY_MARKDOWN=${SUMMARY_MARKDOWN},JOB_WORK_ROOT=${JOB_WORK_ROOT}/summary-multiseed" \
    "${PROJECT_ROOT}/benchmarks/llzo_qnep/summarize_multiseed.slurm")

printf 'Experiment root:       %s\n' "${EXPERIMENT_ROOT}"
printf 'Blocked data:          %s\n' "${DATA_ROOT}"
printf 'MACE nl0 train/eval:   %s %s\n' "${mace_one_job}" "${baseline_one_job}"
printf 'MACE nl1 train/eval:   %s %s\n' "${mace_two_job}" "${baseline_two_job}"
printf 'Multi-seed summary:    %s\n' "${summary_job}"
printf 'Final report:          %s\n' "${SUMMARY_MARKDOWN}"
