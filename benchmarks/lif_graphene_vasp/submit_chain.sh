#!/usr/bin/env bash
# Submit only incomplete PBE -> PBE0 -> PBE0+rVV10L array members.

set -Eeuo pipefail

usage() {
    echo "Usage: $0 [debug-gpu|gpu] [max_parallel] [--dry-run]" >&2
}

PARTITION=${1:-debug-gpu}
MAX_PARALLEL=${2:-2}
DRY_RUN=${3:-}
if [[ "${PARTITION}" != "debug-gpu" && "${PARTITION}" != "gpu" ]]; then
    usage
    exit 2
fi
if [[ ! "${MAX_PARALLEL}" =~ ^[1-9][0-9]*$ ]]; then
    usage
    exit 2
fi
if [[ -n "${DRY_RUN}" && "${DRY_RUN}" != "--dry-run" ]]; then
    usage
    exit 2
fi

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "${ROOT}"
mkdir -p logs
mapfile -t CASES < cases.list
N_CASES=${#CASES[@]}
if (( N_CASES == 0 )); then
    echo "cases.list is empty" >&2
    exit 2
fi
LAST_INDEX=$((N_CASES - 1))
REQUESTED_ARRAY_SPEC=${CASE_ARRAY:-0-${LAST_INDEX}}
if [[ ! "${REQUESTED_ARRAY_SPEC}" =~ ^[0-9,-]+$ ]]; then
    echo "Invalid CASE_ARRAY=${REQUESTED_ARRAY_SPEC}" >&2
    exit 2
fi
if [[ "${PARTITION}" == "debug-gpu" ]]; then
    WALLTIME=02:00:00
else
    WALLTIME=48:00:00
fi

SELECTED_INDICES=()
declare -A SELECTED_SEEN=()

append_selected_index() {
    local index=$1
    if (( index < 0 || index > LAST_INDEX )); then
        echo "CASE_ARRAY index ${index} is outside 0-${LAST_INDEX}" >&2
        exit 2
    fi
    if [[ -z "${SELECTED_SEEN[${index}]:-}" ]]; then
        SELECTED_INDICES+=("${index}")
        SELECTED_SEEN[${index}]=1
    fi
}

parse_selected_indices() {
    local token start stop index
    local -a tokens
    IFS=',' read -r -a tokens <<< "${REQUESTED_ARRAY_SPEC}"
    for token in "${tokens[@]}"; do
        if [[ "${token}" =~ ^([0-9]+)-([0-9]+)$ ]]; then
            start=${BASH_REMATCH[1]}
            stop=${BASH_REMATCH[2]}
            if (( start > stop )); then
                echo "Descending CASE_ARRAY range is not supported: ${token}" >&2
                exit 2
            fi
            for ((index = start; index <= stop; index++)); do
                append_selected_index "${index}"
            done
        elif [[ "${token}" =~ ^[0-9]+$ ]]; then
            append_selected_index "${token}"
        else
            echo "Invalid CASE_ARRAY token: ${token}" >&2
            exit 2
        fi
    done
    if (( ${#SELECTED_INDICES[@]} == 0 )); then
        echo "CASE_ARRAY selected no cases" >&2
        exit 2
    fi
}
parse_selected_indices

VALIDATOR=${ROOT}/validate_vasprun.py

pending_array_spec() {
    local stage=$1
    local index case_name vasprun_path
    local -a pending=()
    local completed=0
    for index in "${SELECTED_INDICES[@]}"; do
        case_name=${CASES[${index}]}
        vasprun_path=${ROOT}/calculations/${case_name}/${stage}/vasprun.xml
        if PYTHONNOUSERSITE=1 python3 "${VALIDATOR}" "${vasprun_path}" \
            --expected-atoms 36 --require-static --quiet; then
            completed=$((completed + 1))
        else
            pending+=("${index}")
        fi
    done
    if (( ${#pending[@]} == 0 )); then
        echo "${stage}: ${completed}/${#SELECTED_INDICES[@]} selected cases complete" >&2
        return
    fi
    local pending_spec
    pending_spec=$(IFS=,; echo "${pending[*]}")
    echo "${stage}: ${completed}/${#SELECTED_INDICES[@]} selected cases complete; submitting ${pending_spec}" >&2
    echo "${pending_spec}"
}

submit_stage() {
    local stage=$1
    local array_spec=$2
    local dependency=${3:-}
    local command=(
        sbatch --parsable
        --partition="${PARTITION}"
        --time="${WALLTIME}"
        --array="${array_spec}%${MAX_PARALLEL}"
        --job-name="lifg-${stage}"
        --export="ALL,STAGE=${stage}"
    )
    if [[ -n "${dependency}" ]]; then
        command+=(--dependency="afterok:${dependency}")
    fi
    command+=(run_stage.slurm)
    if [[ "${DRY_RUN}" == "--dry-run" ]]; then
        printf '%q ' "${command[@]}" >&2
        printf '\n' >&2
        echo "DRYRUN-${stage}"
    else
        local submitted
        submitted=$("${command[@]}")
        echo "${submitted%%;*}"
    fi
}

submit_if_pending() {
    local stage=$1
    local array_spec=$2
    local dependency=${3:-}
    if [[ -z "${array_spec}" ]]; then
        return
    fi
    submit_stage "${stage}" "${array_spec}" "${dependency}"
}

report_stage() {
    local label=$1
    local job_id=$2
    local array_spec=$3
    if [[ -z "${array_spec}" ]]; then
        printf '%-18s %s\n' "${label}:" "skipped (all selected cases complete)"
    else
        printf '%-18s %s (array %s)\n' "${label}:" "${job_id}" "${array_spec}"
    fi
}

PBE_ARRAY=$(pending_array_spec pbe)
PBE_JOB=$(submit_if_pending pbe "${PBE_ARRAY}")

PBE0_ARRAY=$(pending_array_spec pbe0)
PBE0_JOB=$(submit_if_pending pbe0 "${PBE0_ARRAY}" "${PBE_JOB}")

RVV10_ARRAY=$(pending_array_spec pbe0_rvv10)
RVV10_JOB=$(submit_if_pending pbe0_rvv10 "${RVV10_ARRAY}" "${PBE0_JOB}")

report_stage "PBE job" "${PBE_JOB}" "${PBE_ARRAY}"
report_stage "PBE0 job" "${PBE0_JOB}" "${PBE0_ARRAY}"
report_stage "PBE0+rVV10L job" "${RVV10_JOB}" "${RVV10_ARRAY}"
