#!/usr/bin/env bash
# Submit PBE -> PBE0 -> PBE0+rVV10L arrays with whole-stage dependencies.

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
N_CASES=$(wc -l < cases.list)
LAST_INDEX=$((N_CASES - 1))
ARRAY_SPEC=${CASE_ARRAY:-0-${LAST_INDEX}}
if [[ ! "${ARRAY_SPEC}" =~ ^[0-9,-]+$ ]]; then
    echo "Invalid CASE_ARRAY=${ARRAY_SPEC}" >&2
    exit 2
fi
if [[ "${PARTITION}" == "debug-gpu" ]]; then
    WALLTIME=02:00:00
else
    WALLTIME=48:00:00
fi

submit_stage() {
    local stage=$1
    local dependency=${2:-}
    local command=(
        sbatch --parsable
        --partition="${PARTITION}"
        --time="${WALLTIME}"
        --array="${ARRAY_SPEC}%${MAX_PARALLEL}"
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

PBE_JOB=$(submit_stage pbe)
PBE0_JOB=$(submit_stage pbe0 "${PBE_JOB}")
RVV10_JOB=$(submit_stage pbe0_rvv10 "${PBE0_JOB}")

echo "PBE job:          ${PBE_JOB}"
echo "PBE0 job:         ${PBE0_JOB}"
echo "PBE0+rVV10L job:  ${RVV10_JOB}"
