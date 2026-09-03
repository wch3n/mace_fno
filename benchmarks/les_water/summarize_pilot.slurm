#!/usr/bin/env bash
#SBATCH --job-name=les-water-summary
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --account=htbase
#SBATCH --time=00:05:00
#SBATCH --mem=2G

set -Eeuo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)}
BASELINE_RESULT=${BASELINE_RESULT:?Set BASELINE_RESULT}
REPORT_ROOT=${REPORT_ROOT:?Set REPORT_ROOT}
SUMMARY_JSON=${SUMMARY_JSON:-${REPORT_ROOT}/pilot-summary.json}
SUMMARY_MARKDOWN=${SUMMARY_MARKDOWN:-${REPORT_ROOT}/pilot-summary.md}
JOB_WORK_ROOT=${JOB_WORK_ROOT:-${REPORT_ROOT}/work}
PILOT_VARIANTS=${PILOT_VARIANTS:?Set colon-separated PILOT_VARIANTS}

mkdir -p "${REPORT_ROOT}" "${JOB_WORK_ROOT}"
cd "${JOB_WORK_ROOT}"

IFS=: read -r -a variants <<< "${PILOT_VARIANTS}"
arguments=(
    --baseline "${BASELINE_RESULT}"
    --output-json "${SUMMARY_JSON}"
    --output-markdown "${SUMMARY_MARKDOWN}"
)
for variant in "${variants[@]}"; do
    arguments+=(--run "${variant}" "${REPORT_ROOT}/${variant}.json")
done

python3 "${PROJECT_ROOT}/benchmarks/les_water/summarize_pilot.py" "${arguments[@]}"
