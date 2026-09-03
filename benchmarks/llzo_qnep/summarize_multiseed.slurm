#!/usr/bin/env bash
#SBATCH --job-name=llzo-multiseed-summary
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --partition=batch
#SBATCH --account=htbase
#SBATCH --time=00:10:00
#SBATCH --mem=4G

set -Eeuo pipefail

module --force purge
module load EasyBuild/2024a Python

PROJECT_ROOT=${PROJECT_ROOT:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)}
MANIFEST=${MANIFEST:?Set MANIFEST}
MACE_ONE_RESULT=${MACE_ONE_RESULT:?Set MACE_ONE_RESULT}
MACE_TWO_RESULT=${MACE_TWO_RESULT:?Set MACE_TWO_RESULT}
FNO_SEEDS_CSV=${FNO_SEEDS_CSV:?Set FNO_SEEDS_CSV}
REPORT_ROOT=${REPORT_ROOT:?Set REPORT_ROOT}
SUMMARY_JSON=${SUMMARY_JSON:?Set SUMMARY_JSON}
SUMMARY_MARKDOWN=${SUMMARY_MARKDOWN:?Set SUMMARY_MARKDOWN}
JOB_WORK_ROOT=${JOB_WORK_ROOT:-$(dirname "${SUMMARY_JSON}")/work}

mkdir -p "$(dirname "${SUMMARY_JSON}")" "${JOB_WORK_ROOT}"
cd "${JOB_WORK_ROOT}"

IFS=',' read -r -a seeds <<< "${FNO_SEEDS_CSV}"
run_args=()
for seed in "${seeds[@]}"; do
    fno_result="${REPORT_ROOT}/mace-fno-seed${seed}.json"
    audit_result="${REPORT_ROOT}/mace-fno-audit-seed${seed}.json"
    spectral_result="${REPORT_ROOT}/mace-fno-spectral-seed${seed}.json"
    test -s "${fno_result}"
    test -s "${audit_result}"
    test -s "${spectral_result}"
    run_args+=(--run "${seed}" "${fno_result}" "${audit_result}" "${spectral_result}")
done

python3 -u "${PROJECT_ROOT}/benchmarks/llzo_qnep/summarize_multiseed.py" \
    --manifest "${MANIFEST}" \
    --mace-one "${MACE_ONE_RESULT}" \
    --mace-two "${MACE_TWO_RESULT}" \
    "${run_args[@]}" \
    --output-json "${SUMMARY_JSON}" \
    --output-markdown "${SUMMARY_MARKDOWN}"

test -s "${SUMMARY_JSON}"
test -s "${SUMMARY_MARKDOWN}"
