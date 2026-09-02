#!/usr/bin/env bash
#SBATCH --job-name=llzo-summary
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --partition=batch
#SBATCH --account=htbase
#SBATCH --time=00:10:00
#SBATCH --mem=4G
# Scheduler output is supplied by submit.sh so it stays outside the checkout.

set -Eeuo pipefail

module --force purge
module load EasyBuild/2024a Python

PROJECT_ROOT=${PROJECT_ROOT:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)}
MANIFEST=${MANIFEST:?Set MANIFEST}
MACE_ONE_RESULT=${MACE_ONE_RESULT:?Set MACE_ONE_RESULT}
MACE_TWO_RESULT=${MACE_TWO_RESULT:?Set MACE_TWO_RESULT}
FNO_RESULT=${FNO_RESULT:?Set FNO_RESULT}
AUDIT_OUTPUT=${AUDIT_OUTPUT:?Set AUDIT_OUTPUT}
SPECTRUM_OUTPUT=${SPECTRUM_OUTPUT:?Set SPECTRUM_OUTPUT}
SUMMARY_JSON=${SUMMARY_JSON:?Set SUMMARY_JSON}
SUMMARY_MARKDOWN=${SUMMARY_MARKDOWN:?Set SUMMARY_MARKDOWN}
JOB_WORK_ROOT=${JOB_WORK_ROOT:-$(dirname "${SUMMARY_JSON}")/work}

mkdir -p "$(dirname "${SUMMARY_JSON}")" "${JOB_WORK_ROOT}"
cd "${JOB_WORK_ROOT}"

published_args=()
if [[ -n "${PUBLISHED_RESULT:-}" ]]; then
    test -s "${PUBLISHED_RESULT}"
    published_args+=(--published "${PUBLISHED_RESULT}")
fi

python3 -u "${PROJECT_ROOT}/benchmarks/llzo_qnep/summarize_results.py" \
    --manifest "${MANIFEST}" \
    --mace-one "${MACE_ONE_RESULT}" \
    --mace-two "${MACE_TWO_RESULT}" \
    --fno "${FNO_RESULT}" \
    --audit "${AUDIT_OUTPUT}" \
    --spectral "${SPECTRUM_OUTPUT}" \
    "${published_args[@]}" \
    --output-json "${SUMMARY_JSON}" \
    --output-markdown "${SUMMARY_MARKDOWN}"

test -s "${SUMMARY_JSON}"
test -s "${SUMMARY_MARKDOWN}"
