#!/usr/bin/env bash
# Build a portable, submission-ready benchmark directory outside the checkout.

set -Eeuo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)}
SOURCE_ROOT=${SOURCE_ROOT:-/gpfs/home/acad/ucl-modl/wchen/scratch/mxene/mlip/pbe0-rvv10_8/Ti}
STAGE_ROOT=${STAGE_ROOT:-/gpfs/home/acad/ucl-modl/wchen/mace_fno_runs/ti2co2_adsorbates}
BENCHMARK_ROOT=${PROJECT_ROOT}/benchmarks/ti2co2_adsorbates

case "${STAGE_ROOT}/" in
    "${PROJECT_ROOT}/"*)
        printf 'STAGE_ROOT must remain outside the source checkout: %s\n' "${STAGE_ROOT}" >&2
        exit 2
        ;;
esac

mkdir -p "${STAGE_ROOT}/data"
python3 "${BENCHMARK_ROOT}/prepare_dataset.py" \
    --source-root "${SOURCE_ROOT}" \
    --output-root "${STAGE_ROOT}/data" \
    --seed 17 \
    --validation-fraction 0.05 \
    --test-fraction 0.10

install -m 755 "${BENCHMARK_ROOT}/submit_staged.sh" "${STAGE_ROOT}/submit.sh"
install -m 755 "${BENCHMARK_ROOT}/prepare_dataset.py" "${STAGE_ROOT}/prepare_dataset.py"
for script in train_mace evaluate_mace train_fno_2d audit_fno_2d audit_spectral; do
    install -m 755 "${BENCHMARK_ROOT}/${script}.slurm" "${STAGE_ROOT}/${script}.slurm"
done
install -m 644 "${BENCHMARK_ROOT}/README.md" "${STAGE_ROOT}/README.md"

for filename in train.xyz validation.xyz test.xyz split_manifest.json; do
    test -s "${STAGE_ROOT}/data/${filename}"
done

printf 'Prepared portable Ti2CO2 benchmark at %s\n' "${STAGE_ROOT}"
printf 'After copying this directory to scratch, run: bash submit.sh\n'
