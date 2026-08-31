#!/usr/bin/env bash
# Prepare Water-SCAN data and submit the MACE to FNO validation chain.

set -Eeuo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}
cd "${PROJECT_ROOT}"
mkdir -p logs

python3 benchmarks/water_scan_qnep/prepare_dataset.py --download

mace_job=$(sbatch --parsable jobs/train_water_scan_mace.slurm)
baseline_job=$(sbatch --parsable --dependency="afterok:${mace_job}" \
    jobs/evaluate_water_scan_mace.slurm)
fno_job=$(sbatch --parsable --dependency="afterok:${mace_job}" \
    jobs/train_water_scan_fno_3d.slurm)
audit_job=$(sbatch --parsable --dependency="afterok:${fno_job}" \
    jobs/audit_water_scan_fno_3d.slurm)

printf 'MACE training:      %s\n' "${mace_job}"
printf 'MACE validation:    %s\n' "${baseline_job}"
printf 'MACE+FNO training:  %s\n' "${fno_job}"
printf 'MACE+FNO audit:     %s\n' "${audit_job}"
