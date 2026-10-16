#!/usr/bin/env bash
set -Eeuo pipefail
campaign_root=${1:?campaign directory}
task_index=${2:?task index}
resume_driver=${3:?resume driver}
report_path=${4:?report path}
module --force purge
module load EasyBuild/2024a Python
export PYTHONPATH="${campaign_root}/source/src:${campaign_root}/mace_source"
export PYTHONDONTWRITEBYTECODE=1 TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
exec python3 -B -u "$resume_driver" "$campaign_root" "$task_index" --check --report "$report_path"
