#!/usr/bin/env bash
#SBATCH --job-name=synthetic-yukawa-data
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=40
#SBATCH --mem=100G
#SBATCH --partition=shared
#SBATCH --qos=shared
#SBATCH --account=htbase
#SBATCH --time=12:00:00
# Run one stage of prepare_dataset.sh on a CPU node, for example
#   sbatch --export=ALL,STAGE=static benchmarks/synthetic_yukawa/prepare_dataset.slurm
#   sbatch --export=ALL,STAGE=md,KAPPA=0.05 benchmarks/synthetic_yukawa/prepare_dataset.slurm
#   sbatch --export=ALL,STAGE=contacts --time=01:00:00 benchmarks/synthetic_yukawa/prepare_dataset.slurm
# Pass --output to keep the scheduler log outside the checkout.

set -Eeuo pipefail

module --force purge
module load EasyBuild/2024a Python

PROJECT_ROOT=${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}
export PROJECT_ROOT
export NPROC=${NPROC:-${SLURM_CPUS_PER_TASK:-40}}
STAGE=${STAGE:?Set STAGE to static, md, contacts or verify}

if [[ "${STAGE}" == md ]]; then
    bash "${PROJECT_ROOT}/benchmarks/synthetic_yukawa/prepare_dataset.sh" md "${KAPPA:?Set KAPPA for the md stage}"
else
    bash "${PROJECT_ROOT}/benchmarks/synthetic_yukawa/prepare_dataset.sh" "${STAGE}"
fi
