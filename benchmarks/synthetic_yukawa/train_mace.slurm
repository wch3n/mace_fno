#!/usr/bin/env bash
#SBATCH --job-name=synthetic-mace
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH --account=htbase
#SBATCH --time=1-00:00:00
#SBATCH --mem=64G
# Train a local MACE baseline or MACE-LES for the synthetic benchmark, e.g.
#   sbatch --export=ALL,KAPPA=0.05,MODEL=mace1 benchmarks/synthetic_yukawa/train_mace.slurm
#   sbatch --export=ALL,KAPPA=0.5,MODEL=les,DATASET=md benchmarks/synthetic_yukawa/train_mace.slurm
# MODEL: mace1 (one interaction), mace2 (two interactions), les (MACE1-LES).
# DATASET: static (the benchmark data, kappa = 0.05, 0.2 or 0.5) or md (the
# 2800 K data with close-contact frames, kappa = 0.05 or 0.5).
# MACE1 takes about 1.5 h, MACE2 about 8 h, MACE1-LES about 5 h on one A100.
# Pass --output to keep the scheduler log outside the checkout.

set -Eeuo pipefail

module --force purge
module load EasyBuild/2024a Python CUDA
ulimit -s unlimited

PROJECT_ROOT=${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}
source "${PROJECT_ROOT}/benchmarks/runtime_paths.sh"
HERE=${PROJECT_ROOT}/benchmarks/synthetic_yukawa
DATA_ROOT=${DATA_ROOT:-${PROJECT_ROOT}/data/synthetic_yukawa}
DATASET=${DATASET:-static}
MODEL=${MODEL:-mace1}
KAPPA=${KAPPA:?Set KAPPA}
MACE_SEED=${MACE_SEED:-17}
RESTART_LATEST=${RESTART_LATEST:-0}
tag=${KAPPA/./p}

case "${DATASET}" in
    static)
        data=${DATA_ROOT}/static
        train=${data}/kappa${tag}_train.xyz
        name=${MODEL}_kappa${tag}
        ;;
    md)
        [[ "${KAPPA}" == 0.05 || "${KAPPA}" == 0.5 ]] || { echo "md data exist for kappa 0.05 and 0.5" >&2; exit 2; }
        data=${DATA_ROOT}/md2800/data_k${tag}
        train=${data}/kappa${tag}_train_contact.xyz
        name=${MODEL}c_md2800_kappa${tag}
        ;;
    *) echo "DATASET must be static or md" >&2; exit 2 ;;
esac
case "${MODEL}" in
    mace1) model=MACE;    interactions=1; epochs=200;  start_swa=150 ;;
    mace2) model=MACE;    interactions=2; epochs=200;  start_swa=150 ;;
    les)   model=MACELES; interactions=1; epochs=1200; start_swa=450 ;;
    *) echo "MODEL must be mace1, mace2 or les" >&2; exit 2 ;;
esac
valid=${data}/kappa${tag}_valid.xyz
test_file=${data}/kappa${tag}_test.xyz
for f in "${train}" "${valid}" "${test_file}"; do test -s "${f}"; done

RUN_ROOT=${RUN_ROOT:-${MACE_FNO_WORK_ROOT}/synthetic_yukawa/${DATASET}/kappa_${tag}/${MODEL}_seed${MACE_SEED}}
mkdir -p "${RUN_ROOT}"
cd "${RUN_ROOT}"

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export PYTHONUNBUFFERED=1
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1

python3 - "${HERE}/mace.yaml" config.yaml <<PY
import sys, yaml
config = yaml.safe_load(open(sys.argv[1]))
config.update(
    name="${name}", seed=${MACE_SEED}, work_dir=".", log_dir="./logs", model_dir="./models",
    checkpoints_dir="./checkpoints", results_dir="./results",
    train_file="${train}", valid_file="${valid}", test_file="${test_file}",
    model="${model}", num_interactions=${interactions},
    max_num_epochs=${epochs}, start_swa=${start_swa},
)
if "${RESTART_LATEST}" in ("1", "true", "yes"):
    config["restart_latest"] = True
yaml.safe_dump(config, open(sys.argv[2], "w"), sort_keys=False)
PY

mace_run_train --config config.yaml
test -s "models/${name}_stagetwo.model"
