#!/usr/bin/env bash
#SBATCH --partition=ia
#SBATCH --qos=ia
#SBATCH --account=htbase
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --no-requeue
set -Eeuo pipefail
campaign_root=${1:?campaign directory}
task_index=${2:?task index}
resume_driver=${3:?resume driver}
previous_job=${4:-}
if [[ -n "$previous_job" ]]; then
    previous_state=$(sacct -X -n -j "$previous_job" --format=State | awk 'NF {print $1; exit}')
    case "$previous_state" in
        TIMEOUT) ;;
        COMPLETED) echo "Previous allocation completed; no follow-on needed."; exit 0 ;;
        *) echo "Refusing automatic continuation after state $previous_state for job $previous_job." >&2; exit 1 ;;
    esac
fi
module --force purge
module load EasyBuild/2024a Python
ulimit -s unlimited
export PYTHONPATH="${campaign_root}/source/src:${campaign_root}/mace_source"
export PYTHONDONTWRITEBYTECODE=1 TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
cd "$campaign_root"
nvidia-smi
exec python3 -B -u "$resume_driver" "$campaign_root" "$task_index"
