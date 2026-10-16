#!/usr/bin/env bash
#SBATCH --job-name=hetscreen-md
#SBATCH --partition=shared
#SBATCH --qos=shared
#SBATCH --account=htbase
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --array=0-3
# MD stage of the heterogeneous-screening dataset as a job array: task i runs
# trajectories [i*PER_TASK, (i+1)*PER_TASK) with one process each. Fewer
# processes per node than one 80-process job reduce memory-bandwidth
# contention. Label afterwards with STAGE=label gen_heterogeneous.slurm, e.g.
#   A=$(sbatch --parsable --output=/path/md_%A_%a.log gen_heterogeneous_md.slurm OUTDIR)
#   sbatch --dependency=afterok:$A --export=ALL,STAGE=label gen_heterogeneous.slurm OUTDIR
set -euo pipefail
OUT=${1:?output directory}
PER_TASK=${PER_TASK:-20}
TEMPLATES=${TEMPLATES:-${HOME}/mace_fno_runs/yukawa_L32_md2800/data_k0p05}
lo=$((SLURM_ARRAY_TASK_ID * PER_TASK))
module --force purge
module load EasyBuild/2024a Python
cd "${SLURM_SUBMIT_DIR}"
python3 -u gen_heterogeneous.py --stage md --traj-range "${lo}:$((lo + PER_TASK))" \
    --templates "${TEMPLATES}" --out "${OUT}" --nproc "${PER_TASK}"
