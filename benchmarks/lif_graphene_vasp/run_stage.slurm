#!/usr/bin/env bash
#SBATCH --job-name=lifg-pbe
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --gpu-bind=none
#SBATCH --partition=gpu
#SBATCH --account=htbase
#SBATCH --time=48:00:00
#SBATCH --mem-per-cpu=50G
#SBATCH --output=logs/%x-%A_%a.out

set -Eeuo pipefail

: "${STAGE:?Set STAGE to pbe, pbe0, or pbe0_rvv10}"
case "${STAGE}" in
    pbe|pbe0|pbe0_rvv10) ;;
    *) echo "Unsupported STAGE=${STAGE}" >&2; exit 2 ;;
esac

BENCHMARK_ROOT=${SLURM_SUBMIT_DIR}
mapfile -t CASES < "${BENCHMARK_ROOT}/cases.list"
CASE_NAME=${CASES[${SLURM_ARRAY_TASK_ID}]}
WORKDIR=${BENCHMARK_ROOT}/calculations/${CASE_NAME}/${STAGE}

if [[ ! -d "${WORKDIR}" ]]; then
    echo "Missing work directory: ${WORKDIR}" >&2
    exit 3
fi
cd "${WORKDIR}"

module --force purge
module use ~/.local/easybuild/modules/all
module load EasyBuild/2024a VASP OpenMPI/5.0.3-NVHPC-25.1-CUDA-12.6.0

VALIDATOR=${BENCHMARK_ROOT}/validate_vasprun.py
VALIDATOR_ARGS=(
    "${WORKDIR}/vasprun.xml"
    --expected-atoms 36
    --require-static
)
if python3 "${VALIDATOR}" "${VALIDATOR_ARGS[@]}" --quiet; then
    echo "Already complete and validated: ${CASE_NAME}/${STAGE}"
    exit 0
fi
if [[ -e vasprun.xml ]]; then
    echo "Existing vasprun.xml is not a successful complete run; recalculating." >&2
    python3 "${VALIDATOR}" "${VALIDATOR_ARGS[@]}" || true
fi

if [[ "${STAGE}" == "pbe0" ]]; then
    RESTART_DIR=../pbe
elif [[ "${STAGE}" == "pbe0_rvv10" ]]; then
    RESTART_DIR=../pbe0
else
    RESTART_DIR=
fi

if [[ -n "${RESTART_DIR}" ]]; then
    for restart_file in WAVECAR CHGCAR; do
        if [[ ! -s "${RESTART_DIR}/${restart_file}" ]]; then
            echo "Missing restart ${RESTART_DIR}/${restart_file}" >&2
            exit 4
        fi
        cp -f "${RESTART_DIR}/${restart_file}" "${restart_file}"
    done
fi

ulimit -s unlimited

export OMP_NUM_THREADS=1
export NO_STOP_MESSAGE=1
export OMPI_MCA_btl_openib_warn_no_device_params_found=0
export OMPI_MCA_btl_openib_warn_default_gid_prefix=0
export OMPI_MCA_btl_smcuda_use_cuda_ipc=0
export PMIX_MCA_psec=native

echo "case=${CASE_NAME} stage=${STAGE} host=$(hostname) start=$(date --iso-8601=seconds)"
srun -u ~/install/vasp.6.5.0/build_nv/std/vasp
python3 "${VALIDATOR}" "${VALIDATOR_ARGS[@]}"
echo "case=${CASE_NAME} stage=${STAGE} end=$(date --iso-8601=seconds)"
