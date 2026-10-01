#!/usr/bin/env bash
# Generate the synthetic screened-Coulomb datasets used in the manuscript.
#
#   bash benchmarks/synthetic_yukawa/prepare_dataset.sh STAGE [ARGS]
#
# Stages:
#   static        static benchmark (Table III, Figs. 2-3): L = 32 A, 1146 ions,
#                 kappa = 0.05, 0.2, 0.5, 40 trajectories x 30 frames, split
#                 22/8/10 by trajectory. All kappa share one set of positions.
#   md KAPPA      2800 K MD data for one kappa (0.05 or 0.5), driven and labelled
#                 by E_LJ + s E_Yukawa with the static-benchmark scale s.
#   contacts      close-contact frames for both MD datasets (needs both md stages).
#   verify        compare every generated file with checksums.sha256.
#
# Data go to DATA_ROOT (default: <repository>/data/synthetic_yukawa, which Git
# ignores). NPROC sets the number of parallel trajectories (default 40; the
# published files were generated with 40).

set -Eeuo pipefail

HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=${PROJECT_ROOT:-$(cd -- "${HERE}/../.." && pwd)}
DATA_ROOT=${DATA_ROOT:-${PROJECT_ROOT}/data/synthetic_yukawa}
NPROC=${NPROC:-40}
PYTHON=${PYTHON:-python3}

export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

stage=${1:?usage: prepare_dataset.sh static|md KAPPA|contacts|verify}
shift

case "${stage}" in
    static)
        # About 30 min on 40 cores. Defaults: 400 K, 2 fs, gamma = 2 ps^-1,
        # LJ epsilon = 0.20 eV, sigma = 3.0 A, cutoff 4.5 A, Gaussian smearing
        # 0.6 A, k_cut = 3.5 A^-1. The trajectories are driven by the unscaled
        # kappa = 0.05 potential and relabelled for each kappa with the fitted
        # scale s_kappa (1.851, 2.048, 2.992).
        "${PYTHON}" "${HERE}/gen_yukawa.py" --out "${DATA_ROOT}/static" \
            --cells 32.0 --kappa 0.05 0.2 0.5 --ntraj 40 --nsample 30 \
            --seed 17 --nproc "${NPROC}" --nsteps 2625 --nburn 1500
        ;;
    md)
        # About 9 h per kappa on 40 cores: 10 ps burn-in, then 30 frames 1 ps apart.
        kappa=${1:?md needs KAPPA (0.05 or 0.5)}
        case "${kappa}" in
            0.05) scale=1.851164318158901; tag=k0p05 ;;
            0.5)  scale=2.9920082180339316; tag=k0p5 ;;
            *) printf 'md: KAPPA must be 0.05 or 0.5, got %s\n' "${kappa}" >&2; exit 2 ;;
        esac
        "${PYTHON}" "${HERE}/gen_yukawa.py" --out "${DATA_ROOT}/md2800/data_${tag}" \
            --cells 32.0 --kappa "${kappa}" --fixed-scale "${scale}" --temp 2800 \
            --ntraj 40 --nsample 30 --nsteps 20000 --nburn 5000 --seed 17 \
            --nproc "${NPROC}"
        ;;
    contacts)
        "${PYTHON}" "${HERE}/gen_contacts.py" --data-root "${DATA_ROOT}/md2800"
        ;;
    verify)
        # The static metadata.json is not compared: the published file predates
        # the generator's "generation" record and differs only by it.
        (cd "${DATA_ROOT}" && sha256sum -c "${HERE}/checksums.sha256")
        ;;
    *)
        printf 'unknown stage: %s\n' "${stage}" >&2
        exit 2
        ;;
esac
