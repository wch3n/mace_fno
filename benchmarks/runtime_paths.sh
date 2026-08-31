#!/usr/bin/env bash

# Keep generated models, checkpoints, caches, and reports outside the Git
# checkout. Override this site-specific default when running elsewhere.
MACE_FNO_WORK_ROOT=${MACE_FNO_WORK_ROOT:-/gpfs/scratch/acad/htbase/wchen/mace_fno/runs}
export MACE_FNO_WORK_ROOT
