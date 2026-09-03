#!/usr/bin/env bash

# Keep generated models, checkpoints, caches, and reports outside the Git
# checkout. Override this site-specific default when running elsewhere.
MACE_FNO_WORK_ROOT=${MACE_FNO_WORK_ROOT:-/gpfs/scratch/acad/htbase/wchen/mace_fno/runs}
export MACE_FNO_WORK_ROOT

# Append an explicitly defined environment variable as a CLI override to a
# named Bash array. YAML remains the source of benchmark defaults.
mace_fno_append_env_option() {
    local -n arguments=$1
    local variable_name=$2
    local option_name=$3
    if [[ -v "${variable_name}" ]]; then
        arguments+=("${option_name}" "${!variable_name}")
    fi
}
