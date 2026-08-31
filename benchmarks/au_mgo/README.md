# Au2-MgO surface benchmark

This benchmark uses the Au-MgO-Al split and one-interaction local MACE model
from [`ChengUCB/les_fit`](https://github.com/ChengUCB/les_fit), pinned during
the original benchmark to commit `a886785caa4182a0effb95d95f0e402881adfc4d`.
The data contain 4,500 training and 500 independent test structures. The
tracked `validation_indices_seed123.txt` fixes the model-selection subset.

Generated data, models, caches, checkpoints, and reports live outside the Git
checkout under `MACE_FNO_WORK_ROOT`. On the original cluster,
`benchmarks/runtime_paths.sh` sets this to
`/gpfs/scratch/acad/htbase/wchen/mace_fno/runs`.

## Prepare

Clone the upstream repository, then stage and checksum the required files:

```bash
source benchmarks/runtime_paths.sh
python3 benchmarks/au_mgo/prepare_benchmark.py \
  --les-fit-root /path/to/les_fit
```

This copies the XYZ files to the ignored `data/les_au_mgo/` directory and the
published one-interaction MACE model to
`$MACE_FNO_WORK_ROOT/les_au_mgo/pretrained/`.

## Run

The ordinary local-MACE baseline and the 2D/2.5D residual variants are:

```bash
sbatch benchmarks/au_mgo/evaluate_mace.slurm
sbatch benchmarks/au_mgo/train_fno_2d.slurm
sbatch benchmarks/au_mgo/train_fno_2p5d.slurm
```

Audit the selected 2.5D checkpoint with:

```bash
sbatch --export=ALL,CHECKPOINT=/path/to/checkpoint.pt \
  benchmarks/au_mgo/audit_2p5d.slurm
```

## Remote-Al wetting sign reversal

The preparation command also stages the four DFT-optimized endpoints used in
the published Au2-MgO wetting test. Structures `1` and `3` are the
non-wetting and wetting endpoints, respectively. The evaluator follows the
published protocol: the Mg/O/Al substrate is fixed, only Au is relaxed with
ASE FIRE, and convergence is requested at 0.01 eV/A within 500 steps. It
reports `Delta E = E(wetting) - E(non-wetting)` for both compositions.

Submit a frozen-MACE plus FNO test while keeping all output outside Git:

```bash
source benchmarks/runtime_paths.sh
run_root="$MACE_FNO_WORK_ROOT/les_au_mgo/wetting_switch"
sbatch --output="$run_root/fno-%j.out" \
  --export=ALL,MODEL_KIND=mace-fno,MODEL=/path/to/fno.pt,RUN_NAME=fno \
  benchmarks/au_mgo/evaluate_wetting.slurm
```

Use `MODEL_KIND=mace` with a MACE `.model` file for the local-model
controls. The DFT target is a sign reversal from positive (undoped) to
negative (doped), not merely a low aggregate validation RMSE.

`train_mace.slurm` is retained as the two-interaction local-MACE
capacity control. All paths and optimization settings can be overridden by
environment variables.
