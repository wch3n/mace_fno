# LLZO 3D MACE+FNO benchmark

This workflow tests whether a learned three-dimensional nonlocal residual helps
a deliberately short-ranged MACE model for crystalline
Li7La3Zr2O12 (LLZO). It uses the 1,978 PBEsol reference structures released
with the qNEP study in Zenodo record
[18335947](https://zenodo.org/records/18335947).

Every structure contains 192 atoms (Li56La24Zr16O96), full three-dimensional
periodicity, energy, forces, and virial labels. The cells are aligned and
orthogonal, but their three lengths vary independently. The source contains
1,201 orthorhombic, 533 cubic, and 244 tetragonal configurations. This makes it
a useful test of heterogeneous bulk cells and of long-range contributions in a
polar solid electrolyte.

## Controlled comparison

The default workflow trains all three held-out comparisons from the same
deterministic split:

1. MACE with a 4.5 A cutoff and one message-passing interaction (`nl0`);
2. MACE with the same cutoff and two interactions (`nl1`);
3. the frozen `nl0` checkpoint plus a learned 3D FNO residual.

The third model isolates the value of the global correction from simply
increasing the local MACE receptive field. The MACE weights remain fixed during
FNO optimization. Only the latent-source projection and FNO correction are
updated.

The FNO uses `cell_mode=anisotropic`. It deposits atoms in fractional
coordinates and supplies the nonlinear operator with seven constant cell
features: log of the volume length and the six independent entries of the
volume-normalized lattice metric. These features distinguish cell size and
shape while remaining invariant to rigid Cartesian rotation of the complete
cell. The `metric_eqgino` operator generates each retained spectral weight from
the physical reciprocal magnitude
`|2*pi*A^-1*n|^2`. It is an isotropic, rigid-rotation-invariant operator for
arbitrary nonsingular cell shapes and supports heterogeneous batches.

## Data protocol

The Zenodo record provides one structure file rather than an official test
split, and it does not identify independent trajectories. The preparation
script verifies the published byte count and MD5 digest, audits every structure
and label, and supports two deterministic protocols. The default reproduces an
80:10:10 frame-level split stratified by cubic, tetragonal, and orthorhombic
cell class. The stricter `source-blocked` protocol assigns whole contiguous
source-order blocks to one split. This is a transparent block-level surrogate
for trajectory separation; it is not described as a true trajectory split.

The default frame-level split with seed 17 contains:

- 1,582 training structures;
- 198 validation structures;
- 198 held-out test structures.

Prepare without submitting jobs using:

    PREPARE_ONLY=1 bash benchmarks/llzo_qnep/submit.sh

Alternatively, invoke the preparation script directly:

    python3 benchmarks/llzo_qnep/prepare_dataset.py \
        --data-root /path/outside/the/repository/llzo_qnep \
        --download

Prepare the blocked protocol in a separate directory using:

    python3 benchmarks/llzo_qnep/prepare_dataset.py \
        --data-root /path/outside/the/repository/llzo_qnep \
        --split-method source-blocked \
        --block-size 20 \
        --prepared-name prepared-block20

The prepared XYZ files carry `source_index`, `benchmark_split`, and
`benchmark_group` metadata. `split_manifest.json` records exact source indices,
per-class counts, source provenance, and prepared-file checksums.

The cell classes are geometric strata for error reporting; they are not a
crystallographic phase assignment for individual finite-temperature snapshots.

## Running the complete benchmark

From the repository root:

    bash benchmarks/llzo_qnep/submit.sh

The controlled follow-up uses a blocked source-order split, metric-aware
EqGINO, mesh-origin augmentation, full interlaced validation, and FNO seeds 17,
29, and 41:

    MACE_FNO_WORK_ROOT=$HOME/mace_fno_runs \
        bash benchmarks/llzo_qnep/submit_followup.sh

This launcher trains the two MACE baselines once, trains three FNO corrections
against the same frozen one-interaction checkpoint, runs strict physics and
spectral audits for every seed, and writes `summary-multiseed.md` only after all
dependencies complete.

The launcher creates a timestamped experiment under
`$MACE_FNO_WORK_ROOT/llzo_qnep/runs/`. Data, logs, MACE checkpoints, FNO caches,
trained models, and reports all remain outside the Git checkout. Because GPFS
scratch can occasionally be unreliable, the complete runtime root can instead
be placed in the home filesystem:

    MACE_FNO_WORK_ROOT=$HOME/mace_fno_runs \
        bash benchmarks/llzo_qnep/submit.sh

Useful overrides include:

    RUN_ID=llzo-f64 STEPS=40000 bash benchmarks/llzo_qnep/submit.sh
    GRID=32 MODES=6 CHANNELS=8 FNO_HIDDEN_CHANNELS=32 \
        bash benchmarks/llzo_qnep/submit.sh
    PRETRAINED_MACE_NL0=/path/to/nl0.model \
    PRETRAINED_MACE_NL1=/path/to/nl1.model \
        bash benchmarks/llzo_qnep/submit.sh
    METRIC_HIDDEN_CHANNELS=16 bash benchmarks/llzo_qnep/submit.sh

The three-seed blocked follow-up uses the same metric-aware operator:

    RUN_ID=llzo-block20-metric-f64 \
    METRIC_HIDDEN_CHANNELS=16 \
        bash benchmarks/llzo_qnep/submit_followup.sh

`RESTART_LATEST=1` resumes the most recent MACE checkpoint in each run root.
The default FNO run uses float64, a 24x24x24 grid, four modes per direction,
four latent channels, two Fourier layers, and 20,000 steps. Metric-aware
EqGINO is the default. `INTERLACING_TRAINING=random` samples one of the eight
mesh origins per optimization batch while validation and inference average all
eight; this provides mesh-origin augmentation without an eightfold training
cost.
Float64 is intentionally conservative for the finite-difference force audit;
`MODEL_DTYPE=float32` is available for a faster exploratory run.

## Outputs and acceptance criteria

The `reports` directory contains:

- `mace-nl0.json` and `mace-nl1.json`: independent MACE test metrics;
- `mace-fno.json`: frozen-baseline and corrected validation/test metrics;
- `mace-fno-audit.json`: conservative-force and periodicity checks;
- `mace-fno-spectral.json`: scalar low-k and anisotropic inverse-quadratic
  response diagnostics;
- `summary.json` and `summary.md`: the combined comparison.

The summary reports overall and cell-class-resolved energy and force RMSE. A
credible nonlocal gain requires both held-out energy and force improvement over
one-interaction MACE, improvement beyond the train-fitted constant energy-offset
control, and passage of all promised exact checks. The two-interaction MACE
shows whether the same gain can be obtained by a larger local receptive field.
The low-k diagnostic is evidence about the learned response, not a training
loss and not proof that a latent field is physical charge density.

This first workflow is an energy-and-force residual benchmark. The preparation
script audits the source virials but the model does not train or report stress,
because residual cell derivatives have not yet been validated. For the same
reason, an NPT heating/cooling reproduction of the approximately 900 K LLZO
phase transition is intentionally outside this workflow. It should be added
after conservative residual stress is implemented and audited.

## Optional published NEP/QNEP context

Set `RUN_PUBLISHED_MODELS=1` to download and evaluate the published NEP,
qNEP-mode1, and qNEP-mode2 parameter files through the optional `calorine`
`CPUNEP` calculator:

    RUN_PUBLISHED_MODELS=1 bash benchmarks/llzo_qnep/submit.sh

This optional job requires `calorine` in the loaded Python environment. The
published models were trained on the complete 1,978-structure source file, so
their values on our deterministic test subset are in-sample context only. They
must not be presented as held-out numbers directly comparable to the three
models trained by this workflow.
