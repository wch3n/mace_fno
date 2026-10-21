# SCAN liquid-water benchmark

We test whether a global FNO correction improves the energies and forces of
local MACE models for liquid water. The SCAN dataset comes from Zenodo record
[18335947](https://zenodo.org/records/18335947). Each structure contains
128 water molecules (384 atoms) in a periodic cubic cell. Cell lengths span
approximately 11.28–15.86 Å.

## Data protocol

Run from the repository root:

```bash
python3 benchmarks/water_scan_qnep/prepare_dataset.py --download \
  --data-root /path/to/external/water_scan/source \
  --output-dir /path/to/external/water_scan/prepared
```

The script verifies the source MD5 digests and produces **471 training,
52 validation, and 134 test structures**. `split_manifest.json` records source
checksums, duplicate groups, selected indices, label checks, and split hashes.
Add `--download-published-models` to obtain the published NEP/qNEP parameter files.

The released training file contains 1,388 entries but 523 distinct geometries.
The released validation file contains 500 entries and 267 distinct geometries.
Of these, 366 entries representing 133 geometries also occur in the training
pool. We use the first occurrence of each training geometry and reserve
52 representatives by cell-length-stratified sampling with seed 17. The test
set contains the 134 source-validation geometries absent from the training pool.

Identity uses exact float64 positions, species, cell, and periodicity and
allows atom permutations. It does not identify rotated, translated, or nearly
identical configurations. Thus geometry disjointness does not establish
trajectory independence. Repeated labels must agree within 1e-6 eV per
structure and 1e-5 eV/Å per force component. Labels are not averaged.

The protocol is `exact-geometry-disjoint-source-roles-v1`. Train the local
backbones on this split and use compatible caches. Models trained on the
overlapping split cannot be reused. Literature NEP/qNEP errors use a
different evaluation protocol and provide context rather than a matched test.

## Models

The comparison includes one- and two-interaction MACE backbones at a 4.5 Å
cutoff, charge-only LES, and frozen and joint linear/nonlinear FNO models.
All reported water FNO models use **one spectral group (`G=1`)** and seeds
17, 29, 41, 53, and 67. The SCAN-water LES comparison is part of this benchmark.

The 3D FNO uses the physical reciprocal metric and shell-spline radial weights.
`cell_mode: isotropic` accepts uniform scalings of a cubic reference cell.
Deposition uses the cell and voxel volume of each structure. The nonlinear
operator also receives a constant log-cell-length feature. The linear
operator uses the reciprocal metric without appending this feature to its
source field.

The completed campaigns are documented in:

- [Nonlinear G=1 fits and convergence checks](groups1_complete/README.md).
- [Linear G=1 fits](../physical_linear_gpu/README.md).
- [Single-seed grouping comparison](channel_groups/README.md).

The manuscript uses the complete five-seed test summaries. Water
budgets are 120,000 steps, except joint MACE1 at 200,000. Checkpoint selection
uses the combined validation loss. Energy errors are unshifted per-atom RMSEs,
and force errors average Cartesian components. Seed spreads describe
individual fits, not ensemble predictions.

## Introductory launchers

```bash
export MACE_FNO_WORK_ROOT=/path/to/external/runs
bash benchmarks/water_scan_qnep/submit.sh
```

The launcher creates a timestamped directory under
`$MACE_FNO_WORK_ROOT/water_scan_qnep/runs/`. Set `RUN_ID` to name the run.
The tracked `train_fno_3d.yaml` and `train_joint_3d.yaml` use `G=1` and the
frozen or joint MACE1 budgets given above. For joint MACE2, set `STEPS=120000`.
They are introductory configurations; use the saved campaign configuration
for exact reproduction of the model settings and numerical precision. Pass
a separate YAML through `FNO_CONFIG` to customize a run. FNO training requests
the `gpu` partition for up to 48 hours.

For a short setup check:

```bash
RUN_ID=small-fno STEPS=200 GRID=16 MODES=3 \
  SBATCH_PARTITION=debug-gpu SBATCH_TIMELIMIT=02:00:00 \
  bash benchmarks/water_scan_qnep/submit.sh
```

The launcher evaluates the baseline and submits a numerical audit after FNO
training. Spectral diagnostics are disabled in the supplied configurations.

## Evaluation

Compare the selected model with its starting local backbone on the same test
set. The baseline report includes constant-offset controls. They help
separate a shift of the energy reference from a change in configuration-dependent
errors. Neither these controls nor test errors select the trained model.

Numerical checks cover conservative forces, periodic translations, and the
applicable cubic symmetries. Runtime and memory comparisons use the same
hardware, precision, and evaluation protocol. Convergence variants use
validation errors only. The data do not provide a separate long-range energy,
so improved test accuracy alone does not identify the correction as electrostatic.
