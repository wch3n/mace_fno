# MACE–FNO: Learning nonlocal interactions

Local interatomic potentials describe atomic environments within a finite
range. MACE–FNO adds a global field response to a local MACE energy. We map
local descriptors to neutral latent sources, deposit them on a mesh, and
transform the source fields with a Fourier neural operator. Their contraction
gives an energy correction. Differentiating the total energy gives conservative
forces, including the position dependence of the local descriptors.

The MACE backbone can be frozen or trained jointly with the correction.
A linear FNO learns a fixed propagation kernel. A nonlinear FNO allows the
response to depend on the complete source field. The latent sources are
learned quantities and need not represent physical charges.

New users can start with the [MACE-FNO quick-start tutorial](TUTORIAL.md),
which covers data preparation, residual training, audits, and ASE inference.
Supply training options through `mace-fno-train --config train.yaml`.
Explicit command-line options override YAML values.

## Table of contents

- [Quick-start tutorial](TUTORIAL.md)
- [Geometries and benchmarks](#geometries-and-benchmarks)
- [Installation and tests](#installation-and-tests)
- [Repository layout](#repository-layout)
- [Frozen and joint training modes](#frozen-and-joint-training-modes)
  - [Frozen-backbone training](#frozen-backbone-training)
  - [Energy-prioritized checkpoints](#energy-prioritized-checkpoints)
  - [Resuming training](#resuming-training)
  - [Checkpoint-initialized fine-tuning](#checkpoint-initialized-fine-tuning)
  - [Two-stage optimization](#two-stage-optimization)
  - [Joint training](#joint-training)
- [Geometry-specific settings](#geometry-specific-settings)
  - [2D FNO for slabs](#2d-fno-for-slabs)
  - [Fully periodic 3D](#fully-periodic-3d)
- [Spectral diagnostics](#spectral-diagnostics)
- [ASE inference](#ase-inference)
- [Current limitations](#current-limitations)
- [Contributing and citation](#contributing-and-citation)

## Geometries and benchmarks

Implemented geometries are:

- **Slabs (`2.5d`):** in-plane Fourier transforms and an explicit finite height
  mesh. Full z-mixing couples channels and heights through a spectral matrix.
- **Bulk (`3d`):** Fourier transforms along all three periodic directions.
  Metric-aware radial weights support different cell sizes and shapes.
- **Planar projection (`2d`):** a projected field without an explicit height
  mesh, available for comparisons.

The benchmark workflows cover:

1. [Au₂/MgO](benchmarks/au_mgo/README.md), testing full z-mixing for adsorption
   energies, forces, and the wetting response to remote Al substitution.
2. [Water-SCAN](benchmarks/water_scan_qnep/README.md), testing periodic 3D FNO.
3. [LLZO](benchmarks/llzo_qnep/README.md), testing metric-aware 3D FNO on
   heterogeneous cubic, tetragonal, and orthorhombic cells.
4. [Synthetic Yukawa](benchmarks/synthetic_yukawa/README.md), comparing local
   MACE, MACE-LES, and frozen, joint, and linear FNO models for a known
   screened-Coulomb interaction.

Generated data, MACE models, graph caches, FNO checkpoints, and audit reports
are not versioned. Benchmark workflows write them beneath
`MACE_FNO_WORK_ROOT`, whose original-cluster default is defined in
`benchmarks/runtime_paths.sh`.

## Installation and tests

Install the core package with:

```bash
python3 -m pip install -e .
```

Training or loading a MACE checkpoint additionally requires:

```bash
python3 -m pip install -e '.[mace]'
```

Run the verification suite with:

```bash
PYTHONPATH=src python3 -m unittest discover -s tests -v
```

The reusable commands are installed as `mace-fno-train`,
`mace-fno-evaluate-mace`, `mace-fno-evaluate`, `mace-fno-audit-2p5d`,
`mace-fno-audit-3d`, and `mace-fno-audit-spectral`. The benchmark jobs invoke
the same modules directly so they also work from an editable checkout.

## Repository layout

- `src/mace_fno/`: model, particle-mesh, checkpoint, ASE, and training code.
- `benchmarks/au_mgo/`: complete Au2-MgO preparation, training, and audit workflow.
- `benchmarks/water_scan_qnep/`: complete Water-SCAN preparation, training, and audit workflow.
- `benchmarks/llzo_qnep/`: heterogeneous-cell LLZO training and audit workflow.
- `benchmarks/synthetic_yukawa/`: known screened interactions and charge-correlation tests.
- `benchmarks/result_catalog/`: indexed results and provenance checks.
- `tests/`: numerical, symmetry, checkpoint, diagnostic, and ASE tests.

Benchmark launchers belong to their benchmark directory. Reusable
implementations belong to the package.

## Frozen and joint training modes

### Frozen-backbone training

Frozen residual training is the default. MACE weights are fixed, but
the autograd graph includes descriptor derivatives with respect to atom
positions. Detaching or evaluating the descriptors under `torch.no_grad()` would
omit part of the residual force.

A generic fixed-cell `train.yaml` is:

```yaml
mace_model: /path/to/frozen.model

data:
  train_file: /path/to/train.xyz
  test_file: /path/to/test.xyz
  energy_key: REF_energy
  forces_key: REF_forces
  train_cache: /path/to/run/cache/train.pt
  test_cache: /path/to/run/cache/test.pt

model:
  spatial_scheme: 3d
  cell_mode: fixed
  grid: 24
  z_grid: 24
  modes: 4
  z_modes: 4
  spectral_symmetry: metric_eqgino
  spectral_groups: 1

checkpoint: /path/to/run/model.pt
```

Run it with:

```bash
mace-fno-train --config train.yaml
```

Graph caches are optional, regenerable preprocessing files, not model
checkpoints. Omit the cache paths to keep preprocessing in memory without
writing caches to disk. For multiseed runs, share compatible caches instead
of copying them into every run directory. Store these files on scratch when
home-directory quotas are limited.

Remove a shared cache only after **all** dependent training, held-out
evaluation, and diagnostic jobs have succeeded and their outputs are saved.
Do not delete it automatically when the first seed finishes. Keep the source
XYZ files, split indices, MACE models, configuration files, and FNO checkpoints
(including `.last.pt` for continuation). The training CLI can rebuild a missing
cache from those inputs. Campaigns that enforce read-only caches and cache-only
evaluation commands need a separate cache-preparation step first.
Failed cache writes clean up their own temporary
files while preserving any previously saved cache, except when the process
is forcibly terminated before cleanup can run.

The saved frozen-mode checkpoint contains the learned residual state and
reconstruction metadata, but does not duplicate the MACE weights. The trainer
writes the initialized model and then atomically replaces that file whenever
the validation objective improves, so the best model is available while
training is running. Optional step-based early stopping follows the same
validation objective:

```yaml
training:
  eval_interval: 500
  early_stopping_patience_steps: 5000
```

The patience counts optimizer steps since the best validation result. Because
validation is evaluated periodically, stopping occurs at the first validation
check that reaches or exceeds the requested patience. A value of zero disables
early stopping.

### Energy-prioritized checkpoints

An optional **energy-prioritized checkpoint** selects the lowest validation
energy RMSE among checkpoints close to the best combined validation loss:

```yaml
checkpoint: /path/to/run/model.pt
training:
  energy_checkpoint_tolerance: 0.01  # Allow loss up to 1% above the best
  energy_checkpoint_constraint: loss
  energy_checkpoint_metric: raw
```

This writes `model.energy.pt` in addition to the usual best-loss `model.pt`.
The selection rule is `argmin(E_RMSE)` subject to
`validation_loss <= (1 + tolerance) * minimum_validation_loss`. Both quantities
come from validation, not training minibatches or the test set. Candidates are
the initialized model and subsequent validation checks, so `eval_interval`
controls the selection frequency. Choose the tolerance before inspecting test
errors and apply the same rule across seeds.

Set `energy_checkpoint_constraint: forces` to constrain validation force RMSE
instead of combined loss. Set `energy_checkpoint_metric: centered` to rank by
`sqrt(E_RMSE**2 - E_ME**2)` instead of raw energy RMSE. Centered selection does
**not** calibrate the energy reference or apply a shift to the saved model.
Raw selection is the default and is appropriate when absolute energy accuracy
is the goal. Omitting `energy_checkpoint_tolerance` disables this feature.

The energy-selected file is updated atomically during training, including when
a lower best loss tightens the eligibility threshold. It contains one complete
model, including its matching MACE weights for joint training, and selection
metadata. At completion, both selected models have their energy and force
errors evaluated together, with raw validation and test metrics stored under
`evaluation_metrics` in their respective checkpoints. The energy-selected file
does not reuse the spectral diagnostic of the primary model. Run a separate
diagnostic on that file if needed. The training loss, scheduler, early stopping,
and primary best-loss selection are unchanged.

To enforce the final threshold exactly, training keeps eligible, nondominated
energy/constraint candidates in CPU memory and in `model.last.pt`. This adds
memory and checkpoint-size overhead that grows with the number of such
candidates, especially for joint training. Resuming preserves these candidates
and requires unchanged selection settings. Selection cannot be enabled
retroactively when resuming an older run that did not save them.

### Resuming training

Training also writes a separate `model.last.pt` alongside `model.pt`. The former
contains the latest training state, while the latter contains the best model
selected by validation. The latest file includes optimizer moments, scheduler
state, sampling and global random-number-generator states, warm-up flags,
early-stopping counters, the best weights so far, and spectral-monitor history.
Both files are written atomically. This works for frozen and joint training.

To resume an interrupted run with the same configuration:

```bash
mace-fno-train --config train.yaml --resume /path/to/run/model.last.pt
```

Alternatively, add `resume: /path/to/run/model.last.pt` to the YAML file.
`steps` is the **total** target, not the number of additional steps. It can
be increased when continuing a completed run. A saved early-stopping decision
is preserved, rather than resetting patience. Resuming rejects changed model,
loss, batching, validation, or diagnostic settings and checks SHA-256 hashes of
the input data and original MACE checkpoint. Cache and output locations can
change. Keep the same software and device type for reproducibility. GPU
nondeterminism can still prevent bitwise-identical results.

By default, training state is saved at initialization, validation checks, and
the end of optimization, before restoring the best model. For more frequent
saves or a custom location:

```yaml
last_checkpoint: /path/to/run/last.pt
checkpoint_interval: 100  # Additional saves every 100 optimizer steps
```

A sudden termination loses only progress since the last completed save. No
automatic Slurm requeue is performed. Older best-model or weights-only files
cannot provide a full resume. Extending a completed run continues its saved
state, including the final validation/scheduler update, so it can differ from
a run originally configured with a longer budget and a different validation
schedule at that boundary.

### Checkpoint-initialized fine-tuning

To start a **new fine-tuning stage** from existing weights, use `init_from`
instead of `resume`. For example, keep the model/data settings from the parent
configuration and change the loss scales and learning rate:

```yaml
init_from: /path/to/parent/model.pt
checkpoint: /path/to/new_run/model.pt
training:
  steps: 2000
  energy_scale: 0.30
  force_scale: 1.0
  learning_rate: 1.25e-4
```

The command-line equivalent is `--init-from /path/to/parent/model.pt`.
Best-model (`model.pt`), energy-selected (`model.energy.pt`), and latest-state
(`model.last.pt`) checkpoints with saved training configuration are accepted.
For a latest-state checkpoint, initialization uses its **latest** weights, not
the best weights embedded in its training history.

Fine-tuning restores the source head, FNO, and, for joint training, the learned
MACE weights. Adam moments, scheduler state, random streams, early-stopping
counters, and checkpoint-selection history start fresh. `steps` is the budget
for this new stage, and the learning rate comes from the new configuration.
No energy-reference shift is fitted or applied. Fresh Adam state can introduce
an initial transient, so compare sustained validation accuracy as well as the
selected checkpoint.

The architecture, dtype, frozen/joint mode, MACE head, and original MACE
reference must match. Compatible new data and different optimization settings
are allowed. The parent reference cell and spectral interpolation anchors are
preserved. Older selected checkpoints without a MACE fingerprint require their
original reference file to be accessible. Parent path, SHA-256, and step
are recorded in the new checkpoints. Output paths must not overwrite the parent.

`init_from` and `resume` are mutually exclusive. To exactly resume an interrupted
fine-tuning stage, remove `init_from` from its YAML and set `resume` to the
new `model.last.pt` of that stage. Its parent provenance is preserved. This does not
relax the unchanged-loss requirement of ordinary `resume`.

### Two-stage optimization

An optional **two-stage loss schedule** changes the energy/force balance within
one run, for either frozen or joint MACE training. For example:

```yaml
checkpoint: /path/to/run/model.pt
training:
  steps: 12000
  energy_weight: 1.0
  force_weight: 10.0
  energy_scale: 0.5
  force_scale: 1.0
  stage_two_start_step: 10000
  stage_two_energy_scale: 0.1
  # Optional: otherwise continue with the current, possibly decayed LR.
  # stage_two_learning_rate: 1.0e-4
```

Here, updates 1–10000 follow the original objective. Updates 10001–12000
have 25 times more energy weight relative to forces, because the loss divides
the energy squared error by `energy_scale**2`. These numbers illustrate the
schedule, not a universal prescription. Stronger energy weighting does not
guarantee smaller offsets without affecting force accuracy. No energy shift is
fitted or applied.

`stage_two_force_scale`, `stage_two_energy_weight`, and `stage_two_force_weight`
can also be specified. Omitted loss settings inherit their first-stage values.
`stage_two_learning_rate` and, for joint training,
`stage_two_mace_learning_rate` optionally replace the current learning rates.
The switch follows any configured warm-up and continues from the latest
weights with the same Adam moments. This changes the loss weights. It does not apply the optional weight averaging
of MACE.

Validation runs at the transition even if it falls between regular checks.
The current weights are re-scored under the new objective. Best-loss selection,
the optional energy selector, plateau-scheduler history, and patience counters
restart, so losses with different weights are never compared for selection.
Early stopping applies only after the switch, ensuring stage two is reached.
The ordinary learning-rate floor also applies to the second-stage FNO rate.

With `checkpoint: model.pt`, the first-stage selection is saved to
`model.stage1.pt` (and `model.stage1.energy.pt` if enabled). The second-stage
selections use `model.pt` and `model.energy.pt`. `model.last.pt` saves the active
stage, objective, optimizer, and random states for continuation on either side
of the transition. Physical energy/force RMSEs are comparable across stages,
but the weighted objective changes. Final test evaluation concerns the
second-stage selections.

The schedule is **off by default**. Existing YAML files and older checkpoints
keep their single-stage behavior, including exact resume from older `.last.pt`
files. A scheduled run requires the same schedule when resumed. To introduce a
new schedule using the weights of an older run, start a separate run with `init_from`
and new output paths, not `resume`.

### Joint training

Joint training uses the same total model,

```text
atoms -> MACE descriptors -> latent sources -> mesh -> FNO -> residual energy
   |          |                                               |
   +----------+---------- MACE energy -------------------------+-> total energy
```

but optimizes the total DFT energy-and-force loss through both branches. Enable
it with a smaller learning rate for the pretrained MACE parameters:

```yaml
training:
  mace_training: joint
  learning_rate: 3.0e-4
  mace_learning_rate: 1.0e-5
  mace_warmup_steps: 500
  output_initialization_scale: 0.1
```

During `mace_warmup_steps`, only the FNO and latent-source parameters change.
Afterward, backpropagation through both the MACE energy and the descriptors
updates MACE and FNO together in every optimizer step. Joint checkpoints embed
the updated MACE state. The original MACE file is still needed to reconstruct
the architecture and its atom-to-graph conversion settings.

## Geometry-specific settings

### 2D FNO for slabs

For a slab, we Fourier transform the field in the periodic plane and describe
height on a finite mesh. The configuration value is `2.5d`. It distinguishes
this representation from the height-integrated `2d` projection.

```yaml
model:
  spatial_scheme: 2.5d
  cell_mode: fixed
  grid: 24
  modes: 8
  z_grid: 24
  z_extent: 22.0
  z_center: mean
  z_mixing: full
  architecture: nonlinear
  channels: 16
  fno_hidden_channels: 16
  fno_layers: 4
  spectral_symmetry: metric_eqgino
  metric_parameterization: shell_spline
  spectral_groups: 1
  planar_symmetry: none
  lateral_interlacing: 2
```

These are the operator settings of the dense full z-mixing Au₂/MgO campaign.
They specify the architecture, not a complete training protocol. The
[benchmark guide](benchmarks/au_mgo/README.md) identifies the data, backbones,
and campaign records.

The mesh has layout `(channels, nz, nx, ny)`. Height is measured along the
normal to the first two cell vectors. The window must contain every atom in
every configuration and may include empty space around the slab. It is finite:
there is no Fourier transform or periodic wrapping along z. `z_center: mean`
centers this window on the mean atomic height, making the correction invariant
to a rigid translation of the whole slab along its normal. Tilting the third
cell vector does not shear the field. Add `allow_periodic_z: true` under `data`
when the input file uses periodic boundary conditions across the vacuum.

#### Full z-mixing and channel-wise height mixing

Full z-mixing uses a matrix over the combined channel and height indices at
each in-plane wavevector. For one hidden layer,

```text
v_hat_out[a, z, k] = sum_(b,z') R[a,z,b,z'](|k|²) v_hat_in[b,z',k].
```

Thus the propagation between two heights can also change the channel, and its
weights depend on the in-plane wavelength. A pointwise channel branch is added
to the inverse-transformed spectral field, followed by GELU. Learned lifting
and projection maps connect the source and response channels to the hidden
layers. Full z-mixing uses no separate height branch.

The earlier **channel-wise height mixing** is available as
`z_mixing: global`. This identifier is preserved for configuration and
checkpoint compatibility. Each layer adds three contributions before GELU:

```text
spectral:   R[a,b](|k|²), shared by all heights
height:     Z[a,z,z'], independent of the in-plane wavevector
pointwise:  W[a,b], applied at each mesh point
```

The height branch connects all heights within each channel. Channel coupling
occurs in the spectral and pointwise branches. This separates height coupling
from the wavelength-dependent channel matrix and reduces the parameter count.
The additional `z_mixing: local` option uses a zero-padded height convolution.

For one spectral group, hidden width `H`, `N_z` heights, and `N_s` radial
shells, the full spectral matrix has `N_s (H N_z)²` parameters per layer.
The channel-wise spectral and height branches together have
`N_s H² + H N_z²`. At `H = 16`, `N_z = 24`, and `N_s = 34`, these counts are
**5,013,504** and **17,920**, respectively. Both exclude the pointwise branch,
lifting, projection, and source head.

Full mixing is an explicit configuration choice and requires its own fit.
Changing the setting does not convert an existing channel-wise checkpoint.

#### Radial weights and in-plane symmetry

With `spectral_symmetry: metric_eqgino`, the matrices depend on the physical
squared wavevector. For row cell vectors `A = cell[:2]`,

```text
|k_parallel|² = (2*pi)² n.T @ inverse(A @ A.T) @ n.
```

The third cell vector and the vacuum thickness do not enter this metric.
`metric_parameterization: shell_spline` places independent matrix values at
the reciprocal shells of an equal-area reference square and interpolates them
with a natural cubic spline. On square cells, equal mesh sizes and mode cutoffs
give intrinsic D4 symmetry. Use `planar_symmetry: none` for this operator.

Fixed rectangular and skew cells use their actual metric. Variable in-plane
cells require `cell_mode: anisotropic`. The height window, mesh dimensions,
and integer mode counts stay fixed, so mesh spacing and Fourier coverage
change with cell size. Transfer to a new size range requires validation.

`spectral_groups` partitions the channel matrix. It must divide the hidden
width in a stacked operator or the source-channel count in a direct linear
operator. The current physical benchmark FNO models use **one group (`G=1`)**.
Pointwise layers in a nonlinear stack still mix channels across groups.

Without `metric_eqgino`, the spectral layer learns complex matrices per
in-plane mode. Optional `planar_symmetry: c4` or `d4` then averages symmetry
images at evaluation and cycles through them during training. These options
require a square mesh and a square physical plane and cannot be combined with
`metric_eqgino`.

`lateral_interlacing: 2` averages four half-grid origins to reduce mesh-origin
dependence. It is independent of the spectral symmetry. The strict
`mace-fno-audit-2p5d` command checks the applicable symmetry and force identities.

#### Two linear slab implementations

The code supports two linear field operators. Their configuration distinction
matters when reproducing a benchmark:

| Configuration | Field transformation |
|---|---|
| `architecture: linear`, `z_mixing: local` or `global` | One dense matrix over source channels and heights, with no pointwise branch or activation. |
| `architecture: linear`, `z_mixing: full` | A stack of full z-mixing layers, including lifting, pointwise branches, and projection, with identity activations. |

The **reported linear Au₂/MgO benchmarks use the first implementation**
(`LinearMetricSlabFNO2D`). Despite the historical `global` configuration value,
this direct linear operator already couples channels and heights jointly.
It is not the channel-wise nonlinear scheme. The second implementation is a
linear control matched to the width and depth of the nonlinear full-mixing
stack. Both are linear in the source field at fixed cell and give an energy
correction quadratic in that field. The source head is a learned function
of the atomic environments.

#### Low-rank option

For a full-mixing stack with metric shell-spline weights, `spectral_rank: r`
factorizes each channel-height matrix into two factors. The factors are
interpolated independently and applied in sequence, avoiding construction of
the dense matrix. `spectral_rank: 0` selects the dense implementation.
A positive rank is defined per group and must not exceed `H N_z / G`.

For `G=1`, the spectral parameter count is `2 N_s H N_z r` per layer. At the
sizes above, rank 32 uses **835,584** parameters and rank 64 uses **1,671,168**.
Lifting, pointwise mixing, and projection are unchanged. This option reduces
the parameter and memory requirements but requires a new fit. The
[single-seed trials](benchmarks/au_mgo/low_rank_full_z/README.md) are exploratory
and are not the main benchmark results.

The factors have tangent-linear extrapolation beyond the shell anchors, so
their product can grow quadratically outside the fitted range. Dense matrices
use tangent-linear extrapolation directly. Check the cell range when comparing
these parameterizations.

### Fully periodic 3D

Select bulk 3D with:

```yaml
model:
  spatial_scheme: 3d
  grid: 24
  z_grid: 24
  modes: 4
  z_modes: 4
  spectral_symmetry: metric_eqgino
  metric_parameterization: shell_spline
  spectral_groups: 1
```

`--spectral-symmetry metric_eqgino` evaluates radial weights at the physical
reciprocal magnitude `|2*pi*A^-1*n|^2` of every retained mode. This preserves an
isotropic spectral operator under rigid Cartesian rotation without incorrectly
equating integer modes that have different wavelengths in an anisotropic cell.
`--spectral-groups` controls block-diagonal channel grouping.

New models default to `--metric-parameterization shell_spline`: each reference
shell has an independently learned matrix. The fixed physical anchors are
`q_s = (2*pi/L0)^2*s`, where `s` runs over the distinct retained integer squared
radii (including zero) and `L0 = abs(det(reference_cell))^(1/3)`. The training
data preparation supplies the reference cell. For a cubic reference cell,
these anchors are its actual reciprocal shells, so copying the original EqGINO
shell matrices and all other model weights reproduces its fields, energies,
forces, and shell-parameter gradients to numerical precision. Matching the
grid, retained modes, normalization, and cell conditioning is also required.
This is a reference-cell equivalence, not a guarantee of identical independently
trained fits or accurate transfer to other cells.

Between anchors we use a natural cubic spline in physical squared wavenumber,
with tangent-linear extrapolation outside the interval (constant for one
anchor). Values and first and second derivatives are continuous. The anchor
positions never move with an input structure: noncubic reference cells use a
volume-equivalent reference cube, while evaluation always uses the actual
reciprocal metric. Strong extrapolation still requires validation. No Coulomb
kernel or positivity constraint is imposed. Low-level field/mesh constructors
accept `metric_reference_length` in angstrom (default 1). The combined MACE
model requires an explicit reference cell for this parameterization.

`--metric-parameterization radial_mlp` selects a shared radial
network. Only this option uses `--metric-hidden-channels`. Historical radial-MLP
checkpoints load unchanged without that new metadata field. New checkpoints
record the parameterization and spline buffers for training and ASE inference.
Loading never silently converts an old network into a spline.

Water-SCAN uses `--cell-mode isotropic`, which
accepts positive uniform scalings of a cubic reference cell and conditions the
nonlinear operator on cell length. Use `--cell-mode anisotropic` when cell sizes
or shapes vary within the dataset. The unconstrained 3D FNO is available
with `--spectral-symmetry none` as an ablation.

The linear metric-aware operator also supports variable cells. Its radial
weights depend on the physical reciprocal metric, but no cell-feature channels
are appended to the source field. It is therefore homogeneous and linear
in that field at a fixed cell. The source head and the atomic energy model are
not thereby made linear. Existing nonlinear checkpoints are unchanged.

`mace-fno-audit-3d --strict` distinguishes two symmetry checks. For a cubic
cell held fixed, an EqGINO signed-axis transformation is exact only if it
preserves both the mesh sizes and Fourier cutoffs. For example, a
`24 x 24 x 32` xyz mesh permits exact x/y exchanges but not exchanges with z.
All tested transformations still appear in the JSON report, including measured
errors and reasons for diagnostic-only classification. Passing the strict
checks therefore does not imply full cubic symmetry on an unequal mesh.
For `cell_mode: anisotropic`, the checker also rotates atoms and cell together,
including neighbor-image shifts, and strictly checks residual energies and
forces. This separate test applies to noncubic cells and unequal meshes too.

`--volume-interlacing 2` averages eight half-grid origins. Interlacing is
conservative but more expensive, and returned mesh fields are undefined
because the replicas have different origins.

## Spectral diagnostics

The post-deposition response can be probed without adding a spectral training
loss:

```bash
PYTHONPATH=src python3 -m mace_fno.cli.audit_spectral \
  --checkpoint /path/to/model.pt \
  --samples 4 \
  --max-mode 2
```

Cosine probes are the default. Add `--probe-phase sine` to check the other
Fourier phase, using the same checkpoint, sample-selection seed, sample count,
amplitudes, and mode range for a paired comparison. The JSON records the phase.
Each probe is zero-mean and normalized to RMS one. No retraining is needed.

The diagnostic compares the learned low-k curvature with the geometry-specific
Coulomb form:

- planar projection: the thin-sheet `1/k_parallel` response.
- 2D FNO slab (`2.5d`):
  `2*pi*exp(-k_parallel*|z-z'|)/k_parallel` on finite z profiles.
- 3D: scalar `1/k^2` and anisotropic `1/(k^T B k)` fits.

During training, `--spectral-diagnostic-samples N` enables a cheap fixed
validation probe. Routine 2D FNO slab validation uses only the monopole z
profile. The selected checkpoint uses `--spectral-diagnostic-z-profiles`. Add
`--spectral-diagnostic-depth deep` for a one-time final amplitude-convergence
sweep. Diagnostics never affect the loss or checkpoint selection.

## ASE inference

Use the combined model directly in ASE:

```python
from mace_fno import MACEFNOCalculator

atoms.calc = MACEFNOCalculator("/path/to/model.pt", device="cuda")
energy = atoms.get_potential_energy()
forces = atoms.get_forces()
```

The result dictionary also exposes `mace_energy` and `residual_energy`. Stress
is intentionally unavailable because FNO virial derivatives have not yet been
validated. The checkpoint cell and periodicity contracts are enforced during
inference.

## Current limitations

- The latent fields are not physical charge densities and have no unique
  channel basis.
- Identifying `1/k`, `1/k^2`, or slab-kernel-like response is diagnostic
  evidence, not proof that the residual is exclusively electrostatic.
- Batches share one mesh shape. Heterogeneous physical cells require explicit
  cell conditioning or shape-bucketed batches.
- Metric-aware EqGINO is invariant to rigid Cartesian cell rotations but uses
  an isotropic radial response. It does not yet learn a general anisotropic
  dielectric tensor.
- The current ASE adapter provides energy and forces, but not stress or a
  production LAMMPS deployment.

## Contributing and citation

Development conventions and the verification checklist are collected in
[CONTRIBUTING.md](CONTRIBUTING.md). Citation metadata are provided in
[CITATION.cff](CITATION.cff) and can be rendered directly by GitHub.
