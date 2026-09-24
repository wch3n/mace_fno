# MACE-FNO long-range residuals

This repository develops conservative Fourier neural-operator corrections for
local MACE potentials. MACE supplies local invariant descriptors and a baseline
energy. A neutral latent source head deposits descriptor-dependent fields onto
a periodic mesh, and an FNO maps those fields to a global response. The MACE
backbone can remain frozen or be fine-tuned jointly with the FNO. The resulting
total scalar energy is differentiated to obtain forces.

New users can start with the [MACE-FNO quick-start tutorial](TUTORIAL.md),
which covers data preparation, residual training, audits, and ASE inference.
Training options may be supplied through `mace-fno-train --config train.yaml`;
explicit command-line options override YAML values.

Implemented geometries are:

- planar projection, periodic along the first two cell vectors;
- 2D FNO for slabs, Fourier transformed in-plane with an explicit nonperiodic
  z axis (configuration value `2.5d`);
- fully periodic 3D, including a metric-aware EqGINO spectral contraction for
  arbitrary nonsingular cells.

The benchmark surface is intentionally narrow. Only the systems assessed with
complete reproducible workflows are retained:

1. [Au2-MgO](benchmarks/au_mgo/README.md), comparing planar projection and the
   slab-resolved 2D FNO correction;
2. [Water-SCAN](benchmarks/water_scan_qnep/README.md), testing periodic 3D FNO;
3. [LLZO](benchmarks/llzo_qnep/README.md), testing metric-aware 3D FNO on
   heterogeneous cubic, tetragonal, and orthorhombic cells;
4. [LES liquid water](benchmarks/les_water/README.md), revisiting the fixed-cell
   RPBE-D3 water benchmark with the current metric-aware EqGINO operator.

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

- `src/mace_fno/`: model, particle-mesh, checkpoint, ASE, and training code;
- `benchmarks/au_mgo/`: complete Au2-MgO preparation, training, and audit workflow;
- `benchmarks/water_scan_qnep/`: complete Water-SCAN preparation, training, and audit workflow;
- `benchmarks/llzo_qnep/`: heterogeneous-cell LLZO training and audit workflow;
- `benchmarks/les_water/`: fixed-cell RPBE-D3 liquid-water EqGINO workflow;
- `tests/`: numerical, symmetry, checkpoint, diagnostic, and ASE tests.

There are deliberately no separate top-level `jobs/` or `examples/`
directories: benchmark-specific launchers belong to their benchmark, while
reusable implementations belong to the package.

## Frozen and joint training modes

Frozen residual training is the default. MACE weights remain fixed, but
descriptor derivatives with respect to atom positions remain in the autograd
graph. Detaching or evaluating the descriptors under `torch.no_grad()` would
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
  spatial_scheme: 2d
  cell_mode: fixed

checkpoint: /path/to/run/model.pt
```

Run it with:

```bash
mace-fno-train --config train.yaml
```

The saved frozen-mode checkpoint contains the learned residual state and
reconstruction metadata, but does not duplicate the MACE weights. The trainer
writes the initialized model and then atomically replaces that file whenever
the validation objective improves, so the best model remains available while
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
does not reuse the primary model's spectral diagnostic. Run a separate
diagnostic on that file if needed. The training loss, scheduler, early stopping,
and primary best-loss selection are unchanged.

To enforce the final threshold exactly, training keeps eligible, nondominated
energy/constraint candidates in CPU memory and in `model.last.pt`. This adds
memory and checkpoint-size overhead that grows with the number of such
candidates, especially for joint training. Resuming preserves these candidates
and requires unchanged selection settings. Selection cannot be enabled
retroactively when resuming an older run that did not save them.

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
`steps` remains the **total** target, not the number of additional steps. It can
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
original reference file to remain accessible. Parent path, SHA-256, and step
are recorded in the new checkpoints. Output paths must not overwrite the parent.

`init_from` and `resume` are mutually exclusive. To exactly resume an interrupted
fine-tuning stage, remove `init_from` from its YAML and set `resume` to that
stage's new `model.last.pt`. Its parent provenance is preserved. This does not
relax the unchanged-loss requirement of ordinary `resume`.

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
the updated MACE state; the original MACE file is still needed to reconstruct
the architecture and its atom-to-graph conversion settings.

### 2D FNO for slabs

Select the slab representation in YAML with, for example:

```yaml
model:
  spatial_scheme: 2.5d
  z_grid: 16
  z_extent: 22.0
  z_center: mean
  z_mixing: global
```

The mesh layout is `(channels, nz, nx, ny)`. Only x/y are Fourier transformed;
z is finite and never circularly padded or wrapped. `--z-center mean` makes
the residual invariant to rigid translation of the complete slab along its
normal. `--z-mixing local` uses a zero-padded z CNN, while `global` learns a
dense nonperiodic z response. The physical z window must contain every atom.
Lateral deposition uses only the first two cell vectors, so tilting the third
vector does not shear the latent field or break normal-translation invariance
with `z_center: mean`.

`--lateral-interlacing 2` averages four half-grid mesh origins to reduce the
particle-mesh egg-box force. `--planar-symmetry c4` or `d4` requires both a
square lateral mesh and a square physical plane (orthogonal first two cell
vectors of equal length). Rectangular or skew planes must use `none`.
Evaluation averages all group images to enforce discrete energy invariance
and force covariance; training cycles through one image per forward pass.
Direct calls to `SlabFNOFieldOperator2D` must supply `cell` when C4/D4 is enabled.

For intrinsic in-plane EqGINO symmetry, add these settings to the slab model
section of a training YAML (the same options work with frozen or joint MACE):

```yaml
  architecture: nonlinear
  spectral_symmetry: metric_eqgino
  metric_parameterization: shell_spline
  spectral_groups: 1
  planar_symmetry: none
```

This replaces each lateral spectral convolution by a real radial channel
matrix `W(|k_parallel|^2)`, shared over the finite z layers. With row vectors
`A = cell[:2]`, the physical squared wavevector is
`(2*pi)^2 n.T @ inverse(A @ A.T) @ n`: neither the third cell vector nor the
vacuum thickness enters the lateral kernel. Independent shell matrices are
anchored at a reference square of side `sqrt(|a x b|)` and interpolated with
the same natural cubic shell spline as the 3D operator. At that square, the
spectral layer exactly reproduces a direct integer-shell table. Local/global
nonperiodic z mixing, nonlinear activations, and the energy readout are unchanged.

On square cells and meshes with equal lateral mode counts, this scalar EqGINO
adaptation is intrinsically D4-equivariant: training and evaluation use the
same function, without symmetry cycling or eight-image averaging. Do not
combine it with `planar_symmetry: c4/d4`. Fixed rectangular or skew cells are
also supported, using their actual metric; they do not acquire square-cell
D4 symmetry. General variable-cell slab training remains unsupported. The
standalone field operator accepts a shared cell or one per field in a batch.
For grouped spectral matrices, `spectral_groups` must divide
`fno_hidden_channels`; pointwise layers still mix the groups.

This path currently requires `architecture: nonlinear`. Existing vanilla and
C4/D4-averaged slab checkpoints retain their original architecture and load
unchanged. Switching an existing YAML to EqGINO requires a new FNO fit, not a
reinterpretation of old weights. Intrinsic symmetry removes the evaluation
ensemble but does not eliminate mesh-discretization forces, guarantee a
training speedup, or establish improved prediction accuracy. Interlacing is
still a separate choice. `mace-fno-audit-2p5d --strict` checks intrinsic D4
symmetry when the cell, grid, and mode cutoff support it.

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
reciprocal metric. Strong extrapolation still requires validation; no Coulomb
kernel or positivity constraint is imposed. Low-level field/mesh constructors
accept `metric_reference_length` in angstrom (default 1); the combined MACE
model requires an explicit reference cell for this parameterization.

`--metric-parameterization radial_mlp` retains the previous shared radial
network; only this option uses `--metric-hidden-channels`. Historical radial-MLP
checkpoints load unchanged without that new metadata field. New checkpoints
record the parameterization and spline buffers for training and ASE inference;
loading never silently converts an old network into a spline.

Water-SCAN uses `--cell-mode isotropic`, which
accepts positive uniform scalings of a cubic reference cell and conditions the
nonlinear operator on cell length. Use `--cell-mode anisotropic` when cell sizes
or shapes vary within the dataset. The unconstrained 3D FNO remains available
with `--spectral-symmetry none` as an ablation.

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
Each probe is zero-mean and normalized to RMS one; no retraining is needed.

The diagnostic compares the learned low-k curvature with the geometry-specific
Coulomb form:

- planar projection: the thin-sheet `1/k_parallel` response;
- 2D FNO slab (`2.5d`):
  `2*pi*exp(-k_parallel*|z-z'|)/k_parallel` on finite z profiles;
- 3D: scalar `1/k^2` and anisotropic `1/(k^T B k)` fits.

During training, `--spectral-diagnostic-samples N` enables a cheap fixed
validation probe. Routine 2D FNO slab validation uses only the monopole z
profile; the selected checkpoint uses `--spectral-diagnostic-z-profiles`. Add
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
  an isotropic radial response; it does not yet learn a general anisotropic
  dielectric tensor.
- The current ASE adapter provides energy and forces, but not stress or a
  production LAMMPS deployment.

## Contributing and citation

Development conventions and the verification checklist are collected in
[CONTRIBUTING.md](CONTRIBUTING.md). Citation metadata are provided in
[CITATION.cff](CITATION.cff) and can be rendered directly by GitHub.
