# MACE-FNO long-range residuals

This repository develops conservative Fourier neural-operator corrections for
a frozen local MACE potential. MACE supplies local invariant descriptors and a
baseline energy. A neutral latent source head deposits descriptor-dependent
fields onto a periodic mesh, and an FNO maps those fields to a global response.
The resulting scalar residual energy is differentiated to obtain forces.

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
   heterogeneous cubic, tetragonal, and orthorhombic cells.

Generated data, MACE models, graph caches, FNO checkpoints, and audit reports
are not versioned. Benchmark workflows write them beneath
`MACE_FNO_WORK_ROOT`, whose original-cluster default is defined in
`benchmarks/runtime_paths.sh`.

## Installation and tests

Install the core package with:

```bash
python3 -m pip install -e .
```

Training or loading a frozen MACE checkpoint additionally requires:

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
- `tests/`: numerical, symmetry, checkpoint, diagnostic, and ASE tests.

There are deliberately no separate top-level `jobs/` or `examples/`
directories: benchmark-specific launchers belong to their benchmark, while
reusable implementations belong to the package.

## Frozen-MACE residual training

The MACE weights remain fixed, but descriptor derivatives with respect to atom
positions remain in the autograd graph. Detaching or evaluating the MACE
descriptors under `torch.no_grad()` would omit part of the residual force.

A generic fixed-cell run is:

```bash
source benchmarks/runtime_paths.sh
mace-fno-train \
  --mace-model /path/to/frozen.model \
  --train-file train.xyz \
  --test-file test.xyz \
  --energy-key REF_energy \
  --forces-key REF_forces \
  --train-cache "$MACE_FNO_WORK_ROOT/cache/train.pt" \
  --test-cache "$MACE_FNO_WORK_ROOT/cache/test.pt" \
  --checkpoint "$MACE_FNO_WORK_ROOT/model.pt"
```

The saved checkpoint contains the learned residual state and reconstruction
metadata, but does not duplicate the frozen MACE weights.

### 2D FNO for slabs

Select the slab representation with, for example:

```bash
  --spatial-scheme 2.5d \
  --z-grid 16 \
  --z-extent 22.0 \
  --z-center mean \
  --z-mixing global
```

The mesh layout is `(channels, nz, nx, ny)`. Only x/y are Fourier transformed;
z is finite and never circularly padded or wrapped. `--z-center mean` makes
the residual invariant to rigid translation of the complete slab along its
normal. `--z-mixing local` uses a zero-padded z CNN, while `global` learns a
dense nonperiodic z response. The physical z window must contain every atom.

`--lateral-interlacing 2` averages four half-grid mesh origins to reduce the
particle-mesh egg-box force. For square cells, `--planar-symmetry c4` or `d4`
enforces in-plane discrete energy invariance and force covariance.

### Fully periodic 3D

Select bulk 3D with:

```bash
  --spatial-scheme 3d \
  --grid 24 \
  --z-grid 24 \
  --modes 4 \
  --z-modes 4 \
  --spectral-symmetry metric_eqgino
```

`--spectral-symmetry metric_eqgino` evaluates a small radial network at the
physical reciprocal magnitude
`|2*pi*A^-1*n|^2` of every retained mode. This preserves an isotropic operator
under rigid Cartesian rotation without incorrectly equating integer modes that
have different wavelengths in an anisotropic cell. The radial-network width is
controlled by `--metric-hidden-channels`, while `--spectral-groups` controls
block-diagonal channel grouping. Water-SCAN uses `--cell-mode isotropic`, which
accepts positive uniform scalings of a cubic reference cell and conditions the
nonlinear operator on cell length. Use `--cell-mode anisotropic` when cell sizes
or shapes vary within the dataset. The unconstrained 3D FNO remains available
with `--spectral-symmetry none` as an ablation.

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
