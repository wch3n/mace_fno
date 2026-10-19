# Geometry and field operators

[Quick start](../TUTORIAL.md) · [Training](training.md) · [Validation](validation.md)

Choose the field geometry before training:

| System | `spatial_scheme` | `cell_mode` |
|---|---|---|
| Bulk in one fixed cell | `3d` | `fixed` |
| Cubic bulk at different volumes | `3d` | `isotropic` |
| Bulk with different cell lengths or shapes | `3d` | `anisotropic` |
| Slab in one fixed in-plane cell | `2.5d` | `fixed` |
| Slab with changing in-plane cells | `2.5d` | `anisotropic` |

The variable-cell slab scheme requires `spectral_symmetry: metric_eqgino`.
The projected `2d` scheme has no explicit height mesh and is available for
comparisons. The [bulk](../examples/train_bulk.yaml) and
[slab](../examples/train_slab.yaml) examples are short workflow checks.

## Resolution and channel mixing

- `grid` and `z_grid` set the mesh dimensions.
- `modes` and `z_modes` set the Fourier cutoffs. For each transformed direction,
  the cutoff must satisfy `2 * modes <= grid`.
- `channels` sets the number of latent source fields.
- `fno_hidden_channels` sets the width of the nonlinear operator.
- `spectral_groups: 1` mixes all channels within each spectral matrix.

Check grid, mode, and channel convergence on validation data. For bulk models,
`architecture: linear` selects a single spectral operation without a pointwise
branch or activation. The two linear slab implementations are distinguished
below.

## 2D FNO for slabs

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
[benchmark guide](../benchmarks/au_mgo/README.md) identifies the data, backbones,
and campaign records.

The mesh has layout `(channels, nz, nx, ny)`. Height is measured along the
normal to the first two cell vectors. The window must contain every atom in
every configuration and may include empty space around the slab. It is finite:
there is no Fourier transform or periodic wrapping along z. `z_center: mean`
centers this window on the mean atomic height, making the correction invariant
to a rigid translation of the whole slab along its normal. Tilting the third
cell vector does not shear the field. Add `allow_periodic_z: true` under `data`
when the input file uses periodic boundary conditions across the vacuum.

### Full z-mixing and channel-wise height mixing

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

**Channel-wise height mixing** uses `z_mixing: global`. Each layer adds
three contributions before GELU:

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

### Radial weights and in-plane symmetry

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
operator. The physical benchmark FNO models use **one group (`G=1`)**.
Pointwise layers in a nonlinear stack still mix channels across groups.

Without `metric_eqgino`, the spectral layer learns complex matrices per
in-plane mode. Optional `planar_symmetry: c4` or `d4` then averages symmetry
images at evaluation and cycles through them during training. These options
require a square mesh and a square physical plane and cannot be combined with
`metric_eqgino`.

`lateral_interlacing: 2` averages four half-grid origins to reduce mesh-origin
dependence. It is independent of the spectral symmetry. The strict
`mace-fno-audit-2p5d` command checks the applicable symmetry and force identities.

### Two linear slab implementations

The code supports two linear field operators. Their configuration distinction
matters when reproducing a benchmark:

| Configuration | Field transformation |
|---|---|
| `architecture: linear`, `z_mixing: local` or `global` | One dense matrix over source channels and heights, with no pointwise branch or activation. |
| `architecture: linear`, `z_mixing: full` | A stack of full z-mixing layers, including lifting, pointwise branches, and projection, with identity activations. |

The **reported linear Au₂/MgO benchmarks use the first implementation**
(`LinearMetricSlabFNO2D`, configured with `z_mixing: global`).
This direct linear operator couples channels and heights jointly.
It is not the channel-wise nonlinear scheme. The second implementation is a
linear control matched to the width and depth of the nonlinear full-mixing
stack. Both are linear in the source field at fixed cell and give an energy
correction quadratic in that field. The source head is a learned function
of the atomic environments.

### Low-rank option

For a full-mixing stack with metric shell-spline weights, `spectral_rank: r`
factorizes each channel-height matrix into two factors. The factors are
interpolated independently and applied in sequence, avoiding construction of
the dense matrix. `spectral_rank: 0` selects the dense implementation.
A positive rank is defined per group and must not exceed `H N_z / G`.

For `G=1`, the spectral parameter count is `2 N_s H N_z r` per layer. At the
sizes above, rank 32 uses **835,584** parameters and rank 64 uses **1,671,168**.
Lifting, pointwise mixing, and projection are unchanged. This option reduces
the parameter and memory requirements but requires a new fit. The
[single-seed trials](../benchmarks/au_mgo/low_rank_full_z/README.md) are exploratory
and are not the main benchmark results.

The factors have tangent-linear extrapolation beyond the shell anchors, so
their product can grow quadratically outside the fitted range. Dense matrices
use tangent-linear extrapolation directly. Check the cell range when comparing
these parameterizations.

## Fully periodic 3D

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

The default is `--metric-parameterization shell_spline`: each reference
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
network. Only this option uses `--metric-hidden-channels`. Checkpoints record
the parameterization and spline buffers for training and ASE inference. A
checkpoint without parameterization metadata is identified as a shell spline
when its weights contain spline knots, and as a radial MLP otherwise. Loading
preserves the saved operator type.

Water-SCAN uses `--cell-mode isotropic`, which
accepts positive uniform scalings of a cubic reference cell and conditions the
nonlinear operator on cell length. Use `--cell-mode anisotropic` when cell sizes
or shapes vary within the dataset. The unconstrained 3D FNO is available
with `--spectral-symmetry none` as an ablation.

The linear metric-aware operator also supports variable cells. Its radial
weights depend on the physical reciprocal metric, but no cell-feature channels
are appended to the source field. It is therefore homogeneous and linear
in that field at a fixed cell. The source head and the atomic energy model are
not thereby made linear. This choice requires a linear-operator checkpoint.

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


## Scope and limitations

- The latent fields are not physical charge densities and have no unique
  channel basis.
- Identifying `1/k`, `1/k^2`, or slab-kernel-like response is diagnostic
  evidence, not proof that the residual is exclusively electrostatic.
- Batches share one mesh shape. Heterogeneous physical cells require explicit
  cell conditioning or shape-bucketed batches.
- Metric-aware EqGINO is invariant to rigid Cartesian cell rotations but uses
  an isotropic radial response. It does not yet learn a general anisotropic
  dielectric tensor.
- The ASE adapter provides energy and forces, but not stress or a
  production LAMMPS deployment.
