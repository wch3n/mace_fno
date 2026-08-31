# MACE–FNO long-range prototype

This repository implements a set of deliberately restricted milestones toward
a MACE–FNO potential:

- smooth cubic B-spline assignment of atom-centred scalar values to a
  two-dimensional periodic mesh;
- an analytic planar Coulomb operator evaluated with FFTs;
- a conservative particle–mesh energy whose forces follow from PyTorch
  autograd;
- a direct reciprocal-space reference used to quantify mesh convergence;
- native PyTorch linear and nonlinear 2D FNOs with learned complex spectral
  convolutions;
- a hybrid 2.5D slab FNO that retains an explicit finite z axis, transforms
  only the periodic x/y directions, and mixes z layers without circular
  padding;
- a fully periodic 3D particle mesh and FNO for bulk systems, with Fourier
  modes and cubic B-spline wrapping along all three lattice directions;
- a normalized field-operator adapter that can replace the analytic kernel in
  the same conservative particle–mesh energy; and
- a frozen-MACE adapter that selects the exact even-scalar (`0e`) columns from
  every interaction layer, predicts graph-neutral latent sources, and adds a
  conservative FNO residual trained jointly against energies and forces.

The analytic operator is a validation target, not the final learned model. Once
the particle–mesh mapping and gradients are trusted, the analytic spectral
operator can be replaced by an FNO without changing the surrounding energy
construction.

## Present assumptions

- Fixed in-plane cell vectors within a 2D/2.5D training run, or a fixed full
  cell within a 3D training run.
- A fixed, non-degenerate three-dimensional cell.
- Scalar or multichannel atom values, with every channel neutral by default.
- Periodicity in the first two lattice directions for 2D/2.5D, or all three
  lattice directions for the bulk 3D model.
- For the optional 2.5D model, a fixed physical z window shared by every
  structure in a run. It must contain the full slab and adsorbate height range.
- A truncated planar kernel, `2*pi/|k|`, with the zero mode removed.
- Main-lobe deconvolution of the cubic assignment window in the analytic
  operator; this can be disabled for diagnostics.

The MACE-coupled wrapper accepts ordinary batched MACE dictionaries. The
training entry point stacks the MACE graphs and particle meshes so several
configurations share one forward/backward pass. It remains intentionally
restricted to a fixed cell (in-plane for surface models and the complete cell
for 3D). Three-dimensional periodic input requires `--allow-periodic-z` only
when deliberately using a 2D/2.5D approximation; the 3D scheme handles those
periodic images explicitly.

The model includes the reciprocal-space self contribution associated with the
finite mode cutoff. It is constant in the direct point-charge reference, but a
finite particle mesh introduces a small grid-position ("egg-box") error. The
cubic assignment and low-mode truncation make this error converge rapidly with
mesh refinement.

## Run the verification suite

Install the core package in editable mode with:

```bash
python3 -m pip install -e .
```

Training against a frozen MACE checkpoint additionally requires:

```bash
python3 -m pip install -e '.[mace]'
```

The implementation is organized by geometry: `fno_2d.py` contains planar
operators, `fno_slab.py` contains finite-z 2.5D operators, and `fno_3d.py`
contains periodic bulk and EqGINO-style operators. The historical imports from
`mace_fno.fno` remain supported, as do the original residual state-dict keys.
Reusable dataset, cache, initialization, evaluation, and checkpoint utilities
live under `mace_fno.training`.

No test framework beyond the Python standard library is required:

```bash
PYTHONPATH=src python3 -m unittest discover -s tests -v
```

Run the small demonstration with:

```bash
PYTHONPATH=src python3 examples/toy_charges.py
```

Train the FNO against synthetic analytic fields with:

```bash
PYTHONPATH=src python3 examples/train_toy_fno.py --device cpu
```

Train the residual branch against an extended XYZ dataset while keeping a MACE
checkpoint frozen with:

```bash
PYTHONPATH=src python3 examples/train_mace_residual.py \
  --mace-model frozen_mace.model \
  --train-file train.xyz \
  --test-file test.xyz \
  --energy-key REF_energy \
  --forces-key REF_forces \
  --batch-size 4 \
  --evaluation-batch-size 8 \
  --train-cache artifacts/cache/train-float64.pt \
  --test-cache artifacts/cache/test-float64.pt \
  --checkpoint mace_fno_residual.pt
```

After installation, the equivalent packaged command is `mace-fno-train`. The
example path remains as a compatibility wrapper for existing SLURM scripts.

Select the hybrid 2.5D slab model by adding, for example:

```bash
  --z-grid 12 \
  --z-extent 14.0 \
  --z-center mean \
  --z-mixing global \
  --lateral-interlacing 2 \
  --planar-symmetry d4
```

Here the mesh field is `(channels, 12, nx, ny)`. `--z-extent` is a physical
distance in angstrom, not the full vacuum-cell length unless that is genuinely
the desired modelling window. `--z-center mean` removes rigid translation of
the entire slab along its normal; `--z-center cell` instead anchors the window
to the projected centre of the third cell vector. An atom outside the selected
finite window is rejected rather than silently wrapped or clipped.
The default `--z-mixing local` uses the original zero-padded z CNN and remains
compatible with existing checkpoints. `--z-mixing global` instead learns a
separate dense nonperiodic `(z_out, z_in)` response for every hidden channel,
so every finite z layer can interact in one block without connecting the top
and bottom boundaries periodically. `--z-kernel-size` applies only to the local
mixer.

`--lateral-interlacing 2` averages the conservative residual energy over a
2 x 2 set of half-grid in-plane mesh origins. This suppresses the lateral
particle-mesh "egg-box" force while retaining exact energy-force consistency;
it evaluates four mesh fields and is therefore more expensive than the default
single origin. Because the four fields have different coordinate origins,
`return_fields=True` is intentionally unavailable for an interlaced model.
For square surface cells, `--planar-symmetry c4` group-averages the complete
field operator over four rotations. `--planar-symmetry d4` also includes the
four reflected operations and is the physical default for a square, achiral
surface. Both enforce exact in-plane energy invariance and force equivariance
at validation and inference. During optimization, the implementation cycles
deterministically through one group image per forward pass; this balanced
symmetry augmentation avoids evaluating all four or eight images at every
force-training step.

For a file containing several fixed-cell supercell families, select one family
with `--num-atoms`, or explicitly retain only the first labeled structure's
cell with `--skip-cell-mismatch`. A supplied `--test-file` is evaluated only
before and after optimization. If structures are periodic along the vacuum
direction, `--allow-periodic-z` acknowledges that MACE stays 3D-periodic while
the learned residual remains nonperiodic along z.

Install the optional integration dependency with `pip install -e '.[mace]'` if
MACE is not already available. The saved residual checkpoint deliberately
excludes the frozen MACE weights and records the path and architecture needed
to reconstruct the combined model.

The example defaults to the linear FNO because its analytic Coulomb target is a
linear Green-function map. Use `--architecture nonlinear` to exercise the
nonlinear network; it is retained for later environment-dependent responses,
but is intentionally a harder and less identifiable model for this toy target.

For the 2.5D residual, the two architectures have a more specific meaning:

- `nonlinear` is the economical prototype: each block applies a shared 2D
  spectral convolution to every z slice and a zero-padded 1D CNN along z;
- `linear` directly learns the dense complex response
  `R(k_parallel, z, z')` at each retained in-plane mode. It is physically
  transparent but grows quadratically with the number of z layers.

Neither architecture FFTs or wraps z. Both use the volumetric conservative
energy `0.5 * integral rho*phi dA dz`, so autograd includes normal as well as
in-plane residual-force components.

Select the fully periodic bulk model with, for example:

```bash
  --spatial-scheme 3d \
  --grid 32 \
  --z-grid 32 \
  --modes 8 \
  --z-modes 8
```

For a cubic cell and cubic mesh, the 3D operator can instead use the native
EqGINO-style spectral layer:

```bash
  --spatial-scheme 3d \
  --grid 32 \
  --z-grid 32 \
  --modes 8 \
  --z-modes 8 \
  --spectral-symmetry eqgino \
  --spectral-groups 4 \
  --volume-interlacing 2
```

This path follows EqGINO's two central EqFNO choices: a full complex FFT and
one shared channel-mixing matrix for every exact reciprocal-radius shell
`kz^2 + kx^2 + ky^2`. It therefore enforces the complete signed-axis group of
the cubic grid natively, without the 48-fold inference average. The optional
channel groups use EqGINO's block-diagonal contraction to offset the full-FFT
cost; the spectral channel count (`--channels` for a linear model or
`--fno-hidden-channels` for a nonlinear model) must be divisible by the group
count. `--spectral-groups 1` retains dense mixing.

`--volume-interlacing 2` averages the conservative 3D residual energy over
the eight combinations of zero and half-grid shifts along the three lattice
directions. The replicas are packed into one enlarged mesh/FFT batch. This
suppresses continuous-translation egg-box errors while preserving EqGINO's
cubic energy invariance and force covariance. It raises the particle-mesh/FNO
work and memory by roughly a factor of eight; the frozen MACE backbone is still
evaluated only once. As for slab interlacing, `return_fields=True` is
unavailable because the eight fields live on different mesh origins.

Unlike the reference EqGINO implementation, this real density-to-potential
model stores real radial weights. For a scalar isotropic real-to-real operator,
`W(-k)=W(k)` and Hermitian consistency together require this restriction. The
implementation is based on the [EqGINO paper](https://arxiv.org/abs/2606.03260)
and its [official code](https://github.com/sung-won-kim/EqGINO), without adding
EqGINO's irregular-mesh GNO encoder/decoder because the differentiable
particle-mesh assignment already provides that mapping here.

The 3D mesh layout is `(channels, nz, nx, ny)`. Atomic positions are converted
with the complete (including triclinic) cell and wrapped in fractional x, y,
and z. `--z-extent`, `--z-center`, `--z-kernel-size`, `--z-mixing`,
`--lateral-interlacing`, and `--planar-symmetry` are slab concepts and do not
apply. The input configurations must be periodic in all three directions and
share the same complete cell. The saved checkpoint records `spatial_scheme=3d`,
the volume-interlacing factor, and the three retained mode counts.

This is a software-identifiability test: it asks whether the FNO can learn a
known global operator after particle-to-mesh conversion. It is not evidence
that an FNO has learned the missing physics in a DFT dataset.

With the deterministic defaults, the linear FNO is expected to recover the toy
operator to approximately numerical precision. The nonlinear result should be
treated separately because matching field values does not by itself identify
the correct input derivative needed for atomistic forces.

## Training performance

`--batch-size` is a true multi-configuration batch; `--accumulation-steps`
accumulates several such batches. Persistent train/validation/test caches store
the pruned MACE neighbor graphs and frozen-MACE predictions. Cache metadata
includes the source data and checkpoint size and modification time, so stale
caches are rebuilt automatically. Training uses `reference - frozen MACE`
energy and force labels directly. This avoids cancellation between large total
energies while preserving the same residual objective.

The default nonlinear model initializes its final field projection to zero, so
the combined prediction begins exactly at frozen MACE. A short staged warm-up
can make the projection nonzero before the complete residual branch is trained:

```bash
  --output-warmup-steps 250 \
  --output-warmup-learning-rate 3e-3 \
  --learning-rate 3e-4
```

During those first steps only the final output projection is trainable. The
source head and remaining FNO layers are then unfrozen, the main learning rate
is restored, and the same Adam optimizer is retained. The validation scheduler
is inactive during warm-up, while validation checkpoints are still eligible
for selection. This option applies only to the nonlinear architecture; zero
warm-up steps preserve the original training behavior.

This schedule is experimental and remains disabled by default. In the matched
interlaced EqGINO water test, 250 projection-only steps at `3e-3` delayed rather
than accelerated force learning and worsened the 3,000-step held-out force
RMSE. It should therefore not be enabled for the current water model without a
new controlled justification.

A less disruptive alternative keeps every residual parameter trainable from
the first step while scaling the random final projection to a small nonzero
amplitude:

```bash
  --output-initialization-scale 0.1
```

This sacrifices the exactly frozen-MACE initial prediction but immediately
provides gradients to the source head and upstream FNO layers. A scale of zero
retains the original exact-zero initialization. Scaled initialization is
mutually exclusive with projection-only warm-up and with the fully random
residual initialization option.

For the matched interlaced EqGINO water run, scale 0.1 advanced the force
trajectory by roughly 100 optimizer steps and improved the 3,000-step held-out
RMSE from `25.0118` to `24.9398` meV/A. This is the preferred exploratory
initialization for that setup, although the default remains zero for backward
compatibility and for workflows requiring an exactly frozen-MACE initial
prediction.

`--evaluation-scope validation-test` omits repeated full-training and offset
controls after those diagnostics have been established. Reference labels and
cached baseline predictions remain float64 even for a float32 model, allowing
the latter to be benchmarked without first rounding large LES total energies.
For mixed-precision runs, the checkpoint is first loaded in float64; MACE
descriptors and interaction energies then run in float32, while the large
composition-dependent atomic reference energy is reconstructed and accumulated
in float64. The exact reference table is included in residual checkpoints.

## Learned-operator limitations

- The FNO weights are indexed by fractional-grid modes and currently assume a
  fixed cell. They are not yet conditioned on physical reciprocal vectors or
  the cell metric.
- The default 3D model is discretely translation equivariant and maps a zero
  field to zero, but is not automatically rotation or lattice-basis
  equivariant. `--spectral-symmetry eqgino` additionally enforces signed-axis
  equivariance for cubic cells and grids; it does not make a Cartesian mesh
  equivariant to arbitrary continuous rotations or general lattice-basis
  changes.
- The bilinear energy gives conservative forces, although training only on
  potentials does not guarantee accurate force derivatives. Energy-and-force
  residual training is required for the MACE experiment.
- Batch members must currently share a mesh shape and the cell vectors used by
  their scheme (two for 2D/2.5D, all three for 3D).
- The hybrid FNO itself accepts different lateral resolutions as long as all
  retain the requested Fourier modes. End-to-end heterogeneous-cell training
  still requires shape-bucketed batches and a physical-wavevector-conditioned
  kernel; mode indices alone do not make different supercells equivalent.

## Current MACE coupling

`MACEFNOResidual` freezes every MACE parameter but retains the descriptor
derivatives with respect to positions. Do not wrap its forward pass in
`torch.no_grad()` when forces are requested. During training it sets
`create_graph=True`, so force-error backpropagation updates the latent source
head and FNO without updating MACE. Frozen baseline energies and forces can be
cached, but MACE descriptors are deliberately recomputed: detaching cached
descriptors would omit part of the residual force derivative.

The next scientific milestone is a matched larger-supercell experiment with
distance-stratified diagnostics showing that improvements come from
interactions beyond the MACE receptive field rather than from extra generic
model capacity.

Audit a trained Au2-MgO 2.5D checkpoint independently of its training RMSE with:

```bash
PYTHONPATH=src python3 examples/audit_au_mgo_2p5d.py \
  --checkpoint artifacts/les_au_mgo/model.pt \
  --samples 32 \
  --strict
```

The audit checks latent-source neutrality, residual-force finite differences,
the acoustic sum rule, rigid translations, exact one-grid translations and
held-out force errors resolved by Cartesian direction.

For a fully periodic 3D checkpoint, run the corresponding bulk audit with:

```bash
PYTHONPATH=src python3 examples/audit_les_water_3d.py \
  --checkpoint artifacts/les_water/les_water_fno_3d_seed17_float64.pt \
  --samples 8 \
  --strict
```

In addition to the checks above, the 3D audit covers all three grid and lattice
directions, verifies total-force decomposition, and measures signed-axis cubic
energy invariance and force equivariance. Arbitrary continuous translations
remain a diagnostic. Signed-axis invariance/covariance is promised only for
EqGINO checkpoints; 2x2x2 interlacing preserves that exact discrete symmetry.

For a cubic bulk cell, test inference-only group averaging of an existing 3D
checkpoint with:

```bash
PYTHONPATH=src python3 examples/evaluate_les_water_3d_cubic_average.py \
  --checkpoint artifacts/les_water/les_water_fno_3d_seed17_float64.pt \
  --output artifacts/les_water/seed17_cubic_average.json \
  --device cuda \
  --strict
```

This evaluates the raw residual, the 24 proper cubic rotations (`O`) and the
full 48 signed-axis operations (`O_h`). MACE descriptors are evaluated once per
structure; only the particle-mesh/FNO branch is expanded over the group. Forces
are obtained by differentiating each averaged scalar energy, so descriptor
derivatives are retained and the symmetrized correction remains conservative.

The first Ti2CO2 + O*/OH*/OOH* pilot, including frozen-checkpoint comparison,
composition-offset controls, fixed-cell filtering, and held-out metrics, is in
[`artifacts/ti_1331_pilot_results.md`](artifacts/ti_1331_pilot_results.md).
The batching, caching, residual-target, evaluation and hybrid-precision LES
benchmarks are recorded in
[`artifacts/les_au_mgo/optimization_benchmark.md`](artifacts/les_au_mgo/optimization_benchmark.md).
