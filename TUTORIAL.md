# MACE-FNO quick-start tutorial

This tutorial shows how to train a conservative FNO correction on top of an
existing local MACE model, check that the correction behaves correctly, and use
the combined model in ASE. Generated data, caches, and checkpoints should be
kept outside the Git checkout.

The default frozen workflow is:

```text
labeled XYZ -> frozen MACE -> latent atomic sources -> mesh -> FNO -> residual energy
                       |                                      |
                       +---------- remains fixed              +-- optimized
```

The combined prediction is

\[
E = E_{\mathrm{MACE}} + \Delta E_{\mathrm{FNO}},
\qquad
\mathbf F_i = -\frac{\partial E}{\partial \mathbf R_i}.
\]

MACE supplies the baseline energy and local descriptors. In the default mode,
its weights remain fixed and only the latent-source head and FNO are optimized.
In joint mode, the same total loss is backpropagated through both branches, so
the MACE parameters can adapt to information supplied by the FNO loss. Forces
are obtained by differentiating the complete energy, so either model is
conservative by construction.

## 1. Install the package

Clone the repository, create an environment, and install the package with the
optional MACE dependency:

```bash
git clone https://github.com/wch3n/mace_fno.git
cd mace_fno
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install -e '.[mace]'
```

Install a CUDA-enabled PyTorch build appropriate for the local system before
the last command if GPU support is required. Confirm the installation with:

```bash
mace-fno-train --help
python3 -c "import torch; print(torch.cuda.is_available())"
```

## 2. Prepare the inputs

Two inputs are required:

1. a trained local MACE checkpoint;
2. extended-XYZ files containing reference total energies and atomic forces.

Use separate training, validation, and test files. For molecular-dynamics data,
split complete trajectories or contiguous trajectory blocks rather than
randomly separating neighboring frames.

The energy must be an `Atoms.info` value or an ASE calculator result. Forces
must be an `Atoms.arrays` value or calculator result. For example, custom labels
can be written as:

```python
import numpy as np
from ase.io import write

# `structures`, `energies`, and `forces` come from the reference calculations.
for atoms, energy, force in zip(structures, energies, forces, strict=True):
    atoms.info["REF_energy"] = float(energy)          # eV per structure
    atoms.arrays["REF_forces"] = np.asarray(force)   # eV/angstrom, shape (N, 3)

write("train.xyz", structures, format="extxyz")
```

The following check catches common label and periodicity problems before a GPU
job is submitted:

```python
from ase.io import read
from mace_fno.training import reference_energy, reference_forces

for filename in ("train.xyz", "validation.xyz", "test.xyz"):
    frames = read(filename, index=":")
    assert frames, f"no structures in {filename}"
    for atoms in frames:
        energy = reference_energy(atoms, "REF_energy")
        forces = reference_forces(atoms, "REF_forces")
        assert forces.shape == (len(atoms), 3)
    print(filename, len(frames), "structures", "pbc=", frames[0].pbc)
```

Choose the geometry and cell treatment from the physical system:

| System | Spatial scheme | Cell mode |
|---|---|---|
| Fixed-cell bulk | `3d` | `fixed` |
| Cubic bulk at different volumes | `3d` | `isotropic` |
| Bulk with different cell lengths or shapes | `3d` | `anisotropic` |
| Surface or slab, periodic in-plane | `2.5d` | `fixed` |

The 3D scheme requires periodicity along all three directions. The slab scheme
requires periodicity in x and y and treats z as a finite, nonperiodic response
direction.

## 3. Measure the frozen-MACE baseline

Evaluate the local model before training a correction:

```bash
mace-fno-evaluate-mace \
  --model /absolute/path/to/local_mace.model \
  --train-file /absolute/path/to/train.xyz \
  --test-file /absolute/path/to/test.xyz \
  --energy-key REF_energy \
  --forces-key REF_forces \
  --device cuda \
  --output /absolute/path/to/run/mace_baseline.json
```

This establishes whether the residual problem is meaningful. The output also
reports constant energy-offset controls; a useful FNO should improve more than
a trivial fitted offset and should improve forces as well as energies.

## 4. Run a small 3D smoke test

Create a run directory outside the repository:

```bash
export RUN_ROOT=/absolute/path/to/mace_fno_run
mkdir -p "$RUN_ROOT/cache"
```

Store the run settings in `$RUN_ROOT/train.yaml`. Paths in YAML may be absolute
or relative to the YAML file; shell variables are not expanded inside it. A
short metric-aware EqGINO configuration for heterogeneous bulk cells is:

```yaml
mace_model: /absolute/path/to/local_mace.model

data:
  train_file: /absolute/path/to/data/train.xyz
  validation_file: /absolute/path/to/data/validation.xyz
  test_file: /absolute/path/to/data/test.xyz
  energy_key: REF_energy
  forces_key: REF_forces
  train_cache: cache/train.pt
  validation_cache: cache/validation.pt
  test_cache: cache/test.pt

model:
  spatial_scheme: 3d
  cell_mode: anisotropic
  grid: 24
  z_grid: 24
  modes: 4
  z_modes: 4
  channels: 4
  fno_hidden_channels: 16
  fno_layers: 2
  spectral_symmetry: metric_eqgino
  metric_hidden_channels: 16
  output_initialization_scale: 0.1

training:
  learning_rate: 3.0e-4
  lr_scheduler: plateau
  energy_weight: 1.0
  force_weight: 1.0
  energy_scale: 0.01
  force_scale: 0.10
  batch_size: 2
  evaluation_batch_size: 4
  steps: 200
  eval_interval: 50
  evaluation_scope: validation-test
  seed: 17

runtime:
  dtype: float32
  device: cuda

checkpoint: mace_fno_3d.pt
```

Run it with:

```bash
mace-fno-train --config "$RUN_ROOT/train.yaml"
```

The example above uses the default `mace_training: frozen` mode and fits cached
DFT-minus-MACE residual targets. To fine-tune both parts against total DFT
energies and forces, add the following settings:

```yaml
training:
  mace_training: joint
  learning_rate: 3.0e-4
  mace_learning_rate: 1.0e-5
  mace_warmup_steps: 50
  output_initialization_scale: 0.1
```

The warm-up initially protects the local model while the new global branch
develops a useful signal. Joint mode then uses two optimizer parameter groups:
the FNO learning rate applies to the latent-source and field-operator branches,
whereas `mace_learning_rate` applies to MACE. Validation selects and restores
both states together.

Section names such as `data`, `model`, and `training` are organizational; every
leaf key must match a documented command-line option, using underscores or
hyphens. Unknown and duplicate keys are rejected. An explicit command-line
option overrides YAML, which is convenient for controlled repeats:

```bash
mace-fno-train --config "$RUN_ROOT/train.yaml" --seed 29 --steps 25000
```

Two hundred steps only verify that data loading, batching, differentiation, and
checkpoint writing work. They are not expected to produce a converged model.
For a serious fit, use the validation curve to choose the training length; the
current benchmarks typically use about 20,000 steps and at least three random
seeds. Train the checkpoint with `--dtype float64` when preparing the final
conservative-force finite-difference audit; float32 is useful for quick
development runs. Replace `--device cuda` by `--device cpu` when no GPU is
available.

`energy_scale` and `force_scale` normalize the two errors in the loss. Replace
the illustrative values above by approximate frozen-MACE validation RMSEs in
eV/atom and eV/angstrom. This makes equal energy and force weights easier to
interpret.

The main resolution parameters are:

- `grid` and `z-grid`: mesh points along each direction;
- `modes` and `z-modes`: retained Fourier modes; small values emphasize the
  longest wavelengths;
- `channels`: number of latent source fields;
- `fno-hidden-channels`: internal nonlinear field width;
- `metric-hidden-channels`: width of the radial network that maps physical
  \(\lvert\mathbf k\rvert^2\) to spectral channel-mixing weights.

Increase these parameters only after the small run succeeds. Grid density,
mode count, and channel width should be checked by convergence tests on the
validation set. Every retained mode count must satisfy
`2 * modes <= grid`. Once the single-origin model is stable,
`--volume-interlacing 2 --interlacing-training random` can reduce the 3D
particle-mesh egg-box error without evaluating all eight origins at every
training step; validation and inference still average all origins.

## 5. Adapt the command to a slab

For a surface or adsorption system, replace the `model` section by:

```yaml
model:
  spatial_scheme: 2.5d
  cell_mode: fixed
  grid: 24
  modes: 4
  z_grid: 16
  z_extent: 22.0
  z_center: mean
  z_mixing: global
  lateral_interlacing: 1
  channels: 4
  fno_hidden_channels: 16
  fno_layers: 2
```

The slab operator Fourier transforms only the periodic x-y plane and retains
an explicit finite z direction. `z_extent` must contain every atom in every
configuration.

Many electronic-structure files mark the vacuum direction as periodic even
though the learned response should be nonperiodic along z. Add
`--allow-periodic-z` explicitly in that case. For a square in-plane cell,
`--planar-symmetry c4` or `d4` can enforce the corresponding discrete planar
symmetry.

## 6. Evaluate the trained correction

The checkpoint records the cache paths used during training, so the held-out
sets can be evaluated with:

```bash
mace-fno-evaluate \
  --checkpoint "$RUN_ROOT/mace_fno_3d.pt" \
  --splits validation test \
  --batch-size 4 \
  --device cuda \
  --output "$RUN_ROOT/evaluation.json"
```

Compare the `frozen_mace` and `mace_fno` sections. Report energy RMSE in
eV/atom and force RMSE in eV/angstrom, and repeat the fit with multiple seeds.
The training command also writes `mace_fno_3d.config.yaml` next to the
checkpoint and embeds the same resolved configuration in the checkpoint. This
record contains the effective YAML and command-line values with absolute paths.

## 7. Run the physics audits

For a periodic 3D checkpoint:

```bash
mace-fno-audit-3d \
  --checkpoint "$RUN_ROOT/mace_fno_3d.pt" \
  --samples 8 \
  --fd-components 12 \
  --strict \
  --device cuda \
  --output "$RUN_ROOT/audit.json"
```

For a slab checkpoint, use `mace-fno-audit-2p5d` with the same pattern. These
audits check source neutrality, periodic translations, conservative forces by
finite differences, and the relevant energy/force symmetries.

The learned long-wavelength response can be examined separately:

```bash
mace-fno-audit-spectral \
  --checkpoint "$RUN_ROOT/mace_fno_3d.pt" \
  --samples 4 \
  --max-mode 2 \
  --fit-shells 3 \
  --device cuda \
  --output "$RUN_ROOT/spectral_response.json"
```

The spectral result is a diagnostic, not a training target. Agreement with a
\(1/k^2\) trend indicates an electrostatic-like low-wavevector component, but
does not make the latent fields physical charge densities.

When a checkpoint and in-training spectral diagnostics are configured, the
trainer writes the history automatically beside the checkpoint as
`<checkpoint-stem>_spectral_training.json`.

## 8. Use the combined model in ASE

Frozen-mode checkpoints store only the learned correction. Joint checkpoints
also embed the updated MACE state. In both cases, keep the original MACE file
because it defines the architecture and graph conversion used at inference:

```python
from ase.io import read
from mace_fno import MACEFNOCalculator

atoms = read("structure.xyz")
atoms.calc = MACEFNOCalculator(
    "mace_fno_3d.pt",
    mace_model_path="local_mace.model",  # overrides the path saved at training
    device="cuda",
)

energy = atoms.get_potential_energy()
forces = atoms.get_forces()

print("total energy:", energy, "eV")
print("MACE energy:", atoms.calc.results["mace_energy"], "eV")
print("FNO correction:", atoms.calc.results["residual_energy"], "eV")
print("force array:", forces.shape)
```

The ASE calculator validates periodicity and cell compatibility. It currently
provides energy and forces, but not stress.

## 9. Common problems

- **Missing labels:** confirm that `--energy-key` and `--forces-key` match the
  extended-XYZ fields exactly.
- **Cell mismatch:** use `fixed` only for one cell, `isotropic` for uniformly
  scaled cubes, and `anisotropic` for general 3D cells.
- **Slab outside the z window:** increase `z_extent` or inspect the centering
  convention.
- **Out of GPU memory:** reduce `batch-size` first, then grid or channel width.
- **Stale caches:** add `--rebuild-cache` after changing data or preprocessing.
- **Good energies but worse forces:** verify label units, loss scales, and the
  finite-difference audit before increasing model size.
- **Moved checkpoint:** pass `mace_model_path` to the ASE calculator and keep
  the recorded sample caches available when using the evaluation commands.

## Complete benchmark workflows

The maintained benchmarks provide cluster-ready examples with explicit data
provenance and validation:

- [Au2-MgO](benchmarks/au_mgo/README.md): 2D slab FNO;
- [Water-SCAN](benchmarks/water_scan_qnep/README.md): periodic 3D water;
- [LLZO](benchmarks/llzo_qnep/README.md): heterogeneous bulk cells with
  metric-aware EqGINO.

Each benchmark keeps its default FNO architecture and optimization settings in
a tracked `train_fno_*.yaml` file. Its Slurm launcher supplies runtime paths
and accepts an alternative configuration through `FNO_CONFIG`.

Their launchers keep all generated files outside the source tree through
`MACE_FNO_WORK_ROOT`.
