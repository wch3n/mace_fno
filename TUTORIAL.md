# MACE-FNO quick-start tutorial

We train a global FNO correction to a local MACE energy, check the resulting
forces and symmetries, and use the combined model in ASE. Store generated
data, caches, and checkpoints outside the Git checkout.

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
its weights are fixed and only the latent-source head and FNO are optimized.
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

1. a trained local MACE checkpoint.
2. extended-XYZ files containing reference total energies and atomic forces.

Use separate training, validation, and test files. For molecular-dynamics data,
assign complete trajectories or contiguous trajectory blocks to each split.
Keep neighboring frames from the same block in the same subset.

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

This gives the reference errors for the local model. The output also reports
constant energy-offset controls. Compare both energy and force errors when
assessing the FNO correction.

## 4. Run a small 3D smoke test

Create a run directory outside the repository:

```bash
export RUN_ROOT=/absolute/path/to/mace_fno_run
mkdir -p "$RUN_ROOT/cache"
```

Store the run settings in `$RUN_ROOT/train.yaml`. Paths in YAML may be absolute
or relative to the YAML file. Shell variables are not expanded inside it. A
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
  metric_parameterization: shell_spline
  spectral_groups: 1
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

Section names such as `data`, `model`, and `training` organize the file. Every
leaf key must match a documented command-line option, using underscores or
hyphens. Unknown and duplicate keys are rejected. An explicit command-line
option overrides YAML, which is convenient for controlled repeats:

```bash
mace-fno-train --config "$RUN_ROOT/train.yaml" --seed 29 --steps 25000
```

Two hundred steps only verify that data loading, batching, differentiation, and
checkpoint writing work. They are not expected to produce a converged model.
For a production fit, follow the validation curve within a declared step budget
and compare independent seeds. The manuscript uses seeds 17, 29, 41, 53, and
67 for the main FNO comparisons. Budgets depend on the system and architecture.
Use the corresponding benchmark records for reproduction. Train with
`--dtype float64` for the final finite-difference force checks. Float32 is
useful for short development runs. Replace `--device cuda` by `--device cpu`
when no GPU is available.

`energy_scale` and `force_scale` normalize the two errors in the loss. Replace
the illustrative values above by approximate frozen-MACE validation RMSEs in
eV/atom and eV/angstrom. This makes equal energy and force weights easier to
interpret.

The main resolution parameters are:

- `grid` and `z_grid`: mesh points along the lateral and third directions.
- `modes` and `z_modes`: Fourier cutoff along these directions.
- `channels`: number of latent source fields.
- `fno_hidden_channels`: number of hidden channels in the nonlinear operator.
- `spectral_groups`: number of channel groups in the spectral matrix. Use
  `1` for coupling among all channels, as in the current physical benchmarks.
- `metric_parameterization`: `shell_spline` for independent radial-shell
  matrices. The alternative `radial_mlp` uses a shared radial network whose
  width is set by `metric_hidden_channels`.

Increase these parameters only after the small run succeeds. Grid density,
mode count, and channel width should be checked by convergence tests on the
validation set. Every retained mode count must satisfy
`2 * modes <= grid`, and similarly for `z_modes` and `z_grid`. Once the
single-origin model is stable,
`--volume-interlacing 2 --interlacing-training random` can reduce the 3D
particle-mesh egg-box error without evaluating all eight origins at every
training step. Validation and inference still average all origins.

## 5. Adapt the command to a slab

For a surface or adsorption system, replace the `model` section by:

```yaml
model:
  spatial_scheme: 2.5d
  cell_mode: fixed
  grid: 24
  modes: 4
  z_grid: 24
  z_extent: 22.0
  z_center: mean
  z_mixing: full
  architecture: nonlinear
  spectral_symmetry: metric_eqgino
  metric_parameterization: shell_spline
  spectral_groups: 1
  planar_symmetry: none
  lateral_interlacing: 1
  channels: 4
  fno_hidden_channels: 16
  fno_layers: 2
```

The slab operator Fourier transforms only the periodic plane. Full z-mixing
uses a wavelength-dependent matrix over channels and heights. It therefore
couples different heights and channels in the spectral branch. A pointwise
channel branch is added before GELU in each layer. The height direction is
finite and has no periodic wrapping. `z_extent` must contain every atom in
every configuration.

Many electronic-structure files mark the vacuum direction as periodic even
though the learned response should be nonperiodic along z. Add
`--allow-periodic-z` explicitly in that case. The metric-aware operator already
has D4 symmetry on a square cell with equal in-plane mesh sizes and mode
cutoffs. Keep `planar_symmetry: none` for this configuration.

The earlier channel-wise height scheme is available as `z_mixing: global`.
It uses a separate height matrix within each channel and costs less than
dense full z-mixing. A low-rank factorization is also available for full
mixing. See the [slab operator description](README.md#2d-fno-for-slabs) for
the parameter counts and the distinction between the two linear slab
implementations. The small example here checks the workflow. It does not
reproduce the full Au₂/MgO training campaign.

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

## 7. Check forces and symmetries

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

An optional spectral diagnostic examines the learned long-wavelength response:

```bash
mace-fno-audit-spectral \
  --checkpoint "$RUN_ROOT/mace_fno_3d.pt" \
  --samples 4 \
  --max-mode 2 \
  --fit-shells 3 \
  --device cuda \
  --output "$RUN_ROOT/spectral_response.json"
```

This diagnostic is not required for the current liquid-water workflow.
It does not enter training or checkpoint selection. Agreement with a
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

- [Au2-MgO](benchmarks/au_mgo/README.md): 2D slab FNO.
- [Water-SCAN](benchmarks/water_scan_qnep/README.md): periodic 3D water.
- [LLZO](benchmarks/llzo_qnep/README.md): heterogeneous bulk cells with
  metric-aware EqGINO.

The introductory launchers read tracked `train_fno_*.yaml` files and accept an
alternative through `FNO_CONFIG`. These files include earlier configurations.
For manuscript reproduction, use the campaign links in each benchmark guide
and the saved resolved configuration of the selected checkpoint.

Their launchers keep all generated files outside the source tree through
`MACE_FNO_WORK_ROOT`.
