# Validation and diagnostics

[Quick start](../TUTORIAL.md) · [Training](training.md) · [Operators](operators.md)

Energy and force errors measure prediction accuracy. Numerical checks test
whether the implementation obeys its force and symmetry identities. Spectral
probes characterize the learned response and are optional.

## Reference labels

Use eV for total energies, angstrom for positions, and eV/angstrom for forces.
Custom labels can be written to extended XYZ as follows:

```python
import numpy as np
from ase.io import write

# `structures`, `energies`, and `forces` come from the reference calculations.
for atoms, energy, force in zip(structures, energies, forces, strict=True):
    atoms.info["REF_energy"] = float(energy)
    atoms.arrays["REF_forces"] = np.asarray(force)

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

## Local-MACE baseline

Evaluate the local model before training a correction:

```bash
mace-fno-evaluate-mace \
  --model /absolute/path/to/local_mace.model \
  --train-file /absolute/path/to/train.xyz \
  --test-file /absolute/path/to/validation.xyz \
  --energy-key REF_energy \
  --forces-key REF_forces \
  --device cuda \
  --output /absolute/path/to/run/mace_validation.json
```

Here, `--test-file` points to validation data so these errors can guide loss
scaling. Reserve the test set for the final accuracy comparison. The output
also reports constant energy-offset controls. Compare both energy and force errors when
assessing the FNO correction.

## Forces and symmetries

For a periodic 3D model, run:

```bash
mace-fno-audit-3d \
  --checkpoint /absolute/path/to/run/model.pt \
  --samples 8 \
  --fd-components 12 \
  --strict \
  --device cuda \
  --output /absolute/path/to/run/audit.json
```

For a slab, use `mace-fno-audit-2p5d`. The commands check source neutrality,
translations, finite-difference forces, and the applicable energy/force
symmetries. Train with `dtype: float64` for quantitative finite-difference
checks. A short training run can test the workflow but does not establish
converged accuracy or stability in molecular dynamics.

The precise symmetry guarantees depend on the cell, grid, and Fourier cutoffs.
See the [operator guide](operators.md#fully-periodic-3d), especially for unequal
mesh dimensions and variable cells.

## Spectral diagnostics

The post-deposition response can be probed without adding a spectral training
loss:

```bash
mace-fno-audit-spectral \
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

## Common problems

| Symptom | Check |
|---|---|
| Missing energy or force labels | Match `energy_key` and `forces_key` to the XYZ fields. |
| Cell mismatch | Use the [cell mode](operators.md) appropriate to the dataset. |
| Atoms outside the slab window | Check `z_extent` and the centering convention. |
| Out of GPU memory | Reduce batch size first, then inspect grid and channel sizes. |
| Stale graph caches | Rebuild from the source data with `--rebuild-cache`. |
| Good energies but worse forces | Check label units, loss scales, and finite-difference forces. |
| Relocated files | Override `mace_model_path` in ASE and cache paths in `mace-fno-evaluate`. |

Do not delete shared caches until their training and evaluation jobs have
finished. Keep the source data, original MACE model, training configuration,
and FNO checkpoints. See [checkpoint and cache handling](training.md#data-and-graph-caches).
