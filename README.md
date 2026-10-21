# MACE–FNO: Learning nonlocal interactions using Fourier neural operator

MACE–FNO adds a learned nonlocal energy correction to a local MACE interatomic
potential. Atomic descriptors are mapped to latent source fields on a mesh.
A Fourier neural operator transforms these fields, and their contraction gives
the energy correction. Forces follow by differentiating the total energy.

The MACE backbone can be frozen or trained jointly with the correction.
A linear FNO learns a fixed propagation kernel. A nonlinear FNO makes the
response depend on the complete source field. The latent sources need not
represent physical charges.

The code supports periodic 3D systems and slabs. For slabs, Fourier transforms
act in the periodic plane, and **full z-mixing** couples channels and heights
on a finite height mesh.

## Install

Use Python 3.10 or newer in a virtual environment:

```bash
git clone https://github.com/wch3n/mace_fno.git
cd mace_fno
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -e '.[mace]'
mace-fno-train --help
```

For GPU use, install a CUDA-enabled PyTorch build appropriate for your system.
The `mace` extra is needed to train or load MACE checkpoints.

## Get started

Follow the [quick-start tutorial](TUTORIAL.md) to train a correction, evaluate
energy and force errors, and use the model in ASE. Start from one of these
configurations:

| System | Example |
|---|---|
| Periodic bulk, including variable cells | [train_bulk.yaml](examples/train_bulk.yaml) |
| Slab with full z-mixing | [train_slab.yaml](examples/train_slab.yaml) |

Both examples run for 200 steps to check the workflow. They are not converged
benchmark fits. Set the paths to your local MACE model and labelled data, then
run:

```bash
mace-fno-train --config /path/to/train.yaml
```

For inference from an existing checkpoint, see the
[ASE example](TUTORIAL.md#5-use-the-model-in-ase). Keep the original MACE model
alongside the FNO checkpoint. The calculator provides energies and forces;
stress is not yet supported.

## Benchmarks and data

| Benchmark | Physical setting |
|---|---|
| [Synthetic Yukawa](benchmarks/synthetic_yukawa/README.md) | Known screened interactions and charge correlations |
| [Inhomogeneous screening](benchmarks/heterogeneous_screening/README.md) | Spatially varying screening and remote-medium modulation of interionic coupling |
| [Liquid water](benchmarks/water_scan_qnep/README.md) | Periodic 3D liquid, SCAN reference data |
| [LLZO](benchmarks/llzo_qnep/README.md) | Ionic conductor with different cell sizes and shapes |
| [Au₂/MgO](benchmarks/au_mgo/README.md) | Slab energies, forces, and dopant-induced wetting reversal |

[Datasets and checkpoints are available on Zenodo](https://doi.org/10.5281/zenodo.23459503),
together with training configurations, data-generation scripts, and MD
trajectories. Follow the campaign links in each benchmark guide to reproduce
the reported results. The introductory configurations are separate examples.

## Further reading

- [Training and checkpoints](guides/training.md): joint training, resuming,
  fine-tuning, two-stage optimization, and model selection.
- [Geometry and operators](guides/operators.md): bulk and slab settings,
  full versus channel-wise height mixing, linear operators, and low-rank mixing.
- [Validation and diagnostics](guides/validation.md): labels, baseline errors,
  force and symmetry checks, and optional spectral probes.
- [Contributing](CONTRIBUTING.md): development setup and tests.
- [Citation](CITATION.cff): software citation metadata.
