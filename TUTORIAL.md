# Train and use a MACE–FNO model

This tutorial trains a short frozen-backbone model, compares it with local
MACE, and evaluates a structure in ASE. Install the package using the
[README instructions](README.md#install) first.

## 1. Prepare the inputs

You need a local MACE checkpoint and three extended-XYZ files: training,
validation, and test data. The examples expect these labels:

| Quantity | Field | Unit |
|---|---|---|
| Total energy per structure | `REF_energy` | eV |
| Atomic forces, shape `(N, 3)` | `REF_forces` | eV/angstrom |
| Positions and cell | Standard XYZ fields | angstrom |

Use independent validation and test sets. For trajectory data, keep neighboring
frames together when splitting by trajectory or contiguous block. Validation
selects the model; test data are evaluated before and after training.
See [label preparation and checks](guides/validation.md#reference-labels) if your
files use different keys.

## 2. Choose a configuration

From the repository root, copy an example to a run directory:

```bash
export RUN_ROOT="$HOME/mace_fno_example"
mkdir -p "$RUN_ROOT"
cp examples/train_bulk.yaml "$RUN_ROOT/train.yaml"
```

For a surface or slab, copy `examples/train_slab.yaml` instead. Edit
`$RUN_ROOT/train.yaml` to set:

- `mace_model`: the local MACE checkpoint.
- `train_file`, `validation_file`, and `test_file`: the labelled data files.
- `energy_key` and `forces_key`: the label names, if different from the defaults.
- `energy_scale` and `force_scale`: approximate local-MACE validation RMSEs in
  eV/atom and eV/angstrom. The supplied values are illustrative.

Relative paths are resolved from the YAML file. Cache and checkpoint paths in
these examples therefore stay inside the run directory. Shell variables are
not expanded in YAML. The [baseline command](guides/validation.md#local-mace-baseline)
can help set the loss scales.

The bulk example requires periodicity in all three directions. It uses
`cell_mode: anisotropic` to accept different cell sizes and shapes.
The slab example is periodic in-plane and uses full z-mixing on a finite
height mesh. Set `z_extent` to contain every atom in every configuration. Set
`allow_periodic_z: true` if the XYZ files mark the vacuum direction as periodic.
See the [geometry guide](guides/operators.md) for other cell treatments.

## 3. Train the correction

```bash
mace-fno-train --config "$RUN_ROOT/train.yaml"
```

The examples keep MACE fixed and train the source head and nonlinear FNO.
`device: auto` uses a GPU when available. Double precision supports
finite-difference force checks.

**The 200-step budget checks the workflow; it does not produce a converged fit.**
For a longer run, choose a budget from validation behavior and repeat with
independent seeds. Explicit command-line options override the YAML:

```bash
mace-fno-train --config "$RUN_ROOT/train.yaml" --steps 25000 --seed 29 \
  --checkpoint "$RUN_ROOT/seed29/model.pt"
```

The main outputs are:

| File | Purpose |
|---|---|
| `model.pt` | Best model selected by validation loss |
| `model.last.pt` | Latest training state for continuation |
| `model.config.yaml` | Resolved training settings |
| `cache/` | Preprocessed samples for training and evaluation |

For an interrupted run, continue from its latest state:

```bash
mace-fno-train --config "$RUN_ROOT/train.yaml" --resume "$RUN_ROOT/model.last.pt"
```

Keep the model, data, and training settings unchanged when resuming. `steps`
is the total target. See [training and checkpoints](guides/training.md) for
joint training, fine-tuning, early stopping, and two-stage optimization.

## 4. Evaluate the model

```bash
mace-fno-evaluate \
  --checkpoint "$RUN_ROOT/model.pt" \
  --splits validation test \
  --batch-size 1 \
  --output "$RUN_ROOT/evaluation.json"
```

Compare the `frozen_mace` and `mace_fno` errors in the output. Energy RMSE is
reported in eV/atom and force RMSE in eV/angstrom. This command loads the sample
caches recorded by training, so keep them until evaluation is complete.

Before using a trained model for simulations, run the
[force and symmetry checks](guides/validation.md#forces-and-symmetries).
Accuracy and simulation stability require validation beyond this short run.

## 5. Use the model in ASE

Keep the original MACE model as well as the FNO checkpoint. Frozen checkpoints
store the correction; joint checkpoints also store jointly trained MACE weights, but
still need the original model to reconstruct the architecture.

```python
import os
from pathlib import Path

from ase.io import read
from mace_fno import MACEFNOCalculator

run = Path(os.environ["RUN_ROOT"])
atoms = read("/absolute/path/to/structure.xyz")
atoms.calc = MACEFNOCalculator(
    run / "model.pt",
    mace_model_path="/absolute/path/to/local_mace.model",
    device="cpu",  # use "cuda" for GPU inference
)
print("Energy:", atoms.get_potential_energy(), "eV")
print("Forces:", atoms.get_forces(), "eV/angstrom")
```

The calculator checks cell and periodicity compatibility. It also exposes
`mace_energy` and `residual_energy` in `atoms.calc.results`. Stress is not yet
supported.

For numerical checks or troubleshooting, use the
[validation guide](guides/validation.md). To reproduce the paper, follow the
[benchmark workflows](README.md#benchmarks-and-data) and their saved training
configurations.
