# Training and checkpoints

[Quick start](../TUTORIAL.md) · [Operators](operators.md) · [Validation](validation.md)

Use this guide to configure joint training, checkpoint selection, continuation,
and changes to the training objective. For all options, run `mace-fno-train --help`.

## Configuration files

Pass a YAML file through `mace-fno-train --config train.yaml`. Relative paths
are resolved from the YAML file, and shell variables are not expanded.
Section names such as `data`, `model`, and `training` organize the file. Leaf
keys match command-line options, using underscores or hyphens. Unknown options
and duplicate leaf keys across sections are rejected. Explicit command-line
options override YAML values:

```bash
mace-fno-train --config train.yaml --seed 29 --device cpu
```

The trainer writes the resolved options to `model.config.yaml` and embeds them
in the checkpoint. This record includes absolute input paths.

`energy_scale` and `force_scale` normalize the errors in the loss. A starting
choice is the approximate local-MACE validation RMSE in eV/atom and eV/angstrom,
respectively. These scales and the relative loss weights require validation.

## Frozen-backbone training

Frozen residual training is the default. MACE weights are fixed, but
the autograd graph includes descriptor derivatives with respect to atom
positions. Detaching or evaluating the descriptors under `torch.no_grad()` would
omit part of the residual force.

Start from the [bulk](../examples/train_bulk.yaml) or
[slab](../examples/train_slab.yaml) example and follow the
[tutorial](../TUTORIAL.md).

## Data and graph caches

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
files while preserving the saved cache, except when the process
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

## Energy-prioritized checkpoints

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
when resuming a run that did not save these candidates.

## Resuming training

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
automatic Slurm requeue is performed. Best-model or weights-only files
cannot provide a full resume. Extending a completed run continues its saved
state, including the final validation/scheduler update, so it can differ from
a run originally configured with a longer budget and a different validation
schedule at that boundary.

## Checkpoint-initialized fine-tuning

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
preserved. Selected checkpoints without a MACE fingerprint require their
original reference file to be accessible. Parent path, SHA-256, and step
are recorded in the new checkpoints. Output paths must not overwrite the parent.

`init_from` and `resume` are mutually exclusive. To exactly resume an interrupted
fine-tuning stage, remove `init_from` from its YAML and set `resume` to the
new `model.last.pt` of that stage. Its parent provenance is preserved. This does not
relax the unchanged-loss requirement of ordinary `resume`.

## Two-stage optimization

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

The schedule is **off by default**. Runs without a stage-two setting use one
objective throughout training. Resuming requires the saved schedule. To change
the schedule, initialize a separate run with `init_from` and distinct output
paths.

## Joint training

Joint training uses the same total model,

```text
atoms -> MACE descriptors -> latent sources -> mesh -> FNO -> residual energy
   |          |                                               |
   +----------+---------- MACE energy -------------------------+-> total energy
```

but optimizes the total reference energy-and-force loss through both branches. Enable
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
