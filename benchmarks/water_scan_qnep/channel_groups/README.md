# Liquid-water spectral grouping: one-seed check

## Scope

This historical seed-29 comparison tests G=1 against G=4 for frozen MACE1 on
the clean SCAN-water split. The manuscript now uses the complete
[G=1 campaign](../groups1_complete/README.md). The comparison below records
the earlier controlled test and its configuration.

Only `spectral_groups` changes from 4 to 1, apart from the output checkpoint
path. The experiment preserves the reference code snapshot, backbone, data,
471/52 train/validation split, 24-cubed mesh, m=8, C=16, hidden width 16,
two Fourier layers, float64, loss scales, batch size, scheduler, seed, and
120000-step budget. Seed 29 is the first seed in the existing matched
convergence campaign. It was not selected by its error.

Both models are selected by the minimum combined validation loss. Raw energy
RMSE and force RMSE come from that same checkpoint. No test labels are loaded.
The archived G=4 control has validation RMSEs 0.4936945564 meV/atom and
75.4976005240 meV/angstrom, selected at step 118000.

Native initialization is preserved. Changing the grouping also changes the
spectral parameter count, its fan-in-dependent initialization, and random
number consumption. This tests the complete G setting under the existing
protocol. It does not isolate connectivity from initialization effects.
One seed provides a preliminary comparison, without a seed-variability estimate.

## Preparation and execution

```bash
python3 -B prepare.py --reference-campaign /path/to/physical_convergence_matched_20260930 --output /path/to/new_campaign
```

The new campaign copies the reference source and archives its control records
and raw training/validation inputs. Existing feature caches are shared read-only.
SHA-256 checks cover the source, configurations, inputs, caches, and control.
Run `preflight.py` with the matching Python packages and `source/src` of the campaign on PYTHONPATH. It verifies the full validation history of the control,
checkpoint, configuration differences, and cached data provenance.

Within the campaign, `python3 -B submit.py smoke` submits a short debug-gpu
check of eight optimizer steps with force gradients and a frozen backbone.
`python3 -B submit.py production` queues one gpu job after successful completion
of that check. The training wrapper also requires passed preflight and smoke
records. Slurm enforces a 48-hour limit, with no automatic requeue or retry.
Optimizer, scheduler, and RNG state are saved every 500 steps in model.last.pt.

On completion, `collect.py` writes analysis/comparison.json and comparison.csv
with G=1 minus G=4 differences. It does not modify manuscript tables. A failed
or timed-out job requires inspection before any restart.
