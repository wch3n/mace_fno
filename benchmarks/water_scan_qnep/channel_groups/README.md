# Liquid-water spectral grouping: one-seed check

## Material Passport

- Workflow: academic-research-suite experiment-agent, run mode.
- Scope: frozen MACE1, clean SCAN-water split, seed 29, G=1 versus existing G=4.
- Status: inspect preflight.json, smoke.json, submission records, and run status.
- Question: does using a single spectral group improve validation accuracy
  under the existing training protocol?

Only `spectral_groups` changes from 4 to 1, apart from the output checkpoint
path. The experiment preserves the reference code snapshot, backbone, data,
471/52 train/validation split, 24-cubed mesh, m=8, C=16, hidden width 16,
two Fourier layers, float64, loss scales, batch size, scheduler, seed, and
120000-step budget. Seed 29 is the first seed in the existing matched
convergence campaign; it was not selected by its error.

Both models are selected by the minimum combined validation loss. Raw energy
RMSE and force RMSE come from that same checkpoint. No test labels are loaded.
The archived G=4 control has validation RMSEs 0.4936945564 meV/atom and
75.4976005240 meV/angstrom, selected at step 118000.

Native initialization is preserved. Changing the grouping also changes the
spectral parameter count, its fan-in-dependent initialization, and random
number consumption. This tests the complete G setting under the existing
protocol; it does not isolate connectivity from initialization effects.
One seed provides a preliminary comparison, without a seed-variability estimate.

## Preparation and execution

```bash
python3 -B prepare.py --reference-campaign /path/to/physical_convergence_matched_20260930 --output /path/to/new_campaign
```

The new campaign copies the reference source and archives its control records
and raw training/validation inputs. Existing feature caches are shared read-only.
SHA-256 checks cover the source, configurations, inputs, caches, and control.
Run `preflight.py` with the matching Python packages and the campaign's
`source/src` on PYTHONPATH. It verifies the control's full validation history,
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
