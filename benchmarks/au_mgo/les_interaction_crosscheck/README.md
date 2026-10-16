# Au₂/MgO LES first-interaction cross-check

One fresh seed-17 fit changes `interaction_first` from
`RealAgnosticInteractionBlock` to `RealAgnosticResidualInteractionBlock`,
the block found in the published LES checkpoint. The completed scalar–vector
seed-17 fit is the control. All other training configuration values, data splits,
and MACE/LES source files are preserved. Only the model name also changes.

The model retains `128x0e+128x1o` product outputs, self-interaction removal,
Ewald normalization 90.4756, float64, and the native MACE training loop.
The budget is 1000 epochs, with stage two starting at epoch 500.
The expected exported parameter count is 266,496.
This is a controlled interaction-block comparison, not a full reproduction
of the published training protocol; that checkpoint also differs in LES settings.

`campaign.py prepare ROOT --parent TRAINING --wetting CONTROL_WETTING
--published PUBLISHED_EVALUATION` freezes source/data and records checksums.
Run `sbatch --output=ROOT/logs/%j.out job.slurm ROOT` on the GPU partition.
The job verifies inputs, trains a two-epoch/two-stage smoke model, checks saved
architectures and configuration round trips, then starts production training.
Any failed smoke check prevents the production fit.

After training, the existing evaluator measures validation/test energy and force
RMSE without an energy shift. Four Au-only FIRE relaxations use the same DFT
starting structures and 0.01 eV/Å threshold as Table VII. Substrate positions,
cell, convergence, and wetting/non-wetting orientation are checked.
`comparison.json` reports the new fit, the seed-17 control, and the published
checkpoint. `runs/les_nl1_seed17/status.json` tracks the current stage.
No manuscript results are replaced automatically. Reaching the epoch budget
does not itself establish convergence.

For cuEquivariance, prepare a separate campaign with `--enable-cueq
--parity-reference PATH_TO_E3NN_SMOKE_MODEL`. First submit `job.slurm ROOT
validate-cueq` (a short GPU job). This checks float64 energy/force/latent-charge
parity, conversion round trips, one Adam step, all-parameter optimizer coverage,
and batch-10 training-step timing. It then exercises both training stages and
e3nn export. Production `job.slurm ROOT run` requires these checks to pass.
The process-local converter wrapper preserves explicit LES settings; frozen
MACE/LES source remains byte-identical to the control. This variant starts fresh
from seed 17 and does not transfer optimizer state between backends.
