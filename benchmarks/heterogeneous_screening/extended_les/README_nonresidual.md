# Nonresidual first block for extended LES

Fresh five-seed fits (17, 29, 41, 53, 67) use
`interaction_first: RealAgnosticInteractionBlock`, matching the first-block choice
of the synthetic local MACE and ordinary LES reruns. All other training settings
are inherited from the corrected five-seed extended LES campaign: 660/240
train/validation structures, up to 1200 total epochs, stage two at epoch 450, patience
50, float64, batch four, EMA, and cuEquivariance. All extra multipole and response
readouts are included in the optimizer. The scalar/vector product representation and
permanent/induced response settings stay unchanged. Matching this block does not
make extended LES identical in local representation or parameter count to
ordinary charge-only LES.

`prepare_nonresidual.py ROOT --parent PARENT` creates separate configurations,
preserves the parent manifest and configs, and reuses the pinned environment and
immutable datasets. It never resumes a residual checkpoint into the changed
architecture. The nonresidual-first model has 485,376 parameters and the
residual-first model has 337,920. Only the shape of `interactions.0.skip_tp.weight` changes between native model constructors.

The nonresidual architecture requires its own cuEquivariance certificate.
The GPU check compares energy, forces and all latent outputs before and
after conversion and one optimizer step, checks complete optimizer coverage,
then runs four training epochs through stage two, export/reload and a
finite-difference force test. Five production jobs depend on successful checks.
Each fit evaluates the unchanged test set, 40 remote-medium probe sets and 80
ionic null variants. The existing collector records seed means/sample SD and
whether validation early stopping was reached. Manuscript results remain unchanged.
When the residual-first summary is available, collection also writes per-seed
differences and their mean/sample SD for energy/force RMSE, probe slope, force
constant RMSE, and null-response RMS in `results/first_block_comparison.json`.

On 2026-10-11 the user matched the total budget to 1200 epochs for both
first-block choices. Array 16195547 continues the five nonresidual checkpoints
(saved epochs 84--152), replacing 16194803. Collector 16194804 depends on
16195547 and the residual collector 16193550. See `budget_resume.json` in the
campaign and `README_response.md` for checkpoint provenance and restart semantics.
