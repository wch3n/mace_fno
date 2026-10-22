# Au₂/MgO full-z budget revision, 16 October 2026

Following the validation-curve review, the user approved finalizing MACE1 joint
seeds 41, 53, and 67 without further training and capping the ten MACE2 full-z
fits at 100,000 optimizer steps. MACE1 joint seeds 17 and 29 already completed
240,000 steps. Their final test results are reused for new wetting evaluations.
The periodic 3D comparison jobs and completed frozen-MACE1 fits are unaffected.

The campaign is stored at
`~/mace_fno_runs/_campaigns/au2_mgo_full_z_finalize_20261016/`.
Its manifest records the original campaigns, agreed budgets, input hashes,
selection rule, and Slurm predecessors. `submission.json` records the actual
replacement jobs and cancellations. Parent manifests and input configurations
remain intact. Per-task `training.yaml` changes only the total step budget.

`prepare.py` snapshots the checked continuation and evaluation helpers and
wetting structures. Preflight checks validate all 15 checkpoints before
`submit.py` changes the queue. Replacements are submitted held. Only after all
submissions succeed are duplicate continuations cancelled and MACE1 stopped.
MACE2 continues in its current allocation, then resumes natively up to 100,000
steps. Its optimizer, scheduler, RNG, and validation-selection history persist.

Each task archives its incoming checkpoints and status. Native zero-step resume
finalizes the remaining MACE1 fits, restoring the best validation-selected
weights. The three new MACE1 tests and all ten MACE2 tests use the 500 held-out
structures, without fitting an energy offset. Wetting evaluations use immutable
snapshots of those exact checkpoints, the four established starting geometries,
Au-only relaxation, and the existing force threshold and basin checks.

Results and actual stopping steps are in `results/<task>/status.json`.
Wetting/test snapshots are in `results/<task>/evaluation/results/<task>/`.
The former 240,000-step budget must not be reported as the completed budget for
these newly finalized runs. Checkpoints remain selected using validation loss,
never the held-out test or wetting results.
