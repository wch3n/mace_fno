# Scalar-vector MACE2-LES reruns for Au₂/MgO

Five fresh fits use seeds 17, 29, 41, 53 and 67, two interactions, a 5.5 Å cutoff,
and `128x0e+128x1o` at both product outputs. The first interaction is
`RealAgnosticInteractionBlock`. The second is residual. Local parameter shapes
must match the scalar-vector MACE2 baseline used for the new full-z FNO fits.

The protocol matches scalar-vector MACE1-LES: 1,000 epochs, stage two from
epoch 500, batch size 10, float64, unchanged charge-only LES settings and
4,275/225/500 training/validation/test structures. The original nested MACE2-LES
campaign timed out for all five seeds after roughly 812–827 epochs. Its saved
checkpoints lack EMA/RNG histories and no final models or evaluations were
exported. Those files are preserved. New fits start independently from scratch.

The only change to the original MACE2-LES training configuration is enabling
cuEquivariance. Before any production fit, a separate GPU validation job checks
energy, forces, latent charges and LES-energy parity, conversion round trips,
optimizer coverage, one Adam step, both training stages and e3nn model export.
The converter wrapper preserves LES settings and the last-layer irreps.
The original campaign source is copied unchanged and all inputs are hashed.

Each production job automatically evaluates the native stage-two model selected
by validation loss on 225 validation and 500 test structures, without energy
shifts. It then performs the four Table VII Au-only wetting relaxations with
the fixed substrate/cell, 0.01 eV/Å force tolerance and at most 500 FIRE steps.
Endpoint checks require convergence and the intended orientation basin. Failed
basin checks are reported for review. No test or wetting metric selects a model.

Campaign: `~/mace_fno_runs/_campaigns/au2_mgo_matched_mace2_les_cueq_5seed_20261016/`.
`submission.json` records GPU jobs. Each `runs/les_nl2_seed*/status.json` records
progress. Test results go to `evaluation/metrics.json` and wetting results to
`wetting/results.json` under each run. Manuscript rows are not automatically
replaced before the new results are checked.

Validation job **16275151** passed on an A100 40 GB. The cuEquivariance
training-step benchmark measured 0.134 s versus 0.547 s for e3nn (4.08×).
Energy/force parity, optimizer coverage, two-stage training and model export
checks passed. Production array **16275152**, tasks 0–4 in seed order, was
submitted on `gpu` with a 48-hour limit and an `afterok` validation dependency.
The JSON reports in this directory record these checks and the submission.
