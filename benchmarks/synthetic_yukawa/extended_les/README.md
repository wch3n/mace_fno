# Extended LES: homogeneous Yukawa at kappa = 0.5 A^-1

## Scope

This exploratory calculation tests static errors and charge correlations for
the extended LES architecture used in the heterogeneous-medium study. It is
outside the paper. The inputs are the published synthetic datasets,
the pinned extended-LES environment, and the established
`yukawa_L32_md2800/md` driver and S_ZZ analysis. The dated submission records
below describe the original jobs. Completion evidence is stored in the campaign.

Two independent seed-17 fits use the matched nonresidual first interaction block,
one MACE layer, 128 scalar plus 128 vector channels, permanent monopoles/dipoles/
quadrupoles and induced charges/dipoles. All response readouts are optimized.
Training uses cuEquivariance, float64, batch four, up to 1200 total epochs, stage two
from epoch 450, and validation-loss early stopping after 50 epochs. Hitting the
budget is recorded as unconverged. Test errors or SZZ never select a checkpoint.

The static fit uses 660/240/300 training/validation/test structures. The MD fit
uses 660 thermal plus 150 close-contact training structures, 240 validation
structures, 300 thermal test structures and 45 held-out contact tests. Data hashes
must match `benchmarks/synthetic_yukawa/checksums.sha256`.

An isolated GPU check trains four epochs on small subsets of both datasets,
crosses stage two, checks optimizer coverage and updates of every response head,
compares native and cuEq energies/forces/latent outputs, tests finite-difference
forces and exercises the actual MD driver. Production jobs depend on this check.

After the MD fit, use the original ASE Langevin protocol: test frame zero,
seed 17, 2 ps equilibration, then 1 ns at 2800 K, dt 2 fs, friction 2 ps^-1,
1146 ions in a 32 A cell. Save every 0.5 ps. A separate 100 ps NVE diagnostic
starts from the same equilibrated state. The integrator and logging are copied
unchanged. Only calculator construction enables the verified cuEq backend.

SZZ uses the fixed reference charges (+/-0.4 e) from the initial state of the trajectory,
not learned latent charges. Use the existing 15 reciprocal shells and ten time
blocks for the sampling SE, propagate the reference SE into model/reference
ratios, and reject incomplete or unstable trajectories from the final summary.
These are sampling errors for one trained model, not training-seed variability.
No finite-cell result establishes the k->0 limit.

The reference and existing MACE/LES/FNO curves are preserved in a hashed CSV
snapshot. Existing ordinary LES MD differs in first-block/local representation.
That comparison cannot isolate the effect of induced response alone. A matched
nonresidual static ordinary-LES campaign is already running separately.
No manuscript data or figures are replaced automatically.

## Submission, 2026-10-11

Campaign: `~/mace_fno_runs/_campaigns/extended_les_yukawa_k0p5_seed17_20261011`.
GPU check 16195342 passed in seven minutes. Both models contain 408,704 parameters.
Native/cuEq output parity and finite-difference force checks passed. The analysis
reproduced all 15 existing ordinary LES SZZ means and sampling errors.

Training array 16195345 maps task 0 to the static fit and task 1 to the
contact-trained fluid fit. Both started on `gpu`. Job 16195346 depends on task 1
and performs equilibration, 1 ns NVT and SZZ analysis. Job 16195347 then performs
the 100 ps NVE diagnostic from the same equilibrated state.
`submission_20261011.json` preserves the exact submission commands and check records.
The campaign writes `results/szz.json`, `results/szz_comparison.csv`, and
`results/nve.json`. Each fit writes test errors in its own `evaluation.json`.

1200-epoch continuation, 2026-10-11: array **16195548** replaces 16195345 and
continues the static and fluid checkpoints after epochs 44 and 36, respectively,
with a total cap of 1200 epochs. Job 16195346 depends on 16195548_1. The NVE
dependency remains unchanged. `budget_resume.json` preserves the original
configuration, logs, checkpoints and certificate binding. Model, optimizer and
learning-rate state are verified before resuming at the next epoch. Native
checkpointing does not store EMA history, RNG state or patience counters.
The restart therefore is not bitwise identical to uninterrupted training.
