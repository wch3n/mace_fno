# Complete liquid-water G=1 replacement

Replace all twenty nonlinear water FNO accuracy models with one spectral
group: one- and two-interaction backbones, frozen/joint optimization, five
seeds (17, 29, 41, 53, 67). Reuse the already completed frozen MACE1 seed-29
G=1 fit after checking its protocol. Refit the fifteen water convergence
variants (five settings, seeds 29/41/53) with G=1.

The starting backbones, clean 471/52/134 split, source snapshot, loss scales,
optimizer, batch size, and validation-selection rule are preserved.
Training budgets are 120000 steps, except joint MACE1 at 200000. Only
spectral_groups changes, along with output paths and removal of old G=4
resume paths. New fits start from the original local backbone, not a G=4
FNO checkpoint. Shared caches and original campaigns are read-only.

Preparation verifies source and input hashes. GPU smoke tests exercise
eight native optimizer steps for all nine architecture/regime settings,
including the joint warm-up transition. Production verifies the full
validation history and evaluates only its lowest-loss model on the test
set. Convergence variants use validation only. No energy shifts or test
selection are used. Failed or timed-out runs require inspection.

The final collector requires all 35 results and exports per-seed values and
sample standard deviations. It never writes the manuscript automatically.
The manuscript should be switched only after the entire replacement is
verified, including the following dependent quantities:

- Water accuracy table/macros and comparisons, plus the G=1 descriptions
  in the main text and Appendix A.
- Water convergence panel, reusing the production frozen MACE1 seeds
  29/41/53 as its reference setting.
- Parameter counts, GPU runtime/memory, and seed-17 interlacing measurements.
- Optional residual parity and standalone spectral/numerical diagnostics,
  regenerated from the new checkpoints before describing them as G=1.

Keep the archived G=4 results identifiable until all replacements exist.
Do not relabel historical measurements as G=1.

## Current scope, 11 October 2026

All water accuracy and GPU cost/interlacing replacements have entered the
manuscript. Water convergence is omitted until its G=1 fits finish. The user
has discontinued water spectral checks. Only numerical checks are required
for the remaining audits, using numerical_only.py and numerical_only.slurm.
Historical spectral-recovery scripts and immutable submitted snapshots are
kept for provenance, not for resubmission. See SUBMITTED.md for cancellations
and the numerical-only replacement array.
