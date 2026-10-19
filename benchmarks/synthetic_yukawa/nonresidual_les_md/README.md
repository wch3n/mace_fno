# Nonresidual charge-only LES for Fig. 4

## Completion, 13 October 2026

Both production tasks completed successfully: kappa 0.05 at 08:49 CEST and
kappa 0.5 at 08:35 CEST. Each completed 1200 training epochs, 2 ps of
equilibration, and 1000 ps of production NVT. Both trajectories passed the
completion and stability checks and provide 2001 saved frames, 15 shells,
and ten time blocks for sampling standard errors.

The dependent figure job 16205159 failed with signal 11 before writing a
stage status. Its empty staging directory was preserved as
`results/figure4_update_failed_16205159`. The unchanged update script then
completed using the verified local Python runtime. Its results are in
`results/figure4_update`. All 30 LES rows and the canonical Fig. 4 PDF were
updated, all 300 other raw rows were preserved, and both manuscript PDFs
were rebuilt without changing the prose or numerical macros.

The four existing tests, PDF text audit, and strict collision audit passed.
Both panels and insets were visually reviewed. The width warning of the source validator is resolved by the measured 183 x 86 mm PDF size. The repository
record is `docs/source/data/md_structure/nonresidual_les/verification.json`.

At the lowest shell, LES gives S_ZZ = 0.00547116 +/- 0.00006360 for
kappa 0.05 and 0.01616851 +/- 0.00023477 for kappa 0.5. The corresponding
reference values are 0.00543727 and 0.02825264. These errors are sampling
standard errors for a single trained seed.

## Protocol

Two fresh seed-17 contact-trained fits replace the ordinary LES curves at
kappa 0.05 and 0.5 inverse angstrom. These are the two conditions in Fig. 4.
Kappa 0.2 has no panel or existing matched contact-trained MD protocol.
The 810 training, 240 validation, 300 thermal-test and 45 contact-test
structures are unchanged and verified against the published dataset hashes.

The first interaction is `RealAgnosticInteractionBlock`. Other architecture,
LES, optimization and loss settings follow the original MD-specific fits:
one MACE layer, scalar final products, 1200 total epochs, stage two at 450,
batch four, float64, EMA and patience 50. Export selection uses validation
loss, never test errors or S_ZZ. Fits start fresh. Static benchmark models
are not substituted for contact-trained MD models. Finishing the epoch budget
does not itself establish convergence.

MACE and LES sources and cuEquivariance compatibility code are copied from
the completed synthetic nonresidual campaign. Per-condition GPU checks train
native and cuEq smoke models through both stages, verify complete optimizer
coverage, conversion/one-step parity and export architecture, test finite-
difference forces, and exercise the actual MD driver. The shell estimator
must reproduce all 15 existing LES shell means and sampling errors at each
kappa. Production runs depend on all checks succeeding.

Each production GPU task trains and validates its model, then runs 2 ps of
equilibration and 1 ns of NVT dynamics at 2800 K, dt 2 fs, friction 2/ps,
from test frame zero with seed 17. The unmodified ASE driver uses the same
1146 ions in a 32-angstrom cell and saves every 0.5 ps. Only calculator
construction enables the checked cuEq backend. The physical +/-0.4 e charges
define S_ZZ. Completion and stability checks precede all trajectory averages.
Fifteen shells and ten time blocks reproduce the existing estimator. Sampling
errors describe one trained seed, not training-seed variation. Existing
reference and model trajectories are preserved.

A dependent CPU job stages the figure update only after both trajectories
finish. It changes exactly 30 LES rows, preserves all other raw data, checks
the reference hashes/values, exports the existing Python figure, and runs
source, text-size, collision and manuscript data tests. The input curves
and PDF are archived. Failed rendered checks preserve the candidate for review
instead of installing it. A successful update rebuilds the manuscript without
editing prose. The final rendered figure still requires visual review.

Figure layout: preserve the existing 183 x 86 mm quantitative comparison
of charge correlations under weak and strong screening, both panels and
insets, all 15 shells, Helvetica-style type, colors, and sampling-error bars.
Interpretation follows the measured results. No expected ranking is imposed.

## Submission, 12 October 2026

Campaign: `~/mace_fno_runs/_campaigns/synthetic_les_nonresidual_md_20261012`.
GPU check array **16205157** and production array **16205158** map tasks 0
and 1 to kappa 0.05 and 0.5, respectively. Each production task performs
training, held-out evaluation, equilibration, 1 ns NVT and S_ZZ analysis on
one GPU, with a 48-hour allocation. CPU job **16205159** stages and installs
the Fig. 4 update after both production tasks succeed. Exact commands and
dependencies are saved in `submission_20261012.json` and in the campaign.

Both GPU checks passed, including the actual MD driver and exact reproduction
of the 15 saved shell means and sampling errors at each kappa. Production
tasks `16205158_0` and `16205158_1` started on GPUs `cna037` and `cna038`.
`validation_20261012.json` archives the successful check stages and exit codes.

Preparation and native configuration checks passed against the pinned MACE
sources and dataset hashes. Four existing figure-data/style tests pass.
An in-memory importer check reproduced the existing rows exactly and rejected
incomplete, unstable, wrong-block and short trajectories without writing data.
The existing-data figure preflight passes PDF text and strict collision checks
at 183 x 86 mm, with 5.6 pt minimum glyph size. Both panels and insets were
visually checked. These checks do not certify future trajectory results.
The dependent update repeats the data and rendered checks on the new curves.
