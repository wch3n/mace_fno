# Submitted liquid-water G=1 replacement

**Historical submission record.** Queue states and pending imports below refer
to their dated entries. The accuracy, cost, interlacing, and convergence G=1
replacements are now imported. See [README.md](README.md) for the current scope.

Campaign: `/gpfs/home/acad/ucl-modl/wchen/mace_fno_runs/_campaigns/water_scan_groups1_complete_20261009`.
Submitted 9 October 2026. The campaign contains immutable training/source
snapshots, original and replacement configurations, input hashes, and exact
Slurm submission records (`submission_*.json`).

| Job | Purpose | Dependency |
|---|---|---|
| 16181132 | Nine GPU smoke tests; all passed on an A100-SXM4-40GB | None |
| 16181135 | 35 array tasks, at most four concurrent GPUs | Successful smoke test |
| 16181140 | Numerical and spectral audits of all 20 accuracy models | Corresponding successful training task |
| 16181141 | Seed-17 GPU runtime, memory, parameters, and interlacing | Successful frozen and joint MACE1 training tasks |
| 16181142 | Accuracy and convergence result collection | Entire training array succeeds |

The training array has 19 new accuracy fits, 15 new convergence fits, and
one evaluation of the existing frozen MACE1 seed-29 G=1 checkpoint.
The seed-29 evaluation completed successfully: energy RMSE
0.786836103584 meV/atom and force RMSE 82.427355188677 meV/angstrom
on the 134-frame clean test set. These are single-seed results.

The GPU cost job measures seed-17 frozen and joint MACE1-FNO with one and
eight mesh origins, alongside the same MACE1, MACE2, and MACE1-LES controls
as the original inference table. It uses an A100-SXM4-40GB, float64,
four CPU threads, 90 synchronized timed energy/force calls per setting,
and nine memory measurements in a fresh process per setting. Saved model
metadata records the actual parameter count and verifies G=1.

The numerical/spectral jobs use 32 clean test frames per model, sampling
seed 947, coordinate finite differences of 1e-4 angstrom, and cosine/sine
amplitude sweeps at 0.0125, 0.025, 0.05, and 0.1. Raw mode-level reports
are retained for refitting the original spectral windows.

## Manuscript transition

At submission on 9 October, the manuscript contained the G=4 water results.
Neither submission nor the result collector relabels them as G=1. After
the new fits and measurements complete, verify and replace together:

1. All 20 water FNO accuracy records and their five-seed summaries/macros.
   Local MACE and LES accuracy results are unchanged.
2. The six water convergence rows: three production reference seeds
   29/41/53, plus five variants with the same three seeds.
3. Water parameter counts, GPU costs, and interlacing rows. Recalculate
   any cross-system cost ratios or ranges that include water.
4. The standalone numerical/spectral rows, optional spectral-window and
   residual-parity plots, and any related claims. Rerun the ASE checkpoint
   round-trip checks before preserving the aggregate round-trip claim.
5. The G=1 descriptions in the main text and Appendix A, and the historical
   split/training-budget descriptions for replaced diagnostic models.

`analysis/results.json` is produced only after all 35 array tasks complete.
`measurements/` and `audits/` contain the dependent reports. The existing
`docs/source/data/water_clean/collect_results.py` still selects the original
campaigns. Its selection must be updated and the old source-data snapshots
archived when importing the completed replacement. Compile both manuscript
and standalone appendix after updating the tables and descriptions.

Failed or timed-out tasks require inspection before resubmission. The
wrappers refuse to overwrite existing run/measurement directories and do
not select checkpoints using the test set.

## Status and diagnostic recovery, 10 October 2026

Eight accuracy models completed by the morning check: all five frozen
MACE1 seeds and joint MACE1 seeds 17, 29, and 41. The seed-17 GPU cost and
interlacing job 16181141 completed successfully in 4 minutes 38 seconds.

Audit tasks 16181140_0 through 16181140_4 reached their two-hour walltime
after writing complete numerical and cosine-sweep reports. All promised
exact numerical checks passed for these five models. Their sine sweeps
did not finish. The remaining original audit tasks were extended to eight
hours using `scontrol update`. The command and result are recorded in
`audit_walltime_update_20261010.json` in the campaign.

Recovery array **16182317**, tasks 0--4 with at most two concurrent GPUs,
reruns only those unfinished sine sweeps with an eight-hour walltime.
`recover_audit.py --check-only` passed against all five existing reports
before submission. It verifies the checkpoint/cache identities, G=1,
sample counts, all numerical criteria, and the complete four-amplitude
cosine sweeps. Original numerical/cosine JSON files and failed-attempt logs
are retained. Recovery records store their checksums, and final status is
written only after the sine sweep completes and the inputs are rechecked.

## MACE2 diagnostic memory recovery, 10 October 2026 evening

All ten MACE1 accuracy fits and their numerical/spectral audits completed.
The five-seed joint G=1 test RMSEs are 0.696824 +/- 0.026991 meV/atom
and 71.777284 +/- 0.479744 meV/angstrom (sample standard deviations).

The first four frozen MACE2 fits and test evaluations also completed, but
their diagnostic tasks 16181140_10--13 exhausted the 40 GB GPU memory in
the numerical audit. The failure occurred when all 24 displaced structures
for the 12-component finite-difference check were evaluated together.

The audit now accepts `--fd-batch-size` (default 4). Recovery uses batches
of two without changing the 12 components, displacement, 32 sampled test
frames, strict thresholds, or spectral probes. All 16 audit unit/integration
tests passed, including equivalence of chunked and unchunked finite
differences. The training source snapshot and model checkpoints are unchanged.

Replacement array **16189094**, tasks 10--19 with at most two concurrent
GPUs and an eight-hour walltime, depends on the corresponding successful
training tasks. It runs a separately hashed audit/recovery snapshot under
`audit_mace2_recovery_20261010/`. Preflight passed for the four completed
MACE2 fits. The six pending original MACE2 audits 16181140_14--19 were
cancelled after the replacement was submitted. Failed-attempt logs are
preserved, and recovery uses distinct log names. Submission and patch
provenance are recorded in `submission_mace2_audit_recovery_20261010.json`
and `audit_mace2_batching_fix_20261010.json` in the campaign.

## Spectral checks discontinued, 11 October 2026

The user no longer requests water spectral checks. Cancelled array tasks
16189094_15--19: two running spectral stages and three pending combined
audits. Tasks 15 and 16 had already completed all numerical criteria. Their
numerical reports remain unchanged and have separate numerical_status.json
completion records. Earlier completed spectral reports remain archived.

Numerical-only array **16191146**, tasks 17--19, completes the three missing
joint-MACE2 numerical checks. It has a 30-minute GPU budget, at most two
concurrent tasks, and the validated finite-difference batch size of two.
All five relevant preflights passed. The new immutable wrapper and audit are
in numerical_only_20261011/, with hashed inputs and submission.json.
Cancellation is recorded in spectral_checks_cancelled_20261011.json.
No training, accuracy, convergence, or GPU measurement job was cancelled.

The manuscript now reports G=1 water accuracy and costs. Water spectral rows
and macros have been removed, and the numerical exporter no longer requires
spectral completion. Future water follow-up submissions are numerical only.
