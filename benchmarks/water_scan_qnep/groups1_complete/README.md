# Complete liquid-water G=1 replacement

Replace all twenty nonlinear water FNO accuracy models with one spectral
group: one- and two-interaction backbones, frozen/joint optimization, five
seeds (17, 29, 41, 53, 67). Reuse the already completed frozen MACE1 seed-29
G=1 fit after checking its protocol. Refit the fifteen water convergence
variants (five settings, seeds 29/41/53) with G=1.

The starting backbones, clean 471/52/134 split, source snapshot, loss scales,
optimizer, batch size, and validation-selection rule are preserved.
Training budgets are 120000 steps, except joint MACE1 at 200000. Only
spectral_groups changes, along with output paths and removal of G=4
resume paths. The fits start from the original local backbone, not a G=4
FNO checkpoint. Shared caches and original campaigns are read-only.

Preparation verifies source and input hashes. GPU smoke tests exercise
eight native optimizer steps for all nine architecture/regime settings,
including the joint warm-up transition. Production verifies the full
validation history and evaluates only its lowest-loss model on the test
set. Convergence variants use validation only. No energy shifts or test
selection are used. Failed or timed-out runs require inspection.

The final collector requires all 35 results and exports per-seed values and
sample standard deviations. It never writes the manuscript automatically.
Import requires complete verification of the following dependent quantities:

- Water accuracy table/macros and comparisons, plus the G=1 descriptions
  in the main text and Appendix A.
- Water convergence panel, reusing the production frozen MACE1 seeds
  29/41/53 as its reference setting.
- Parameter counts, GPU runtime/memory, and seed-17 interlacing measurements.
- Optional residual parity and standalone spectral/numerical diagnostics,
  regenerated from the G=1 checkpoints before describing them as G=1.

Archived grouped results keep their original model identities. They do not
supply the manuscript entries.

## Completed imports

The twenty accuracy fits, GPU costs, and interlacing results were imported on
11 October. All fifteen convergence variants and three production controls
completed 120,000 steps and entered the appendix on 12 October. The manuscript
uses G=1 throughout. Further water spectral checks have been discontinued.
Use `numerical_only.py` and `numerical_only.slurm` for numerical checks.
Spectral-recovery scripts and submitted snapshots are separate from the
numerical-only checks. [SUBMITTED.md](SUBMITTED.md) records submissions and
cancellations. The local manuscript evidence is documented in
`docs/source/data/water_groups1/README.md`.
