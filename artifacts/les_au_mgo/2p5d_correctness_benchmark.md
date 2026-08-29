# Au2-MgO 2.5D FNO correctness benchmark

Date: 2026-08-29

## Purpose

This benchmark supplements held-out energy and force RMSEs with checks that a
learned residual is a physically admissible atomistic energy: conservative
forces, neutral sources, rigid-translation behaviour, the acoustic sum rule,
and the square-plane point-group symmetries of the MgO(001) cell.

## Findings from the previous checkpoint

The 20,000-step local-z, cell-centred checkpoint achieved a held-out test RMSE
of 0.1433 meV/atom and 15.1378 meV/Angstrom. Its forces agree with finite
differences of the learned energy, and its four source channels are neutral.
It nevertheless has two representation defects:

- Cell-centred z coordinates change under a rigid normal translation. On one
  held-out structure, a 0.1 Angstrom rigid shift changed the residual energy by
  4.11 meV and produced a net normal residual force of 36.19 meV/Angstrom.
- The unconstrained planar spectral weights violate square-cell symmetry. A
  90-degree rotation changed the residual energy by 33.07 meV and gave a force
  equivariance RMSE of 29.08 meV/Angstrom, while frozen MACE was invariant to
  numerical precision.

The audit is stored in `2p5d_cell_center_audit.json`.

## Global-z and mean-centred ablation

Replacing the local z CNN with a dense nonperiodic global-z mixer and using the
mean atomic height as the z origin gives:

| Model | Test E RMSE (meV/atom) | Test F RMSE (meV/Angstrom) |
|---|---:|---:|
| Frozen one-layer MACE | 2.3119 | 56.9900 |
| Local-z, cell-centred 2.5D FNO | 0.1433 | 15.1378 |
| Global-z, mean-centred 2.5D FNO | 0.1103 | 13.6571 |

The mean-centred model has zero normal acoustic-sum-rule error to numerical
precision and is exactly invariant to normal translations. Its 32-structure
audit still finds lateral net residual-force RMS values of 21.56 and 18.51
meV/Angstrom, up to 24.15 and 21.25 meV energy changes under a 0.1 Angstrom
lateral rigid shift, a 17.05 meV/Angstrom C4 force-equivariance error, and an
8.18 meV/Angstrom reflection-equivariance error. These are mesh/architecture
errors, not force-gradient inconsistencies. The audit is stored alongside the
checkpoint as `les_au_mgo_fno_2p5d_global_mean_seed17_z16_float64_audit.json`.

## Corrected model

The corrected Au2-MgO model combines:

1. mean-z centring for exact rigid normal-translation invariance;
2. dense global-z mixing so every finite z layer communicates in one block;
3. a conservative 2 x 2 interlaced lateral mesh to suppress egg-box forces;
4. exact D4 group averaging at validation/inference for fourfold rotations and
   in-plane reflections; and
5. deterministic cycling through one D4 image per optimizer forward, avoiding
   the eightfold force-training cost while providing balanced augmentation.

The final two-seed accuracy and strict-audit results will be appended after the
running jobs complete.
