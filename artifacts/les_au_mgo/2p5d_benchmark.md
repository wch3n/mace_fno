# LES Au2-MgO: 2D versus hybrid 2.5D FNO benchmark

Date: 2026-08-29

## Question

Does retaining the surface-normal structure improve the conservative FNO
residual relative to the established planar 2D correction on the Au2-MgO LES
benchmark?

## Matched protocol

- Dataset: 4,500 upstream training structures and 500 untouched test
  structures, each with 110 atoms and the same
  9.0474 x 9.0474 x 26.4589 Angstrom cell.
- Frozen backbone: `Au2-MgO_r5.5_nl0_stagetwo.model`.
- Split: the same 225 MACE seed-123 validation indices, leaving 4,275 FNO-fit
  structures.
- Optimisation: 1,500 updates, batch size 4, learning rate 3e-4, float64,
  seeds 17 and 29, and validation every 250 updates.
- Shared operator settings: 24 x 24 lateral grid, four in-plane Fourier modes,
  four neutral source channels, 16 hidden channels, two nonlinear FNO blocks,
  and zero-residual initialization.
- Shared normalized loss: energy scale 2.260 meV/atom, force scale
  56.280 meV/Angstrom, with equal normalized energy and force weights.
- 2.5D-only settings: 16 finite z layers over a cell-centred 22 Angstrom
  window and a zero-padded three-point z kernel.

The only trainable-architecture change is the nonperiodic z CNN. The 2D FNO
has 33,664 parameters and the hybrid FNO has 35,200 (+4.6%). The frozen MACE
and neutral source head are identical.

## z-window audit

Across all 5,000 train/test structures, the atomic height span is at most
16.144 Angstrom and the maximum distance from the cell centre is 8.110
Angstrom. The 22 Angstrom window therefore includes every atom and retains the
full cubic B-spline support at `Nz=16`. Cell centring is used because the Au
cluster lies below the MgO slab; mean centring would move the coordinate origin
with the asymmetric adsorbate/substrate distribution.

## Results

| Model | Seed | Validation E RMSE (meV/atom) | Validation F RMSE (meV/A) | Test E RMSE (meV/atom) | Test F RMSE (meV/A) |
|---|---:|---:|---:|---:|---:|
| Frozen one-layer MACE | - | 2.2621 | 57.4096 | 2.3119 | 56.9900 |
| 2D FNO | 17 | 0.3627 | 34.5577 | 0.4049 | 34.5897 |
| 2D FNO | 29 | 0.4542 | 36.4937 | 0.5032 | 36.3863 |
| Hybrid 2.5D FNO | 17 | 0.5202 | 39.2377 | 0.5368 | 38.6914 |
| Hybrid 2.5D FNO | 29 | 0.5525 | 38.4058 | 0.6039 | 38.1766 |

Submitted Slurm jobs: seed 17 `15380001`; seed 29 `15380002`.

The two-seed held-out means and half-ranges are:

| Model | Test E RMSE (meV/atom) | Test F RMSE (meV/A) |
|---|---:|---:|
| 2D FNO | 0.4541 +/- 0.0492 | 35.4880 +/- 0.8983 |
| Hybrid 2.5D FNO | 0.5704 +/- 0.0336 | 38.4340 +/- 0.2574 |

At the matched 1,500 updates, 2.5D has 25.6% higher mean energy RMSE and 8.3%
higher mean force RMSE than 2D. The ordering is consistent for both seeds. The
2.5D correction still improves the frozen backbone by 75.3% in energy RMSE and
32.6% in force RMSE.

The degradation is not distributed uniformly across compositions. Averaged
over seeds, the doped test energy RMSE is almost unchanged (0.4849 versus
0.4789 meV/atom), whereas the undoped value increases from 0.4283 to 0.6372
meV/atom. Force RMSE increases for both groups: by 11.3% for doped structures
and 5.0% for undoped structures.

The seed-17 and seed-29 2.5D runs took 175.11 and 167.95 s internally. Mean
optimisation-plus-validation time increased from 97.87 s for 2D to 155.98 s
for 2.5D (+59.4%).

## Outcome

The first hybrid 2.5D prototype does not outperform the simpler projected 2D
model on this fixed-cell benchmark. This should not yet be interpreted as
evidence that z resolution is intrinsically unhelpful. With two three-point z
CNN blocks, one output layer sees only five neighbouring z planes, spanning
5.5 Angstrom between layer centres at the present spacing. The projected 2D
model, in contrast, couples
all heights immediately. A decisive architectural follow-up is therefore a
global-z treatment (a wider/dilated z mixer, attention, or the explicit learned
`R(k_parallel,z,z')`) rather than tuning the present local-z model on the test
set.

## Interpretation guardrails

This comparison tests whether explicit z resolution helps on the fixed small
LES cell. It does not by itself identify long-range electrostatics: the local
two-interaction MACE control remains necessary, and a later larger-cell or
distance-stratified test is needed to establish genuinely nonlocal transfer.
