# MoS2 frozen-MACE/FNO pilot

## Inputs

- Frozen model: `mos2_xs_stagetwo.model`
- Dataset: `pbe_1331_train.xyz`
- MACE: two interaction layers, 6 A radial cutoff, float64
- Labeled structures: 157 configurations in 15 fixed in-plane-cell families
- Residual model: linear learned Fourier operator, two neutral latent channels,
  32 x 32 planar mesh, four retained modes per direction

The two isolated-atom reference entries lack force labels and were excluded.
Although the periodic structures are marked periodic along z, the learned
residual uses only the first two lattice vectors. The cells contain 25--40 A
along z, but this remains an explicit planar approximation.

## Frozen-MACE size scan

| Atoms | Frames | Energy MAE (meV/atom) | Force RMSE (meV/A) |
|---:|---:|---:|---:|
| 42 | 16 | 0.4610 | 8.0510 |
| 78 | 14 | 0.4821 | 6.0587 |
| 114 | 14 | 0.4657 | 6.4016 |
| 150 | 6 | 0.6166 | 7.3226 |
| 186 | 14 | 0.5136 | 6.4862 |
| 222 | 6 | 0.5315 | 6.2179 |
| 258 | 13 | 0.5107 | 6.5555 |
| 294 | 7 | 0.5404 | 6.8294 |
| 366 | 7 | 0.5919 | 6.5340 |
| 402 | 16 | 0.7185 | 6.4929 |
| 438 | 7 | 0.5929 | 6.8522 |
| 474 | 8 | 0.7328 | 6.7657 |
| 546 | 16 | 0.8607 | 7.4070 |
| 582 | 8 | 0.9616 | 7.4954 |
| 618 | 5 | 1.1825 | 7.8073 |

Across all 157 labeled structures, the energy MAE is 0.6253 meV/atom and
the force RMSE is 6.9604 meV/A. The increasing energy error with cell size is
worth investigating, but it is not by itself evidence for missing long-range
physics; an extensive energy calibration error can produce the same trend.

## Fixed-cell residual experiments

Both experiments used 12 training and four deterministic validation
configurations. The FNO branch was initialized to zero, so the initial combined
prediction was exactly frozen MACE. Energy and force errors were normalized by
1 meV/atom and 10 meV/A, respectively. The tuned runs accumulated four
configurations per optimizer update and used four times more force weight than
energy weight.

### 402 atoms, 26.13 A in-plane cell

| Model | Validation energy MAE (meV/atom) | Validation force RMSE (meV/A) |
|---|---:|---:|
| Frozen MACE | 0.6850 | 5.9770 |
| Constant per-atom energy offset | 0.3492 | 5.9770 |
| Initial FNO pilot | 0.6743 | 6.0678 |
| Tuned FNO | 0.6851 | 5.9885 |

The constant-offset control explains substantially more energy residual than
the FNO. The tuned FNO reduces training force RMSE from 6.6560 to 6.6044 meV/A,
but the reduction does not transfer to validation.

### 546 atoms, 30.45 A in-plane cell

| Model | Validation energy RMSE (meV/atom) | Validation force RMSE (meV/A) |
|---|---:|---:|
| Frozen MACE | 1.3492 | 7.6627 |
| Constant per-atom energy offset | 1.4236 | 7.6627 |
| Tuned FNO | 1.3502 | 7.6768 |

Here the energy residual is configuration-dependent and the constant offset
worsens validation. Nevertheless, the FNO does not improve validation energy
or forces. Training force RMSE decreases from 7.3198 to 7.2777 meV/A, again
indicating weak fitting without generalization.

## Conclusion

The software pathway is operational: MACE remains frozen, invariant features
drive neutral latent sources, force loss backpropagates through the learned
Fourier branch, and the total force is the gradient of the combined energy.

These data do **not** yet provide evidence that the FNO captures missing
long-range interactions. The fixed-cell families contain only 5--16
configurations each, the baseline was trained on this same dataset, and its
remaining force error is already only about 6--8 meV/A. The present FNO learns
small training-set corrections but fails to improve held-out configurations.

The saved pilot checkpoints should therefore be treated as diagnostics, not
production potentials.

## Required next experiment

1. Add reciprocal-cell-metric conditioning so one model can use all 157
   variable-cell configurations without assigning different physical
   wavelengths to the same Fourier index.
2. Evaluate on an independent DFT test set not used to fit the frozen MACE.
3. Include cutoff-ablation or deliberately separated perturbations so the
   target residual contains an identifiable beyond-cutoff component.
4. Compare against constant offsets, a larger local MACE, and an explicit
   electrostatic/dispersion baseline before attributing gains to long range.
