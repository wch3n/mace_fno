# LES Au-MgO-Al learned-FNO pilot

Date: 2026-08-28

## Question

Can a conservative learned planar FNO correct an intentionally short-range,
one-interaction MACE model on the LES Au-MgO-Al benchmark?

## Protocol

- Official upstream split: 4,500 training structures and 500 untouched test
  structures, all with 110 atoms and the same 9.0474 x 9.0474 x 26.4589
  Angstrom cell.
- The training file contains 2,264 doped `Al3Au2Mg51O54` structures and 2,236
  undoped `Au2Mg54O54` structures. The test file contains 236 doped and 264
  undoped structures.
- Frozen backbone: upstream `mace-r5.5-nl-0` checkpoint, with one interaction
  layer and a 5.5 Angstrom local receptive field.
- MACE seed-123 split: 4,275 FNO fit structures and 225 validation structures.
  The same validation indices were used in both FNO runs.
- Nonlinear 2D FNO: 24 x 24 mesh, four modes per direction, four neutral latent
  source channels, 16 hidden channels, and two operator layers.
- Conservative energy-and-force training: 1,500 steps, four structures per
  optimizer update, float64, and a zero residual at initialization.
- The energy and force loss terms were normalized by the frozen backbone's
  training RMSEs: 2.260 meV/atom and 56.280 meV/Angstrom.
- The FNO used the in-plane coordinates only. The frozen MACE retained the
  original 3D-periodic graph.

## Held-out results

| Model | Energy RMSE (meV/atom) | Force RMSE (meV/Angstrom) |
| --- | ---: | ---: |
| One-interaction frozen MACE | 2.312 | 56.990 |
| Frozen MACE + constant energy offset | 2.310 | 56.990 |
| Frozen MACE + formula energy offsets | 2.311 | 56.990 |
| Frozen MACE + FNO, seed 17 | 0.405 | 34.590 |
| Frozen MACE + FNO, seed 29 | 0.503 | 36.386 |
| Two-interaction local-MACE control | **0.130** | **3.676** |

Across the two FNO seeds, the held-out result is 0.454 +/- 0.049 meV/atom and
35.488 +/- 0.898 meV/Angstrom (mean +/- half-range). Relative to the frozen
one-interaction backbone, this lowers energy RMSE by 80.4% and force RMSE by
37.7%. The improvement is reproducible and appears in both doped and undoped
subsets.

The two-interaction local-MACE control is nevertheless about 3.5 times more
accurate in energy and 9.7 times more accurate in forces than the mean FNO
result. It has an 11 Angstrom total receptive field, which spans the roughly
9 Angstrom in-plane cell.

## Conclusion

The learned FNO provides a substantial, conservative, and reproducible
configuration-dependent correction to a one-layer frozen MACE. This is a
positive implementation result, especially because both energy and force
errors improve and constant/composition offsets cannot explain the gain.

It is not yet evidence that the FNO has isolated long-range electrostatics.
The much stronger two-interaction local model shows that this fixed, small-cell
test can be solved primarily by extending local message passing. A decisive
long-range benchmark must hold local depth and cutoff fixed while testing
larger cells, adsorbate/dopant separation beyond the receptive field, or an
electrostatic response observable. Direct comparison with the upstream
one-interaction MACE-LES checkpoint is also required.

## Artifacts

- One-layer baseline: `Au2-MgO_r5.5_nl0_baseline.json`
- Two-layer capacity control: `Au2-MgO_r5.5_nl1_l0_baseline.json`
- FNO checkpoints: `les_au_mgo_fno_seed17.pt` and
  `les_au_mgo_fno_seed29.pt`
- Slurm logs: `logs/les-au-mgo-fno-15377028.out`,
  `logs/les-au-mgo-fno-15377047.out`, and
  `logs/les-au-mgo-eval-15377046.out`
