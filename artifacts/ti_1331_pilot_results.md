# Ti2CO2 + adsorbates learned-FNO pilot

Date: 2026-08-28

## Question

Can a conservative learned 2D Fourier neural operator (FNO) improve a frozen
MACE potential for Ti2CO2 surfaces bearing O*, OH*, and OOH* adsorbates?

## Data audit

- Source: `pbe0-rvv10_1331_train.xyz` and the supplied independent
  `pbe0-rvv10_1331_test.xyz`.
- Labels: `energy_dft` and `forces_dft`.
- Fully labeled structures: 956 train and 108 test. Four isolated-atom
  reference entries without force labels were excluded.
- Composition counts in the full training set: 54 bare Ti2CO2, 123 O*, 220
  OH*, and 559 OOH* structures.
- Composition counts in the full test set: 6 bare, 14 O*, 25 OH*, and 63 OOH*
  structures.
- There are two in-plane cells. The fixed-cell FNO pilot used the majority
  9.098748 x 10.506329 Angstrom cell: 895 train structures and 105 test
  structures. The 9.029556 x 10.426433 Angstrom cell (61 train and 3 test)
  was excluded rather than silently mixed into a fixed-cell operator.
- All surface structures are 3D-periodic with a 25 Angstrom c axis. MACE kept
  the original 3D periodic graph; the residual used a planar mesh and therefore
  assumes the vacuum makes inter-slab residual coupling negligible.

## Frozen-MACE checkpoint selection

Metrics below use energy RMSE per structure divided by atom count and force
RMSE over Cartesian force components. They were recomputed directly on all 108
test structures rather than copied from training logs.

| Checkpoint | Energy RMSE (meV/atom) | Force RMSE (meV/Angstrom) |
| --- | ---: | ---: |
| `ft-omat_0-01_stagetwo.model` | 2.887 | 60.874 |
| `ft-omat_0-02_stagetwo.model` | 4.659 | 62.483 |
| `ft-omat_0_multi-00_stagetwo.model`, `Default` head | 2.044 | 113.352 |

`ft-omat_0-01_stagetwo.model` was frozen for residual learning because it is
the strongest balanced energy-and-force checkpoint. The multi-head checkpoint
has the best energy but substantially worse forces.

## FNO protocol

- Majority cell only: 805 fit / 90 internal validation / 105 untouched test.
  Equal-weight runs used independent seeds 17 and 29, changing the fit split,
  initialization, and sampling sequence while retaining the same test set.
- Frozen MACE invariant node features were mapped to four graph-neutral latent
  source channels.
- Nonlinear 2D FNO: 24 x 24 mesh, four Fourier modes per direction, 16 hidden
  channels, and two FNO layers.
- Total residual energy was differentiated to obtain conservative forces.
- Zero residual initialization made step zero exactly equal to frozen MACE.
- 1,500 optimizer steps, four structures accumulated per update, float64.
- Null controls were fitted only on the fit partition: one constant per-atom
  energy shift and one per-composition energy shift. The latter is the relevant
  control because the four adsorbate compositions have distinct energy biases.
- The supplied test file was evaluated only before and after optimization.

## Held-out majority-cell results

| Model/control | Energy RMSE (meV/atom) | Force RMSE (meV/Angstrom) |
| --- | ---: | ---: |
| Frozen MACE | 2.785 | 59.578 |
| Frozen MACE + global energy offset | 2.773 | 59.578 |
| Frozen MACE + composition energy offsets | 2.343 | 59.578 |
| Equal-weight learned FNO, step 1500 | **2.120** | 61.916 |
| Equal-weight learned FNO, seed 29, selected step 1500 | **2.128** | 62.061 |
| Force-weight-4 learned FNO, validation-selected step 1250 | 2.396 | 60.289 |

Across the two independent seeds, the equal-weight FNO gives 2.124 +/- 0.004
meV/atom energy RMSE and 61.988 +/- 0.072 meV/Angstrom force RMSE (mean +/-
half-range). It lowers energy RMSE by about 23.7% relative to raw MACE and by
about 9.3% relative to the composition-offset control, but worsens force RMSE
by about 4.0%. Under the normalized equal-weight energy-plus-force objective,
its gain over the composition-offset control is only 1--2%. The
force-emphasized run does not beat the composition-offset control on either
metric.

The equal-weight energy result is also composition-dependent (seed 17 shown):

| Subset | Structures | Composition-offset E RMSE | FNO E RMSE | MACE F RMSE | FNO F RMSE |
| --- | ---: | ---: | ---: | ---: | ---: |
| bare Ti2CO2 | 4 | 0.252 | 1.191 | 32.889 | 33.126 |
| O* | 14 | 1.163 | 0.981 | 49.041 | 49.519 |
| OH* | 24 | 1.166 | 1.889 | 55.486 | 56.478 |
| OOH* | 63 | 2.885 | 2.415 | 64.199 | 67.362 |

The aggregate energy gain is dominated by the OOH* majority; it does not
generalize uniformly across adsorbate classes.

## Conclusion

This pilot does not yet support adopting the FNO residual as an improved MLIP.
The independent-seed repeat confirms that the branch learns reproducible
spatial energy corrections beyond composition offsets, especially for OOH*,
but it also confirms that no run improves forces. The aggregate advantage is
marginal after a strong null control.

It also cannot establish that the learned correction is genuinely long-range.
The approximately 9 x 10.5 Angstrom cell is comparable to the receptive field
of a two-layer MACE with a 6 Angstrom cutoff, so an FNO improvement could reflect
generic extra model capacity. A decisive next experiment needs matched larger
supercells and a split based on adsorbate separation or cell size, ideally with
the local MACE receptive field held fixed.

## Reproducibility artifacts

- Baseline JSON files: `ft-omat_0-01_stagetwo_1331_baseline.json`,
  `ft-omat_0-02_stagetwo_1331_baseline.json`, and
  `ft-omat_0_multi-00_stagetwo_1331_baseline.json`.
- Equal-weight diagnostic checkpoint: `ti_1331_fno_seed17.pt`.
- Independent equal-weight checkpoint: `ti_1331_fno_seed29.pt`.
- Force-emphasized validation-selected checkpoint:
  `ti_1331_fno_fw4_seed17.pt`.
- Slurm logs: `logs/ti-fno-1331-15376740.out` and
  `logs/ti-fno-1331-15376765.out`, and
  `logs/ti-fno-1331-15376789.out`.
