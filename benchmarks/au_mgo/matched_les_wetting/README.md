# Scalar-vector MACE1-LES wetting rerun

Re-evaluates the five completed scalar-vector MACE1-LES checkpoints used in
Table VI, seeds 17, 29, 41, 53, and 67. These are the charge-only LES models
from `au2_mgo_matched_irreps_20260929T182640Z`. This campaign performs inference
and relaxation only.

The four DFT starting geometries and evaluator are copied from the shared
wetting inputs. Only Au moves, the cell and substrate are fixed, and ASE
FIRE runs in float64 to 0.01 eV/angstrom, at most 500 steps. Input/source and
model hashes are checked. Each saved model must have `128x0e+128x1o` product
outputs. Endpoint checks cover force convergence, substrate/cell constraints,
and the intended orientation basin.

Campaign: `~/mace_fno_runs/_campaigns/au2_mgo_matched_les_wetting_20261011`.
GPU job **16193846** completed on one A100 in **58 seconds**, 11 October 2026.
All 20 endpoints passed, and all five models predict the wetting sign reversal.
Preference means and sample SDs are **941.717 +/- 4.176 meV** (undoped) and
**-77.199 +/- 4.133 meV** (doped).

`collect.py CAMPAIGN --data-dir docs/source/data/au2_mgo_converged` verifies
checkpoint identity against the accuracy snapshot, replaces only the five
MACE1-LES records, and regenerates the macros of Table VII. The input snapshot
is archived before replacement. All other model records are checked to be
unchanged. Paired energy differences are computed before calculating seed
means and sample SDs. All endpoint energies use the same DFT non-wetting zero
for the corresponding substrate composition.

The complete table is reproducible offline using
`python3 -B docs/source/data/au2_mgo_converged/collect_wetting.py`.
