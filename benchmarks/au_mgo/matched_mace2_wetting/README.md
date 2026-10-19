# Matched MACE2 wetting baseline

Evaluate the seed-17 scalar-vector MACE2 model used in Table VI at the four
Table VII endpoints. The checkpoint has two `128x0e+128x1o` product outputs,
960144 parameters, and SHA256
`a3c44f24fe8519caf8a94aa8132404d636f4e5e8c057166df7cf862ad291db8a`.

The campaign copies the existing DFT starting structures and unchanged
adsorption evaluator. Only Au atoms move under ASE FIRE, with a fixed cell
and substrate, float64 inference, a 0.01 eV/angstrom force tolerance and a
500-step cap. Saved geometries are checked for convergence, fixed substrate,
fixed cell and the intended wetting/non-wetting orientation.

The prepared campaign is
`~/mace_fno_runs/_campaigns/au2_mgo_matched_mace2_wetting_20261015`.
Job 16250841 completed on debug-gpu in 39 seconds. The four relaxations
needed 0, 2, 3 and 0 FIRE steps, respectively. Preferences are 902.699097991
meV undoped and -37.0221352205 meV doped. Both signs match DFT.

The scripts derive from `../matched_les_wetting/`. Preparation verifies the
model hash and copies its training source so the saved scalar-vector model
loads with the matching implementation. The manuscript import is handled by
`docs/source/data/au2_mgo_converged/collect_wetting.py --refresh-matched-mace2`.
The importer archives its input results before writing the evaluated values.
