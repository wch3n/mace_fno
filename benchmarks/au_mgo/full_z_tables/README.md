# Full-z results for Au2/MgO Tables VI and VII

**Import status, 16 October:** frozen MACE1 test and wetting results use all
five completed fits. Joint wetting uses all five finalized fits from the
[finalization campaign](../full_z_budget_20261016/README.md). The joint test row
still uses the 13 October snapshot. The dated evaluation recipes below record
each import separately.

The initial evaluation covered ten selected dense full z-mixing MACE1–FNO
checkpoints (frozen/joint, seeds 17/29/41/53/67) while training continued.
The first snapshot was prepared on 2026-10-11 in
`~/mace_fno_runs/_campaigns/au2_mgo_full_z_tables_20261011` and submitted as
Slurm job 16191184 on debug-gpu (four GPUs, four independent workers).

`prepare.py` verifies the training source and data hashes and takes stable
checkpoint copies. It checks the full channel-height operator, the declared rank,
G=1, 16 source/hidden channels, 24 heights, four in-plane mesh origins, and
the shared backbone. Selection is the minimum combined validation loss
recorded through each saved checkpoint, with no test-based selection.
The snapshot records selected and observed progress separately and marks fits
that have not completed 240000 steps as preliminary.

`run.py` uses the test evaluator of the training campaign and the ASE evaluator
of the wetting campaign. Each GPU handles one model at a time. Table VI
uses raw energy/force errors on the same 500 held-out structures, without
energy shifts. Table VII uses four Au-only FIRE relaxations, fixed substrate
and cell, force tolerance 0.01 eV/angstrom, and a 500-step limit. Final
structures must remain in the intended wetting/non-wetting basins. The
same checkpoint bytes supply test and wetting results.

`collect.py CAMPAIGN` requires every model to finish and all endpoint
checks to pass. It exports test/wetting per-seed CSVs and means and
sample standard deviations, including endpoint energies relative to the DFT
non-wetting zero for each composition. It never edits the manuscript.
MACE2 and low-rank experiments are not included in these dense MACE1 rows.

The same scripts also accept a low-rank training campaign. They preserve its
own source snapshot and group results by spectral rank and training regime.
Single-seed groups have no standard deviation. On 2026-10-11 the rank-32 and
rank-64 seed-17 frozen/joint checkpoints were copied into
`~/mace_fno_runs/_campaigns/au2_mgo_low_rank_tables_20261011`. Rank 32 had
completed its training budget. Rank 64 was still running. Test structures,
backbone, wetting structures, and all evaluation settings match the dense
snapshot. Comparisons must use dense seed 17 and record the unequal training
progress. These scripts do not decide equivalence or replace manuscript rows.

`prepare_low_rank_gpu.py` reuses the existing appendix GPU benchmark for
joint seed 17, dense/rank 32/rank 64, all measured serially on the same A100.
`compare.py` archives matched test and wetting results, parameter counts,
and optional GPU measurements. It checks that the GPU and accuracy checkpoint
hashes agree. The collected 2026-10-11 comparison is in
`docs/source/data/au2_mgo_low_rank_current/`.

For the completed frozen-MACE1 wetting refresh on 15 October, preparation uses
`--regime frozen --require-complete`. This requires all selected fits to finish
240,000 steps and checks their checkpoint hashes against completed status.
The saved test evaluations are reused only when they identify exactly those
checkpoint bytes. The five models receive new wetting relaxations. Test reuse
is recorded explicitly in the output JSON. This avoids repeating the 500-frame
test evaluation while aligning Tables VI and VII to the same models.

The campaign is `au2_mgo_full_z_frozen_wetting_completed_20261015`.
Job 16255199 uses two A100 GPUs in debug-gpu, 8 CPUs, 64 GB host memory and a
30-minute time limit. After convergence and basin checks pass, import with:

```bash
python3 -B docs/source/data/au2_mgo_full_z_current/collect_results.py \
  --refresh-completed-frozen-wetting \
  ~/mace_fno_runs/_campaigns/au2_mgo_full_z_frozen_wetting_completed_20261015 \
  --write-manuscript
```

This importer changes only frozen wetting records. Joint results and all test
metrics keep their own provenance. The input manuscript data are archived
before replacement, and offline collection reapplies the completed wetting
overlay rather than restoring the interim values.
