# Linear FNO wetting evaluations for Table VII

**Completed and imported on 13 October:** all twenty models and eighty
relaxations passed the endpoint checks. The four five-seed groups supply
the linear rows of Table VII. The submission record below describes their
original dependencies.

Submitted 13 October 2026 as campaign `au2_mgo_linear_wetting_20261013`.
Job 16209413 evaluates the thirteen ready checkpoints on four debug GPUs.
Jobs 16209414--16209420 evaluate tasks 13--19 after training jobs
16191322_22--16191322_28, respectively. Final collector 16209421 depends
on all eight evaluation jobs. Submission commands and the completed CPU
checkpoint/model-load checks are archived in submission.json and preflight.json.

Preparation obtains the MACE2 slab backbone from the GPU campaign and the
MACE1 backbone from the parent campaign. Configuration hashes are read from
`source_hashes`. The archived preparation records preserve input provenance.

Twenty Au2/MgO models cover MACE1/MACE2, frozen/joint, and seeds 17/29/41/53/67.
The campaign evaluates only completed 240000-step training runs. Thirteen
checkpoints are ready at preparation. The seven remaining MACE2 evaluations
depend individually on their training tasks in array 16191322.

All fits use G=1 and a single dense linear channel-height matrix. The model
loader must instantiate LinearMetricSlabFNO2D. Training source snapshots are
identical across the original and GPU campaigns and are copied without edits.
Checkpoint, backbone, configuration, data and validation-history checks tie
each relaxation to the same minimum-validation-loss weights as its held-out
accuracy evaluation. Completed checkpoint bytes are copied before inference.

The unchanged Table VII evaluator relaxes only the two Au atoms from the four
DFT starting geometries. Mg/O/Al positions and the cell stay fixed. ASE FIRE
uses float64, fmax=0.01 eV/angstrom and at most 500 steps. GPU preflight checks
the loaded operator, backbone depth and a force finite difference. Every
endpoint must converge and stay in the intended orientation basin. A wetting
sign reversal is reported as a result, not imposed as an acceptance criterion.

The ready job uses four GPUs with one sequential worker per GPU. Each pending
fit gets a separate one-GPU job after its own training succeeds. Final CPU
collection depends on all evaluations. Every job stays within the debug-gpu
two-hour limit. No training is performed by this campaign.

Collection records all completed per-seed evaluations, but computes table
means and sample standard deviations only for complete five-seed groups.
Endpoint energies share the DFT non-wetting zero for each substrate. Wetting
differences are formed per seed before averaging. No energy offsets are fitted.
The collector does not edit manuscript files.

Prepare with campaign.py prepare ROOT --parent PARENT --gpu GPU --wetting WETTING_ROOT.
Review submit.py ROOT --dry-run before calling submit.py ROOT. For a partial
export use collect.py ROOT --allow-partial --output-dir DESTINATION.
