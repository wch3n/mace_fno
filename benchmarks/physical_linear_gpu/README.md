# Linear FNO: G=1 water and Au2/MgO on gpu

**Completed:** all twenty G=1 water fits and nine replacement Au₂/MgO fits
finished. Their results entered Table VI by 13 October, alongside the completed
parent fits. All twenty Au₂/MgO linear wetting evaluations also entered Table VII.
The submission details below describe the original queue arrangement.

This campaign reuses the exact source, trainer, evaluator, data and MACE
backbones from `physical_linear_5seed_20261005`.

Twenty fresh water fits cover MACE1/MACE2, frozen/joint training, and seeds
17/29/41/53/67. The only training change is spectral_groups=1. Budgets are
120000 steps except MACE1 joint at 200000. No G=4 checkpoint or
optimizer state initializes a G=1 model.

Nine Au2/MgO MACE2 fits replace the never-started parent tasks 51–59:
four frozen seeds (29/41/53/67) and five joint seeds (17/29/41/53/67).
Training settings, including the batch size of four and 240000-step budget,
are unchanged. Evaluation batch size decreases from eight to one to fit
the 40-GB GPUs. This changes batching, not the dataset or objective.
The linear slab uses one dense radial matrix over the compound channel and
height index: full mixing, with no low-rank factorization, activation or
pointwise W branch. The original completed Au2/MgO results are valid.

`prepare.py --parent PARENT --output NEW_DIRECTORY` verifies original hashes,
archives input files and enforces allowed configuration changes. Run
`preflight.py` using the copied source. Before production, a debug-gpu job
runs `smoke.py`: six fresh six-step fits, covering every depth/regime/system
combination, then checkpoint reload and force inference. Actual training and
inference GPU peaks are recorded. No spectral diagnostics are requested.

Production runs on gpu and is gated on successful smoke completion. Each
fit uses the original validation-selected checkpoint and unshifted held-out
test evaluation. Existing model checkpoints and paper tables are untouched.
Submission and migration receipts record replacement job IDs. The old held tasks were cancelled after their GPU replacements were queued.

`submit.py CAMPAIGN` queues separate arrays for water (four concurrent fits)
and Au2/MgO (two concurrent fits) after `smoke_passed.json` exists.
`submit.py CAMPAIGN --retire-parent` verifies the replacement jobs and cancels
the nine held original entries. `collect.py` combines the new fits with the
31 completed G=1 parent fits, producing a 60-model per-seed table and five-seed
means/sample standard deviations. It makes no manuscript edits.

The 2026-10-11 campaign is
`~/mace_fno_runs/_campaigns/physical_linear_gpu_g1_20261011`:

- Smoke job 16191317: completed. All six training/reload/force checks passed.
- Water array 16191321_0–19: submitted to gpu, concurrency four.
- Au2/MgO array 16191322_20–28: submitted to gpu, concurrency two.
- Collector 16191323: depends on successful completion of both arrays.
- Original ia jobs 15848117_51–59: cancelled after replacements were queued.

The 40-GB A100 smoke peaks for Au2/MgO MACE2 were 25.18/26.00 GiB allocated
during frozen/joint training and 4.86/4.94 GiB during evaluation. These are
smoke-test memory measurements, not the manuscript inference benchmark.
Water peak training memory ranged from 6.34 to 10.52 GiB. All production
training batches and optimizer settings are as specified above.
