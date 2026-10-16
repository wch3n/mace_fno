# Central collection of completed calculations

`centralize.py` builds `~/mace_fno_runs/results/` without moving or modifying
the original runs. The collection is organized as
`system/category/model/campaign/run/`. Each entry links to the original files,
configuration and completion evidence and includes a provenance record.

```bash
python3 -B benchmarks/result_catalog/centralize.py
```

The top-level README and CSV index are the starting points. A new snapshot is
created on each refresh, and the `current` link switches only after validation.
The collection is an index with links, not an independent backup. Existing
absolute paths in job scripts and manuscript provenance remain valid.

Use a read-only draft before changing classification rules:

```bash
python3 -B benchmarks/result_catalog/centralize.py \
  --inventory-only /tmp/completed-results-review.json
python3 -B -m unittest discover -s benchmarks/result_catalog -p 'test_*.py'
```

Completion evidence includes terminal run status, explicit completion markers,
finished GPU benchmark records, and successful finalization in legacy training
logs. Explicit early stopping is accepted. Active jobs, unsuccessful runs,
smoke tests, pilots, archives and scratch directories are excluded. Obsolete
per-leg `resume_*` status records do not override a completed parent run.
Completion does not certify convergence, scientific accuracy, or selection
for the manuscript. Completed evaluations of interim checkpoints remain
evaluations, not completed training runs.

Water splits, G values and LES first-block settings remain separate. Metadata
is taken from explicit task/configuration fields where available. Unknown
values stay blank. The `z_mixing` CSV column is the configuration value, so
the `global` setting of the linear slab must not be interpreted as channel-wise
mixing: its dispatch can use a full compound channel-height matrix.

The scanner does not deserialize checkpoints, rehash large model/trajectory
files, submit jobs, or update the manuscript. It verifies completion evidence,
configuration hashes and every generated link. The live Slurm queue is read
before each inventory. Inability to read it aborts the refresh.

## Cross-campaign seed groups

`group_au_channelwise.py` collects the twenty completed channel-wise Au2/MgO FNO runs referenced by the archived per-seed table into `results/collections/au2_mgo_channelwise/`. It verifies all four five-seed sets, completion records, configurations and evaluation files before linking them. These curated groups persist across general catalog refreshes.

`audit_benchmarks.py` checks the selected physical and synthetic accuracy records
against their collectors, saved evaluations, configurations, checkpoint hashes
(where recorded), and complete seed sets. It also reproduces the current Au2/MgO
linear and full-z wetting rows from their archived evaluations. It performs no
training or inference and does not edit the manuscript.

```bash
python3 -B benchmarks/result_catalog/audit_benchmarks.py \
  --output /tmp/benchmark-audit.json \
  --collection ~/mace_fno_runs/results/collections/benchmarks
```

The optional collection groups exact selected checkpoints across campaigns.
It includes single-checkpoint local baselines and labels the archived Au2/MgO
channel-wise models separately. Ongoing full-z fits and their interim evaluation
snapshots are not promoted into these completed accuracy groups. The broader
15 October audit, including the probe, MD, numerical macros, and outstanding
Au2/MgO snapshot updates, is in `audit_20261015.md`.
