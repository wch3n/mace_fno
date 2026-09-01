# Ti2CO2 adsorbate benchmark

This benchmark tests whether a global 2D FNO residual can recover accuracy that
would otherwise require a deeper local MACE model. The common DFT target is
PBE0+rVV10 for bare Ti2CO2 and O*, OH*, and OOH* adsorbates.

The controlled comparison is:

1. a one-interaction MACE trained from scratch with a 4.5 A cutoff;
2. a two-interaction MACE with the same cutoff, hidden width, loss, split, and
   optimization schedule;
3. the frozen one-interaction MACE augmented by the slab-resolved 2D FNO.

Both local models use `128x0e` hidden features. The comparison therefore asks
whether global residual propagation can substitute for the second local
message-passing interaction without changing the local cutoff.

## Data preparation

The `1331` and `2332` files are alternative partitions of the same 1,045 unique
geometries. `prepare_dataset.py` verifies this equality, checks label agreement,
deduplicates before splitting, and makes a deterministic split stratified by
chemical formula and the source temperature/configuration label.

The current slab FNO has a fixed in-plane-cell contract. Accordingly, all three
models use only the dominant 9.098748 by 10.506329 by 25.0 A cell. The 64
uniformly contracted-cell configurations are excluded equally and recorded in
`split_manifest.json`; they are not silently dropped by a training job.
Generated XYZ files and the manifest are written outside Git.

Prepare only:

```bash
source benchmarks/runtime_paths.sh
python3 benchmarks/ti2co2_adsorbates/prepare_dataset.py \
  --source-root ~/scratch/mxene/mlip/pbe0-rvv10_8/Ti \
  --output-root "$MACE_FNO_WORK_ROOT/ti2co2_adsorbates/data/prepared"
```

For systems where the scratch filesystem is unreliable during data
preparation, build a portable package in home storage first:

```bash
bash benchmarks/ti2co2_adsorbates/stage_inputs.sh
```

This creates `~/mace_fno_runs/ti2co2_adsorbates` with the prepared data,
Slurm inputs, and a portable `submit.sh`. Copy the complete directory to
scratch and run `bash submit.sh` from the copied directory. Its data, logs,
models, caches, and audits remain beneath that copied directory; only the
versioned Python source is imported from the repository checkout.

## Submit the comparison

From the repository root:

```bash
bash benchmarks/ti2co2_adsorbates/submit.sh
```

The two MACE controls run independently. Their evaluations depend on the
corresponding training jobs. FNO training depends only on the completed
one-interaction baseline evaluation, whose training RMSEs set the residual loss
scales. Finite-difference/symmetry and low-k spectral audits follow the selected
FNO checkpoint.

All generated data, models, caches, checkpoints, logs, and audit JSON files are
placed under the printed external experiment root. No benchmark is executed
inside the source checkout.

## Interpretation

The primary comparison is held-out energy and force RMSE for:

- one-interaction MACE;
- two-interaction MACE;
- one-interaction MACE plus FNO.

Metrics should also be reported separately for bare, O*, OH*, and OOH* systems.
The close-contact `ti2c` configurations contain very large repulsive forces and
should be shown as a separate diagnostic subset when analyzing the final run.
An FNO improvement over the one-interaction model demonstrates a useful global
residual. Matching the two-interaction control is the stronger capacity result;
neither aggregate comparison alone proves that the learned residual is purely
electrostatic.
