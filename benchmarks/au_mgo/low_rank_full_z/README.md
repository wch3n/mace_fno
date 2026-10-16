# Low-rank full z-mixing trials for Au2/MgO

This prepares wavelength-dependent rank-32 and rank-64 factors of the full
channel-height spectral matrices. The dense campaign supplies the matched
data, MACE1 backbone, configuration, and training/evaluation wrappers.
The default comparison uses seed 17 for both frozen and joint training.

Run `prepare.py --dense-campaign PATH --output-dir NEW_PATH` from this directory
with the repository dependencies available. The destination must not exist.
It snapshots source and inputs with SHA-256 hashes. Then, from the new campaign:

```bash
PYTHONPATH=source/src:mace_source python3 -B preflight.py
python3 -B submit.py smoke
python3 -B submit.py production
```

Four 12-step smoke fits exercise both ranks and both training protocols,
including checkpoint reload and force inference. They use `debug-gpu`, with
two concurrent tasks. The production array uses `gpu`, also at concurrency two,
and is gated on successful completion of the entire smoke array. Each task
requests one GPU, four CPUs, 64 GB host memory, and up to 48 hours. The walltime
limit does not guarantee completion of the 240,000-step budget. Checkpoints for
explicit continuation are saved every 500 steps; jobs are not automatically
retried or resubmitted.

Production changes only the spectral rank. Grid 24x24x24, mode cutoff 8,
16 source and hidden channels, four Fourier layers, 22-angstrom height window,
four lateral mesh origins, float64, and all optimization settings match the
dense campaign. Four one-structure microbatches preserve the effective batch
size of four. Joint training preserves the 500-step MACE warm-up.

Both factors are interpolated independently in squared wavevector magnitude,
then contracted without constructing the full matrix. This maintains the
rank bound between shell anchors. The rank-32 and rank-64 spectral branches
contain 835,584 and 1,671,168 parameters per layer, compared with 5,013,504 for
the dense branch. Their random initialization matches the dense entry variance
at shell knots; the final output projection remains zero-initialized.

The fixed split contains 4,275 training, 225 validation, and 500 test structures.
The primary checkpoint is selected by minimum validation loss. Test labels are
not used in training or selection. Held-out errors are evaluated only after
training, without an energy shift. Smoke inference uses validation structures.
The single-seed trials are a first comparison and do not establish seed-averaged
accuracy. No dense runs, manuscript results, or existing checkpoints are changed.

Per-run `status.json`, `training.log`, `validation_history.jsonl`, `memory.json`,
and `test.json` distinguish successful submission from completed training.
