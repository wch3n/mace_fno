# Five-seed full-response LES training

Follow-up to the 60-epoch extended-LES pilot. Train only the model with permanent
monopoles, dipoles and quadrupoles plus induced charges and anisotropic induced
dipoles. Seeds: 17, 29, 41, 53, 67. Every seed starts fresh under the same schedule.

Use the same 660 training / 240 validation structures, local MACE architecture,
float64, loss weights, EMA, and cuEquivariance 0.6.1 as the pilot. Restore the
full-protocol stage-two switch at epoch 450. Allow up to 1200 total epochs and use
validation-loss early stopping after 50 epochs without improvement. A run that
hits the epoch ceiling is explicitly flagged as needing a convergence review.
The pilot measured about 40 seconds per full epoch, suggesting about 13 hours at
the ceiling. Every training job requests one A100 GPU and 48 hours on `gpu`.

## Optimizer correction

The pinned upstream MACE optimizer includes `les_readouts` (monopoles) but omits
98,944 trainable weights in the additional multipole and response readouts.
These omissions affected the pilots: the shared backbone and monopole readout
were trained, while the omitted heads were not stepped by the optimizer.
`optimizer_compat.py` adds those readouts at the base learning rate and zero
weight decay, following the monopole-readout convention. It asserts complete,
nonduplicated optimizer coverage and records actual updates to every added
parameter tensor on the first training step. This correction applies only to
the new training processes. Pilot artifacts and installed packages are preserved.

The existing cuEquivariance conversion certificate is inherited for the identical
architecture. A new GPU check exercises the corrected optimizer for four epochs,
including stage two, standard e3nn model export/reload and finite-difference
forces. The five training jobs depend on that check succeeding. Check results
are tied to the full campaign manifest hash.

## Data and evaluation

`prepare_response.py ROOT --pilot PILOT_ROOT` creates an isolated campaign,
links the immutable pilot datasets and environment, snapshots scripts/configs,
and records source/input/package hashes. No pilot checkpoint is used as an
initialization because its stage-two schedule and optimizer coverage differ.

After each fit, select by validation loss and evaluate the original 300 test
structures, all 40 medium-probe sets from 20 trajectories and 80 ionic null
variants. Neither test nor probe results control training or checkpoint selection.
`collect.py` writes per-seed metrics and means/sample standard deviations across
the five seeds, recording the convergence status of each seed. No manuscript results
are automatically replaced.

## Submission, 2026-10-11

Campaign: `~/mace_fno_runs/_campaigns/extended_les_response_5seed_20261011`.
GPU check **16193545** passed: all 337,920 trainable parameters are covered,
all nine formerly omitted readout tensors changed on the first optimizer step,
and the exported model passed a finite-difference force check (5.8e-10 eV/A).

Production array **16193549**, tasks 0 through 4, maps to seeds
17, 29, 41, 53 and 67. All five started on `gpu`. Collector **16193550** depends
on successful completion of the array. Submission commands and scheduler states
are recorded in `submission_*.json`. Per-run training and convergence states are
under `runs/les_response_seed*/`.

## Budget revision, 2026-10-11

The user set all extended-LES fits to the ordinary synthetic LES budget of
1200 total epochs, keeping stage two at 450. Array 16195546 continues the
five residual-first checkpoints (saved epochs 523--529), replacing 16193549.
Collector 16193550 now depends on the replacement array. The archived manifests,
configs, checkpoints and histories are referenced by `budget_resume.json`.

The guarded restart verifies exact model, optimizer and learning-rate state
restoration and starts at the saved epoch plus one. The saved model uses EMA
weights. Native checkpoints omit EMA history, random-number state and patience
counters, so those restart afresh. This is not a bitwise continuation. Existing
GPU architecture checks are stored with their original manifest binding.
The budget revision is verified separately, without claiming a new GPU certificate.
