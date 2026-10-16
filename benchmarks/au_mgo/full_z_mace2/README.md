# Au2/MgO: MACE2 nonlinear FNO with dense full z-mixing

Ten fresh fits cover frozen and joint MACE2 training for seeds 17, 29, 41, 53
and 67. Every fit has a total budget of 240,000 optimizer steps, not per-job
steps. The starting MACE2 model and the 4,275/225/500 train/validation/test
structures are the matched physical-benchmark inputs. The frozen software is
copied from the MACE1 dense full-z campaign; current working-tree edits are
not imported into the model code.

Relative to the archived MACE2 nonlinear configuration, global height mixing
becomes dense full z-mixing. The FNO keeps G=1, 16 latent and hidden channels,
24 heights, eight in-plane modes and four nonlinear blocks. The effective
batch size stays four through four one-structure microbatches. Evaluation
uses one structure per batch. All other model and loss settings are retained.

CPU preflight verifies the immutable inputs/configurations, two MACE
interactions, cutoff, seeds, fresh-start status and effective batch size.
Two GPU smoke tasks use the complete production train/validation sets for
representative memory allocation. They take 12 optimization steps; the joint
warmup is shortened to two steps so that both FNO and backbone gradients are
exercised. Selected checkpoints must reload and evaluate successfully. GPU
peak allocated memory must stay below 95% of physical GPU memory.

The jobs were initially submitted to ia. On 2026-10-14, the smoke array was
moved to debug-gpu and production plus its continuations to gpu, both using
40 GB A100 GPUs. These are scheduler overrides; the frozen submission scripts
retain their original ia defaults. See ../gpu_migration_20261014/ for the
commands and memory evidence. The production array has no additional
array-throttle cap and waits for both 40 GB smoke tasks to finish
successfully. Each fit has up to two conditional 48-hour continuation jobs
after the first 48-hour allocation. They restore the full optimizer,
scheduler, RNG and validation-selection state from model.last.pt and never
raise the 240,000-step budget. Follow-ons run training only after TIMEOUT,
skip after COMPLETED, and refuse other terminal states. All attempts preserve
their input checkpoints and validation history. A failed check prevents
production; a training failure is not automatically retried as a timeout.

Checkpoints are saved every 500 steps. Final held-out energy and force RMSE
are evaluated using the minimum-validation-loss model, without energy shifts
or test-based model selection. No persistent graph caches are created.

Reproduction: run prepare.py with a new campaign directory, then run that
directory's preflight.py in its recorded source/MACE environment. Finally run
its submit.py. Submission IDs and exact commands are saved incrementally in
submission.json. Run status and memory measurements live in each task folder.
