# Au2/MgO continuations, 12 October 2026

Author-requested continuation of rank-64 frozen/joint seed 17 and dense joint
seeds 53/67. The total training budget stays at 240,000 steps. Original campaign
sources, datasets, model settings, losses, seeds and checkpoint selection are
unchanged. Each job uses its campaign's frozen software snapshot.

The driver adapts the already-used dense continuation workflow. Native resume
validation checks the original training contract and input hashes. The full
`model.last.pt` state restores optimizer, scheduler, random generators,
training samples and validation selection; the best weights are not used as
an optimization restart. Committed history must agree with the saved step and
best objective. An exclusive file lock prevents overlapping continuations.
Each attempt preserves input checkpoints by hard links before atomic native
checkpoint replacement, plus copies of status and validation history.

Rank-64 jobs request the gpu partition for 12 hours. Dense jobs request ia
for 48 hours, followed by one conditional 12-hour allocation. A follow-on
trains only if its parent stopped with TIMEOUT; it exits on COMPLETED and
refuses to retry other failure states. No job increases the total step cap.
All jobs request one GPU, four CPUs and 64 GB host memory.

The native training code writes checkpoints every 500 steps. After reaching
240,000 steps, the driver verifies the validation-selected model and runs the
existing held-out test evaluator automatically. Wetting evaluation is not part
of these continuation jobs.

Preflight records establish source/config/data compatibility, optimizer and
scheduler presence, saved CUDA RNG state, energy-selector state, and history
consistency for all four runs. Submission IDs and exact commands are recorded
incrementally in submission.json locally and beside the frozen driver in both
campaign directories. Original manifests are left unchanged.
