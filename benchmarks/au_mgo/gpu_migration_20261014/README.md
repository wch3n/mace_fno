# Au2/MgO GPU allocation update, 2026-10-14

This directory records the GPU migration of 14 October. Use the
[finalization campaign](../full_z_budget_20261016/README.md) for full-mixing
stopping settings and evaluation. The job IDs below identify the migration
allocations.

Dense MACE1 full z-mixing fits in the 40 GB A100 cards of the gpu partition at
microbatch size one, accumulation four, and evaluation batch size one.
The archived training smoke tests measured peak allocated/reserved device
memory of 15.04/15.67 GiB (frozen) and 15.24/15.68 GiB (joint). Live checks
near the end of the 48-hour training allocations measured 16.51 and
16.60 GiB in nvidia-smi, respectively, including CUDA overhead.
The 64 GiB Slurm memory request is host RAM, independent of GPU memory.
Measured host MaxRSS was approximately 10.3 GiB frozen and 23.6 GiB joint.
The host-memory requests were unchanged.

## Queue changes

- MACE1 joint seeds 41/53/67: existing jobs 16190334, 16206375 and 16206376
  moved from ia to gpu and started there. Existing timeout follow-ons
  16206377 and 16206378 moved to gpu too.
- Seven running ia allocations were left undisturbed. Jobs 16234043–16234049
  provide six-hour gpu continuations for frozen seeds 17/29/41/53/67 and
  joint seeds 17/29. Each waits for its current allocation and resumes only
  after TIMEOUT, skips COMPLETED, and refuses other terminal states.
- Job 16234050 is a twelve-hour gpu timeout follow-on for joint seed 41,
  matching the existing fallback coverage of seeds 53 and 67.
- MACE2 smoke array 16208989 moved to debug-gpu. Its two tasks check frozen
  and joint training on 40 GB cards. The debug QoS runs one task at a time.
- MACE2 production array 16208990 and twenty conditional continuations
  16208991–16209010 moved to gpu. Production still requires both smoke tasks
  to succeed. The smoke checks exercise backbone gradients for joint training
  and reject a peak allocation above 95% of the device memory. The measured
  MACE2 memory result was still pending when this migration was prepared.

Training configurations, full-state resume drivers, seeds, and 240,000-step
total budgets are unchanged. The continuation wrapper is copied from the
existing October 12 implementation. The validated October 8/11 drivers are
reused for their corresponding tasks. New scheduler records are stored in
the resume_gpu_20261014 directory of the MACE1 campaign. No manuscript results
were changed.

## Records

- memory_evidence.json: original memory records and live measurements.
- migration.json: before/after scheduler records and exact update commands.
- continuations.json: new job IDs, dependencies, commands, and driver hashes.
