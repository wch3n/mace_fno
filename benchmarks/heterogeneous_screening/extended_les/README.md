# Extended LES on the heterogeneous-screening probe

This exploratory seed-17 pilot is outside the current paper. Its status and
inputs are recorded in the external campaign. Reference: [Park et al., arXiv:2605.05746v2](https://arxiv.org/abs/2605.05746v2).

The five-seed full-response follow-up is documented in
[README_response.md](README_response.md). Its optimizer audit found that the
pinned upstream optimizer omitted the additional multipole and response readout
weights in these pilots. The backbone and monopole readout were optimized. The
full-response follow-up corrects all 98,944 omitted weights and starts fresh.

We test whether induced charges and dipoles reproduce the dependence of the
cross force constant on the remote screening medium. The comparison uses
three fresh MACE1 fits with the same pinned MACE/LES versions:

1. Monopole LES (control).
2. Fixed monopoles, dipoles and quadrupoles (multipole ablation).
3. The same multipoles plus induced charges and anisotropic induced dipoles,
   using `maceles-uQiqiu-r-4.5-nl-1/les.yaml` from the original study.

The third model can mediate an A–B interaction through response on remote atoms.
Fixed multipoles alone do not introduce this mechanism. This is an architectural
possibility. Learning the correct medium response must be measured.

The campaign is `~/mace_fno_runs/_campaigns/extended_les_heterogeneous_20261011`.
Its venv contains separate MACE and LES installations. The existing PyTorch and
other dependencies are reused read-only through a `.pth` path. The old MACE/LES
environment is not upgraded. Jobs disable user-site processing and clear
`PYTHONPATH`. Preparation and each run verify the imported module locations.
Upstream commits, installed source hashes, configs, input hashes and package
versions are recorded in `manifest.json` and `environment.json`.
Python is pinned to the module `Python/3.12.3-GCCcore-13.3.0`, including its shared
library path: the unversioned module selected Python 3.12.12 on the compute node
and caused the original 3.12.3 venv to crash while importing `ctypes`. The failed
startup jobs are recorded in the campaign log. They performed no training.

The pilots use the existing MACE1-LES seed-17 data and model settings: 660 training and
240 validation structures, isolated-atom E0s, cutoff 4.5 A, one interaction,
128x0e+128x1o, float64, batch 4, EMA and the original loss weights/scheduler.
The budget is **60 epochs**. Stage two starts at **epoch 23** (zero-based).
This shortens the original 1200-epoch / epoch-450 schedule, keeping approximately
the same stage-one fraction, to fit each training and evaluation task within the
**two-hour debug-gpu walltime**. These are exploratory fits, not converged accuracy
results. A nonzero correctly aligned response would demonstrate learned medium
modulation. A weak response would not establish an architectural limitation.
The updated MACELES includes vector
features in its last layer. All three new fits share this backbone. Published
older LES results provide a historical comparison, not an exact implementation
control. LES uses sigma=1 A, dl=2 A, self-interaction removal and the legacy
dynamic reciprocal-vector enumeration (no potentially undersized N_max grid).
The response coefficients use the unconstrained defaults of the original study. No
charges, screening fields or response labels are supplied during training.

Models are selected by validation loss under the existing stage-two/EMA
procedure. Test and probe files are absent from training configs. After training,
evaluate the original 300-structure test set and all 40 probe sets from 20 held-out
trajectories: six medium variants and two ionic-swap null variants per set,
six force evaluations at displacement 0.05 A per variant. Reuse the original
evaluator and trajectory-bootstrap analysis. Report energy/force RMSE, slope b,
bootstrap interval, absolute K RMSE, captured modulation and null-response RMS.
No manuscript or existing figure data are changed by this campaign.

The smoke test uses training/validation subsets only and covers all three models,
force-loss backpropagation, transition to stage two, model reload, active response
flags, force versus finite-difference energy, and GPU memory. Production requires
`smoke_passed.json`. Each pilot job has a two-hour scheduler limit and records running,
completed or failed status. Failed tasks are not silently retried. Evaluation
automatically follows each fit. `collect.py` requires all tasks to finish.

cuEquivariance can be enabled for the multipole and induced-response fits using
`prepare.py --cueq`. The monopole control uses e3nn. `check_cueq.py ROOT INDEX`
must pass on a GPU for each accelerated variant before production can start.
It compares energies, forces, LES energies and latent quantities against e3nn,
checks conversion back to e3nn, differentiates a force loss, checks a force by
finite differences, and runs four training epochs through stage two and export.
The production launcher verifies the code and LES-config hashes recorded by the check.

The pinned upstream converter omits LES arguments and retention of vector
features in the last layer. `cueq_compat.py` preserves both and normalizes the
cuEquivariance irreps at the LES tensor-readout constructors. This compatibility
fix is applied only inside the accelerated training process. Installed packages
are unchanged. Standard e3nn models are exported for the existing test/probe
evaluation. `collect.py` records the training backend for each fit.

Submitted on 2026-10-11:

- Corrected smoke job **16192163**: all three variants passed on an A100 40 GB,
  including force-loss training, stage two, checkpoint reload and finite-difference
  forces (maximum discrepancy 2.1e-9 eV/A). These four-epoch smoke models are not
  accuracy results.
- The original 1200-epoch GPU array **16192164** and collector **16192167** were
  superseded before training started by the two-hour pilots requested by the user.
  The original protocol is preserved in `provenance/full_budget_before_pilot/`.
- Debug-GPU pilot array **16192176**, tasks 0–2: monopole, multipoles, and full
  response, respectively. All seed 17. Batch collector **16192177** runs afterwards.
  Submission receipts are `submission_pilot_train.json` and
  `submission_pilot_collect.json` in the campaign.
- As requested, pending tasks **16192176_1** and **16192176_2** use
  cuEquivariance 0.6.1 after GPU validation job **16192757** passed for both.
  Task 0 completed with e3nn. Energies agreed within 9.1e-13 eV (including
  roundtrip conversion), forces within 3.6e-16 eV/A, and all active latent
  quantities agreed. Sampled single-structure force-loss steps were 3.6x and
  2.9x faster, respectively. These are not full-epoch speedups. Training budgets,
  data, losses and the evaluation backend are unchanged. The migration is
  recorded in `cueq_enablement.json`, with original configs preserved under
  `provenance/before_cueq_enablement/`.
- Output: `results/summary.csv` and `results/summary.json` in the campaign.

The original e3nn smoke timings estimate approximately 52, 61 and 72 seconds per full epoch
for the three variants (165 training batches and 60 validation batches). The
60-epoch full-response fit therefore takes about 72 minutes before startup,
checkpoint exports and the 300-structure/40-set evaluation. The budget leaves
about 45 minutes of nominal headroom. `pilot_timing_budget.json` records the
calculation. Actual runtime can vary with GPU sharing and filesystem load.
