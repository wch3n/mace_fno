# Extended LES on the heterogeneous-screening probe

Material passport: code experiment; limited-epoch seed-17 pilot; status is recorded
in the external campaign. Reference: [Park et al., arXiv:2605.05746v2](https://arxiv.org/abs/2605.05746v2).

The five-seed full-response follow-up is documented in
[README_response.md](README_response.md). Its optimizer audit found that the
pinned upstream optimizer omitted the additional multipole and response readout
weights in these pilots. The backbone and monopole readout were optimized; the
full-response follow-up corrects all 98,944 omitted weights and starts fresh.

The question is whether induced charge/dipole response captures the dependence of
the cross force constant on the remote screening medium. Compare three freshly
trained MACE1 models using the same pinned MACE/LES versions:

1. Monopole LES (control).
2. Fixed monopoles, dipoles and quadrupoles (multipole ablation).
3. The same multipoles plus induced charges and anisotropic induced dipoles,
   using the authors' `maceles-uQiqiu-r-4.5-nl-1/les.yaml` settings.

The third model can mediate an A–B interaction through response on remote atoms.
Fixed multipoles alone do not introduce this mechanism. This is an architectural
possibility; learning the correct medium response must be measured.

The campaign is `~/mace_fno_runs/_campaigns/extended_les_heterogeneous_20261011`.
Its venv contains separate MACE and LES installations. The existing PyTorch and
other dependencies are reused read-only through a `.pth` path; the old MACE/LES
environment is not upgraded. Jobs disable user-site processing and clear
`PYTHONPATH`; preparation and each run verify the imported module locations.
Upstream commits, installed source hashes, configs, input hashes and package
versions are recorded in `manifest.json` and `environment.json`.
Python is pinned to the module `Python/3.12.3-GCCcore-13.3.0`, including its shared
library path: the unversioned module selected Python 3.12.12 on the compute node
and caused the original 3.12.3 venv to crash while importing `ctypes`. The failed
startup jobs are retained in the campaign log; they performed no training.

The pilots retain the existing MACE1-LES seed-17 data and model settings: 660 training and
240 validation structures, isolated-atom E0s, cutoff 4.5 A, one interaction,
128x0e+128x1o, float64, batch 4, EMA and the original loss weights/scheduler.
The budget is **60 epochs**, with stage two starting at **epoch 23** (zero-based).
This shortens the original 1200-epoch / epoch-450 schedule, retaining approximately
the same stage-one fraction, to fit each training and evaluation task within the
**two-hour debug-gpu walltime**. These are exploratory fits, not converged accuracy
results. A nonzero correctly aligned response would demonstrate learned medium
modulation; a weak response would not establish an architectural limitation.
The updated MACELES retains vector
features in its last layer; all three new fits share this backbone. Published
older LES results remain a historical comparison, not an exact implementation
control. LES uses sigma=1 A, dl=2 A, self-interaction removal and the legacy
dynamic reciprocal-vector enumeration (no potentially undersized N_max grid).
The authors' unconstrained response-coefficient defaults are retained; no
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
completed or failed status; failed tasks are not silently retried. Evaluation
automatically follows each fit. `collect.py` requires all tasks to finish.

cuEquivariance can be enabled for the multipole and induced-response fits using
`prepare.py --cueq`. The monopole control uses e3nn. `check_cueq.py ROOT INDEX`
must pass on a GPU for each accelerated variant before production can start.
It compares energies, forces, LES energies and latent quantities against e3nn,
checks conversion back to e3nn, differentiates a force loss, checks a force by
finite differences, and runs four training epochs through stage two and export.
The production launcher verifies the check's code and LES-config hashes.

The pinned upstream converter omits LES arguments and retention of vector
features in the last layer. `cueq_compat.py` preserves both and normalizes the
cuEquivariance irreps at the LES tensor-readout constructors. This compatibility
fix is applied only inside the accelerated training process; installed packages
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
  response, respectively; all seed 17. Batch collector **16192177** runs afterwards.
  Submission receipts are `submission_pilot_train.json` and
  `submission_pilot_collect.json` in the campaign.
- At the user's request, pending tasks **16192176_1** and **16192176_2** use
  cuEquivariance 0.6.1 after GPU validation job **16192757** passed for both.
  Task 0 completed with e3nn. Energies agreed within 9.1e-13 eV (including
  roundtrip conversion), forces within 3.6e-16 eV/A, and all active latent
  quantities agreed. Sampled single-structure force-loss steps were 3.6x and
  2.9x faster, respectively; these are not full-epoch speedups. Training budgets,
  data, losses and the evaluation backend are unchanged. The migration is
  recorded in `cueq_enablement.json`, with original configs preserved under
  `provenance/before_cueq_enablement/`.
- Output: `results/summary.csv` and `results/summary.json` in the campaign.

The original e3nn smoke timings estimate approximately 52, 61 and 72 seconds per full epoch
for the three variants (165 training batches and 60 validation batches). The
60-epoch full-response fit therefore takes about 72 minutes before startup,
checkpoint exports and the 300-structure/40-set evaluation. The budget retains
about 45 minutes of nominal headroom. `pilot_timing_budget.json` records the
calculation; actual runtime can vary with GPU sharing and filesystem load.
