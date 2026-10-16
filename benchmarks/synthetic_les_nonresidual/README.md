# Synthetic LES first-block comparison

Tables IV and V and Figs. 3, 5, and 6 use these nonresidual-first results.
The separate [contact-trained MD campaign](../synthetic_yukawa/nonresidual_les_md/README.md)
supplies the revised LES curves of Fig. 4.

The twenty completed charge-only MACE1–LES fits change `interaction_first` from
`RealAgnosticResidualInteractionBlock` to `RealAgnosticInteractionBlock`.
The latter matches the local MACE backbones used by these synthetic benchmarks.

The 20 tasks cover homogeneous kappa = 0.05, 0.2, 0.5 inverse angstrom and
heterogeneous screening, each at seeds 17, 29, 41, 53, 67. All original data files
are copied byte-for-byte. The optimizer, losses, stage-two start at epoch 450,
1,200-epoch maximum, patience 50, batch size four, and charge-only LES defaults
are retained. The one-layer product output stays scalar (`128x0e`). The existing
`128x0e+128x1o` hidden representation setting is unchanged. Fits start fresh.

Preparation snapshots unmodified installed MACE/LES sources, all 20 original
stage-two models, configuration files, and old full-precision metrics.
`manifest.json` records hashes, package versions, and the exact config changes.
No installed package or original run is edited. Backend conversion preserves
explicit LES settings through a process-local wrapper around the native trainer.

Two GPU validation tasks (two-species homogeneous and three-species heterogeneous)
run native and cuEquivariance two-epoch, two-stage smoke fits. They check saved
architectures, LES defaults, parameter counts, energy/force/charge equivalence,
round-trip conversion, optimizer coverage, and a force-loss optimization step.
Production is gated on both validation tasks succeeding. The expected native
parameter counts are 278,528 (homogeneous) and 347,776 (heterogeneous), including
the 128-parameter charge readout. First-block parameter count is part of the
architecture change, not held artificially constant.

Each production task evaluates its native validation-selected stage-two model on
the original validation and test splits. It re-evaluates the original residual
checkpoint on the identical test data and checks that published manuscript source
metrics are reproduced. Heterogeneous tasks also evaluate the unchanged 40-set
probe and analyze it alongside the original residual-first responses.
`collect.py` produces per-seed RMSE differences and mean/sample-SD summaries of
the paired differences. Test data never select models. Manuscript numbers are
not replaced automatically.

Run `campaign.py prepare ROOT --runs-root RUNS_ROOT --repo REPO` in the pinned
Python environment, then use `job.slurm ROOT validate` as array `0,15`.
After validation, submit `job.slurm ROOT run` as array `0-19%5` and
`collect.slurm ROOT` with an `afterok` dependency on the production array.
The four consecutive blocks of five indices follow the dataset order above.
