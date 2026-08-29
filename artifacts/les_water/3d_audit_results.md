## Material Passport

- Origin Skill: experiment-agent
- Origin Mode: validate
- Origin Date: 2026-08-29
- Verification Status: VERIFIED
- Version Label: validation_v1

## Validation Report

- **Source**: LES-water frozen-MACE + 3D-FNO pilot
- **Overall Confidence**: SOLID for the EqGINO + interlacing numerical
  contracts; the earlier generic 3D-FNO symmetry failure is retained below as
  historical motivation
- **Audit jobs**: `15387074` (seed 17, 14 s) and `15387077` (controlled seed
  29, 13 s), both `COMPLETED` with exit code `0:0`
- **Audit subset**: the same eight deterministic held-out indices for both
  checkpoints: 42, 34, 44, 2, 21, 37, 31 and 30

### EqGINO + 2 x 2 x 2 interlacing update

The cubic EqGINO layer removed the signed-axis symmetry defect. Native 3D
interlacing was then tested in a strictly matched run: seed 17, a `24 x 24 x
24` mesh, four modes on every axis, one dense EqGINO spectral group and 3,000
optimizer steps. Training job `15388338` and strict audit job `15388339` both
completed with exit code `0:0`.

| Metric | EqGINO, one origin | EqGINO, 8 origins | Change |
|---|---:|---:|---:|
| Held-out E RMSE (meV/atom, 50 structures) | 0.2319 | 0.2308 | -0.5% |
| Held-out F RMSE (meV/A, 50 structures) | 25.0268 | 25.0118 | -0.06% |
| Audit-subset corrected E RMSE (meV/atom) | 0.171897 | 0.171591 | -0.2% |
| Audit-subset corrected F RMSE (meV/A) | 24.8627 | 24.8505 | -0.05% |
| Max 0.1-A rigid-translation E change (meV) | 0.105132 | 0.005127 | 20.5 x smaller |
| Max translated residual-force RMS (meV/A) | 0.033861 | 0.002446 | 13.8 x smaller |
| Max translated residual-force change (meV/A) | 0.280350 | 0.025500 | 11.0 x smaller |
| Largest net residual-force RMS component (meV/A) | 0.478654 | 0.019113 | 25.0 x smaller |
| Optimization + validation time (s) | 207.39 | 855.55 | 4.13 x |

All strict checks pass for the interlaced checkpoint. In particular, the
largest cubic residual-energy change is `2.13e-7` meV and the largest cubic
residual-force covariance RMSE is `1.26e-7` meV/A. The residual-force
finite-difference maximum is `1.56e-6` meV/A. Interlacing therefore preserves
EqGINO's cubic symmetry and conservativity while reducing the continuous
translation egg-box diagnostics by one to two orders of magnitude, without a
measurable loss of held-out accuracy. Its observed training-time cost is about
fourfold because the frozen MACE evaluation and other work are shared.

### Rejected projection-only warm-up

A matched run tested 250 output-projection-only steps at a `3e-3` learning
rate, followed by full residual training at `3e-4` through the same total budget
of 3,000 steps. Training job `15388628` and strict audit job `15388629` both
completed successfully. The schedule operated as intended (64 of 19,392
residual parameters active during warm-up), but delayed useful force learning
and is rejected for this model.

| Metric | No warm-up | 250-step warm-up | Change |
|---|---:|---:|---:|
| Validation F RMSE at step 1,500 (meV/A) | 25.6166 | 26.9414 | worse |
| Validation F RMSE at step 3,000 (meV/A) | 24.3407 | 25.2139 | +3.6% |
| Held-out E RMSE (meV/atom) | 0.2308 | 0.2414 | +4.6% |
| Held-out F RMSE (meV/A) | 25.0118 | 25.7444 | +2.9% |
| Max 0.1-A rigid-translation E change (meV) | 0.005127 | 0.012490 | 2.44 x larger |
| Max translated residual-force change (meV/A) | 0.025500 | 0.109421 | 4.29 x larger |

All exact checks still pass, including source neutrality, conservative-force
finite differences and EqGINO cubic covariance. The negative result is thus an
optimization effect rather than a software-contract failure. Projection-only
training initially fits an energy-biased mapping using fixed random upstream
features; after unfreezing, the source head and field operator must undo that
mapping before their force derivatives improve. The useful force-learning
transition moved from approximately step 1,500 to step 2,500. Warm-up remains
available as a default-off diagnostic, but it is not recommended here.

### Accepted scale-0.1 soft start

A second matched run kept all 19,392 residual parameters trainable from step
one and scaled only the random final projection by 0.1. Training job `15388819`
and strict audit job `15388822` completed successfully. Unlike projection-only
warm-up, the soft start provided immediate upstream gradients and modestly
advanced the force-learning trajectory without changing the 3,000-step cost.

| Metric | Exact-zero start | Scale-0.1 start | Change |
|---|---:|---:|---:|
| Validation F RMSE at step 1,500 (meV/A) | 25.6166 | 25.4446 | -0.7% |
| Validation F RMSE at step 3,000 (meV/A) | 24.3407 | 24.2784 | -0.3% |
| Held-out E RMSE (meV/atom) | 0.2308 | 0.2286 | -1.0% |
| Held-out F RMSE (meV/A) | 25.0118 | 24.9398 | -0.3% |
| Audit-subset corrected E RMSE (meV/atom) | 0.171591 | 0.166404 | -3.0% |
| Audit-subset corrected F RMSE (meV/A) | 24.8505 | 24.7637 | -0.3% |
| Max 0.1-A rigid-translation E change (meV) | 0.005127 | 0.004808 | -6.2% |
| Max translated residual-force change (meV/A) | 0.025500 | 0.017646 | -30.8% |
| Largest net residual-force RMS component (meV/A) | 0.019113 | 0.013885 | -27.4% |

All strict checks pass. Scale 0.1 is therefore preferred over projection-only
warm-up for exploratory water fits. The exact-zero start remains the default
for backward compatibility and when the initial combined prediction must equal
frozen MACE exactly. Because initialization variants have now been selected
using this development split, the small accuracy gain requires replication
across seeds or a fresh holdout before it is treated as confirmatory.

### Result summary

The following section records the earlier generic, unsymmetrized 3D-FNO pilot.
All promised implementation invariants pass for both checkpoints. The learned
seed-17 correction is nevertheless strongly dependent on the mesh origin and
Cartesian orientation. The controlled seed-29 correction is nearly zero, so
the same architectural violations are correspondingly tiny in absolute units.

| Diagnostic | Seed 17 | Seed 29, fixed validation |
|---|---:|---:|
| Predicted residual E RMS (meV/atom) | 0.085873 | 0.001753 |
| Predicted residual F RMS (meV/A) | 8.3321 | 0.01235 |
| Largest net residual-force component RMS (meV/A) | 4.3766 | 0.01370 |
| Max E change under a 0.1-A rigid translation (meV) | 1.2323 | 0.002676 |
| Max residual-force change under that translation (meV/A) | 2.9544 | 0.004769 |
| Max E change under a cubic signed-axis transformation (meV) | 35.8290 | 0.001674 |
| Max cubic residual-force equivariance RMSE (meV/A) | 10.5110 | 0.001100 |

For seed 17, the largest cubic energy change is comparable to or greater than
the learned residual-energy scale, and the cubic force-equivariance error is
larger than the predicted residual-force RMS. This means the apparent energy
improvement cannot yet be interpreted as a physically valid long-range
correction.

The frozen MACE control validates the audit transformation: across the same
cubic operations, its maximum energy change is only `4.17e-5` meV and its
maximum force-equivariance RMSE is `6.99e-6` meV/A.

### Promised exact invariants

For seed 17, which supplies the more stringent nonzero-residual test:

| Check | Maximum observed | Required | Verdict |
|---|---:|---:|---|
| Latent-source neutrality | `5.52e-14` | `1e-10` | PASS |
| Total = frozen + residual force | `4.48e-12` meV/A | `1e-5` meV/A | PASS |
| One-grid translation energy | `2.05e-13` meV | `1e-7` meV | PASS |
| Full-lattice translation energy | `3.50e-13` meV | `1e-7` meV | PASS |
| Translation derivative vs net force | `2.25e-6` meV/A | `0.05` meV/A | PASS |
| Residual-force finite difference | `1.47e-6` meV/A | `0.05` meV/A | PASS |

These results confirm that the reported residual forces are conservative and
that periodic indexing, batching and source neutrality are correct. They do
not establish invariance under arbitrary translations or cubic rotations.

### Interpretation

1. The present single-origin particle mesh has a measurable continuous
   translation (egg-box) error. Exact grid and lattice translations still pass.
2. The generic 3D spectral weights do not respect the cubic signed-permutation
   group. This is the dominant physical defect in the seed-17 model.
3. Seed 29 is not evidence that the symmetry problem disappears: its learned
   residual is approximately zero, so every residual-dependent error shrinks.
4. Hyperparameter or multi-seed optimization should be deferred until the
   symmetry treatment is corrected; otherwise a model may fit the fixed
   orientation of the training cell rather than a rotationally valid response.

### Recommended implementation order

1. Completed: native EqGINO radial spectral weights enforce the full cubic
   signed-axis group without an inference-time 48-fold average.
2. Completed: three-dimensional `2 x 2 x 2` mesh interlacing reduces the
   egg-box diagnostics by 11-25 x at about 4.1 x training time.
3. Completed: projection-only warm-up was rejected; scale-0.1 joint soft-start
   training modestly advanced force convergence and preserved every audit
   contract.
4. Next: replicate the scale-0.1 comparison across independent seeds or a
   fresh holdout before changing the global default.

### Fallacy scan

- **Coverage**: 11/11 statistical fallacy types checked.
- Simpson, ecological, Berkson, collider, base-rate, regression-to-mean,
  survivorship, causal-inference and reverse-causality fallacies are not
  applicable to these deterministic invariant checks.
- **Look-elsewhere / garden of forking paths (CAUTION)**: multiple seeds and
  implementation variants have already been inspected. The current published
  test set should therefore be treated as development evidence; a fresh final
  holdout will be needed for any confirmatory performance claim.

### Output files

- `les_water_fno_3d_seed17_float64_audit.json`
- `les_water_fno_3d_seed29_fixedval_float64_audit.json`
- `les_water_eqgino_g1_m4_3d_seed17_steps3000_float64_audit.json`
- `les_water_eqgino_g1_m4_i2_3d_seed17_steps3000_float64_audit.json`
- `les_water_eqgino_g1_m4_i2_warmup250_lr3e-3_3d_seed17_steps3000_float64_audit.json`
- `les_water_eqgino_g1_m4_i2_softstart0.1_3d_seed17_steps3000_float64_audit.json`
- `../../logs/audit-les-water-3d-15387074.out`
- `../../logs/audit-les-water-3d-15387077.out`
- `../../logs/les-water-fno-3d-15388338.out`
- `../../logs/audit-les-water-3d-15388339.out`
- `../../logs/les-water-fno-3d-15388628.out`
- `../../logs/audit-les-water-3d-15388629.out`
- `../../logs/les-water-fno-3d-15388819.out`
- `../../logs/audit-les-water-3d-15388822.out`
