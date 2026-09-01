# Water-SCAN 3D MACE+FNO benchmark

This benchmark uses the neutral bulk-water SCAN data from Zenodo record
[18335947](https://zenodo.org/records/18335947). Each configuration contains
128 water molecules (384 atoms), full three-dimensional periodicity, reference
energy and forces, and an aligned cubic cell. The cell length is not fixed: it
spans approximately 11.28--15.86 A.

## Data protocol

Run from the repository root:

    python3 benchmarks/water_scan_qnep/prepare_dataset.py --download

Add --download-published-models to verify and retain the published NEP/QNEP
parameter files too. The script checks the official MD5 digests, audits every
frame and creates:

- data/water_scan_qnep/prepared/train.xyz: 1249 configurations
- data/water_scan_qnep/prepared/validation.xyz: 139 configurations
- data/water_scan_qnep/prepared/test.xyz: 500 configurations
- data/water_scan_qnep/prepared/split_manifest.json: provenance, checksums and
  the exact deterministic split

The 1388 official training configurations are divided 90:10. Validation
members are sampled deterministically across cell-length bins with seed 17, so
model selection covers the complete density range. The official 500-frame
validation file is treated as the held-out test set and is never used for
gradient updates or model selection.

## Models

The ordinary MACE baseline is deliberately local: r_max=4.5 A and one
message-passing interaction. This corresponds to the nl-0 naming used by the
liquid-water LES setup and creates a meaningful residual-learning test.

The residual job freezes this MACE checkpoint and trains a nonlinear 3D
EqGINO/FNO correction. The data use different cubic volumes, so the job selects
the explicit isotropic cell mode. That mode:

- accepts only positive uniform scalings of the reference cubic cell;
- deposits each structure using its own cell and voxel volume;
- supplies log(cubic cell length in A) as a spatially constant operator input;
- preserves the existing translation and signed-axis EqGINO symmetries.

The fixed-cell mode remains the default for all existing jobs and checkpoints.
The Water-SCAN job uses no volume interlacing by default, which keeps the
20,000-step run tractable.

## Running

Prepare the data and submit the complete dependency graph with:

    bash benchmarks/water_scan_qnep/submit.sh

The launcher creates a timestamped experiment below
`$MACE_FNO_WORK_ROOT/water_scan_qnep/runs/`. Dataset preparation, scheduler
logs, job working directories, models, caches, checkpoints, and reports all
remain outside the Git checkout. Set `RUN_ID` to give the experiment a stable
name.

To reuse an existing frozen MACE model and run a longer residual optimization:

    RUN_ID=eqgino-60k-s17-f64 \
    PRETRAINED_MACE_MODEL=/path/to/water-SCAN-r4p5-nl0_stagetwo.model \
    STEPS=60000 MODEL_DTYPE=float64 \
      bash benchmarks/water_scan_qnep/submit.sh

Important outputs within the experiment directory are:

- `mace/models/water-SCAN-r4p5-nl0_stagetwo.model` when MACE is retrained;
- `baseline.json`;
- `fno/water_scan_fno_3d_seed17_float32.pt`;
- the corresponding suffix _audit.json
- the corresponding suffix _spectral_response.json

All job settings can be overridden as environment variables. Useful examples:

    RUN_ID=mace-200 MAX_EPOCHS=200 START_STAGE_TWO=100 bash benchmarks/water_scan_qnep/submit.sh
    RUN_ID=small-fno STEPS=200 GRID=16 MODES=3 bash benchmarks/water_scan_qnep/submit.sh
    RUN_ID=interlaced VOLUME_INTERLACING=2 bash benchmarks/water_scan_qnep/submit.sh

The default FNO loss scales are 0.01 eV/atom and 0.10 eV/A. After the first
baseline evaluation, these can be replaced with the frozen-MACE validation
RMSE values through ENERGY_SCALE and FORCE_SCALE; doing so gives equal
dimensionless influence when the energy and force weights are both one.

The Water-SCAN training job also records a validation-only low-wavevector
diagnostic at every ordinary validation check. It uses four fixed validation
snapshots and the first three *physical* 3D reciprocal shells, but does **not**
add a spectral loss. The resulting
`water_scan_fno_3d_seed17_<dtype>_spectral_training.json` records the fitted
free exponent and the fixed-
\(1/k^2\) log-space \(R^2\) alongside the validation objective. It is
appropriate only for the fully periodic 3D FNO with one mesh origin.

For non-cubic bulk data the same implementation additionally fits individual
physical reciprocal vectors to
\(R(\mathbf k)=1/(\mathbf k^{\mathsf T}B\mathbf k)\), providing a
trace-normalized dielectric-like tensor and tensor-fit quality. A separate
2D FNO slab path resolves monopole, dipole, and quadrupole z profiles and
compares its full response with
\(2\pi e^{-k_\parallel |z-z'|}/k_\parallel\); it therefore does not misuse the
3D scalar \(1/k^2\) test for a slab.

The spectral-response audit perturbs the deposited latent mesh fields after
training and fits the dominant low-wavevector curvature to a free power law and
to a fixed \(1/k^2\) form.  This diagnoses whether an electrostatic-like
response has emerged; it does not assign a physical charge meaning to an
individual latent channel.

## Validation criteria

The baseline JSON reports unshifted energy and force metrics plus constant
energy-offset controls. The FNO trainer reports the frozen baseline before
optimization and restores the checkpoint with the best validation objective.
The strict audit then checks:

- held-out metrics from the restored residual checkpoint;
- conservative forces against finite differences;
- continuous and mesh-step translation behavior;
- full-lattice periodicity;
- cubic signed-axis invariance/covariance for EqGINO.

A useful result requires held-out force and energy improvement over frozen
MACE, not merely a lower training residual. Compare with the constant-offset
energy control and require the strict physics audit to pass. This dataset does
not label a separately defined long-range energy, so it tests whether a
nonlocal correction improves transfer beyond the deliberately local baseline;
it does not by itself prove that the learned residual is electrostatic.
