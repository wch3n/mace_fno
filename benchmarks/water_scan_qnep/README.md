# Water-SCAN 3D MACE+FNO benchmark

This benchmark uses the neutral bulk-water SCAN data from Zenodo record
[18335947](https://zenodo.org/records/18335947). Each configuration contains
128 water molecules (384 atoms), full three-dimensional periodicity, reference
energy and forces, and an aligned cubic cell. The cell length is not fixed: it
spans approximately 11.28--15.86 A.

## Data protocol

Run from the repository root:

    python3 benchmarks/water_scan_qnep/prepare_dataset.py --download \
      --data-root /path/outside/the/checkout/water_scan/source \
      --output-dir /path/outside/the/checkout/water_scan/prepared

Add --download-published-models to verify and retain the published NEP/QNEP
parameter files too. The script checks the official MD5 digests, audits every
frame and creates:

- train.xyz: 471 unique geometries
- validation.xyz: 52 unique geometries
- test.xyz: 134 unique geometries
- split_manifest.json: source checksums, all duplicate-group memberships,
  retained source indices, label consistency checks, and split checksums

The released training file has 1388 entries but 523 distinct geometries. The
released validation file has 500 entries and 267 distinct geometries; 366
entries (133 geometries) also occur in the source training pool. This overlap
is present in the checksum-verified source files, not introduced by our split.

We keep the first occurrence of each source-training geometry, then reserve
52 of these 523 representatives using cell-length-stratified sampling with
seed 17. The test contains the first occurrence of each source-validation
geometry absent from the entire source-training pool: 134 structures.
No exact geometry occurs twice within or across prepared splits.

Identity includes exact float64 Cartesian coordinates, species, cell and PBC,
and permits atom permutations. It does not round positions or identify rotated,
periodic-image-equivalent, or near-duplicate configurations. This check does
not establish trajectory independence. Duplicate energy/force labels are
checked for consistency before selecting representatives, with tolerances of
1e-6 eV per structure and 1e-5 eV/A per force component; labels are not averaged.

The protocol is `exact-geometry-disjoint-source-roles-v1`. Existing legacy
1249/139/500 outputs are not silently accepted or overwritten; use a new output
directory. Prepared files are checksum-verified when reused. Raw data are
never rewritten.

All MACE baselines must be trained from scratch on this split. Do not reuse
a legacy MACE checkpoint, residual checkpoint, or preprocessed cache: those
models saw geometries that are no longer in the training split. Train both
one- and two-interaction MACE models at r_max=4.5 A, then use the new one-layer
checkpoint for frozen and joint MACE-FNO runs. The benchmark's current shell
interpolation is selected with `metric_parameterization: shell_spline`.

Literature NEP/qNEP errors use a different evaluation protocol and are contextual
references, not a matched clean-test comparison. The old water-SCAN results
and spectral diagnostics must not be relabeled as results on this cleaned set.

## Models

The ordinary MACE baseline is deliberately local: r_max=4.5 A and one
message-passing interaction. This corresponds to the nl-0 naming used by the
liquid-water LES setup and creates a meaningful residual-learning test.

The residual job freezes this MACE checkpoint and trains a nonlinear 3D
metric-aware EqGINO correction. The data use different cubic volumes, so the
job selects the explicit isotropic cell mode. That mode:

- accepts only positive uniform scalings of the reference cubic cell;
- deposits each structure using its own cell and voxel volume;
- supplies log(cubic cell length in A) as a spatially constant operator input;
- preserves translation and rigid-rotation symmetry through physical
  reciprocal-space magnitudes.

The fixed-cell mode remains the default for all existing jobs and checkpoints.
The Water-SCAN job uses no volume interlacing by default, which keeps the
20,000-step run tractable.
Its reproducible FNO defaults are stored in
[`train_fno_3d.yaml`](train_fno_3d.yaml); the Slurm wrapper supplies runtime
paths, seed, dtype, and explicit environment overrides. Set `FNO_CONFIG` to a
copied and edited YAML file for a distinct configuration.

[`train_joint_3d.yaml`](train_joint_3d.yaml) retains the established
60,000-step float64 protocol and jointly fine-tunes MACE at `1e-5` after a
500-step FNO-only warm-up. It is the controlled end-to-end comparison with the
frozen-backbone result reported above.

## Running

Prepare the data and submit the complete dependency graph with:

    bash benchmarks/water_scan_qnep/submit.sh

The launcher creates a timestamped experiment below
`$MACE_FNO_WORK_ROOT/water_scan_qnep/runs/`. Dataset preparation, scheduler
logs, job working directories, models, caches, checkpoints, and reports all
remain outside the Git checkout. Set `RUN_ID` to give the experiment a stable
name.

To reuse an existing frozen MACE model and run a longer residual optimization:

    RUN_ID=metric-eqgino-60k-s17-f64 \
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
add a spectral loss. The checkpoint stores the diagnostic history, including
the fitted free exponent and fixed-\(1/k^2\) log-space \(R^2\), alongside the
selected model. The trainer also writes the same history to the adjacent
`water_scan_fno_3d_seed17_<dtype>_spectral_training.json`. It is appropriate
only for the fully periodic 3D FNO with one mesh origin.

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
- cubic signed-axis invariance/covariance for metric-aware EqGINO.

A useful result requires held-out force and energy improvement over frozen
MACE, not merely a lower training residual. Compare with the constant-offset
energy control and require the strict physics audit to pass. This dataset does
not label a separately defined long-range energy, so it tests whether a
nonlocal correction improves transfer beyond the deliberately local baseline;
it does not by itself prove that the learned residual is electrostatic.
