# Inhomogeneous screening benchmark

This benchmark tests whether an interatomic potential captures changes in
interionic coupling when the screening medium rearranges far from both ions.
It complements the [homogeneous Yukawa benchmark](../synthetic_yukawa/README.md)
by making the screening depend on the spatial distribution of neutral particles.
The data support the energy and force comparisons in Table V and the remote-medium
probe in Fig. 6.

## Download

The [Zenodo release](https://doi.org/10.5281/zenodo.23459503) contains the
datasets, selected checkpoints, training configurations, and generation scripts.
Download `heterogeneous_screening_20261018.tar.gz` and
`shared_code_and_manifests_20261018.tar.gz`, then extract both into the same
directory. Within the release, this benchmark is under
`benchmarks/heterogeneous_screening/`:

| Path | Contents |
|---|---|
| `data/hetscreen_{train,valid,test}.xyz` | Structures and reference energies and forces |
| `data/metadata.json` | Screening parameters, sampling settings, split identities, and acceptance checks |
| `data/probes_40sets/` | Displaced probe configurations and reference coupling tensors |
| `checkpoints/` | Selected models and required MACE backbones |
| `checkpoint_index.csv` | Model paths, methods, and seeds |
| `results/remote_medium_probe/` | Model responses, analysis, and CSV exports |

The release includes `DATA_FORMAT.md` and `GENERATION.md` for this benchmark.
Training recipes are indexed in the release-level `training_config_index.csv`;
`shared/TRAINING.md` explains their formats and path settings.

## System and reference

Each periodic cubic cell has side length 32 Å and contains 430 positive ions,
430 negative ions, and 286 neutral screening particles, denoted S. The XYZ
species are `Na`, `Cl`, and `Ar`, respectively. These are species identifiers
for a synthetic model; `Ar` encodes S. The ionic charges are ±0.4 e, smeared
by Gaussians of width 0.6 Å.

The normalized, Gaussian-smoothed concentration of S sets the screening field:

$$
\kappa^2(\mathbf r)=\kappa_0^2+\kappa_1^2 c_S(\mathbf r),
\qquad \langle c_S\rangle=1.
$$

The smoothing width is 3 Å, and the released parameters are
`kappa0 = 0.05 Å⁻¹` and `kappa1 = 0.2947772349941876 Å⁻¹`.
The [reference solver](screening_reference.py) solves the periodic screened
Poisson equation,

$$
[-\nabla^2+\kappa^2(\mathbf r)]\varphi(\mathbf r)=4\pi\rho_\sigma(\mathbf r),
$$

in a plane-wave basis. The total energy combines a species-independent,
short-range Lennard-Jones interaction and the screened electrostatic energy.
A constant self-energy in an S-free medium is subtracted. The nonlocal term
is scaled so its standard deviation over training configurations is
5 meV/atom. Forces include derivatives with respect to both ions and S.
Rearranging S changes the propagation operator even at fixed ionic positions.

## Data and splits

The configurations are sampled at 2800 K using a homogeneous Yukawa driver
and a long-wavelength external field acting on S. They are then labelled
with the inhomogeneous reference. The external field is excluded from the
labels. These are samples for supervised learning, rather than equilibrium
trajectories of the labelled Hamiltonian.

| Split | Trajectories | Structures |
|---|---|---|
| Training | 44 | 660 |
| Validation | 16 | 240 |
| Test | 20 | 300 |

Each trajectory contributes 15 frames. The training file also contains three
isolated-atom entries for the atomic reference energies. Two initial frames
are taken from each homogeneous MD trajectory; both child trajectories
inherit its split, so parent trajectories are not shared across splits.

Energies are total energies in eV, forces are in eV/Å, and positions are in Å.
ASE reads the energy and force labels through `atoms.get_potential_energy()`
and `atoms.get_forces()`. The files also contain `ref_charges`, `e_local`, and
`e_nonlocal` for reference analysis. Model training uses the structures,
energies, and forces, without physical-charge or screening-field targets.

## Remote-medium probe

The probe contains 40 ion-pair sets, two from each of the 20 test trajectories.
The selected ions A and B are 10–12 Å apart. All positions are fixed across
variants. Species within 10 Å of either ion are also fixed, while
composition-preserving species swaps change the remote environment.

Each set has six medium variants: the base configuration, four independently
biased S distributions, and one distribution sampled without the external
field. Two additional null variants exchange only remote Na/Cl identities
while keeping S fixed.

For every variant, ion B is displaced by ±0.05 Å along each Cartesian axis.
The six force evaluations give the 3 × 3 cross force-constant matrix:

$$
K_{AB}^{\alpha\beta}\simeq-
\frac{F_{A\alpha}(\mathbf R_B+\delta\mathbf e_\beta)
      -F_{A\alpha}(\mathbf R_B-\delta\mathbf e_\beta)}{2\delta}.
$$

There are 1,920 displaced configurations in total. The file
`probeiii_test_configs.xyz` contains the configurations;
`probeiii_test_reference.json` supplies the reference tensors in eV/Å².
Join them using `probe_set`/`set` and `variant`/`kind`. The JSON field
`E_ref_00` is the energy of the undisplaced variant, not a training label
for each displaced configuration.

Medium modulation is measured by centering each tensor component over the
six medium variants within a set, separately for the model and reference.
The two null variants are compared against the base configuration. Their
reference coupling is unchanged because only remote ionic identities change;
this does not imply unchanged total energies or forces.

## Generation and evaluation

The supplied data can be used directly. To regenerate the dataset, install
NumPy, SciPy, and ASE and run the following from the repository root on a
CPU allocation. Set `--nproc` to the allocated CPU count and use a separate
output directory.

```bash
python3 benchmarks/heterogeneous_screening/gen_heterogeneous.py \
  --stage all --templates /path/to/homogeneous/data/md2800/data_k0p05 \
  --out /path/to/new/inhomogeneous-data --nproc 20
```

The templates are the κ = 0.05 Å⁻¹, 2800 K data in
`homogeneous_yukawa_20261018.tar.gz`. The generator defaults specify 80
trajectories, a 2 fs timestep, 6 ps relaxation, and 15 frames spaced by 0.5 ps.
It records sampling settings and label checks in `metadata.json`.
Full MD regeneration can depend on package versions and hardware; the
released files define the benchmark inputs.

To build the 40-set test probe from those inputs:

```bash
python3 benchmarks/heterogeneous_screening/build_probe2.py \
  --data /path/to/inhomogeneous-data --split test --design iii \
  --out /path/to/new/probe-test --n-frames 20 --frame-in-traj 7 \
  --sets-per-frame 2 --n-field 4 --n-null 2 --frozen 10 \
  --sweeps 1600 --delta 0.05 --seed 20261005 --nproc 20
```

[eval_probe2.py](eval_probe2.py) evaluates MACE, MACE–LES, and MACE–FNO
checkpoints on these configurations using CUDA. FNO evaluation uses eight
mesh origins and requires the associated MACE backbone.
[analyze_probe2.py](analyze_probe2.py) compares the model and reference
responses, including slopes, trajectory-bootstrap intervals, and null controls.
Both scripts provide command-line options through `--help`.

[validate_reference.py](validate_reference.py) checks the uniform-medium
limit, grid convergence, finite-difference forces, and symmetries of the
reference. The [LES training guide](../synthetic_les_nonresidual/README.md)
documents the matched nonresidual first interaction block used in the paper.
