# Synthetic screened-Coulomb (Yukawa) benchmark

This workflow reproduces the synthetic benchmark of the manuscript, in which the
nonlocal interaction is known by construction. It covers the data generation and
the training of every model compared on these data: the local MACE baselines,
MACE-LES, and frozen and joint MACE1-FNO.

## System

A charge-neutral, NaCl-like ionic system of 1146 ions with charges +-0.4 e in a
periodic cubic cell of side 32 A. The reference energy is

    E = E_LJ + s_kappa E_Y(kappa)

- `E_LJ` is a Lennard-Jones interaction (epsilon = 0.20 eV, sigma = 3.0 A),
  identical for all ion pairs and switched smoothly to zero between 3.6 and
  4.5 A, the cutoff of the MACE backbone.
- `E_Y` is a Gaussian-smeared Yukawa interaction with the reciprocal-space
  kernel `4 pi / (k^2 + kappa^2) exp(-sigma^2 k^2)`, with sigma = 0.6 A and
  k_cut = 3.5 A^-1, evaluated as a reciprocal-space sum up to k_cut.
- The scale `s_kappa` fixes the standard deviation of the nonlocal energy at
  5 meV/atom: 1.851, 2.048 and 2.992 for kappa = 0.05, 0.2 and 0.5 A^-1.

`gen_yukawa.py` generates the configurations with Langevin MD and labels them.
It also provides the analytical reference as an ASE calculator
(`YukawaLJCalculator`).

## Datasets

| dataset | kappa (A^-1) | frames (train/valid/test) | state | used for |
|---|---|---|---|---|
| `static/` | 0.05, 0.2, 0.5 | 660/240/300 | 400 K, see caveats | Static accuracy and spectral comparisons |
| `md2800/data_k<tag>/` | 0.05, 0.5 | 660/240/300, plus 150/45 close-contact frames | 2800 K fluid | MD models of Fig. 4 |

Both datasets have 40 trajectories of 30 frames each, split by trajectory into
22/8/10 for training, validation and testing, so that no trajectory is shared
between subsets. The training files also contain two isolated-atom entries
(`config_type=IsolatedAtom`, `traj=-1`) for the MACE atomic reference energies.

**Static benchmark.** All three kappa share one set of configurations. These
are driven by the unscaled kappa = 0.05 potential at 400 K for 5.25 ps from
random positions, then relabelled for each kappa with `s_kappa`. They are thus
not an equilibrium sample of the labelled Hamiltonian, and the system is an
arrested, disordered state at this temperature. This does not affect the static
benchmark, but it rules these data out for MD.

**MD data.** Each kappa is driven and labelled by its own scaled potential
(`--fixed-scale`), with `s_kappa` from the static benchmark, so the Hamiltonian
is the one the static models learn. At 2800 K the system is a diffusive fluid,
with self-diffusion coefficients of about 4.2e-5 cm^2/s for both ions and
negligible dependence on kappa. Each trajectory has a 10 ps burn-in, followed
by 30 frames 1 ps apart. Models trained only on these frames turn attractive
just below the closest sampled pair separations (2.58 A Na-Cl, 2.62 A like
pairs), which MD at 2800 K occasionally reaches. `gen_contacts.py` therefore
adds 150 training frames per kappa, built from training frames, in which the
closest pair of one type (Na-Cl, Na-Na and Cl-Cl in turn, 50 each) is moved
symmetrically to r ~ U(2.25, 2.55) A with all other atoms fixed. It also adds a
held-out set of 45 such frames built from test frames. The MD models train on
`kappa<tag>_train_contact.xyz`, the training file followed by the 150 contact
frames, and use the unchanged thermal validation and test sets.

## Generate the data

The data are written to `DATA_ROOT`, by default `data/synthetic_yukawa/` in the
repository (ignored by Git). On a CPU node with 40 cores:

    bash benchmarks/synthetic_yukawa/prepare_dataset.sh static       # ~75 min
    bash benchmarks/synthetic_yukawa/prepare_dataset.sh md 0.05      # ~9 h
    bash benchmarks/synthetic_yukawa/prepare_dataset.sh md 0.5       # ~9 h
    bash benchmarks/synthetic_yukawa/prepare_dataset.sh contacts     # a few min
    bash benchmarks/synthetic_yukawa/prepare_dataset.sh verify

`prepare_dataset.slurm` runs one stage per job:

    sbatch --export=ALL,STAGE=md,KAPPA=0.05 --output=/path/outside/the/checkout/%x_%j.log \
        benchmarks/synthetic_yukawa/prepare_dataset.slurm

`verify` compares every file with `checksums.sha256`, which holds the SHA-256
digests of the data used in the manuscript. Generation is deterministic for a
given seed and number of parallel trajectories (`NPROC=40` for the published
files). The static `metadata.json` is not compared. The published file predates
the `generation` record of the generator, which is the only difference.

Verification status:
- the MD datasets regenerate byte-identically (checked on 2026-09-25).
- `gen_contacts.py` reproduces all six contact files and `data_contact_summary.json` byte-identically (2026-10-01).
- the static benchmark regenerates byte-identically (2026-10-01).

## LES protocol

The manuscript uses the nonresidual first block
`RealAgnosticInteractionBlock`, matching the synthetic local backbones.
The [matched LES campaign](../synthetic_les_nonresidual/README.md) supplies
all twenty static/heterogeneous fits. Each completed 1,200 epochs, switching
to stage two at epoch 450. Fig. 4 uses separate
[contact-trained LES fits and MD](nonresidual_les_md/README.md), also with
1,200 epochs and the nonresidual first block.

The generic `mace.yaml` below does not specify `interaction_first` and depends
on the installed MACE default. Use the explicit matched campaign for the
manuscript LES results. The generic configurations are available for
comparisons using the installed MACE defaults.

## Train the models

Runs are written below `MACE_FNO_WORK_ROOT` (see `benchmarks/runtime_paths.sh`),
in `synthetic_yukawa/<dataset>/kappa_<tag>/<model>_seed<seed>/`.

**MACE baselines and MACE-LES** (`train_mace.slurm`, settings in `mace.yaml`):

    sbatch --export=ALL,KAPPA=0.05,MODEL=mace1 benchmarks/synthetic_yukawa/train_mace.slurm
    sbatch --export=ALL,KAPPA=0.05,MODEL=mace2 benchmarks/synthetic_yukawa/train_mace.slurm
    sbatch --export=ALL,KAPPA=0.05,MODEL=les   benchmarks/synthetic_yukawa/train_mace.slurm

All three use a 4.5 A cutoff and 128x0e+128x1o hidden irreps, with
energy:force weights 1:100 and then 100:100 in stage two (SWA). MACE1 and MACE2
run 200 epochs with stage two from epoch 150. MACE1-LES runs 1200 epochs with
stage two from epoch 450, and needs a MACE installation that provides the
`MACELES` model. Add `DATASET=md` for the MD models, which train on the
contact-augmented data. Use `RESTART_LATEST=1` to continue an interrupted run.

**MACE1-FNO** (`train_fno.slurm`, settings in `train_fno_frozen.yaml` and
`train_fno_joint.yaml`):

    sbatch --export=ALL,KAPPA=0.05,MODE=frozen benchmarks/synthetic_yukawa/train_fno.slurm
    sbatch --export=ALL,KAPPA=0.05,MODE=joint  benchmarks/synthetic_yukawa/train_fno.slurm

The default backbone is the MACE1 run with the same dataset, kappa and
`MACE_SEED`. Override it with `MACE_MODEL`. The synthetic FNO settings are:
- a 44^3 mesh, m = 11, C = 4 and two Fourier layers.
- learning rate 1e-3, and 1e-5 for the backbone in joint training.
- energy and force scales of 0.65/0.50/0.40 eV/atom (set by kappa) and 1.0 eV/A.
- validation every 100 steps.
- checkpoints selected by the validation objective.
- early stopping after 2000 steps without improvement.

Frozen runs have a 100k-step budget, about 36 h on one A100. Joint runs have
200k steps. One 48 h job reaches about 125-155k, so resubmit with `RESUME=1` to
continue from the saved training state. The published joint runs were trained
in two legs: the first to 100k steps, then continued to 200k.

For the static benchmark, the in-training spectral diagnostic samples eight
validation configurations, which gives the exponents of the synthetic spectral table. The MD
models train without it (`SPECTRAL_DIAGNOSTIC_SAMPLES=0`). The diagnostic uses
its own random generator and does not change the training trajectory.

Seeds in the manuscript:
- MACE backbones, MACE2 and the MD models use seed 17.
- The FNO models use FNO seeds 17, 29, 41, 53 and 67 on the seed-17 backbone (`FNO_SEED`).
- MACE1-LES uses seeds 17, 29, 41, 53 and 67, trained independently (`MACE_SEED`).

The architecture variants of Appendix B.2 change one or two settings through
the overrides `CHANNELS`, `FNO_LAYERS`, `MODES` and `GRID`, for example
`CHANNELS=16` or `FNO_LAYERS=4 MODES=8`.

`MODE=linear` (settings in `train_fno_linear.yaml`) trains a linear control on
the frozen backbone. A single linear spectral operator replaces the Fourier
layers, so that the nonlocal energy is a quadratic form of the source fields
with a learned isotropic kernel. The source head and all other settings are
those of the frozen runs, except that the output starts from zero
(`output_initialization_scale: 0.0`).

The YAML files reproduce the original configurations. The explicit
nonresidual-first LES replacements use the campaign linked above.
- **FNO:** parsing `train_fno_{frozen,joint,linear}.yaml` with the options
  passed by `train_fno.slurm` gives the same resolved configuration as the
  original run for all ten frozen and joint runs (static at three kappa, MD at
  two) and for the three static linear runs.
- **MACE and MACE-LES:** `mace.yaml`, rendered as in `train_mace.slurm`, matches
  all fifteen MACE, MACE2 and MACE-LES configurations, apart from file paths and
  run names.

## Analysis and molecular dynamics

The nonresidual-first LES MD workflow and Fig. 4 update are in
[nonresidual_les_md](nonresidual_les_md/README.md). The original reference,
local-MACE, and FNO trajectories keep their recorded inputs. Shell-resolved
analysis and spectral-fit plotting data are documented in the local manuscript
source under `docs/source/data/yukawa_shells/` and `yukawa_spectral/`.
Extended LES is a separate exploratory calculation and is not included in
the paper.
