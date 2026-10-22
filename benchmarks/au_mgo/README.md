# Au₂/MgO surface benchmark

We compare local MACE, charge-only LES, and MACE–FNO for Au₂ adsorption on
MgO(001). Al substitution in the substrate changes the wetting preference,
providing a test of the response to a remote environment. The benchmark
reports energy and force errors on 500 test structures and separate Au-only
relaxations for the wetting comparison.

## Data and model comparisons

The Au–MgO–Al data and original one-interaction MACE model come from
[`ChengUCB/les_fit`](https://github.com/ChengUCB/les_fit), pinned to commit
`a886785caa4182a0effb95d95f0e402881adfc4d`. Of the 4,500 structures in the
training pool, 225 form the fixed validation subset. The resulting split is
**4,275/225/500** training/validation/test structures.
`validation_indices_seed123.txt` specifies the validation indices.

The nonlinear slab campaign uses **full z-mixing**: a spectral matrix
couples hidden channels and heights jointly at each in-plane wavevector.
Only the plane is Fourier transformed. The height mesh is finite and does
not connect periodic slab images through the vacuum. The model uses a
24 × 24 lateral mesh, 24 heights in a 22 Å window, mode parameter 8, 16 source
and hidden channels, four Fourier layers, one spectral group, and four lateral
mesh origins. Radial shell-spline weights impose the in-plane metric symmetry.

Channel-wise height mixing is a separate comparison. Its
configuration key is `z_mixing: global`. The direct **linear** slab model uses
one dense channel-height matrix when configured with `z_mixing: global`.
It has no pointwise branch or activation. See the
[operator description](../../guides/operators.md#2d-fno-for-slabs) for the equations,
parameter counts, and low-rank option.

## Campaign records

| Comparison | Preparation and evaluation records |
|---|---|
| Dense full z-mixing, MACE1 | [Test and wetting evaluations](full_z_tables/README.md) |
| Dense full z-mixing, MACE2 | [Five-seed frozen and joint campaign](full_z_mace2/README.md) |
| Training budgets | [16 October finalization](full_z_budget_20261016/README.md) |
| Direct linear FNO | [Physical linear campaign](../physical_linear_gpu/README.md), [wetting evaluation](linear_wetting/README.md) |
| Scalar–vector MACE1–LES | [Matched wetting results](matched_les_wetting/README.md) |
| Scalar–vector MACE2–LES | [Five-seed rerun](matched_mace2_les/README.md) |
| Scalar–vector local MACE2 | [Matched wetting baseline](matched_mace2_wetting/README.md) |
| LES first interaction | [Controlled seed-17 comparison](les_interaction_crosscheck/README.md) |
| Low-rank full z-mixing | [Exploratory seed-17 fits](low_rank_full_z/README.md) |

These records identify the model representation and selected checkpoint.
The scalar-only and scalar–vector MACE2 backbones have different parameter
counts. Compare results only after checking which representation was used.
The original full-mixing budget is 240,000 steps. The
[finalization record](full_z_budget_20261016/README.md) gives the lower stopping
budgets adopted after inspecting the validation curves.

## Data preparation

Run from the repository root:

```bash
export MACE_FNO_WORK_ROOT=/path/to/external/runs
source benchmarks/runtime_paths.sh
python3 benchmarks/au_mgo/prepare_benchmark.py \
  --les-fit-root /path/to/les_fit
```

This copies the XYZ files to the ignored `data/les_au_mgo/` directory and the
published MACE model to `$MACE_FNO_WORK_ROOT/les_au_mgo/pretrained/`.
The runtime helper uses a site-specific scratch path if the environment
variable is unset. Set it explicitly for a new installation.

## Introductory launchers

The launchers cover local-MACE evaluation, planar projection, and
channel-wise slab training:

```bash
sbatch benchmarks/au_mgo/evaluate_mace.slurm
sbatch benchmarks/au_mgo/train_fno_2d.slurm
sbatch benchmarks/au_mgo/train_fno_2p5d.slurm
```

`train_fno_2p5d.yaml` is an introductory channel-wise configuration.
`train_joint_2p5d.yaml` describes a 20,000-step, symmetry-averaged
joint run. Neither reproduces the full z-mixing campaign. Use the
campaign records above for manuscript reproduction, or supply a separate
configuration through `FNO_CONFIG` for a new calculation.

The slab launcher accepts grid, mode, channel, height, and training overrides.
Give each run an external `CHECKPOINT` path and a separate Slurm output path.
The launcher uses `/tmp` for scheduler output when none is supplied.

## Force and symmetry checks

```bash
source benchmarks/runtime_paths.sh
run_root="$MACE_FNO_WORK_ROOT/les_au_mgo/checks"
mkdir -p "$run_root"
sbatch --output="$run_root/check-2p5d-%j.out" \
  --export=ALL,CHECKPOINT=/path/to/checkpoint.pt,CHECK_OUTPUT="$run_root/check.json" \
  benchmarks/au_mgo/check_2p5d.slurm
```

The check tests source neutrality, translations, applicable in-plane symmetry,
and forces against finite differences. Passing these numerical checks does
not establish predictive accuracy.

## Wetting preference

The preparation command stages four DFT starting geometries. Structures `1`
and `3` are the non-wetting and wetting endpoints. The evaluator fixes the
Mg/O/Al substrate and cell, relaxes only the two Au atoms with ASE FIRE, and
requests a force tolerance of 0.01 eV/Å within 500 steps. It reports
`Delta E = E(wetting) - E(non-wetting)` for each composition. DFT gives a
positive value for undoped MgO and a negative value for Al-doped MgO.

```bash
source benchmarks/runtime_paths.sh
run_root="$MACE_FNO_WORK_ROOT/les_au_mgo/wetting_switch"
mkdir -p "$run_root"
sbatch --output="$run_root/fno-%j.out" \
  --export=ALL,MODEL_KIND=mace-fno,MODEL=/path/to/fno.pt,RUN_NAME=fno \
  benchmarks/au_mgo/evaluate_wetting.slurm
```

Use `MODEL_KIND=mace` for local MACE or MACE–LES `.model` files. The campaign
wrappers also check convergence, fixed substrate and cell, and the intended
orientation of Au₂. Form wetting differences for each seed before averaging.
Test errors and wetting outcomes do not select the checkpoint.
