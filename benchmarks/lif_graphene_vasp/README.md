# Symmetric LiF/graphene/LiF VASP pilot

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: plan-to-execution handoff
- Origin Date: 2026-08-29
- Verification Status: generated inputs require local validation before submission
- Version Label: lif_graphene_vasp_v1

## Scientific scope

This is a tractable DFT pilot for comparing long-range response across PBE,
PBE0, and an exploratory PBE0+rVV10L calculation. Each configuration contains a
4x4 graphene sheet and two LiF molecules related by inversion through a graphene
hollow site. Consequently the total dipole vanishes and the INCAR files contain
neither `LDIPOL`/`IDIPOL` nor Coulomb-kernel truncation.

- Graphene: 32 C atoms, `a = 2.460 A`, 4x4 supercell.
- Cell: `9.84 x 9.84 x 40 A` (120-degree lateral lattice).
- Adsorbates: two LiF molecules, gas-phase bond length `1.564 A`.
- Height: LiF bond midpoint to the graphene plane.
- Heights: 3.5, 4, 5, 6, 7, 8, and 10 A.
- Orientations: Li closest on both faces, F closest on both faces, and a
  fully inversion-paired in-plane orientation.
- Sampling: Gamma-centred 4x4x1 mesh and `ENCUT = 600 eV`.
- Task: static energies and conservative forces; no relaxation.

The 10 A member of each orientation is the primary internal zero. The reported
curve is

```text
Delta E(h) / LiF = [E(h) - E(10 A)] / 2.
```

This avoids a polar single-molecule reference calculation. The 10 A zero and
40 A cell are pilot settings, not proof of asymptotic convergence; selected
vacuum, lateral-size, k-mesh, and height-reference checks are still required.

## Functionals

- `pbe`: PBE starting from atomic charge densities.
- `pbe0`: unscreened PBE0, restarted from the corresponding PBE WAVECAR and
  CHGCAR; `ALGO=Normal` activates VASP 6's ACE path.
- `pbe0_rvv10`: PBE0 plus the rVV10 nonlocal term with `IVDW_NL=2`,
  `BPARAM=10.0`, and `CPARAM=0.0093`, restarted from PBE0.

The last stage is deliberately labelled **exploratory PBE0+rVV10L(b=10)**.
The `b=10` damping is inherited from published PBE+rVV10L; it has not been
established here as a universally parametrized PBE0+rVV10 functional. It should
therefore be interpreted as a dispersion-sensitivity calculation.

PBE 5.4 PAW datasets are used in the order `C`, `Li_sv`, `F`. The more-electron
Li dataset is also appropriate for checking the PAW sensitivity of adsorption.

## Generate and validate

From this directory:

```bash
python3 generate_inputs.py
python3 validate_inputs.py
python3 validate_vasprun.py calculations/000_h03p50_li_near/pbe/vasprun.xml \
    --expected-atoms 36 --require-static
bash -n run_stage.slurm submit_chain.sh
./submit_chain.sh debug-gpu 2 --dry-run
```

Generation writes 21 case directories, each containing complete `pbe`, `pbe0`,
and `pbe0_rvv10` VASP inputs. `cases.tsv`, `cases.list`, and `generation.json`
are reproducibility manifests.

## Submit

Run a two-job-at-a-time debug chain:

```bash
./submit_chain.sh debug-gpu 2
```

For the mandatory one-case end-to-end smoke test, select one array index with
`CASE_ARRAY`:

```bash
CASE_ARRAY=0 ./submit_chain.sh debug-gpu 1
```

Comma-separated selections such as `CASE_ARRAY=0,6,14` are also accepted.

or a four-job-at-a-time production chain:

```bash
./submit_chain.sh gpu 4
```

Each array member uses one node and four GPUs, matching the module stack and
VASP 6.5.0 OpenACC executable in `~/bin/sub-vasp_gpu`. Whole-array `afterok`
dependencies ensure PBE0 starts only after all PBE cases finish, and rVV10
starts only after all PBE0 cases finish. On resubmission, a calculation is
skipped only when `validate_vasprun.py` confirms a complete XML document,
electronic and ionic convergence, a static 36-atom result, and finite energy
and forces. A newly run calculation that fails this validation exits nonzero,
preventing dependent functional stages from starting. No jobs are submitted by
the generator or validator.

## Collect

After calculations finish:

```bash
python3 collect_results.py
```

`results.csv` includes total energy, relative energy per LiF, the net force on
each LiF molecule, their antisymmetric interaction-force estimate, and the
top-plus-bottom force mismatch. It uses the same `validate_vasprun.py` success
criterion as the submission script. For a well-converged inversion-symmetric
run, the last quantity should be close to zero.

## Mandatory checks before using as reference data

1. Run one `debug-gpu` case through all three stages and inspect electronic
   convergence, GPU distribution, force inversion, and the PBE0 restart.
2. Repeat representative near, intermediate, and 10 A points with a 50 A cell.
3. Check one near-binding and one long-distance point with at least a 5x5x1 or
   6x6x1 k mesh.
4. Repeat selected points in a larger graphene supercell to quantify lateral
   LiF-image interactions.
5. Treat PBE0+rVV10L(b=10) as sensitivity evidence unless its damping choice is
   separately validated for this system.

## Authoritative VASP documentation

- PBE0 setup: <https://vasp.at/wiki/List_of_hybrid_functionals>
- Hybrid minimization and `ALGO`: <https://vasp.at/wiki/ALGO>
- rVV10 and PBE+rVV10L settings: <https://vasp.at/wiki/Nonlocal_vdW-DF_functionals>
- Fock singularity treatment: <https://vasp.at/wiki/HFRCUT>
