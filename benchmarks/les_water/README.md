# LES liquid-water metric-aware EqGINO benchmark

This workflow revisits the RPBE-D3 liquid-water benchmark used in the LES
paper. It contains 604 training configurations and 50 published test
configurations, each with 64 water molecules in a fixed 12.429 A cubic cell.
The frozen baseline is the published one-interaction MACE model with a 4.5 A
cutoff.

The earlier cubic EqGINO run reduced test force RMSE from 27.419 to 24.203
meV/A and energy RMSE from 0.2352 to 0.2133 meV/atom. The best validation
checkpoint occurred at step 10,500 of 20,000, and only one seed was run. The
present workflow uses the current metric-aware EqGINO implementation and first
tests whether gradient noise or force weighting limited that result.

## Data and model

Clone `ChengUCB/les_fit` outside this repository, then stage the exact inputs
under the external runtime root:

    python3 benchmarks/les_water/prepare_benchmark.py \
      --les-fit-root /path/to/les_fit

The staging script verifies SHA-256 checksums for both XYZ files and the
published MACE checkpoint. The 30 validation indices are versioned here so
every fit uses the same 574/30 development split. The 50 published test
structures are never used for gradient updates or checkpoint selection.

## Optimization pilot

Submit three matched 8,000-step runs with:

    bash benchmarks/les_water/submit_pilot.sh

All outputs are written below
`$MACE_FNO_WORK_ROOT/les_water/runs/<RUN_ID>/`, not inside the checkout. The
pilot compares:

- `b2-fw1`: batch two and equal normalized energy/force weights, reproducing
  the main optimization choices of the older run with the current operator;
- `b8-fw1`: batch eight to reduce gradient noise;
- `b8-fw4`: batch eight with a fourfold force weight in both optimization and
  validation-based checkpoint selection.

Every run uses float64, a 24 x 24 x 24 mesh, four retained modes per axis,
four latent source channels, two nonlinear EqGINO blocks, and scale-0.1 output
initialization. Metric-aware EqGINO evaluates its radial weights at physical
reciprocal magnitudes. During optimization, one randomly selected half-grid
origin is used per batch; validation and inference average all eight origins.

The launcher writes `reports/pilot-summary.md` after all jobs complete. The
pilot is an optimization screen rather than a final statistical comparison:
the selected setting should subsequently be repeated for at least three
initialization seeds, followed by strict symmetry, conservativity, and
low-wavevector response audits.

## Interpretation

This comparison is deliberately more demanding than the LES fit. LES jointly
optimizes the local MLIP and its latent-charge branch while prescribing an
Ewald kernel. Here, the published local MACE and its descriptors remain fixed,
and EqGINO must learn both latent source fields and a global response operator
from the remaining energy/force residual. A smaller improvement does not by
itself show that the learned operator lacks capacity; it may also mean that a
separately optimized local baseline leaves a residual that is poorly aligned
with the frozen descriptors.

A later architectural ablation should therefore compare a frozen local
residual head, EqGINO alone, and the two heads together. That control will test
whether EqGINO is learning a genuinely nonlocal contribution rather than being
asked to absorb residual local fitting error.

## Joint MACE-EqGINO fit

`train_joint_3d.yaml` enables the end-to-end alternative while retaining the
same mesh, split, loss normalization, and held-out test set. It first trains the
new EqGINO branch for 500 steps, then fine-tunes MACE at `1e-5` while the FNO
continues at `3e-4`. Use the existing launcher with a distinct external run
directory and checkpoint name:

```bash
FNO_CONFIG="$PWD/benchmarks/les_water/train_joint_3d.yaml" \
RUN_ROOT="$MACE_FNO_WORK_ROOT/les_water/joint" \
CHECKPOINT="$MACE_FNO_WORK_ROOT/les_water/joint/model.pt" \
sbatch benchmarks/les_water/train_fno_3d.slurm
```

The frozen and joint fits should use identical development/test splits. The
joint checkpoint is larger because it contains the fine-tuned MACE state in
addition to the EqGINO state.
