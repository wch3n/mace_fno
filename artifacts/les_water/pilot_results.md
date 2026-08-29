## Material Passport

- Origin Skill: experiment-agent
- Origin Mode: run
- Origin Date: 2026-08-29
- Verification Status: UNVERIFIED
- Version Label: exp_result_v1

## Experiment Result

- **ID**: les-water-frozen-mace-3d-fno-pilot
- **Type**: training
- **Status**: completed
- **Working Directory**: `/gpfs/home/acad/ucl-modl/wchen/mxene_proj/mace_fno`
- **Baseline Job**: `15382131` (44 s wall time)
- **3D-FNO Smoke Job**: `15382136` (38 s wall time)
- **Seed-17 Pilot Job**: `15382138` (131 s wall time)
- **Controlled Seed-29 Replication Job**: `15383269` (127 s wall time)
- **Exit Codes**: all zero

### Material and protocol

The baseline is the published `H20_stagetwo.model` from ChengUCB/les_fit. Its
fit script specifies a one-interaction-layer MACE with `r_max=4.5` angstrom,
`128x0e + 128x1o` hidden irreps, and a 5% validation fraction. The downloaded
checkpoint has Git blob `38b42fdc8765bad6dbbae2397989b7b06b7881eb`.

The benchmark contains 604 published training and 50 independent test
configurations. Every configuration is H128O64 (192 atoms), has full 3D PBC,
and uses the identical 12.429-angstrom cubic cell. There are no exact
coordinate/cell duplicates between the train and test files. The test set was
used only after validation-based checkpoint selection.

For controlled FNO replication, both initialization seeds use the same 30
validation indices generated in the original seed-17 pilot. The remaining 574
training structures are identical between runs.

### 3D-FNO configuration

- Fully periodic cubic B-spline particle mesh: `24 x 24 x 24`
- Retained Fourier modes: `4 x 4 x 4`
- Neutral latent source channels: 4
- Source hidden channels: 64
- Nonlinear 3D FNO: 16 hidden channels, 2 layers
- Batch size: 2
- Optimizer: Adam, learning rate `3e-4`
- Training: 1,500 steps, validation every 250 steps
- Loss scales from frozen-MACE training RMSE only: 0.291 meV/atom and
  26.590 meV/angstrom
- Energy and force weights: 1:1 after normalization
- Model dtype: float64 (the published float32 checkpoint is promoted before
  both baseline and residual evaluation)

### Aggregate results

All energy values are RMSE in meV/atom; forces are RMSE in meV/angstrom.

| Model | Best step | Validation E | Validation F | Test E | Test F |
|---|---:|---:|---:|---:|---:|
| Frozen one-layer MACE | 0 | 0.2904 | 26.8975 | 0.2352 | 27.4193 |
| MACE + 3D FNO, seed 17 | 1500 | 0.2147 | 27.0626 | 0.1975 | 27.5265 |
| MACE + 3D FNO, seed 29, fixed validation | 250 | 0.2903 | 26.8975 | 0.2353 | 27.4195 |

Relative to frozen MACE, seed 17 reduces validation energy RMSE by 26.1% and
test energy RMSE by 16.0%, while increasing force RMSE by 0.6% and 0.4%,
respectively. Its normalized validation objective decreases by 21.7%. The
controlled seed-29 replication is effectively unchanged from the baseline.

An earlier seed-29 run used a seed-dependent validation split and is retained
only as a protocol diagnostic (`les_water_fno_3d_seed29_float64.pt`); it is not
used in the controlled comparison above.

### Interpretation boundary

The software result is successful: fully periodic deposition, 3D FFTs,
conservative force backpropagation, frozen-MACE coupling, caching, validation
selection, and held-out evaluation all run end to end. Peak reported memory was
about 2.3 GiB and a 1,500-step run took roughly two minutes on one A100.

The scientific result is preliminary. One seed finds a transferable energy
correction, while the second controlled seed finds essentially no correction;
neither improves forces. Therefore this pilot does not yet establish a robust
long-range correction. Because every snapshot has the same cell and system
size, it also cannot distinguish genuinely long-range learning from additional
cell-specific collective model capacity.

### Recommended next experiment

1. Keep the fixed validation indices and run at least three initialization
   seeds.
2. Increase the effective batch size substantially; the current batch of two
   is noisy relative to the very small residual signal and memory headroom is
   ample.
3. Test an energy-first warm-up followed by joint energy/force optimization,
   or reduce the force weight, because the observed benefit is energy-only.
4. Benchmark transfer to a larger water supercell or a cell-size series. This
   is the decisive control for whether the learned correction is long-ranged.

### Output files

| File | Purpose |
|---|---|
| `H20_r4.5_nl0_baseline.json` | Complete frozen-MACE aggregate metrics |
| `les_water_fno_3d_smoke_float64.pt` | Ten-step end-to-end smoke checkpoint |
| `les_water_fno_3d_seed17_float64.pt` | Seed-17 selected checkpoint |
| `les_water_fno_3d_seed29_fixedval_float64.pt` | Controlled seed-29 checkpoint |
| `cache/train-float64.pt` | Frozen train predictions and residual labels |
| `cache/test-float64.pt` | Frozen test predictions and residual labels |

### Anomalies detected

- Strong initialization-seed sensitivity.
- Energy improvements did not extend to forces.
- No crashes, stalls, timeout, non-finite values, or resource anomalies.

### Sources

- Fit script and checkpoint:
  https://github.com/ChengUCB/les_fit/tree/main/MLIPs/MACE-LES/water/mace-r-4.5-nl-0
- Published benchmark data:
  https://github.com/ChengUCB/les_fit/tree/main/data-benchmark

