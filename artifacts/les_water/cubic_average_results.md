## Material Passport

- Origin Skill: experiment-agent
- Origin Mode: validate
- Origin Date: 2026-08-29
- Verification Status: VERIFIED
- Version Label: validation_v1

## Validation Report

- **Source**: seed-17 LES-water 3D-FNO checkpoint
- **Job**: `15387234`
- **Status**: `COMPLETED`, exit code `0:0`, wall time 24 s
- **Evaluation**: all 50 independent test structures
- **Group treatment**: raw, 24 proper cubic rotations (`O`) and all 48
  signed-axis operations (`O_h`)
- **Overall confidence**: SOLID for the controlled inference comparison;
  CAUTION for generalization because this published test set has already been
  used during model development

### Performance

| Model | E RMSE (meV/atom) | F RMSE (meV/A) | F MAE (meV/A) |
|---|---:|---:|---:|
| Frozen one-layer MACE | 0.235152 | 27.419320 | 21.010673 |
| Raw seed-17 FNO | 0.197461 | 27.526468 | 21.147122 |
| `O` average, 24 operations | 0.244412 | 26.740603 | 20.514576 |
| `O_h` average, 48 operations | 0.243818 | 26.739772 | 20.513555 |

Relative to frozen MACE, the raw checkpoint reduces energy RMSE by 16.0% but
increases force RMSE by 0.39%. Full `O_h` averaging instead reduces force RMSE
by 2.48% and force MAE by 2.37%, while increasing energy RMSE by 3.69%.

The force improvement is present in every Cartesian component:

| Axis | Frozen MACE | Raw FNO | `O_h` FNO | `O_h` vs frozen |
|---|---:|---:|---:|---:|
| x | 27.9116 | 27.9415 | 27.1386 | -2.77% |
| y | 27.0866 | 27.3236 | 26.5011 | -2.16% |
| z | 27.2528 | 27.3096 | 26.5751 | -2.49% |

### Interpretation

The cubic anisotropy was a major reason the learned FNO correction failed to
improve forces. Averaging removes the orientation-dependent component, lowers
the residual-force RMS from 8.389 to 4.631 meV/A, and converts a harmful raw
force correction into a modest useful one.

Conversely, the raw energy gain is not symmetry-robust. Across the 48 equivalent
orientations, the raw residual energy has a mean range of 68.8 meV per
structure and a maximum range of 112.9 meV. Once this dependence is averaged
out, the energy RMSE is slightly worse than frozen MACE. The raw energy result
therefore should not be treated as evidence of a valid long-range correction.

The 24- and 48-operation averages have nearly identical accuracy, but only the
48-element treatment enforces the complete signed-axis group, including
inversion/reflection operations.

### Numerical verification

- Raw metrics reproduce the original held-out evaluation: 0.1975 meV/atom and
  27.5265 meV/A.
- `O_h` invariance error: at most `1.62e-7` meV.
- `O_h` force-equivariance RMSE: at most `5.07e-8` meV/A.
- `O_h` energy-force finite-difference error: `4.35e-7` meV/A.
- Latent-source neutrality: at most `1.24e-13`.
- The complete unit suite passes: 64/64 tests.

The `O_h` average does not remove particle-mesh origin dependence. Its net
residual-force component RMS remains 3.08--3.57 meV/A, so three-dimensional
interlacing remains the next independent correction.

### Fallacy scan

- **Coverage**: 11/11 statistical fallacy types checked.
- No structural or causal fallacy applies to the deterministic symmetry test.
- **Look-elsewhere / garden of forking paths (CAUTION)**: the test set is now a
  development set for architecture decisions. A fresh holdout is required for
  a final confirmatory accuracy claim.

### Output

- `les_water_fno_3d_seed17_cubic_average.json`
- `../../logs/eval-les-water-oh48-15387234.out`
