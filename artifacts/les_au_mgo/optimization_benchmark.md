# LES Au-MgO-Al FNO optimization benchmark

Date: 2026-08-28  
GPU: NVIDIA A100-SXM4 40 GB  
Workload: 1,500 optimizer updates, four sampled configurations per update,
225 validation and 500 held-out test structures, seed 17 unless noted.

## Changes

1. Stacked MACE graphs, particle meshes and FNO fields so a batch is handled by
   one forward/backward pass.
2. Persistent caches for pruned MACE neighbor graphs, frozen-MACE energies and
   forces, and reference-minus-MACE residual labels.
3. Direct residual energy/force loss, avoiding cancellation between large LES
   total energies.
4. Batched evaluation and a `validation-test` scope that skips established
   full-training and offset controls in repeat jobs.
5. Optional hybrid float32 compute with the large atomic reference-energy sum
   reconstructed in float64.

The optimized float64 run uses `batch_size=4, accumulation_steps=1`; the old
run used `batch_size=1, accumulation_steps=4`. Both therefore use the same four
sampled configurations and one optimizer update. The first-step loss and final
seed-17 metrics are identical.

## Timing and accuracy

| Run | Cache | Slurm wall | Internal total | Optimize + validation | Test E RMSE (meV/atom) | Test F RMSE (meV/A) |
|---|---:|---:|---:|---:|---:|---:|
| Original float64, seed 17 (`15377028`) | none | 8:32 | — | ~4:21 | 0.4049 | 34.5897 |
| Optimized float64, seed 17 (`15377107`) | cold | 3:00 | 169.55 s | 96.90 s | 0.4049 | 34.5897 |
| Original float64, seed 29 (`15377047`) | none | 8:27 | — | — | 0.5032 | 36.3863 |
| Optimized float64, seed 29 (`15377113`) | warm | 2:08 | 117.98 s | 98.84 s | 0.5032 | 36.3863 |
| Plain float32 control (`15377112`) | cold | 2:37 | 146.34 s | 77.13 s | 0.7667 | 40.3563 |
| Hybrid float32, seed 17 (`15377129`) | cold | 2:39 | 147.11 s | 75.92 s | 0.4049 | 34.5915 |
| Hybrid float32, seed 29 (`15377137`) | warm | 1:47 | 96.36 s | 78.41 s | 0.5032 | 36.3870 |

The cold-cache optimized float64 job is 2.84 times faster end to end than the
original. With a warm cache, the matched seed-29 job is 3.96 times faster. The
cache occupies about 1.5 GB for all 5,000 structures and lowers peak host
memory from about 4.7 to 3.7 GiB in the float64 benchmark.

The warm-cache hybrid job is 4.74 times faster end to end than the original
matched seed-29 run (1:47 versus 8:27). It also reduces the measured internal
runtime from 117.98 s for optimized float64 to 96.36 s.

Plain float32 is not acceptable for LES absolute energies: direct float32
accumulation shifts the frozen baseline by about -3.9 meV/atom. Reconstructing
the atomic reference contribution in float64 reduces that error to
8.0e-5 meV/atom on a checked LES structure. Hybrid training then retains the
float64 energy result and changes the force RMSE by only 0.0018 meV/A for seed
17. Across the two seeds, hybrid compute reduces optimization time by 20-22%.

## Verification

- 31 unit tests pass, including batched mesh equivalence, batched graph edge
  offsets, conservative force gradients, cache round trips, residual-target
  identity, and hybrid atomic-energy precision.
- A real two-structure MACE check gave zero batched-versus-separate energy
  difference and a maximum force difference of 2.44e-15 eV/A.
- The optimized seed-17 first-step loss components exactly match the original:
  total `1.421387`, energy `0.6483886`, forces `0.7729980`.
- Frozen and final held-out metrics from optimized float64 exactly match the
  original logs.

The conservative default in the Slurm scripts remains float64. Set
`MODEL_DTYPE=float32` to use the validated hybrid path when throughput is more
important than bitwise reproduction.
