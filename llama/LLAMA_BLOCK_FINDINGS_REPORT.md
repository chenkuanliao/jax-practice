# Llama Block Findings Report

## Run Summary

Benchmarks were run for one Llama block with:

- Shape: `seq=8192`, `hidden=4096`, `heads=32`, `kv_heads=8`, `head_dim=128`, `ffn=11008`
- Dtype: `float32`
- Timing: `warmup=2`, `steps=10`
- 4-GPU topology: `1_island_4gpu_v100`, mesh `(2, 2)` with axes `("x", "y")`
- 8-GPU topology: `2_islands_8gpu_v100`, mesh `(2, 4)` with axes `("x", "y")`
- Estimated total FLOPs per block: `4.003e12`

Plots generated from these runs:

- `data/mean_runtime_4gpu.png`
- `data/mean_runtime_8gpu.png`
- `data/mean_gpu_speedup.png`

## Mean Runtime

| Variant | Strategy | 4-GPU Mean (ms) | 8-GPU Mean (ms) | 4-to-8 Speedup |
| --- | --- | ---: | ---: | ---: |
| `v1` | user baseline | 100.220 | 56.751 | 1.766x |
| `v2` | minimal manual constraints | 102.345 | 57.665 | 1.775x |

## Main Findings

`v1` is the fastest absolute runtime on both GPU counts. It ran in `100.220 ms`
(mean) on 4 GPUs and `56.751 ms` on 8 GPUs. This indicates the original mixed layout
from the plan is the best steady-state choice for this shape among the tested
variants.

`v2` shows slightly better 4-to-8 **mean** speedup (`1.775x` vs `1.766x`) but is
not the fastest absolute runtime. It trails `v1` by about `2.1%` on 4 GPUs and about
`1.6%` on 8 GPUs. This mirrors the main lesson from the ABCD benchmarks: scaling ratio
and fastest wall-clock time are related but not interchangeable metrics.

The benchmark set has been reduced to the two strongest layouts. The old
attention-head, FFN-only, and sequence-stress variants were removed from the
runnable code path, and the old `v5` minimal-constraint strategy is now `v2`.

## Throughput

| Variant | 4-GPU Effective TFLOPS | 8-GPU Effective TFLOPS |
| --- | ---: | ---: |
| `v1` | 39.941 | 70.535 |
| `v2` | 39.112 | 69.417 |

`v1` has the highest effective TFLOPS on both topologies. `v2` is close, but the
extra freedom from fewer manual constraints did not improve steady-state
runtime for this run.

## Sharding Notes

No K/V-head fallback was recorded in these results. The benchmark used
`kv_heads=8`, which is divisible by both tested `y` mesh sizes: `2` on 4 GPUs
and `4` on 8 GPUs.

The strongest layouts are the two mixed approaches:

- `v1`: explicit constraints throughout major attention and FFN intermediates.
- `v2`: explicit input/output sharding with constraints only at major phase
  boundaries.

## Conclusion

Use `v1` as the best current implementation for absolute runtime. Treat `v2` as
the best scaling experiment and a useful comparison point when testing other
sequence lengths or dtypes.
