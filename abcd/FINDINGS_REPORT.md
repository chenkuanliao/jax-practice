# JAX Sharding Benchmark Report (V100, 4 GPU vs 8 GPU)

## Scope

This report summarizes benchmark results in `JAX-practice/abcd/` for the matrix
chain expression:

- `a @ b @ c @ d`
- matrix size: `8192 x 8192`
- dtype: `float32`
- platform: NVIDIA V100
- GPU topologies:
  - 4 GPU: `1_island_4gpu_v100` with mesh `(2, 2)`
  - 8 GPU: `2_islands_8gpu_v100` with mesh `(2, 4)`

Data sources:

- Benchmark scripts: `abcd_4gpu_v*.py`, `abcd_8gpu_v*.py`
- Stats JSON for this write-up: `8129_data/abcd_*_stats.json`
- Plot scripts and outputs (mean runtime from `average_runtime_ms` in JSON):
  - `8129_data/plot_mean_runtime_4gpu.py` -> `8129_data/mean_runtime_4gpu.png`
  - `8129_data/plot_mean_runtime_8gpu.py` -> `8129_data/mean_runtime_8gpu.png`
  - `8129_data/compare_gpu_speedup.py` -> `8129_data/mean_gpu_speedup.png`

## Variant Definitions

Version naming follows the plot annotations:

- `v1-*`: `multi_dot` pipeline (implicit multiply chain)
- `v2-*`: explicit stepwise matmul pipeline (`x=a@b; x=x@c; x=x@d`) with output constraints after each step
- `-1`: row+column sharding on all inputs (`P("island","lane")`)
- `-2`: row-only sharding on all inputs (`P("island", None)`)
- `-3`: mixed sharding (`a: P("island",None)`, `b/c/d: P(None,"lane")`)

All runs use `jax.jit`, `NamedSharding`, and `PartitionSpec`; each result includes compile time, timed rounds, and effective TFLOPS.

## Mean Runtime Findings

### 4 GPU (from `mean_runtime_4gpu.png` and JSON)

- `v2-3`: **65.96 ms** (best)
- `v1-3`: 72.83 ms
- `v2-1`: 73.73 ms
- `v1-1`: 78.35 ms
- `v1-2`: 84.33 ms
- `v2-2`: 86.70 ms (worst)

### 8 GPU (from `mean_runtime_8gpu.png` and JSON)

- `v2-3`: **32.96 ms** (best)
- `v2-1`: 40.16 ms
- `v1-1`: 41.19 ms
- `v1-2`: 41.37 ms
- `v2-2`: 42.09 ms
- `v1-3`: 44.12 ms (worst)

### Throughput (effective TFLOPS)

- Best 4 GPU throughput: `v2-3` at about **50.0 TFLOPS**
- Best 8 GPU throughput: `v2-3` at about **100.1 TFLOPS**

This mirrors runtime results: the fastest runtime variant is also the highest-throughput variant.

## 4->8 GPU Scaling Findings

Mean speedup (`4 GPU mean / 8 GPU mean`) from `mean_gpu_speedup.png` and JSON:

- `v2-2`: **2.06x**
- `v1-2`: **2.04x**
- `v2-3`: **2.00x**
- `v1-1`: **1.90x**
- `v2-1`: **1.84x**
- `v1-3`: **1.65x**

Interpretation:

- Most variants scale near ideal 2x when moving from 4 to 8 GPUs.
- `v1-3` is the major outlier in scaling efficiency.
- Absolute performance and scaling efficiency are not identical:
  - `v2-2` scales best but is slow in absolute terms.
  - `v2-3` delivers the best end-to-end runtime while still scaling almost 2x.

## Compile-Time Observations

- Compile times are broadly in the ~3.0s to ~4.2s range.
- Differences exist by variant, but runtime ranking is more meaningful for steady-state workloads where kernels are reused.
- Warmup rounds are important to avoid conflating compilation and first-run effects with steady-state timing.

## Recommended JAX Practices (Evidence-Based)

Based on these runs, the following practices are useful for multi-GPU JAX matmul chains:

1. Use explicit `NamedSharding` and `PartitionSpec` for each operand.
   - The benchmark variants expose that sharding choice strongly affects runtime and scaling.

2. Prefer explicit stepwise matmul with intermediate `with_sharding_constraint` for deep chains.
   - `v2-*` formulation (explicit `@` steps) is more controllable and produced the overall best variant (`v2-3`).

3. Evaluate sharding layouts empirically; do not assume symmetric sharding is always fastest.
   - Mixed sharding (`-3`) won absolute performance.
   - Row-only sharding (`-2`) scaled well but had weaker absolute runtime.

4. Separate compile, warmup, and timed phases in benchmarks.
   - The shared API correctly tracks `compile_ms`, warmup rounds, and timed rounds; this should be standard benchmarking hygiene in JAX.

5. Keep benchmark output schema stable (JSON contract) for automated comparisons.
   - The project setup enables repeatable plotting and cross-run analysis with minimal ad-hoc parsing.

6. Track both runtime and scaling metrics.
   - Choosing only by speedup can pick a slower absolute configuration; choosing only by runtime can hide scaling bottlenecks.

## Practical Recommendation for This Workload

For `8192 x 8192`, `float32`, and this V100 topology:

- Use the **`v2-3` strategy** as the baseline production candidate.
  - It is best in absolute runtime on both 4 and 8 GPUs.
  - It has near-ideal mean scaling (~2.00x).
  - It achieves the highest effective TFLOPS at both GPU counts.

Follow-up experiments that would strengthen confidence:

- Repeat at larger `timed_rounds` (for tighter confidence intervals).
- Test additional matrix sizes and dtypes (`bfloat16`, `float16`) to validate ranking stability.
- Measure communication traces (NCCL/XLA profiling) to explain why `v1-3` scaling degrades.

## Repro Commands

Run all benchmark scripts from the project root:

```bash
cd JAX-practice/abcd
./run_all_benchmarks.sh --n 8192 --warmup 2 --steps 10 --dtype float32
```

Generate plots (requires the `jax-v100` conda environment for Matplotlib):

```bash
cd JAX-practice/abcd/8129_data
conda run -n jax-v100 python3 plot_mean_runtime_4gpu.py
conda run -n jax-v100 python3 plot_mean_runtime_8gpu.py
conda run -n jax-v100 python3 compare_gpu_speedup.py
```
