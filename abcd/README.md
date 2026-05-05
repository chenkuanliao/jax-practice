# ABCD Benchmark Layout

This directory benchmarks the matrix chain `a @ b @ c @ d` with several JAX
sharding strategies on 4 and 8 GPUs. Every implementation uses the same
benchmarking API and produces uniform JSON plus terminal output.

## Files

- `benchmark_api.py`: Shared benchmark interface (timing, stats, persistence, summary print).
- `abcd_4gpu_v1-*.py`: 4-GPU `multi_dot` variants on a `(2, 2)` mesh.
- `abcd_4gpu_v2-*.py`: 4-GPU explicit stepwise matmul variants.
- `abcd_8gpu_v1-*.py`: 8-GPU `multi_dot` variants on a `(2, 4)` mesh.
- `abcd_8gpu_v2-*.py`: 8-GPU explicit stepwise matmul variants.
- `run_all_benchmarks.sh`: runs all 4-GPU scripts, then all 8-GPU scripts.
- `8129_data/`, `16284_data/`: saved benchmark JSON and generated plots for
  previous dimension runs. Regenerate figures by running `python3` on
  `plot_mean_runtime_*.py` and `compare_gpu_speedup.py` inside those folders
  (requires Matplotlib).
- `FINDINGS_REPORT.md`: summary of the recorded benchmark results.

## Run

From the project root (`JAX-practice/`):

```bash
cd abcd
python3 abcd_4gpu_v1-1.py --n 8192 --warmup 2 --steps 10 --dtype float32
python3 abcd_8gpu_v2-3.py --n 8192 --warmup 2 --steps 10 --dtype float32
./run_all_benchmarks.sh --n 8192 --warmup 2 --steps 10 --dtype float32
```

## Add a New JAX Implementation

1. Create a new script in this folder, following the existing
   `abcd_<gpu>gpu_v<strategy>-<layout>.py` naming pattern.
2. Build your JAX function (`jax.jit` / sharding strategy).
3. Call shared API helpers from `benchmark_api.py`:
   - `make_sharded_array(...)`
   - `run_expression_benchmark(...)`
   - `make_base_result(...)`
   - `append_run_to_json(...)`
   - `print_run_summary(...)`
4. Keep output format stable:
   - one run payload with `expressions`
   - write into the appropriate data directory as `<your_file>_stats.json`
   - print shared summary via `print_run_summary`

If every implementation follows this contract, benchmarking output stays comparable across scripts.
