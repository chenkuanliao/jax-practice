# Llama Block Benchmark Layout

This directory mirrors the `abcd/` benchmark workflow for one Llama
block. Each script runs the same configurable block with a named sharding
strategy and writes a compatible `data/*_stats.json` payload.

## Files

- `benchmark_api.py`: shared timing, JSON writing, FLOP-rate reporting, and
  terminal summaries.
- `llama_block_common.py`: shared block implementation, CLI parsing, shape
  validation, mesh setup, sharding specs, tensor initialization, and FLOP
  accounting.
- `llama_4gpu_v1.py`, `llama_4gpu_v2.py`: 4-GPU variants on a `(2, 2)` mesh
  named `("x", "y")`.
- `llama_8gpu_v1.py`, `llama_8gpu_v2.py`: 8-GPU variants on a `(2, 4)` mesh
  named `("x", "y")`.
- `run_all_benchmarks.sh`: runs all 4-GPU scripts, then all 8-GPU scripts.
- `data/*.py`: plotting helpers for mean runtime and 4-to-8 GPU speedup
  (requires Matplotlib in your Python environment).

## Variants

- `v1`: user baseline with explicit constraints through the major attention
  and FFN intermediates.
- `v2`: minimal manual constraints, keeping constraints only at major phase
  boundaries. This is the former `v5` strategy renamed to `v2`.

## Run

```bash
python3 llama_4gpu_v1.py \
  --seq 8192 \
  --hidden 4096 \
  --heads 32 \
  --kv-heads 8 \
  --head-dim 128 \
  --ffn 11008 \
  --dtype float32 \
  --warmup 2 \
  --steps 10
```

Run the full set with shared arguments:

```bash
./run_all_benchmarks.sh --seq 8192 --dtype float32 --warmup 2 --steps 10
```

For a tiny smoke run before committing GPU time:

```bash
python3 llama_4gpu_v1.py \
  --seq 16 \
  --hidden 64 \
  --heads 4 \
  --kv-heads 2 \
  --head-dim 16 \
  --ffn 128 \
  --dtype float32 \
  --warmup 1 \
  --steps 1
```

## Plot

After benchmark JSON exists in `data/`, run (from this directory, with
Matplotlib available):

```bash
python3 data/plot_mean_runtime_4gpu.py
python3 data/plot_mean_runtime_8gpu.py
python3 data/compare_gpu_speedup.py
```
