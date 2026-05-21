#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON:-python3}"

BENCHMARKS=(
  "1gpu/llama_1gpu_1k.py"
  "1gpu/llama_1gpu_1kx8.py"
  "1gpu/llama_1gpu_1kx128.py"
  "1gpu/llama_1gpu_8k.py"
  "4gpu/llama_4gpu_1k_tensor_parallel.py"
  "4gpu/llama_4gpu_1k_seq_model_parallel.py"
  "4gpu/llama_4gpu_1kx8_tensor_parallel.py"
  "4gpu/llama_4gpu_1kx8_seq_model_parallel.py"
  "4gpu/llama_4gpu_1kx128_tensor_parallel.py"
  "4gpu/llama_4gpu_1kx128_seq_model_parallel.py"
  "4gpu/llama_4gpu_8k_tensor_parallel.py"
  "4gpu/llama_4gpu_8k_seq_model_parallel.py"
  "8gpu/llama_8gpu_1k_tensor_parallel.py"
  "8gpu/llama_8gpu_1k_seq_model_parallel.py"
  "8gpu/llama_8gpu_1kx8_tensor_parallel.py"
  "8gpu/llama_8gpu_1kx8_seq_model_parallel.py"
  "8gpu/llama_8gpu_1kx128_tensor_parallel.py"
  "8gpu/llama_8gpu_1kx128_seq_model_parallel.py"
  "8gpu/llama_8gpu_8k_tensor_parallel.py"
  "8gpu/llama_8gpu_8k_seq_model_parallel.py"
)

for benchmark in "${BENCHMARKS[@]}"; do
  echo "Running ${benchmark} $*"
  "${PYTHON_BIN}" "${SCRIPT_DIR}/${benchmark}" "$@"
done
