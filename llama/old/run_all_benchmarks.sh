#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

COMMON_ARGS=("$@")

run_group() {
  local gpu_label="$1"
  shift
  local scripts=("$@")

  echo "================================================================"
  echo "Running ${gpu_label} Llama block benchmarks (${#scripts[@]} scripts)"
  echo "================================================================"
  for script in "${scripts[@]}"; do
    echo
    echo ">>> python3 ${script} ${COMMON_ARGS[*]}"
    python3 "$script" "${COMMON_ARGS[@]}"
  done
}

FOUR_GPU_SCRIPTS=(
  "llama_4gpu_v1.py"
  "llama_4gpu_v2.py"
)

EIGHT_GPU_SCRIPTS=(
  "llama_8gpu_v1.py"
  "llama_8gpu_v2.py"
)

run_group "4-GPU" "${FOUR_GPU_SCRIPTS[@]}"
run_group "8-GPU" "${EIGHT_GPU_SCRIPTS[@]}"

echo
echo "All Llama block benchmark scripts completed successfully."
