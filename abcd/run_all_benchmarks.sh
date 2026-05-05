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
  echo "Running ${gpu_label} benchmarks (${#scripts[@]} scripts)"
  echo "================================================================"
  for script in "${scripts[@]}"; do
    echo
    echo ">>> python3 ${script} ${COMMON_ARGS[*]}"
    python3 "$script" "${COMMON_ARGS[@]}"
  done
}

FOUR_GPU_SCRIPTS=(
  "abcd_4gpu_v1-1.py"
  "abcd_4gpu_v1-2.py"
  "abcd_4gpu_v1-3.py"
  "abcd_4gpu_v2-1.py"
  "abcd_4gpu_v2-2.py"
  "abcd_4gpu_v2-3.py"
)

EIGHT_GPU_SCRIPTS=(
  "abcd_8gpu_v1-1.py"
  "abcd_8gpu_v1-2.py"
  "abcd_8gpu_v1-3.py"
  "abcd_8gpu_v2-1.py"
  "abcd_8gpu_v2-2.py"
  "abcd_8gpu_v2-3.py"
)

run_group "4-GPU" "${FOUR_GPU_SCRIPTS[@]}"
run_group "8-GPU" "${EIGHT_GPU_SCRIPTS[@]}"

echo
echo "All benchmark scripts completed successfully."
