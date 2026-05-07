#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

SCRIPTS=(
  "abcd_4gpu_v1-1.py"
  "abcd_4gpu_v1-2.py"
  "abcd_4gpu_v1-3.py"
  "abcd_4gpu_v2-1.py"
  "abcd_4gpu_v2-2.py"
  "abcd_4gpu_v2-3.py"
)

for script in "${SCRIPTS[@]}"; do
  echo "================================================================"
  echo "Running ${script}"
  echo "================================================================"
  python "${SCRIPT_DIR}/${script}" "$@"
  echo
done

echo "All 6 profile benchmarks completed."
