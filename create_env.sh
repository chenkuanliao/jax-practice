#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_FILE="${SCRIPT_DIR}/environment.yml"

if ! command -v conda >/dev/null 2>&1; then
  echo "Error: conda is not available on PATH." >&2
  echo "Install Miniconda/Mambaforge or initialize conda before running this script." >&2
  exit 1
fi

if [[ ! -f "${ENV_FILE}" ]]; then
  echo "Error: missing ${ENV_FILE}" >&2
  exit 1
fi

ENV_NAME="$(awk -F': *' '/^name:/ {print $2; exit}' "${ENV_FILE}")"

if [[ -z "${ENV_NAME}" ]]; then
  echo "Error: could not read environment name from ${ENV_FILE}" >&2
  exit 1
fi

if conda env list | awk '{print $1}' | grep -Fxq "${ENV_NAME}"; then
  echo "Updating existing conda environment '${ENV_NAME}' from ${ENV_FILE}..."
  conda env update --name "${ENV_NAME}" --file "${ENV_FILE}" --prune
else
  echo "Creating conda environment '${ENV_NAME}' from ${ENV_FILE}..."
  conda env create --file "${ENV_FILE}"
fi

echo
echo "Done. Activate it with:"
echo "  conda activate ${ENV_NAME}"
