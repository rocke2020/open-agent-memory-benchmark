#!/usr/bin/env bash

# Run the generic Hugging Face dataset downloader in OAMB's locked environment.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/../.." && pwd)"
readonly PYTHON_SCRIPT_PATH="$REPO_ROOT/scripts/download/hf_dataset.py"

exec uv run --locked --group download --project "$REPO_ROOT" python "$PYTHON_SCRIPT_PATH"
