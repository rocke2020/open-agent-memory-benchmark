#!/usr/bin/env bash
# Download and verify the recorded BEAM-10M input without generation calls.
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"
exec uv run --locked --group download --project "$REPO_ROOT" python "$SCRIPT_DIR/beam10m.py" "$@"
