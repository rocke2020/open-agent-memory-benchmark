#!/usr/bin/env bash

# Download the exact MemoryAgentBench inputs required by OAMB.

set -euo pipefail

readonly DATASET_REPOSITORY="ai-hyz/MemoryAgentBench"
readonly DATASET_REVISION="7ea066982b140a19337e17e60d45d4076e042faf"

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPOSITORY_ROOT=$(cd "$SCRIPT_DIR/../.." && pwd)
OUTPUT_DIR="$REPOSITORY_ROOT/datasets/MemoryAgentBench"

usage() {
  cat <<'EOF'
Download and verify the OAMB MemoryAgentBench inputs.

Usage:
  memoryagentbench.sh [--output DIR]

Options:
  --output DIR  Destination directory.
                Default: datasets/MemoryAgentBench
  -h, --help    Show this help.
EOF
}

fail() {
  printf 'ERROR: %s\n' "$1" >&2
  exit 1
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    --output)
      [ "$#" -ge 2 ] || fail "--output requires a directory"
      OUTPUT_DIR=$2
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      fail "unknown argument: $1"
      ;;
  esac
done

env \
  DATASET_REPOSITORY="$DATASET_REPOSITORY" \
  DATASET_REVISION="$DATASET_REVISION" \
  DATASET_FILE= \
  OUTPUT_DIRECTORY="$OUTPUT_DIR" \
  EXPECTED_SHA256= \
  EXPECTED_CHECKSUMS_FILE="$SCRIPT_DIR/memoryagentbench.sha256" \
  REVISION_FILE="$OUTPUT_DIR/REVISION" \
  CHECKSUM_FILE="$OUTPUT_DIR/SHA256SUMS" \
  "$SCRIPT_DIR/hf_dataset.sh"
