#!/usr/bin/env bash

# Download one verified file from the LongMemEval cleaned dataset repository.
#
# From the OAMB repository root:
#   ./scripts/download/longmemeval.sh
#   ./scripts/download/longmemeval.sh --file longmemeval_m_cleaned.json
#   ./scripts/download/longmemeval.sh --file longmemeval_oracle.json
#
# The default downloads the S variant used by the published Hindsight benchmark.
# The default revision and supported file hashes are frozen for reproducibility.

set -euo pipefail

readonly DATASET_REPOSITORY="xiaowu0162/longmemeval-cleaned"
readonly DEFAULT_REVISION="98d7416c24c778c2fee6e6f3006e7a073259d48f"
readonly DEFAULT_DATASET_FILE="longmemeval_s_cleaned.json"

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPOSITORY_ROOT=$(cd "$SCRIPT_DIR/../.." && pwd)
OUTPUT_DIR="$REPOSITORY_ROOT/datasets/longmemeval-cleaned"
REVISION="$DEFAULT_REVISION"
DATASET_FILE="$DEFAULT_DATASET_FILE"

usage() {
  cat <<'EOF'
Download and verify a file from xiaowu0162/longmemeval-cleaned.

Usage:
  longmemeval.sh [--output DIR] [--revision REVISION] [--file PATH]

Options:
  --output DIR         Destination directory.
                       Default: datasets/longmemeval-cleaned
  --revision REVISION  Hugging Face branch, tag, or commit.
                       Default: pinned OAMB dataset revision
  --file PATH          Repository file to download.
                       Default: longmemeval_s_cleaned.json
  -h, --help           Show this help.

Examples:
  # Hindsight/LongMemEval-S benchmark input
  ./scripts/download/longmemeval.sh

  # Larger retrieval-scale variant, approximately 2.74 GB
  ./scripts/download/longmemeval.sh --file longmemeval_m_cleaned.json

  # Oracle-retrieval variant containing only evidence sessions
  ./scripts/download/longmemeval.sh --file longmemeval_oracle.json
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
    --revision)
      [ "$#" -ge 2 ] || fail "--revision requires a value"
      REVISION=$2
      shift 2
      ;;
    --file)
      [ "$#" -ge 2 ] || fail "--file requires a repository path"
      DATASET_FILE=$2
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

case "$DATASET_FILE" in
  longmemeval_s_cleaned.json)
    EXPECTED_SHA256="d6f21ea9d60a0d56f34a05b609c79c88a451d2ae03597821ea3d5a9678c3a442"
    ;;
  longmemeval_m_cleaned.json)
    EXPECTED_SHA256="9d79e5524794a2e6900a3aa9cb7d9152c5a3e8319c9a87c25494ba1eacee495f"
    ;;
  longmemeval_oracle.json)
    EXPECTED_SHA256="821a2034d219ab45846873dd14c14f12cfe7776e73527a483f9dac095d38620c"
    ;;
  *)
    fail "unsupported LongMemEval file: $DATASET_FILE"
    ;;
esac
readonly EXPECTED_SHA256

env \
  DATASET_REPOSITORY="$DATASET_REPOSITORY" \
  DATASET_REVISION="$REVISION" \
  DATASET_FILE="$DATASET_FILE" \
  OUTPUT_DIRECTORY="$OUTPUT_DIR" \
  EXPECTED_SHA256="$EXPECTED_SHA256" \
  EXPECTED_CHECKSUMS_FILE= \
  REVISION_FILE="$OUTPUT_DIR/REVISION" \
  CHECKSUM_FILE="$OUTPUT_DIR/SHA256SUMS" \
  "$SCRIPT_DIR/hf_dataset.sh"
