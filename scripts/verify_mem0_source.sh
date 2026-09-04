#!/bin/sh
set -eu

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
REPOSITORY_ROOT=$(CDPATH= cd -- "$SCRIPT_DIR/.." && pwd)
VERSIONS_FILE="$REPOSITORY_ROOT/provider-services/versions.env"

# Public, tracked release pins only.
. "$VERSIONS_FILE"

if [ "$#" -ne 1 ]; then
  printf 'Usage: %s MEM0_CHECKOUT\n' "$0" >&2
  exit 2
fi

MEM0_CHECKOUT=$1

if ! ACTUAL_COMMIT=$(git -C "$MEM0_CHECKOUT" \
  rev-parse "v${MEM0_VERSION}^{commit}" 2>/dev/null); then
  printf 'FAIL: cannot resolve Mem0 tag v%s in %s\n' \
    "$MEM0_VERSION" "$MEM0_CHECKOUT" >&2
  exit 1
fi

if [ "$ACTUAL_COMMIT" != "$MEM0_COMMIT" ]; then
  printf 'FAIL: Mem0 v%s expected %s, got %s\n' \
    "$MEM0_VERSION" "$MEM0_COMMIT" "$ACTUAL_COMMIT" >&2
  exit 1
fi

printf 'PASS: Mem0 v%s resolves to %s\n' "$MEM0_VERSION" "$ACTUAL_COMMIT"
