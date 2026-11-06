#!/bin/bash
set -euo pipefail

source_dir=${1:?Usage: build-image.sh OAM_CHECKOUT COMMIT}
revision=${2:?Usage: build-image.sh OAM_CHECKOUT COMMIT}
commit=$(git -C "$source_dir" rev-parse --verify "$revision^{commit}")
context=$(mktemp -d "${TMPDIR:-/tmp}/oamb-oam-image.XXXXXXXX")
trap 'rm -rf -- "$context"' EXIT
git -C "$source_dir" archive "$commit" | tar -x -C "$context"
docker build --label "org.opencontainers.image.revision=$commit" \
  -t "oamb/oam:0.0.1-${commit:0:12}" \
  -f "$(dirname "$0")/Dockerfile" "$context"
