#!/bin/bash

set -euo pipefail

readonly ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly STATE_FILE="$ROOT/.local-demo/quick-start-current.json"
readonly CELLS=("hindsight-lme60" "mem0-lme60" "openviking-lme60")

MODE="smoke"
MODE_SELECTED=false
DRY_RUN=false

usage() {
  cat <<'EOF'
Usage: ./run.sh [--smoke_test | --full_test] [--dry-run]

--smoke_test  Run one LME-60 question on every provider (default).
--full_test   Run the bounded proof, then all 60 questions on every provider.
--dry-run     Validate configuration and readiness with zero model/provider calls.
EOF
}

die() {
  printf 'run: FAIL: %s\n' "$*" >&2
  exit 1
}

open_report() {
  local report=$1
  if [[ "${OAMB_NO_OPEN:-0}" == "1" ]]; then
    printf 'report: %s\n' "$report"
    return
  fi
  case "$(uname -s)" in
    Darwin) command -v open >/dev/null 2>&1 || die "open command not found"; open "$report" ;;
    Linux) command -v xdg-open >/dev/null 2>&1 || die "xdg-open command not found"; xdg-open "$report" ;;
    *) die "cannot open report on this operating system: $report" ;;
  esac
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --smoke_test|--full_test)
      [[ "$MODE_SELECTED" == false ]] || die "choose only one of --smoke_test or --full_test"
      [[ "$1" == --smoke_test ]] && MODE="smoke" || MODE="full"
      MODE_SELECTED=true
      shift
      ;;
    --dry-run)
      DRY_RUN=true
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      usage >&2
      exit 2
      ;;
  esac
done

command -v jq >/dev/null 2>&1 || die "required command not found: jq"
command -v uv >/dev/null 2>&1 || die "required command not found: uv"
[[ -f "$STATE_FILE" ]] || die "run ./precheck.sh first"

RUN_LABEL="$(jq -er '.run_label | select(type == "string" and length > 0)' "$STATE_FILE")"
WORK_DIR="$(jq -er '.work_dir | select(type == "string" and length > 0)' "$STATE_FILE")"
PLAN="$(jq -er '.resolved_plan | select(type == "string" and length > 0)' "$STATE_FILE")"
DATASET_SOURCE="$(jq -er '.dataset_source | select(type == "string" and length > 0)' "$STATE_FILE")"
QUESTION_ID="$(jq -er '.question_id | select(type == "string" and length > 0)' "$STATE_FILE")"
case "$WORK_DIR" in
  "$ROOT/.local-demo/"*) ;;
  *) die "precheck state points outside .local-demo" ;;
esac
[[ -f "$PLAN" ]] || die "resolved plan is missing; rerun ./precheck.sh"
[[ -f "$DATASET_SOURCE" ]] || die "dataset is missing; rerun ./precheck.sh"
PLAN_HASH="$(jq -er '.resolved_plan_hash | select(type == "string" and test("^[0-9a-f]{64}$"))' "$PLAN")"

if [[ "$DRY_RUN" == true ]]; then
  "$ROOT/provider-services/bin/provider-services" doctor
  uv run --locked python - "$PLAN" "$ROOT/.env" "$ROOT/provider-services/.runtime" \
    "${CELLS[@]}" <<'PY'
import sys
from pathlib import Path

from oamb.config.doctor import load_resolved_plan_for_run
from oamb.live import load_live_environment, validate_live_readiness_receipt

plan_path = Path(sys.argv[1])
env_file = Path(sys.argv[2])
runtime = Path(sys.argv[3])
expected_cells = tuple(sys.argv[4:])
plan = load_resolved_plan_for_run(plan_path)
actual_cells = tuple(cell.cell_id for cell in plan.cells)
if actual_cells != expected_cells:
    raise SystemExit(f"unexpected plan cells: {actual_cells!r}")
environment = load_live_environment(
    provider_env_path=env_file,
    model_env_path=env_file,
    provider_runtime_directory=runtime,
    base_environment={},
)
validate_live_readiness_receipt(
    plan=plan,
    provider_runtime_directory=runtime,
    environment=environment,
)
PY
  [[ "$MODE" == smoke ]] && QUESTION_COUNT=1 || QUESTION_COUNT=60
  printf 'run: PASS (dry-run, %s, %s questions, 3 providers, zero model/provider calls)\n' \
    "$MODE" "$QUESTION_COUNT"
  exit 0
fi

MODE_DIR="$WORK_DIR/$MODE"
mkdir -p "$MODE_DIR/results" "$MODE_DIR/validations" "$MODE_DIR/capsules"

BOUNDED_ROOTS=()
BOUNDED_VALIDATIONS=()
VALIDATION_PATH=""
RESULT_MAP_PATH=""

result_map_is_complete() {
  local result_map=$1
  shift
  local expected_cells
  expected_cells="$(jq -cn --args '$ARGS.positional | sort' "$@")"
  jq -e --arg plan_hash "$PLAN_HASH" --argjson expected "$expected_cells" '
    . as $map |
    (keys | sort) == ["capsule_roots", "cells", "resolved_plan_hash",
                      "schema_name", "schema_version", "status"] and
    .schema_name == "live_run_result_map" and
    .schema_version == 1 and
    .resolved_plan_hash == $plan_hash and
    .status == "completed" and
    (.capsule_roots | type) == "object" and
    (.capsule_roots | keys | sort) == $expected and
    (.cells | type) == "array" and
    ([.cells[].cell_id] | sort) == $expected and
    all(.cells[];
      (keys | sort) == ["capsule_root", "cell_id", "detail", "status"] and
      .status == "completed" and
      .detail == null and
      (.capsule_root | type) == "string" and
      (.capsule_root | length) > 0 and
      $map.capsule_roots[.cell_id] == .capsule_root)
  ' "$result_map" >/dev/null 2>&1
}

find_reusable_result_map() {
  local preferred=$1
  shift
  local candidate cell capsule_root complete
  for candidate in "$preferred" "${preferred%.json}"-retry-*.json; do
    [[ -f "$candidate" ]] || continue
    result_map_is_complete "$candidate" "$@" || continue
    complete=true
    for cell in "$@"; do
      capsule_root="$(jq -er --arg cell "$cell" '.capsule_roots[$cell]' "$candidate")"
      if [[ ! -d "$capsule_root" ]]; then
        complete=false
        break
      fi
    done
    if [[ "$complete" == true ]]; then
      RESULT_MAP_PATH=$candidate
      return 0
    fi
  done
  RESULT_MAP_PATH=""
  return 1
}

validate_capsule() {
  local capsule_root=$1
  local preferred_path=$2
  local validation_path=$preferred_path
  if [[ -f "$preferred_path" ]] && \
    jq -e '.disposition == "validated"' "$preferred_path" >/dev/null 2>&1; then
    VALIDATION_PATH=$preferred_path
    return
  fi
  if [[ -e "$preferred_path" ]]; then
    validation_path="${preferred_path%.json}-retry-$(date -u +%Y%m%d-%H%M%S)-$$.json"
  fi
  [[ ! -e "$validation_path" ]] || die "validation output already exists: $validation_path"
  uv run --locked oamb capsule validate "$capsule_root" --output "$validation_path"
  jq -e '.disposition == "validated"' "$validation_path" >/dev/null || \
    die "capsule validation is not valid: $validation_path"
  VALIDATION_PATH=$validation_path
}

run_bounded_cells() {
  local cell provider result_map capsule_root validation
  # These small proof slices stay serial because they share one operator-provided
  # model/embedding endpoint without a separate smoke-test rate budget.
  for cell in "${CELLS[@]}"; do
    provider="${cell%-lme60}"
    result_map="$MODE_DIR/results/bounded-$provider.json"
    validation="$MODE_DIR/validations/bounded-$provider.json"
    capsule_root=""
    if find_reusable_result_map "$result_map" "$cell"; then
      result_map=$RESULT_MAP_PATH
      capsule_root="$(jq -er --arg cell "$cell" '.capsule_roots[$cell]' "$result_map")"
    fi
    if [[ -z "$capsule_root" || ! -d "$capsule_root" ]]; then
      if [[ -e "$result_map" ]]; then
        result_map="${result_map%.json}-retry-$(date -u +%Y%m%d-%H%M%S)-$$.json"
      fi
      uv run --locked oamb run "$PLAN" \
        --cell "$cell" --question "$QUESTION_ID" \
        --run-label "$RUN_LABEL-$MODE-bounded-$provider-retry-$$" \
        --output-root "$MODE_DIR/capsules/bounded" \
        --result-map "$result_map"
      capsule_root="$(jq -er --arg cell "$cell" '.capsule_roots[$cell]' "$result_map")"
    else
      printf 'run: reusing completed %s capsule\n' "$cell"
    fi
    [[ -d "$capsule_root" ]] || die "bounded capsule is missing for $cell"
    validate_capsule "$capsule_root" "$validation"
    BOUNDED_ROOTS+=("$capsule_root")
    BOUNDED_VALIDATIONS+=("$VALIDATION_PATH")
  done
}

build_comparison() {
  local expected_cases=60
  local expected_results=180
  local roots=()
  local validations=()
  if [[ "$MODE" == "smoke" ]]; then
    expected_cases=1
    expected_results=3
    roots=("${BOUNDED_ROOTS[@]}")
    validations=("${BOUNDED_VALIDATIONS[@]}")
  else
    roots=("${FULL_ROOTS[@]}")
    validations=("${FULL_VALIDATIONS[@]}")
  fi
  local comparison="$MODE_DIR/comparison"
  if [[ -e "$comparison" ]]; then
    comparison="$MODE_DIR/comparison-retry-$(date -u +%Y%m%d-%H%M%S)-$$"
  fi
  if [[ "$MODE" == "smoke" ]]; then
    uv run --locked oamb compare "$PLAN" \
      --cell-root "${CELLS[0]}=${roots[0]}" \
      --cell-root "${CELLS[1]}=${roots[1]}" \
      --cell-root "${CELLS[2]}=${roots[2]}" \
      --validation "${CELLS[0]}=${validations[0]}" \
      --validation "${CELLS[1]}=${validations[1]}" \
      --validation "${CELLS[2]}=${validations[2]}" \
      --dataset-source "$DATASET_SOURCE" \
      --output-root "$comparison" \
      --diagnostic
  else
    uv run --locked oamb compare "$PLAN" \
      --cell-root "${CELLS[0]}=${roots[0]}" \
      --cell-root "${CELLS[1]}=${roots[1]}" \
      --cell-root "${CELLS[2]}=${roots[2]}" \
      --validation "${CELLS[0]}=${validations[0]}" \
      --validation "${CELLS[1]}=${validations[1]}" \
      --validation "${CELLS[2]}=${validations[2]}" \
      --dataset-source "$DATASET_SOURCE" \
      --output-root "$comparison"
  fi

  jq -e \
    --argjson cases "$expected_cases" \
    --argjson results "$expected_results" \
    '.coverage == {
       "cell_count": 3,
       "unique_case_count": $cases,
       "provider_specific_result_count": $results
     }' "$comparison/report.json" >/dev/null || die "comparison coverage is invalid"
  [[ -s "$comparison/report.html" ]] || die "comparison HTML is missing or empty"
  open_report "$comparison/report.html"
  printf 'run: PASS (%s, %s questions, %s provider results)\n' \
    "$MODE" "$expected_cases" "$expected_results"
  printf 'report: %s\n' "$comparison/report.html"
}

cd "$ROOT"
run_bounded_cells

FULL_ROOTS=()
FULL_VALIDATIONS=()
if [[ "$MODE" == "full" ]]; then
  full_result="$MODE_DIR/results/full.json"
  full_complete=false
  if find_reusable_result_map "$full_result" "${CELLS[@]}"; then
    full_result=$RESULT_MAP_PATH
    full_complete=true
  fi
  if [[ "$full_complete" == true ]]; then
    printf 'run: reusing completed full capsules\n'
  else
    if [[ -e "$full_result" ]]; then
      full_result="${full_result%.json}-retry-$(date -u +%Y%m%d-%H%M%S)-$$.json"
    fi
    uv run --locked oamb run "$PLAN" \
      --run-label "$RUN_LABEL-full-retry-$$" \
      --output-root "$MODE_DIR/capsules/full" \
      --result-map "$full_result" \
      --bounded-capsule "${CELLS[0]}=${BOUNDED_ROOTS[0]}" \
      --bounded-capsule "${CELLS[1]}=${BOUNDED_ROOTS[1]}" \
      --bounded-capsule "${CELLS[2]}=${BOUNDED_ROOTS[2]}" \
      --bounded-validation "${CELLS[0]}=${BOUNDED_VALIDATIONS[0]}" \
      --bounded-validation "${CELLS[1]}=${BOUNDED_VALIDATIONS[1]}" \
      --bounded-validation "${CELLS[2]}=${BOUNDED_VALIDATIONS[2]}"
  fi
  for cell in "${CELLS[@]}"; do
    provider="${cell%-lme60}"
    capsule_root="$(jq -er --arg cell "$cell" '.capsule_roots[$cell]' "$full_result")"
    validation="$MODE_DIR/validations/full-$provider.json"
    [[ -d "$capsule_root" ]] || die "full capsule is missing for $cell"
    validate_capsule "$capsule_root" "$validation"
    FULL_ROOTS+=("$capsule_root")
    FULL_VALIDATIONS+=("$VALIDATION_PATH")
  done
fi

build_comparison
