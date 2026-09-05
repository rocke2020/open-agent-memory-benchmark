#!/bin/bash

set -euo pipefail

readonly ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly OUTPUTS_ROOT="$ROOT/outputs"
readonly PRECHECK_ROOT="$OUTPUTS_ROOT/tmp/precheck"
readonly STATE_FILE="$OUTPUTS_ROOT/tmp/quick-start-current.json"
readonly CELLS=("hindsight-lme60" "mem0-lme60" "openviking-lme60")
readonly STATUS_INTERVAL_SECONDS=30

. "$ROOT/provider-services/lib/plan_environment.sh"

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

count_progress_artifacts() {
  local output_root=$1
  local artifact_kind=$2
  if [[ ! -d "$output_root" ]]; then
    printf '0\n'
    return
  fi
  find "$output_root" -type f \
    -path "*/source/$artifact_kind/*.json" -print 2>/dev/null | \
    wc -l | tr -d ' '
}

status_heartbeat() {
  local label=$1
  local output_root=$2
  local initial_attempts=$3
  local initial_cases=$4
  local total_questions=$5
  local elapsed_seconds=0
  local timer_pid=""
  local attempts cases completed_operations completed_questions question_progress
  trap '[ -z "$timer_pid" ] || kill "$timer_pid" 2>/dev/null || true; exit 0' TERM INT HUP
  while :; do
    sleep "$STATUS_INTERVAL_SECONDS" &
    timer_pid=$!
    wait "$timer_pid" || exit 0
    elapsed_seconds=$((elapsed_seconds + STATUS_INTERVAL_SECONDS))
    attempts="$(count_progress_artifacts "$output_root" attempts)"
    cases="$(count_progress_artifacts "$output_root" cases)"
    completed_operations=$((attempts - initial_attempts))
    completed_questions=$((cases - initial_cases))
    ((completed_operations >= 0)) || completed_operations=0
    ((completed_questions >= 0)) || completed_questions=0
    question_progress=$((completed_questions * 100 / total_questions))
    ((question_progress <= 100)) || question_progress=100
    printf 'run: %s status=running, elapsed=%ss, completed_operations=%s, completed_questions=%s, question_progress=%s%% (%s/%s)\n' \
      "$label" "$elapsed_seconds" "$completed_operations" "$completed_questions" \
      "$question_progress" "$completed_questions" "$total_questions"
  done
}

run_with_status() {
  local label=$1
  local context=$2
  local output_root=$3
  local total_questions=$4
  shift 4
  local initial_attempts initial_cases heartbeat_pid command_code
  initial_attempts="$(count_progress_artifacts "$output_root" attempts)"
  initial_cases="$(count_progress_artifacts "$output_root" cases)"
  printf 'run: %s status=starting (%s)\n' "$label" "$context"
  status_heartbeat \
    "$label" "$output_root" "$initial_attempts" "$initial_cases" "$total_questions" &
  heartbeat_pid=$!
  command_code=0
  "$@" || command_code=$?
  kill "$heartbeat_pid" 2>/dev/null || true
  wait "$heartbeat_pid" 2>/dev/null || true
  if ((command_code != 0)); then
    printf 'run: %s status=failed\n' "$label" >&2
    return "$command_code"
  fi
  printf 'run: %s status=execution-completed, question_progress=100%% (%s/%s)\n' \
    "$label" "$total_questions" "$total_questions"
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

if [[ "$MODE" == "smoke" ]]; then
  readonly TOTAL_STEPS=5
  readonly COMPARISON_STEP=4
  readonly REPORT_STEP=5
else
  readonly TOTAL_STEPS=6
  readonly COMPARISON_STEP=5
  readonly REPORT_STEP=6
fi

command -v jq >/dev/null 2>&1 || die "required command not found: jq"
command -v uv >/dev/null 2>&1 || die "required command not found: uv"
[[ -f "$STATE_FILE" ]] || die "run ./precheck.sh first"

RUN_LABEL="$(jq -er '.run_label | select(type == "string" and length > 0)' "$STATE_FILE")"
WORK_DIR="$(jq -er '.work_dir | select(type == "string" and length > 0)' "$STATE_FILE")"
PLAN="$(jq -er '.resolved_plan | select(type == "string" and length > 0)' "$STATE_FILE")"
DATASET_SOURCE="$(jq -er '.dataset_source | select(type == "string" and length > 0)' "$STATE_FILE")"
QUESTION_ID="$(jq -er '.question_id | select(type == "string" and length > 0)' "$STATE_FILE")"
case "$RUN_LABEL" in
  ""|"."|".."|*[!A-Za-z0-9._-]*) die "precheck state has invalid run label" ;;
esac
[[ "$WORK_DIR" == "$PRECHECK_ROOT/$RUN_LABEL" ]] || \
  die "precheck state work directory is outside outputs/tmp/precheck"
[[ "$PLAN" == "$WORK_DIR/plan/resolved-plan.json" ]] || \
  die "precheck state resolved plan is outside its work directory"
[[ -f "$PLAN" ]] || die "resolved plan is missing; rerun ./precheck.sh"
[[ -f "$DATASET_SOURCE" ]] || die "dataset is missing; rerun ./precheck.sh"
PLAN_HASH="$(jq -er '.resolved_plan_hash | select(type == "string" and test("^[0-9a-f]{64}$"))' "$PLAN")"
load_plan_model_environment "$PLAN" || die "cannot load model configuration from resolved plan"

if [[ "$DRY_RUN" == true ]]; then
  "$ROOT/provider-services/bin/provider-services" doctor
  uv run --locked python - "$PLAN" "$ROOT/.env" "$ROOT/provider-services/.runtime" \
    "${CELLS[@]}" <<'PY'
import os
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
    base_environment=os.environ,
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

if [[ "$MODE" == "smoke" ]]; then
  MODE_DIR="$OUTPUTS_ROOT/smoke-test/$RUN_LABEL"
else
  MODE_DIR="$OUTPUTS_ROOT/full-test/$RUN_LABEL"
fi
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
  local cell provider result_map capsule_root validation step_index step_label
  local combined_result pending_index provider_list
  local -a pending_cells=()
  local -a pending_arguments=()
  local -a pending_providers=()
  local -a pending_steps=()
  local -a cell_roots=()
  step_index=0
  for cell in "${CELLS[@]}"; do
    step_index=$((step_index + 1))
    provider="${cell%-lme60}"
    step_label="step $step_index/$TOTAL_STEPS provider=$provider"
    result_map="$MODE_DIR/results/bounded-$provider.json"
    validation="$MODE_DIR/validations/bounded-$provider.json"
    capsule_root=""
    if find_reusable_result_map "$result_map" "$cell"; then
      result_map=$RESULT_MAP_PATH
      capsule_root="$(jq -er --arg cell "$cell" '.capsule_roots[$cell]' "$result_map")"
      printf 'run: %s status=reusing-completed-capsule\n' "$step_label"
      cell_roots+=("$capsule_root")
    else
      pending_cells+=("$cell")
      pending_providers+=("$provider")
      pending_steps+=("$step_index")
      cell_roots+=("")
    fi
  done

  combined_result="$MODE_DIR/results/bounded.json"
  if ((${#pending_cells[@]} > 0)); then
    if find_reusable_result_map "$combined_result" "${pending_cells[@]}"; then
      combined_result=$RESULT_MAP_PATH
      for pending_index in "${!pending_cells[@]}"; do
        printf 'run: step %s/%s provider=%s status=reusing-completed-capsule\n' \
          "${pending_steps[$pending_index]}" "$TOTAL_STEPS" \
          "${pending_providers[$pending_index]}"
      done
    else
      if [[ -e "$combined_result" ]]; then
        combined_result="${combined_result%.json}-retry-$(date -u +%Y%m%d-%H%M%S)-$$.json"
      fi
      provider_list=""
      for pending_index in "${!pending_cells[@]}"; do
        cell="${pending_cells[$pending_index]}"
        provider="${pending_providers[$pending_index]}"
        provider_list="${provider_list}${provider_list:+,}${provider}"
        pending_arguments+=(--cell "$cell")
        printf 'run: step %s/%s provider=%s status=starting (%s, question %s)\n' \
          "${pending_steps[$pending_index]}" "$TOTAL_STEPS" "$provider" "$MODE" "$QUESTION_ID"
      done
      run_with_status "providers=$provider_list" "$MODE, question $QUESTION_ID" \
        "$MODE_DIR/capsules/bounded" "${#pending_cells[@]}" \
        uv run --locked oamb run "$PLAN" \
        "${pending_arguments[@]}" --question "$QUESTION_ID" \
        --run-label "$RUN_LABEL-$MODE-bounded-retry-$$" \
        --output-root "$MODE_DIR/capsules/bounded" \
        --result-map "$combined_result"
    fi
  fi

  step_index=0
  for cell in "${CELLS[@]}"; do
    provider="${cell%-lme60}"
    step_label="step $((step_index + 1))/$TOTAL_STEPS provider=$provider"
    capsule_root="${cell_roots[$step_index]}"
    if [[ -z "$capsule_root" ]]; then
      capsule_root="$(jq -er --arg cell "$cell" '.capsule_roots[$cell]' "$combined_result")"
    fi
    validation="$MODE_DIR/validations/bounded-$provider.json"
    [[ -d "$capsule_root" ]] || die "bounded capsule is missing for $cell"
    printf 'run: %s status=validating\n' "$step_label"
    validate_capsule "$capsule_root" "$validation"
    printf 'run: %s status=completed\n' "$step_label"
    BOUNDED_ROOTS+=("$capsule_root")
    BOUNDED_VALIDATIONS+=("$VALIDATION_PATH")
    step_index=$((step_index + 1))
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
  printf 'run: step %s/%s status=building-comparison\n' \
    "$COMPARISON_STEP" "$TOTAL_STEPS"
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
  printf 'run: step %s/%s status=completed\n' "$COMPARISON_STEP" "$TOTAL_STEPS"
  printf 'run: step %s/%s status=opening-report\n' "$REPORT_STEP" "$TOTAL_STEPS"
  open_report "$comparison/report.html"
  printf 'run: step %s/%s status=completed\n' "$REPORT_STEP" "$TOTAL_STEPS"
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
    printf 'run: step 4/%s providers=hindsight,mem0,openviking status=reusing-completed-capsules\n' \
      "$TOTAL_STEPS"
  else
    if [[ -e "$full_result" ]]; then
      full_result="${full_result%.json}-retry-$(date -u +%Y%m%d-%H%M%S)-$$.json"
    fi
    run_with_status "step 4/$TOTAL_STEPS providers=hindsight,mem0,openviking" \
      "60 questions, 3 providers" "$MODE_DIR/capsules/full" 180 \
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
  printf 'run: step 4/%s providers=hindsight,mem0,openviking status=validating\n' \
    "$TOTAL_STEPS"
  for cell in "${CELLS[@]}"; do
    provider="${cell%-lme60}"
    capsule_root="$(jq -er --arg cell "$cell" '.capsule_roots[$cell]' "$full_result")"
    validation="$MODE_DIR/validations/full-$provider.json"
    [[ -d "$capsule_root" ]] || die "full capsule is missing for $cell"
    validate_capsule "$capsule_root" "$validation"
    FULL_ROOTS+=("$capsule_root")
    FULL_VALIDATIONS+=("$VALIDATION_PATH")
  done
  printf 'run: step 4/%s providers=hindsight,mem0,openviking status=completed\n' \
    "$TOTAL_STEPS"
fi

build_comparison
