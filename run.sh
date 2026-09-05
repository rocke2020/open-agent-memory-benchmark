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
RESUME=false

usage() {
  cat <<'EOF'
Usage: ./run.sh [--smoke_test | --full_test] [--resume] [--dry-run]

--smoke_test  Run one LME-60 question on every provider (default).
--full_test   Run the bounded proof, then all 60 questions on every provider.
--resume      Resume an interrupted --full_test from immutable whole-group parts.
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
    --resume)
      RESUME=true
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

if [[ "$RESUME" == true && "$MODE" != "full" ]]; then
  die "--resume requires --full_test"
fi
if [[ "$RESUME" == true && "$DRY_RUN" == true ]]; then
  die "--resume cannot be combined with --dry-run"
fi

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
RESUME_LOCK=""

release_resume_lock() {
  if [[ -n "$RESUME_LOCK" ]]; then
    rmdir "$RESUME_LOCK" 2>/dev/null || true
    RESUME_LOCK=""
  fi
}

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

resume_result_map_is_bound() {
  local result_map=$1
  local expected_cells
  expected_cells="$(jq -cn --args '$ARGS.positional' "${CELLS[@]}")"
  jq -e --arg plan_hash "$PLAN_HASH" --argjson expected "$expected_cells" '
    (keys | sort) == ["capsule_roots", "cells", "resolved_plan_hash",
                      "schema_name", "schema_version", "status"] and
    .schema_name == "live_run_result_map" and
    .schema_version == 1 and
    .resolved_plan_hash == $plan_hash and
    (.status == "completed" or .status == "failed") and
    (.capsule_roots | type) == "object" and
    (.cells | type) == "array" and
    [.cells[].cell_id] == $expected and
    all(.cells[];
      (keys | sort) == ["capsule_root", "cell_id", "detail", "status"] and
      (.status == "completed" or .status == "failed" or .status == "not_started") and
      (.capsule_root | type) == "string" and
      (.capsule_root | length) > 0)
  ' "$result_map" >/dev/null 2>&1
}

resume_state_is_bound() {
  local state_path=$1
  local expected_cells
  expected_cells="$(jq -cn --args '$ARGS.positional' "${CELLS[@]}")"
  jq -e --arg plan_hash "$PLAN_HASH" --argjson expected "$expected_cells" '
    (keys | sort) == ["attempt", "cells", "resolved_plan_hash",
                      "schema_name", "schema_version"] and
    .schema_name == "full_test_resume_state" and
    .schema_version == 1 and
    .resolved_plan_hash == $plan_hash and
    (.attempt | type) == "number" and .attempt >= 0 and
    .attempt == (.attempt | floor) and
    [.cells[].cell_id] == $expected and
    all(.cells[];
      (keys | sort) == ["cell_id", "final_capsule_root", "final_validation", "parts"] and
      (.parts | type) == "array" and
      (.parts | length) > 0 and
      all(.parts[]; type == "string" and length > 0) and
      (.parts | unique | length) == (.parts | length) and
      (.final_capsule_root == null or (
        (.final_capsule_root | type) == "string" and
        (.final_capsule_root | length) > 0)) and
      (.final_validation == null or (
        (.final_validation | type) == "string" and
        (.final_validation | length) > 0)))
  ' "$state_path" >/dev/null 2>&1
}

atomic_replace_resume_state() {
  local input=$1
  local temporary
  temporary="$(mktemp "$MODE_DIR/results/.full-resume-state.XXXXXX")"
  jq -cS . "$input" > "$temporary" || die "cannot serialize resume state"
  mv "$temporary" "$RESUME_STATE"
  rm -f "$input"
}

initialize_resume_state() {
  local candidate root temporary_state
  local -a candidates=()
  if [[ -e "$RESUME_STATE" ]]; then
    [[ -f "$RESUME_STATE" && ! -L "$RESUME_STATE" ]] || \
      die "resume state is not a regular file: $RESUME_STATE"
    resume_state_is_bound "$RESUME_STATE" || \
      die "resume state is malformed or does not match the current resolved plan"
  else
    for candidate in "$MODE_DIR/results/full.json" "$MODE_DIR"/results/full-retry-*.json; do
      [[ -f "$candidate" && ! -L "$candidate" ]] || continue
      resume_result_map_is_bound "$candidate" || continue
      candidates+=("$candidate")
    done
    ((${#candidates[@]} > 0)) || \
      die "no resumable full-run result map; --resume never starts a fresh full run"
    ((${#candidates[@]} == 1)) || \
      die "ambiguous full-run source selection; preserve one explicit resume state"
    candidate=${candidates[0]}
    temporary_state="$(mktemp "$MODE_DIR/results/.full-resume-initial.XXXXXX")"
    jq -cS --arg plan_hash "$PLAN_HASH" '
      {
        schema_name: "full_test_resume_state",
        schema_version: 1,
        resolved_plan_hash: $plan_hash,
        attempt: 0,
        cells: [.cells[] | {
          cell_id,
          parts: [.capsule_root],
          final_capsule_root: null,
          final_validation: null
        }]
      }
    ' "$candidate" > "$temporary_state" || die "cannot initialize resume state"
    atomic_replace_resume_state "$temporary_state"
  fi
  while IFS= read -r root; do
    [[ -d "$root" ]] || die "resume source capsule is missing: $root"
  done < <(jq -r '.cells[].parts[]' "$RESUME_STATE")
}

update_resume_state() {
  local temporary
  temporary="$(mktemp "$MODE_DIR/results/.full-resume-update.XXXXXX")"
  jq "$@" "$RESUME_STATE" > "$temporary" || die "cannot update resume state"
  atomic_replace_resume_state "$temporary"
}

append_resume_part() {
  local cell=$1
  local capsule_root=$2
  update_resume_state --arg cell "$cell" --arg root "$capsule_root" '
    .cells |= map(
      if .cell_id == $cell then
        .parts = (
          if (.parts | index($root)) == null then .parts + [$root] else .parts end
        ) |
        .final_capsule_root = null |
        .final_validation = null
      else . end
    )
  '
}

complete_resume_cell() {
  local cell=$1
  local capsule_root=$2
  local validation=$3
  update_resume_state --arg cell "$cell" --arg root "$capsule_root" \
    --arg validation "$validation" '
      .cells |= map(
        if .cell_id == $cell then
          .final_capsule_root = $root | .final_validation = $validation
        else . end
      )
    '
}

increment_resume_attempt() {
  update_resume_state '.attempt += 1'
  RESUME_ATTEMPT="$(jq -er '.attempt' "$RESUME_STATE")"
}

resume_parts_for_cell() {
  local cell=$1
  jq -er --arg cell "$cell" '.cells[] | select(.cell_id == $cell) | .parts[]' \
    "$RESUME_STATE"
}

resume_result_root() {
  local result_map=$1
  local cell=$2
  jq -er --arg plan_hash "$PLAN_HASH" --arg cell "$cell" '
    if (
      (keys | sort) == ["capsule_roots", "cells", "resolved_plan_hash",
                        "schema_name", "schema_version", "status"] and
      .schema_name == "live_run_result_map" and
      .schema_version == 1 and
      .resolved_plan_hash == $plan_hash and
      (.status == "completed" or .status == "failed") and
      (.cells | length) == 1 and
      .cells[0].cell_id == $cell and
      (.cells[0].status == "completed" or .cells[0].status == "failed") and
      (.cells[0].capsule_root | type) == "string" and
      (.cells[0].capsule_root | length) > 0
    ) then .cells[0].capsule_root else false end
  ' "$result_map"
}

validate_capsule_result() {
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
  if [[ -e "$validation_path" ]]; then
    printf 'run: capsule validation output already exists: %s\n' "$validation_path" >&2
    return 1
  fi
  if ! uv run --locked oamb capsule validate "$capsule_root" --output "$validation_path"; then
    return 1
  fi
  if ! jq -e '.disposition == "validated"' "$validation_path" >/dev/null; then
    printf 'run: capsule validation is not valid: %s\n' "$validation_path" >&2
    return 1
  fi
  VALIDATION_PATH=$validation_path
}

validate_capsule() {
  local capsule_root=$1
  local preferred_path=$2
  validate_capsule_result "$capsule_root" "$preferred_path" || \
    die "capsule validation failed: $capsule_root"
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
      if [[ "$RESUME" == true ]]; then
        die "full-test resume requires all completed bounded proof capsules"
      fi
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

run_resume_analysis() {
  local cell=$1
  local provider=$2
  local analysis_output=$3
  local log_path=$4
  local -a arguments=(
    uv run --locked oamb run "$PLAN"
    --cell "$cell"
    --run-label "$RUN_LABEL-full-resume-analysis-$RESUME_ATTEMPT-$provider"
    --output-root "$MODE_DIR/capsules/resume-analysis-$RESUME_ATTEMPT/$provider"
    --recovery-analysis-output "$analysis_output"
  )
  local part index
  while IFS= read -r part; do
    arguments+=(--recover-from "$part")
  done < <(resume_parts_for_cell "$cell")
  for index in "${!CELLS[@]}"; do
    arguments+=(--bounded-capsule "${CELLS[$index]}=${BOUNDED_ROOTS[$index]}")
    arguments+=(--bounded-validation "${CELLS[$index]}=${BOUNDED_VALIDATIONS[$index]}")
  done
  "${arguments[@]}" > "$log_path" 2>&1
}

resume_status_heartbeat() {
  local label=$1
  local output_root=$2
  local reused=$3
  local remaining=$4
  local quarantined=$5
  local elapsed_seconds=0
  local timer_pid=""
  local completed unfinished progress
  trap '[ -z "$timer_pid" ] || kill "$timer_pid" 2>/dev/null || true; exit 0' TERM INT HUP
  while :; do
    sleep "$STATUS_INTERVAL_SECONDS" &
    timer_pid=$!
    wait "$timer_pid" || exit 0
    elapsed_seconds=$((elapsed_seconds + STATUS_INTERVAL_SECONDS))
    completed="$(count_progress_artifacts "$output_root" cases)"
    ((completed <= remaining)) || completed=$remaining
    unfinished=$((remaining - completed))
    progress=$(((reused + completed) * 100 / 60))
    printf 'run: %s status=running, elapsed=%ss, reused=%s, remaining=%s, running=unavailable, completed=%s, failed=unavailable, quarantined=%s, question_progress=%s%% (%s/60)\n' \
      "$label" "$elapsed_seconds" "$reused" "$unfinished" "$completed" \
      "$quarantined" "$progress" "$((reused + completed))"
  done
}

wait_for_child() {
  local child_pid=$1
  local child_code
  while :; do
    if wait "$child_pid"; then
      return 0
    else
      child_code=$?
      kill -0 "$child_pid" 2>/dev/null || return "$child_code"
    fi
  done
}

run_resume_worker() {
  local cell=$1
  local provider=$2
  local output_root=$3
  local result_map=$4
  local log_path=$5
  local reused=$6
  local remaining=$7
  local quarantined=$8
  local command_pid=""
  local heartbeat_pid=""
  local command_code=0
  local index part
  local -a arguments=(
    uv run --locked oamb run "$PLAN"
    --cell "$cell"
    --run-label "$RUN_LABEL-full-resume-$RESUME_ATTEMPT-$provider"
    --output-root "$output_root"
    --result-map "$result_map"
  )
  while IFS= read -r part; do
    arguments+=(--recover-from "$part")
  done < <(resume_parts_for_cell "$cell")
  for index in "${!CELLS[@]}"; do
    arguments+=(--bounded-capsule "${CELLS[$index]}=${BOUNDED_ROOTS[$index]}")
    arguments+=(--bounded-validation "${CELLS[$index]}=${BOUNDED_VALIDATIONS[$index]}")
  done

  forward_worker_stop() {
    [[ -z "$command_pid" ]] || kill -TERM "$command_pid" 2>/dev/null || true
  }
  trap forward_worker_stop TERM INT HUP
  resume_status_heartbeat "$cell" "$output_root" "$reused" "$remaining" \
    "$quarantined" &
  heartbeat_pid=$!
  "${arguments[@]}" >> "$log_path" 2>&1 &
  command_pid=$!
  while :; do
    if wait "$command_pid"; then
      command_code=0
      break
    else
      command_code=$?
      kill -0 "$command_pid" 2>/dev/null || break
    fi
  done
  kill "$heartbeat_pid" 2>/dev/null || true
  wait "$heartbeat_pid" 2>/dev/null || true
  if ((command_code != 0)); then
    printf 'run: %s status=failed\n' "$cell" >> "$log_path"
    return "$command_code"
  fi
  printf 'run: %s status=execution-completed\n' "$cell" >> "$log_path"
}

compose_resume_cell() {
  local cell=$1
  local provider=$2
  local output="$MODE_DIR/capsules/composed/$provider-$RESUME_ATTEMPT"
  local validation="$MODE_DIR/validations/resume-$RESUME_ATTEMPT-$provider.json"
  local part
  local -a arguments=(
    uv run --locked oamb capsule compose --plan "$PLAN" --cell "$cell" --output "$output"
  )
  while IFS= read -r part; do
    arguments+=(--part "$part")
  done < <(resume_parts_for_cell "$cell")
  printf 'run: provider=%s status=composing\n' "$provider"
  "${arguments[@]}"
  validate_capsule "$output" "$validation"
  complete_resume_cell "$cell" "$output" "$VALIDATION_PATH"
  printf 'run: provider=%s status=completed\n' "$provider"
}

run_full_resume() {
  local cell provider analysis log_path remaining reusable quarantined progress status
  local final_root final_validation result_map output_root capsule_root part_validation
  local provider_cap batch_start batch_end index pid any_failed admitted_count completed
  local worker_pid_count=0
  local stop_count=0
  local -a pending_cells=()
  local -a pending_providers=()
  local -a pending_reused=()
  local -a pending_remaining=()
  local -a pending_quarantined=()
  local -a pending_statuses=()
  local -a active_pids=()

  RESUME_STATE="$MODE_DIR/results/full-resume-state.json"
  RESUME_LOCK="$MODE_DIR/results/full-resume.lock"
  mkdir "$RESUME_LOCK" 2>/dev/null || \
    die "another full-test resume owns the local state: $RESUME_LOCK"
  trap release_resume_lock EXIT
  initialize_resume_state
  increment_resume_attempt
  provider_cap="$(jq -er '
    .execution.max_parallel_providers_per_dataset |
    select(type == "number" and . >= 1 and . == floor)
  ' "$PLAN")"
  ((provider_cap <= ${#CELLS[@]})) || provider_cap=${#CELLS[@]}

  printf 'run: resume attempt=%s status=validating-sources\n' "$RESUME_ATTEMPT"
  for cell in "${CELLS[@]}"; do
    provider="${cell%-lme60}"
    final_root="$(jq -r --arg cell "$cell" \
      '.cells[] | select(.cell_id == $cell) | .final_capsule_root // empty' "$RESUME_STATE")"
    if [[ -n "$final_root" ]]; then
      [[ -d "$final_root" ]] || die "completed resume capsule is missing: $final_root"
      validate_capsule "$final_root" \
        "$MODE_DIR/validations/resume-recheck-$RESUME_ATTEMPT-$provider.json"
      complete_resume_cell "$cell" "$final_root" "$VALIDATION_PATH"
    fi
    analysis="$MODE_DIR/results/resume-analysis-$RESUME_ATTEMPT-$provider.json"
    log_path="$MODE_DIR/results/resume-analysis-$RESUME_ATTEMPT-$provider.log"
    [[ ! -e "$analysis" && ! -e "$log_path" ]] || \
      die "resume analysis output already exists for attempt $RESUME_ATTEMPT: $provider"
    run_resume_analysis "$cell" "$provider" "$analysis" "$log_path" &
    active_pids+=("$!")
  done

  any_failed=false
  index=0
  for cell in "${CELLS[@]}"; do
    pid=${active_pids[$index]}
    if wait_for_child "$pid"; then
      :
    else
      any_failed=true
    fi
    provider="${cell%-lme60}"
    log_path="$MODE_DIR/results/resume-analysis-$RESUME_ATTEMPT-$provider.log"
    sed -n '1,240p' "$log_path"
    index=$((index + 1))
  done
  [[ "$any_failed" == false ]] || \
    die "resume source validation failed; zero recovery dispatches were started"

  for cell in "${CELLS[@]}"; do
    provider="${cell%-lme60}"
    final_root="$(jq -r --arg cell "$cell" \
      '.cells[] | select(.cell_id == $cell) | .final_capsule_root // empty' "$RESUME_STATE")"
    [[ -z "$final_root" ]] || continue
    analysis="$MODE_DIR/results/resume-analysis-$RESUME_ATTEMPT-$provider.json"
    reusable="$(jq -er '.reusable_ingestion_plan_ids | length' "$analysis")"
    remaining="$(jq -er '.remaining_case_manifest_entry_ids | length' "$analysis")"
    quarantined="$(jq -er '.quarantined_ingestion_plan_ids | length' "$analysis")"
    progress=$((reusable * 100 / 60))
    printf 'run: provider=%s status=ready, reused=%s, remaining=%s, running=0, completed=0, failed=0, quarantined=%s, question_progress=%s%% (%s/60)\n' \
      "$provider" "$reusable" "$remaining" "$quarantined" \
      "$progress" "$reusable"
    if ((remaining == 0)); then
      compose_resume_cell "$cell" "$provider"
    else
      pending_cells+=("$cell")
      pending_providers+=("$provider")
      pending_reused+=("$reusable")
      pending_remaining+=("$remaining")
      pending_quarantined+=("$quarantined")
      pending_statuses+=("not_started")
    fi
  done

  RESUME_STOP_REQUESTED=false
  RESUME_WORKER_PIDS=()
  forward_resume_stop() {
    local worker_pid
    stop_count=$((stop_count + 1))
    RESUME_STOP_REQUESTED=true
    if ((stop_count > 1)); then
      printf 'run: repeated stop requested; forwarding the provider hard-stop signal\n' >&2
    fi
    if ((worker_pid_count > 0)); then
      for worker_pid in "${RESUME_WORKER_PIDS[@]}"; do
        kill -TERM "$worker_pid" 2>/dev/null || true
      done
    fi
  }
  trap forward_resume_stop INT TERM HUP

  any_failed=false
  admitted_count=0
  batch_start=0
  while ((batch_start < ${#pending_cells[@]})); do
    batch_end=$((batch_start + provider_cap))
    ((batch_end <= ${#pending_cells[@]})) || batch_end=${#pending_cells[@]}
    RESUME_WORKER_PIDS=()
    worker_pid_count=0
    for ((index = batch_start; index < batch_end; index++)); do
      if [[ "$RESUME_STOP_REQUESTED" == true ]]; then
        break
      fi
      cell=${pending_cells[$index]}
      provider=${pending_providers[$index]}
      output_root="$MODE_DIR/capsules/resume/$RESUME_ATTEMPT/$provider"
      result_map="$MODE_DIR/results/resume-$RESUME_ATTEMPT-$provider.json"
      log_path="$MODE_DIR/results/resume-$RESUME_ATTEMPT-$provider.log"
      [[ ! -e "$result_map" && ! -e "$log_path" ]] || \
        die "resume worker output already exists for attempt $RESUME_ATTEMPT: $provider"
      pending_statuses[$index]="running"
      printf 'run: provider=%s status=running, reused=%s, remaining=%s, running=unavailable, completed=0, failed=unavailable, quarantined=%s, question_progress=%s%% (%s/60)\n' \
        "$provider" "${pending_reused[$index]}" "${pending_remaining[$index]}" \
        "${pending_quarantined[$index]}" \
        "$((pending_reused[$index] * 100 / 60))" "${pending_reused[$index]}"
      run_resume_worker "$cell" "$provider" "$output_root" "$result_map" "$log_path" \
        "${pending_reused[$index]}" "${pending_remaining[$index]}" \
        "${pending_quarantined[$index]}" &
      pid=$!
      RESUME_WORKER_PIDS+=("$pid")
      worker_pid_count=$((worker_pid_count + 1))
      admitted_count=$((index + 1))
      if [[ "$RESUME_STOP_REQUESTED" == true ]]; then
        kill -TERM "$pid" 2>/dev/null || true
        break
      fi
    done
    batch_end=$admitted_count
    for ((index = batch_start; index < batch_end; index++)); do
      pid=${RESUME_WORKER_PIDS[$((index - batch_start))]}
      if wait_for_child "$pid"; then
        pending_statuses[$index]="execution_completed"
      else
        pending_statuses[$index]="failed"
        any_failed=true
      fi
    done
    RESUME_WORKER_PIDS=()
    worker_pid_count=0
    batch_start=$batch_end
    if [[ "$any_failed" == true || "$RESUME_STOP_REQUESTED" == true ]]; then
      break
    fi
  done

  for ((index = 0; index < admitted_count; index++)); do
    cell=${pending_cells[$index]}
    provider=${pending_providers[$index]}
    result_map="$MODE_DIR/results/resume-$RESUME_ATTEMPT-$provider.json"
    log_path="$MODE_DIR/results/resume-$RESUME_ATTEMPT-$provider.log"
    [[ ! -f "$log_path" ]] || sed -n '1,320p' "$log_path"
    if [[ -f "$result_map" && ! -L "$result_map" ]]; then
      capsule_root="$(resume_result_root "$result_map" "$cell")" || {
        any_failed=true
        pending_statuses[$index]="failed"
        capsule_root=""
      }
      if [[ -n "$capsule_root" && -d "$capsule_root" ]]; then
        part_validation="$MODE_DIR/validations/resume-part-$RESUME_ATTEMPT-$provider.json"
        if validate_capsule_result "$capsule_root" "$part_validation"; then
          append_resume_part "$cell" "$capsule_root"
          if [[ "${pending_statuses[$index]}" == "execution_completed" ]]; then
            pending_statuses[$index]="completed"
          fi
        else
          pending_statuses[$index]="failed"
          any_failed=true
        fi
      else
        pending_statuses[$index]="failed"
        any_failed=true
      fi
    else
      pending_statuses[$index]="failed"
      any_failed=true
    fi
  done

  if [[ "$any_failed" == true || "$RESUME_STOP_REQUESTED" == true ]]; then
    for index in "${!pending_cells[@]}"; do
      cell=${pending_cells[$index]}
      provider=${pending_providers[$index]}
      status=${pending_statuses[$index]}
      analysis="$MODE_DIR/results/resume-post-analysis-$RESUME_ATTEMPT-$provider.json"
      log_path="$MODE_DIR/results/resume-post-analysis-$RESUME_ATTEMPT-$provider.log"
      if [[ "$status" == "not_started" ]]; then
        reusable=${pending_reused[$index]}
        remaining=${pending_remaining[$index]}
        quarantined=${pending_quarantined[$index]}
        completed=0
        progress=$((reusable * 100 / 60))
        printf 'run: provider=%s status=not_started, reused=%s, remaining=%s, running=0, completed=0, failed=0, quarantined=%s, question_progress=%s%% (%s/60), resume_state=%s\n' \
          "$provider" "$reusable" "$remaining" "$quarantined" \
          "$progress" "$reusable" "$RESUME_STATE" >&2
      elif run_resume_analysis "$cell" "$provider" "$analysis" "$log_path"; then
        reusable="$(jq -er '.reusable_ingestion_plan_ids | length' "$analysis")"
        remaining="$(jq -er '.remaining_case_manifest_entry_ids | length' "$analysis")"
        quarantined="$(jq -er '.quarantined_ingestion_plan_ids | length' "$analysis")"
        completed=$((reusable - pending_reused[$index]))
        ((completed >= 0)) || completed=0
        progress=$((reusable * 100 / 60))
        if [[ "$status" == "completed" ]]; then
          printf 'run: provider=%s status=completed, reused=%s, remaining=%s, running=0, completed=%s, failed=0, quarantined=%s, question_progress=%s%% (%s/60), resume_state=%s\n' \
            "$provider" "$reusable" "$remaining" "$completed" "$quarantined" \
            "$progress" "$reusable" "$RESUME_STATE" >&2
        else
          printf 'run: provider=%s status=failed, reused=%s, remaining=%s, running=0, completed=%s, failed=unavailable, quarantined=%s, question_progress=%s%% (%s/60), resume_state=%s\n' \
            "$provider" "$reusable" "$remaining" "$completed" "$quarantined" \
            "$progress" "$reusable" "$RESUME_STATE" >&2
        fi
      else
        printf 'run: provider=%s status=%s, progress=unavailable, resume_state=%s\n' \
          "$provider" "$status" "$RESUME_STATE" >&2
      fi
    done
    trap - INT TERM HUP
    die "full-test recovery stopped or failed; preserved parts will be reused by the next --resume"
  fi

  for index in "${!pending_cells[@]}"; do
    compose_resume_cell "${pending_cells[$index]}" "${pending_providers[$index]}"
    if [[ "$RESUME_STOP_REQUESTED" == true ]]; then
      trap - INT TERM HUP
      die "operator stop preserved resume state before report"
    fi
    printf 'run: provider=%s status=completed, reused=%s, remaining=0, running=0, completed=%s, failed=0, quarantined=0, question_progress=100%% (60/60)\n' \
      "${pending_providers[$index]}" "${pending_reused[$index]}" \
      "${pending_remaining[$index]}"
  done

  FULL_ROOTS=()
  FULL_VALIDATIONS=()
  for cell in "${CELLS[@]}"; do
    final_root="$(jq -er --arg cell "$cell" \
      '.cells[] | select(.cell_id == $cell) | .final_capsule_root | select(type == "string")' \
      "$RESUME_STATE")"
    final_validation="$(jq -er --arg cell "$cell" \
      '.cells[] | select(.cell_id == $cell) | .final_validation | select(type == "string")' \
      "$RESUME_STATE")"
    [[ -d "$final_root" && -f "$final_validation" ]] || \
      die "resume did not produce one validated final capsule for $cell"
    FULL_ROOTS+=("$final_root")
    FULL_VALIDATIONS+=("$final_validation")
  done
  trap - INT TERM HUP
  if [[ "$RESUME_STOP_REQUESTED" == true ]]; then
    die "operator stop preserved resume state before report"
  fi
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
  if [[ "$RESUME" == true ]]; then
    run_full_resume
    build_comparison
    exit 0
  fi
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
