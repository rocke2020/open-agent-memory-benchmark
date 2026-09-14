#!/bin/bash

set -euo pipefail

readonly ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly ENV_FILE="$ROOT/.env"
readonly OUTPUTS_ROOT="$ROOT/outputs"
readonly PRECHECK_ROOT="$OUTPUTS_ROOT/tmp/precheck"
readonly STATE_FILE="$OUTPUTS_ROOT/tmp/quick-start-current.json"
readonly CELLS=("hindsight-lme60" "mem0-lme60" "openviking-lme60")
readonly STATUS_INTERVAL_SECONDS=30
readonly EMBEDDING_STARTUP_ATTEMPTS="${OAMB_EMBEDDING_STARTUP_ATTEMPTS:-180}"
readonly OAMB_RUN_SH_PROCESS_ID=$$
export OAMB_RUN_SH_PROCESS_ID
ACTIVE_COMMAND_PID=""
ACTIVE_COMMAND_WATCHER_PID=""
ACTIVE_HEARTBEAT_PID=""
STOP_REQUESTED=false
STOP_EXIT_CODE=0

. "$ROOT/provider-services/lib/env.sh"
. "$ROOT/provider-services/lib/host_embedding.sh"
. "$ROOT/provider-services/lib/plan_environment.sh"

MODE="smoke"
MODE_SELECTED=false
DRY_RUN=false
RESUME=false
GENERATE_REPORT=false
RESULT_DIR=""

usage() {
  cat <<'EOF'
Usage: ./run.sh [--smoke_test | --full_test] [--resume] [--dry-run]
       ./run.sh --generate-report --result-dir=DIR

--smoke_test  Run one LME-60 question on every provider (default).
--full_test   Run all 60 questions on every provider in parallel.
--resume      Skip saved --full_test question IDs and run only missing questions.
--dry-run     Validate configuration and readiness with zero model/provider calls.
--generate-report
              Build a report from saved results and cached concise LLM analysis.
--result-dir  Directory containing one saved snapshot per provider.
EOF
}

die() {
  printf 'run: FAIL: %s\n' "$*" >&2
  exit 1
}

finish_logging() {
  local run_exit_code=$1
  local log_exit_code=0
  exec 1>&3 2>&4 3>&- 4>&-
  wait "$LOG_STDOUT_PID" || log_exit_code=$?
  wait "$LOG_STDERR_PID" || log_exit_code=$?
  if ((log_exit_code != 0)); then
    printf 'run: FAIL: could not finish writing log: %s\n' "$LOG_FILE" >&2
    ((run_exit_code != 0)) || run_exit_code=$log_exit_code
  fi
  exit "$run_exit_code"
}

stop_owned_process() {
  local process_id=$1
  kill -KILL -- "-$process_id" 2>/dev/null || true
}

request_stop() {
  local exit_code=$1
  STOP_REQUESTED=true
  STOP_EXIT_CODE=$exit_code
  if [[ -n "$ACTIVE_HEARTBEAT_PID" ]]; then
    kill -TERM "$ACTIVE_HEARTBEAT_PID" 2>/dev/null || true
    wait "$ACTIVE_HEARTBEAT_PID" 2>/dev/null || true
    ACTIVE_HEARTBEAT_PID=""
  fi
  if [[ -n "$ACTIVE_COMMAND_PID" ]]; then
    stop_owned_process "$ACTIVE_COMMAND_PID"
    wait "$ACTIVE_COMMAND_PID" 2>/dev/null || true
    ACTIVE_COMMAND_PID=""
  fi
  if [[ -n "$ACTIVE_COMMAND_WATCHER_PID" ]]; then
    kill "$ACTIVE_COMMAND_WATCHER_PID" 2>/dev/null || true
    wait "$ACTIVE_COMMAND_WATCHER_PID" 2>/dev/null || true
    ACTIVE_COMMAND_WATCHER_PID=""
  fi
  if [[ "$GENERATE_REPORT" == true ]]; then
    exit "$exit_code"
  fi
  "$ROOT/provider-services/bin/provider-services" stop || true
  exit "$exit_code"
}

run_owned_command() {
  local command_pid watcher_pid command_code=0
  set -m
  (trap - INT TERM HUP; exec "$@") &
  command_pid=$!
  set +m
  ACTIVE_COMMAND_PID=$command_pid
  (
    trap 'exit 0' INT TERM HUP
    while kill -0 "$OAMB_RUN_SH_PROCESS_ID" 2>/dev/null && \
      kill -0 "$command_pid" 2>/dev/null; do
      sleep 1
    done
    if ! kill -0 "$OAMB_RUN_SH_PROCESS_ID" 2>/dev/null; then
      stop_owned_process "$command_pid"
      if [[ "$GENERATE_REPORT" == false ]]; then
        "$ROOT/provider-services/bin/provider-services" stop || true
      fi
    fi
  ) &
  watcher_pid=$!
  ACTIVE_COMMAND_WATCHER_PID=$watcher_pid
  wait "$command_pid" || command_code=$?
  if [[ "$STOP_REQUESTED" == true ]]; then
    wait "$command_pid" 2>/dev/null || true
    if [[ "$GENERATE_REPORT" == false ]]; then
      if ! "$ROOT/provider-services/bin/provider-services" stop; then
        printf 'run: provider services could not be stopped\n' >&2
      fi
    fi
    command_code=$STOP_EXIT_CODE
  fi
  ACTIVE_COMMAND_PID=""
  kill "$watcher_pid" 2>/dev/null || true
  wait "$watcher_pid" 2>/dev/null || true
  ACTIVE_COMMAND_WATCHER_PID=""
  return "$command_code"
}

report_provider_progress() {
  local results_root=""
  [[ "$MODE" != "full" ]] || results_root="$MODE_DIR/results"
  python3 - "$@" "$results_root" <<'PY'
import json
import sys
from datetime import datetime
from pathlib import Path

output_root = Path(sys.argv[1])
total_questions = int(sys.argv[2])
elapsed_seconds = int(sys.argv[3])
command_active = sys.argv[4] == "true"
results_root = Path(sys.argv[5]) if sys.argv[5] else None
providers = ("hindsight", "mem0", "openviking")
provider_specs = {provider: [] for provider in providers}
terminal_statuses = {
    "finalized": "execution-completed",
    "aborted": "aborted",
    "infrastructure_blocked": "infrastructure_blocked",
}


for path in output_root.glob("*/source/specs/run-spec.json"):
    try:
        spec = json.loads(path.read_text(encoding="utf-8"))
        provider = spec["memory_system_id"]
        if provider in provider_specs:
            provider_specs[provider].append((path.parents[2], spec))
    except (OSError, ValueError, KeyError, TypeError):
        continue

for provider, specs in provider_specs.items():
    status = elapsed = progress = "unavailable"
    completed = None
    if results_root is not None:
        try:
            results = json.loads((results_root / f"{provider}.json").read_text(encoding="utf-8"))
            if not isinstance(results, dict) or len(results) > total_questions:
                raise ValueError("invalid question results")
            completed = len(results)
        except (OSError, ValueError):
            pass
    if len(specs) == 1:
        capsule_root, spec = specs[0]
        if results_root is None:
            completed = min(
                sum(path.is_file() for path in (capsule_root / "source/cases").glob("*.json")),
                total_questions,
            )
        records = list((capsule_root / "source/run").glob("*.json"))
        try:
            run_id = spec["run_id"]
            if not isinstance(run_id, str) or not run_id:
                raise ValueError("missing run identity")
            if len(records) == 1:
                record = json.loads(records[0].read_text(encoding="utf-8"))
                if (
                    record["schema_name"] != "run_record"
                    or record["schema_version"] != 1
                    or record["run_id"] != run_id
                    or records[0].stem != run_id
                ):
                    raise ValueError("terminal run identity differs")
                terminal_status = terminal_statuses[record["state"]]
                started = datetime.fromisoformat(record["started_at"])
                ended = datetime.fromisoformat(record["ended_at"])
                if started.tzinfo is None or ended.tzinfo is None or ended < started:
                    raise ValueError("invalid terminal run timestamps")
                status = terminal_status
                elapsed = f"{int((ended - started).total_seconds())}s"
            elif not records and command_active and not (capsule_root / "capsule-manifest.json").exists():
                status = "running"
                elapsed = f"{elapsed_seconds}s"
        except (OSError, ValueError, KeyError, TypeError):
            pass
    elif not specs and completed is not None:
        if completed == total_questions:
            status = "execution-completed"
        elif command_active:
            status = "starting"
            elapsed = f"{elapsed_seconds}s"
    if completed is not None:
        progress = f"{completed} ({completed}/{total_questions}, {completed * 100 // total_questions}%)"
    print(f"provider={provider} status={status}, elapsed={elapsed}, completed_questions={progress}")
PY
}

provider_status_heartbeat() {
  local output_root=$1
  local total_questions=$2
  local owner_pid=$3
  local started_seconds=$SECONDS
  local timer_pid=""
  trap '[ -z "$timer_pid" ] || kill "$timer_pid" 2>/dev/null || true; exit 0' TERM INT HUP
  while :; do
    sleep "$STATUS_INTERVAL_SECONDS" &
    timer_pid=$!
    while kill -0 "$timer_pid" 2>/dev/null; do
      sleep 1
      if ! kill -0 "$owner_pid" 2>/dev/null; then
        kill "$timer_pid" 2>/dev/null || true
        wait "$timer_pid" 2>/dev/null || true
        exit 0
      fi
    done
    wait "$timer_pid" || exit 0
    report_provider_progress "$output_root" "$total_questions" \
      "$((SECONDS - started_seconds))" true
  done
}

run_provider_cells_with_status() {
  local output_root=$1
  local context=$2
  local total_questions=$3
  shift 3
  local provider heartbeat_pid command_code
  local started_seconds=$SECONDS
  printf 'run: providers=hindsight,mem0,openviking status=starting (%s)\n' "$context"
  if [[ "$MODE" == "full" ]]; then
    report_provider_progress "$output_root" "$total_questions" 0 true
  else
    for provider in hindsight mem0 openviking; do
      printf 'run: provider=%s status=starting, completed_operations=0, completed_questions=0, question_progress=0%% (0/%s)\n' \
        "$provider" "$total_questions"
    done
  fi
  provider_status_heartbeat "$output_root" "$total_questions" "$OAMB_RUN_SH_PROCESS_ID" &
  heartbeat_pid=$!
  ACTIVE_HEARTBEAT_PID=$heartbeat_pid
  command_code=0
  run_owned_command "$@" || command_code=$?
  kill "$heartbeat_pid" 2>/dev/null || true
  wait "$heartbeat_pid" 2>/dev/null || true
  ACTIVE_HEARTBEAT_PID=""
  report_provider_progress "$output_root" "$total_questions" \
    "$((SECONDS - started_seconds))" false
  if ((command_code != 0)); then
    printf 'run: provider execution command status=failed\n' >&2
    return "$command_code"
  fi
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
      [[ "$MODE_SELECTED" == false && "$GENERATE_REPORT" == false ]] || \
        die "choose only one run or report mode"
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
    --generate-report)
      [[ "$GENERATE_REPORT" == false && "$MODE_SELECTED" == false ]] || \
        die "choose only one run or report mode"
      GENERATE_REPORT=true
      MODE="report"
      shift
      ;;
    --result-dir=*)
      [[ -z "$RESULT_DIR" ]] || die "--result-dir may be supplied only once"
      RESULT_DIR="${1#--result-dir=}"
      [[ -n "$RESULT_DIR" ]] || die "--result-dir requires a directory"
      shift
      ;;
    --result-dir)
      [[ -z "$RESULT_DIR" ]] || die "--result-dir may be supplied only once"
      [[ $# -ge 2 && -n "$2" ]] || die "--result-dir requires a directory"
      RESULT_DIR=$2
      shift 2
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

command -v tee >/dev/null 2>&1 || die "required command not found: tee"
mkdir -p "$OUTPUTS_ROOT/tmp" || die "cannot create log directory"
LOG_FILE="$OUTPUTS_ROOT/tmp/run-$MODE-$(date -u +%Y%m%d-%H%M%S)-$$.log"
(umask 077; set -C; : > "$LOG_FILE") || die "cannot create log: $LOG_FILE"
LOG_PIPE_DIR="$(mktemp -d "$OUTPUTS_ROOT/tmp/.run-log.XXXXXX")" || die "cannot create log pipes"
mkfifo "$LOG_PIPE_DIR/stdout" "$LOG_PIPE_DIR/stderr"
# Keep stdout/stderr separate on the terminal; append both to one private log.
# Ordinary background children can be waited for on macOS Bash 3.2.
exec 3>&1 4>&2
(trap '' INT TERM HUP; exec tee -a "$LOG_FILE" < "$LOG_PIPE_DIR/stdout" >&3) &
LOG_STDOUT_PID=$!
(trap '' INT TERM HUP; exec tee -a "$LOG_FILE" < "$LOG_PIPE_DIR/stderr" >&4) &
LOG_STDERR_PID=$!
exec > "$LOG_PIPE_DIR/stdout" 2> "$LOG_PIPE_DIR/stderr"
trap 'finish_logging "$?"' EXIT
trap 'request_stop 130' INT
trap 'request_stop 143' TERM
trap 'request_stop 129' HUP
rm "$LOG_PIPE_DIR/stdout" "$LOG_PIPE_DIR/stderr"
rmdir "$LOG_PIPE_DIR"
printf 'run: log=%s\n' "$LOG_FILE"

if [[ "$GENERATE_REPORT" == true ]]; then
  [[ -n "$RESULT_DIR" ]] || die "--generate-report requires --result-dir"
  [[ "$RESUME" == false && "$DRY_RUN" == false ]] || \
    die "--generate-report cannot be combined with --resume or --dry-run"
  case "$RESULT_DIR" in
    /*) ;;
    *) RESULT_DIR="$ROOT/$RESULT_DIR" ;;
  esac
  [[ -d "$RESULT_DIR" && ! -L "$RESULT_DIR" ]] || \
    die "saved result directory is missing or symlinked: $RESULT_DIR"
  [[ -f "$ENV_FILE" && ! -L "$ENV_FILE" ]] || \
    die "report analysis model environment is missing or symlinked: $ENV_FILE"
  command -v jq >/dev/null 2>&1 || die "required command not found: jq"
  command -v uv >/dev/null 2>&1 || die "required command not found: uv"
  REPORT_OUTPUT="$RESULT_DIR/comparison"
  if [[ -e "$REPORT_OUTPUT" || -L "$REPORT_OUTPUT" ]]; then
    REPORT_OUTPUT="$RESULT_DIR/comparison-$(date -u +%Y%m%d-%H%M%S)-$$"
  fi
  cd "$ROOT"
  printf 'run: status=building-saved-report\n'
  run_owned_command uv run --locked oamb report saved-results "$RESULT_DIR" \
    --output-root "$REPORT_OUTPUT" \
    --analysis-model-env "$ENV_FILE" \
    --analysis-cache-root "$RESULT_DIR/report-analysis-cache"
  jq -e '.coverage == {
    "cell_count": 3,
    "unique_case_count": 60,
    "provider_specific_result_count": 180
  }' "$REPORT_OUTPUT/report.json" >/dev/null || die "saved report coverage is invalid"
  [[ -s "$REPORT_OUTPUT/report.html" ]] || die "saved report HTML is missing or empty"
  [[ -s "$REPORT_OUTPUT/report-analysis.json" ]] || \
    die "saved report concise analysis is missing or empty"
  open_report "$REPORT_OUTPUT/report.html"
  printf 'run: PASS (saved report, 60 questions, 180 provider results)\n'
  printf 'report: %s\n' "$REPORT_OUTPUT/report.html"
  exit 0
fi

[[ -z "$RESULT_DIR" ]] || die "--result-dir requires --generate-report"

if [[ "$RESUME" == true && "$MODE" != "full" ]]; then
  die "--resume requires --full_test"
fi
if [[ "$RESUME" == true && "$DRY_RUN" == true ]]; then
  die "--resume cannot be combined with --dry-run"
fi

command -v jq >/dev/null 2>&1 || die "required command not found: jq"
command -v uv >/dev/null 2>&1 || die "required command not found: uv"
[[ -f "$STATE_FILE" ]] || die "run ./precheck.sh first"

RUN_LABEL="$(jq -er '.run_label | select(type == "string" and length > 0)' "$STATE_FILE")"
WORK_DIR="$(jq -er '.work_dir | select(type == "string" and length > 0)' "$STATE_FILE")"
PRECHECK_PLAN="$(jq -er '.resolved_plan | select(type == "string" and length > 0)' "$STATE_FILE")"
QUESTION_ID="$(jq -er '.question_id | select(type == "string" and length > 0)' "$STATE_FILE")"
STATE_EMBEDDING_OWNERSHIP=""
if [[ "$RESUME" == false ]]; then
  STATE_EMBEDDING_OWNERSHIP="$(jq -er '
    .embedding_ownership |
    select(. == "embedding_local_fallback" or . == "external")
  ' "$STATE_FILE")" || \
    die "precheck state has invalid embedding ownership"
fi
case "$RUN_LABEL" in
  ""|"."|".."|*[!A-Za-z0-9._-]*) die "precheck state has invalid run label" ;;
esac
[[ "$WORK_DIR" == "$PRECHECK_ROOT/$RUN_LABEL" ]] || \
  die "precheck state work directory is outside outputs/tmp/precheck"
[[ "$PRECHECK_PLAN" == "$WORK_DIR/plan/resolved-plan.json" ]] || \
  die "precheck state resolved plan is outside its work directory"
if [[ "$MODE" == "smoke" ]]; then
  MODE_ROOT="$OUTPUTS_ROOT/smoke-test"
else
  MODE_ROOT="$OUTPUTS_ROOT/full-test"
fi
FULL_RUN_SELECTOR="$OUTPUTS_ROOT/tmp/full-test-current"
if [[ "$RESUME" == true ]]; then
  [[ -f "$FULL_RUN_SELECTOR" && ! -L "$FULL_RUN_SELECTOR" ]] || \
    die "no resumable full run; start one with ./run.sh --full_test"
  EXECUTION_LABEL="$(<"$FULL_RUN_SELECTOR")"
  case "$EXECUTION_LABEL" in
    ""|"."|".."|*[!A-Za-z0-9._-]*) die "full-run selector is invalid" ;;
  esac
  MODE_DIR="$MODE_ROOT/$EXECUTION_LABEL"
  [[ -d "$MODE_DIR" && ! -L "$MODE_DIR" ]] || \
    die "resumable full-run output is missing: $MODE_DIR"
  PLAN="$MODE_DIR/resolved-plan.json"
else
  PLAN="$PRECHECK_PLAN"
fi
[[ -f "$PLAN" && ! -L "$PLAN" ]] || die "resolved plan is missing"
DATASET_PATH="$(jq -er '.dataset.path | select(type == "string" and length > 0)' "$PLAN")"
case "$DATASET_PATH" in
  /*) DATASET_SOURCE="$DATASET_PATH" ;;
  *) DATASET_SOURCE="$ROOT/$DATASET_PATH" ;;
esac
[[ -f "$DATASET_SOURCE" && ! -L "$DATASET_SOURCE" ]] || die "stored-plan dataset is missing"
RESUME_EMBEDDING_ENDPOINT=""
if [[ "$RESUME" == true ]]; then
  RESUME_EMBEDDING_ENDPOINT="$(read_optional_env_value \
    "$ENV_FILE" OAMB_EMBEDDING_BASE_URL)" || \
    die "cannot load the resume embedding endpoint"
  if [[ -z "$RESUME_EMBEDDING_ENDPOINT" || "$RESUME_EMBEDDING_ENDPOINT" == change-me* ]]; then
    RESUME_EMBEDDING_ENDPOINT="$(jq -er '
      .embedding_endpoint.effective_endpoint |
      select(type == "string" and length > 0)
    ' "$PLAN" 2>/dev/null)" || \
      die "resume requires a reachable embedding endpoint"
  fi
  export OAMB_RESUME_EMBEDDING_ENDPOINT="$RESUME_EMBEDDING_ENDPOINT"
  export OAMB_EMBEDDING_BASE_URL="$RESUME_EMBEDDING_ENDPOINT"
  export OAMB_EMBEDDING_OWNERSHIP=external
fi
load_plan_model_environment "$PLAN" "$ENV_FILE" || \
  die "cannot load model configuration from resolved plan and current .env"
if [[ "$RESUME" == true ]]; then
  export OAMB_EMBEDDING_BASE_URL="$RESUME_EMBEDDING_ENDPOINT"
  export OAMB_EMBEDDING_OWNERSHIP=external
else
  [[ "$STATE_EMBEDDING_OWNERSHIP" == "$OAMB_EMBEDDING_OWNERSHIP" ]] || \
    die "precheck state and resolved plan embedding ownership differ"
fi

"$ROOT/provider-services/bin/provider-services" doctor
uv run --locked python - "$PLAN" "$ENV_FILE" "$ROOT/provider-services/.runtime" \
  "$DATASET_SOURCE" "${CELLS[@]}" <<'PY'
import os
import sys
from pathlib import Path

from oamb.config.doctor import load_resolved_plan_for_run
from oamb.live import (
    load_live_environment,
    validate_live_readiness_receipt,
    validate_lme60_mem0_input_encoding,
)

plan_path = Path(sys.argv[1])
env_file = Path(sys.argv[2])
runtime = Path(sys.argv[3])
dataset_source = Path(sys.argv[4])
expected_cells = tuple(sys.argv[5:])
resume_embedding_endpoint = os.environ.get("OAMB_RESUME_EMBEDDING_ENDPOINT")
plan = load_resolved_plan_for_run(
    plan_path,
    resume_embedding_endpoint=resume_embedding_endpoint,
)
actual_cells = tuple(cell.cell_id for cell in plan.cells)
if actual_cells != expected_cells:
    raise SystemExit(f"unexpected plan cells: {actual_cells!r}")
source_count = validate_lme60_mem0_input_encoding(dataset_source)
print(
    f"input encoding: PASS (60 questions, {source_count} sources, "
    "zero model/provider calls)"
)
environment = load_live_environment(
    plan=plan,
    provider_env_path=env_file,
    model_env_path=env_file,
    provider_runtime_directory=runtime,
    base_environment=os.environ,
)
if resume_embedding_endpoint is None:
    validate_live_readiness_receipt(
        plan=plan,
        provider_runtime_directory=runtime,
        environment=environment,
    )
PY
if [[ "$RESUME" == true ]]; then
  embedding_url=$OAMB_EMBEDDING_BASE_URL
  embedding_api_key="$(read_optional_env_value "$ENV_FILE" OAMB_EMBEDDING_API_KEY)" || \
    die "cannot load the prechecked embedding API key"
  embedding_request="$ROOT/provider-services/.runtime/embedding-ready-request.json"
  [[ -f "$embedding_request" && ! -L "$embedding_request" ]] || \
    die "last-run embedding configuration is missing"
  embedding_dimension="$(jq -er '
    .dimensions |
    select(type == "number" and . == floor and . > 0)
  ' "$embedding_request")" || \
    die "last-run embedding dimension is invalid"
  host_embedding_url="$(resolve_host_embedding_base "$embedding_url")" || \
    die "cannot resolve the resume embedding endpoint"
  probe_embedding \
    "$host_embedding_url" "$embedding_api_key" "$embedding_dimension" >/dev/null 2>&1 || \
    die "resume embedding must return one finite $embedding_dimension-dimensional vector"
  printf 'embedding: PASS (finite %s-dimensional vector)\n' "$embedding_dimension"
  "$ROOT/provider-services/bin/provider-services" up
  "$ROOT/provider-services/bin/provider-services" verify --services
fi
if [[ "$DRY_RUN" == true ]]; then
  [[ "$MODE" == smoke ]] && QUESTION_COUNT=1 || QUESTION_COUNT=60
  printf 'run: PASS (dry-run, %s, %s questions, 3 providers, zero model/provider calls)\n' \
    "$MODE" "$QUESTION_COUNT"
  exit 0
fi

if [[ "$RESUME" == false ]]; then
  EXECUTION_LABEL="$RUN_LABEL"
  if [[ -e "$MODE_ROOT/$EXECUTION_LABEL" ]]; then
    EXECUTION_LABEL="$RUN_LABEL-$(date -u +%Y%m%d-%H%M%S)-$$"
  fi
  MODE_DIR="$MODE_ROOT/$EXECUTION_LABEL"
  mkdir -p "$MODE_ROOT"
  mkdir "$MODE_DIR" || die "cannot create fresh run output: $MODE_DIR"
fi
mkdir -p "$MODE_DIR/results" "$MODE_DIR/validations" "$MODE_DIR/capsules"
if [[ "$MODE" == "full" && "$RESUME" == false ]]; then
  cp "$PLAN" "$MODE_DIR/resolved-plan.json" || die "cannot save the resolved plan"
  PLAN="$MODE_DIR/resolved-plan.json"
  uv run --locked python - "$PLAN" "$MODE_DIR/results" <<'PY'
import sys
from pathlib import Path

from oamb.config.doctor import load_resolved_plan_for_run
from oamb.runtime.question_results import (
    initialize_question_results,
    provider_result_path,
)
from oamb.workloads.longmemeval import build_longmemeval_bundle

plan = load_resolved_plan_for_run(Path(sys.argv[1]))
results_root = Path(sys.argv[2])
dataset_path = Path(plan.dataset.path)
if not dataset_path.is_absolute():
    dataset_path = Path.cwd() / dataset_path
manifest = build_longmemeval_bundle(dataset_path, plan.dataset.selection).case_manifest
ordered_question_ids = tuple(case.raw_question_id for case in manifest.cases)
if (
    plan.dataset.selection != "lme60"
    or len(ordered_question_ids) != 60
    or manifest.manifest_hash != plan.dataset.case_manifest_hash
):
    raise SystemExit("fresh full results do not match the frozen LME-60 manifest")
paths = tuple(
    provider_result_path(results_root, cell.provider_id) for cell in plan.cells
)
if any(path.exists() or path.is_symlink() for path in paths):
    raise SystemExit("fresh full result target already exists")
for cell, path in zip(plan.cells, paths, strict=True):
    initialize_question_results(path)
    print(f"question results: initialized cell={cell.cell_id} reused=0 remaining=60")
PY
  selector_tmp="$OUTPUTS_ROOT/tmp/.full-test-current.$$"
  (umask 077; printf '%s\n' "$EXECUTION_LABEL" > "$selector_tmp") || \
    die "cannot write full-run selector"
  mv "$selector_tmp" "$FULL_RUN_SELECTOR" || die "cannot publish full-run selector"
fi

VALIDATION_PATH=""

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
  if ! run_owned_command uv run --locked oamb capsule validate \
    "$capsule_root" --output "$validation_path"; then
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

build_comparison() {
  local expected_cases=60
  local expected_results=180
  local roots=()
  local validations=()
  local comparison="$MODE_DIR/comparison"
  if [[ "$MODE" == "smoke" ]]; then
    expected_cases=1
    expected_results=3
    roots=("${RUN_ROOTS[@]}")
    validations=("${RUN_VALIDATIONS[@]}")
  fi
  if [[ -e "$comparison" ]]; then
    comparison="$MODE_DIR/comparison-retry-$(date -u +%Y%m%d-%H%M%S)-$$"
  fi
  printf 'run: status=building-comparison\n'
  if [[ "$MODE" == "smoke" ]]; then
    run_owned_command uv run --locked oamb compare "$PLAN" \
      --cell-root "${CELLS[0]}=${roots[0]}" \
      --cell-root "${CELLS[1]}=${roots[1]}" \
      --cell-root "${CELLS[2]}=${roots[2]}" \
      --validation "${CELLS[0]}=${validations[0]}" \
      --validation "${CELLS[1]}=${validations[1]}" \
      --validation "${CELLS[2]}=${validations[2]}" \
      --dataset-source "$DATASET_SOURCE" \
      --analysis-model-env "$ROOT/.env" \
      --analysis-cache-root "$MODE_DIR/report-analysis-cache" \
      --output-root "$comparison" \
      --diagnostic
  else
    run_owned_command uv run --locked oamb compare "$PLAN" \
      --results-root "$MODE_DIR/results" \
      --dataset-source "$DATASET_SOURCE" \
      --analysis-model-env "$ROOT/.env" \
      --analysis-cache-root "$MODE_DIR/report-analysis-cache" \
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
  printf 'run: status=comparison-completed\n'
  printf 'run: status=opening-report\n'
  open_report "$comparison/report.html"
  printf 'run: status=report-opened\n'
  printf 'run: PASS (%s, %s questions, %s provider results)\n' \
    "$MODE" "$expected_cases" "$expected_results"
  printf 'report: %s\n' "$comparison/report.html"
}

run_simple_resume() {
  local resume_label="simple-resume-$(date -u +%Y%m%d-%H%M%S)-$$"
  local resume_output="$MODE_DIR/capsules/resume/$resume_label"
  mkdir -p "$MODE_DIR/capsules/resume"
  printf 'run: status=resuming-from-saved-question-results\n'
  run_provider_cells_with_status "$resume_output" "full resume, 60 questions each" 60 \
    uv run --locked oamb run "$PLAN" \
    --run-label "$resume_label" \
    --output-root "$resume_output" \
    --results-root "$MODE_DIR/results" \
    --resume-concurrency-config "$ROOT/configs/benchmark.yml"
  printf 'run: status=resume-completed\n'
}

cd "$ROOT"
RUN_ROOTS=()
RUN_VALIDATIONS=()
if [[ "$MODE" == "full" && "$RESUME" == true ]]; then
  run_simple_resume
  build_comparison
  exit 0
fi

run_result="$MODE_DIR/results/$MODE.json"
run_output="$MODE_DIR/capsules/$MODE"
run_arguments=(
  uv run --locked oamb run "$PLAN"
  --run-label "$EXECUTION_LABEL-$MODE-$$"
  --output-root "$run_output"
  --result-map "$run_result"
)
if [[ "$MODE" == "smoke" ]]; then
  run_arguments+=(
    --cell "${CELLS[0]}"
    --cell "${CELLS[1]}"
    --cell "${CELLS[2]}"
    --question "$QUESTION_ID"
  )
  question_count=1
  run_context="smoke, 1 question each"
else
  question_count=60
  run_context="full, 60 questions each"
  run_arguments+=(
    --results-root "$MODE_DIR/results"
  )
fi

run_provider_cells_with_status "$run_output" "$run_context" "$question_count" \
  "${run_arguments[@]}"

for cell in "${CELLS[@]}"; do
  provider="${cell%-lme60}"
  capsule_root="$(jq -er --arg cell "$cell" '.capsule_roots[$cell]' "$run_result")"
  validation="$MODE_DIR/validations/$MODE-$provider.json"
  [[ -d "$capsule_root" ]] || die "$MODE capsule is missing for $cell"
  printf 'run: provider=%s status=validating\n' "$provider"
  validate_capsule "$capsule_root" "$validation"
  printf 'run: provider=%s status=completed\n' "$provider"
  if [[ "$MODE" == "smoke" ]]; then
    RUN_ROOTS+=("$capsule_root")
    RUN_VALIDATIONS+=("$VALIDATION_PATH")
  fi
done

build_comparison
