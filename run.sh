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

. "$ROOT/provider-services/lib/env.sh"
. "$ROOT/provider-services/lib/host_embedding.sh"
. "$ROOT/provider-services/lib/plan_environment.sh"

MODE="smoke"
MODE_SELECTED=false
DRY_RUN=false
RESUME=false

usage() {
  cat <<'EOF'
Usage: ./run.sh [--smoke_test | --full_test] [--resume] [--dry-run]

--smoke_test  Run one LME-60 question on every provider (default).
--full_test   Run all 60 questions on every provider in parallel.
--resume      Resume only unfinished --full_test questions from canonical progress.
--dry-run     Validate configuration and readiness with zero model/provider calls.
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

report_provider_progress() {
  python3 - "$@" <<'PY'
import json
import sys
from datetime import datetime
from pathlib import Path

output_root = Path(sys.argv[1])
total_questions = int(sys.argv[2])
elapsed_seconds = int(sys.argv[3])
command_active = sys.argv[4] == "true"
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
    if len(specs) == 1:
        capsule_root, spec = specs[0]
        completed = min(
            sum(path.is_file() for path in (capsule_root / "source/cases").glob("*.json")),
            total_questions,
        )
        progress = f"{completed} ({completed}/{total_questions}, {completed * 100 // total_questions}%)"
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
    print(f"provider={provider} status={status}, elapsed={elapsed}, completed_questions={progress}")
PY
}

provider_status_heartbeat() {
  local output_root=$1
  local total_questions=$2
  local started_seconds=$SECONDS
  local timer_pid=""
  trap '[ -z "$timer_pid" ] || kill "$timer_pid" 2>/dev/null || true; exit 0' TERM INT HUP
  while :; do
    sleep "$STATUS_INTERVAL_SECONDS" &
    timer_pid=$!
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
  for provider in hindsight mem0 openviking; do
    printf 'run: provider=%s status=starting, completed_operations=0, completed_questions=0, question_progress=0%% (0/%s)\n' \
      "$provider" "$total_questions"
  done
  provider_status_heartbeat "$output_root" "$total_questions" &
  heartbeat_pid=$!
  command_code=0
  "$@" || command_code=$?
  kill "$heartbeat_pid" 2>/dev/null || true
  wait "$heartbeat_pid" 2>/dev/null || true
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
rm "$LOG_PIPE_DIR/stdout" "$LOG_PIPE_DIR/stderr"
rmdir "$LOG_PIPE_DIR"
printf 'run: log=%s\n' "$LOG_FILE"

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

if [[ "$RESUME" == true ]]; then
  early_resume_pointer="$WORK_DIR/full-test-current"
  [[ -f "$early_resume_pointer" && ! -L "$early_resume_pointer" ]] || \
    die "no resumable full run; start one with ./run.sh --full_test"
  early_execution_label="$(<"$early_resume_pointer")"
  case "$early_execution_label" in
    ""|"."|".."|*[!A-Za-z0-9._-]*) die "full-run resume pointer is invalid" ;;
  esac
  early_mode_dir="$OUTPUTS_ROOT/full-test/$early_execution_label"
  [[ -d "$early_mode_dir" && ! -L "$early_mode_dir" ]] || \
    die "resumable full-run output is missing: $early_mode_dir"
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
plan = load_resolved_plan_for_run(plan_path)
actual_cells = tuple(cell.cell_id for cell in plan.cells)
if actual_cells != expected_cells:
    raise SystemExit(f"unexpected plan cells: {actual_cells!r}")
source_count = validate_lme60_mem0_input_encoding(dataset_source)
print(
    f"input encoding: PASS (60 questions, {source_count} sources, "
    "zero model/provider calls)"
)
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
if [[ "$RESUME" == true ]]; then
  embedding_url="$(read_env_value "$ENV_FILE" OAMB_EMBEDDING_BASE_URL)" || \
    die "cannot load the prechecked embedding endpoint"
  case "$embedding_url" in
    http://host.docker.internal:*)
      embedding_stamp="$(date -u +%Y%m%d-%H%M%S)-$$"
      start_local_embedding "$embedding_url" \
        "$WORK_DIR/embedding-resume-$embedding_stamp.log" \
        "$WORK_DIR/embedding.pid"
      ;;
    *)
      host_embedding_url="$(resolve_host_embedding_base "$embedding_url")" || \
        die "cannot resolve the prechecked embedding endpoint"
      probe_embedding "$host_embedding_url" >/dev/null 2>&1 || \
        die "prechecked external embedding endpoint is unavailable"
      printf 'embedding: PASS (prechecked external endpoint reachable)\n'
      ;;
  esac
  "$ROOT/provider-services/bin/provider-services" up
  "$ROOT/provider-services/bin/provider-services" verify --services
  uv run --locked oamb run "$PLAN" \
    --run-label "simple-resume-rehearsal" \
    --output-root "$early_mode_dir/capsules/simple-resume-rehearsal" \
    --full-progress-root "$early_mode_dir/results" \
    --full-resume-lock "$WORK_DIR/full-test-resume.lock" \
    --full-resume-rehearsal
fi
if [[ "$DRY_RUN" == true ]]; then
  [[ "$MODE" == smoke ]] && QUESTION_COUNT=1 || QUESTION_COUNT=60
  printf 'run: PASS (dry-run, %s, %s questions, 3 providers, zero model/provider calls)\n' \
    "$MODE" "$QUESTION_COUNT"
  exit 0
fi

if [[ "$MODE" == "smoke" ]]; then
  MODE_ROOT="$OUTPUTS_ROOT/smoke-test"
else
  MODE_ROOT="$OUTPUTS_ROOT/full-test"
fi
FULL_RESUME_POINTER="$WORK_DIR/full-test-current"
if [[ "$RESUME" == true ]]; then
  [[ -f "$FULL_RESUME_POINTER" && ! -L "$FULL_RESUME_POINTER" ]] || \
    die "no resumable full run; start one with ./run.sh --full_test"
  EXECUTION_LABEL="$(<"$FULL_RESUME_POINTER")"
  case "$EXECUTION_LABEL" in
    ""|"."|".."|*[!A-Za-z0-9._-]*) die "full-run resume pointer is invalid" ;;
  esac
  MODE_DIR="$MODE_ROOT/$EXECUTION_LABEL"
  [[ -d "$MODE_DIR" && ! -L "$MODE_DIR" ]] || \
    die "resumable full-run output is missing: $MODE_DIR"
else
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
  uv run --locked python - "$PLAN" "$MODE_DIR/results" <<'PY'
import sys
from pathlib import Path

from oamb.config.doctor import load_resolved_plan_for_run
from oamb.runtime.full_progress import (
    canonical_full_progress_path,
    empty_full_progress,
    initialize_full_progress,
)
from oamb.workloads.longmemeval import build_longmemeval_bundle

plan = load_resolved_plan_for_run(Path(sys.argv[1]))
progress_root = Path(sys.argv[2])
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
    raise SystemExit("fresh full progress does not match the frozen LME-60 manifest")
paths = tuple(
    canonical_full_progress_path(progress_root, cell.provider_id) for cell in plan.cells
)
if any(path.exists() or path.is_symlink() for path in paths):
    raise SystemExit("fresh full progress target already exists")
for cell, path in zip(plan.cells, paths, strict=True):
    initialize_full_progress(
        path,
        empty_full_progress(
            resolved_plan_hash=plan.resolved_plan_hash,
            cell_id=cell.cell_id,
            provider_id=cell.provider_id,
            workload_id=cell.workload_id,
            case_manifest_hash=cell.case_manifest_hash,
            ordered_question_ids=ordered_question_ids,
        ),
    )
    print(f"full progress: initialized cell={cell.cell_id} reused=0 remaining=60")
PY
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
    uv run --locked oamb compare "$PLAN" \
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
    uv run --locked oamb compare "$PLAN" \
      --full-progress-root "$MODE_DIR/results" \
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
  printf 'run: status=resuming-from-canonical-progress\n'
  uv run --locked oamb run "$PLAN" \
    --run-label "$resume_label" \
    --output-root "$resume_output" \
    --full-progress-root "$MODE_DIR/results" \
    --full-resume-lock "$WORK_DIR/full-test-resume.lock"
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
    --full-progress-root "$MODE_DIR/results"
    --full-resume-lock "$WORK_DIR/full-test-resume.lock"
    --full-resume-pointer "$FULL_RESUME_POINTER"
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
