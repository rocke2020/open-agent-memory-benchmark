#!/bin/bash

set -euo pipefail

readonly ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly ENV_FILE="$ROOT/.env"
readonly RUNTIME_DIR="$ROOT/provider-services/.runtime"
readonly STATE_FILE="$ROOT/.local-demo/quick-start-current.json"
readonly QUESTION_ID="72e3ee87"
readonly DEFAULT_EMBEDDING_URL="http://host.docker.internal:18000/v1"
readonly DEFAULT_HINDSIGHT_PORT="18888"
readonly DEFAULT_MEM0_PORT="18889"
readonly DEFAULT_MEM0_INSPECTOR_PORT="16333"
readonly DEFAULT_OPENVIKING_PORT="19330"
readonly DEFAULT_OPENVIKING_ACCOUNT_ID="oamb-benchmark"
readonly DEFAULT_OPENVIKING_ADMIN_USER_ID="oamb-admin"
readonly EMBEDDING_STARTUP_ATTEMPTS="${OAMB_EMBEDDING_STARTUP_ATTEMPTS:-180}"

. "$ROOT/provider-services/lib/host_embedding.sh"
. "$ROOT/provider-services/lib/plan_environment.sh"

START_LOCAL_EMBEDDING=true
EMBEDDING_API_URL=""

usage() {
  cat <<'EOF'
Usage: ./precheck.sh [--embedding-api-url URL | --no-start-embedding]

Prepare inputs, configure local files, start and verify providers, and freeze
the exact LME-60 plan. Local embedding startup is enabled by default.
EOF
}

die() {
  printf 'precheck: FAIL: %s\n' "$*" >&2
  exit 1
}

need_command() {
  command -v "$1" >/dev/null 2>&1 || die "required command not found: $1"
}

read_env_value() {
  local file=$1
  local key=$2
  awk -v key="$key" '
    index($0, key "=") == 1 {
      if (found) exit 3
      print substr($0, length(key) + 2)
      found = 1
    }
    END { if (!found) exit 2 }
  ' "$file"
}

set_env_value() {
  local file=$1
  local key=$2
  local value=$3
  OAMB_ENV_FILE="$file" OAMB_ENV_KEY="$key" OAMB_ENV_VALUE="$value" python3 - <<'PY'
import os
from pathlib import Path

path = Path(os.environ["OAMB_ENV_FILE"])
key = os.environ["OAMB_ENV_KEY"]
value = os.environ["OAMB_ENV_VALUE"]
if "\n" in value or "\r" in value:
    raise SystemExit(f"{key} must fit on one dotenv line")
lines = path.read_text(encoding="utf-8").splitlines()
matches = [index for index, line in enumerate(lines) if line.startswith(f"{key}=")]
if len(matches) > 1:
    raise SystemExit(f"duplicate dotenv key: {key}")
replacement = f"{key}={value}"
if matches:
    lines[matches[0]] = replacement
else:
    lines.append(replacement)
temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
temporary.write_text("\n".join(lines) + "\n", encoding="utf-8")
temporary.chmod(0o600)
temporary.replace(path)
PY
}

random_secret() {
  python3 -c 'import secrets; print(secrets.token_hex(32))'
}

remove_plan_owned_environment_values() {
  OAMB_ENV_FILE="$ENV_FILE" python3 - <<'PY'
import os
from pathlib import Path

path = Path(os.environ["OAMB_ENV_FILE"])
plan_owned = {
    "OMBA_ANSWER_LLM",
    "OMBA_ANSWER_MODEL",
    "OMBA_JUDGE_LLM",
    "OMBA_JUDGE_MODEL",
    "OPENAI_BASE_URL",
    "OPENAI_API_KEY",
    "OAMB_EMBEDDING_MODEL",
    "OAMB_HINDSIGHT_LLM_PROVIDER",
    "OAMB_HINDSIGHT_LLM_MODEL",
    "OAMB_HINDSIGHT_LLM_REASONING_EFFORT",
    "OAMB_MEM0_LLM_MODEL",
    "OAMB_MEM0_LLM_REASONING_EFFORT",
    "OAMB_OPENVIKING_VLM_PROVIDER",
    "OAMB_OPENVIKING_VLM_MODEL",
    "OAMB_OPENVIKING_VLM_REASONING_EFFORT",
}
stale_comment_lines = {
    "# AMB's OpenAI-compatible adapter consumes these names.",
    "# tcai deepseek url and keys",
    "# legacy alternate credentials",
}


def keep(line: str) -> bool:
    stripped = line.strip()
    if stripped in stale_comment_lines:
        return False
    if stripped.startswith("#"):
        commented = stripped[1:].strip()
        if "=" in commented and commented.split("=", 1)[0] in {
            "DEEPSEEK_BASE_URL",
            "DEEPSEEK_API_KEY",
        }:
            return False
        return True
    return "=" not in line or line.split("=", 1)[0] not in plan_owned


lines = path.read_text(encoding="utf-8").splitlines()
kept = [line for line in lines if keep(line)]
temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
temporary.write_text("\n".join(kept) + "\n", encoding="utf-8")
temporary.chmod(0o600)
temporary.replace(path)
PY
}

ensure_env_default() {
  local key=$1
  local default_value=$2
  local current_value
  current_value="$(read_env_value "$ENV_FILE" "$key" 2>/dev/null || true)"
  if [[ -z "$current_value" || "$current_value" == change-me* ]]; then
    set_env_value "$ENV_FILE" "$key" "$default_value"
  fi
}

ensure_model_environment() {
  [[ -f "$ENV_FILE" ]] || \
    die "prepare .env from .env.example, set DEEPSEEK_BASE_URL and DEEPSEEK_API_KEY, then rerun"
  chmod 600 "$ENV_FILE"
  remove_plan_owned_environment_values

  local base_url api_key
  base_url="$(read_env_value "$ENV_FILE" DEEPSEEK_BASE_URL 2>/dev/null || true)"
  api_key="$(read_env_value "$ENV_FILE" DEEPSEEK_API_KEY 2>/dev/null || true)"
  if [[ -n "$base_url" && "$base_url" != change-me* && -n "$api_key" && "$api_key" != change-me* ]]; then
    return
  fi
  die "configure DEEPSEEK_BASE_URL and DEEPSEEK_API_KEY in .env, then rerun"
}

ensure_provider_environment() {
  local run_label=$1
  local mem0_checkout=$2
  chmod 600 "$ENV_FILE"

  local model_url model_key value
  model_url="$(read_env_value "$ENV_FILE" DEEPSEEK_BASE_URL)"
  model_key="$(read_env_value "$ENV_FILE" DEEPSEEK_API_KEY)"
  value="$(read_env_value "$ENV_FILE" OAMB_PROVIDER_PROJECT 2>/dev/null || true)"
  if [[ -z "$value" || "$value" == change-me* ]]; then
    set_env_value "$ENV_FILE" OAMB_PROVIDER_PROJECT "oamb-providers-$run_label"
  fi
  value="$(read_env_value "$ENV_FILE" OAMB_MEM0_SOURCE_CHECKOUT 2>/dev/null || true)"
  if [[ -z "$value" || "$value" == change-me* ]]; then
    set_env_value "$ENV_FILE" OAMB_MEM0_SOURCE_CHECKOUT "$mem0_checkout"
  fi

  ensure_env_default OAMB_HINDSIGHT_PORT "$DEFAULT_HINDSIGHT_PORT"
  ensure_env_default OAMB_MEM0_PORT "$DEFAULT_MEM0_PORT"
  ensure_env_default OAMB_MEM0_INSPECTOR_PORT "$DEFAULT_MEM0_INSPECTOR_PORT"
  ensure_env_default OAMB_OPENVIKING_PORT "$DEFAULT_OPENVIKING_PORT"
  ensure_env_default OAMB_OPENVIKING_ACCOUNT_ID "$DEFAULT_OPENVIKING_ACCOUNT_ID"
  ensure_env_default OAMB_OPENVIKING_ADMIN_USER_ID "$DEFAULT_OPENVIKING_ADMIN_USER_ID"

  local key
  for key in \
    OAMB_HINDSIGHT_LLM_BASE_URL \
    OAMB_MEM0_LLM_BASE_URL \
    OAMB_OPENVIKING_VLM_BASE_URL; do
    value="$(read_env_value "$ENV_FILE" "$key" 2>/dev/null || true)"
    if [[ -z "$value" || "$value" == change-me* ]]; then
      set_env_value "$ENV_FILE" "$key" "$model_url"
    fi
  done
  for key in \
    OAMB_HINDSIGHT_LLM_API_KEY \
    OAMB_MEM0_LLM_API_KEY \
    OAMB_OPENVIKING_VLM_API_KEY; do
    value="$(read_env_value "$ENV_FILE" "$key" 2>/dev/null || true)"
    if [[ -z "$value" || "$value" == change-me* ]]; then
      set_env_value "$ENV_FILE" "$key" "$model_key"
    fi
  done
  for key in \
    OAMB_MEM0_ADMIN_API_KEY \
    OAMB_MEM0_JWT_SECRET \
    OAMB_MEM0_POSTGRES_PASSWORD \
    OAMB_MEM0_INSPECTOR_API_KEY \
    OAMB_OPENVIKING_ROOT_API_KEY; do
    value="$(read_env_value "$ENV_FILE" "$key" 2>/dev/null || true)"
    if [[ -z "$value" || "$value" == change-me* ]]; then
      set_env_value "$ENV_FILE" "$key" "$(random_secret)"
    fi
  done
  if [[ -n "$EMBEDDING_API_URL" ]]; then
    set_env_value "$ENV_FILE" OAMB_EMBEDDING_BASE_URL "$EMBEDDING_API_URL"
  fi
  ensure_env_default OAMB_EMBEDDING_BASE_URL "$DEFAULT_EMBEDDING_URL"
}

probe_embedding() {
  local base_url=$1
  local request
  request="$(jq -cn --arg model "$OAMB_EMBEDDING_MODEL" \
    '{model: $model, input: "OAMB startup probe", dimensions: 1024}')"
  curl --noproxy '*' --fail --silent --show-error \
    --connect-timeout 2 --max-time 10 \
    -H 'Authorization: Bearer oamb-local-embedding' \
    -H 'Content-Type: application/json' \
    --data-binary "$request" "${base_url%/}/embeddings" | \
    python3 -c '
import json
import math
import sys

payload = json.load(sys.stdin)
data = payload.get("data")
if not isinstance(data, list) or len(data) != 1 or not isinstance(data[0], dict):
    raise SystemExit(1)
vector = data[0].get("embedding")
if not isinstance(vector, list) or len(vector) != 1024:
    raise SystemExit(1)
if any(isinstance(item, bool) or not isinstance(item, (int, float)) or not math.isfinite(item) for item in vector):
    raise SystemExit(1)
'
}

start_local_embedding() {
  local configured_url=$1
  local host_url=$configured_url
  local helper_bind_host=""
  case "$(uname -s)" in
    Darwin) ;;
    Linux)
      helper_bind_host="$(resolve_docker_bridge_gateway)" || \
        die "cannot resolve the Docker bridge gateway for local Ollama"
      ;;
    *) die "local embedding startup supports only macOS and Linux; use --embedding-api-url" ;;
  esac
  host_url="$(resolve_host_embedding_base "$configured_url" "$helper_bind_host")" || \
    die "cannot resolve the host embedding URL"
  if probe_embedding "$host_url" >/dev/null 2>&1; then
    printf 'embedding: PASS (reusing %s)\n' "$host_url"
    return
  fi

  local helper
  case "$(uname -s)" in
    Darwin) helper="$ROOT/scripts/start_local_embedding/start_vllm_metal.sh" ;;
    Linux) helper="$ROOT/scripts/start_local_embedding/start_ollama_embedding.sh" ;;
  esac
  local log_file="$WORK_DIR/embedding.log"
  if [[ -n "$helper_bind_host" ]]; then
    OAMB_OLLAMA_BIND_HOST="$helper_bind_host" "$helper" >"$log_file" 2>&1 &
  else
    "$helper" >"$log_file" 2>&1 &
  fi
  local embedding_pid=$!
  printf '%s\n' "$embedding_pid" > "$WORK_DIR/embedding.pid"

  local attempt
  for ((attempt = 1; attempt <= EMBEDDING_STARTUP_ATTEMPTS; attempt++)); do
    if probe_embedding "$host_url" >/dev/null 2>&1; then
      printf 'embedding: PASS (%s, pid %s)\n' "$(basename "$helper")" "$embedding_pid"
      return
    fi
    if ! kill -0 "$embedding_pid" 2>/dev/null; then
      wait "$embedding_pid" || true
      die "embedding helper exited before readiness; inspect $log_file"
    fi
    sleep 1
  done
  die "embedding did not become ready; inspect $log_file"
}

validate_existing_readiness() {
  uv run --locked python - "$PLAN" "$ENV_FILE" "$RUNTIME_DIR" <<'PY'
import os
import sys
from pathlib import Path

from oamb.config.doctor import load_resolved_plan_for_run
from oamb.live import (
    load_live_environment,
    validate_live_readiness_receipt,
)

plan_path, env_file, runtime = map(Path, sys.argv[1:])
plan = load_resolved_plan_for_run(plan_path)
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
print("model readiness: PASS (existing six-role receipt revalidated)")
PY
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --embedding-api-url)
      [[ $# -ge 2 && -n "$2" ]] || die "--embedding-api-url requires a URL"
      EMBEDDING_API_URL=$2
      START_LOCAL_EMBEDDING=false
      shift 2
      ;;
    --no-start-embedding)
      START_LOCAL_EMBEDDING=false
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *) die "unknown argument: $1" ;;
  esac
done

for command_name in git uv curl jq python3 awk docker; do
  need_command "$command_name"
done

cd "$ROOT"
readonly RUN_LABEL="lme60-$(date -u +%Y%m%d-%H%M%S)-$$"
readonly WORK_DIR="$ROOT/.local-demo/$RUN_LABEL"
readonly MEM0_CHECKOUT="$ROOT/.local-demo/provider-source/mem0"
readonly DATASET_SOURCE="$ROOT/datasets/longmemeval-cleaned/longmemeval_s_cleaned.json"
readonly PLAN="$WORK_DIR/plan/resolved-plan.json"
mkdir -p "$WORK_DIR/plan" "$ROOT/.local-demo/provider-source"

uv sync --locked --all-groups
"$ROOT/scripts/download/longmemeval.sh"

if [[ ! -d "$MEM0_CHECKOUT/.git" ]]; then
  [[ ! -e "$MEM0_CHECKOUT" ]] || die "$MEM0_CHECKOUT exists but is not a Git checkout"
  git clone --branch v2.0.19 --single-branch \
    https://github.com/mem0ai/mem0.git "$MEM0_CHECKOUT"
fi
"$ROOT/scripts/verify_mem0_source.sh" "$MEM0_CHECKOUT"

ensure_model_environment
ensure_provider_environment "$RUN_LABEL" "$MEM0_CHECKOUT"

uv run --locked oamb doctor "$ROOT/configs/benchmark.yml" --output "$WORK_DIR/plan"
load_plan_model_environment "$PLAN" || die "cannot load model configuration from resolved plan"

if [[ "$START_LOCAL_EMBEDDING" == true ]]; then
  start_local_embedding "$(read_env_value "$ENV_FILE" OAMB_EMBEDDING_BASE_URL)"
else
  printf 'embedding: local startup skipped; configured API will be verified directly\n'
fi

"$ROOT/provider-services/bin/provider-services" doctor
"$ROOT/provider-services/bin/provider-services" build
"$ROOT/provider-services/bin/provider-services" up
"$ROOT/provider-services/bin/provider-services" verify --services
if [[ -f "$RUNTIME_DIR/model-readiness-receipt.json" ]]; then
  validate_existing_readiness
else
  "$ROOT/provider-services/bin/provider-services" verify --model-readiness \
    --resolved-plan "$PLAN" --model-env "$ENV_FILE"
  validate_existing_readiness
fi
"$ROOT/provider-services/bin/provider-services" status

state_temporary="$STATE_FILE.tmp-$$"
jq -n \
  --arg run_label "$RUN_LABEL" \
  --arg work_dir "$WORK_DIR" \
  --arg plan "$PLAN" \
  --arg dataset "$DATASET_SOURCE" \
  --arg question "$QUESTION_ID" \
  '{run_label: $run_label, work_dir: $work_dir,
    resolved_plan: $plan, dataset_source: $dataset, question_id: $question}' \
  > "$state_temporary"
mv "$state_temporary" "$STATE_FILE"

printf 'precheck: PASS\n'
printf 'next: ./run.sh --smoke_test\n'
