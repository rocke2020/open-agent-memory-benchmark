#!/bin/bash

set -euo pipefail

MODEL_NAME="qwen3-embedding:0.6b"
EMBEDDING_DIMENSION=1024
OLLAMA_CONTEXT_LENGTH_TOKENS=8192
OLLAMA_PORT="${OAMB_OLLAMA_PORT:-18000}"
OLLAMA_BIND_HOST="${OAMB_OLLAMA_BIND_HOST:-}"
OLLAMA_BIND_ADDRESS="$OLLAMA_BIND_HOST:$OLLAMA_PORT"
OLLAMA_CLIENT_URL="http://$OLLAMA_BIND_ADDRESS"
STARTUP_ATTEMPTS=60
STARTUP_INTERVAL_SECONDS=1

if [[ ! "$OLLAMA_PORT" =~ ^[0-9]+$ ]] || ((OLLAMA_PORT < 1 || OLLAMA_PORT > 65535)); then
  echo "OAMB_OLLAMA_PORT must be an integer from 1 through 65535" >&2
  exit 1
fi
if [[ ! "$OLLAMA_BIND_HOST" =~ ^[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+$ ]] || \
   [[ "$OLLAMA_BIND_HOST" == "0.0.0.0" ]]; then
  echo "OAMB_OLLAMA_BIND_HOST must be the Docker bridge gateway IPv4 address" >&2
  exit 1
fi

if ! OLLAMA_BIN="$(command -v ollama)"; then
  echo "Ollama is not installed; follow https://docs.ollama.com/linux" >&2
  exit 1
fi
if ! CURL_BIN="$(command -v curl)"; then
  echo "curl is required" >&2
  exit 1
fi
if ! PYTHON_BIN="$(command -v python3)"; then
  echo "Python 3 is required" >&2
  exit 1
fi

server_pid=""

cleanup() {
  if [[ -n "$server_pid" ]] && kill -0 "$server_pid" 2>/dev/null; then
    kill "$server_pid" 2>/dev/null || true
    wait "$server_pid" 2>/dev/null || true
  fi
}

trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

server_is_ready() {
  "$CURL_BIN" --noproxy '*' --fail --silent --show-error \
    --connect-timeout 1 --max-time 2 "$OLLAMA_CLIENT_URL/api/version" >/dev/null 2>&1
}

if ! server_is_ready; then
  env \
    OLLAMA_HOST="$OLLAMA_BIND_ADDRESS" \
    OLLAMA_CONTEXT_LENGTH="$OLLAMA_CONTEXT_LENGTH_TOKENS" \
    "$OLLAMA_BIN" serve &
  server_pid=$!

  for ((attempt = 1; attempt <= STARTUP_ATTEMPTS; attempt++)); do
    if server_is_ready; then
      break
    fi
    if ! kill -0 "$server_pid" 2>/dev/null; then
      wait "$server_pid" || true
      echo "Ollama failed to start on $OLLAMA_BIND_ADDRESS; the port may already be in use" >&2
      exit 1
    fi
    sleep "$STARTUP_INTERVAL_SECONDS"
  done

  if ! server_is_ready; then
    echo "Ollama did not become ready within $STARTUP_ATTEMPTS seconds" >&2
    exit 1
  fi
fi

env OLLAMA_HOST="$OLLAMA_CLIENT_URL" "$OLLAMA_BIN" pull "$MODEL_NAME"

EMBEDDING_REQUEST=$(printf \
  '{"model":"%s","input":"OAMB embedding readiness probe","encoding_format":"float","dimensions":%s}' \
  "$MODEL_NAME" "$EMBEDDING_DIMENSION")

"$CURL_BIN" --noproxy '*' --fail --silent --show-error \
  --connect-timeout 3 --max-time 120 \
  -H 'Content-Type: application/json' \
  --data-binary "$EMBEDDING_REQUEST" \
  "$OLLAMA_CLIENT_URL/v1/embeddings" | \
  "$PYTHON_BIN" -c '
import json
import math
import sys

expected_model = sys.argv[1]
expected_dimension = int(sys.argv[2])
try:
    response = json.load(sys.stdin)
except (json.JSONDecodeError, UnicodeDecodeError) as exc:
    raise SystemExit(f"invalid Ollama embedding JSON: {exc}") from exc
if response.get("model") != expected_model:
    raise SystemExit(f"expected embedding model {expected_model}")
data = response.get("data")
if not isinstance(data, list) or len(data) != 1 or not isinstance(data[0], dict):
    raise SystemExit("expected exactly one embedding result")
vector = data[0].get("embedding")
if not isinstance(vector, list) or len(vector) != expected_dimension:
    raise SystemExit(f"expected exactly {expected_dimension} embedding values")
if any(
    isinstance(value, bool)
    or not isinstance(value, (int, float))
    or not math.isfinite(float(value))
    for value in vector
):
    raise SystemExit("embedding contains a non-finite or non-numeric value")
' "$MODEL_NAME" "$EMBEDDING_DIMENSION"

"$CURL_BIN" --noproxy '*' --fail --silent --show-error \
  --connect-timeout 3 --max-time 5 "$OLLAMA_CLIENT_URL/api/ps" | \
  "$PYTHON_BIN" -c '
import json
import sys

expected_model = sys.argv[1]
minimum_context = int(sys.argv[2])
try:
    response = json.load(sys.stdin)
except (json.JSONDecodeError, UnicodeDecodeError) as exc:
    raise SystemExit(f"invalid Ollama process JSON: {exc}") from exc
models = response.get("models")
if not isinstance(models, list):
    raise SystemExit("Ollama process response has no model list")
matches = [
    model
    for model in models
    if isinstance(model, dict)
    and (model.get("model") == expected_model or model.get("name") == expected_model)
]
if len(matches) != 1:
    raise SystemExit(f"expected exactly one loaded {expected_model} model")
context_length = matches[0].get("context_length")
if (
    isinstance(context_length, bool)
    or not isinstance(context_length, int)
    or context_length < minimum_context
):
    raise SystemExit(f"expected a loaded context of at least {minimum_context} tokens")
' "$MODEL_NAME" "$OLLAMA_CONTEXT_LENGTH_TOKENS"

printf 'PASS: %s returned one finite %s-dimensional vector with an %s-token context\n' \
  "$MODEL_NAME" "$EMBEDDING_DIMENSION" "$OLLAMA_CONTEXT_LENGTH_TOKENS"

if [[ -n "$server_pid" ]]; then
  printf 'Ollama is listening on %s; keep this terminal open.\n' "$OLLAMA_BIND_ADDRESS"
  wait "$server_pid"
fi
