#!/usr/bin/env bash

# Manual external probe. This sends one potentially billable DeepSeek request.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
ENV_FILE="${DEEPSEEK_ENV_FILE:-$REPO_ROOT/.env}"

if [[ -f "$ENV_FILE" ]]; then
  set -a
  # shellcheck disable=SC1090
  source "$ENV_FILE"
  set +a
fi

: "${DEEPSEEK_BASE_URL:?FAIL: set DEEPSEEK_BASE_URL or add it to $ENV_FILE}"
: "${DEEPSEEK_API_KEY:?FAIL: set DEEPSEEK_API_KEY or add it to $ENV_FILE}"

MODEL="${DEEPSEEK_TEST_MODEL:-deepseek-chat}"
MODEL_URL="${DEEPSEEK_BASE_URL%/}/chat/completions"
PROMPT='Reply with exactly OAMB_DEEPSEEK_OK'
EXPECTED_CONTENT='OAMB_DEEPSEEK_OK'

printf 'input_prompt=%s\n' "$PROMPT"
printf 'expected_content=%s\n' "$EXPECTED_CONTENT"
printf 'model_url=%s\n' "$MODEL_URL"
printf 'api_key_prefix=%.6s\n' "$DEEPSEEK_API_KEY"

if ! RESPONSE="$(curl --silent --show-error --fail-with-body \
  --connect-timeout 15 \
  --max-time 120 \
  --header "Authorization: Bearer $DEEPSEEK_API_KEY" \
  --header 'Content-Type: application/json' \
  --data "$(jq -nc --arg model "$MODEL" --arg prompt "$PROMPT" '{
    model: $model,
    messages: [{role: "user", content: $prompt}],
    max_tokens: 1024,
    stream: false
  }')" \
  "$MODEL_URL")"; then
  printf 'FAIL: DeepSeek request failed.\n' >&2
  exit 1
fi

if ! SUMMARY="$(jq -er '
  .choices[0] as $choice
  | select(($choice.finish_reason // "") != "")
  | select($choice.finish_reason != "length")
  | select(($choice.message.content // "") != "")
  | "runtime_model=\(.model // "unknown")\n"
    + "finish_reason=\($choice.finish_reason)"
' <<<"$RESPONSE")"; then
  printf 'FAIL: response was empty, malformed, or ended with finish_reason=length.\n' >&2
  exit 1
fi

REPLIED_CONTENT="$(jq -r '.choices[0].message.content' <<<"$RESPONSE")"
printf 'replied_content=%s\n' "$REPLIED_CONTENT"

if [[ "$REPLIED_CONTENT" != "$EXPECTED_CONTENT" ]]; then
  printf 'FAIL: replied content does not exactly match expected content.\n' >&2
  exit 1
fi

printf 'PASS: DeepSeek chat request succeeded.\n'
printf 'requested_model=%s\n%s\n' "$MODEL" "$SUMMARY"
