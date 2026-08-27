#!/bin/sh
set -eu

ROOT=${1:?provider-services root required}
ENV_FILE=${2:?provider-services env file required}
. "$ROOT/lib/env.sh"

env_value() {
  read_env_value "$ENV_FILE" "$1"
}

required_env_value() {
  value=$(env_value "$1") || { printf 'openviking bootstrap: missing %s\n' "$1" >&2; exit 1; }
  [ -n "$value" ] || { printf 'openviking bootstrap: empty %s\n' "$1" >&2; exit 1; }
  printf '%s' "$value"
}

OAMB_OPENVIKING_PORT=$(env_value OAMB_OPENVIKING_PORT || printf '19330')
OAMB_OPENVIKING_ROOT_API_KEY=$(required_env_value OAMB_OPENVIKING_ROOT_API_KEY)
OAMB_OPENVIKING_ACCOUNT_ID=$(required_env_value OAMB_OPENVIKING_ACCOUNT_ID)
OAMB_OPENVIKING_ADMIN_USER_ID=$(required_env_value OAMB_OPENVIKING_ADMIN_USER_ID)

BASE_URL="http://127.0.0.1:$OAMB_OPENVIKING_PORT"
RUNTIME_DIR="$ROOT/.runtime"
KEY_FILE="$RUNTIME_DIR/openviking-user-key"
ROOT_HEADER_FILE="$RUNTIME_DIR/openviking-root-header"
USER_HEADER_FILE="$RUNTIME_DIR/openviking-user-header"
mkdir -p "$RUNTIME_DIR"
umask 077
printf 'X-API-Key: %s\n' "$OAMB_OPENVIKING_ROOT_API_KEY" > "$ROOT_HEADER_FILE"
chmod 600 "$ROOT_HEADER_FILE"

root_get() {
  curl --noproxy '*' --fail --silent --show-error --connect-timeout 3 --max-time 15 \
    -H "@$ROOT_HEADER_FILE" "$BASE_URL$1"
}

if [ -f "$KEY_FILE" ]; then
  user_key=$(sed -n '1p' "$KEY_FILE")
else
  accounts=$(root_get /api/v1/admin/accounts)
  if printf '%s' "$accounts" | jq -e --arg account "$OAMB_OPENVIKING_ACCOUNT_ID" \
      '.result | any((.account_id? // .) == $account)' >/dev/null; then
    printf 'openviking bootstrap: account exists but local user key is absent; refusing to rotate it\n' >&2
    exit 1
  fi
  response=$(jq -n \
      --arg account "$OAMB_OPENVIKING_ACCOUNT_ID" \
      --arg user "$OAMB_OPENVIKING_ADMIN_USER_ID" \
      '{account_id: $account, admin_user_id: $user}' | \
    curl --noproxy '*' --fail --silent --show-error --connect-timeout 3 --max-time 30 \
      -H "@$ROOT_HEADER_FILE" -H 'Content-Type: application/json' \
      --data-binary @- "$BASE_URL/api/v1/admin/accounts")
  user_key=$(printf '%s' "$response" | jq -er '.result.user_key')
  printf '%s\n' "$user_key" > "$KEY_FILE"
  chmod 600 "$KEY_FILE"
fi

printf 'X-API-Key: %s\n' "$user_key" > "$USER_HEADER_FILE"
chmod 600 "$USER_HEADER_FILE"

curl --noproxy '*' --fail --silent --show-error --connect-timeout 3 --max-time 15 \
  -H "@$USER_HEADER_FILE" "$BASE_URL/api/v1/system/status" | \
  jq -e --arg user "$OAMB_OPENVIKING_ADMIN_USER_ID" \
    '.status == "ok" and .result.initialized == true and .result.user == $user' >/dev/null

curl --noproxy '*' --fail --silent --show-error --connect-timeout 3 --max-time 15 \
  -H "@$USER_HEADER_FILE" "$BASE_URL/health" | \
  jq -e --arg account "$OAMB_OPENVIKING_ACCOUNT_ID" \
    --arg user "$OAMB_OPENVIKING_ADMIN_USER_ID" \
    '.status == "ok" and .account_id == $account and .user_id == $user and
     .role == "admin"' >/dev/null

root_health=$(root_get /health)
printf '%s' "$root_health" | jq -e '.status == "ok" and .role == "root"' >/dev/null
printf 'openviking bootstrap: PASS (dedicated account ADMIN identity verified as non-ROOT)\n'
