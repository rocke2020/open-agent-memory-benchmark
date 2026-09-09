#!/bin/sh
set -eu

ROOT=${1:?provider-services root required}
ENV_FILE=${2:?provider-services env file required}
. "$ROOT/lib/env.sh"
. "$ROOT/lib/compose.sh"

env_value() {
  case "$1" in
    OAMB_EMBEDDING_MODEL|OAMB_MEM0_LLM_MODEL|OAMB_MEM0_LLM_REASONING_EFFORT) read_runtime_env_value "$ENV_FILE" "$1" ;;
    *) read_env_value "$ENV_FILE" "$1" ;;
  esac
}

required_env_value() {
  value=$(env_value "$1") || { printf 'mem0 bootstrap: missing %s\n' "$1" >&2; exit 1; }
  [ -n "$value" ] || { printf 'mem0 bootstrap: empty %s\n' "$1" >&2; exit 1; }
  printf '%s' "$value"
}

OAMB_MEM0_PORT=$(env_value OAMB_MEM0_PORT || printf '18889')
OAMB_MEM0_LLM_API_KEY=$(required_env_value OAMB_MEM0_LLM_API_KEY)
OAMB_MEM0_LLM_MODEL=$(required_env_value OAMB_MEM0_LLM_MODEL)
OAMB_MEM0_LLM_REASONING_EFFORT=$(required_env_value OAMB_MEM0_LLM_REASONING_EFFORT)
OAMB_MEM0_LLM_BASE_URL=$(required_env_value OAMB_MEM0_LLM_BASE_URL)
OAMB_EMBEDDING_MODEL=$(required_env_value OAMB_EMBEDDING_MODEL)
OAMB_EMBEDDING_BASE_URL=$(resolve_container_embedding_base "$(required_env_value OAMB_EMBEDDING_BASE_URL)")
OAMB_EMBEDDING_API_KEY=$(effective_embedding_api_key "$ENV_FILE")
OAMB_MEM0_POSTGRES_PASSWORD=$(required_env_value OAMB_MEM0_POSTGRES_PASSWORD)
OAMB_MEM0_ADMIN_API_KEY=$(required_env_value OAMB_MEM0_ADMIN_API_KEY)

compose() {
  oamb_compose "$ROOT" "$ENV_FILE" "$@"
}

BASE_URL="http://127.0.0.1:$OAMB_MEM0_PORT"
RUNTIME_DIR="$ROOT/.runtime"
mkdir -p "$RUNTIME_DIR"
CONFIG_FILE="$RUNTIME_DIR/mem0-config.json"
EXPECTED_FILE="$RUNTIME_DIR/mem0-config-redacted.json"
ACTUAL_FILE="$RUNTIME_DIR/mem0-config-actual.json"
ADMIN_HEADER_FILE="$RUNTIME_DIR/mem0-admin-header"
umask 077
printf 'X-API-Key: %s\n' "$OAMB_MEM0_ADMIN_API_KEY" > "$ADMIN_HEADER_FILE"
chmod 600 "$ADMIN_HEADER_FILE"

printf '%s' "$OAMB_EMBEDDING_API_KEY" | jq -Rs \
  --arg llm_key "$OAMB_MEM0_LLM_API_KEY" \
  --arg llm_model "$OAMB_MEM0_LLM_MODEL" \
  --arg llm_effort "$OAMB_MEM0_LLM_REASONING_EFFORT" \
  --arg llm_base "$OAMB_MEM0_LLM_BASE_URL" \
  --arg embed_model "$OAMB_EMBEDDING_MODEL" \
  --arg embed_base "$OAMB_EMBEDDING_BASE_URL" \
  --arg postgres_password "$OAMB_MEM0_POSTGRES_PASSWORD" \
  '. as $embed_key | {
    version: "v1.1",
    vector_store: {provider: "pgvector", config: {
      host: "mem0-postgres", port: 5432, dbname: "postgres",
      user: "oamb_mem0", password: $postgres_password,
      collection_name: "oamb_memories", embedding_model_dims: 1024,
      diskann: false, hnsw: true
    }},
    llm: {provider: "openai", config: {
      api_key: $llm_key, model: $llm_model, openai_base_url: $llm_base,
      temperature: 0.0, max_tokens: 2000,
      reasoning_effort: $llm_effort, is_reasoning_model: true
    }},
    embedder: {provider: "openai", config: {
      api_key: $embed_key, model: $embed_model,
      openai_base_url: $embed_base, embedding_dims: 1024
    }},
    history_db_path: "/app/history/history.db",
    reranker: null
  }' > "$CONFIG_FILE"

jq 'walk(if type == "object" then with_entries(. as $entry |
      if (["admin_api_key","api_key","authorization","jwt_secret","password","password_hash","secret","token"] | index($entry.key | ascii_downcase))
      then .value = (if .value then "[redacted]" else .value end) else . end)
    else . end)' "$CONFIG_FILE" > "$EXPECTED_FILE"

curl --noproxy '*' --fail --silent --show-error --connect-timeout 3 --max-time 30 \
  -H "@$ADMIN_HEADER_FILE" -H 'Content-Type: application/json' \
  --data-binary "@$CONFIG_FILE" "$BASE_URL/configure" | \
  jq -e '.message == "Configuration set successfully"' >/dev/null

readback() {
  curl --noproxy '*' --fail --silent --show-error --connect-timeout 3 --max-time 10 \
    -H "@$ADMIN_HEADER_FILE" "$BASE_URL/configure" | jq -S . > "$ACTUAL_FILE"
  jq -S . "$EXPECTED_FILE" | cmp -s - "$ACTUAL_FILE" || {
    printf 'mem0 bootstrap: redacted configuration readback mismatch\n' >&2
    exit 1
  }
}

readback
compose exec -T mem0-postgres psql -U oamb_mem0 -d mem0_app -tAc \
  "SELECT count(*) = 1 FROM settings WHERE key = 'config_overrides' AND value <> '';" | \
  grep -qx 't'
compose exec -T mem0 alembic current | grep -Eq '^006( |$)'
compose restart mem0 >/dev/null

attempts=0
until curl --noproxy '*' --fail --silent --show-error --connect-timeout 2 --max-time 5 \
  "$BASE_URL/openapi.json" >/dev/null 2>&1; do
  attempts=$((attempts + 1))
  [ "$attempts" -lt 60 ] || { printf 'mem0 bootstrap: restart readiness timeout\n' >&2; exit 1; }
  sleep 2
done
readback
printf 'mem0 bootstrap: PASS (configuration persisted across restart)\n'
