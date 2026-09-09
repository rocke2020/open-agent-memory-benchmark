#!/bin/sh

resolve_docker_bridge_gateway() {
    _oamb_gateway=$(docker network inspect bridge \
        --format '{{(index .IPAM.Config 0).Gateway}}' 2>/dev/null) || return 1
    python3 - "$_oamb_gateway" <<'PY' >/dev/null || return 1
import ipaddress
import sys

address = ipaddress.ip_address(sys.argv[1])
if (
    address.version != 4
    or address.is_unspecified
    or address.is_loopback
    or address.is_multicast
):
    raise SystemExit(1)
PY
    printf '%s\n' "$_oamb_gateway"
}

resolve_host_embedding_base() {
    _oamb_url=$1
    case "$_oamb_url" in
        http://host.docker.internal:*) ;;
        *) printf '%s\n' "$_oamb_url"; return ;;
    esac
    case "$(uname -s)" in
        Darwin) _oamb_host=127.0.0.1 ;;
        Linux)
            _oamb_host=${2:-}
            if [ -z "$_oamb_host" ]; then
                _oamb_host=$(resolve_docker_bridge_gateway) || return 1
            fi
            ;;
        *) return 1 ;;
    esac
    printf 'http://%s:%s\n' "$_oamb_host" "${_oamb_url#http://host.docker.internal:}"
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
    local log_file=${2:-$WORK_DIR/embedding.log}
    local pid_file=${3:-$WORK_DIR/embedding.pid}
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
    if [ -n "$helper_bind_host" ]; then
        OAMB_OLLAMA_BIND_HOST="$helper_bind_host" "$helper" >"$log_file" 2>&1 &
    else
        "$helper" >"$log_file" 2>&1 &
    fi
    local embedding_pid=$!
    printf '%s\n' "$embedding_pid" > "$pid_file"

    local attempt=1
    while [ "$attempt" -le "$EMBEDDING_STARTUP_ATTEMPTS" ]; do
        if probe_embedding "$host_url" >/dev/null 2>&1; then
            printf 'embedding: PASS (%s, pid %s)\n' "$(basename "$helper")" "$embedding_pid"
            return
        fi
        if ! kill -0 "$embedding_pid" 2>/dev/null; then
            wait "$embedding_pid" || true
            die "embedding helper exited before readiness; inspect $log_file"
        fi
        sleep 1
        attempt=$((attempt + 1))
    done
    die "embedding did not become ready; inspect $log_file"
}
