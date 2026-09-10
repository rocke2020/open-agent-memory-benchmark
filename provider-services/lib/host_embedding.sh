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

request_embedding() {
    local base_url=$1
    local api_key=$2
    local request=$3
    local connect_timeout=${4:-2}
    local maximum_time=${5:-10}
    set -- --noproxy '*' --fail --silent --show-error \
        --connect-timeout "$connect_timeout" --max-time "$maximum_time" \
        -H 'Content-Type: application/json'
    if [ -n "$api_key" ]; then
        printf 'header = "Authorization: Bearer %s"\n' "$api_key" | \
            curl "$@" --config - --data-binary "$request" "${base_url%/}/embeddings"
        return
    fi
    curl "$@" --data-binary "$request" "${base_url%/}/embeddings"
}

probe_embedding() {
    local base_url=$1
    local api_key=$2
    local request
    request="$(jq -cn --arg model "$OAMB_EMBEDDING_MODEL" \
        '{model: $model, input: "OAMB startup probe", dimensions: 1024}')"
    request_embedding "$base_url" "$api_key" "$request" | \
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
    local api_key=$2
    local log_file=${3:-$WORK_DIR/embedding.log}
    local pid_file=${4:-$WORK_DIR/embedding.pid}
    local host_url=$configured_url
    local helper_bind_host=""
    local helper_relay_host=""
    local relay_url=""
    case "$(uname -s)" in
        Darwin) ;;
        Linux)
            helper_relay_host="$(resolve_docker_bridge_gateway)" || \
                die "cannot resolve the Docker bridge gateway for local Ollama"
            helper_bind_host=127.0.0.1
            case "$configured_url" in
                http://127.0.0.1:*)
                    relay_url="http://$helper_relay_host${configured_url#http://127.0.0.1}"
                    ;;
                *) die "managed-local Linux embedding endpoint must use 127.0.0.1" ;;
            esac
            ;;
        *) die "local embedding startup supports only macOS and Linux; use --embedding-api-url" ;;
    esac
    host_url="$(resolve_host_embedding_base "$configured_url" "$helper_bind_host")" || \
        die "cannot resolve the host embedding URL"
    if probe_embedding "$host_url" "$api_key" >/dev/null 2>&1 && \
        { [ -z "$relay_url" ] || probe_embedding "$relay_url" "$api_key" >/dev/null 2>&1; }
    then
        printf 'embedding: PASS (reusing %s)\n' "$host_url"
        return
    fi

    [ ! -L "$pid_file" ] || die "embedding PID path is unsafe: $pid_file"
    local embedding_pid=""
    local started_here=false
    if [ -f "$pid_file" ]; then
        embedding_pid=$(cat "$pid_file") || die "cannot read embedding PID: $pid_file"
        case "$embedding_pid" in
            ""|*[!0-9]*) embedding_pid="" ;;
        esac
        if [ -n "$embedding_pid" ] && [ "$embedding_pid" -gt 0 ] 2>/dev/null && \
            kill -0 "$embedding_pid" 2>/dev/null; then
            printf 'embedding: waiting for existing startup (pid %s)\n' "$embedding_pid"
        else
            embedding_pid=""
        fi
    fi

    local helper
    case "$(uname -s)" in
        Darwin) helper="$ROOT/scripts/start_local_embedding/start_vllm_metal.sh" ;;
        Linux) helper="$ROOT/scripts/start_local_embedding/start_ollama_embedding.sh" ;;
    esac
    if [ -z "$embedding_pid" ]; then
        if [ -n "$helper_relay_host" ]; then
            OAMB_OLLAMA_BIND_HOST="$helper_bind_host" \
                OAMB_OLLAMA_RELAY_HOST="$helper_relay_host" \
                "$helper" >"$log_file" 2>&1 &
        else
            "$helper" >"$log_file" 2>&1 &
        fi
        embedding_pid=$!
        printf '%s\n' "$embedding_pid" > "$pid_file"
        started_here=true
    fi

    local attempt=1
    while [ "$attempt" -le "$EMBEDDING_STARTUP_ATTEMPTS" ]; do
        if probe_embedding "$host_url" "$api_key" >/dev/null 2>&1 && \
            { [ -z "$relay_url" ] || probe_embedding "$relay_url" "$api_key" >/dev/null 2>&1; }
        then
            if [ "$started_here" = true ]; then
                printf 'embedding: PASS (%s, pid %s)\n' "$(basename "$helper")" "$embedding_pid"
            else
                printf 'embedding: PASS (existing startup, pid %s)\n' "$embedding_pid"
            fi
            return
        fi
        if ! kill -0 "$embedding_pid" 2>/dev/null; then
            if [ "$started_here" = true ]; then
                wait "$embedding_pid" || true
                die "embedding helper exited before readiness; inspect $log_file"
            fi
            die "recorded embedding helper exited before readiness: $embedding_pid"
        fi
        sleep 1
        attempt=$((attempt + 1))
    done
    if [ "$started_here" = true ]; then
        die "embedding did not become ready; inspect $log_file"
    fi
    die "recorded embedding helper did not become ready: $embedding_pid"
}
