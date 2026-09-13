#!/bin/sh

# Keep the operator file inside the simple grammar interpreted identically by
# this bundle and Docker Compose. Values remain inert text.
read_file_mode() {
    target=$1
    if mode=$(stat -f '%Lp' "$target" 2>/dev/null); then
        printf '%s\n' "$mode"
        return 0
    fi
    if mode=$(stat -c '%a' "$target" 2>/dev/null); then
        printf '%s\n' "$mode"
        return 0
    fi
    return 1
}

validate_env_file() {
    awk '
        /^[[:space:]]*$/ || /^[[:space:]]*#/ { next }
        {
            separator = index($0, "=")
            if (separator == 0) {
                print "invalid dotenv line without =" > "/dev/stderr"
                failed = 1
                next
            }
            key = substr($0, 1, separator - 1)
            value = substr($0, separator + 1)
            if (key !~ /^[A-Za-z_][A-Za-z0-9_]*$/) {
                print "invalid dotenv key: " key > "/dev/stderr"
                failed = 1
            }
            if (++seen[key] > 1) {
                print "duplicate dotenv key: " key > "/dev/stderr"
                failed = 1
            }
            trimmed = value
            sub(/^[[:space:]]+/, "", trimmed)
            sub(/[[:space:]]+$/, "", trimmed)
            if (value != trimmed || value ~ /["$`\\#]/ || index(value, sprintf("%c", 39))) {
                print "unsupported dotenv value syntax for: " key > "/dev/stderr"
                failed = 1
            }
        }
        END { exit failed ? 1 : 0 }
    ' "$1"
}

# Read one unique dotenv assignment as inert text. Provider model connection
# aliases are derived from the canonical generic LLM pair and never stored again.
read_env_value() {
    env_file=$1
    wanted_name=$2
    case "$wanted_name" in
        OAMB_HINDSIGHT_LLM_BASE_URL|OAMB_MEM0_LLM_BASE_URL|OAMB_OPENVIKING_VLM_BASE_URL)
            wanted_name=LLM_BASE_URL
            ;;
        OAMB_HINDSIGHT_LLM_API_KEY|OAMB_MEM0_LLM_API_KEY|OAMB_OPENVIKING_VLM_API_KEY)
            wanted_name=LLM_API_KEY
            ;;
    esac
    awk -F= -v wanted="$wanted_name" \
        '$1 == wanted {sub(/^[^=]*=/, ""); value=$0; count+=1}
         END {if (count != 1) exit 1; print value}' \
        "$env_file"
}

# Read zero or one assignment as inert text. A missing optional assignment and
# an explicitly empty assignment both produce an empty string.
read_optional_env_value() {
    env_file=$1
    wanted_name=$2
    awk -F= -v wanted="$wanted_name" \
        '$1 == wanted {sub(/^[^=]*=/, ""); value=$0; count+=1}
         END {if (count > 1) exit 1; print value}' \
        "$env_file"
}

set_env_value() {
    env_file=$1
    env_key=$2
    env_value=$3
    OAMB_ENV_FILE="$env_file" OAMB_ENV_KEY="$env_key" OAMB_ENV_VALUE="$env_value" python3 - <<'PY'
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
    unset env_file env_key env_value
}

# Provider containers reach a host-local embedding server through Docker's
# host gateway, while host-side probes keep using the user-configured URL.
resolve_container_embedding_base() {
    embedding_endpoint=$1
    case "$embedding_endpoint" in
        https://127.0.0.1|https://127.0.0.1:*|https://127.0.0.1/*|https://localhost|https://localhost:*|https://localhost/*|http://\[::1\]|http://\[::1\]:*|http://\[::1\]/*|https://\[::1\]|https://\[::1\]:*|https://\[::1\]/*)
            return 1
            ;;
        http://127.0.0.1|http://127.0.0.1:*|http://127.0.0.1/*)
            printf 'http://host.docker.internal%s\n' "${embedding_endpoint#http://127.0.0.1}"
            ;;
        http://localhost|http://localhost:*|http://localhost/*)
            printf 'http://host.docker.internal%s\n' "${embedding_endpoint#http://localhost}"
            ;;
        *) printf '%s\n' "$embedding_endpoint" ;;
    esac
    unset embedding_endpoint
}

# OpenAI client libraries used inside the provider containers require a
# non-empty string even when the target embedding service does not authenticate.
effective_embedding_api_key() {
    embedding_env_file=$1
    embedding_api_key=$(read_optional_env_value "$embedding_env_file" OAMB_EMBEDDING_API_KEY) || return 1
    if [ -n "$embedding_api_key" ]; then
        printf '%s\n' "$embedding_api_key"
    else
        printf 'oamb-no-auth\n'
    fi
    unset embedding_env_file embedding_api_key
}

# Derive the producer-only application proxy bypass from the canonical LLM endpoint.
derive_llm_no_proxy() {
    llm_endpoint=$1
    case "$llm_endpoint" in
        http://*|https://*) ;;
        *) return 1 ;;
    esac
    llm_authority=${llm_endpoint#*://}
    llm_authority=${llm_authority%%/*}
    case "$llm_authority" in
        ""|*@*|*','*|*' '*|*'['*|*']'*) return 1 ;;
    esac
    llm_host=${llm_authority%%:*}
    case "$llm_host" in
        ""|.*|*.|*..*|*[!A-Za-z0-9.-]*) return 1 ;;
    esac
    printf '127.0.0.1,localhost,host.docker.internal,%s\n' "$llm_host"
    unset llm_endpoint llm_authority llm_host
}

# Provider aliases and plan-derived controls are exported by the runtime loader.
# Generative model names come from the current root dotenv file.
read_runtime_env_value() {
    _env_file=$1
    wanted_name=$2
    printenv "$wanted_name" 2>/dev/null
}

reject_plan_owned_env_values() {
    awk '
        BEGIN {
            split("OMBA_ANSWER_LLM OMBA_ANSWER_MODEL OMBA_JUDGE_LLM OMBA_JUDGE_MODEL OPENAI_BASE_URL OPENAI_API_KEY DEEPSEEK_BASE_URL DEEPSEEK_API_KEY OAMB_EMBEDDING_MODEL OAMB_HINDSIGHT_LLM_PROVIDER OAMB_HINDSIGHT_LLM_MODEL OAMB_HINDSIGHT_LLM_REASONING_EFFORT OAMB_MEM0_LLM_MODEL OAMB_MEM0_LLM_REASONING_EFFORT OAMB_OPENVIKING_VLM_PROVIDER OAMB_OPENVIKING_VLM_MODEL OAMB_OPENVIKING_VLM_REASONING_EFFORT OAMB_HINDSIGHT_LLM_BASE_URL OAMB_HINDSIGHT_LLM_API_KEY OAMB_MEM0_LLM_BASE_URL OAMB_MEM0_LLM_API_KEY OAMB_OPENVIKING_VLM_BASE_URL OAMB_OPENVIKING_VLM_API_KEY", names)
            for (i in names) forbidden[names[i]] = 1
        }
        /^[[:space:]]*$/ || /^[[:space:]]*#/ { next }
        {
            separator = index($0, "=")
            key = substr($0, 1, separator - 1)
            if (forbidden[key]) {
                print "plan-owned, derived, or stale key is forbidden in dotenv: " key > "/dev/stderr"
                failed = 1
            }
        }
        END { exit failed ? 1 : 0 }
    ' "$1"
}
