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

# Plan-derived non-secret values must be exported by the frozen-plan loader.
# They never fall back to the private dotenv file.
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
