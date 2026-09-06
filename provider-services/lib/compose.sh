#!/bin/sh

# Resolve every Compose input from its authoritative source before invoking
# Docker. Private/runtime inputs come from the root dotenv file; model settings
# come from the frozen plan exported into the process environment.
oamb_compose() {
    oamb_compose_root=$1
    oamb_compose_env_file=$2
    shift 2
    oamb_llm_no_proxy=$(derive_llm_no_proxy "$(read_env_value "$oamb_compose_env_file" LLM_BASE_URL)") || {
        printf 'invalid LLM_BASE_URL for producer proxy bypass\n' >&2
        return 1
    }

    env \
        OAMB_LLM_NO_PROXY="$oamb_llm_no_proxy" \
        OAMB_PROVIDER_PROJECT="$(read_env_value "$oamb_compose_env_file" OAMB_PROVIDER_PROJECT)" \
        OAMB_HINDSIGHT_PORT="$(read_env_value "$oamb_compose_env_file" OAMB_HINDSIGHT_PORT)" \
        OAMB_MEM0_PORT="$(read_env_value "$oamb_compose_env_file" OAMB_MEM0_PORT)" \
        OAMB_MEM0_INSPECTOR_PORT="$(read_env_value "$oamb_compose_env_file" OAMB_MEM0_INSPECTOR_PORT)" \
        OAMB_OPENVIKING_PORT="$(read_env_value "$oamb_compose_env_file" OAMB_OPENVIKING_PORT)" \
        OAMB_EMBEDDING_BASE_URL="$(read_env_value "$oamb_compose_env_file" OAMB_EMBEDDING_BASE_URL)" \
        OAMB_EMBEDDING_MODEL="$(read_runtime_env_value "$oamb_compose_env_file" OAMB_EMBEDDING_MODEL)" \
        OAMB_HINDSIGHT_LLM_PROVIDER=openai \
        OAMB_HINDSIGHT_LLM_MODEL="$(read_runtime_env_value "$oamb_compose_env_file" OAMB_HINDSIGHT_LLM_MODEL)" \
        OAMB_HINDSIGHT_LLM_REASONING_EFFORT="$(read_runtime_env_value "$oamb_compose_env_file" OAMB_HINDSIGHT_LLM_REASONING_EFFORT)" \
        OAMB_HINDSIGHT_LLM_BASE_URL="$(read_env_value "$oamb_compose_env_file" OAMB_HINDSIGHT_LLM_BASE_URL)" \
        OAMB_HINDSIGHT_LLM_API_KEY="$(read_env_value "$oamb_compose_env_file" OAMB_HINDSIGHT_LLM_API_KEY)" \
        OAMB_MEM0_LLM_MODEL="$(read_runtime_env_value "$oamb_compose_env_file" OAMB_MEM0_LLM_MODEL)" \
        OAMB_MEM0_LLM_REASONING_EFFORT="$(read_runtime_env_value "$oamb_compose_env_file" OAMB_MEM0_LLM_REASONING_EFFORT)" \
        OAMB_MEM0_LLM_BASE_URL="$(read_env_value "$oamb_compose_env_file" OAMB_MEM0_LLM_BASE_URL)" \
        OAMB_MEM0_LLM_API_KEY="$(read_env_value "$oamb_compose_env_file" OAMB_MEM0_LLM_API_KEY)" \
        OAMB_MEM0_ADMIN_API_KEY="$(read_env_value "$oamb_compose_env_file" OAMB_MEM0_ADMIN_API_KEY)" \
        OAMB_MEM0_JWT_SECRET="$(read_env_value "$oamb_compose_env_file" OAMB_MEM0_JWT_SECRET)" \
        OAMB_MEM0_POSTGRES_PASSWORD="$(read_env_value "$oamb_compose_env_file" OAMB_MEM0_POSTGRES_PASSWORD)" \
        OAMB_MEM0_INSPECTOR_API_KEY="$(read_env_value "$oamb_compose_env_file" OAMB_MEM0_INSPECTOR_API_KEY)" \
        OAMB_OPENVIKING_VLM_PROVIDER=openai \
        OAMB_OPENVIKING_VLM_MODEL="$(read_runtime_env_value "$oamb_compose_env_file" OAMB_OPENVIKING_VLM_MODEL)" \
        OAMB_OPENVIKING_VLM_REASONING_EFFORT="$(read_runtime_env_value "$oamb_compose_env_file" OAMB_OPENVIKING_VLM_REASONING_EFFORT)" \
        OAMB_OPENVIKING_VLM_BASE_URL="$(read_env_value "$oamb_compose_env_file" OAMB_OPENVIKING_VLM_BASE_URL)" \
        OAMB_OPENVIKING_VLM_API_KEY="$(read_env_value "$oamb_compose_env_file" OAMB_OPENVIKING_VLM_API_KEY)" \
        OAMB_OPENVIKING_ROOT_API_KEY="$(read_env_value "$oamb_compose_env_file" OAMB_OPENVIKING_ROOT_API_KEY)" \
        docker compose \
        -p "$(read_env_value "$oamb_compose_env_file" OAMB_PROVIDER_PROJECT)" \
        --env-file "$oamb_compose_env_file" -f "$oamb_compose_root/compose.yaml" "$@"
    oamb_compose_status=$?
    unset oamb_llm_no_proxy
    return "$oamb_compose_status"
}
