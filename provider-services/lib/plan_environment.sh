#!/bin/sh

# Export provider-owned model identities, embedding endpoint, and extraction retry count from one
# already-frozen OAMB plan.
load_plan_model_environment() {
  plan_environment_plan=$1
  [ -f "$plan_environment_plan" ] || return 1

  while IFS='|' read -r plan_environment_role plan_environment_variable \
    plan_environment_effort_variable; do
    plan_environment_value="$(jq -er --arg role "$plan_environment_role" '
      [.model_roles[] | select(.role_id == $role) | .model] |
      select(length == 1) | .[0] |
      select(type == "string" and length > 0)
    ' "$plan_environment_plan")" || return 1
    export "$plan_environment_variable=$plan_environment_value"
    if [ "$plan_environment_effort_variable" != "not_applicable" ]; then
      plan_environment_effort="$(jq -er --arg role "$plan_environment_role" '
        [.model_roles[] | select(.role_id == $role) | .thinking_effort] |
        select(length == 1) | .[0] |
        select(. == "low" or . == "high" or . == "max")
      ' "$plan_environment_plan")" || return 1
      export "$plan_environment_effort_variable=$plan_environment_effort"
    fi
  done <<'EOF'
hindsight_extraction|OAMB_HINDSIGHT_LLM_MODEL|OAMB_HINDSIGHT_LLM_REASONING_EFFORT
mem0_extraction|OAMB_MEM0_LLM_MODEL|OAMB_MEM0_LLM_REASONING_EFFORT
openviking_semantic_understanding|OAMB_OPENVIKING_VLM_MODEL|OAMB_OPENVIKING_VLM_REASONING_EFFORT
embedding|OAMB_EMBEDDING_MODEL|not_applicable
EOF

  plan_environment_extraction_max_retries="$(jq -er '
    .execution.extraction_max_retries |
    select(type == "number" and . == 10)
  ' "$plan_environment_plan")" || return 1
  export OAMB_EXTRACTION_MAX_RETRIES="$plan_environment_extraction_max_retries"
  plan_environment_embedding_base_url="$(jq -er '
    .embedding_endpoint.effective_endpoint |
    select(type == "string" and length > 0)
  ' "$plan_environment_plan")" || return 1
  export OAMB_EMBEDDING_BASE_URL="$plan_environment_embedding_base_url"
  plan_environment_embedding_ownership="$(jq -er '
    .embedding_endpoint.ownership |
    select(. == "embedding_local_fallback" or . == "external")
  ' "$plan_environment_plan")" || return 1
  export OAMB_EMBEDDING_OWNERSHIP="$plan_environment_embedding_ownership"

  # These provider processes expose an OpenAI-compatible protocol; this is an
  # implementation detail, not a user-selectable model setting.
  export OAMB_HINDSIGHT_LLM_PROVIDER=openai
  export OAMB_OPENVIKING_VLM_PROVIDER=openai

  unset plan_environment_plan plan_environment_role plan_environment_variable \
    plan_environment_effort_variable plan_environment_value plan_environment_effort \
    plan_environment_extraction_max_retries plan_environment_embedding_base_url \
    plan_environment_embedding_ownership
}
