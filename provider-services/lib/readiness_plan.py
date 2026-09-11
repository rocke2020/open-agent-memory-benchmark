"""Validate one resolved plan and expose its secret-free live-readiness inputs."""

from __future__ import annotations

import os
import sys
from pathlib import Path

from oamb.config.doctor import load_resolved_plan_for_run
from oamb.contracts.ids import canonical_json_bytes
from oamb.live import (
    _runtime_model,
    live_readiness_environment_hash,
    load_live_environment,
    resolve_service_verification_receipt,
    validate_live_service_verification_receipt,
)


def readiness_plan_document(
    path: Path,
    *,
    provider_env_path: Path,
    model_env_path: Path,
    provider_runtime_directory: Path,
) -> dict[str, object]:
    plan = load_resolved_plan_for_run(path)
    environment = load_live_environment(
        plan=plan,
        provider_env_path=provider_env_path,
        model_env_path=model_env_path,
        provider_runtime_directory=provider_runtime_directory,
        base_environment=os.environ,
    )
    service_receipt_path = resolve_service_verification_receipt(provider_runtime_directory)
    validate_live_service_verification_receipt(
        plan=plan,
        provider_runtime_directory=provider_runtime_directory,
        environment=environment,
        service_receipt_path=service_receipt_path,
    )
    return {
        "schema_name": "oamb_live_readiness_plan",
        "schema_version": 1,
        "resolved_plan_hash": plan.resolved_plan_hash,
        "max_retries_per_operation": plan.execution.max_retries_per_operation,
        "operation_timeout_seconds": plan.execution.operation_timeout_seconds,
        "environment_hash": live_readiness_environment_hash(plan, environment),
        "service_verification_receipt_sha256": service_receipt_path.stem,
        "model_roles": [
            {
                "role_id": role.role_id,
                "model": _runtime_model(role, environment),
                "thinking_effort": role.thinking_effort,
                "endpoint_variable": role.endpoint_variable,
                "credential_variable": role.credential_variable,
                "maximum_output_tokens_per_call": role.maximum_output_tokens_per_call,
            }
            for role in plan.model_roles
        ],
    }


def main() -> None:
    if len(sys.argv) != 5:
        raise SystemExit(
            "usage: readiness_plan.py RESOLVED_PLAN PROVIDER_ENV MODEL_ENV PROVIDER_RUNTIME"
        )
    sys.stdout.buffer.write(
        canonical_json_bytes(
            readiness_plan_document(
                Path(sys.argv[1]),
                provider_env_path=Path(sys.argv[2]),
                model_env_path=Path(sys.argv[3]),
                provider_runtime_directory=Path(sys.argv[4]),
            )
        )
    )


if __name__ == "__main__":
    main()
