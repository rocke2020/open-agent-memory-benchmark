from __future__ import annotations

import hashlib
import re
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = ROOT.parent


class ServiceBundleContractTests(unittest.TestCase):
    def read(self, relative_path: str) -> str:
        path = ROOT / relative_path
        self.assertTrue(path.is_file(), f"missing required file: {relative_path}")
        return path.read_text(encoding="utf-8")

    def read_root_env_example(self) -> str:
        path = REPOSITORY_ROOT / ".env.example"
        self.assertTrue(path.is_file(), "missing required root .env.example")
        return path.read_text(encoding="utf-8")

    def test_release_and_image_pins_are_immutable(self) -> None:
        versions = self.read("versions.env")
        expected = {
            "HINDSIGHT_VERSION": "0.9.2",
            "HINDSIGHT_COMMIT": "ebad478240d3171bb88201ececda5e8d9883d22d",
            "HINDSIGHT_IMAGE_DIGEST": "sha256:7635a15739361dbdf221ba796ad25a813f876144fe113022eea8e26cb6ee75e7",
            "MEM0_VERSION": "2.0.19",
            "MEM0_COMMIT": "dc82354e143c2581d505d581a00286d6ef8c3605",
            "OPENVIKING_VERSION": "0.4.16",
            "OPENVIKING_COMMIT": "499995f3ed2e7f551a715179c4053772c51ff819",
            "OPENVIKING_IMAGE_DIGEST": "sha256:46f9e34cd37238c28cbd9535033773d179006bdf7f3e528dd1c46567abce7701",
        }
        parsed = dict(
            line.split("=", 1)
            for line in versions.splitlines()
            if line and not line.startswith("#")
        )
        for key, value in expected.items():
            self.assertEqual(parsed.get(key), value, key)
        self.assertNotIn("MEM0_SERVER_OVERLAY_SHA256", parsed)
        self.assertNotIn("MEM0_QDRANT_VERSION", parsed)
        self.assertNotIn("MEM0_QDRANT_IMAGE_DIGEST", parsed)
        self.assertEqual(
            hashlib.sha256((ROOT / "retry-guard" / "sitecustomize.py").read_bytes()).hexdigest(),
            parsed["RETRY_GUARD_SHA256"],
        )

    def test_compose_has_five_long_running_services_and_read_only_storage_probe(self) -> None:
        compose = self.read("compose.yaml")
        for service in (
            "hindsight",
            "mem0-postgres",
            "mem0",
            "mem0-inspector",
            "openviking",
        ):
            self.assertRegex(compose, rf"(?m)^  {re.escape(service)}:$")
        self.assertNotRegex(compose, r"(?m)^  mem0-qdrant:$")
        self.assertNotRegex(compose, r"(?m)^  mem0_qdrant:$")
        self.assertNotIn("QDRANT", compose)
        self.assertEqual(compose.count("    healthcheck:\n"), 5)
        self.assertRegex(compose, r"(?m)^  openviking-storage-probe:$")
        probe = compose.split("  openviking-storage-probe:\n", 1)[1].split("\nvolumes:\n", 1)[0]
        self.assertIn('profiles: ["verification"]', probe)
        self.assertIn("network_mode: none", probe)
        self.assertIn("read_only: true", probe)
        self.assertIn("target: /probe/workspace", probe)
        self.assertIn("read_only: true", probe)
        self.assertIn('cap_drop: ["ALL"]', probe)
        self.assertIn('security_opt: ["no-new-privileges:true"]', probe)
        self.assertNotRegex(probe, r"(?i)(api_key|credential|secret)")
        self.assertNotRegex(compose, r"(?m)^\s*image:\s*\S*:latest(?:\s|$)")
        self.assertIn('HINDSIGHT_API_SKIP_LLM_VERIFICATION: "true"', compose)
        self.assertRegex(compose, r"(?m)^    shm_size: 1gb$")
        self.assertNotIn('"5432:', compose)
        self.assertNotIn('"6333:', compose)
        for port in ("18888", "18889", "16333", "19330"):
            self.assertIn("127.0.0.1:${OAMB_", compose)
            self.assertIn(port, compose)

    def test_inspector_uses_private_postgres_without_runner_credentials(self) -> None:
        compose = self.read("compose.yaml")
        env_example = self.read_root_env_example()
        self.assertIn("OAMB_MEM0_INSPECTOR_API_KEY", compose)
        self.assertIn("POSTGRES_HOST: mem0-postgres", compose)
        self.assertIn("POSTGRES_DB: postgres", compose)
        self.assertIn("POSTGRES_USER: oamb_mem0", compose)
        self.assertNotIn("QDRANT", compose)
        self.assertNotIn("QDRANT", env_example)
        self.assertNotIn("OAMB_RUNNER_POSTGRES", compose)

    def test_mem0_image_is_fixed_source_and_not_dev_reinstall(self) -> None:
        dockerfile = self.read("mem0/Dockerfile")
        self.assertIn(
            "python:3.12.11-slim-bookworm@sha256:519591d6871b7bc437060736b9f7456b8731f1499a57e22e6c285135ae657bf7",
            dockerfile,
        )
        self.assertIn("org.opencontainers.image.revision", dockerfile)
        self.assertIn("dc82354e143c2581d505d581a00286d6ef8c3605", dockerfile)
        self.assertNotIn("--force-reinstall", dockerfile)
        self.assertNotIn("--reload", dockerfile)
        self.assertNotRegex(dockerfile, r"(?m)^CMD .*pip install")
        self.assertNotIn("server_state.py", dockerfile)
        self.assertNotIn("server-overlay-sha256", dockerfile)
        self.assertIn("io.oamb.mem0.build-input-sha256", dockerfile)

    def test_mem0_uses_official_pgvector_config_without_overlay(self) -> None:
        bootstrap = self.read("mem0/bootstrap.sh")
        dockerignore = self.read(".dockerignore")
        operator = self.read("bin/provider-services")
        self.assertIn('vector_store: {provider: "pgvector"', bootstrap)
        self.assertIn('host: "mem0-postgres"', bootstrap)
        self.assertIn('dbname: "postgres"', bootstrap)
        self.assertIn('collection_name: "oamb_memories"', bootstrap)
        self.assertIn("embedding_model_dims: 1024", bootstrap)
        self.assertIn("hnsw: true", bootstrap)
        self.assertIn("diskann: false", bootstrap)
        self.assertNotIn("server_state.py", dockerignore)
        self.assertNotIn("MEM0_SERVER_OVERLAY_SHA256", operator)
        self.assertNotIn("MEM0_QDRANT_IMAGE_DIGEST", operator)

    def test_operator_surface_preserves_state(self) -> None:
        command = self.read("bin/provider-services")
        lifecycle = self.read("lib/lifecycle.sh")
        self.assertIn("doctor", command)
        self.assertIn("build", command)
        self.assertIn("verify --services", command)
        self.assertIn("verify --model-readiness", command)
        self.assertIn("stop", command)
        self.assertIn("active-operation", lifecycle)
        self.assertIn("provider-project.attestation", command)
        self.assertIn("verify_project_attestation", command)
        self.assertIn("provider-lifecycle.lock", command)
        self.assertIn("acquire_lifecycle_lock", command)
        self.assertIn("docker volume ls", command)
        self.assertIn("docker network ls", command)
        self.assertIn('--filter "publish=$port"', command)
        self.assertIn("org.opencontainers.image.revision", command)
        self.assertIn("ensure_image", command)
        self.assertIn('docker image inspect "$image"', command)
        self.assertIn("mem0_image_matches_pin", command)
        self.assertIn("mem0_build_input_sha256", command)
        self.assertIn("--build-arg MEM0_BUILD_INPUT_SHA256=", command)
        self.assertIn("/v1/projection?run_id=", command)
        self.assertIn("/chat/completions", command)
        self.assertIn("/embeddings", command)
        self.assertIn("dimensions: 1024", command)
        self.assertIn("/ready", command)
        self.assertIn('.version == "v0.4.16"', command)
        self.assertIn("openviking-storage-probe", command)
        self.assertIn("compose stop openviking", command)
        self.assertIn("compose start openviking", command)
        self.assertIn("--resolved-plan", command)
        self.assertIn("readiness_plan.py", command)
        self.assertIn("readiness_response.py", command)
        self.assertIn("environment_hash: $environment_hash", command)
        self.assertIn("model_calls_dispatched: 7", command)
        self.assertIn("MODEL_READINESS_CHAT_MAX_TOKENS=1024", command)
        self.assertIn("max_tokens: $max_tokens", command)
        self.assertNotIn("max_tokens: 16", command)
        self.assertIn("model-readiness-attempt.json", command)
        self.assertIn("billing_complete: false", command)
        self.assertIn("service_verification_receipt_sha256", command)
        self.assertNotIn('set -- "$service_receipt_directory"/*.json', command)
        self.assertLess(
            command.index('readiness_plan_temporary="$readiness_plan.partial.$$"'),
            command.index('create_model_readiness_attempt "$project"'),
        )
        self.assertLess(
            command.index("service_verification_receipt_sha256"),
            command.index('create_model_readiness_attempt "$project"'),
        )
        self.assertNotIn('find "$RUNTIME_DIR/service-verification-receipts"', command)
        self.assertIn("record_model_dispatch", command)
        self.assertIn("record_model_terminal", command)
        self.assertIn("restore_openviking_on_exit", command)
        self.assertIn("OPENVIKING_QUIESCED", command)
        all_text = "\n".join(
            path.read_text(encoding="utf-8")
            for path in ROOT.rglob("*")
            if path.is_file()
            and "__pycache__" not in path.parts
            and "tests" not in path.parts
            and ".build" not in path.parts
            and ".runtime" not in path.parts
            and path.name != ".env"
        )
        for destructive in (
            "down " + "-v",
            "docker rm " + "-f",
            "qdrant/" + "delete",
            "collections/" + "delete",
        ):
            self.assertNotIn(destructive, all_text)

    def test_model_readiness_uses_the_shared_host_embedding_resolver(self) -> None:
        command = self.read("bin/provider-services")
        resolver = ROOT / "lib" / "host_embedding.sh"

        self.assertTrue(resolver.is_file())
        self.assertIn('. "$ROOT/lib/host_embedding.sh"', command)
        self.assertIn("resolve_host_embedding_base", command)
        self.assertNotIn(
            'embedding_base="http://127.0.0.1:${embedding_base#http://host.docker.internal:}"',
            command,
        )

    def test_openviking_isolated_api_configuration(self) -> None:
        compose = self.read("compose.yaml")
        config = self.read("openviking/ov.conf")
        self.assertIn('OPENVIKING_WITH_BOT: "0"', compose)
        self.assertIn('"workspace": "/var/lib/openviking"', config)
        self.assertIn('"provider": "openai"', config)
        self.assertIn('"model": "${OAMB_EMBEDDING_MODEL}"', config)
        self.assertNotIn('"rerank"', config)
        self.assertNotIn("~/.openviking", compose)

    def test_provider_internal_thinking_effort_comes_from_plan_environment(self) -> None:
        compose = self.read("compose.yaml")
        mem0_bootstrap = self.read("mem0/bootstrap.sh")
        openviking_config = self.read("openviking/ov.conf")
        operator = self.read("bin/provider-services")
        plan_environment = self.read("lib/plan_environment.sh")

        self.assertIn(
            "HINDSIGHT_API_LLM_REASONING_EFFORT: ${OAMB_HINDSIGHT_LLM_REASONING_EFFORT:?required}",
            compose,
        )
        self.assertIn("OAMB_MEM0_LLM_REASONING_EFFORT", mem0_bootstrap)
        self.assertIn("reasoning_effort: $llm_effort", mem0_bootstrap)
        self.assertIn("is_reasoning_model: true", mem0_bootstrap)
        self.assertIn(
            '"extra_request_body": {"reasoning_effort": "${OAMB_OPENVIKING_VLM_REASONING_EFFORT}"}',
            openviking_config,
        )
        for variable in (
            "OAMB_HINDSIGHT_LLM_REASONING_EFFORT",
            "OAMB_MEM0_LLM_REASONING_EFFORT",
            "OAMB_OPENVIKING_VLM_REASONING_EFFORT",
        ):
            self.assertIn(variable, plan_environment)
        self.assertIn("reasoning_effort: $effort", operator)
        self.assertIn("hindsight-model-config.json", operator)
        self.assertIn("openviking-model-config.json", operator)
        self.assertNotIn("TARGET_PROVIDER_MODEL", operator)
        self.assertEqual(operator.count(".model == $target_model"), 3)
        self.assertEqual(operator.count("$target_effort"), 3)
        self.assertIn(".llm.config.model == $target_model", operator)

    def test_all_provider_internal_retry_paths_are_disabled_and_proven(self) -> None:
        compose = self.read("compose.yaml")
        openviking_config = self.read("openviking/ov.conf")
        retry_guard = self.read("retry-guard/sitecustomize.py")
        operator = self.read("bin/provider-services")

        self.assertIn('HINDSIGHT_API_LLM_MAX_RETRIES: "0"', compose)
        self.assertIn('HINDSIGHT_API_RETAIN_LLM_MAX_RETRIES: "0"', compose)
        self.assertIn('HINDSIGHT_API_WORKER_MAX_RETRIES: "0"', compose)
        self.assertEqual(compose.count("./retry-guard/sitecustomize.py:"), 3)
        self.assertEqual(openviking_config.count('"max_retries": 0'), 2)
        self.assertIn('kwargs["max_retries"] = INTERNAL_RETRY_COUNT', retry_guard)
        self.assertIn("module._MEMORY_EXTRACTION_MAX_RETRIES =", retry_guard)
        for filename in (
            "hindsight-retry-config.json",
            "mem0-retry-config.json",
            "openviking-retry-config.json",
        ):
            self.assertIn(filename, operator)

    def test_bootstrap_treats_dotenv_as_data(self) -> None:
        mem0 = self.read("mem0/bootstrap.sh")
        openviking = self.read("openviking/bootstrap.sh")
        operator = self.read("bin/provider-services")
        for script in (mem0, openviking):
            self.assertNotIn('. "$ENV_FILE"', script)
            self.assertIn("env_value()", script)
        self.assertIn('read_runtime_env_value "$ENV_FILE" "$1"', mem0)
        self.assertNotIn("qwen3-embedding:0.6b", mem0)
        self.assertIn('read_env_value "$ENV_FILE" "$1"', openviking)
        self.assertIn('. "$ROOT/lib/compose.sh"', mem0)
        self.assertIn('. "$ROOT/lib/compose.sh"', operator)
        self.assertNotIn("docker compose", mem0)

    def test_openviking_bootstrap_proves_full_non_root_identity(self) -> None:
        script = self.read("openviking/bootstrap.sh")
        self.assertIn('"$BASE_URL/health"', script)
        self.assertIn('.role == "admin"', script)
        self.assertIn(".account_id == $account", script)
        self.assertIn(".user_id == $user", script)

    def test_example_env_has_placeholders_not_credentials(self) -> None:
        example = self.read_root_env_example()
        values = dict(
            line.split("=", 1) for line in example.splitlines() if line and not line.startswith("#")
        )
        for name in (
            "OAMB_PROVIDER_PROJECT",
            "OAMB_HINDSIGHT_LLM_API_KEY",
            "OAMB_MEM0_LLM_API_KEY",
            "OAMB_OPENVIKING_VLM_API_KEY",
            "OAMB_MEM0_INSPECTOR_API_KEY",
        ):
            self.assertRegex(example, rf"(?m)^{name}=.*$")
        for name in (
            "OMBA_ANSWER_LLM",
            "OMBA_ANSWER_MODEL",
            "OMBA_JUDGE_LLM",
            "OMBA_JUDGE_MODEL",
            "OPENAI_BASE_URL",
            "OPENAI_API_KEY",
            "OAMB_EMBEDDING_MODEL",
            "OAMB_HINDSIGHT_LLM_PROVIDER",
            "OAMB_HINDSIGHT_LLM_MODEL",
            "OAMB_MEM0_LLM_MODEL",
            "OAMB_OPENVIKING_VLM_PROVIDER",
            "OAMB_OPENVIKING_VLM_MODEL",
        ):
            self.assertNotIn(name, values)
        for name in (
            "OAMB_HINDSIGHT_LLM_BASE_URL",
            "OAMB_MEM0_LLM_BASE_URL",
            "OAMB_OPENVIKING_VLM_BASE_URL",
        ):
            self.assertRegex(example, rf"(?m)^{name}=change-me$")
        self.assertNotRegex(example, r"sk-[A-Za-z0-9]{12,}")


if __name__ == "__main__":
    unittest.main()
