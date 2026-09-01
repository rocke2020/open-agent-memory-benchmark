from __future__ import annotations

import ast
import hashlib
import re
import unittest
from copy import deepcopy
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class ServiceBundleContractTests(unittest.TestCase):
    def read(self, relative_path: str) -> str:
        path = ROOT / relative_path
        self.assertTrue(path.is_file(), f"missing required file: {relative_path}")
        return path.read_text(encoding="utf-8")

    def test_release_and_image_pins_are_immutable(self) -> None:
        versions = self.read("versions.env")
        expected = {
            "HINDSIGHT_VERSION": "0.9.2",
            "HINDSIGHT_COMMIT": "ebad478240d3171bb88201ececda5e8d9883d22d",
            "HINDSIGHT_IMAGE_DIGEST": "sha256:7635a15739361dbdf221ba796ad25a813f876144fe113022eea8e26cb6ee75e7",
            "MEM0_VERSION": "2.0.19",
            "MEM0_COMMIT": "dc82354e143c2581d505d581a00286d6ef8c3605",
            "MEM0_SERVER_OVERLAY_SHA256": "fc6d575c66f8c27816cc8f293849a0ce1612f0f5048e3232a867593755ede2d8",
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
        self.assertEqual(
            hashlib.sha256((ROOT / "mem0" / "server_state.py").read_bytes()).hexdigest(),
            parsed["MEM0_SERVER_OVERLAY_SHA256"],
        )

    def test_compose_has_six_long_running_services_and_read_only_storage_probe(self) -> None:
        compose = self.read("compose.yaml")
        for service in (
            "hindsight",
            "mem0-postgres",
            "mem0-qdrant",
            "mem0",
            "mem0-inspector",
            "openviking",
        ):
            self.assertRegex(compose, rf"(?m)^  {re.escape(service)}:$")
        self.assertEqual(compose.count("    healthcheck:\n"), 6)
        self.assertRegex(compose, r"(?m)^  openviking-storage-probe:$")
        probe = compose.split("  openviking-storage-probe:\n", 1)[1].split(
            "\nvolumes:\n", 1
        )[0]
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

    def test_runner_never_receives_qdrant_backend_key(self) -> None:
        compose = self.read("compose.yaml")
        self.assertIn("OAMB_MEM0_INSPECTOR_API_KEY", compose)
        self.assertIn("QDRANT_API_KEY", compose)
        self.assertNotIn("OAMB_RUNNER_QDRANT", compose)

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
        self.assertIn("COPY mem0/server_state.py /app/server_state.py", dockerfile)
        self.assertIn("io.oamb.mem0.build-input-sha256", dockerfile)

    def test_mem0_provider_switch_replaces_incompatible_config(self) -> None:
        source = self.read("mem0/server_state.py")
        tree = ast.parse(source)
        merge_function = next(
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "_merge_config"
        )
        namespace = {"Any": object, "Dict": dict, "deepcopy": deepcopy}
        exec(compile(ast.Module(body=[merge_function], type_ignores=[]), "server_state.py", "exec"), namespace)
        merge = namespace["_merge_config"]
        base = {
            "vector_store": {
                "provider": "pgvector",
                "config": {"dbname": "postgres", "user": "oamb"},
            }
        }
        switched = merge(
            base,
            {"vector_store": {"provider": "qdrant", "config": {"url": "http://qdrant"}}},
        )
        self.assertEqual(
            switched["vector_store"],
            {"provider": "qdrant", "config": {"url": "http://qdrant"}},
        )
        same_provider = merge(
            switched,
            {"vector_store": {"provider": "qdrant", "config": {"api_key": "secret"}}},
        )
        self.assertEqual(
            same_provider["vector_store"]["config"],
            {"url": "http://qdrant", "api_key": "secret"},
        )

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
        self.assertIn('org.opencontainers.image.revision', command)
        self.assertIn('ensure_image', command)
        self.assertIn('docker image inspect "$image"', command)
        self.assertIn('mem0_image_matches_pin', command)
        self.assertIn('mem0_build_input_sha256', command)
        self.assertIn('--build-arg MEM0_BUILD_INPUT_SHA256=', command)
        self.assertIn('/v1/projection?run_id=', command)
        self.assertIn('/chat/completions', command)
        self.assertIn('/embeddings', command)
        self.assertIn('dimensions: 1024', command)
        self.assertIn('/ready', command)
        self.assertIn('.version == "v0.4.16"', command)
        self.assertIn('openviking-storage-probe', command)
        self.assertIn('compose stop openviking', command)
        self.assertIn('compose start openviking', command)
        self.assertIn('oamb-provider-model-readiness-budget-v1', command)
        self.assertIn('model-readiness-attempt.json', command)
        self.assertIn('billing_complete: false', command)
        self.assertIn('record_model_dispatch', command)
        self.assertIn('record_model_terminal', command)
        self.assertIn('restore_openviking_on_exit', command)
        self.assertIn('OPENVIKING_QUIESCED', command)
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

    def test_openviking_isolated_api_configuration(self) -> None:
        compose = self.read("compose.yaml")
        config = self.read("openviking/ov.conf")
        self.assertIn('OPENVIKING_WITH_BOT: "0"', compose)
        self.assertIn('"workspace": "/var/lib/openviking"', config)
        self.assertIn('"provider": "openai"', config)
        self.assertIn('"model": "${OAMB_EMBEDDING_MODEL}"', config)
        self.assertNotIn('"rerank"', config)
        self.assertNotIn("~/.openviking", compose)

    def test_all_provider_internal_model_calls_pin_low_reasoning_effort(self) -> None:
        compose = self.read("compose.yaml")
        mem0_bootstrap = self.read("mem0/bootstrap.sh")
        openviking_config = self.read("openviking/ov.conf")
        operator = self.read("bin/provider-services")

        self.assertIn('HINDSIGHT_API_LLM_REASONING_EFFORT: "low"', compose)
        self.assertIn('reasoning_effort: "low"', mem0_bootstrap)
        self.assertIn('is_reasoning_model: true', mem0_bootstrap)
        self.assertIn('"extra_request_body": {"reasoning_effort": "low"}', openviking_config)
        self.assertIn('reasoning_effort: "low"', operator)
        self.assertIn("hindsight-model-config.json", operator)
        self.assertIn("openviking-model-config.json", operator)
        self.assertEqual(operator.count('--arg target_model "$TARGET_PROVIDER_MODEL"'), 3)
        self.assertEqual(operator.count("'.model == $target_model"), 2)
        self.assertIn(".llm.config.model == $target_model", operator)

    def test_bootstrap_treats_dotenv_as_data(self) -> None:
        for script_name in ("mem0/bootstrap.sh", "openviking/bootstrap.sh"):
            script = self.read(script_name)
            self.assertNotIn('. "$ENV_FILE"', script)
            self.assertIn("env_value()", script)
            self.assertIn('read_env_value "$ENV_FILE" "$1"', script)

    def test_openviking_bootstrap_proves_full_non_root_identity(self) -> None:
        script = self.read("openviking/bootstrap.sh")
        self.assertIn('"$BASE_URL/health"', script)
        self.assertIn('.role == "admin"', script)
        self.assertIn('.account_id == $account', script)
        self.assertIn('.user_id == $user', script)

    def test_example_env_has_placeholders_not_credentials(self) -> None:
        example = self.read(".env.example")
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
            "OAMB_HINDSIGHT_LLM_MODEL",
            "OAMB_MEM0_LLM_MODEL",
            "OAMB_OPENVIKING_VLM_MODEL",
        ):
            self.assertRegex(example, rf"(?m)^{name}=deepseek-v4-flash$")
        for name in (
            "OAMB_HINDSIGHT_LLM_BASE_URL",
            "OAMB_MEM0_LLM_BASE_URL",
            "OAMB_OPENVIKING_VLM_BASE_URL",
        ):
            self.assertRegex(example, rf"(?m)^{name}=change-me$")
        self.assertNotRegex(example, r"sk-[A-Za-z0-9]{12,}")
        self.assertEqual(values["OAMB_HINDSIGHT_LLM_MODEL"], "deepseek-v4-flash")
        self.assertEqual(values["OAMB_MEM0_LLM_MODEL"], "deepseek-v4-flash")
        self.assertEqual(values["OAMB_OPENVIKING_VLM_MODEL"], "deepseek-v4-flash")


if __name__ == "__main__":
    unittest.main()
