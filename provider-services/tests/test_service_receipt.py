from __future__ import annotations

import hashlib
import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "lib" / "service_receipt.py"
PROFILE_FILES = {
    "hindsight-rest-v1": (
        "hindsight-health.json",
        "hindsight-version.json",
        "hindsight-model-config.json",
    ),
    "mem0-rest-v1": (
        "mem0-openapi.json",
        "mem0-empty-projection.json",
        "mem0-config-redacted.json",
    ),
    "openviking-rest-v1": (
        "openviking-health.json",
        "openviking-auth-identity.json",
        "openviking-storage.json",
        "openviking-model-config.json",
    ),
}


def load_module():
    spec = importlib.util.spec_from_file_location("service_receipt", MODULE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load service receipt module")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_proof_files(directory: Path, profile_id: str) -> dict[str, bytes]:
    expected: dict[str, bytes] = {}
    for ordinal, filename in enumerate(PROFILE_FILES[profile_id], start=1):
        content = json.dumps(
            {"ordinal": ordinal, "status": "ok"},
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        (directory / filename).write_bytes(content)
        expected[filename] = content
    return expected


def write_all_proof_files(directory: Path) -> None:
    for profile_id in PROFILE_FILES:
        write_proof_files(directory, profile_id)


class ServiceReceiptTests(unittest.TestCase):
    def test_profile_proof_seal_allows_a_system_symlink_above_the_trusted_root(self) -> None:
        module = load_module()
        with tempfile.TemporaryDirectory(dir="/var/tmp") as temporary:
            root = Path(temporary)
            attempt = root / "attempt"
            attempt.mkdir()
            write_proof_files(attempt, "hindsight-rest-v1")

            manifest_hash = module.seal_profile_proof_manifest(
                root / "receipts" / "proofs",
                profile_id="hindsight-rest-v1",
                proof_directory=attempt,
                trusted_root=root,
            )

            self.assertRegex(manifest_hash, r"^[0-9a-f]{64}$")

    def test_same_proof_bytes_in_random_directories_have_one_manifest_hash(self) -> None:
        module = load_module()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            first_attempt = root / "attempt.random-a"
            second_attempt = root / "attempt.random-b"
            first_attempt.mkdir()
            second_attempt.mkdir()
            expected = write_proof_files(first_attempt, "hindsight-rest-v1")
            write_proof_files(second_attempt, "hindsight-rest-v1")
            proof_store = root / "receipts" / "proofs"

            first_hash = module.seal_profile_proof_manifest(
                proof_store,
                profile_id="hindsight-rest-v1",
                proof_directory=first_attempt,
                trusted_root=root,
            )
            second_hash = module.seal_profile_proof_manifest(
                proof_store,
                profile_id="hindsight-rest-v1",
                proof_directory=second_attempt,
                trusted_root=root,
            )

            self.assertEqual(first_hash, second_hash)
            manifest_path = proof_store / "manifests" / f"{first_hash}.json"
            manifest_bytes = manifest_path.read_bytes()
            self.assertEqual(hashlib.sha256(manifest_bytes).hexdigest(), first_hash)
            self.assertNotIn(str(first_attempt).encode(), manifest_bytes)
            self.assertNotIn(str(second_attempt).encode(), manifest_bytes)
            manifest = json.loads(manifest_bytes)
            self.assertEqual(manifest["profile_id"], "hindsight-rest-v1")
            self.assertEqual(
                tuple(entry["relative_path"] for entry in manifest["files"]),
                PROFILE_FILES["hindsight-rest-v1"],
            )
            for entry in manifest["files"]:
                filename = entry["relative_path"]
                blob = proof_store / "blobs" / entry["sha256"]
                self.assertEqual(blob.read_bytes(), expected[filename])
                self.assertEqual(entry["byte_count"], len(expected[filename]))

    def test_profile_proof_seal_redacts_provider_authorities(self) -> None:
        module = load_module()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            attempt = root / "attempt"
            attempt.mkdir()
            write_proof_files(attempt, "mem0-rest-v1")
            (attempt / "mem0-config-redacted.json").write_text(
                json.dumps(
                    {
                        "vector_store": {
                            "provider": "qdrant",
                            "config": {"url": "http://mem0-qdrant:6333"},
                        },
                        "llm": {
                            "provider": "openai",
                            "config": {"openai_base_url": "https://models.example/v1"},
                        },
                    }
                ),
                encoding="utf-8",
            )

            manifest_hash = module.seal_profile_proof_manifest(
                root / "receipts" / "proofs",
                profile_id="mem0-rest-v1",
                proof_directory=attempt,
                trusted_root=root,
            )

            manifest = json.loads(
                (
                    root
                    / "receipts"
                    / "proofs"
                    / "manifests"
                    / f"{manifest_hash}.json"
                ).read_bytes()
            )
            config_entry = next(
                entry
                for entry in manifest["files"]
                if entry["relative_path"] == "mem0-config-redacted.json"
            )
            sealed = (
                root / "receipts" / "proofs" / "blobs" / config_entry["sha256"]
            ).read_bytes()
            self.assertNotIn(b"://", sealed)
            self.assertEqual(
                json.loads(sealed)["vector_store"]["config"]["url"],
                "[redacted]",
            )

    def test_new_nested_proof_directories_fsync_each_parent_in_order(self) -> None:
        module = load_module()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            attempt = root / "attempt"
            attempt.mkdir()
            write_proof_files(attempt, "hindsight-rest-v1")
            proof_store = root / "first" / "second" / "proofs"
            observed: list[Path] = []

            with mock.patch.object(module, "_fsync_directory", side_effect=observed.append):
                module.seal_profile_proof_manifest(
                    proof_store,
                    profile_id="hindsight-rest-v1",
                    proof_directory=attempt,
                    trusted_root=root,
                )

            self.assertEqual(
                observed[:4],
                [root, root / "first", root / "first" / "second", proof_store],
            )

    def test_profile_proof_seal_rejects_a_symlink_in_output_ancestors(self) -> None:
        module = load_module()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            attempt = root / "attempt"
            attempt.mkdir()
            write_proof_files(attempt, "hindsight-rest-v1")
            real_parent = root / "real-parent"
            real_parent.mkdir()
            linked_parent = root / "linked-parent"
            linked_parent.symlink_to(real_parent, target_is_directory=True)

            with self.assertRaisesRegex(ValueError, "symlink|symbolic link"):
                module.seal_profile_proof_manifest(
                    linked_parent / "proofs",
                    profile_id="hindsight-rest-v1",
                    proof_directory=attempt,
                    trusted_root=root,
                )

    def test_profile_proof_seal_rejects_a_leaf_symlink(self) -> None:
        module = load_module()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            attempt = root / "attempt"
            attempt.mkdir()
            write_proof_files(attempt, "hindsight-rest-v1")
            target = root / "outside.json"
            target.write_text('{"status":"ok"}', encoding="utf-8")
            (attempt / "hindsight-health.json").unlink()
            (attempt / "hindsight-health.json").symlink_to(target)

            with self.assertRaisesRegex(ValueError, "symlink|regular proof file"):
                module.seal_profile_proof_manifest(
                    root / "receipts" / "proofs",
                    profile_id="hindsight-rest-v1",
                    proof_directory=attempt,
                    trusted_root=root,
                )

    def test_profile_proof_seal_rejects_a_symlink_in_input_ancestors(self) -> None:
        module = load_module()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            real_parent = root / "real-parent"
            real_parent.mkdir()
            attempt = real_parent / "attempt"
            attempt.mkdir()
            write_proof_files(attempt, "hindsight-rest-v1")
            linked_parent = root / "linked-parent"
            linked_parent.symlink_to(real_parent, target_is_directory=True)

            with self.assertRaisesRegex(ValueError, "symlink|symbolic link"):
                module.seal_profile_proof_manifest(
                    root / "receipts" / "proofs",
                    profile_id="hindsight-rest-v1",
                    proof_directory=linked_parent / "attempt",
                    trusted_root=root,
                )

    def test_receipt_keeps_five_gates_separate_and_references_manifests(self) -> None:
        module = load_module()

        receipt = module.build_service_verification_receipt(
            provider_project="oamb-providers-test-alpha",
            project_attestation_sha256="a" * 64,
            verified_at_utc="2026-08-27T12:00:00Z",
            profile_proof_manifest_sha256={
                "hindsight-rest-v1": "b" * 64,
                "mem0-rest-v1": "c" * 64,
                "openviking-rest-v1": "d" * 64,
            },
        )

        self.assertEqual(
            tuple(profile["profile_id"] for profile in receipt["profiles"]),
            ("hindsight-rest-v1", "mem0-rest-v1", "openviking-rest-v1"),
        )
        for profile in receipt["profiles"]:
            self.assertRegex(profile["proof_manifest_sha256"], r"^[0-9a-f]{64}$")
            self.assertEqual(profile["liveness"], "passed")
            self.assertEqual(profile["storage_configuration"], "passed")
            self.assertEqual(profile["runtime_identity"], "passed")
            self.assertEqual(profile["model_readiness"], "not_run")
            self.assertEqual(profile["memory_conformance"], "not_run")

    def test_receipt_is_content_addressed_create_only_and_contains_no_private_path(self) -> None:
        module = load_module()
        receipt = module.build_service_verification_receipt(
            provider_project="oamb-providers-test-alpha",
            project_attestation_sha256="a" * 64,
            verified_at_utc="2026-08-27T12:00:00Z",
            profile_proof_manifest_sha256={
                "hindsight-rest-v1": "b" * 64,
                "mem0-rest-v1": "c" * 64,
                "openviking-rest-v1": "d" * 64,
            },
        )
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary).resolve()

            first = module.seal_service_verification_receipt(output, receipt)
            second = module.seal_service_verification_receipt(output, receipt)

            self.assertEqual(first, second)
            parsed = json.loads(first.read_text(encoding="utf-8"))
            self.assertEqual(parsed, receipt)
            self.assertNotIn(temporary, first.read_text(encoding="utf-8"))
            self.assertRegex(first.name, r"^[0-9a-f]{64}\.json$")

    def test_cli_seals_proof_store_and_one_receipt_for_operator(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            proof_directory = root / "attempt.random"
            proof_directory.mkdir()
            write_all_proof_files(proof_directory)
            output_directory = root / "receipts"
            result = subprocess.run(
                [
                    sys.executable,
                    str(MODULE_PATH),
                    "--output-dir",
                    str(output_directory),
                    "--proof-dir",
                    str(proof_directory),
                    "--trusted-root",
                    str(root),
                    "--project",
                    "oamb-providers-test-alpha",
                    "--attestation-sha256",
                    "a" * 64,
                    "--verified-at-utc",
                    "2026-08-27T12:00:00Z",
                ],
                check=False,
                capture_output=True,
                text=True,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            receipt_path = Path(result.stdout.strip())
            self.assertTrue(receipt_path.is_file())
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
            for profile in receipt["profiles"]:
                manifest_hash = profile["proof_manifest_sha256"]
                self.assertTrue(
                    (output_directory / "proofs" / "manifests" / f"{manifest_hash}.json").is_file()
                )


if __name__ == "__main__":
    unittest.main()
