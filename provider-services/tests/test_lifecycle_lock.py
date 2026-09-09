from __future__ import annotations

import subprocess
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class LifecycleLockTests(unittest.TestCase):
    def run_acquire(self, runtime: Path) -> subprocess.CompletedProcess[str]:
        program = """
set -eu
die() { printf '%s\\n' "$*" >&2; exit 1; }
RUNTIME_DIR=$1
LIFECYCLE_LOCK="$RUNTIME_DIR/provider-lifecycle.lock"
LIFECYCLE_LOCK_HELD=false
. "$2"
acquire_lifecycle_lock
printf 'acquired\\n'
"""
        return subprocess.run(
            ["sh", "-c", program, "sh", str(runtime), str(ROOT / "lib" / "lifecycle.sh")],
            capture_output=True,
            text=True,
        )

    def run_stopped_cleanup(self, runtime: Path) -> subprocess.CompletedProcess[str]:
        program = """
set -eu
die() { printf '%s\n' "$*" >&2; exit 1; }
RUNTIME_DIR=$1
LIFECYCLE_LOCK="$RUNTIME_DIR/provider-lifecycle.lock"
LIFECYCLE_LOCK_HELD=false
. "$2"
acquire_lifecycle_stop_lock
clear_stopped_provider_lifecycle
printf 'stopped-clean\n'
"""
        return subprocess.run(
            ["sh", "-c", program, "sh", str(runtime), str(ROOT / "lib" / "lifecycle.sh")],
            capture_output=True,
            text=True,
        )

    def test_stop_lock_clears_only_ephemeral_lifecycle_markers(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            runtime = Path(temporary)
            domain = runtime / "lifecycle-domains" / "mem0"
            attempts = domain / "active-provider-attempts"
            attempts.mkdir(parents=True)
            (domain / "active-operation").write_text("{}\n", encoding="utf-8")
            (attempts / f"{'a' * 64}.json").write_text("{}\n", encoding="utf-8")
            preserved = domain / "preserved-diagnostic.json"
            preserved.write_text("{}\n", encoding="utf-8")

            result = self.run_stopped_cleanup(runtime)

            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(result.stdout, "stopped-clean\n")
            self.assertFalse((domain / "active-operation").exists())
            self.assertFalse(tuple(attempts.glob("*.json")))
            self.assertTrue(preserved.is_file())
            self.assertFalse((runtime / "provider-lifecycle.lock").exists())

    def test_stopped_cleanup_rejects_lifecycle_domains_symlink(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            runtime = root / "runtime"
            runtime.mkdir()
            external = root / "external" / "mem0"
            external.mkdir(parents=True)
            marker = external / "active-operation"
            marker.write_text("{}\n", encoding="utf-8")
            (runtime / "lifecycle-domains").symlink_to(
                external.parent,
                target_is_directory=True,
            )

            result = self.run_stopped_cleanup(runtime)

            self.assertNotEqual(result.returncode, 0)
            self.assertIn("lifecycle domains path is unsafe", result.stderr)
            self.assertTrue(marker.is_file())
            self.assertFalse((runtime / "provider-lifecycle.lock").exists())

    def test_successful_lock_is_released_on_exit(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            runtime = Path(temporary)
            result = self.run_acquire(runtime)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(result.stdout, "acquired\n")
            self.assertFalse((runtime / "provider-lifecycle.lock").exists())

    def test_benchmark_active_operation_fails_and_releases_lifecycle_lock(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            runtime = Path(temporary)
            (runtime / "active-operation").write_text(
                '{"kind":"benchmark_run"}\n', encoding="utf-8"
            )
            result = self.run_acquire(runtime)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("active OAMB provider operation", result.stderr)
            self.assertFalse((runtime / "provider-lifecycle.lock").exists())

    def test_conformance_active_operation_fails_between_provider_attempts(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            runtime = Path(temporary)
            (runtime / "active-operation").write_text(
                '{"kind":"memory_conformance"}\n', encoding="utf-8"
            )
            result = self.run_acquire(runtime)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("active OAMB provider operation", result.stderr)
            self.assertFalse((runtime / "provider-lifecycle.lock").exists())

    def test_legacy_active_run_lease_fails_and_releases_lifecycle_lock(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            runtime = Path(temporary)
            (runtime / "active-run-lease").write_text("legacy\n", encoding="utf-8")

            result = self.run_acquire(runtime)

            self.assertNotEqual(result.returncode, 0)
            self.assertIn("legacy active OAMB run lease", result.stderr)
            self.assertFalse((runtime / "provider-lifecycle.lock").exists())

    def test_dispatched_provider_attempt_fails_and_releases_lifecycle_lock(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            runtime = Path(temporary)
            (runtime / "active-provider-attempt").write_text(
                '{"schema_name":"attempt_intent_record","schema_version":1}\n',
                encoding="utf-8",
            )
            result = self.run_acquire(runtime)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("active provider attempt", result.stderr)
            self.assertFalse((runtime / "provider-lifecycle.lock").exists())

    def test_per_attempt_pointer_fails_and_releases_lifecycle_lock(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            runtime = Path(temporary)
            attempts = runtime / "active-provider-attempts"
            attempts.mkdir()
            (attempts / f"{'a' * 64}.json").write_text("{}\n", encoding="utf-8")

            result = self.run_acquire(runtime)

            self.assertNotEqual(result.returncode, 0)
            self.assertIn("active provider attempt", result.stderr)
            self.assertFalse((runtime / "provider-lifecycle.lock").exists())

    def test_provider_domain_active_operation_blocks_stack_mutation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            runtime = Path(temporary)
            domain = runtime / "lifecycle-domains" / "mem0-rest-v1"
            domain.mkdir(parents=True)
            (domain / "active-operation").write_text('{"kind":"benchmark_run"}\n', encoding="utf-8")

            result = self.run_acquire(runtime)

            self.assertNotEqual(result.returncode, 0)
            self.assertIn("active OAMB provider operation", result.stderr)
            self.assertFalse((runtime / "provider-lifecycle.lock").exists())

    def test_provider_domain_active_attempt_blocks_stack_mutation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            runtime = Path(temporary)
            attempts = (
                runtime
                / "lifecycle-domains"
                / "openviking-session-rest-v1"
                / "active-provider-attempts"
            )
            attempts.mkdir(parents=True)
            (attempts / f"{'a' * 64}.json").write_text("{}\n", encoding="utf-8")

            result = self.run_acquire(runtime)

            self.assertNotEqual(result.returncode, 0)
            self.assertIn("active provider attempt", result.stderr)
            self.assertFalse((runtime / "provider-lifecycle.lock").exists())

    def test_provider_domain_admission_lock_blocks_stack_mutation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            runtime = Path(temporary)
            domain_lock = (
                runtime / "lifecycle-domains" / "hindsight-rest-v1" / "provider-lifecycle.lock"
            )
            domain_lock.mkdir(parents=True)

            result = self.run_acquire(runtime)

            self.assertNotEqual(result.returncode, 0)
            self.assertIn("provider lifecycle domain admission", result.stderr)
            self.assertFalse((runtime / "provider-lifecycle.lock").exists())

    def test_empty_provider_domains_do_not_block_stack_mutation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            runtime = Path(temporary)
            for provider in ("hindsight-rest-v1", "mem0-rest-v1"):
                (runtime / "lifecycle-domains" / provider).mkdir(parents=True)

            result = self.run_acquire(runtime)

            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(result.stdout, "acquired\n")
            self.assertFalse((runtime / "provider-lifecycle.lock").exists())

    def test_symbolic_lifecycle_domains_path_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            runtime = Path(temporary)
            target = runtime / "elsewhere"
            target.mkdir(parents=True)
            (runtime / "lifecycle-domains").symlink_to(target, target_is_directory=True)

            result = self.run_acquire(runtime)

            self.assertNotEqual(result.returncode, 0)
            self.assertIn("lifecycle domains path is unsafe", result.stderr)
            self.assertFalse((runtime / "provider-lifecycle.lock").exists())

    def test_dangling_root_operation_pointer_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            runtime = Path(temporary)
            (runtime / "active-operation").symlink_to(runtime / "missing-operation")

            result = self.run_acquire(runtime)

            self.assertNotEqual(result.returncode, 0)
            self.assertIn("active OAMB provider operation", result.stderr)
            self.assertFalse((runtime / "provider-lifecycle.lock").exists())

    def test_dangling_domain_attempt_pointer_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            runtime = Path(temporary)
            domain = runtime / "lifecycle-domains" / "mem0-rest-v1"
            domain.mkdir(parents=True)
            (domain / "active-provider-attempt").symlink_to(domain / "missing-attempt")

            result = self.run_acquire(runtime)

            self.assertNotEqual(result.returncode, 0)
            self.assertIn("active provider attempt", result.stderr)
            self.assertFalse((runtime / "provider-lifecycle.lock").exists())

    def test_symbolic_runtime_path_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            target = root / "target"
            target.mkdir()
            runtime = root / "runtime"
            runtime.symlink_to(target, target_is_directory=True)

            result = self.run_acquire(runtime)

            self.assertNotEqual(result.returncode, 0)
            self.assertIn("provider runtime path is unsafe", result.stderr)
            self.assertFalse((target / "provider-lifecycle.lock").exists())

    def test_unreadable_domain_attempts_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            runtime = Path(temporary)
            attempts = runtime / "lifecycle-domains" / "mem0-rest-v1" / "active-provider-attempts"
            attempts.mkdir(parents=True)
            (attempts / f"{'a' * 64}.json").write_text("{}\n", encoding="utf-8")
            attempts.chmod(0)
            try:
                result = self.run_acquire(runtime)
            finally:
                attempts.chmod(0o700)

            self.assertNotEqual(result.returncode, 0)
            self.assertIn("active provider attempts path is unsafe", result.stderr)
            self.assertFalse((runtime / "provider-lifecycle.lock").exists())

    def test_existing_lifecycle_lock_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            runtime = Path(temporary)
            (runtime / "provider-lifecycle.lock").mkdir()
            result = self.run_acquire(runtime)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("another provider lifecycle operation", result.stderr)


if __name__ == "__main__":
    unittest.main()
