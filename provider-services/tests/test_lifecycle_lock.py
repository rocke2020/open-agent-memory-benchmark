from __future__ import annotations

import subprocess
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class LifecycleLockTests(unittest.TestCase):
    def run_acquire(self, runtime: Path) -> subprocess.CompletedProcess[str]:
        program = """
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

    def test_successful_lock_is_released_on_exit(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            runtime = Path(temporary)
            result = self.run_acquire(runtime)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(result.stdout, "acquired\n")
            self.assertFalse((runtime / "provider-lifecycle.lock").exists())

    def test_active_run_fails_and_releases_lifecycle_lock(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            runtime = Path(temporary)
            (runtime / "active-run-lease").write_text("active\n", encoding="utf-8")
            result = self.run_acquire(runtime)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("active OAMB run lease", result.stderr)
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
