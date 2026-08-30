from __future__ import annotations

import os
import subprocess
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class EnvLoaderTests(unittest.TestCase):
    def test_file_mode_ignores_failed_stat_stdout_before_portable_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            env_file = directory / ".env"
            env_file.write_text("OAMB_KEY=value\n", encoding="utf-8")
            env_file.chmod(0o600)
            fake_stat = directory / "stat"
            fake_stat.write_text(
                "#!/bin/sh\n"
                "if [ \"$1\" = \"-f\" ]; then\n"
                "  printf 'unexpected filesystem details\\n'\n"
                "  exit 1\n"
                "fi\n"
                "printf '600\\n'\n",
                encoding="utf-8",
            )
            fake_stat.chmod(0o755)
            command = '. "$1"; read_file_mode "$2"'
            result = subprocess.run(
                ["sh", "-c", command, "sh", str(ROOT / "lib" / "env.sh"), str(env_file)],
                check=False,
                capture_output=True,
                text=True,
                env={**os.environ, "PATH": f"{directory}:{os.environ['PATH']}"},
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(result.stdout, "600\n")

    def test_shell_syntax_in_value_is_never_executed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            env_file = directory / ".env"
            marker = directory / "executed"
            hostile = f"$(touch {marker})"
            env_file.write_text(f"HOSTILE={hostile}\n", encoding="utf-8")
            command = '. "$1"; read_env_value "$2" HOSTILE'
            result = subprocess.run(
                ["sh", "-c", command, "sh", str(ROOT / "lib" / "env.sh"), str(env_file)],
                check=True,
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.stdout.rstrip("\n"), hostile)
            self.assertFalse(marker.exists())

    def test_validation_rejects_duplicate_keys_and_compose_syntax(self) -> None:
        for content in (
            "OAMB_KEY=first\nOAMB_KEY=second\n",
            'OAMB_KEY="quoted"\n',
            "OAMB_KEY=${EXPANDED}\n",
        ):
            with self.subTest(content=content), tempfile.TemporaryDirectory() as temporary:
                env_file = Path(temporary) / ".env"
                env_file.write_text(content, encoding="utf-8")
                command = '. "$1"; validate_env_file "$2"'
                result = subprocess.run(
                    ["sh", "-c", command, "sh", str(ROOT / "lib" / "env.sh"), str(env_file)],
                    check=False,
                    capture_output=True,
                    text=True,
                )
                self.assertNotEqual(result.returncode, 0)
                self.assertNotIn("not found", result.stderr)

    def test_validation_accepts_the_tracked_simple_value_grammar(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            env_file = Path(temporary) / ".env"
            env_file.write_text(
                "OAMB_URL=https://example.invalid/v1\nOAMB_PATH=/tmp/path with spaces\n",
                encoding="utf-8",
            )
            command = '. "$1"; validate_env_file "$2"'
            subprocess.run(
                ["sh", "-c", command, "sh", str(ROOT / "lib" / "env.sh"), str(env_file)],
                check=True,
                capture_output=True,
                text=True,
            )


if __name__ == "__main__":
    unittest.main()
