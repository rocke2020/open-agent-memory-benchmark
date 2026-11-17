from __future__ import annotations

import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path


class OamBuildImageTests(unittest.TestCase):
    def test_source_image_uses_committed_tree_and_revision_identity(self) -> None:
        script = Path(__file__).resolve().parents[1] / "oam/build-image.sh"
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            source.mkdir()
            subprocess.run(["git", "init", "-q", str(source)], check=True)
            (source / "evidence.txt").write_text("committed")
            subprocess.run(["git", "-C", str(source), "add", "."], check=True)
            subprocess.run(
                [
                    "git",
                    "-C",
                    str(source),
                    "-c",
                    "user.name=Test",
                    "-c",
                    "user.email=test@example.invalid",
                    "commit",
                    "-qm",
                    "fixture",
                ],
                check=True,
            )
            revision = subprocess.check_output(
                ["git", "-C", str(source), "rev-parse", "HEAD"], text=True
            ).strip()
            (source / "evidence.txt").write_text("uncommitted")
            (source / "private.txt").write_text("must not enter image")
            binaries = root / "bin"
            binaries.mkdir()
            docker = binaries / "docker"
            docker.write_text(
                "#!/usr/bin/env python3\n"
                "import json, os, pathlib, sys\n"
                "context = pathlib.Path(sys.argv[-1])\n"
                'pathlib.Path(os.environ["OAMB_BUILD_TRACE"]).write_text(json.dumps({\n'
                '"args": sys.argv[1:], "evidence": (context/"evidence.txt").read_text(),\n'
                '"private_exists": (context/"private.txt").exists(), "context": str(context)}))\n'
            )
            docker.chmod(0o755)
            trace = root / "trace.json"
            subprocess.run(
                [str(script), str(source), revision],
                check=True,
                capture_output=True,
                env={
                    **os.environ,
                    "PATH": f"{binaries}:{os.environ['PATH']}",
                    "OAMB_BUILD_TRACE": str(trace),
                },
            )
            result = json.loads(trace.read_text())
            self.assertEqual(result["evidence"], "committed")
            self.assertFalse(result["private_exists"])
            self.assertIn(f"org.opencontainers.image.revision={revision}", result["args"])
            self.assertIn(f"oamb/oam:source-{revision[:12]}", result["args"])
            self.assertFalse(Path(result["context"]).exists())
