from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DOWNLOAD_SCRIPTS = REPOSITORY_ROOT / "scripts" / "download"
LONGMEMEVAL_REVISION = "98d7416c24c778c2fee6e6f3006e7a073259d48f"
LONGMEMEVAL_SHA256 = "d6f21ea9d60a0d56f34a05b609c79c88a451d2ae03597821ea3d5a9678c3a442"
MEMORY_AGENT_BENCH_REVISION = "7ea066982b140a19337e17e60d45d4076e042faf"


def _write_fake_huggingface_hub(fake_python_root: Path) -> None:
    package = fake_python_root / "huggingface_hub"
    package.mkdir(parents=True, exist_ok=True)
    (package / "__init__.py").write_text(
        """
import os
from pathlib import Path


def _fail_if_hub_access_is_forbidden():
    if os.environ.get("FAKE_FAIL_ON_HUB_ACCESS") == "1":
        raise AssertionError("Hub access is forbidden")


class _DatasetInfo:
    sha = os.environ["FAKE_RESOLVED_REVISION"]


class HfApi:
    def dataset_info(self, repository, revision):
        _fail_if_hub_access_is_forbidden()
        return _DatasetInfo()

    def list_repo_files(self, *args, **kwargs):
        raise AssertionError("pinned downloads must not enumerate the repository")


def hf_hub_download(*, repo_id, filename, repo_type, revision):
    _fail_if_hub_access_is_forbidden()
    return str(Path(os.environ["FAKE_HUB_ROOT"]) / filename)
""".lstrip(),
        encoding="utf-8",
    )


def _run_python_downloader(
    tmp_path: Path,
    expected_checksums: str,
    output_directory: Path | None = None,
) -> subprocess.CompletedProcess[str]:
    fake_hub_root = tmp_path / "hub"
    (fake_hub_root / "data").mkdir(parents=True, exist_ok=True)
    (fake_hub_root / "data" / "sample.parquet").write_bytes(b"abc")
    (fake_hub_root / "entity2id.json").write_bytes(b"hello")

    fake_python_root = tmp_path / "fake-python"
    _write_fake_huggingface_hub(fake_python_root)

    expected_file = tmp_path / "expected.sha256"
    expected_file.write_text(expected_checksums, encoding="utf-8")
    output_directory = output_directory or tmp_path / "output"

    environment = os.environ.copy()
    environment.update(
        {
            "PYTHONPATH": str(fake_python_root),
            "DATASET_REPOSITORY": "example/dataset",
            "DATASET_REVISION": "requested-revision",
            "OUTPUT_DIRECTORY": str(output_directory),
            "EXPECTED_CHECKSUMS_FILE": str(expected_file),
            "REVISION_FILE": str(output_directory / "REVISION"),
            "CHECKSUM_FILE": str(output_directory / "SHA256SUMS"),
            "FAKE_HUB_ROOT": str(fake_hub_root),
            "FAKE_RESOLVED_REVISION": "resolved-revision",
        }
    )
    return subprocess.run(
        [sys.executable, str(DOWNLOAD_SCRIPTS / "hf_dataset.py")],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )


def test_python_downloader_installs_only_checksum_manifest_files(tmp_path: Path) -> None:
    checksums = (
        "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        "  data/sample.parquet\n"
        "2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824"
        "  entity2id.json\n"
    )

    result = _run_python_downloader(tmp_path, checksums)

    assert result.returncode == 0, result.stderr
    output_directory = tmp_path / "output"
    assert (output_directory / "data" / "sample.parquet").read_bytes() == b"abc"
    assert (output_directory / "entity2id.json").read_bytes() == b"hello"
    assert (output_directory / "REVISION").read_text(encoding="utf-8") == ("resolved-revision\n")
    assert (output_directory / "SHA256SUMS").read_text(encoding="utf-8") == checksums


def test_python_downloader_rejects_pinned_checksum_mismatch(tmp_path: Path) -> None:
    result = _run_python_downloader(
        tmp_path,
        f"{'0' * 64}  data/sample.parquet\n",
    )

    assert result.returncode != 0
    assert "does not match expected SHA-256" in result.stderr
    assert not (tmp_path / "output" / "data" / "sample.parquet").exists()


def test_python_downloader_preserves_existing_mismatched_file(tmp_path: Path) -> None:
    output_directory = tmp_path / "output"
    existing_file = output_directory / "data" / "sample.parquet"
    existing_file.parent.mkdir(parents=True)
    existing_file.write_bytes(b"preserve-me")
    checksums = (
        "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad  data/sample.parquet\n"
    )

    result = _run_python_downloader(tmp_path, checksums, output_directory)

    assert result.returncode != 0
    assert "existing file does not match selected Hub revision" in result.stderr
    assert existing_file.read_bytes() == b"preserve-me"


def test_python_downloader_is_idempotent_for_verified_files(tmp_path: Path) -> None:
    checksums = (
        "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad  data/sample.parquet\n"
    )

    first_result = _run_python_downloader(tmp_path, checksums)
    assert first_result.returncode == 0, first_result.stderr
    first_manifest = (tmp_path / "output" / "SHA256SUMS").read_bytes()
    second_result = _run_python_downloader(tmp_path, checksums)

    assert second_result.returncode == 0, second_result.stderr
    assert "Already downloaded and verified" in second_result.stdout
    assert (tmp_path / "output" / "data" / "sample.parquet").read_bytes() == b"abc"
    assert (tmp_path / "output" / "SHA256SUMS").read_bytes() == first_manifest


def test_python_downloader_rejects_missing_checksum_before_hub_access(
    tmp_path: Path,
) -> None:
    fake_python_root = tmp_path / "fake-python"
    _write_fake_huggingface_hub(fake_python_root)
    environment = os.environ.copy()
    for name in (
        "DATASET_FILE",
        "EXPECTED_SHA256",
        "EXPECTED_CHECKSUMS_FILE",
        "REVISION_FILE",
        "CHECKSUM_FILE",
    ):
        environment.pop(name, None)
    environment.update(
        {
            "PYTHONPATH": str(fake_python_root),
            "DATASET_REPOSITORY": "example/dataset",
            "DATASET_REVISION": "requested-revision",
            "OUTPUT_DIRECTORY": str(tmp_path / "output"),
            "FAKE_FAIL_ON_HUB_ACCESS": "1",
            "FAKE_RESOLVED_REVISION": "resolved-revision",
        }
    )

    result = subprocess.run(
        [sys.executable, str(DOWNLOAD_SCRIPTS / "hf_dataset.py")],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )

    assert result.returncode != 0
    assert "a pinned checksum is required" in result.stderr
    assert "Hub access is forbidden" not in result.stderr


def _capture_wrapper_environment(tmp_path: Path, wrapper_name: str) -> dict[str, str]:
    repository = tmp_path / "repository"
    download_directory = repository / "scripts" / "download"
    download_directory.mkdir(parents=True)
    shutil.copy2(DOWNLOAD_SCRIPTS / wrapper_name, download_directory / wrapper_name)
    source_manifest = DOWNLOAD_SCRIPTS / "memoryagentbench.sha256"
    if source_manifest.exists():
        shutil.copy2(source_manifest, download_directory / source_manifest.name)

    capture_file = tmp_path / "environment.txt"
    (download_directory / "hf_dataset.sh").write_text(
        """#!/usr/bin/env bash
set -euo pipefail
for name in DATASET_REPOSITORY DATASET_REVISION DATASET_FILE OUTPUT_DIRECTORY \\
  EXPECTED_SHA256 EXPECTED_CHECKSUMS_FILE REVISION_FILE CHECKSUM_FILE; do
  printf '%s=%s\\n' "$name" "${!name-}"
done > "$CAPTURE_FILE"
""",
        encoding="utf-8",
    )
    (download_directory / "hf_dataset.sh").chmod(0o755)

    environment = os.environ.copy()
    environment["CAPTURE_FILE"] = str(capture_file)
    subprocess.run(
        ["bash", str(download_directory / wrapper_name)],
        check=True,
        env=environment,
    )
    return dict(
        line.split("=", 1) for line in capture_file.read_text(encoding="utf-8").splitlines()
    )


def test_longmemeval_wrapper_uses_pinned_s_dataset_and_repository_output(
    tmp_path: Path,
) -> None:
    environment = _capture_wrapper_environment(tmp_path, "longmemeval.sh")
    repository = tmp_path / "repository"
    output = repository / "datasets" / "longmemeval-cleaned"

    assert environment == {
        "DATASET_REPOSITORY": "xiaowu0162/longmemeval-cleaned",
        "DATASET_REVISION": LONGMEMEVAL_REVISION,
        "DATASET_FILE": "longmemeval_s_cleaned.json",
        "OUTPUT_DIRECTORY": str(output),
        "EXPECTED_SHA256": LONGMEMEVAL_SHA256,
        "EXPECTED_CHECKSUMS_FILE": "",
        "REVISION_FILE": str(output / "REVISION"),
        "CHECKSUM_FILE": str(output / "SHA256SUMS"),
    }


def test_hf_dataset_wrapper_activates_locked_download_group(tmp_path: Path) -> None:
    repository = tmp_path / "repository"
    download_directory = repository / "scripts" / "download"
    download_directory.mkdir(parents=True)
    shutil.copy2(DOWNLOAD_SCRIPTS / "hf_dataset.sh", download_directory / "hf_dataset.sh")
    (download_directory / "hf_dataset.py").write_text("", encoding="utf-8")

    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    capture_file = tmp_path / "uv-arguments.txt"
    (fake_bin / "uv").write_text(
        '#!/usr/bin/env bash\nprintf \'%s\\n\' "$@" > "$CAPTURE_FILE"\n',
        encoding="utf-8",
    )
    (fake_bin / "uv").chmod(0o755)

    environment = os.environ.copy()
    environment["PATH"] = f"{fake_bin}{os.pathsep}{environment['PATH']}"
    environment["CAPTURE_FILE"] = str(capture_file)
    subprocess.run(
        ["bash", str(download_directory / "hf_dataset.sh")],
        check=True,
        env=environment,
    )

    assert capture_file.read_text(encoding="utf-8").splitlines() == [
        "run",
        "--locked",
        "--group",
        "download",
        "--project",
        str(repository),
        "python",
        str(download_directory / "hf_dataset.py"),
    ]
