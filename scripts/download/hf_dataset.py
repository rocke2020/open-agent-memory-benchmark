#!/usr/bin/env python3

import hashlib
import os
import shutil
import string
import tempfile
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download


def sha256_file(file_path):
    digest = hashlib.sha256()
    with file_path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_expected_checksums(checksum_path):
    checksums = {}
    for line_number, raw_line in enumerate(
        checksum_path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not raw_line:
            continue
        digest, separator, dataset_file = raw_line.partition("  ")
        if (
            not separator
            or len(digest) != 64
            or any(character not in string.hexdigits for character in digest)
            or not dataset_file
        ):
            raise RuntimeError(f"invalid checksum manifest line {line_number}: {checksum_path}")
        if dataset_file in checksums:
            raise RuntimeError(f"duplicate checksum manifest path: {dataset_file}")
        checksums[dataset_file] = digest.lower()
    if not checksums:
        raise RuntimeError(f"checksum manifest is empty: {checksum_path}")
    return checksums


def write_atomic(file_path, content):
    file_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{file_path.name}.", suffix=".tmp", dir=file_path.parent
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as destination:
            destination.write(content)
        os.replace(temporary_path, file_path)
    finally:
        temporary_path.unlink(missing_ok=True)


def install_verified_file(cached_path, destination_path, expected_sha256):
    destination_path.parent.mkdir(parents=True, exist_ok=True)

    if destination_path.exists() or destination_path.is_symlink():
        if destination_path.is_symlink() or not destination_path.is_file():
            raise RuntimeError(f"existing destination is not a regular file: {destination_path}")
        if sha256_file(destination_path) != expected_sha256:
            raise RuntimeError(
                f"existing file does not match selected Hub revision: {destination_path}"
            )
        print(f"Already downloaded and verified: {destination_path}")
        return

    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination_path.name}.",
        suffix=".tmp",
        dir=destination_path.parent,
    )
    os.close(descriptor)
    temporary_path = Path(temporary_name)
    try:
        shutil.copyfile(cached_path, temporary_path)
        if sha256_file(temporary_path) != expected_sha256:
            raise RuntimeError(f"copied file failed SHA-256 verification: {temporary_path}")
        try:
            os.link(temporary_path, destination_path)
        except FileExistsError as error:
            raise RuntimeError(
                f"destination appeared during download: {destination_path}"
            ) from error
    finally:
        temporary_path.unlink(missing_ok=True)

    print(f"Downloaded and verified: {destination_path}")


def main():
    repository = os.environ.get("DATASET_REPOSITORY", "ai-hyz/MemoryAgentBench")
    requested_revision = os.environ.get("DATASET_REVISION", "main")
    requested_file = os.environ.get("DATASET_FILE", "")
    expected_sha256 = os.environ.get("EXPECTED_SHA256", "").lower()
    expected_checksums_file_value = os.environ.get("EXPECTED_CHECKSUMS_FILE")
    output_directory = Path(os.environ.get("OUTPUT_DIRECTORY", "./datasets")).expanduser().resolve()

    if expected_sha256 and not requested_file:
        raise RuntimeError("EXPECTED_SHA256 requires DATASET_FILE")
    if expected_sha256 and (
        len(expected_sha256) != 64
        or any(character not in string.hexdigits for character in expected_sha256)
    ):
        raise RuntimeError("EXPECTED_SHA256 must be 64 hexadecimal characters")
    if requested_file and expected_checksums_file_value:
        raise RuntimeError("DATASET_FILE and EXPECTED_CHECKSUMS_FILE cannot be used together")

    expected_checksums = (
        read_expected_checksums(
            Path(expected_checksums_file_value).expanduser().resolve(strict=True)
        )
        if expected_checksums_file_value
        else ({requested_file: expected_sha256} if expected_sha256 else {})
    )
    if not expected_checksums:
        raise RuntimeError(
            "a pinned checksum is required through EXPECTED_SHA256 or EXPECTED_CHECKSUMS_FILE"
        )

    if requested_file in {".", ".."} or requested_file.startswith(("/", "./", "../")):
        raise RuntimeError(f"unsafe dataset file path: {requested_file}")

    revision_file_value = os.environ.get("REVISION_FILE")
    checksum_file_value = os.environ.get("CHECKSUM_FILE")
    revision_file = (
        Path(revision_file_value).expanduser().resolve(strict=False)
        if revision_file_value
        else None
    )
    checksum_file = (
        Path(checksum_file_value).expanduser().resolve(strict=False)
        if checksum_file_value
        else None
    )
    if revision_file and revision_file == checksum_file:
        raise RuntimeError(f"revision and checksum manifests conflict: {revision_file}")
    if requested_file:
        requested_destination = (output_directory / requested_file).resolve(strict=False)
        for manifest_file in (revision_file, checksum_file):
            if manifest_file == requested_destination:
                raise RuntimeError(f"dataset destination conflicts with manifest: {manifest_file}")

    resolved_revision = HfApi().dataset_info(repository, revision=requested_revision).sha
    if not resolved_revision:
        raise RuntimeError(f"could not resolve revision: {requested_revision}")

    dataset_files = list(expected_checksums)

    checksums = []
    for dataset_file in dataset_files:
        if not dataset_file or Path(dataset_file).is_absolute():
            raise RuntimeError(f"unsafe dataset file path: {dataset_file}")
        destination_path = output_directory / dataset_file
        resolved_destination = destination_path.resolve(strict=False)
        if (
            resolved_destination != output_directory
            and output_directory not in resolved_destination.parents
        ):
            raise RuntimeError(f"dataset file escapes output directory: {dataset_file}")
        for manifest_file in (revision_file, checksum_file):
            if manifest_file == resolved_destination:
                raise RuntimeError(f"dataset destination conflicts with manifest: {manifest_file}")

        cached_path = Path(
            hf_hub_download(
                repo_id=repository,
                filename=dataset_file,
                repo_type="dataset",
                revision=resolved_revision,
            )
        )
        if not cached_path.is_file():
            raise RuntimeError(f"Hugging Face cache entry is not a file: {cached_path}")

        cached_sha256 = sha256_file(cached_path)
        pinned_sha256 = expected_checksums.get(dataset_file)
        if pinned_sha256 and cached_sha256 != pinned_sha256:
            raise RuntimeError(
                f"{dataset_file} does not match expected SHA-256: "
                f"expected {pinned_sha256}, got {cached_sha256}"
            )
        install_verified_file(cached_path, destination_path, cached_sha256)
        checksums.append(f"{cached_sha256}  {dataset_file}")

    if revision_file:
        write_atomic(revision_file, f"{resolved_revision}\n")

    if checksum_file:
        write_atomic(checksum_file, "\n".join(checksums) + "\n")

    print(f"Revision: {resolved_revision}")
    print(f"Downloaded files: {len(dataset_files)}")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        raise SystemExit(f"ERROR: {error}") from error
