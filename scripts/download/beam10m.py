#!/usr/bin/env python3
"""Export pinned official BEAM shards using the recorded AMB JSON serialization."""

import argparse
import json
import tempfile
from pathlib import Path

import pyarrow.parquet as pq
from hf_dataset import install_verified_file, sha256_file, write_atomic
from huggingface_hub import hf_hub_download

REPOSITORY = "Mohammadta/BEAM-10M"
REVISION = "9b2096193fe74e2837e4713e483351e19817773c"
JSON_SHA256 = "c16643a7cc7d93c1ed5509568cdd9e4c387dcd60414e68dde740b54064e68ba8"
SHARDS = {
    "data/10M-00000-of-00002.parquet": "31d96fd47ec56221d202e68792f26c00e49467dd4b36ee105c36ebd19ef78ad5",
    "data/10M-00001-of-00002.parquet": "a4f13fe25af51d57405ae41008689c31d1421377f3efde56a024b441deb2ee65",
}


def export_parquet(shards: list[Path], destination: Path, expected_sha256: str) -> None:
    if destination.exists() or destination.is_symlink():
        install_verified_file(destination, destination, expected_sha256)
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".beam-export-", dir=destination.parent) as temporary:
        staged = Path(temporary) / "10m.json"
        with staged.open("w", encoding="utf-8") as stream:
            stream.write("[")
            first = True
            for shard in shards:
                for batch in pq.ParquetFile(shard).iter_batches(batch_size=1):
                    for row in batch.to_pylist():
                        if not first:
                            stream.write(", ")
                        json.dump(row, stream)
                        first = False
            stream.write("]")
        install_verified_file(staged, destination, expected_sha256)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("datasets/beam"))
    args = parser.parse_args()
    destination = args.output_dir / "10m.json"
    if destination.exists() or destination.is_symlink():
        export_parquet([], destination, JSON_SHA256)
    else:
        paths = []
        for name, digest in SHARDS.items():
            path = Path(
                hf_hub_download(
                    repo_id=REPOSITORY, filename=name, repo_type="dataset", revision=REVISION
                )
            )
            if sha256_file(path) != digest:
                raise RuntimeError("BEAM source shard checksum mismatch")
            paths.append(path)
        export_parquet(paths, destination, JSON_SHA256)
    write_atomic(args.output_dir / "REVISION", REVISION + "\n")
    write_atomic(args.output_dir / "SHA256SUMS", JSON_SHA256 + "  10m.json\n")


if __name__ == "__main__":
    main()
