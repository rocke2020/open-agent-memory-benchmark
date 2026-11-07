import hashlib
import runpy
from pathlib import Path

import pyarrow as pa  # type: ignore[import-untyped]
import pyarrow.parquet as pq  # type: ignore[import-untyped]
import pytest


def test_export_preserves_order_unicode_and_campaign_json_serialization(tmp_path, monkeypatch):
    script = Path(__file__).resolve().parents[2] / "scripts/download/beam10m.py"
    assert script.exists(), "pinned BEAM download/export implementation is missing"
    monkeypatch.syspath_prepend(str(script.parent))
    export = runpy.run_path(str(script))["export_parquet"]
    shards = []
    for index, row in enumerate(
        [{"conversation_id": "1", "text": "茶"}, {"conversation_id": "2", "text": "red"}]
    ):
        shard = tmp_path / f"{index}.parquet"
        pq.write_table(pa.Table.from_pylist([row]), shard)
        shards.append(shard)
    expected = (
        b'[{"conversation_id": "1", "text": "\\u8336"}, {"conversation_id": "2", "text": "red"}]'
    )
    destination = tmp_path / "10m.json"
    digest = hashlib.sha256(expected).hexdigest()
    export(shards, destination, digest)
    assert destination.read_bytes() == expected
    before = destination.stat().st_mtime_ns
    export(shards, destination, digest)
    assert destination.stat().st_mtime_ns == before
    with pytest.raises(RuntimeError):
        export(shards, destination, "0" * 64)
    assert destination.read_bytes() == expected
    with pytest.raises(RuntimeError):
        export(shards, tmp_path / "bad.json", "0" * 64)
    assert not (tmp_path / "bad.json").exists()
