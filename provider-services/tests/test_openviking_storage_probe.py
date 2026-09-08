from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

PROBE_PATH = Path(__file__).resolve().parents[1] / "openviking" / "storage_probe.py"


def load_probe():
    spec = importlib.util.spec_from_file_location("oamb_openviking_storage_probe", PROBE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load OpenViking storage probe")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class OpenVikingStorageProbeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.module = load_probe()

    def make_workspace(self, root: Path) -> Path:
        workspace = root / "workspace"
        collection = workspace / "vectordb" / "context"
        index = collection / "index" / "default"
        agfs = workspace / "viking" / "accounts" / "oamb-benchmark"
        index.mkdir(parents=True)
        agfs.mkdir(parents=True)
        (collection / "collection_meta.json").write_text(
            json.dumps(
                {
                    "CollectionName": "context",
                    "HasPrimaryKey": True,
                    "PrimaryKey": "id",
                    "Dimension": 1024,
                    "Fields": [
                        {"FieldName": "id", "FieldType": "string", "IsPrimaryKey": True},
                        {"FieldName": "vector", "FieldType": "vector", "Dim": 1024},
                    ],
                    "FieldsDict": {
                        "id": {"FieldName": "id", "FieldType": "string", "IsPrimaryKey": True},
                        "vector": {"FieldName": "vector", "FieldType": "vector", "Dim": 1024},
                    },
                }
            ),
            encoding="utf-8",
        )
        (index / "index_meta.json").write_text(
            json.dumps(
                {
                    "CollectionName": "context",
                    "VectorIndex": {
                        "IndexType": "flat",
                        "Dimension": 1024,
                        "Distance": "ip",
                        "NormalizeVector": True,
                    },
                }
            ),
            encoding="utf-8",
        )
        (agfs / "profile.json").write_text('{"initialized":true}\n', encoding="utf-8")
        return workspace

    def test_intact_empty_workspace_is_read_without_changes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            workspace = self.make_workspace(Path(temporary))
            before = self.module.snapshot_required_files(workspace)
            result = self.module.probe_storage(workspace, exact_release=False)
            after = self.module.snapshot_required_files(workspace)
            self.assertEqual(before, after)
            self.assertEqual(result["status"], "ok")
            self.assertEqual(result["collection"], "context")
            self.assertEqual(result["dimension"], 1024)
            self.assertGreaterEqual(result["agfs_entries"], 1)

    def test_large_preserved_workspace_has_no_fixed_entry_gate(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            workspace = self.make_workspace(Path(temporary))
            entries = workspace / "viking" / "preserved"
            entries.mkdir()
            for ordinal in range(10_001):
                (entries / str(ordinal)).mkdir()

            result = self.module.probe_storage(workspace, exact_release=False)

            self.assertEqual(result["status"], "ok")
            self.assertGreater(result["agfs_entries"], 10_000)

    def test_missing_or_corrupt_vector_metadata_fails(self) -> None:
        for mode in ("missing", "corrupt"):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as temporary:
                workspace = self.make_workspace(Path(temporary))
                metadata = workspace / "vectordb" / "context" / "collection_meta.json"
                if mode == "missing":
                    metadata.unlink()
                else:
                    metadata.write_text("not-json", encoding="utf-8")
                with self.assertRaises(self.module.ProbeError):
                    self.module.probe_storage(workspace, exact_release=False)

    def test_symlink_in_required_storage_tree_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            workspace = self.make_workspace(root)
            (workspace / "viking" / "escape").symlink_to(root)
            with self.assertRaises(self.module.ProbeError):
                self.module.probe_storage(workspace, exact_release=False)

    def test_mountinfo_requires_workspace_to_be_read_only(self) -> None:
        mountinfo = (
            "36 25 0:32 / /probe/workspace ro,nosuid,nodev - virtiofs host rw\n"
        )
        self.module.require_read_only_mount(Path("/probe/workspace"), mountinfo)
        with self.assertRaises(self.module.ProbeError):
            self.module.require_read_only_mount(
                Path("/probe/workspace"),
                "36 25 0:32 / /probe/workspace rw,nosuid,nodev - virtiofs host rw\n",
            )

    def test_exact_release_path_avoids_mutating_provider_constructors(self) -> None:
        source = PROBE_PATH.read_text(encoding="utf-8")
        self.assertIn("get_binding_client", source)
        self.assertIn("FileStore", source)
        self.assertNotIn("CollectionMeta(", source)
        self.assertNotIn("IndexMeta(", source)
        self.assertNotIn("initialize_openviking_config", source)
        self.assertNotIn("create_agfs_client", source)


if __name__ == "__main__":
    unittest.main()
