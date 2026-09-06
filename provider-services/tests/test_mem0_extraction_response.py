from __future__ import annotations

import asyncio
import copy
import importlib.util
import json
import subprocess
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "mem0" / "extraction_response.py"
FIXTURE_PATH = Path(__file__).parent / "fixtures" / "mem0_extraction_stage.py.txt"
PATCH_ANCHOR = "        if not extracted_memories:\n"
MEANINGFUL_TEXT = "  User's tea preference: 乌龙茶; no sugar.  "
SECOND_TEXT = "Assistant recommended a ceramic teapot."


def load_module():
    spec = importlib.util.spec_from_file_location("oamb_extraction_response", MODULE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load Mem0 extraction response module")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class MemoryConsumer:
    """Only records local calls made after the original parser-stage boundary."""

    def __init__(self):
        self.embedded = []
        self.persisted = []
        self.saved_messages = []
        self.db = types.SimpleNamespace(save_messages=self.save_messages)
        self.embedding_model = types.SimpleNamespace(embed_batch=self.embed_batch)

    def embed_batch(self, texts, operation):
        self.embedded.append((list(texts), operation))
        return [[0.0] for _text in texts]

    def persist(self, memories):
        self.persisted.extend(copy.deepcopy(memories))
        return [{"event": "ADD", "memory": memory["text"]} for memory in memories]

    def save_messages(self, messages, scope):
        self.saved_messages.append((copy.deepcopy(messages), scope))


def execute_stage(module, source, value, consumer, *, asynchronous):
    namespace = {
        "remove_code_blocks": lambda response: response,
        "extract_json": lambda response: response,
        "logger": types.SimpleNamespace(error=lambda _message: None),
    }
    provider_modules = {
        "mem0": types.ModuleType("mem0"),
        "mem0.memory": types.ModuleType("mem0.memory"),
        "mem0.memory.oamb_extraction_response": module,
    }
    with patch.dict(sys.modules, provider_modules):
        exec(compile(source, str(FIXTURE_PATH), "exec"), namespace)
        method = namespace["AsyncMemory" if asynchronous else "Memory"]._add_to_vector_store
        result = method(
            consumer,
            json.dumps({"memory": value}, ensure_ascii=False),
            [{"role": "user", "content": "A synthetic test message."}],
            "run_id=fresh-fixture",
        )
        return asyncio.run(result) if asynchronous else result


class Mem0ExtractionResponseTests(unittest.TestCase):
    def test_supported_shapes_reach_add_with_exact_text_in_both_native_stages(self):
        module = load_module()
        source = module.patch_source(FIXTURE_PATH.read_text(encoding="utf-8"))
        native = {
            "id": "0",
            "text": MEANINGFUL_TEXT,
            "attributed_to": "user",
            "linked_memory_ids": ["existing-memory"],
            "provider_extension": {"retained": True},
        }
        cases = (
            ("native_objects", [native], [native]),
            ("strings", [MEANINGFUL_TEXT], [{"text": MEANINGFUL_TEXT}]),
            ("single_object", native, [native]),
            ("single_string", MEANINGFUL_TEXT, [{"text": MEANINGFUL_TEXT}]),
            ("mixed", [native, SECOND_TEXT], [native, {"text": SECOND_TEXT}]),
        )
        for asynchronous in (False, True):
            for label, value, expected in cases:
                with self.subTest(asynchronous=asynchronous, shape=label):
                    original = copy.deepcopy(value)
                    consumer = MemoryConsumer()
                    result = execute_stage(
                        module, source, value, consumer, asynchronous=asynchronous
                    )
                    expected_texts = [item["text"] for item in expected]
                    self.assertEqual(
                        result, [{"event": "ADD", "memory": text} for text in expected_texts]
                    )
                    self.assertEqual(consumer.embedded, [(expected_texts, "add")])
                    self.assertEqual(consumer.persisted, expected)
                    self.assertEqual(len(consumer.saved_messages), 1)
                    self.assertEqual(consumer.saved_messages[0][1], "run_id=fresh-fixture")
                    self.assertEqual(value, original)

    def test_native_empty_result_saves_messages_without_embedding_or_add(self):
        module = load_module()
        source = module.patch_source(FIXTURE_PATH.read_text(encoding="utf-8"))
        for asynchronous in (False, True):
            with self.subTest(asynchronous=asynchronous):
                consumer = MemoryConsumer()
                result = execute_stage(module, source, [], consumer, asynchronous=asynchronous)
                self.assertEqual(result, [])
                self.assertEqual(consumer.embedded, [])
                self.assertEqual(consumer.persisted, [])
                self.assertEqual(len(consumer.saved_messages), 1)

    def test_invalid_member_rejects_entire_batch_outside_native_parse_catch(self):
        module = load_module()
        source = module.patch_source(FIXTURE_PATH.read_text(encoding="utf-8"))
        invalid_values = (
            None,
            False,
            7,
            "",
            " \t\n",
            {},
            {"unrelated": MEANINGFUL_TEXT},
            {"text": None},
            {"text": 7},
            {"text": " \t"},
            [[MEANINGFUL_TEXT]],
            [{"text": MEANINGFUL_TEXT}, None],
            [MEANINGFUL_TEXT, {"text": 7}],
        )
        for asynchronous in (False, True):
            for value in invalid_values:
                with self.subTest(asynchronous=asynchronous, shape=repr(value)):
                    consumer = MemoryConsumer()
                    with self.assertRaises((ValueError, TypeError)):
                        execute_stage(module, source, value, consumer, asynchronous=asynchronous)
                    self.assertEqual(consumer.embedded, [])
                    self.assertEqual(consumer.persisted, [])
                    self.assertEqual(consumer.saved_messages, [])

    def test_direct_normalizer_preserves_native_fields_and_does_not_mutate_input(self):
        module = load_module()
        value = [
            {
                "text": MEANINGFUL_TEXT,
                "id": "7",
                "attributed_to": "assistant",
                "linked_memory_ids": ["a"],
            }
        ]
        original = copy.deepcopy(value)
        self.assertEqual(module.normalize_extracted_memories(value), original)
        self.assertEqual(value, original)
        self.assertEqual(
            module.normalize_extracted_memories(MEANINGFUL_TEXT), [{"text": MEANINGFUL_TEXT}]
        )

    def test_patch_rejects_anchor_drift_and_second_application(self):
        module = load_module()
        source = FIXTURE_PATH.read_text(encoding="utf-8")
        invalid_sources = (
            source.replace(PATCH_ANCHOR, "        if len(extracted_memories) == 0:\n", 1),
            source + PATCH_ANCHOR,
            module.patch_source(source),
        )
        for invalid in invalid_sources:
            with self.subTest(source_length=len(invalid)):
                with self.assertRaises((ValueError, RuntimeError)):
                    module.patch_source(invalid)

    def test_cli_applies_executable_patch_and_preserves_bytes_on_drift(self):
        module = load_module()
        source = FIXTURE_PATH.read_text(encoding="utf-8")
        with tempfile.TemporaryDirectory() as temporary:
            target = Path(temporary) / "main.py"
            target.write_text(source, encoding="utf-8")
            result = subprocess.run(
                [sys.executable, str(MODULE_PATH), str(target)],
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            patched = target.read_text(encoding="utf-8")
            consumer = MemoryConsumer()
            events = execute_stage(module, patched, [MEANINGFUL_TEXT], consumer, asynchronous=False)
            self.assertEqual(events, [{"event": "ADD", "memory": MEANINGFUL_TEXT}])
            before = target.read_bytes()
            repeated = subprocess.run(
                [sys.executable, str(MODULE_PATH), str(target)],
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertNotEqual(repeated.returncode, 0)
            self.assertEqual(target.read_bytes(), before)

    def test_image_fingerprint_changes_when_normalizer_bytes_change(self):
        operator = (ROOT / "bin" / "provider-services").read_text(encoding="utf-8")
        definition = operator.split("mem0_build_input_sha256() {", 1)[1].split(
            "\nmem0_image_matches_pin()", 1
        )[0]
        command = (
            "mem0_build_input_sha256() {"
            + definition
            + '\nROOT=$1\nBUILD_DIR="$ROOT/.build"\nmem0_build_input_sha256\n'
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for relative in (
                ".dockerignore",
                "mem0/Dockerfile",
                "mem0/requirements.lock",
                "mem0/extraction_response.py",
                "mem0/inspector.py",
                ".build/mem0-source.tar",
            ):
                target = root / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(relative, encoding="utf-8")

            def fingerprint():
                result = subprocess.run(
                    ["sh", "-c", command, "sh", str(root)],
                    capture_output=True,
                    text=True,
                    check=False,
                )
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                return result.stdout.strip()

            original = fingerprint()
            self.assertEqual(len(original), 64)
            (root / "mem0/extraction_response.py").write_text(
                "changed normalizer", encoding="utf-8"
            )
            changed = fingerprint()
            self.assertNotEqual(original, changed)
            self.assertEqual(changed, fingerprint())


if __name__ == "__main__":
    unittest.main()
