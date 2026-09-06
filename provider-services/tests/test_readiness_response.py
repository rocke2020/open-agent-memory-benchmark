from __future__ import annotations

import importlib.util
import math
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "lib" / "readiness_response.py"


def load_module():
    spec = importlib.util.spec_from_file_location("readiness_response", MODULE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load readiness response module")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class ReadinessResponseTests(unittest.TestCase):
    def test_chat_response_requires_usable_message_without_resolving_model(self) -> None:
        module = load_module()
        valid = {
            "model": "fixture-chat-model",
            "choices": [
                {
                    "finish_reason": "stop",
                    "message": {"role": "assistant", "content": "OK"},
                }
            ],
        }

        module.validate_chat_response(valid)
        module.validate_chat_response({key: value for key, value in valid.items() if key != "model"})
        module.validate_chat_response({**valid, "model": "supplier-private-model"})

        invalid = (
            {"model": "fixture-chat-model", "choices": []},
            {"model": "fixture-chat-model", "choices": [{}]},
            {
                "model": "fixture-chat-model",
                "choices": [
                    {"message": {"role": "assistant", "content": "OK"}}
                ],
            },
            {
                "model": "fixture-chat-model",
                "choices": [
                    {
                        "finish_reason": "length",
                        "message": {"role": "assistant", "content": "OK"},
                    }
                ],
            },
            {
                "model": "fixture-chat-model",
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {"role": "assistant", "content": ""},
                    }
                ],
            },
        )
        for document in invalid:
            with self.subTest(document=document):
                with self.assertRaises(module.ReadinessResponseError):
                    module.validate_chat_response(document)

    def test_embedding_response_requires_exact_finite_numeric_vector(self) -> None:
        module = load_module()
        valid = {
            "model": "fixture-embedding-model",
            "data": [{"embedding": [0.0] * 1024}],
        }

        module.validate_embedding_response(valid, "fixture-embedding-model", 1024)

        invalid_vectors = (
            [0.0] * 1023,
            [0.0] * 1023 + [None],
            [0.0] * 1023 + ["0"],
            [0.0] * 1023 + [math.inf],
            [0.0] * 1023 + [math.nan],
            [0.0] * 1023 + [True],
        )
        for vector in invalid_vectors:
            with self.subTest(last=vector[-1], length=len(vector)):
                with self.assertRaises(module.ReadinessResponseError):
                    module.validate_embedding_response(
                        {"model": "fixture-embedding-model", "data": [{"embedding": vector}]},
                        "fixture-embedding-model",
                        1024,
                    )

        with self.assertRaises(module.ReadinessResponseError):
            module.validate_embedding_response(
                {**valid, "model": "different-model"},
                "fixture-embedding-model",
                1024,
            )


if __name__ == "__main__":
    unittest.main()
