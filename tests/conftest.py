from __future__ import annotations

import os
from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def source_checkout_imports(monkeypatch: pytest.MonkeyPatch) -> None:
    source = str(Path(__file__).resolve().parents[1] / "src")
    existing = os.environ.get("PYTHONPATH", "")
    monkeypatch.setenv("PYTHONPATH", source + (os.pathsep + existing if existing else ""))
