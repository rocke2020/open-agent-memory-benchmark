"""Disable provider-internal model retries for the fixed OAMB service profile."""

from __future__ import annotations

import importlib.abc
import importlib.machinery
import sys
from types import ModuleType
from typing import Any

INTERNAL_RETRY_COUNT = 0


def _disable_openai_sdk_retries(module: ModuleType) -> None:
    for class_name in ("OpenAI", "AsyncOpenAI"):
        client_class = getattr(module, class_name, None)
        original = getattr(client_class, "__init__", None)
        if original is None or getattr(original, "_oamb_retry_guard", False):
            continue

        def guarded_init(
            self: object,
            *args: object,
            _original: Any = original,
            **kwargs: object,
        ) -> None:
            kwargs["max_retries"] = INTERNAL_RETRY_COUNT
            _original(self, *args, **kwargs)

        guarded_init._oamb_retry_guard = True  # type: ignore[attr-defined]
        client_class.__init__ = guarded_init
    module.OAMB_INTERNAL_RETRY_COUNT = INTERNAL_RETRY_COUNT


class _OpenVikingSessionLoader(importlib.abc.Loader):
    def __init__(self, delegate: importlib.abc.Loader) -> None:
        self._delegate = delegate

    def create_module(self, spec: object) -> ModuleType | None:
        creator = getattr(self._delegate, "create_module", None)
        return creator(spec) if creator is not None else None

    def exec_module(self, module: ModuleType) -> None:
        self._delegate.exec_module(module)
        if not hasattr(module, "_MEMORY_EXTRACTION_MAX_RETRIES"):
            raise RuntimeError("OpenViking memory retry control is unavailable")
        module._MEMORY_EXTRACTION_MAX_RETRIES = INTERNAL_RETRY_COUNT
        module.OAMB_INTERNAL_RETRY_COUNT = INTERNAL_RETRY_COUNT


class _OpenVikingSessionFinder(importlib.abc.MetaPathFinder):
    _TARGET = "openviking.session.session"

    def find_spec(
        self,
        fullname: str,
        path: object = None,
        target: ModuleType | None = None,
    ) -> importlib.machinery.ModuleSpec | None:
        if fullname != self._TARGET:
            return None
        spec = importlib.machinery.PathFinder.find_spec(fullname, path, target)
        if spec is None or spec.loader is None:
            raise RuntimeError("OpenViking session module cannot be resolved")
        if not isinstance(spec.loader, importlib.abc.Loader):
            raise RuntimeError("OpenViking session loader cannot be guarded")
        spec.loader = _OpenVikingSessionLoader(spec.loader)
        return spec


try:
    import openai
except ModuleNotFoundError:
    pass
else:
    _disable_openai_sdk_retries(openai)

sys.meta_path.insert(0, _OpenVikingSessionFinder())
