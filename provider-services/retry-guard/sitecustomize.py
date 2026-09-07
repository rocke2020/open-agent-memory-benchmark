"""Disable provider-internal model retries for the fixed OAMB service profile."""

from __future__ import annotations

import importlib.abc
import importlib.machinery
import json
import os
import sys
from types import ModuleType
from typing import Any

INTERNAL_RETRY_COUNT = 0


def _required_extraction_max_retries() -> int:
    raw = os.environ["OAMB_EXTRACTION_MAX_RETRIES"]
    value = int(raw)
    if value != 10:
        raise RuntimeError("OAMB extraction max retries must equal the frozen value 10")
    return value


EXTRACTION_MAX_RETRIES = _required_extraction_max_retries()


def _extraction_retry_classifier(original: Any) -> Any:
    """Extend the pinned transient classifier only for extraction format failures."""

    def classify(error: Exception) -> bool:
        current: BaseException | None = error
        seen: set[int] = set()
        while current is not None and id(current) not in seen:
            seen.add(id(current))
            error_type = type(current)
            if isinstance(current, json.JSONDecodeError) or (
                error_type.__name__ == "ValidationError"
                and error_type.__module__.startswith(("pydantic", "pydantic_core"))
            ):
                return True
            current = current.__cause__ or current.__context__
        return bool(original(error))

    return classify


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


def _patch_openviking_task_store(module: ModuleType, plugin_error_type: type[Exception]) -> None:
    store_class = getattr(module, "PersistentTaskStore", None)
    original = getattr(store_class, "_mkdir_if_missing", None)
    if original is None or getattr(original, "_oamb_file_exists_guard", False):
        return

    async def mkdir_if_missing(self: object, path: str) -> None:
        try:
            await original(self, path)
        except plugin_error_type as exc:
            if "file exists (os error 17)" in str(exc).lower():
                return
            raise

    mkdir_if_missing._oamb_file_exists_guard = True  # type: ignore[attr-defined]
    store_class._mkdir_if_missing = mkdir_if_missing


class _OpenVikingTaskStoreLoader(importlib.abc.Loader):
    def __init__(self, delegate: importlib.abc.Loader) -> None:
        self._delegate = delegate

    def create_module(self, spec: object) -> ModuleType | None:
        creator = getattr(self._delegate, "create_module", None)
        return creator(spec) if creator is not None else None

    def exec_module(self, module: ModuleType) -> None:
        self._delegate.exec_module(module)
        from openviking.pyagfs.exceptions import AGFSPluginError

        _patch_openviking_task_store(module, AGFSPluginError)


class _OpenVikingTaskStoreFinder(importlib.abc.MetaPathFinder):
    _TARGET = "openviking.service.task_store"

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
            raise RuntimeError("OpenViking task-store module cannot be resolved")
        if not isinstance(spec.loader, importlib.abc.Loader):
            raise RuntimeError("OpenViking task-store loader cannot be guarded")
        spec.loader = _OpenVikingTaskStoreLoader(spec.loader)
        return spec


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
        module._MEMORY_EXTRACTION_MAX_RETRIES = EXTRACTION_MAX_RETRIES
        module.is_retryable_api_error = _extraction_retry_classifier(module.is_retryable_api_error)
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
sys.meta_path.insert(0, _OpenVikingTaskStoreFinder())
