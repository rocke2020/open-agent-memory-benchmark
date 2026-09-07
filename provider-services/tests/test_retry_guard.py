from __future__ import annotations

import asyncio
import importlib.util
import json
import os
import types
import unittest
from pathlib import Path
from unittest.mock import patch

MODULE_PATH = Path(__file__).resolve().parents[1] / "retry-guard" / "sitecustomize.py"
PHASE2_FIXTURE = Path(__file__).parent / "fixtures" / "openviking_phase2_retry.py.txt"


def _load_guard():
    spec = importlib.util.spec_from_file_location("oamb_retry_guard_test", MODULE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    with patch.dict(os.environ, {"OAMB_EXTRACTION_MAX_RETRIES": "10"}):
        spec.loader.exec_module(module)
    return module


class RetryGuardTests(unittest.TestCase):
    def test_pinned_task_store_loader_makes_concurrent_first_directory_create_idempotent(
        self,
    ) -> None:
        guard = _load_guard()

        class AGFSPluginError(RuntimeError):
            pass

        AGFSPluginError.__module__ = "openviking.pyagfs.exceptions"

        class RacingAGFS:
            def __init__(self) -> None:
                self.created = False
                self.arrivals = 0
                self.ready = asyncio.Event()

            async def mkdir(self, _path: str) -> None:
                missing = not self.created
                self.arrivals += 1
                if self.arrivals == 2:
                    self.ready.set()
                await self.ready.wait()
                if missing and not self.created:
                    self.created = True
                    return
                raise AGFSPluginError(
                    "plugin error: failed to create directory: File exists (os error 17)"
                )

        class PinnedTaskStoreDelegate:
            def exec_module(self, module: types.ModuleType) -> None:
                module.PersistentTaskStore = type(
                    "PersistentTaskStore",
                    (),
                    {"_mkdir_if_missing": _pinned_mkdir_if_missing},
                )

        exceptions = types.ModuleType("openviking.pyagfs.exceptions")
        exceptions.AGFSPluginError = AGFSPluginError
        module = types.ModuleType("openviking.service.task_store")
        with patch.dict(
            os.sys.modules,
            {"openviking.pyagfs.exceptions": exceptions},
        ):
            guard._OpenVikingTaskStoreLoader(PinnedTaskStoreDelegate()).exec_module(module)
        store = module.PersistentTaskStore()
        store._agfs = RacingAGFS()

        async def create_concurrently() -> None:
            await asyncio.gather(
                store._mkdir_if_missing("/local/account/_system/tasks/user"),
                store._mkdir_if_missing("/local/account/_system/tasks/user"),
            )

        asyncio.run(create_concurrently())

    def test_task_store_patch_propagates_other_plugin_errors(self) -> None:
        guard = _load_guard()

        class AGFSPluginError(RuntimeError):
            pass

        class FailingAGFS:
            async def mkdir(self, _path: str) -> None:
                raise AGFSPluginError("plugin error: input/output error")

        module = types.ModuleType("openviking.service.task_store")
        module.PersistentTaskStore = type(
            "PersistentTaskStore",
            (),
            {"_mkdir_if_missing": _pinned_mkdir_if_missing},
        )
        guard._patch_openviking_task_store(module, AGFSPluginError)
        store = module.PersistentTaskStore()
        store._agfs = FailingAGFS()

        with self.assertRaisesRegex(AGFSPluginError, "input/output error"):
            asyncio.run(store._mkdir_if_missing("/local/account/_system/tasks/user"))

    def test_pinned_phase2_wrapper_consumes_injected_classifier_for_eleven_attempts(self) -> None:
        guard = _load_guard()
        session = types.ModuleType("pinned_openviking_session")
        session.__dict__["is_retryable_api_error"] = lambda _error: False
        exec(compile(PHASE2_FIXTURE.read_bytes(), str(PHASE2_FIXTURE), "exec"), session.__dict__)
        session._MEMORY_EXTRACTION_MAX_RETRIES = guard.EXTRACTION_MAX_RETRIES
        session.is_retryable_api_error = guard._extraction_retry_classifier(
            session.is_retryable_api_error
        )
        attempts = 0

        async def invalid_json() -> None:
            nonlocal attempts
            attempts += 1
            raise json.JSONDecodeError("bad JSON", "{", 1)

        with self.assertRaises(json.JSONDecodeError):
            session.asyncio.run(session._run_retryable_phase2_step("memory", invalid_json))
        self.assertEqual(attempts, 11)

    def test_extraction_classifier_retries_json_and_schema_failures_only(self) -> None:
        guard = _load_guard()

        class ValidationError(ValueError):
            __module__ = "pydantic_core._pydantic_core"

        def original(error: Exception) -> bool:
            return str(error) == "transient"

        classifier = guard._extraction_retry_classifier(original)

        self.assertIs(classifier(json.JSONDecodeError("bad JSON", "{", 1)), True)
        self.assertIs(classifier(ValidationError("bad schema")), True)
        self.assertIs(classifier(RuntimeError("transient")), True)
        self.assertIs(classifier(ValueError("ordinary provider failure")), False)


async def _pinned_mkdir_if_missing(self: object, path: str) -> None:
    try:
        await self._agfs.mkdir(path)
    except Exception as exc:
        if "already exists" in str(exc).lower():
            return
        raise
