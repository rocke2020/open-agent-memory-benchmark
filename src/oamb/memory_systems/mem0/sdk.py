"""Comparison-ineligible Mem0 SDK profile and supervised IPC boundary."""

from __future__ import annotations

from oamb.contracts.ports import WorkerIPCPort, WorkerOutcome, WorkerReceipt, WorkerRequest

from .adapter import _ZeroDispatchMem0Adapter
from .profiles import MEM0_SDK_PROFILE


class Mem0SdkWorkerBoundary:
    """Forward typed work to an injected supervisor without importing Mem0."""

    def __init__(self, *, worker: WorkerIPCPort) -> None:
        self._worker = worker
        self._closed = False
        self._retired = False

    def request(self, request: WorkerRequest) -> WorkerReceipt:
        if self._closed:
            raise RuntimeError("Mem0 SDK worker boundary is closed")
        if self._retired:
            raise RuntimeError("Mem0 SDK worker boundary is retired")
        receipt = self._worker.request(request)
        if receipt.outcome is WorkerOutcome.UNKNOWN_OUTCOME:
            self._retired = True
        return receipt

    def close(self) -> None:
        if self._closed:
            return
        try:
            self._worker.close()
        finally:
            self._closed = True
            self._retired = True


class Mem0SdkAdapter(_ZeroDispatchMem0Adapter):
    """Reject v2.0.19 before constructing or dispatching an SDK client."""

    def __init__(self, *, worker_boundary: Mem0SdkWorkerBoundary | None = None) -> None:
        super().__init__(profile=MEM0_SDK_PROFILE)
        self._worker_boundary = worker_boundary
        self._closed = False

    async def close(self) -> None:
        if self._closed:
            return
        if self._worker_boundary is not None:
            self._worker_boundary.close()
        self._closed = True


__all__ = ["Mem0SdkAdapter", "Mem0SdkWorkerBoundary"]
