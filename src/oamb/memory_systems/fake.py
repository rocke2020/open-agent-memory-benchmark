"""In-memory deterministic adapter for offline runtime verification."""

from __future__ import annotations

import hashlib
from collections import defaultdict

from oamb.contracts.ids import canonical_json_bytes, canonical_sha256
from oamb.contracts.ports import (
    ArtifactStorePort,
    CapabilitySet,
    IngestionReceipt,
    IngestionRequest,
    InventoryReceipt,
    NativeEvidenceBatch,
    NativeEvidenceCandidate,
    ProjectionReceipt,
    RawPayloadSealRequest,
    RawReferenceHandle,
    ReadinessReceipt,
    ReadinessRequest,
    RetrievalRequest,
    RuntimeResolution,
    ScopeAllocationRequest,
    ScopeReceipt,
    StateDigestReceipt,
)


class ScriptedFakeMemorySystem:
    """Models readiness, partial ingestion, retrieval order, and close offline."""

    def __init__(
        self,
        store: ArtifactStorePort,
        *,
        delayed_readiness_plan_ids: tuple[str, ...] = (),
        partial_ingestion_plan_ids: tuple[str, ...] = (),
    ) -> None:
        self._store = store
        self._delayed = frozenset(delayed_readiness_plan_ids)
        self._partial = frozenset(partial_ingestion_plan_ids)
        self._plan_by_scope: dict[str, str] = {}
        self._accepted_by_scope: dict[str, tuple[str, ...]] = {}
        self._readiness_checks: defaultdict[str, int] = defaultdict(int)
        self._closed = False

    async def resolve(self) -> RuntimeResolution:
        runtime_binding_hash = canonical_sha256(["oamb-fake-runtime-v1"])
        return RuntimeResolution(
            memory_system_id="fake-memory",
            runtime_binding_hash=runtime_binding_hash,
            raw_reference=self._raw_reference(
                {
                    "operation": "resolve",
                    "memory_system_id": "fake-memory",
                    "runtime_binding_hash": runtime_binding_hash,
                }
            ),
        )

    async def capabilities(self) -> CapabilitySet:
        return CapabilitySet(
            capability_ids=("create-only", "delayed-readiness", "projection"),
            provider_order_preserved=True,
            native_reranking_disabled=True,
        )

    async def allocate_ingestion_scope(
        self,
        request: ScopeAllocationRequest,
    ) -> ScopeReceipt:
        self._require_open()
        scope_id = canonical_sha256(
            ["oamb-fake-scope-v1", request.ingestion_occurrence_id, request.ingestion_plan_id]
        )
        self._plan_by_scope[scope_id] = request.ingestion_plan_id
        return ScopeReceipt(
            ingestion_occurrence_id=request.ingestion_occurrence_id,
            scope_id=scope_id,
            raw_reference=self._raw_reference(
                {
                    "operation": "allocate_scope",
                    "ingestion_occurrence_id": request.ingestion_occurrence_id,
                    "ingestion_plan_id": request.ingestion_plan_id,
                    "scope_id": scope_id,
                }
            ),
        )

    async def ingest(self, request: IngestionRequest) -> IngestionReceipt:
        self._require_open()
        plan_id = self._plan_by_scope[request.scope.scope_id]
        source_ids = tuple(item.source_unit_id for item in request.ordered_source_units)
        if plan_id in self._partial and source_ids:
            accepted = source_ids[:1]
            rejected = source_ids[1:]
        else:
            accepted = source_ids
            rejected = ()
        self._accepted_by_scope[request.scope.scope_id] = accepted
        return IngestionReceipt(
            ingestion_occurrence_id=request.scope.ingestion_occurrence_id,
            accepted_source_unit_ids=accepted,
            rejected_source_unit_ids=rejected,
            raw_references=(
                self._raw_reference(
                    {
                        "operation": "ingest",
                        "ingestion_occurrence_id": request.scope.ingestion_occurrence_id,
                        "scope_id": request.scope.scope_id,
                        "source_units": tuple(
                            {
                                "source_unit_id": item.source_unit_id,
                                "payload_sha256": item.payload_sha256,
                                "payload": item.payload_bytes.decode("utf-8"),
                            }
                            for item in request.ordered_source_units
                        ),
                        "accepted_source_unit_ids": accepted,
                        "rejected_source_unit_ids": rejected,
                    }
                ),
            ),
        )

    async def wait_ready(self, request: ReadinessRequest) -> ReadinessReceipt:
        self._require_open()
        plan_id = self._plan_by_scope[request.scope.scope_id]
        self._readiness_checks[request.scope.scope_id] += 1
        source_inventory_matches = request.expected_source_unit_ids == self._accepted_by_scope.get(
            request.scope.scope_id, ()
        )
        ready = source_inventory_matches and plan_id not in self._partial
        if plan_id in self._delayed and self._readiness_checks[request.scope.scope_id] == 1:
            ready = False
        return ReadinessReceipt(
            ingestion_occurrence_id=request.scope.ingestion_occurrence_id,
            ready=ready,
            evidence_references=(
                self._raw_reference(
                    {
                        "operation": "wait_ready",
                        "ingestion_occurrence_id": request.scope.ingestion_occurrence_id,
                        "scope_id": request.scope.scope_id,
                        "check_ordinal": self._readiness_checks[request.scope.scope_id],
                        "ready": ready,
                        "expected_source_unit_ids": request.expected_source_unit_ids,
                    }
                ),
            ),
        )

    async def inventory(self, scope: ScopeReceipt) -> InventoryReceipt:
        self._require_open()
        return InventoryReceipt(
            ingestion_occurrence_id=scope.ingestion_occurrence_id,
            ordered_source_unit_ids=self._accepted_by_scope.get(scope.scope_id, ()),
            raw_reference=self._raw_reference(
                {
                    "operation": "inventory",
                    "ingestion_occurrence_id": scope.ingestion_occurrence_id,
                    "scope_id": scope.scope_id,
                    "ordered_source_unit_ids": self._accepted_by_scope.get(scope.scope_id, ()),
                }
            ),
        )

    async def state_digest(self, scope: ScopeReceipt) -> StateDigestReceipt:
        self._require_open()
        state_sha256 = canonical_sha256(
            ["oamb-fake-state-v1", self._accepted_by_scope.get(scope.scope_id, ())]
        )
        return StateDigestReceipt(
            ingestion_occurrence_id=scope.ingestion_occurrence_id,
            state_sha256=state_sha256,
            raw_reference=self._raw_reference(
                {
                    "operation": "state_digest",
                    "ingestion_occurrence_id": scope.ingestion_occurrence_id,
                    "scope_id": scope.scope_id,
                    "state_sha256": state_sha256,
                }
            ),
        )

    async def project(self, scope: ScopeReceipt) -> ProjectionReceipt:
        return ProjectionReceipt(
            inventory=await self.inventory(scope),
            state_digest=await self.state_digest(scope),
        )

    async def retrieve(self, request: RetrievalRequest) -> NativeEvidenceBatch:
        self._require_open()
        query = request.query_bytes.decode("utf-8")
        candidates: tuple[NativeEvidenceCandidate, ...]
        if "shared" in query or "token" in query or "grounded" in query:
            candidates = (
                NativeEvidenceCandidate(
                    native_id="fake-native-1",
                    native_rank_1_indexed=1,
                    content="shared answer alpha",
                    native_score="0.900",
                ),
                NativeEvidenceCandidate(
                    native_id="fake-native-2",
                    native_rank_1_indexed=2,
                    content="shared answer beta",
                    native_score="0.800",
                ),
            )
        else:
            candidates = ()
        return NativeEvidenceBatch(
            raw_reference=self._raw_reference(
                {
                    "operation": "retrieve",
                    "case_occurrence_id": request.case_occurrence_id,
                    "scope_id": request.scope.scope_id,
                    "query": query,
                    "query_sha256": hashlib.sha256(request.query_bytes).hexdigest(),
                    "top_k": request.top_k,
                    "candidates": tuple(
                        {
                            "native_id": candidate.native_id,
                            "native_rank_1_indexed": candidate.native_rank_1_indexed,
                            "content": candidate.content,
                            "native_score": candidate.native_score,
                        }
                        for candidate in candidates
                    ),
                }
            ),
            candidates=candidates,
        )

    async def close(self) -> None:
        self._closed = True

    def _require_open(self) -> None:
        if self._closed:
            raise RuntimeError("fake memory system is closed")

    def _raw_reference(self, document: object) -> RawReferenceHandle:
        payload = canonical_json_bytes(document)
        sha256 = hashlib.sha256(payload).hexdigest()
        return self._store.seal_raw(
            RawPayloadSealRequest(
                sha256=sha256,
                media_type="application/json",
                compression="gzip",
                payload_bytes=payload,
            )
        )
