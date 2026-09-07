from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import httpx
import pytest

import oamb.runtime.native_run as native
from oamb.artifacts.composition import capsule_execution_configuration_family_hash
from oamb.artifacts.store import ArtifactStore
from oamb.artifacts.validation.native import validate_native_capsule
from oamb.contracts.ids import canonical_sha256, ingestion_occurrence_id
from oamb.contracts.ports import (
    ArtifactStorePort,
    IngestionPlan,
)
from oamb.contracts.specifications import (
    INFRASTRUCTURE_RETRY_POLICY_HASH,
    CasePartitionSpec,
    ModelRole,
)
from oamb.contracts.states import ValidationDisposition
from oamb.memory_systems.hindsight import HindsightAdapter
from oamb.runtime.case_partition import build_case_partition_spec
from oamb.workloads.longmemeval import (
    LME_JUDGE_PROMPT_PACK_ID,
    LongMemEvalMessage,
    LongMemEvalRow,
    LongMemEvalSession,
    LongMemEvalWorkload,
    _build_bundle,
)
from oamb.workloads.visible_evidence import LME_VISIBLE_EVIDENCE_POLICY
from tests.e2e.test_native_recorded_exact_profile import (
    ANSWER_ROLE_ID,
    RUNTIME_BINDING_HASH,
    _bank_id,
    _model_factory,
    _RecordedHindsightService,
    _workload,
)

_FAILURE_DETAIL = (
    "Fact extraction failed: 1/1 chunks failed. "
    "First failures: chunk 0: APIConnectionError: Connection error."
)
_INVALID_JSON_DETAIL = (
    "Fact extraction failed: 1/1 chunks failed. "
    "First failures: chunk 0: JSONDecodeError: Invalid control character at: "
    "line 8 column 36 (char 243)"
)


def _two_source_workload() -> LongMemEvalWorkload:
    sessions = tuple(
        LongMemEvalSession(
            session_id=f"session-{ordinal}",
            raw_timestamp=f"2026/01/0{ordinal} (Thu) 00:00",
            canonical_timestamp=f"2026-01-0{ordinal}T00:00:00+00:00",
            messages=(
                LongMemEvalMessage(role="user", content=f"source {ordinal}", has_answer=True),
            ),
        )
        for ordinal in (1, 2)
    )
    row = LongMemEvalRow(
        source_row_number_1_indexed=1,
        question_id="question-1",
        question_type="multi-session",
        question="What does Alice care about?",
        answer="Alice prefers exact evidence.",
        raw_question_timestamp="2026/01/03 (Fri) 00:00",
        canonical_question_timestamp="2026-01-03T00:00:00+00:00",
        answer_session_ids=tuple(session.session_id for session in sessions),
        message_has_answer_session_ids=tuple(session.session_id for session in sessions),
        sessions=sessions,
    )
    return LongMemEvalWorkload(
        _build_bundle(
            _workload().resolve_sources(), (row,), workload_id="recorded-history-rebuild-v1"
        )
    )


class _RebuildingHindsightService(_RecordedHindsightService):
    """Recorded native REST with retained failed-scope state; no real provider writes."""

    def __init__(
        self,
        run_id: str,
        plans: tuple[IngestionPlan, ...],
        *,
        fail_first_history: bool,
        old_partial_count: int | None,
        journal: Path,
        failure_detail: str = _FAILURE_DETAIL,
    ) -> None:
        assert len(plans) == 1 and len(plans[0].ordered_source_units) == 2
        self.plan = plans[0]
        self.journal = journal
        self.occurrences = tuple(
            ingestion_occurrence_id(
                run_id,
                "hindsight",
                self.plan.ingestion_plan_id,
                history_attempt_ordinal=ordinal,
            )
            for ordinal in (1, 2, 3)
        )
        self.banks = tuple(_bank_id(occurrence) for occurrence in self.occurrences)
        super().__init__({bank: self.plan.ordered_source_units for bank in self.banks})
        self.fail_first_history = fail_first_history
        self.old_partial_count = old_partial_count
        self.failure_detail = failure_detail
        self.retains: list[tuple[str, dict[str, Any]]] = []
        self.recalls: list[str] = []
        self.failed_scope_state: dict[str, object] = {}

    def capture(self, kind: str, **fields: object) -> None:
        with self.journal.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps({"kind": kind, **fields}) + "\n")

    def __call__(self, request: httpx.Request) -> httpx.Response:
        bank = self._bank_id_from_path(request.url.path)
        if request.method == "PUT" and bank is not None and request.url.path.endswith(bank):
            self.capture("create", bank=bank)
        if request.method == "POST" and request.url.path.endswith("/memories"):
            assert bank is not None
            item = json.loads(request.content)["items"][0]
            self.retains.append((bank, item))
            self.capture("retain", bank=bank, item=item)
            if (
                self.fail_first_history
                and bank == self.banks[0]
                and item["document_id"] == self.plan.ordered_source_units[1].source_unit_id
            ):
                self.failed_scope_state = {
                    "bank_id": bank,
                    "partial_count": self.old_partial_count,
                    "sentinel": "OLD_SCOPE_ONLY_SENTINEL",
                    "accepted_source_ids": [self.plan.ordered_source_units[0].source_unit_id],
                }
                self.capture("failure", state=self.failed_scope_state)
                return httpx.Response(500, json={"detail": self.failure_detail})
        if request.method == "POST" and request.url.path.endswith("/memories/recall"):
            assert bank is not None
            self.recalls.append(bank)
            self.capture("recall", bank=bank)
        return super().__call__(request)


def _run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    run_id: str,
    fail_first_history: bool = True,
    old_partial_count: int | None = 3,
    control: native.NativeRunControl | None = None,
    partition: CasePartitionSpec | None = None,
    occurrence_run_id: str | None = None,
    recovery_parts: tuple[Path, ...] = (),
    stop_during_backoff: bool = False,
    failure_detail: str = _FAILURE_DETAIL,
) -> tuple[native.NativeRunArtifacts, _RebuildingHindsightService, list[int]]:
    workload = _two_source_workload()
    dataset = workload.resolve_sources()
    manifest = workload.build_case_manifest(dataset)
    plans = workload.iter_ingestion_plans(manifest)
    cases = workload.iter_case_plans(manifest)
    partition = partition or build_case_partition_spec(
        run_id=run_id,
        resolved_plan_hash=canonical_sha256(["history-rebuild-fixture-plan"]),
        cell_spec_hash=canonical_sha256(["history-rebuild-fixture-cell"]),
        dataset_manifest_hash=dataset.manifest_hash,
        case_manifest=manifest,
        case_plans=cases,
        requested_case_manifest_entry_ids=tuple(case.case_manifest_entry_id for case in cases),
        budget_policy_hash=canonical_sha256(["history-rebuild-fixture-budget"]),
        retry_policy_hash=INFRASTRUCTURE_RETRY_POLICY_HASH,
    )
    service = _RebuildingHindsightService(
        occurrence_run_id or run_id,
        plans,
        fail_first_history=fail_first_history,
        old_partial_count=old_partial_count,
        journal=tmp_path / f"{run_id}-requests.jsonl",
        failure_detail=failure_detail,
    )
    waits: list[int] = []

    async def wait(seconds: int) -> None:
        service.capture("wait", seconds=seconds)
        if stop_during_backoff:
            raise native.NativeRunInterrupted("fixture stop during pending history backoff")

    monkeypatch.setattr(native, "_infrastructure_retry_sleep", wait)

    def memory_factory(
        store: ArtifactStorePort, plans: tuple[IngestionPlan, ...]
    ) -> HindsightAdapter:
        assert plans == (service.plan,)
        return HindsightAdapter(
            store=store,
            base_url="https://hindsight.example",
            extraction_model="fixture-extractor",
            runtime_binding_hash=RUNTIME_BINDING_HASH,
            transport=httpx.MockTransport(service),
            internal_retry_count=10,
        )

    completed = native.run_native_vertical_slice(
        output_root=tmp_path,
        run_id=run_id,
        adapter_profile_id="hindsight-rest-v1",
        workload=workload,
        visible_evidence_policy=LME_VISIBLE_EVIDENCE_POLICY,
        artifact_store_factory=ArtifactStore,
        memory_factory=memory_factory,
        model_factory=lambda store: _model_factory(
            store,
            role=ModelRole.ANSWER,
            binding_id=ANSWER_ROLE_ID,
            model="fixture-answer-model",
            output="An ordinary incorrect answer.",
            thinking_effort="low",
        ),
        answer_role_binding_id=ANSWER_ROLE_ID,
        judge_model_factory=lambda store: _model_factory(
            store,
            role=ModelRole.JUDGE,
            binding_id=LME_JUDGE_PROMPT_PACK_ID,
            model="fixture-judge-model",
            output="no",
            thinking_effort="high",
        ),
        judge_role_binding_id=LME_JUDGE_PROMPT_PACK_ID,
        partition=partition,
        control=control,
        recovery_parts=recovery_parts,
        recovery_execution_configuration_family_hash=(
            capsule_execution_configuration_family_hash(recovery_parts[0])
            if recovery_parts
            else None
        ),
    )
    for line in service.journal.read_text().splitlines():
        event = json.loads(line)
        if event["kind"] == "create":
            service.created_bank_ids.add(event["bank"])
        elif event["kind"] == "retain":
            service.retains.append((event["bank"], event["item"]))
        elif event["kind"] == "failure":
            service.failed_scope_state = event["state"]
        elif event["kind"] == "recall":
            service.recalls.append(event["bank"])
        elif event["kind"] == "wait":
            waits.append(event["seconds"])
    return completed, service, waits


@pytest.mark.parametrize("old_partial_count", (0, 3, None))
@pytest.mark.parametrize(
    "failure_detail",
    (_FAILURE_DETAIL, _INVALID_JSON_DETAIL),
)
def test_hindsight_failure_retries_batch_in_same_bank_then_continues_partial_history(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    old_partial_count: int | None,
    failure_detail: str,
) -> None:
    completed, service, waits = _run(
        tmp_path,
        monkeypatch,
        run_id="recorded-history-rebuild",
        old_partial_count=old_partial_count,
        failure_detail=failure_detail,
    )
    bank = service.banks[0]
    assert service.created_bank_ids == {bank}
    expected = tuple(source.source_unit_id for source in service.plan.ordered_source_units)
    items = [item for target, item in service.retains if target == bank]
    assert tuple(item["document_id"] for item in items) == (
        expected[0],
        expected[1],
        expected[1],
        expected[1],
    )
    assert service.failed_scope_state["partial_count"] == old_partial_count
    assert service.failed_scope_state["sentinel"] == "OLD_SCOPE_ONLY_SENTINEL"
    assert service.recalls == [bank]
    assert waits == [10, 10]
    assert len(completed.ingestion_plan_records) == len(completed.case_records) == 1
    assert completed.ingestion_plan_records[0].ingestion_occurrence_id == service.occurrences[0]
    assert completed.case_records[0].ingestion_occurrence_id == service.occurrences[0]
    histories = sorted(
        (
            json.loads(path.read_bytes())
            for path in (completed.capsule_root / "source/history-attempts").glob("*.json")
        ),
        key=lambda value: value["history_attempt_ordinal"],
    )
    assert [record["status"] for record in histories] == ["ready"]
    assert histories[0]["history_attempt_ordinal"] == 1
    runs = [
        json.loads(path.read_bytes())
        for path in (completed.capsule_root / "source/run").glob("*.json")
    ]
    assert len(runs) == 1
    assert runs[0]["ingestion_occurrence_ids"] == [service.occurrences[0]]
    validation = validate_native_capsule(completed.capsule_root)
    assert validation.disposition == ValidationDisposition.VALIDATED, [
        (issue.rule_id, issue.code) for issue in validation.issues
    ]


def test_incorrect_answer_judge_no_and_empty_native_memory_do_not_rebuild(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    completed, service, waits = _run(
        tmp_path,
        monkeypatch,
        run_id="recorded-no-quality-retry",
        fail_first_history=False,
    )
    assert waits == []
    assert service.created_bank_ids == {service.banks[0]}
    assert service.recalls == [service.banks[0]]
    assert len(service.retains) == 2
    assert completed.case_records[0].evaluation_disposition == "judged"
    assert len(list((completed.capsule_root / "source/history-attempts").glob("*.json"))) == 1
    assert not list((completed.capsule_root / "source/history-retries").glob("*.json"))
