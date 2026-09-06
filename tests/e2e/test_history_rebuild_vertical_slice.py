from __future__ import annotations

import asyncio
import gzip
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import httpx
import pytest

import oamb.runtime.native_run as native
from oamb.artifacts.composition import compose_capsules
from oamb.artifacts.store import ArtifactStore
from oamb.artifacts.validation.native import validate_native_capsule
from oamb.artifacts.validation.source_root import validate_source_root
from oamb.contracts.ids import canonical_sha256, ingestion_occurrence_id
from oamb.contracts.ports import (
    ArtifactStorePort,
    IngestionPlan,
    RawReferenceHandle,
    SettledTransientIngestionFailure,
)
from oamb.contracts.specifications import (
    INFRASTRUCTURE_RETRY_POLICY_HASH,
    CasePartitionSpec,
    ModelRole,
)
from oamb.contracts.states import ValidationDisposition
from oamb.memory_systems.hindsight import HindsightAdapter
from oamb.reporting.native_reduce import reduce_native_run_report
from oamb.reporting.offline_renderer import offline_asset_hashes, offline_renderer_hash
from oamb.reporting.publication import build_report_derivation
from oamb.reporting.roots import build_report_spec
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
            assert bank != self.failed_scope_state.get("bank_id")
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
            internal_retry_count=0,
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
    ("failure_detail", "failure_kind"),
    (
        (_FAILURE_DETAIL, "supplier_connection"),
        (_INVALID_JSON_DETAIL, "supplier_invalid_json_output"),
    ),
)
def test_original_hindsight_failure_rebuilds_complete_history_in_fresh_bank(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    old_partial_count: int | None,
    failure_detail: str,
    failure_kind: str,
) -> None:
    completed, service, waits = _run(
        tmp_path,
        monkeypatch,
        run_id="recorded-history-rebuild",
        old_partial_count=old_partial_count,
        failure_detail=failure_detail,
    )
    old_bank, fresh_bank, _unused_bank = service.banks
    assert service.created_bank_ids == {old_bank, fresh_bank}
    expected = tuple(source.source_unit_id for source in service.plan.ordered_source_units)
    for bank in (old_bank, fresh_bank):
        items = [item for target, item in service.retains if target == bank]
        assert tuple(item["document_id"] for item in items) == expected
        assert tuple(item["content"].encode() for item in items) == tuple(
            source.payload_bytes for source in service.plan.ordered_source_units
        )
    assert service.failed_scope_state["partial_count"] == old_partial_count
    assert service.failed_scope_state["sentinel"] == "OLD_SCOPE_ONLY_SENTINEL"
    assert service.recalls == [fresh_bank]
    assert waits == [1]
    assert len(completed.ingestion_plan_records) == len(completed.case_records) == 1
    assert completed.ingestion_plan_records[0].ingestion_occurrence_id == service.occurrences[1]
    assert completed.case_records[0].ingestion_occurrence_id == service.occurrences[1]
    histories = sorted(
        (
            json.loads(path.read_bytes())
            for path in (completed.capsule_root / "source/history-attempts").glob("*.json")
        ),
        key=lambda value: value["history_attempt_ordinal"],
    )
    assert [record["status"] for record in histories] == ["retryable_failed_settled", "ready"]
    assert histories[0]["failure_kind"] == failure_kind
    runs = [
        json.loads(path.read_bytes())
        for path in (completed.capsule_root / "source/run").glob("*.json")
    ]
    assert len(runs) == 1
    assert set(runs[0]["ingestion_occurrence_ids"]) == set(service.occurrences[:2])
    validation = validate_native_capsule(completed.capsule_root)
    assert validation.disposition == ValidationDisposition.VALIDATED, [
        (issue.rule_id, issue.code) for issue in validation.issues
    ]
    composed = compose_capsules((completed.capsule_root,), tmp_path / "composed")
    composed_validation = validate_source_root(composed.capsule_root)
    assert composed_validation.disposition == ValidationDisposition.VALIDATED, (
        composed_validation.issues
    )
    assert len(composed.composition.ordered_contributions) == 1


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


@pytest.mark.parametrize(
    "changes", ({"failure_kind": "supplier_connection"}, {"settlement_status_code": 200})
)
def test_validator_rechecks_invalid_json_failure_classification(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, changes: dict[str, object]
) -> None:
    from oamb.contracts.evidence import history_attempt_id
    from oamb.contracts.ids import canonical_json_bytes
    from tests.e2e.test_native_fixture_vertical_slice import _reseal_manifest

    completed, _service, _waits = _run(
        tmp_path,
        monkeypatch,
        run_id="recorded-invalid-json-classification-tamper",
        failure_detail=_INVALID_JSON_DETAIL,
    )
    validation = validate_native_capsule(completed.capsule_root)
    assert validation.disposition == ValidationDisposition.VALIDATED, validation.issues
    history_path = next(
        path
        for path in (completed.capsule_root / "source/history-attempts").glob("*.json")
        if json.loads(path.read_bytes())["status"] == "retryable_failed_settled"
    )
    history = json.loads(history_path.read_bytes())
    history.update(changes)
    fields = {key: value for key, value in history.items() if key != "history_attempt_id"}
    history["history_attempt_id"] = history_attempt_id(fields)
    history_path.write_bytes(canonical_json_bytes(history))
    _reseal_manifest(completed.capsule_root)

    invalid = validate_native_capsule(completed.capsule_root)
    assert invalid.disposition != ValidationDisposition.VALIDATED
    assert "history-failure-classification-mismatch" in {issue.code for issue in invalid.issues}


@pytest.mark.parametrize("failure_detail", (_FAILURE_DETAIL, _INVALID_JSON_DETAIL))
def test_rebuilt_native_capsule_publishes_one_result_and_retains_failed_attempts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure_detail: str,
) -> None:
    completed, service, _waits = _run(
        tmp_path,
        monkeypatch,
        run_id="recorded-rebuild-publication",
        failure_detail=failure_detail,
    )
    validation = validate_native_capsule(completed.capsule_root)
    assert validation.disposition == ValidationDisposition.VALIDATED, validation.issues
    report_spec = build_report_spec(
        report_kind="run",
        audience="public",
        preview_max_field_bytes=4096,
        preview_total_bytes=65536,
        display_field_ids=("identity", "completion", "quality", "usage", "limitations"),
        renderer_hash=offline_renderer_hash(),
        asset_hashes=offline_asset_hashes(),
        export_profile_selector_id="public-run-v1",
        export_profile_selector_version=1,
    )
    report = reduce_native_run_report(
        completed.capsule_root,
        validation,
        report_spec=report_spec,
    )
    assert report.summary.completed_cases == 1
    assert report.case_occurrence_ids == (completed.case_records[0].case_occurrence_id,)
    assert report.ingestion_occurrence_ids == (service.occurrences[1],)
    assert report.metric_summaries
    assert all(metric.score_numerator == 0 for metric in report.metric_summaries)
    attempts = [
        json.loads(path.read_bytes())
        for path in (completed.capsule_root / "source/attempts").glob("*.json")
    ]
    failures = [item for item in attempts if item["outcome"] == "failed"]
    assert len(failures) == 1
    assert failures[0]["parent_id"] == service.occurrences[0]
    assert failures[0]["attempt_id"] in report.attempt_ids
    assert set(report.attempt_ids) == {item["attempt_id"] for item in attempts}
    accounting = {
        entry.record_id: (
            entry.record_kind,
            json.loads((completed.capsule_root / entry.relative_path).read_bytes()),
        )
        for entry in completed.manifest.source_entries
        if entry.record_kind in {"token_usage_record", "resource_usage_record", "cost_record"}
    }
    reported_ids: set[str] = set()
    measurements: set[tuple[str, str]] = set()
    for line in report.measurement_lines:
        for record_id in line.source_record_ids:
            measurement = (record_id, line.dimension_id)
            assert measurement not in measurements, "physical measurement counted twice"
            measurements.add(measurement)
            reported_ids.add(record_id)
            kind, record = accounting[record_id]
            if kind == "cost_record":
                assert line.indexing_view == record["indexing_view"]
            elif record["parent_kind"] == "ingestion_plan":
                assert record["parent_id"] in service.occurrences[:2]
                assert line.indexing_view == (
                    "attempted"
                    if record["parent_id"] == service.occurrences[0]
                    else "final_contribution"
                )
            else:
                assert line.indexing_view == "not_applicable"
    assert reported_ids == set(accounting)
    prompt_entry = next(
        entry
        for entry in completed.manifest.source_entries
        if entry.record_kind == "raw_payload"
        and entry.record_id == completed.case_records[0].prompt_raw_ref
    )
    prompt = gzip.decompress((completed.capsule_root / prompt_entry.relative_path).read_bytes())
    assert b"OLD_SCOPE_ONLY_SENTINEL" not in prompt
    publication = build_report_derivation(
        model=report,
        report_spec=report_spec,
        ordered_source_bindings=report.ordered_source_bindings,
        evidence_validations=(validation,),
        evidence_validation_targets=(completed.capsule_root,),
        transform_spec_hash="c" * 64,
        schema_versions=("run_report_model@3", "report_artifact_manifest@2"),
        output_root=tmp_path / "reports",
        committed_at=datetime(2026, 9, 6, tzinfo=UTC),
    )
    assert publication.report_path.is_file()
    assert "<!doctype html>" in publication.report_path.read_text().lower()


def test_history_permit_covers_backoff_without_blocking_an_independent_history(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Exercise the real pipeline/rebuild loop; typed evidence is covered above."""
    backoff_started, release_backoff = asyncio.Event(), asyncio.Event()
    b_started, release_b, c_started = asyncio.Event(), asyncio.Event(), asyncio.Event()
    started: list[tuple[str, int]] = []
    active_writes: set[tuple[str, int]] = set()
    high_water = 0
    raw = json.dumps({"detail": _FAILURE_DETAIL}).encode()

    async def ingest_one(**kwargs: Any) -> tuple[tuple[str], dict[str, object]]:
        nonlocal high_water
        plan_id = kwargs["plans"][0].ingestion_plan_id
        ordinal = kwargs["history_attempt_ordinal"]
        identity = (plan_id, ordinal)
        assert not any(plan == plan_id for plan, _ in active_writes)
        active_writes.add(identity)
        high_water = max(high_water, len(active_writes))
        started.append(identity)
        try:
            if plan_id == "a" and ordinal == 1:
                await b_started.wait()
                raise SettledTransientIngestionFailure(
                    "recorded settled extraction failure",
                    settlement_basis="hindsight_sync_extraction_drained_v1",
                    internal_retry_count=0,
                    failure_kind="supplier_connection",
                    status_code=500,
                    raw_reference=RawReferenceHandle(hashlib.sha256(raw).hexdigest()),
                    raw_response_bytes=raw,
                )
            if plan_id == "b":
                b_started.set()
                await release_b.wait()
            if plan_id == "c":
                c_started.set()
            if plan_id == "a" and ordinal == 2:
                assert release_backoff.is_set()
            return (f"ready-{plan_id}",), {plan_id: object()}
        finally:
            active_writes.remove(identity)

    async def wait(seconds: int) -> None:
        assert seconds == 1
        assert ("a", 1) not in active_writes
        backoff_started.set()
        await release_backoff.wait()

    def seal_attempt(**kwargs: Any) -> SimpleNamespace:
        error = kwargs.get("error")
        assert error is None or isinstance(error, SettledTransientIngestionFailure)
        return SimpleNamespace(status="retryable_failed_settled" if error else "ready")

    monkeypatch.setattr(native, "_execute_ingestion_plans_serial", ingest_one)
    monkeypatch.setattr(native, "_infrastructure_retry_sleep", wait)
    monkeypatch.setattr(native, "_seal_history_attempt", seal_attempt)
    monkeypatch.setattr(
        native,
        "_seal_history_retry",
        lambda state, failed, *, scheduled: (
            SimpleNamespace(backoff_seconds=1) if scheduled else None
        ),
    )
    monkeypatch.setattr(native, "_claim_history_successor", lambda state, event: "fixture-claim")

    async def scenario() -> None:
        plans = tuple(
            SimpleNamespace(ingestion_plan_id=name, ordered_case_manifest_entry_ids=())
            for name in ("a", "b", "c")
        )
        state: Any = SimpleNamespace(
            control=SimpleNamespace(
                max_parallel_history_ingestions=2,
                max_parallel_questions=1,
                max_retries_per_operation=1,
            ),
            history_allowances={},
            history_events={},
            run_id="fixture-history-scheduling",
            history_occurrence_bindings={},
            publish_history_progress=None,
            timestamp=lambda: datetime(2026, 9, 6, tzinfo=UTC),
        )
        operation = asyncio.create_task(
            native._execute_history_question_pipeline(
                state=state,
                workload=state,
                memory=state,
                answer_model=state,
                judge_model=None,
                plans=cast(tuple[IngestionPlan, ...], plans),
                case_plans=(),
                memory_system_id="hindsight",
                runtime_binding_hash=RUNTIME_BINDING_HASH,
                adapter_profile_id="hindsight-rest-v1",
                visible_evidence_policy=LME_VISIBLE_EVIDENCE_POLICY,
                answer_role_binding_id=ANSWER_ROLE_ID,
                judge_role_binding_id=None,
            )
        )
        try:
            await asyncio.wait_for(backoff_started.wait(), timeout=1)
            assert started == [("a", 1), ("b", 1)]
            assert not c_started.is_set(), "backoff released A's history permit"
            release_b.set()
            await asyncio.wait_for(c_started.wait(), timeout=1)
            assert ("a", 2) not in started
            release_backoff.set()
            result: Any = await asyncio.wait_for(operation, timeout=1)
            assert result == (("ready-a", "ready-b", "ready-c"), ())
            assert started == [("a", 1), ("b", 1), ("c", 1), ("a", 2)]
            assert high_water == 2 and not active_writes
        except TimeoutError:
            if operation.done():
                operation.result()
            raise
        finally:
            if not operation.done():
                operation.cancel()
            await asyncio.gather(operation, return_exceptions=True)

    asyncio.run(scenario())
