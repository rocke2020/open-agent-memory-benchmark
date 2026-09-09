from __future__ import annotations

import json
from pathlib import Path

import pytest

from oamb.artifacts.store import ArtifactStore
from oamb.artifacts.validation.native import validate_native_capsule
from oamb.contracts.ids import canonical_json_bytes, canonical_sha256
from oamb.contracts.ports import (
    ModelCallFailure,
    ModelReceipt,
    ModelRequest,
    RawPayloadSealRequest,
    RawReferenceHandle,
)
from oamb.contracts.states import ValidationDisposition
from oamb.runtime.native_results import _model_outer_attempt_count
from oamb.runtime.native_run import NativeRunArtifacts, run_native_vertical_slice
from oamb.workloads.visible_evidence import LME_VISIBLE_EVIDENCE_POLICY
from tests.e2e.test_native_fixture_vertical_slice import (
    _answer_for_prompt,
    _memory_factory,
    _NativeFixtureJudgeWorkload,
    _RecordedNativeModel,
)


class CorrectableModel(_RecordedNativeModel):
    async def complete(self, request: ModelRequest) -> ModelReceipt:
        first = len(request.messages) == 1
        output = (
            ("" if request.stage == "answer" else "perhaps")
            if first
            else _answer_for_prompt(request.messages[0][1])
        )
        if not first:
            assert request.messages[-2] == (
                "assistant",
                "" if request.stage == "answer" else "perhaps",
            )
            assert "previous response failed output validation" in request.messages[-1][1]
        payload = {
            "operation": "model_complete",
            "attempt_id": request.attempt_id,
            "stage": request.stage,
            "prompt": request.messages[0][1],
            "output_text": output,
        }
        raw = self._store.seal_raw(
            RawPayloadSealRequest(
                sha256=canonical_sha256(payload),
                media_type="application/json",
                compression="gzip",
                payload_bytes=canonical_json_bytes(payload),
            )
        )
        return ModelReceipt(
            raw_reference=raw,
            output_text=output,
            usage_reference_ids=(),
            model="fixture-model",
            raw_response_bytes=canonical_json_bytes(payload),
        )


class ExhaustedJudge(CorrectableModel):
    malformed_output = "perhaps"

    async def complete(self, request: ModelRequest) -> ModelReceipt:
        payload = {
            "operation": "model_complete",
            "attempt_id": request.attempt_id,
            "stage": request.stage,
            "prompt": request.messages[0][1],
            "output_text": self.malformed_output,
        }
        raw = self._store.seal_raw(
            RawPayloadSealRequest(
                sha256=canonical_sha256(payload),
                media_type="application/json",
                compression="gzip",
                payload_bytes=canonical_json_bytes(payload),
            )
        )
        return ModelReceipt(
            raw_reference=raw,
            output_text=self.malformed_output,
            usage_reference_ids=(),
            model="fixture-model",
            raw_response_bytes=canonical_json_bytes(payload),
        )


class ExhaustedAnswer(ExhaustedJudge):
    malformed_output = ""


class TransportThenExhaustedJudge(ExhaustedJudge):
    def __init__(self, store: ArtifactStore) -> None:
        super().__init__(store)
        self._failed_messages: set[tuple[tuple[str, str], ...]] = set()

    async def complete(self, request: ModelRequest) -> ModelReceipt:
        if request.messages not in self._failed_messages:
            self._failed_messages.add(request.messages)
            payload = canonical_json_bytes(
                {
                    "operation": "model_complete",
                    "attempt_id": request.attempt_id,
                    "stage": request.stage,
                    "failure": "HTTP 500",
                }
            )
            raise ModelCallFailure(
                "HTTP 500",
                raw_reference=RawReferenceHandle(canonical_sha256(json.loads(payload))),
                raw_response_bytes=payload,
                usage_reference_ids=(),
                retryable=True,
                failure_kind="supplier_error",
                supplier_status_code=500,
            )
        return await super().complete(request)


class UnauthorizedModel(CorrectableModel):
    async def complete(self, request: ModelRequest) -> ModelReceipt:
        payload = canonical_json_bytes(
            {
                "operation": "model_complete",
                "attempt_id": request.attempt_id,
                "stage": request.stage,
                "failure": "HTTP 401",
            }
        )
        raw = self._store.seal_raw(
            RawPayloadSealRequest(
                sha256=canonical_sha256(json.loads(payload)),
                media_type="application/json",
                compression="gzip",
                payload_bytes=payload,
            )
        )
        raise ModelCallFailure(
            "HTTP 401",
            raw_reference=raw,
            raw_response_bytes=payload,
            usage_reference_ids=(),
            retryable=False,
            failure_kind="authentication_error",
            supplier_status_code=401,
        )


def _run(
    tmp_path: Path,
    *,
    answer: type[CorrectableModel] = CorrectableModel,
    judge: type[CorrectableModel] = CorrectableModel,
) -> NativeRunArtifacts:
    return run_native_vertical_slice(
        output_root=tmp_path / "capsules",
        run_id="model-format-retry",
        adapter_profile_id="recorded-native-fixture-v1",
        workload=_NativeFixtureJudgeWorkload(),
        visible_evidence_policy=LME_VISIBLE_EVIDENCE_POLICY,
        artifact_store_factory=ArtifactStore,
        memory_factory=_memory_factory,
        model_factory=answer,
        answer_role_binding_id="recorded-answer-v1",
        judge_model_factory=judge,
        judge_role_binding_id="fake-judge-v1",
    )


def test_answer_and_judge_format_correction_keep_separate_attempts_and_original_prompt(
    tmp_path: Path,
) -> None:
    completed = _run(tmp_path)
    assert all(record.state == "completed" for record in completed.case_records)
    attempts = [
        json.loads(path.read_bytes())
        for path in (completed.capsule_root / "source/attempts").glob("*.json")
    ]
    judged = next(
        record for record in completed.case_records if record.evaluation_disposition == "judged"
    )
    for stage in ("answer", "judge"):
        owned = [
            attempt
            for attempt in attempts
            if attempt["parent_id"] == judged.case_occurrence_id and attempt["stage"] == stage
        ]
        assert len(owned) == 2
        failed_attempt = next(attempt for attempt in owned if attempt["outcome"] == "failed")
        failure_receipt = json.loads(
            (
                completed.capsule_root
                / "source/attempt-receipts"
                / f"{failed_attempt['attempt_id']}.json"
            ).read_bytes()
        )
        assert failure_receipt["failure_kind"] == "output_contract_error"
        assert sorted(attempt["outcome"] for attempt in owned) == ["failed", "succeeded"]
        assert {attempt["attempt_id"] for attempt in owned} <= set(judged.attempt_ids)


def test_judge_format_exhaustion_retains_answer_and_continues_other_cases(tmp_path: Path) -> None:
    completed = _run(tmp_path, judge=ExhaustedJudge)
    failed = [record for record in completed.case_records if record.error_stage == "judge"]
    assert len(failed) == 1
    assert failed[0].evaluation_disposition == "unjudged"
    assert failed[0].answer_raw_ref and failed[0].parsed_answer_sha256
    assert failed[0].metric_numerator is None and failed[0].metric_denominator is None
    assert any(record.state == "completed" for record in completed.case_records)
    attempts = [
        json.loads(path.read_bytes())
        for path in (completed.capsule_root / "source/attempts").glob("*.json")
    ]
    judges = [attempt for attempt in attempts if attempt["stage"] == "judge"]
    assert len(judges) == 6
    assert all(attempt["outcome"] == "failed" for attempt in judges)


def test_answer_format_exhaustion_seals_valid_provider_failed_cases(tmp_path: Path) -> None:
    completed = _run(tmp_path, answer=ExhaustedAnswer)

    assert completed.case_records
    assert all(record.state == "error" for record in completed.case_records)
    assert all(record.error_stage == "answer" for record in completed.case_records)
    assert all(record.evaluation_disposition == "not_run" for record in completed.case_records)
    assert all(record.answer_raw_ref is None for record in completed.case_records)
    assert all(record.parsed_answer_sha256 is None for record in completed.case_records)
    validation = validate_native_capsule(completed.capsule_root)
    assert validation.disposition == ValidationDisposition.VALIDATED, validation.issues


def test_non_output_model_failure_stops_before_later_questions(tmp_path: Path) -> None:
    def reject_nonreportable_case(
        _plan: object,
        _case: object,
        _attempts: object,
        _histories: object,
    ) -> None:
        raise RuntimeError("terminal case is not a reportable question result")

    with pytest.raises(RuntimeError, match="not a reportable question result"):
        run_native_vertical_slice(
            output_root=tmp_path / "capsules",
            run_id="model-format-retry",
            adapter_profile_id="recorded-native-fixture-v1",
            workload=_NativeFixtureJudgeWorkload(),
            visible_evidence_policy=LME_VISIBLE_EVIDENCE_POLICY,
            artifact_store_factory=ArtifactStore,
            memory_factory=_memory_factory,
            model_factory=UnauthorizedModel,
            answer_role_binding_id="recorded-answer-v1",
            judge_model_factory=CorrectableModel,
            judge_role_binding_id="fake-judge-v1",
            terminal_case_publisher=reject_nonreportable_case,
        )

    attempts = [
        json.loads(path.read_bytes())
        for path in (tmp_path / "capsules/model-format-retry/source/attempts").glob("*.json")
    ]
    assert len([attempt for attempt in attempts if attempt["stage"] == "answer"]) == 1


def test_judge_outer_attempt_count_ignores_recovered_transport_retries(tmp_path: Path) -> None:
    completed = _run(tmp_path, judge=TransportThenExhaustedJudge)
    failed = next(record for record in completed.case_records if record.error_stage == "judge")
    attempts = [
        json.loads(path.read_bytes())
        for path in (completed.capsule_root / "source/attempts").glob("*.json")
    ]
    judges = tuple(
        attempt
        for attempt in attempts
        if attempt["parent_id"] == failed.case_occurrence_id and attempt["stage"] == "judge"
    )

    assert len(judges) == 12
    assert _model_outer_attempt_count(judges, max_transport_retries=2) == 6
