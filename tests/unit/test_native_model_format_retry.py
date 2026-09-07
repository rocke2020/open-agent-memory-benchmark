from __future__ import annotations

import json
from pathlib import Path

from oamb.artifacts.store import ArtifactStore
from oamb.contracts.ids import canonical_json_bytes, canonical_sha256
from oamb.contracts.ports import ModelReceipt, ModelRequest, RawPayloadSealRequest
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
    async def complete(self, request: ModelRequest) -> ModelReceipt:
        payload = {
            "operation": "model_complete",
            "attempt_id": request.attempt_id,
            "stage": request.stage,
            "prompt": request.messages[0][1],
            "output_text": "perhaps",
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
            output_text="perhaps",
            usage_reference_ids=(),
            model="fixture-model",
            raw_response_bytes=canonical_json_bytes(payload),
        )


def _run(tmp_path: Path, *, judge: type[CorrectableModel] = CorrectableModel) -> NativeRunArtifacts:
    return run_native_vertical_slice(
        output_root=tmp_path / "capsules",
        run_id="model-format-retry",
        adapter_profile_id="recorded-native-fixture-v1",
        workload=_NativeFixtureJudgeWorkload(),
        visible_evidence_policy=LME_VISIBLE_EVIDENCE_POLICY,
        artifact_store_factory=ArtifactStore,
        memory_factory=_memory_factory,
        model_factory=CorrectableModel,
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
