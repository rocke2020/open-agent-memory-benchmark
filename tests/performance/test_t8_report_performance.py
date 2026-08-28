from __future__ import annotations

import importlib
import multiprocessing
import signal
import time
from collections.abc import Callable
from dataclasses import replace
from multiprocessing.connection import Connection
from multiprocessing.context import SpawnContext
from multiprocessing.process import BaseProcess
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from oamb.contracts.reporting import CompletionSummaryV3, RunReportModelV3
from oamb.contracts.specifications import SourceEvidenceBinding, SourceEvidenceKind
from oamb.reporting.public import build_run_report_model
from tests.reporting.test_t8_offline_renderer import (
    build_claim_boundary,
    build_mab65_report_fixture,
    build_record_projections,
)

SHA_A = "a" * 64
SHA_B = "b" * 64
SHA_C = "c" * 64
SHA_D = "d" * 64
CHILD_BLOCK_SECONDS = 60.0
TEST_PROCESS_CLEANUP_SECONDS = 2.0


class _RecordingProcess:
    def __init__(self, process: BaseProcess) -> None:
        self._process = process
        self.events: list[str] = []
        self.closed = False
        self.alive_before_close: bool | None = None
        self.exitcode_before_close: int | None = None

    @property
    def exitcode(self) -> int | None:
        return self._process.exitcode

    def start(self) -> None:
        self.events.append("start")
        self._process.start()

    def terminate(self) -> None:
        self.events.append("terminate")
        self._process.terminate()

    def kill(self) -> None:
        self.events.append("kill")
        self._process.kill()

    def join(self, timeout: float | None = None) -> None:
        self.events.append("join")
        self._process.join(timeout)

    def is_alive(self) -> bool:
        return self._process.is_alive()

    def close(self) -> None:
        self.alive_before_close = self._process.is_alive()
        self.exitcode_before_close = self._process.exitcode
        self.events.append("close")
        self._process.close()
        self.closed = True

    def emergency_cleanup(self) -> None:
        if self.closed:
            return
        if self._process.is_alive():
            self._process.kill()
        self._process.join(TEST_PROCESS_CLEANUP_SECONDS)
        self._process.close()
        self.closed = True


class _InterruptingReceiver:
    def __init__(self, connection: Connection) -> None:
        self._connection = connection

    def poll(self, _timeout: float) -> bool:
        raise KeyboardInterrupt

    def recv(self) -> object:
        value: object = self._connection.recv()
        return value

    def close(self) -> None:
        self._connection.close()


class _RecordingContext:
    def __init__(
        self,
        delegate: SpawnContext,
        *,
        receiver_wrapper: Callable[[Connection], object] | None = None,
    ) -> None:
        self._delegate = delegate
        self._receiver_wrapper = receiver_wrapper
        self.processes: list[_RecordingProcess] = []

    def Pipe(self, *, duplex: bool) -> tuple[object, Connection]:
        receiver, sender = self._delegate.Pipe(duplex=duplex)
        if self._receiver_wrapper is not None:
            return self._receiver_wrapper(receiver), sender
        return receiver, sender

    def Process(
        self,
        *,
        target: Callable[..., None],
        args: tuple[object, ...],
        name: str,
    ) -> _RecordingProcess:
        process = _RecordingProcess(self._delegate.Process(target=target, args=args, name=name))
        self.processes.append(process)
        return process

    def emergency_cleanup(self) -> None:
        for process in self.processes:
            process.emergency_cleanup()


def _crash_before_sending_measurement(
    _model: RunReportModelV3,
    _case_count: int,
    sender: Connection,
) -> None:
    sender.close()
    raise SystemExit(17)


def _block_until_parent_terminates(
    _model: RunReportModelV3,
    _case_count: int,
    sender: Connection,
) -> None:
    try:
        time.sleep(CHILD_BLOCK_SECONDS)
    finally:
        sender.close()


def _ignore_sigterm_after_sending_invalid_result(
    _model: RunReportModelV3,
    _case_count: int,
    sender: Connection,
) -> None:
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    sender.send(("invalid", None))
    try:
        time.sleep(CHILD_BLOCK_SECONDS)
    finally:
        sender.close()


def _performance() -> ModuleType:
    try:
        return importlib.import_module("oamb.reporting.performance")
    except ModuleNotFoundError:
        pytest.fail("T8 report performance gate is not implemented", pytrace=False)


def _model(
    case_count: int,
    *,
    attempt_count: int | None = None,
    evidence_preview_bytes: int = 0,
) -> RunReportModelV3:
    resolved_attempt_count = case_count if attempt_count is None else attempt_count
    if resolved_attempt_count < 1:
        raise ValueError("performance fixture requires at least one attempt")
    logical_ids = tuple(f"{index:064x}" for index in range(1, case_count + 1))
    plan_ids = tuple(f"{index + 10_000:064x}" for index in range(1, case_count + 1))
    case_ids = tuple(f"{index + 20_000:064x}" for index in range(1, case_count + 1))
    attempt_ids = tuple(f"{index + 30_000:064x}" for index in range(1, resolved_attempt_count + 1))
    summary = CompletionSummaryV3(
        run_id=f"performance-{case_count}",
        intended_logical_contexts=case_count,
        intended_ingestion_plans=case_count,
        ready_ingestion_plans=case_count,
        intended_cases=case_count,
        terminal_cases=case_count,
        completed_cases=case_count,
        errored_cases=0,
        unsupported_cases=0,
        cancelled_cases=0,
        budget_exceeded_cases=0,
        parsed_cases=case_count,
        evaluated_cases=case_count,
        judged_cases=case_count,
        unjudged_cases=0,
        metric_eligible_cases=case_count,
    )
    source = SourceEvidenceBinding(
        binding_id=SHA_A,
        source_kind=SourceEvidenceKind.RUN,
        source_identity=f"performance-{case_count}",
        source_root_hash=SHA_B,
        validation_result_hash=SHA_C,
        source_schema_versions=("capsule_manifest@1",),
    )
    return build_run_report_model(
        report_spec_hash=SHA_A,
        source_binding=source,
        evidence_validation_profile_hash=SHA_C,
        evidence_validation_result_hash=SHA_D,
        reducer_bindings=(("performance-fixture-v1", 1, SHA_A),),
        audience="public",
        origin_kind="native",
        capsule_id=SHA_D,
        protocol_id="performance-fixture-v1",
        workload_id="performance-fixture-v1",
        memory_system_id="performance-fixture-memory",
        claim_boundary=build_claim_boundary(),
        summary=summary,
        metric_summaries=(),
        measurement_lines=(),
        logical_context_ids=logical_ids,
        ingestion_occurrence_ids=plan_ids,
        case_occurrence_ids=case_ids,
        attempt_ids=attempt_ids,
        record_projections=build_record_projections(
            logical_ids=logical_ids,
            plan_ids=plan_ids,
            case_ids=case_ids,
            attempt_ids=attempt_ids,
            evidence_preview_bytes=evidence_preview_bytes,
        ),
        limitations=("synthetic performance fixture",),
    )


def test_50_500_5000_report_builds_pass_absolute_and_linear_growth_budgets(
    tmp_path: Path,
) -> None:
    performance = _performance()
    models = tuple(
        _model(
            case_count,
            attempt_count=case_count * 2,
            evidence_preview_bytes=512,
        )
        for case_count in (50, 500, 5_000)
    )
    assert tuple(len(model.attempt_ids) for model in models) == (100, 1_000, 10_000)
    assert all(
        any(
            preview.text == "e" * 512
            for projection in model.record_projections
            for preview in projection.display_previews
        )
        for model in models
    )
    measurements = tuple(
        performance.measure_report_render(model, case_count=case_count)
        for model, case_count in zip(models, (50, 500, 5_000), strict=True)
    )

    for measurement in measurements:
        performance.enforce_report_performance(measurement)
    performance.enforce_report_scaling(measurements[1], measurements[2])
    (tmp_path / "measurements.json").write_bytes(
        performance.performance_measurements_bytes(measurements)
    )
    assert (tmp_path / "measurements.json").stat().st_size > 0


def test_performance_gate_fails_on_planted_oversize_delay_memory_and_growth() -> None:
    performance = _performance()
    baseline = performance.ReportPerformanceMeasurement(
        case_count=500,
        build_seconds=1.0,
        peak_rss_bytes=64 * 1024 * 1024,
        model_bytes=1_000_000,
        html_bytes=2_000_000,
    )
    with pytest.raises(performance.ReportPerformanceError, match="HTML"):
        performance.enforce_report_performance(replace(baseline, html_bytes=33 * 1024 * 1024))
    with pytest.raises(performance.ReportPerformanceError, match="build"):
        performance.enforce_report_performance(replace(baseline, build_seconds=16.0))
    with pytest.raises(performance.ReportPerformanceError, match="RSS"):
        performance.enforce_report_performance(replace(baseline, peak_rss_bytes=513 * 1024 * 1024))
    with pytest.raises(performance.ReportPerformanceError, match="growth"):
        performance.enforce_report_scaling(
            baseline,
            baseline.__class__(
                case_count=5_000,
                build_seconds=13.0,
                peak_rss_bytes=64 * 1024 * 1024,
                model_bytes=13_000_000,
                html_bytes=26_000_000,
            ),
        )


def test_browser_performance_gate_fails_on_each_planted_latency_overrun() -> None:
    performance = _performance()
    measurement_type = performance.ReportBrowserPerformanceMeasurement
    enforce = performance.enforce_report_browser_performance
    baseline = measurement_type(
        interactive_milliseconds=1_000.0,
        filter_sort_milliseconds=100.0,
        detail_navigation_milliseconds=50.0,
    )

    enforce(baseline)
    with pytest.raises(performance.ReportPerformanceError, match="interactive"):
        enforce(replace(baseline, interactive_milliseconds=5_001.0))
    with pytest.raises(performance.ReportPerformanceError, match="filter/sort"):
        enforce(replace(baseline, filter_sort_milliseconds=251.0))
    with pytest.raises(performance.ReportPerformanceError, match="detail navigation"):
        enforce(replace(baseline, detail_navigation_milliseconds=101.0))


def test_exact_mab65_complete_and_incomplete_reports_keep_29_25_65_render_work() -> None:
    performance = _performance()
    complete = build_mab65_report_fixture()
    unavailable = build_mab65_report_fixture(available=False)

    measurements = tuple(
        performance.measure_report_render(model, case_count=65) for model in (complete, unavailable)
    )

    for measurement in measurements:
        performance.enforce_report_performance(measurement)
        assert measurement.case_count == 65
        assert measurement.model_bytes > 0
        assert measurement.html_bytes > 0
    assert complete.summary.intended_logical_contexts == 29
    assert complete.summary.intended_ingestion_plans == 25
    assert complete.summary.intended_cases == 65
    assert unavailable.logical_context_ids == complete.logical_context_ids
    assert unavailable.ingestion_occurrence_ids == complete.ingestion_occurrence_ids
    assert unavailable.case_occurrence_ids == complete.case_occurrence_ids
    assert complete.mab65_reduction is not None
    assert len(complete.mab65_reduction.plans) == 25
    assert (
        len({item.evidence_binding.ingestion_plan_id for item in complete.mab65_reduction.plans})
        == 25
    )
    assert unavailable.mab65_reduction is not None
    assert unavailable.mab65_reduction.index_value is None


def test_report_peak_rss_measurement_is_isolated_from_parent_process_history(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    performance = _performance()
    monkeypatch.setattr(
        performance.resource,
        "getrusage",
        lambda _target: SimpleNamespace(ru_maxrss=3 * 1024 * 1024 * 1024),
    )

    measurement = performance.measure_report_render(_model(50), case_count=50)

    assert measurement.peak_rss_bytes < 512 * 1024 * 1024


def test_measure_report_render_reaps_and_closes_spawned_child_after_receiver_eof(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    performance = _performance()
    context = _RecordingContext(multiprocessing.get_context("spawn"))
    monkeypatch.setattr(performance.multiprocessing, "get_context", lambda _method: context)
    monkeypatch.setattr(
        performance,
        "_measure_report_render_child",
        _crash_before_sending_measurement,
    )

    try:
        with pytest.raises(
            performance.ReportPerformanceError,
            match="exited with status 17",
        ):
            performance.measure_report_render(_model(1), case_count=1)

        assert len(context.processes) == 1
        process = context.processes[0]
        assert process.events[-2:] == ["join", "close"]
        assert process.alive_before_close is False
        assert process.exitcode_before_close == 17
    finally:
        context.emergency_cleanup()


def test_measure_report_render_reaps_and_closes_child_before_reraising_keyboard_interrupt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    performance = _performance()
    context = _RecordingContext(
        multiprocessing.get_context("spawn"),
        receiver_wrapper=_InterruptingReceiver,
    )
    monkeypatch.setattr(performance.multiprocessing, "get_context", lambda _method: context)
    monkeypatch.setattr(
        performance,
        "_measure_report_render_child",
        _block_until_parent_terminates,
    )

    try:
        with pytest.raises(KeyboardInterrupt):
            performance.measure_report_render(_model(1), case_count=1)

        assert len(context.processes) == 1
        process = context.processes[0]
        assert process.events == ["start", "terminate", "join", "close"]
        assert process.alive_before_close is False
        assert process.exitcode_before_close is not None
    finally:
        context.emergency_cleanup()


def test_measure_report_render_kills_child_that_ignores_terminate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    performance = _performance()
    context = _RecordingContext(multiprocessing.get_context("spawn"))
    monkeypatch.setattr(performance.multiprocessing, "get_context", lambda _method: context)
    monkeypatch.setattr(
        performance,
        "_measure_report_render_child",
        _ignore_sigterm_after_sending_invalid_result,
    )
    monkeypatch.setattr(performance, "REPORT_MEASUREMENT_PROCESS_SHUTDOWN_SECONDS", 0.05)

    try:
        with pytest.raises(performance.ReportPerformanceError):
            performance.measure_report_render(_model(1), case_count=1)

        process = context.processes[0]
        assert process.events == [
            "start",
            "join",
            "terminate",
            "join",
            "kill",
            "join",
            "close",
        ]
        assert process.alive_before_close is False
    finally:
        context.emergency_cleanup()
