"""Fail-capable absolute and scaling gates for deterministic offline reports."""

from __future__ import annotations

import json
import multiprocessing
import resource
import sys
import time
from dataclasses import asdict, dataclass
from multiprocessing.connection import Connection
from typing import Protocol

from oamb.constants import (
    REPORT_BUILD_MAX_RSS_BYTES,
    REPORT_BUILD_MAX_SECONDS,
    REPORT_DETAIL_NAV_MAX_MILLISECONDS,
    REPORT_FILTER_SORT_MAX_MILLISECONDS,
    REPORT_INTERACTIVE_MAX_MILLISECONDS,
    REPORT_MAX_HTML_BYTES,
    REPORT_MEASUREMENT_PROCESS_SHUTDOWN_SECONDS,
    REPORT_MEASUREMENT_PROCESS_TIMEOUT_SECONDS,
    REPORT_SCALE_MAX_RATIO,
)
from oamb.reporting.offline_renderer import (
    OfflineReportModel,
    _canonical_display_model_bytes,
    _render_offline_report,
)


class ReportPerformanceError(ValueError):
    """A report exceeds a frozen absolute or linear-growth boundary."""


@dataclass(frozen=True, slots=True)
class ReportPerformanceMeasurement:
    case_count: int
    build_seconds: float
    peak_rss_bytes: int
    model_bytes: int
    html_bytes: int


@dataclass(frozen=True, slots=True)
class ReportBrowserPerformanceMeasurement:
    interactive_milliseconds: float
    filter_sort_milliseconds: float
    detail_navigation_milliseconds: float


class _ReportMeasurementProcess(Protocol):
    @property
    def exitcode(self) -> int | None: ...

    def start(self) -> None: ...

    def is_alive(self) -> bool: ...

    def terminate(self) -> None: ...

    def kill(self) -> None: ...

    def join(self, timeout: float | None = None) -> None: ...

    def close(self) -> None: ...


def measure_report_render(
    model: OfflineReportModel,
    *,
    case_count: int,
) -> ReportPerformanceMeasurement:
    if case_count < 1:
        raise ValueError("report performance fixture requires at least one case")
    context = multiprocessing.get_context("spawn")
    receiver, sender = context.Pipe(duplex=False)
    process = context.Process(
        target=_measure_report_render_child,
        args=(model, case_count, sender),
        name="oamb-report-performance-measurement",
    )
    started = False
    message: object | None = None
    try:
        process.start()
        started = True
        sender.close()
        if not receiver.poll(REPORT_MEASUREMENT_PROCESS_TIMEOUT_SECONDS):
            raise ReportPerformanceError("report measurement process timed out")
        try:
            message = receiver.recv()
        except (EOFError, OSError):
            pass
        process.join(REPORT_MEASUREMENT_PROCESS_SHUTDOWN_SECONDS)
        if process.is_alive():
            raise ReportPerformanceError("report measurement process did not shut down")
        if process.exitcode != 0:
            raise ReportPerformanceError(
                f"report measurement process exited with status {process.exitcode}"
            )
        if not isinstance(message, tuple) or len(message) != 2:
            raise ReportPerformanceError("report measurement process returned an invalid result")
        status, payload = message
        if status == "error" and isinstance(payload, str):
            raise ReportPerformanceError(f"report measurement process failed: {payload}")
        if status != "ok" or not isinstance(payload, ReportPerformanceMeasurement):
            raise ReportPerformanceError("report measurement process returned an invalid result")
        return payload
    finally:
        try:
            sender.close()
        finally:
            try:
                receiver.close()
            finally:
                _cleanup_report_measurement_process(process, started=started)


def _cleanup_report_measurement_process(
    process: _ReportMeasurementProcess,
    *,
    started: bool,
) -> None:
    if not started:
        process.close()
        return
    if process.is_alive():
        process.terminate()
    process.join(REPORT_MEASUREMENT_PROCESS_SHUTDOWN_SECONDS)
    if process.is_alive():
        process.kill()
        process.join(REPORT_MEASUREMENT_PROCESS_SHUTDOWN_SECONDS)
    if process.is_alive():
        raise ReportPerformanceError(
            "report measurement process survived terminate and kill bounds"
        )
    process.close()


def _measure_report_render_child(
    model: OfflineReportModel,
    case_count: int,
    sender: Connection,
) -> None:
    try:
        sender.send(("ok", _measure_report_render_current_process(model, case_count)))
    except Exception as exc:
        sender.send(("error", f"{type(exc).__name__}: {exc}"))
    finally:
        sender.close()


def _measure_report_render_current_process(
    model: OfflineReportModel,
    case_count: int,
) -> ReportPerformanceMeasurement:
    started = time.perf_counter()
    model_bytes = _canonical_display_model_bytes(model)
    html = _render_offline_report(model, canonical_model_bytes=model_bytes)
    elapsed = time.perf_counter() - started
    raw_peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    peak_rss_bytes = int(raw_peak if sys.platform == "darwin" else raw_peak * 1024)
    return ReportPerformanceMeasurement(
        case_count=case_count,
        build_seconds=elapsed,
        peak_rss_bytes=peak_rss_bytes,
        model_bytes=len(model_bytes),
        html_bytes=len(html),
    )


def enforce_report_performance(measurement: ReportPerformanceMeasurement) -> None:
    if measurement.html_bytes > REPORT_MAX_HTML_BYTES:
        raise ReportPerformanceError("report HTML size budget exceeded")
    if measurement.build_seconds > REPORT_BUILD_MAX_SECONDS:
        raise ReportPerformanceError("report build time budget exceeded")
    if measurement.peak_rss_bytes > REPORT_BUILD_MAX_RSS_BYTES:
        raise ReportPerformanceError("report peak RSS budget exceeded")
    if measurement.model_bytes > REPORT_MAX_HTML_BYTES:
        raise ReportPerformanceError("report model size budget exceeded")


def enforce_report_browser_performance(
    measurement: ReportBrowserPerformanceMeasurement,
) -> None:
    if measurement.interactive_milliseconds > REPORT_INTERACTIVE_MAX_MILLISECONDS:
        raise ReportPerformanceError("report browser interactive time budget exceeded")
    if measurement.filter_sort_milliseconds > REPORT_FILTER_SORT_MAX_MILLISECONDS:
        raise ReportPerformanceError("report browser filter/sort time budget exceeded")
    if measurement.detail_navigation_milliseconds > REPORT_DETAIL_NAV_MAX_MILLISECONDS:
        raise ReportPerformanceError("report browser detail navigation time budget exceeded")


def enforce_report_scaling(
    baseline: ReportPerformanceMeasurement,
    larger: ReportPerformanceMeasurement,
) -> None:
    if baseline.case_count < 1 or larger.case_count != baseline.case_count * 10:
        raise ReportPerformanceError("report growth comparison requires an exact 10x case input")
    ratios = {
        "build": _ratio(larger.build_seconds, baseline.build_seconds),
        "RSS": _ratio(larger.peak_rss_bytes, baseline.peak_rss_bytes),
        "model": _ratio(larger.model_bytes, baseline.model_bytes),
        "HTML": _ratio(larger.html_bytes, baseline.html_bytes),
    }
    exceeded = tuple(label for label, ratio in ratios.items() if ratio > REPORT_SCALE_MAX_RATIO)
    if exceeded:
        raise ReportPerformanceError(f"report growth budget exceeded for {', '.join(exceeded)}")


def performance_measurements_bytes(
    measurements: tuple[ReportPerformanceMeasurement, ...],
) -> bytes:
    payload = tuple(
        {
            **asdict(measurement),
            "build_seconds": format(measurement.build_seconds, ".9f"),
        }
        for measurement in measurements
    )
    return (
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode("utf-8")


def _ratio(numerator: int | float, denominator: int | float) -> float:
    if denominator <= 0:
        return float("inf") if numerator > 0 else 1.0
    return float(numerator / denominator)


__all__ = [
    "ReportPerformanceError",
    "ReportBrowserPerformanceMeasurement",
    "ReportPerformanceMeasurement",
    "enforce_report_browser_performance",
    "enforce_report_performance",
    "enforce_report_scaling",
    "measure_report_render",
    "performance_measurements_bytes",
]
