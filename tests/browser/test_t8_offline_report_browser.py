from __future__ import annotations

import time
from pathlib import Path

import pytest
from playwright.sync_api import BrowserType, Page, sync_playwright

import oamb.reporting.performance as report_performance
from oamb.contracts.accounting import ProofStatus
from oamb.contracts.reporting import (
    CompletionSummaryV3,
    ExactRational,
    MeasurementSummaryLine,
)
from oamb.contracts.specifications import SourceEvidenceBinding, SourceEvidenceKind
from oamb.reporting.offline_renderer import render_offline_report
from oamb.reporting.performance import (
    ReportBrowserPerformanceMeasurement,
    enforce_report_browser_performance,
)
from oamb.reporting.public import build_metric_summary, build_run_report_model
from tests.reporting.test_t8_offline_renderer import (
    _incomparable_report,
    build_claim_boundary,
    build_comparable_report_fixture,
    build_mab65_report_fixture,
    build_record_projections,
)

SHA_A = "a" * 64
SHA_B = "b" * 64
SHA_C = "c" * 64
SHA_D = "d" * 64
MALICIOUS_LIMITATION = "</script><script>globalThis.oambInjected = true</script>"
CONTROL_ARIA_BASELINE = """- text: Filter visible records
- searchbox "Filter visible records"
- text: Record axis
- combobox "Record axis":
  - option "All axes" [selected]
  - option "Report"
  - option "Logical context"
  - option "Ingestion plan"
  - option "Case"
  - option "Attempt"
- text: Terminal status
- listbox "Terminal status":
  - option "All statuses" [selected]
  - option "completed"
  - option "included"
  - option "pass"
  - option "sealed"
  - option "succeeded"
- text: Failure stage
- listbox "Failure stage":
  - option "All failure stages" [selected]
- text: Evaluation status
- listbox "Evaluation status":
  - option "All evaluation states" [selected]
  - option "judged"
- text: Verdict
- listbox "Verdict":
  - option "All verdicts" [selected]
  - option "fail"
  - option "pass"
- text: Capability or type
- listbox "Capability or type":
  - option "All capabilities and types" [selected]
  - option "fixture"
- text: Metric
- listbox "Metric":
  - option "All metrics" [selected]
  - option "fixture-exact-v1"
- text: Proof status
- listbox "Proof status":
  - option "All proof states" [selected]
  - option "measured_complete"
  - option "unavailable"
- text: Raw evidence
- listbox "Raw evidence":
  - option "Any raw evidence" [selected]
  - option "Raw present"
  - option "Raw absent"
- text: Stable order
- combobox "Stable order":
  - option "Manifest/source order" [selected]
  - option "Label"
  - option "Terminal status"
  - option "Metric"
  - option "Latency"
  - option "Ctx tokens"
  - option "Declared usage"
- button "Previous visible record"
- button "Next visible record"
- button "Copy visible record\""""
FOCUS_ORDER = (
    "oamb-filter",
    "oamb-axis-filter",
    "oamb-status-filter",
    "oamb-failure-filter",
    "oamb-evaluation-filter",
    "oamb-verdict-filter",
    "oamb-capability-filter",
    "oamb-metric-filter",
    "oamb-proof-filter",
    "oamb-raw-filter",
    "oamb-sort",
    "oamb-previous",
    "oamb-next",
    "oamb-copy",
)


def _report_path(
    tmp_path: Path,
    *,
    origin_kind: str = "native",
    case_count: int = 2,
) -> Path:
    logical_ids: tuple[str, ...]
    plan_ids: tuple[str, ...]
    case_ids: tuple[str, ...]
    if case_count == 2:
        logical_ids = (SHA_A, SHA_B)
        plan_ids = (SHA_B, SHA_C)
        case_ids = (SHA_C, SHA_D)
    else:
        logical_ids = tuple(f"{index:064x}" for index in range(1, case_count + 1))
        plan_ids = tuple(f"{index + 10_000:064x}" for index in range(1, case_count + 1))
        case_ids = tuple(f"{index + 20_000:064x}" for index in range(1, case_count + 1))
    attempt_ids = (SHA_D, SHA_A)
    summary = CompletionSummaryV3(
        run_id=f"browser-{origin_kind}",
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
        metric_eligible_cases=2,
    )
    source = SourceEvidenceBinding(
        binding_id=SHA_A,
        source_kind=(
            SourceEvidenceKind.RUN if origin_kind == "native" else SourceEvidenceKind.EXTERNAL
        ),
        source_identity=f"browser-{origin_kind}",
        source_root_hash=SHA_B,
        validation_result_hash=SHA_C,
        source_schema_versions=("capsule_manifest@1",),
    )
    model = build_run_report_model(
        report_spec_hash=SHA_A,
        source_binding=source,
        evidence_validation_profile_hash=SHA_C,
        evidence_validation_result_hash=SHA_D,
        reducer_bindings=(("browser-fixture-v1", 1, SHA_A),),
        audience="public",
        origin_kind=origin_kind,  # type: ignore[arg-type]
        capsule_id=SHA_A if origin_kind == "native" else None,
        protocol_id="browser-fixture-v1",
        workload_id="browser-fixture-v1",
        memory_system_id="browser-fixture-memory",
        claim_boundary=build_claim_boundary(),
        summary=summary,
        metric_summaries=(
            build_metric_summary(
                metric_id="fixture-exact-v1",
                metric_version=1,
                stratum_id="all",
                score_numerator=1,
                score_denominator=2,
                input_count=case_count,
                case_occurrence_ids=case_ids,
                claim_note="exact sealed fixture fraction",
            ),
        ),
        measurement_lines=(
            MeasurementSummaryLine(
                dimension_id="input_tokens",
                stage="answer",
                owner_kind="case",
                indexing_view="not_applicable",
                value=ExactRational(numerator=123, denominator=1),
                unit="token",
                proof_status=ProofStatus.MEASURED_COMPLETE,
                basis="token_count",
                currency=None,
                source_record_ids=(SHA_A,),
                reason=None,
            ),
            MeasurementSummaryLine(
                dimension_id="peak_rss_bytes",
                stage="answer",
                owner_kind="run",
                indexing_view="not_applicable",
                value=ExactRational(numerator=4096, denominator=1),
                unit="byte",
                proof_status=ProofStatus.MEASURED_COMPLETE,
                basis="raw_resource",
                currency=None,
                source_record_ids=(SHA_B,),
                reason=None,
            ),
            MeasurementSummaryLine(
                dimension_id="supplier_cost",
                stage="answer",
                owner_kind="case",
                indexing_view="not_applicable",
                value=None,
                unit="currency",
                proof_status=ProofStatus.UNAVAILABLE,
                basis="actual_supplier_charge",
                currency=None,
                source_record_ids=(SHA_D,),
                reason="supplier charge unavailable",
            ),
        ),
        logical_context_ids=logical_ids,
        ingestion_occurrence_ids=plan_ids,
        case_occurrence_ids=case_ids,
        attempt_ids=attempt_ids,
        record_projections=build_record_projections(
            logical_ids=logical_ids,
            plan_ids=plan_ids,
            case_ids=case_ids,
            attempt_ids=attempt_ids,
        ),
        limitations=(MALICIOUS_LIMITATION, "fixture limitation"),
    )
    path = tmp_path / f"report-{origin_kind}.html"
    path.write_bytes(render_offline_report(model))
    return path


def _open_file_report(page: Page, path: Path) -> tuple[list[str], list[str], list[str]]:
    external_requests: list[str] = []
    console_errors: list[str] = []
    page_errors: list[str] = []
    page.on(
        "request",
        lambda request: (
            external_requests.append(request.url)
            if not request.url.startswith(("file:", "data:"))
            else None
        ),
    )
    page.on(
        "console",
        lambda message: console_errors.append(message.text) if message.type == "error" else None,
    )
    page.on("pageerror", lambda error: page_errors.append(str(error)))
    page.goto(path.resolve().as_uri(), wait_until="load")
    return external_requests, console_errors, page_errors


@pytest.mark.parametrize("engine_name", ("chromium", "firefox", "webkit"))
def test_file_report_is_network_free_accessible_and_interactive_across_engines(
    tmp_path: Path,
    engine_name: str,
) -> None:
    path = _report_path(tmp_path)
    with sync_playwright() as playwright:
        engine: BrowserType = getattr(playwright, engine_name)
        browser = engine.launch()
        page = browser.new_page(viewport={"width": 1280, "height": 900})
        external_requests, console_errors, page_errors = _open_file_report(page, path)

        assert external_requests == []
        assert console_errors == []
        assert page_errors == []
        assert page.locator("main").count() == 1
        assert page.locator("section[aria-labelledby]").count() == 6
        assert page.locator('[data-axis="logical-context"]').count() == 2
        assert page.locator('[data-axis="plan"]').count() == 2
        assert page.locator('[data-axis="case"]').count() == 2
        assert page.locator('[data-axis="attempt"]').count() == 2
        assert page.locator("#oamb-quality-summaries li").all_text_contents() == [
            "fixture-exact-v1@1 [all]: 1/2; inputs: 2; exact sealed fixture fraction"
        ]
        measurement_text = page.locator("#oamb-measurement-summaries").text_content()
        assert measurement_text is not None
        for expected in (
            "input_tokens / answer / case / not_applicable: 123/1 token",
            "peak_rss_bytes / answer / run / not_applicable: 4096/1 byte",
            "supplier_cost / answer / case / not_applicable: unavailable",
            "proof: measured_complete",
            "proof: unavailable",
        ):
            assert expected in measurement_text
        assert page.evaluate("globalThis.oambInjected") is None
        assert page.locator(".controls").aria_snapshot() == CONTROL_ARIA_BASELINE
        assert (
            tuple(
                page.locator(".controls input, .controls select, .controls button").evaluate_all(
                    "nodes => nodes.map(node => node.id)"
                )
            )
            == FOCUS_ORDER
        )
        if engine_name != "webkit":
            page.locator("#oamb-filter").focus()
            observed_focus = []
            for _ in FOCUS_ORDER:
                observed_focus.append(page.locator(":focus").get_attribute("id"))
                page.keyboard.press("Tab")
            assert tuple(observed_focus) == FOCUS_ORDER
        else:
            for control_id in FOCUS_ORDER:
                page.locator(f"#{control_id}").focus()
                assert page.evaluate("document.activeElement.id") == control_id

        axis_filter = page.get_by_label("Record axis")
        axis_filter.select_option("case")
        visible = page.locator(".record:not([hidden])")
        assert visible.count() == 2
        assert all(
            item == "case"
            for item in visible.evaluate_all("nodes => nodes.map(n => n.dataset.axis)")
        )

        first_case = visible.first
        first_case.focus()
        page.keyboard.press("j")
        assert page.evaluate("location.hash").startswith(f"#case={SHA_D}")
        assert "axis=case" in page.evaluate("location.hash")
        assert page.locator(":focus").get_attribute("data-axis") == "case"
        page.keyboard.press("ArrowUp")
        assert page.evaluate("location.hash").startswith(f"#case={SHA_C}")
        page.keyboard.press("End")
        assert page.locator(":focus").get_attribute("data-axis") == "case"
        page.keyboard.press("Home")
        assert page.locator(":focus").get_attribute("data-axis") == "case"

        page.reload(wait_until="load")
        assert page.get_by_label("Record axis").input_value() == "case"
        assert page.locator(".record:not([hidden])").count() == 2

        page.evaluate("location.hash = '#axis=attempt'")
        page.wait_for_function("document.getElementById('oamb-axis-filter').value === 'attempt'")
        assert page.get_by_label("Record axis").input_value() == "attempt"
        assert page.locator(".record:not([hidden])").count() == 2
        page.evaluate("location.hash = '#axis=unknown&sort=unknown'")
        page.wait_for_function("document.getElementById('oamb-axis-filter').value === 'all'")
        assert page.get_by_label("Record axis").input_value() == "all"
        assert page.get_by_label("Stable order").input_value() == "source"
        page.evaluate("location.hash = '#%E0%A4%A'")
        page.wait_for_timeout(10)
        assert page.locator(".record:not([hidden])").count() > 0
        assert page_errors == []

        for axis, record_id in (
            ("logical-context", SHA_A),
            ("plan", SHA_B),
            ("case", SHA_C),
            ("attempt", SHA_D),
        ):
            page.evaluate(
                "([axis, id]) => { location.hash = `#${axis}=${id}&axis=${axis}`; }",
                [axis, record_id],
            )
            page.wait_for_function(
                "([axis, id]) => document.activeElement?.id === `${axis}=${id}`",
                arg=[axis, record_id],
            )
            assert page.get_by_label("Record axis").input_value() == axis
            assert page.locator(":focus").get_attribute("id") == f"{axis}={record_id}"

        for label in (
            "Terminal status",
            "Failure stage",
            "Evaluation status",
            "Verdict",
            "Capability or type",
            "Metric",
            "Proof status",
            "Raw evidence",
        ):
            assert page.get_by_label(label, exact=True).count() == 1

        page.get_by_label("Record axis").select_option("case")
        for label, value in (
            ("Terminal status", "completed"),
            ("Evaluation status", "judged"),
            ("Verdict", "pass"),
            ("Capability or type", "fixture"),
            ("Metric", "fixture-exact-v1"),
            ("Proof status", "measured_complete"),
            ("Raw evidence", "present"),
        ):
            select = page.get_by_label(label, exact=True)
            select.select_option(value)
            assert page.locator(".record:not([hidden])").count() > 0
            select.select_option("all")

        page.get_by_label("Filter visible records").fill(SHA_D)
        assert page.locator(".record:not([hidden])").count() == 1
        page.get_by_label("Stable order").select_option("label")

        browser.close()


def test_file_report_without_javascript_keeps_typed_baseline(tmp_path: Path) -> None:
    path = _report_path(tmp_path)
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch()
        context = browser.new_context(java_script_enabled=False)
        page = context.new_page()
        external_requests, console_errors, page_errors = _open_file_report(page, path)

        assert external_requests == []
        assert console_errors == []
        assert page_errors == []
        assert page.locator("#oamb-report-kind").text_content() == "run_report_model"
        assert page.locator("#oamb-origin-kind").text_content() == "native"
        assert page.locator("#oamb-report-id").text_content()
        identity = page.locator("#oamb-identity-summary").text_content()
        assert identity is not None
        assert "browser-fixture-v1" in identity
        assert "browser-fixture-memory" in identity
        assert "browser-native" in identity
        assert SHA_A in identity
        assert SHA_B in identity
        validation = page.locator("#oamb-validation-summary").text_content()
        assert validation is not None
        assert SHA_D in validation
        completion = page.locator("#oamb-completion-summary").text_content()
        assert completion is not None
        assert "intended cases: 2" in completion
        assert "completed: 2" in completion
        assert "errored: 0" in completion
        assert "unsupported: 0" in completion
        assert "cancelled: 0" in completion
        assert "budget exceeded: 0" in completion
        assert "evaluated: 2" in completion
        assert "judged: 2" in completion
        assert "unjudged: 0" in completion
        assert page.locator("#oamb-static-limitations li").all_text_contents() == [
            MALICIOUS_LIMITATION,
            "fixture limitation",
        ]
        nojs = page.locator("#oamb-nojs-explanation")
        assert nojs.text_content() == (
            "JavaScript is disabled; this typed evidence summary remains complete."
        )
        assert nojs.is_visible()

        browser.close()


def test_file_report_copy_theme_print_narrow_and_external_origin(tmp_path: Path) -> None:
    native_path = _report_path(tmp_path)
    external_path = _report_path(tmp_path, origin_kind="external")
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch()

        success_page = browser.new_page(viewport={"width": 320, "height": 800})
        success_page.add_init_script(
            "Object.defineProperty(navigator, 'clipboard', {value: {writeText: async () => {}}})"
        )
        external_requests, console_errors, page_errors = _open_file_report(
            success_page, native_path
        )
        success_page.get_by_role("button", name="Copy visible record").click()
        assert success_page.get_by_role("status").text_content() == "Copied visible record."
        assert success_page.evaluate("document.body.scrollWidth <= innerWidth") is True
        assert success_page.evaluate(
            """() => {
              const style = getComputedStyle(document.documentElement);
              return {
                background: style.getPropertyValue('--background').trim(),
                surface: style.getPropertyValue('--surface').trim(),
                text: style.getPropertyValue('--text').trim(),
                border: style.getPropertyValue('--border').trim(),
                code: style.getPropertyValue('--code-background').trim(),
                controls: getComputedStyle(document.querySelector('.controls')).display,
              };
            }"""
        ) == {
            "background": "#f5f7fa",
            "surface": "#ffffff",
            "text": "#17202a",
            "border": "#c8d1dc",
            "code": "#eef2f7",
            "controls": "grid",
        }
        assert external_requests == []
        assert console_errors == []
        assert page_errors == []
        light_capture = success_page.screenshot(
            path=tmp_path / "native-light-320.png", full_page=True
        )
        assert light_capture == success_page.screenshot(full_page=True)

        fallback_page = browser.new_page(viewport={"width": 1280, "height": 900})
        fallback_page.add_init_script(
            "Object.defineProperty(navigator, 'clipboard', {value: {writeText: async () => {throw new Error('denied')}}})"
        )
        _open_file_report(fallback_page, native_path)
        fallback_page.get_by_role("button", name="Copy visible record").click()
        assert fallback_page.get_by_role("status").text_content() == "Select and copy manually."

        dark_page = browser.new_page(
            viewport={"width": 1280, "height": 900},
            color_scheme="dark",
        )
        _open_file_report(dark_page, external_path)
        assert dark_page.locator("#oamb-origin-kind").text_content() == "external"
        dark_background = dark_page.locator("body").evaluate(
            "node => getComputedStyle(node).backgroundColor"
        )
        assert dark_background == "rgb(16, 21, 27)"
        assert dark_page.evaluate(
            """() => {
              const style = getComputedStyle(document.documentElement);
              return [
                style.getPropertyValue('--surface').trim(),
                style.getPropertyValue('--text').trim(),
                style.getPropertyValue('--code-background').trim(),
              ];
            }"""
        ) == ["#18212a", "#edf2f7", "#0c1117"]
        dark_capture = dark_page.screenshot(
            path=tmp_path / "external-dark-desktop.png", full_page=True
        )
        assert dark_capture == dark_page.screenshot(full_page=True)

        dark_page.emulate_media(media="print")
        assert (
            dark_page.locator(".controls").evaluate("node => getComputedStyle(node).display")
            == "none"
        )
        assert (
            dark_page.evaluate(
                "getComputedStyle(document.documentElement).getPropertyValue('--text').trim()"
            )
            == "#000000"
        )
        dark_page.screenshot(path=tmp_path / "external-print.png", full_page=True)

        browser.close()


def test_exact_mab65_report_exposes_topology_and_precomputed_waterfall(tmp_path: Path) -> None:
    path = tmp_path / "mab65-report.html"
    path.write_bytes(render_offline_report(build_mab65_report_fixture()))
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch()
        page = browser.new_page(viewport={"width": 1280, "height": 900})
        external_requests, console_errors, page_errors = _open_file_report(page, path)

        assert external_requests == []
        assert console_errors == []
        assert page_errors == []
        assert page.locator('[data-axis="logical-context"]').count() == 29
        assert page.locator('[data-axis="plan"]').count() == 25
        assert page.locator('[data-axis="case"]').count() == 65
        assert page.locator("#oamb-mab65-components li").all_text_contents() == [
            "EventQA SubEM: 1/2",
            "ICL Exact Match: 1/2",
            "ReDial Recall@5: 1/2",
            "DetectiveQA Exact Match: 1/2",
            "FactConsolidation SubEM: 1/2",
        ]
        assert page.locator("#oamb-mab65-capabilities li").count() == 4
        assert page.locator("#oamb-mab65-index").text_content() == (
            "MAB-65 Capability-Balanced Index: 50/1 (secondary; complete evidence only)."
        )

        page.get_by_label("Record axis").select_option("plan")
        assert page.locator(".record:not([hidden])").count() == 25
        redial_plan = page.locator(
            '.record:not([hidden]) pre:has-text("mab-redial-recall-at-5-v1")'
        )
        assert redial_plan.count() == 1
        redial_text = redial_plan.text_content()
        assert redial_text is not None
        assert "redial-resolution.json.gz" in redial_text

        page.get_by_label("Metric", exact=True).select_option("mab-redial-recall-at-5-v1")
        assert page.locator(".record:not([hidden])").count() == 1
        page.get_by_label("Metric", exact=True).select_option("all")
        page.get_by_label("Capability or type", exact=True).select_option("recsys")
        recsys_count = page.locator(".record:not([hidden])").count()
        assert recsys_count == 1
        page.get_by_label("Capability or type", exact=True).select_option("cr_sf")
        crsf_count = page.locator(".record:not([hidden])").count()
        assert crsf_count > 0
        page.get_by_label("Capability or type", exact=True).select_option(["recsys", "cr_sf"])
        assert page.locator(".record:not([hidden])").count() == recsys_count + crsf_count
        assert "capability=recsys" in page.evaluate("location.hash")
        assert "capability=cr_sf" in page.evaluate("location.hash")
        page.get_by_label("Capability or type", exact=True).select_option("all")

        page.get_by_label("Record axis").select_option("logical-context")
        grouped_members = page.locator(
            '.record:not([hidden]) pre:has-text("factconsolidation_mh_")'
        )
        assert grouped_members.count() == 4
        page.screenshot(path=tmp_path / "mab65-desktop.png", full_page=True)

        browser.close()


def test_compatible_and_incompatible_comparison_snapshots_preserve_claim_boundary(
    tmp_path: Path,
) -> None:
    paths = []
    for label, model in (
        ("compatible", build_comparable_report_fixture()),
        ("incompatible", _incomparable_report()),
    ):
        path = tmp_path / f"comparison-{label}.html"
        path.write_bytes(render_offline_report(model))
        paths.append((label, path))

    with sync_playwright() as playwright:
        browser = playwright.chromium.launch()
        for label, path in paths:
            page = browser.new_page(viewport={"width": 1280, "height": 900})
            external_requests, console_errors, page_errors = _open_file_report(page, path)
            assert external_requests == []
            assert console_errors == []
            assert page_errors == []
            body_text = page.locator("body").text_content()
            assert body_text is not None
            if label == "compatible":
                assert "winner" in body_text
                assert "signed_delta" in body_text
            else:
                assert "winner" not in body_text
                assert "signed_delta" not in body_text
                assert "dataset revisions differ" in body_text
            page.screenshot(path=tmp_path / f"comparison-{label}.png", full_page=True)
            page.close()
        browser.close()


def test_5000_case_browser_interaction_stays_within_frozen_latency_budgets(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = _report_path(tmp_path, case_count=5_000)
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch()
        page = browser.new_page(viewport={"width": 1280, "height": 900})

        interactive_started = time.perf_counter()
        external_requests, console_errors, page_errors = _open_file_report(page, path)
        interactive_milliseconds = (time.perf_counter() - interactive_started) * 1_000
        assert page.locator('[data-axis="case"]').count() == 5_000

        filter_sort_milliseconds = page.evaluate(
            """() => {
              const filter = document.getElementById('oamb-filter');
              const axis = document.getElementById('oamb-axis-filter');
              const sort = document.getElementById('oamb-sort');
              const started = performance.now();
              axis.value = 'case';
              axis.dispatchEvent(new Event('change', {bubbles: true}));
              filter.value = 'Case occurrence';
              filter.dispatchEvent(new Event('input', {bubbles: true}));
              sort.value = 'label';
              sort.dispatchEvent(new Event('change', {bubbles: true}));
              return performance.now() - started;
            }"""
        )
        detail_navigation_milliseconds = page.evaluate(
            """() => {
              const first = document.querySelector('.record:not([hidden])');
              first.focus();
              const started = performance.now();
              document.dispatchEvent(new KeyboardEvent('keydown', {key: 'j', bubbles: true}));
              return performance.now() - started;
            }"""
        )
        measurement = ReportBrowserPerformanceMeasurement(
            interactive_milliseconds=interactive_milliseconds,
            filter_sort_milliseconds=filter_sort_milliseconds,
            detail_navigation_milliseconds=detail_navigation_milliseconds,
        )
        enforce_report_browser_performance(measurement)
        monkeypatch.setattr(
            report_performance,
            "REPORT_FILTER_SORT_MAX_MILLISECONDS",
            measurement.filter_sort_milliseconds - 0.001,
        )
        with pytest.raises(ValueError, match="filter/sort"):
            enforce_report_browser_performance(measurement)
        assert external_requests == []
        assert console_errors == []
        assert page_errors == []
        browser.close()
