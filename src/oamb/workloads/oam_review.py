"""Offline source/span review of validated native LME and BEAM results."""

from __future__ import annotations

import json
from importlib.metadata import version
from pathlib import Path
from typing import Any

from oamb.workloads.context_noise import (
    aggregate_noise,
    default_assessment,
    digest_json,
    review_noise,
    write_json_new,
)
from oamb.workloads.oam_inputs import load_histories

SCHEMA = "oamb-native-context-review-v1"


def _cases(root: Path, worksheet: Path) -> tuple[str, list[tuple[dict[str, Any], list[Any]]]]:
    # Reuse the scoring validator; importing this module never connects to a service.
    from oamb.oam_workflow import validate_results

    plan = json.loads((root / "plan.json").read_text())
    config = plan["config"]
    histories = load_histories(
        config["dataset"], Path(plan["source"]), plan["source_sha256"], config["question_ids"]
    )
    results = {r["question_id"]: r for r in validate_results(root, plan, histories)}
    tokenizer = {"name": "o200k_base", "version": version("tiktoken"), "method": "encode_ordinary"}
    cases = []
    for history in histories:
        sessions: list[dict[str, Any]] = [
            {
                "session_id": f"{history.history_id}:{index}",
                "session_index": index,
                "messages": [
                    {"message_index": i, **message} for i, message in enumerate(messages, 1)
                ],
            }
            for index, messages in enumerate(history.sources, 1)
        ]
        digest = digest_json(sessions)
        source_path = worksheet.parent / (worksheet.stem + "-sources") / (digest + ".json")
        for question in history.questions:
            result = results[question.question_id]
            cases.append(
                (
                    {
                        "question_id": question.question_id,
                        "question": question.text,
                        "reference": question.reference,
                        "rubric": list(question.rubric),
                        "context": result["context"],
                        "context_sha256": result["context_sha256"],
                        "answer": result["answer"],
                        "score": result["score"],
                        "result_sha256": digest_json(result),
                        "source_view": {
                            "path": str(source_path.resolve()),
                            "sha256": digest,
                            "kind": "complete_prepared_ingestion_sources",
                            "session_count": len(sessions),
                            "message_count": sum(len(s["messages"]) for s in sessions),
                        },
                        "tokenizer": tokenizer,
                    },
                    sessions,
                )
            )
    return digest_json(plan), cases


def export_review(root: Path, worksheet: Path) -> dict[str, Any]:
    if worksheet.exists():
        raise FileExistsError(worksheet)
    identity, cases = _cases(root, worksheet)
    output = []
    for case, sessions in cases:
        path = Path(case["source_view"]["path"])
        payload = {"sessions": sessions}
        if path.exists():
            if json.loads(path.read_text()) != payload:
                raise ValueError("source view differs from frozen dataset")
        else:
            write_json_new(path, payload)
        output.append({**case, "assessment": default_assessment()})
    result = {"schema": SCHEMA, "plan_sha256": identity, "cases": output}
    write_json_new(worksheet, result)
    return result


def summarize_review(root: Path, worksheet: Path) -> dict[str, Any]:
    identity, originals = _cases(root, worksheet)
    review = json.loads(worksheet.read_text())
    cases = review.get("cases")
    if (
        review.get("schema") != SCHEMA
        or review.get("plan_sha256") != identity
        or not isinstance(cases, list)
        or len(cases) != len(originals)
    ):
        raise ValueError("review differs from frozen run identity or coverage")
    expected = {case["question_id"]: (case, sessions) for case, sessions in originals}
    seen = set()
    rows = []
    for case in cases:
        question_id = case.get("question_id")
        if question_id in seen or question_id not in expected:
            raise ValueError("review contains duplicate or unknown questions")
        seen.add(question_id)
        original, sessions = expected[question_id]
        if {k: v for k, v in case.items() if k != "assessment"} != original:
            raise ValueError("review question/context/result/source/tokenizer identity changed")
        path = Path(original["source_view"]["path"])
        if json.loads(path.read_text()) != {"sessions": sessions}:
            raise ValueError("source view differs from frozen dataset")
        rows.append(
            {
                "question_id": question_id,
                **review_noise(
                    original["context"], case.get("assessment"), original["tokenizer"], sessions
                ),
            }
        )
    aggregate = aggregate_noise(rows, expected_cases=len(originals))
    return {
        "schema": SCHEMA,
        "plan_sha256": identity,
        "status": "complete"
        if all(r["noise_status"] in {"reviewed", "empty_context"} for r in rows)
        else "incomplete",
        "aggregation_scope": "fully_reviewed_nonempty_cases_only",
        "noise": aggregate,
        "cases": rows,
    }
