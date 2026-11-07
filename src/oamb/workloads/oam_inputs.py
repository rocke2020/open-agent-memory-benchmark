"""LME and BEAM inputs for the native OAM workflow, independent of local runners."""

from __future__ import annotations

import ast
import hashlib
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from oamb.workloads.longmemeval import load_longmemeval_rows

BEAM_CATEGORIES = (
    "abstention",
    "contradiction_resolution",
    "event_ordering",
    "information_extraction",
    "instruction_following",
    "knowledge_update",
    "multi_session_reasoning",
    "preference_following",
    "summarization",
    "temporal_reasoning",
)
BEAM_DOCUMENT_MAX_CHARS = 100_000


@dataclass(frozen=True)
class Question:
    question_id: str
    text: str
    category: str
    reference: str
    rubric: tuple[str, ...] = ()
    reference_time: str | None = None


@dataclass(frozen=True)
class History:
    history_id: str
    sources: list[list[dict[str, Any]]]
    questions: tuple[Question, ...]
    date_provenance: tuple[dict[str, Any], ...] = ()


def _anchor(value: str | None) -> str | None:
    if not value:
        return None
    normalized = " ".join(value.replace("_", " ").replace("-", " ").replace(",", " ").split())
    for fmt in ("%B %d %Y", "%b %d %Y", "%Y %m %d", "%d %B %Y", "%d %b %Y"):
        try:
            return datetime.strptime(normalized, fmt).replace(tzinfo=UTC).isoformat()
        except ValueError:
            pass
    raise ValueError("unparseable BEAM time anchor")


def _beam_histories(path: Path) -> list[History]:
    rows = json.loads(path.read_text())
    if not isinstance(rows, list):
        raise ValueError("BEAM input must be an array")
    histories = []
    for row in rows:
        owner = str(row["conversation_id"])
        sessions = []
        for plan in row["chat"]:
            if isinstance(plan, list):
                sessions.append((None, plan))
            elif "role" in plan:
                sessions.append((plan.get("time_anchor"), [plan]))
            else:
                for batches in plan.values():
                    for batch in batches or []:
                        turns = []
                        for group in batch["turns"]:
                            turns.extend(group if isinstance(group, list) else [group])
                        sessions.append((batch.get("time_anchor"), turns))
        dates = [
            _anchor(
                anchor or next((t.get("time_anchor") for t in turns if t.get("time_anchor")), None)
            )
            for anchor, turns in sessions
        ]
        sources, provenance = [], []
        for index, (_, turns) in enumerate(sessions):
            date = dates[index]
            donor = index
            if date is None:
                donor = next((i for i in range(index + 1, len(dates)) if dates[i]), -1)
                if donor < 0:
                    donor = next((i for i in range(index - 1, -1, -1) if dates[i]), -1)
                if donor < 0:
                    raise ValueError("BEAM session has no dated neighbor")
                date = dates[donor]
            provenance.append({"session": index, "event_time": date, "donor_session": donor})
            messages = []
            for turn in turns:
                role, content = turn["role"], turn["content"]
                if (
                    role not in {"user", "assistant"}
                    or not isinstance(content, str)
                    or not content.strip()
                ):
                    raise ValueError("invalid BEAM source turn")
                anchor, turn_id = turn.get("time_anchor"), turn.get("id", "")
                labels = ([anchor] if anchor else []) + (
                    [f"Turn {turn_id}"] if turn_id != "" else []
                )
                prefix = "[" + " | ".join(labels) + "] " if labels else ""
                messages.append(
                    {
                        "role": role,
                        "content": f"{prefix}{role.capitalize()}: {content}",
                        "event_time": _anchor(anchor) or date,
                        "learning_eligible": True,
                    }
                )
            if not messages:
                raise ValueError("empty BEAM source session")
            user_info = (row.get("user_profile") or {}).get("user_info", "")
            context = f"Conversation {owner}" + (f" — {user_info}" if user_info else "")
            context += f" (session {index + 1}/{len(sessions)})"
            start = 0
            while start < len(messages):
                end = start + 1
                while (
                    end < len(messages)
                    and len("\n\n".join(m["content"] for m in messages[start : end + 1]))
                    <= BEAM_DOCUMENT_MAX_CHARS
                ):
                    end += 1
                chunk = [
                    {**message, "content": message["content"] + ("\n\n" if i < end - 1 else "")}
                    for i, message in enumerate(messages[start:end], start)
                ]
                sources.append(
                    [
                        {
                            "role": "user",
                            "content": "[AMB document context]\n" + context,
                            "event_time": chunk[0]["event_time"],
                            "learning_eligible": True,
                        },
                        *chunk,
                    ]
                )
                start = end
        probing = row["probing_questions"]
        if isinstance(probing, str):
            try:
                probing = json.loads(probing)
            except ValueError:
                probing = ast.literal_eval(probing)
        if set(probing) - set(BEAM_CATEGORIES):
            raise ValueError("unknown BEAM question category")
        questions = []
        for category in BEAM_CATEGORIES:
            for index, query in enumerate(probing.get(category, [])):
                reference = next(
                    (
                        str(query[k])
                        for k in (
                            "ideal_response",
                            "answer",
                            "expected_answer",
                            "expected",
                            "gold_answer",
                            "reference",
                        )
                        if query.get(k) is not None
                    ),
                    "",
                )
                rubric = query.get("rubric") or (
                    [f"LLM response should contain: {reference}"] if reference else []
                )
                if isinstance(rubric, str):
                    rubric = [rubric]
                if not rubric or not all(isinstance(r, str) and r.strip() for r in rubric):
                    raise ValueError("BEAM question lacks a rubric or reference")
                questions.append(
                    Question(
                        f"{owner}_{category}_{index}",
                        query["question"],
                        category,
                        reference,
                        tuple(rubric),
                    )
                )
        histories.append(History(owner, sources, tuple(questions), tuple(provenance)))
    return histories


def load_histories(dataset: str, path: Path, sha256: str, selection: list[str]) -> list[History]:
    with path.open("rb") as stream:
        if hashlib.file_digest(stream, "sha256").hexdigest() != sha256:
            raise ValueError("dataset checksum mismatch")
    if dataset == "lme500":
        histories = []
        for row in load_longmemeval_rows(path, expected_sha256=sha256):
            sources = [
                [
                    {
                        "role": m.role,
                        "content": m.content,
                        "event_time": s.canonical_timestamp,
                        "learning_eligible": True,
                    }
                    for m in s.messages
                ]
                for s in sorted(
                    row.sessions,
                    key=lambda s: (s.canonical_timestamp, s.original_ordinal_1_indexed),
                )
            ]
            questions = (
                Question(
                    row.question_id,
                    row.question,
                    row.question_type,
                    str(row.answer),
                    reference_time=row.canonical_question_timestamp,
                ),
            )
            histories.append(History(row.question_id, sources, questions))
    elif dataset == "beam10m":
        histories = _beam_histories(path)
    else:
        raise ValueError("unknown OAM dataset")
    ids = [q.question_id for h in histories for q in h.questions]
    if len(set(ids)) != len(ids) or len({h.history_id for h in histories}) != len(histories):
        raise ValueError("duplicate dataset identities")
    if len(set(selection)) != len(selection) or set(selection) - set(ids):
        raise ValueError("invalid question selection")
    if selection:
        selected = set(selection)
        histories = [
            History(
                h.history_id,
                h.sources,
                tuple(q for q in h.questions if q.question_id in selected),
                h.date_provenance,
            )
            for h in histories
            if any(q.question_id in selected for q in h.questions)
        ]
    if not histories:
        raise ValueError("empty question selection")
    return histories
