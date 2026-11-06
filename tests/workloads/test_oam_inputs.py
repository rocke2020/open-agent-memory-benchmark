import hashlib
import json

import pytest

from oamb.workloads.oam_inputs import load_histories


def test_lme_full_history_is_chronological_and_excludes_answer_labels(tmp_path):
    row = {
        "question_id": "q",
        "question_type": "single-session-user",
        "question": "Where?",
        "answer": 42,
        "question_date": "2024/01/03 (Wed) 00:00",
        "answer_session_ids": ["a"],
        "haystack_session_ids": ["b", "a"],
        "haystack_dates": ["2024/01/02 (Tue) 00:00", "2024/01/01 (Mon) 00:00"],
        "haystack_sessions": [
            [{"role": "assistant", "content": "later"}],
            [{"role": "user", "content": "earlier", "has_answer": True}],
        ],
    }
    path = tmp_path / "data.json"
    path.write_text(json.dumps([row]))
    histories = load_histories("lme500", path, hashlib.sha256(path.read_bytes()).hexdigest(), ["q"])
    assert len(histories) == 1
    assert [[m["content"] for m in s] for s in histories[0].sources] == [["earlier"], ["later"]]
    assert "has_answer" not in json.dumps(histories[0].sources)
    assert histories[0].questions[0].reference == "42"


def test_beam_nested_turns_dates_roles_and_shared_owner(tmp_path):
    row = {
        "conversation_id": "c",
        "chat": [
            {
                "plan-0": [
                    {
                        "time_anchor": "March-15-2024",
                        "turns": [
                            [
                                {"id": 0, "role": "user", "content": "red"},
                                {"id": 1, "role": "assistant", "content": "OK"},
                            ]
                        ],
                    }
                ]
            }
        ],
        "probing_questions": {
            "information_extraction": [
                {"question": "Color?", "answer": "red", "rubric": ["Mention red"]},
                {"question": "Again?", "answer": "red", "rubric": ["Mention red"]},
            ]
        },
    }
    path = tmp_path / "beam.json"
    path.write_text(json.dumps([row]))
    histories = load_histories("beam10m", path, hashlib.sha256(path.read_bytes()).hexdigest(), [])
    assert len(histories) == 1
    assert len(histories[0].questions) == 2
    assert [m["role"] for m in histories[0].sources[0]] == ["user", "assistant"]
    assert histories[0].sources[0][0]["event_time"] == "2024-03-15T00:00:00+00:00"
    assert "Turn 0" in histories[0].sources[0][0]["content"]
    assert histories[0].questions[1].question_id == "c_information_extraction_1"


def test_input_hash_and_unknown_selection_fail_closed(tmp_path):
    path = tmp_path / "data.json"
    path.write_text("[]")
    with pytest.raises(ValueError, match="checksum"):
        load_histories("beam10m", path, "0" * 64, [])
    with pytest.raises(ValueError, match="selection"):
        load_histories("beam10m", path, hashlib.sha256(path.read_bytes()).hexdigest(), ["missing"])
