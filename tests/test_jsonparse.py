import json
import sys
from pathlib import Path

import pytest

from loglens.jsonparse import extract_json_object, parse_label

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "training"))
from train_qlora import extract_json, load_chat_jsonl  # noqa: E402

GOOD = {
    "severity": "Error",
    "category": "disk",
    "root_cause": "The filesystem is full.",
    "suggested_action": "Free space on /var.",
}


@pytest.mark.parametrize(
    "text",
    [
        json.dumps(GOOD),
        "Here you go:\n```json\n" + json.dumps(GOOD) + "\n```",
        "prefix {not json} then " + json.dumps(GOOD),
    ],
)
def test_parse_label_tolerates_wrapping(text):
    res = parse_label(text)
    assert res.ok, res.error
    assert res.label.severity == "error"  # case-normalised


def test_braces_and_escaped_quotes_inside_strings():
    obj = {**GOOD, "root_cause": 'Value "{x}" was \\ odd'}
    assert json.loads(extract_json_object("x " + json.dumps(obj) + " y")) == obj
    assert extract_json("x " + json.dumps(obj) + " y") == obj


@pytest.mark.parametrize(
    ("text", "error"),
    [
        ("no json here", "no_json_object"),
        ('{"severity": "info",}', "invalid_json"),
        (json.dumps({**GOOD, "severity": "fatal"}), "schema"),
        (json.dumps({**GOOD, "extra": 1}), "schema"),
    ],
)
def test_parse_label_errors(text, error):
    res = parse_label(text)
    assert not res.ok
    assert res.error.startswith(error)


def test_parse_label_never_raises_on_none():
    assert not parse_label(None).ok


def test_training_loader_splits_prompt_and_completion(tmp_path):
    row = {
        "id": "1",
        "messages": [
            {"role": "system", "content": "s"},
            {"role": "user", "content": "u"},
            {"role": "assistant", "content": json.dumps(GOOD)},
        ],
    }
    path = tmp_path / "train_chat.jsonl"
    path.write_text(json.dumps(row) + "\n", encoding="utf-8")
    loaded = load_chat_jsonl(path)
    assert [m["role"] for m in loaded[0]["prompt"]] == ["system", "user"]
    assert loaded[0]["completion"][0]["role"] == "assistant"
