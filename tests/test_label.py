import asyncio
import json
from types import SimpleNamespace

import pytest

from loglens.data.label import (
    RequestBudget,
    batch_schema,
    parse_batch_response,
    render_batch,
    run_labeling,
)
from loglens.prompts import load_prompt
from loglens.spend import SpendCapError, SpendPlan, confirm_spend

LABEL = {
    "severity": "warning",
    "category": "security",
    "root_cause": "An unknown user tried to log in over SSH.",
    "suggested_action": "Check the source IP for repeated attempts.",
}


class FakeMessages:
    """Answers each batch, optionally dropping one id to exercise the retry path."""

    def __init__(self, drop_first_id: bool = False):
        self.calls = 0
        self.drop_first_id = drop_first_id

    async def create(self, **kwargs):
        self.calls += 1
        text = kwargs["messages"][0]["content"]
        ids = [line.split()[0][3:] for line in text.splitlines() if line.startswith("id=")]
        if self.drop_first_id and len(ids) > 1:
            ids = ids[1:]
        body = {"labels": [{"id": i, **LABEL} for i in ids]}
        return SimpleNamespace(
            usage=SimpleNamespace(input_tokens=100, output_tokens=50),
            stop_reason="end_turn",
            content=[SimpleNamespace(type="text", text=json.dumps(body))],
        )


def _rows(n):
    return [{"id": f"r{i}", "source": "openssh", "message": f"Invalid user u{i}"} for i in range(n)]


def test_parse_batch_response_validates_each_item():
    text = json.dumps(
        {
            "labels": [
                {"id": "a", **LABEL},
                {"id": "b", **{**LABEL, "severity": "panic"}},
                {"id": "zzz", **LABEL},
            ]
        }
    )
    labels, invalid = parse_batch_response(text, {"a", "b"})
    assert set(labels) == {"a"}
    assert invalid == 2


def test_parse_batch_response_bad_json():
    labels, invalid = parse_batch_response("not json", {"a", "b"})
    assert labels == {} and invalid == 2


def test_batch_schema_requires_id():
    assert batch_schema()["properties"]["labels"]["items"]["required"][0] == "id"


def test_render_batch_includes_ids():
    text = render_batch(load_prompt("teacher", "v1"), _rows(2))
    assert "id=r0" in text and "id=r1" in text


def test_run_labeling_batches_and_retries_missing():
    fake = SimpleNamespace(messages=FakeMessages(drop_first_id=True))
    got = {}
    stats = asyncio.run(
        run_labeling(
            fake,
            "claude-haiku-5-5",
            load_prompt("teacher", "v1"),
            _rows(5),
            batch_size=5,
            concurrency=2,
            budget=RequestBudget(10),
            on_label=lambda r, lab: got.__setitem__(r["id"], lab),
        )
    )
    assert len(got) == 5
    assert stats.requests == 2  # one batch + one retry for the dropped line


def test_request_budget_stops_calls():
    fake = SimpleNamespace(messages=FakeMessages())
    stats = asyncio.run(
        run_labeling(
            fake,
            "m",
            load_prompt("teacher", "v1"),
            _rows(30),
            batch_size=10,
            concurrency=1,
            budget=RequestBudget(2),
            on_label=lambda r, lab: None,
        )
    )
    assert fake.messages.calls == 2
    assert stats.labeled == 20
    assert "request cap reached" in stats.errors


def _plan(n):
    return SpendPlan("m", n, 1000, 500, 0.01)


def test_confirm_spend_dry_run_never_proceeds(capsys):
    assert confirm_spend(_plan(3), max_requests=10, dry_run=True, ask=lambda _: "y") is False


def test_confirm_spend_cap():
    with pytest.raises(SpendCapError):
        confirm_spend(_plan(11), max_requests=10, dry_run=False, ask=lambda _: "y")


def test_confirm_spend_asks():
    assert confirm_spend(_plan(3), max_requests=10, dry_run=False, ask=lambda _: "y") is True
    assert confirm_spend(_plan(3), max_requests=10, dry_run=False, ask=lambda _: "") is False
    assert confirm_spend(_plan(3), max_requests=10, dry_run=False, assume_yes=True) is True
