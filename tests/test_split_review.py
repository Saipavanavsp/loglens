import pytest

from loglens.data.review import load_reviews, save_review, select_golden
from loglens.data.split import build_splits, remove_near_duplicates

LABEL = {
    "severity": "info",
    "category": "auth",
    "root_cause": "A user logged in.",
    "suggested_action": "No action needed.",
}


def _row(i, source="openssh", template=None, key=None):
    return {
        "id": f"id{i}",
        "source": source,
        "message": key or f"message {i}",
        "key": key or f"message number {i} unique text {i * 7919}",
        "template": template or f"{source}:t{i}",
        "label": dict(LABEL),
    }


def test_select_golden_one_per_template_and_balanced():
    rows = [_row(i, "a", template="a:same") for i in range(5)]
    rows += [_row(10 + i, "b") for i in range(5)]
    golden = select_golden(rows, 4)
    sources = [g["source"] for g in golden]
    assert sources.count("a") == 1  # only one template available in a
    assert len(golden) == 4


def test_reviews_latest_wins(tmp_path):
    path = tmp_path / "rev.jsonl"
    row = _row(1)
    save_review(row, "accepted", LABEL, path=path)
    save_review(row, "corrected", {**LABEL, "severity": "warning"}, path=path)
    assert load_reviews(path)["id1"]["label"]["severity"] == "warning"


def test_save_review_validates_label(tmp_path):
    with pytest.raises(ValueError):
        save_review(_row(1), "accepted", {**LABEL, "severity": "bad"}, path=tmp_path / "r.jsonl")


def test_near_duplicates_removed():
    golden = [_row(1, key="invalid user admin from <ip>")]
    pool = [
        _row(2, key="invalid user admin from <ip> "),
        _row(3, key="kernel panic not syncing"),
    ]
    kept, dropped = remove_near_duplicates(pool, golden, 90)
    assert [r["id"] for r in kept] == ["id3"]
    assert [r["id"] for r in dropped] == ["id2"]


def test_build_splits_excludes_unreviewed_golden_and_applies_corrections():
    rows = [_row(i) for i in range(50)]
    reviews = {
        "id0": {"status": "corrected", "label": {**LABEL, "severity": "error"}},
        "id1": {"status": "rejected", "label": None},
        "id5": {"status": "corrected", "label": {**LABEL, "category": "network"}},
    }
    s = build_splits(rows, ["id0", "id1", "id2"], reviews)
    golden_ids = [g["id"] for g in s["golden"]]
    assert golden_ids == ["id0"]
    assert s["golden"][0]["label"]["severity"] == "error"
    assert s["unreviewed_golden"] == 2
    pool_ids = {r["id"] for r in s["train"] + s["val"]}
    assert "id0" not in pool_ids and "id1" not in pool_ids
    assert "id2" not in pool_ids  # unreviewed golden candidate is still kept out of training
    corrected = next(r for r in s["train"] + s["val"] if r["id"] == "id5")
    assert corrected["label"]["category"] == "network"


def test_train_and_val_do_not_share_templates():
    rows = [_row(i, template=f"openssh:t{i % 20}") for i in range(200)]
    s = build_splits(rows, [], {}, val_fraction=0.3)
    assert s["val"], "expected some validation rows"
    assert not {r["template"] for r in s["train"]} & {r["template"] for r in s["val"]}
