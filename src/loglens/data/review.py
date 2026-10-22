"""Golden-set selection and storage for human label review.

Reviews are appended to data/interim/reviewed.jsonl; the latest entry per id
wins, so a mistake is fixed by reviewing the item again. Nothing is rewritten
in place.

Usage: python -m loglens.data.review select --n 200
Then:  streamlit run dashboard/review_app.py
"""

import argparse
import time
from collections import defaultdict
from pathlib import Path

from loglens.data.io import append_jsonl, read_jsonl, write_jsonl
from loglens.data.paths import INTERIM, REVIEWED, TEACHER_LABELS
from loglens.schema import TriageLabel

GOLDEN_CANDIDATES = INTERIM / "golden_candidates.jsonl"
STATUSES = ("accepted", "corrected", "rejected")


def select_golden(rows: list[dict], n: int) -> list[dict]:
    """One line per template, round-robin across sources, in input order.

    Input order is already diversity-ranked by sample.py, so taking the first
    unseen template per source keeps the golden set varied.
    """
    by_source: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_source[row["source"]].append(row)
    used_templates: set[str] = set()
    iters = {s: iter(v) for s, v in sorted(by_source.items())}
    out: list[dict] = []
    while iters and len(out) < n:
        for source in list(iters):
            row = next(iters[source], None)
            while row is not None and row.get("template") in used_templates:
                row = next(iters[source], None)
            if row is None:
                del iters[source]
                continue
            used_templates.add(row.get("template"))
            out.append(row)
            if len(out) >= n:
                break
    return out


def load_reviews(path: Path = REVIEWED) -> dict[str, dict]:
    """Latest review per id."""
    latest: dict[str, dict] = {}
    for row in read_jsonl(path):
        latest[row["id"]] = row
    return latest


def save_review(
    row: dict,
    status: str,
    label: dict | None,
    note: str = "",
    path: Path = REVIEWED,
) -> dict:
    if status not in STATUSES:
        raise ValueError(f"status must be one of {STATUSES}")
    if status != "rejected":
        label = TriageLabel.model_validate(label).model_dump(mode="json")
    entry = {
        "id": row["id"],
        "status": status,
        "label": label,
        "teacher_label": row.get("label"),
        "note": note,
        "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    append_jsonl(path, entry)
    return entry


def main() -> None:
    p = argparse.ArgumentParser(description="Golden set candidate selection")
    sub = p.add_subparsers(dest="cmd", required=True)
    sel = sub.add_parser("select", help="pick golden candidates from teacher labels")
    sel.add_argument("--n", type=int, default=200)
    sub.add_parser("progress", help="show review progress")
    args = p.parse_args()

    if args.cmd == "select":
        rows = list(read_jsonl(TEACHER_LABELS))
        if not rows:
            raise SystemExit(f"No teacher labels at {TEACHER_LABELS}. Run labeling first.")
        golden = select_golden(rows, args.n)
        write_jsonl(GOLDEN_CANDIDATES, golden)
        print(f"wrote {len(golden)} golden candidates to {GOLDEN_CANDIDATES}")
    else:
        cands = list(read_jsonl(GOLDEN_CANDIDATES))
        reviews = load_reviews()
        done = [c for c in cands if c["id"] in reviews]
        by_status: dict[str, int] = defaultdict(int)
        for c in done:
            by_status[reviews[c["id"]]["status"]] += 1
        print(f"reviewed {len(done)}/{len(cands)} golden candidates: {dict(by_status)}")


if __name__ == "__main__":
    main()
