"""Freeze a small, fixed subset of a real eval run as the CI regression fixture.

  python -m loglens.evals.make_ci_fixture --run reports/eval/<run_id> --system finetuned --n 40

Picks the subset deterministically (by id hash, stratified by source) so it
stays the same across runs, and stores the gold label, recorded raw output and
model version. Commit the result in evals/fixtures/.
"""

import argparse
import hashlib
from collections import defaultdict
from pathlib import Path

from loglens.data.io import read_jsonl, write_jsonl
from loglens.evals.regression import FIXTURE

KEEP = (
    "id",
    "source",
    "message",
    "template",
    "seen_template",
    "gold",
    "raw",
    "model",
    "prompt_version",
    "system",
)


def pick(preds: list[dict], n: int) -> list[dict]:
    by_source: dict[str, list[dict]] = defaultdict(list)
    for p in sorted(preds, key=lambda r: hashlib.sha256(r["id"].encode()).hexdigest()):
        by_source[p["source"]].append(p)
    out: list[dict] = []
    while len(out) < n and any(by_source.values()):
        for src in sorted(by_source):
            if by_source[src] and len(out) < n:
                out.append(by_source[src].pop(0))
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--system", default="finetuned")
    p.add_argument("--n", type=int, default=40)
    args = p.parse_args()
    preds = [r for r in read_jsonl(args.run / "predictions.jsonl") if r["system"] == args.system]
    subset = [{k: r.get(k) for k in KEEP} for r in pick(preds, args.n)]
    write_jsonl(FIXTURE, subset)
    print(f"wrote {len(subset)} rows to {FIXTURE}")


if __name__ == "__main__":
    main()
