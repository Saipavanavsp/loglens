"""Pick which parsed lines to send for labeling.

Goals:
- one line per masked message (no paying to label the same event twice),
- spread across event types within a source: round-robin over Drain templates,
  with at most --per-template lines from any one template,
- balanced across sources (round-robin over sources),
- deterministic prefix: the first 50 of a 2,000-line sample are the same 50 you
  get when asking for 50, so pilot labels are reused from the cache.

Usage: python -m loglens.data.sample --n 50
"""

import argparse
import random
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path

from loglens.data.io import read_jsonl, write_jsonl
from loglens.data.paths import PARSED, SAMPLE
from loglens.data.templates import mine_templates


def rank_source(rows: list[dict], seed: int, per_template: int = 5) -> list[dict]:
    """Order one source's rows for diversity: one per key, round-robin over templates.

    Rows must carry a "template" field. BGL alert lines are taken first within
    each template so rare failure events are not crowded out by routine lines.
    """
    rng = random.Random(seed)
    first_per_key: dict[str, dict] = {}
    for row in rows:
        first_per_key.setdefault(row["key"], row)
    buckets: dict[str, list[dict]] = defaultdict(list)
    for row in first_per_key.values():
        buckets[row["template"]].append(row)
    for name, bucket in buckets.items():
        rng.shuffle(bucket)
        # Rows are taken with pop() from the end, so alerts sort last.
        bucket.sort(key=lambda r: bool(r.get("alert")))
        buckets[name] = bucket[-per_template:]
    order = sorted(buckets)
    rng.shuffle(order)
    ranked: list[dict] = []
    while order:
        next_round = []
        for name in order:
            ranked.append(buckets[name].pop())
            if buckets[name]:
                next_round.append(name)
        order = next_round
    return ranked


def interleave(ranked_by_source: dict[str, list[dict]]) -> Iterable[dict]:
    """Round-robin across sources until all are exhausted."""
    iters = {s: iter(rows) for s, rows in sorted(ranked_by_source.items())}
    while iters:
        for source in list(iters):
            row = next(iters[source], None)
            if row is None:
                del iters[source]
            else:
                yield row


def add_templates(rows: list[dict]) -> list[dict]:
    """Attach a Drain template to every row (in place) and return the rows."""
    keys: dict[str, list[str]] = defaultdict(list)
    for row in rows:
        keys[row["source"]].append(row["key"])
    templates = mine_templates(keys)
    for row in rows:
        row["template"] = templates[(row["source"], row["key"])]
    return rows


def build_sample(rows: Iterable[dict], n: int, seed: int = 13, per_template: int = 5) -> list[dict]:
    rows = list(rows)
    if rows and "template" not in rows[0]:
        add_templates(rows)
    by_source: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_source[row["source"]].append(row)
    ranked = {s: rank_source(r, seed, per_template) for s, r in by_source.items()}
    out = []
    for row in interleave(ranked):
        out.append(row)
        if len(out) >= n:
            break
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Select log lines to label")
    parser.add_argument("--n", type=int, default=50)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--per-template", type=int, default=5)
    parser.add_argument("--inp", type=Path, default=PARSED)
    parser.add_argument("--out", type=Path, default=SAMPLE)
    args = parser.parse_args()
    rows = add_templates(list(read_jsonl(args.inp)))
    n_templates = len({r["template"] for r in rows})
    sample = build_sample(rows, args.n, args.seed, args.per_template)
    write_jsonl(args.out, sample)
    counts: dict[str, int] = defaultdict(int)
    for row in sample:
        counts[row["source"]] += 1
    print(f"{n_templates} templates across {len(rows):,} parsed rows")
    print(
        f"wrote {len(sample)} rows ({len({r['template'] for r in sample})} templates) "
        f"to {args.out}: {dict(sorted(counts.items()))}"
    )


if __name__ == "__main__":
    main()
