"""Build train / val / golden splits from labeled data.

Rules:
1. Golden = golden candidates that a human reviewed and accepted or corrected.
   Unreviewed candidates are left out (and reported); golden is never trained on.
2. Human corrections replace teacher labels anywhere they exist; rejected lines
   are dropped.
3. Leakage guard: any train/val line whose masked text is >= --fuzzy (default 90)
   similar to a golden line is removed.
4. Train and val are split by Drain template (hash of the template), so the
   two never share an event type and val measures generalisation.

Each golden row gets "seen_template": whether its template also occurs in train.
Evals report metrics for seen and unseen templates separately.

Also writes chat-format files (system/user/assistant) for fine-tuning.

Usage: python -m loglens.data.split
"""

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from rapidfuzz import fuzz, process

from loglens.data.io import read_jsonl, write_jsonl
from loglens.data.paths import PROCESSED, REPORTS, TEACHER_LABELS
from loglens.data.review import GOLDEN_CANDIDATES, load_reviews
from loglens.prompts import load_prompt, triage_messages


def _bucket(template: str, val_fraction: float) -> str:
    h = int(hashlib.sha1(template.encode(), usedforsecurity=False).hexdigest()[:8], 16)
    return "val" if (h % 10_000) / 10_000 < val_fraction else "train"


def apply_reviews(rows: list[dict], reviews: dict[str, dict]) -> list[dict]:
    out = []
    for row in rows:
        rev = reviews.get(row["id"])
        if rev is None:
            out.append({**row, "reviewed": False})
        elif rev["status"] == "rejected":
            continue
        else:
            out.append({**row, "label": rev["label"], "reviewed": True, "label_source": "human"})
    return out


def remove_near_duplicates(
    pool: list[dict], golden: list[dict], threshold: float
) -> tuple[list[dict], list[dict]]:
    """Drop pool rows whose masked key is >= threshold similar to any golden key."""
    if not pool or not golden:
        return pool, []
    scores = process.cdist(
        [r["key"] for r in pool],
        [g["key"] for g in golden],
        scorer=fuzz.ratio,
        score_cutoff=threshold,
        workers=-1,
    )
    leaked_mask = scores.max(axis=1) >= threshold
    kept = [r for r, leak in zip(pool, leaked_mask, strict=True) if not leak]
    dropped = [r for r, leak in zip(pool, leaked_mask, strict=True) if leak]
    return kept, dropped


def to_chat(row: dict, prompt) -> dict:
    messages = triage_messages(prompt, row["source"], row["message"])
    messages.append({"role": "assistant", "content": json.dumps(row["label"], ensure_ascii=False)})
    return {"id": row["id"], "messages": messages}


def build_splits(
    labeled: list[dict],
    golden_ids: list[str],
    reviews: dict[str, dict],
    fuzzy: float = 90.0,
    val_fraction: float = 0.1,
) -> dict:
    rows = apply_reviews(labeled, reviews)
    golden_set = set(golden_ids)
    golden = [r for r in rows if r["id"] in golden_set and r["reviewed"]]
    unreviewed_golden = len(golden_set) - len(golden)
    pool = [r for r in rows if r["id"] not in golden_set]
    pool, leaked = remove_near_duplicates(pool, golden, fuzzy)
    train = [r for r in pool if _bucket(r["template"], val_fraction) == "train"]
    val = [r for r in pool if _bucket(r["template"], val_fraction) == "val"]
    train_templates = {r["template"] for r in train}
    for g in golden:
        g["seen_template"] = g["template"] in train_templates
    return {
        "train": train,
        "val": val,
        "golden": golden,
        "leaked": leaked,
        "unreviewed_golden": unreviewed_golden,
    }


def main() -> None:
    p = argparse.ArgumentParser(description="Create train/val/golden splits")
    p.add_argument("--fuzzy", type=float, default=90.0)
    p.add_argument("--val-fraction", type=float, default=0.1)
    p.add_argument("--prompt-version", default="v1")
    p.add_argument("--out", type=Path, default=PROCESSED)
    args = p.parse_args()

    labeled = list(read_jsonl(TEACHER_LABELS))
    golden_ids = [r["id"] for r in read_jsonl(GOLDEN_CANDIDATES)]
    if not golden_ids:
        raise SystemExit("No golden candidates. Run: python -m loglens.data.review select")
    s = build_splits(labeled, golden_ids, load_reviews(), args.fuzzy, args.val_fraction)

    prompt = load_prompt("triage", args.prompt_version)
    for name in ("train", "val", "golden"):
        write_jsonl(args.out / f"{name}.jsonl", s[name])
    for name in ("train", "val"):
        write_jsonl(args.out / f"{name}_chat.jsonl", (to_chat(r, prompt) for r in s[name]))

    summary = {
        "train": len(s["train"]),
        "val": len(s["val"]),
        "golden": len(s["golden"]),
        "golden_seen_template": sum(g["seen_template"] for g in s["golden"]),
        "removed_near_duplicates": len(s["leaked"]),
        "golden_candidates_not_reviewed": s["unreviewed_golden"],
        "fuzzy_threshold": args.fuzzy,
        "prompt": prompt.ref,
        "train_by_source": dict(Counter(r["source"] for r in s["train"])),
    }
    REPORTS.mkdir(exist_ok=True)
    (REPORTS / "split_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    if s["unreviewed_golden"]:
        print(
            f"WARNING: {s['unreviewed_golden']} golden candidates are not reviewed and were "
            "left out. Review them in the Streamlit app, then re-run this script."
        )


if __name__ == "__main__":
    main()
