"""Error analysis: where does a system get it wrong?

  python -m loglens.evals.errors --run reports/eval/<run_id> --system finetuned

Writes errors_<system>.md next to the run: failures grouped by gold category,
the most common (gold -> predicted) confusions, and example lines.
"""

import argparse
from collections import Counter, defaultdict
from pathlib import Path

from loglens.data.io import read_jsonl


def failures(preds: list[dict]) -> list[dict]:
    out = []
    for p in preds:
        pred = p.get("pred")
        kinds = []
        if not pred:
            kinds.append("invalid_output" if not p.get("error") else "request_error")
        else:
            if pred["severity"] != p["gold"]["severity"]:
                kinds.append("severity")
            if pred["category"] != p["gold"]["category"]:
                kinds.append("category")
        if kinds:
            out.append({**p, "failure": kinds})
    return out


def analyse(preds: list[dict], max_examples: int = 5) -> str:
    fails = failures(preds)
    n = len(preds)
    lines = [f"Failures: {len(fails)}/{n} lines ({100 * len(fails) / max(1, n):.1f}%)", ""]
    kinds = Counter(k for f in fails for k in f["failure"])
    lines += [
        "| Failure type | count |",
        "|---|---|",
        *[f"| {k} | {v} |" for k, v in kinds.most_common()],
        "",
    ]

    confusions = Counter(
        (f["gold"][field], f["pred"][field], field)
        for f in fails
        if f.get("pred")
        for field in ("severity", "category")
        if field in f["failure"]
    )
    lines += [
        "## Most common confusions",
        "",
        "| field | gold | predicted | count |",
        "|---|---|---|---|",
    ]
    lines += [f"| {fld} | {g} | {pr} | {c} |" for (g, pr, fld), c in confusions.most_common(12)]
    lines.append("")

    by_cat: dict[str, list[dict]] = defaultdict(list)
    for f in fails:
        by_cat[f["gold"]["category"]].append(f)
    lines += ["## Failures by gold category", ""]
    for cat, items in sorted(by_cat.items(), key=lambda kv: -len(kv[1])):
        total = sum(1 for p in preds if p["gold"]["category"] == cat)
        lines += [f"### {cat}: {len(items)}/{total} wrong", ""]
        for f in items[:max_examples]:
            pred = f.get("pred") or {}
            lines.append(
                f"- `{f['message'][:140]}`  \n  gold {f['gold']['severity']}/{f['gold']['category']}, "
                f"predicted {pred.get('severity', '-')}/{pred.get('category', '-')}"
                + (f" ({f.get('parse_error') or f.get('error')})" if not pred else "")
            )
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    p = argparse.ArgumentParser(description="Group a system's eval failures")
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--system", default="finetuned")
    args = p.parse_args()
    preds = [r for r in read_jsonl(args.run / "predictions.jsonl") if r["system"] == args.system]
    if not preds:
        raise SystemExit(f"no predictions for system {args.system!r} in {args.run}")
    md = f"# Error analysis: {args.system}\n\n" + analyse(preds)
    out = args.run / f"errors_{args.system}.md"
    out.write_text(md, encoding="utf-8")
    print(md)
    print(f"\nwritten to {out}")


if __name__ == "__main__":
    main()
