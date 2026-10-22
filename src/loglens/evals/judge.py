"""LLM-as-judge for explanation quality, plus a human agreement check.

The judge (Claude) scores each candidate root_cause + suggested_action from 1-5
against a written rubric (prompts/judge/v1.yaml) and the reviewed reference.
Scores are cached on disk, so re-running a report doesn't pay again.

Agreement check: export a sample, score it yourself in the CSV, then compute
exact agreement, within-1 agreement and quadratic-weighted Cohen's kappa
between your scores and the judge's.

  python -m loglens.evals.judge export --run reports/eval/<run_id> --n 30
  (fill the human_score column)
  python -m loglens.evals.judge agreement --csv reports/eval/<run_id>/judge_sample.csv
"""

import argparse
import asyncio
import csv
import hashlib
import json
import random
from pathlib import Path

from loglens.data.io import append_jsonl, read_jsonl
from loglens.data.paths import CACHE
from loglens.pricing import cost_usd, estimate_tokens
from loglens.prompts import load_prompt

JUDGE_CACHE = CACHE / "judge_cache.jsonl"
SCORE_SCHEMA = {
    "type": "object",
    "properties": {
        "score": {"type": "integer", "enum": [1, 2, 3, 4, 5]},
        "reason": {"type": "string"},
    },
    "required": ["score", "reason"],
    "additionalProperties": False,
}


def judge_key(model: str, prompt_ref: str, pred: dict) -> str:
    payload = json.dumps(
        [
            model,
            prompt_ref,
            pred["message"],
            pred["gold"],
            pred["pred"]["root_cause"],
            pred["pred"]["suggested_action"],
        ],
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode()).hexdigest()


def render(prompt, pred: dict) -> str:
    return prompt.render_user(
        source=pred["source"],
        message=pred["message"],
        ref_root_cause=pred["gold"]["root_cause"],
        ref_action=pred["gold"]["suggested_action"],
        cand_root_cause=pred["pred"]["root_cause"],
        cand_action=pred["pred"]["suggested_action"],
    )


def plan_judging(preds: list[dict], model: str, prompt_version: str = "v1"):
    """Return (prompt, todo predictions, cached scores, estimated cost)."""
    prompt = load_prompt("judge", prompt_version)
    cache = {r["key"]: r for r in read_jsonl(JUDGE_CACHE)}
    todo, cached = [], {}
    for p in preds:
        if not p.get("pred"):
            continue
        k = judge_key(model, prompt.ref, p)
        if k in cache:
            cached[(p["system"], p["id"])] = cache[k]["score"]
        else:
            todo.append(p)
    est_in = sum(estimate_tokens(prompt.system + render(prompt, p)) for p in todo)
    est_out = 250 * len(todo)  # short answer + some adaptive thinking
    return prompt, todo, cached, est_in, est_out, cost_usd(model, est_in, est_out)


async def judge_all(client, model: str, prompt, todo: list[dict], concurrency: int = 4) -> dict:
    sem = asyncio.Semaphore(concurrency)
    scores: dict = {}
    usage = {"input_tokens": 0, "output_tokens": 0}

    async def one(p):
        async with sem:
            try:
                resp = await client.messages.create(
                    model=model,
                    max_tokens=2000,
                    system=prompt.system,
                    messages=[{"role": "user", "content": render(prompt, p)}],
                    output_config={
                        "effort": "low",
                        "format": {"type": "json_schema", "schema": SCORE_SCHEMA},
                    },
                )
            except Exception as exc:
                print(f"  judge error on {p['id']}: {type(exc).__name__}")
                return
        usage["input_tokens"] += resp.usage.input_tokens
        usage["output_tokens"] += resp.usage.output_tokens
        text = next((b.text for b in resp.content if b.type == "text"), "")
        try:
            data = json.loads(text)
            score = int(data["score"])
        except (json.JSONDecodeError, KeyError, ValueError, TypeError):
            return
        scores[(p["system"], p["id"])] = score
        append_jsonl(
            JUDGE_CACHE,
            {
                "key": judge_key(model, prompt.ref, p),
                "score": score,
                "reason": data.get("reason", "")[:300],
            },
        )

    await asyncio.gather(*(one(p) for p in todo))
    return {"scores": scores, "usage": usage}


# --- human agreement -----------------------------------------------------------


def export_sample(run_dir: Path, n: int, seed: int = 7) -> Path:
    rows = [r for r in read_jsonl(run_dir / "judged.jsonl") if r.get("judge_score") is not None]
    random.Random(seed).shuffle(rows)
    out = run_dir / "judge_sample.csv"
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "system",
                "id",
                "message",
                "ref_root_cause",
                "cand_root_cause",
                "cand_action",
                "judge_score",
                "human_score",
            ]
        )
        for r in rows[:n]:
            w.writerow(
                [r["system"], r["id"], r["message"], r["gold"]["root_cause"], r["pred"]["root_cause"],
                 r["pred"]["suggested_action"], r["judge_score"], ""]
            )  # fmt: skip
    print(
        f"wrote {min(n, len(rows))} rows to {out}; fill in human_score (1-5) without looking at judge_score"
    )
    return out


def agreement(pairs: list[tuple[int, int]]) -> dict:
    from sklearn.metrics import cohen_kappa_score

    human = [h for h, _ in pairs]
    judge = [j for _, j in pairs]
    n = len(pairs)
    return {
        "n": n,
        "exact_agreement": sum(h == j for h, j in pairs) / n,
        "within_1_agreement": sum(abs(h - j) <= 1 for h, j in pairs) / n,
        "weighted_kappa": cohen_kappa_score(
            human, judge, weights="quadratic", labels=[1, 2, 3, 4, 5]
        ),
        "mean_human": sum(human) / n,
        "mean_judge": sum(judge) / n,
    }


def agreement_from_csv(path: Path) -> dict:
    with path.open(encoding="utf-8") as f:
        pairs = [
            (int(r["human_score"]), int(r["judge_score"]))
            for r in csv.DictReader(f)
            if r["human_score"].strip()
        ]
    if not pairs:
        raise SystemExit("no human_score values filled in")
    result = agreement(pairs)
    (path.parent / "judge_agreement.json").write_text(json.dumps(result, indent=2))
    return result


def main() -> None:
    p = argparse.ArgumentParser(description="Judge agreement tools")
    sub = p.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("export")
    e.add_argument("--run", type=Path, required=True)
    e.add_argument("--n", type=int, default=30)
    a = sub.add_parser("agreement")
    a.add_argument("--csv", type=Path, required=True)
    args = p.parse_args()
    if args.cmd == "export":
        export_sample(args.run, args.n)
    else:
        print(json.dumps(agreement_from_csv(args.csv), indent=2))


if __name__ == "__main__":
    main()
