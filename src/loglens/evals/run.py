"""Run the golden-set eval end to end with one command.

  python -m loglens.evals.run                         # base + finetuned (local, free)
  python -m loglens.evals.run --systems base,finetuned,claude --judge --no-dry-run

Steps: predict with each system -> metrics -> (optional) LLM judge -> report
(markdown + confusion matrices) -> PostgreSQL + MLflow (skipped if unreachable
or disabled).

Paid steps (system "claude" and --judge) go through the spend guard: dry run by
default, request cap, cost estimate and confirmation.
"""

import argparse
import asyncio
import json
import time
from pathlib import Path

from loglens.config import get_settings
from loglens.data.io import read_jsonl, write_jsonl
from loglens.data.paths import PROCESSED, REPORTS
from loglens.evals import judge as judge_mod
from loglens.evals.metrics import compute, compute_by_group
from loglens.evals.report import write_report
from loglens.evals.systems import build_systems, run_system
from loglens.pricing import cost_usd, estimate_tokens
from loglens.prompts import load_prompt
from loglens.spend import SpendPlan, confirm_spend


def _claude_plan(rows: list[dict], model: str) -> SpendPlan:
    prompt = load_prompt("triage", "v1")
    est_in = sum(estimate_tokens(prompt.system + r["message"]) + 20 for r in rows)
    est_out = 350 * len(rows)
    return SpendPlan(model, len(rows), est_in, est_out, cost_usd(model, est_in, est_out))


def main(argv: list[str] | None = None) -> int:
    s = get_settings()
    p = argparse.ArgumentParser(description="Golden-set evaluation")
    p.add_argument("--golden", type=Path, default=PROCESSED / "golden.jsonl")
    p.add_argument("--systems", default="base,finetuned")
    p.add_argument("--finetuned-tag", default=s.local_model)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--judge", action="store_true", help="score explanations with the LLM judge")
    p.add_argument("--max-requests", type=int, default=s.max_requests)
    p.add_argument("--dry-run", action=argparse.BooleanOptionalAction, default=s.dry_run)
    p.add_argument("--yes", action="store_true")
    p.add_argument("--no-db", action="store_true")
    p.add_argument("--no-mlflow", action="store_true")
    p.add_argument("--out", type=Path, default=REPORTS / "eval")
    args = p.parse_args(argv)

    rows = list(read_jsonl(args.golden))[: args.limit]
    if not rows:
        print(
            f"No golden rows at {args.golden}. Build splits first (python -m loglens.data.split)."
        )
        return 1
    names = [n.strip() for n in args.systems.split(",") if n.strip()]

    if "claude" in names and not confirm_spend(
        _claude_plan(rows, s.fallback_model),
        max_requests=args.max_requests,
        dry_run=args.dry_run,
        assume_yes=args.yes,
    ):
        names.remove("claude")
        print("Skipping 'claude' (not confirmed).")

    run_id = time.strftime("%Y%m%d-%H%M%S")
    run_dir = args.out / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    systems = build_systems(names, s, args.finetuned_tag)

    preds_by_system: dict[str, list[dict]] = {}
    for system in systems:
        print(f"Running {system.name} ({system.provider.model}) on {len(rows)} lines")
        preds_by_system[system.name] = asyncio.run(run_system(system, rows))

    notes: list[str] = []
    if args.judge:
        notes += _run_judge(preds_by_system, s, args)

    results = {name: compute(preds) for name, preds in preds_by_system.items()}
    for name, preds in preds_by_system.items():
        scores = [p["judge_score"] for p in preds if p.get("judge_score") is not None]
        results[name]["judge_mean"] = sum(scores) / len(scores) if scores else None
        results[name]["judge_n"] = len(scores)
    by_seen = {n: compute_by_group(p, "seen_template") for n, p in preds_by_system.items()}
    by_source = {n: compute_by_group(p, "source") for n, p in preds_by_system.items()}
    meta = {
        sy.name: {"model": sy.provider.model, "prompt_version": sy.prompt_version} for sy in systems
    }

    all_preds = [p for preds in preds_by_system.values() for p in preds]
    write_jsonl(run_dir / "predictions.jsonl", all_preds)
    if args.judge:
        write_jsonl(
            run_dir / "judged.jsonl", [p for p in all_preds if p.get("judge_score") is not None]
        )
    (run_dir / "metrics.json").write_text(
        json.dumps({"run_id": run_id, "results": results, "meta": meta}, indent=2)
    )
    report = write_report(run_dir, run_id, results, by_seen, by_source, preds_by_system, notes)
    (args.out / "latest.md").write_text(
        report.read_text(encoding="utf-8").replace("](cm_", f"]({run_id}/cm_"), encoding="utf-8"
    )
    print(f"\nReport: {report}")
    print(report.read_text(encoding="utf-8").split("## Seen")[0])

    if not args.no_db:
        try:
            from loglens.evals.store import to_postgres

            n = to_postgres(s.database_url, run_id, results, meta)
            print(f"saved {n} metric rows to PostgreSQL")
        except Exception as exc:
            print(f"PostgreSQL skipped: {type(exc).__name__}: {str(exc)[:120]}")
    if not args.no_mlflow:
        try:
            from loglens.evals.store import to_mlflow

            to_mlflow(s.mlflow_tracking_uri, run_id, results, meta, run_dir)
            print("logged to MLflow")
        except Exception as exc:
            print(f"MLflow skipped: {type(exc).__name__}: {str(exc)[:120]}")
    return 0


def _run_judge(preds_by_system: dict[str, list[dict]], s, args) -> list[str]:
    all_preds = [p for preds in preds_by_system.values() for p in preds]
    prompt, todo, cached, est_in, est_out, est_cost = judge_mod.plan_judging(
        all_preds, s.judge_model
    )
    plan = SpendPlan(s.judge_model, len(todo), est_in, est_out, est_cost)
    scores = dict(cached)
    if todo and confirm_spend(
        plan, max_requests=args.max_requests, dry_run=args.dry_run, assume_yes=args.yes
    ):
        import anthropic

        client = anthropic.AsyncAnthropic(
            api_key=s.anthropic_api_key.get_secret_value(), max_retries=4
        )
        result = asyncio.run(judge_mod.judge_all(client, s.judge_model, prompt, todo))
        scores.update(result["scores"])
        u = result["usage"]
        print(f"judge usage: {u['input_tokens']:,} in / {u['output_tokens']:,} out, "
              f"${cost_usd(s.judge_model, u['input_tokens'], u['output_tokens']):.4f}")  # fmt: skip
    for p in all_preds:
        p["judge_score"] = scores.get((p["system"], p["id"]))
    return [
        f"Judge: {s.judge_model}, rubric prompts/judge/{prompt.version}.yaml, {len(scores)} scored answers."
    ]


if __name__ == "__main__":
    raise SystemExit(main())
