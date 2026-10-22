"""CI regression gate on a small fixed golden subset.

CI must not spend API money and has no GPU, so it does not call Claude. Two modes:

  replay  (default in CI): recompute metrics from recorded predictions committed in
          evals/fixtures/ci_predictions.jsonl. Catches regressions in parsing,
          schema and metric code, and fails if the recorded model's quality is
          below the thresholds (e.g. after someone commits a worse model's outputs).
  ollama: run the subset live against a local Ollama model (the CI job can start
          Ollama in a service container and pull the GGUF). Slower, but tests the
          real model.

Exit code 1 if any threshold is missed.

  python -m loglens.evals.regression --mode replay
"""

import argparse
import asyncio
import json
import sys
from pathlib import Path

from loglens.data.io import read_jsonl
from loglens.evals.metrics import compute
from loglens.jsonparse import parse_label

ROOT = Path(__file__).resolve().parents[3]
FIXTURE = ROOT / "evals" / "fixtures" / "ci_predictions.jsonl"
THRESHOLDS = ROOT / "evals" / "thresholds.json"


def rescore(rows: list[dict]) -> list[dict]:
    """Re-parse the recorded raw text so parser changes are tested too."""
    out = []
    for r in rows:
        parsed = parse_label(r.get("raw") or "")
        out.append({**r, "pred": parsed.label.model_dump(mode="json") if parsed.ok else None})
    return out


def check(metrics: dict, thresholds: dict) -> list[str]:
    failures = []
    for name, minimum in thresholds.items():
        value = metrics.get(name)
        if value is None or value < minimum:
            failures.append(f"{name} = {value} < {minimum}")
    return failures


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Eval regression gate")
    p.add_argument("--mode", choices=["replay", "ollama"], default="replay")
    p.add_argument("--fixture", type=Path, default=FIXTURE)
    p.add_argument("--thresholds", type=Path, default=THRESHOLDS)
    p.add_argument("--ollama-tag", default=None)
    args = p.parse_args(argv)

    if not args.fixture.exists():
        print(f"No fixture at {args.fixture}. Create it after the first real eval run:")
        print("  python -m loglens.evals.make_ci_fixture --run reports/eval/<run_id>")
        return 0 if args.mode == "replay" else 1
    rows = list(read_jsonl(args.fixture))
    if args.mode == "replay":
        preds = rescore(rows)
    else:
        from loglens.config import get_settings
        from loglens.evals.systems import System, run_system
        from loglens.gateway.providers.ollama import OllamaProvider

        s = get_settings()
        tag = args.ollama_tag or s.local_model
        system = System(
            "ci", OllamaProvider(s.ollama_base_url, tag, "ci", num_thread=s.ollama_num_thread)
        )
        golden = [{**r, "label": r["gold"]} for r in rows]
        preds = asyncio.run(run_system(system, golden, progress=False))

    metrics = compute(preds)
    thresholds = json.loads(args.thresholds.read_text())
    print(json.dumps({k: metrics.get(k) for k in ("n", *thresholds)}, indent=2))
    failures = check(metrics, thresholds)
    for f in failures:
        print(f"FAIL {f}", file=sys.stderr)
    print("regression gate:", "FAILED" if failures else "passed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
