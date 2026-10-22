"""Log the measured label-free benchmark and load-test results to MLflow.

python scripts/log_benchmarks_to_mlflow.py   (MLFLOW_TRACKING_URI, default http://localhost:5000)
"""

import csv
import json
from pathlib import Path

import mlflow

from loglens.config import get_settings

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    mlflow.set_tracking_uri(get_settings().mlflow_tracking_uri)
    mlflow.set_experiment("loglens-benchmarks")

    for path in sorted((ROOT / "reports/label_free").glob("*.json")):
        s = json.loads(path.read_text())
        with mlflow.start_run(run_name=f"label-free-{s['model']}"):
            mlflow.set_tags(
                {
                    "kind": "label_free",
                    "model": s["model"],
                    "prompt": s["prompt"],
                    "machine": s["machine"],
                }
            )
            mlflow.log_metrics(
                {
                    k: float(v)
                    for k, v in s.items()
                    if isinstance(v, int | float) and not isinstance(v, bool)
                }
            )
            mlflow.log_artifact(str(path))

    for path in sorted((ROOT / "reports/loadtest").glob("*_stats.csv")):
        with open(path, newline="") as f:
            agg = next(r for r in csv.DictReader(f) if r["Name"] == "Aggregated")
        n = int(agg["Request Count"])
        with mlflow.start_run(run_name=f"loadtest-{path.stem}"):
            mlflow.set_tags(
                {
                    "kind": "loadtest",
                    "users": "8",
                    "duration": "5m",
                    "model": "qwen2.5:1.5b-instruct",
                }
            )
            mlflow.log_metrics(
                {
                    "requests": n,
                    "error_rate": int(agg["Failure Count"]) / n,
                    "rps": float(agg["Requests/s"]),
                    "p50_ms": float(agg["50%"]),
                    "p95_ms": float(agg["95%"]),
                    "p99_ms": float(agg["99%"]),
                }
            )
            mlflow.log_artifact(str(path))
    print("logged to", get_settings().mlflow_tracking_uri)


if __name__ == "__main__":
    main()
