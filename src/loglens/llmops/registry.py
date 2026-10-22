"""Model versioning in MLflow.

Training happens on Colab, where there is no long-lived MLflow server. The
training script writes training_summary.json; this module replays it into the
local MLflow server and registers a model version that points at the exported
artifacts (HF Hub repo + GGUF file + Ollama tag).

MLflow is the record of which versions exist and which one is "champion"
(stable) or "challenger" (canary). The gateway does not call MLflow at runtime;
`promote` writes the chosen versions into configs/gateway.yaml, so a registry
outage can't take the gateway down.

Usage:
  python -m loglens.llmops.registry log-training reports/training/training_summary.json \
      --ollama-tag loglens-qwen2.5-1.5b:v1 --hf-repo you/loglens-qwen2.5-1.5b
  python -m loglens.llmops.registry promote --version 1 --alias champion
"""

import argparse
import contextlib
import json
import tempfile
from pathlib import Path

import yaml

from loglens.config import get_settings

MODEL_NAME = "loglens-triage"
GATEWAY_CONFIG = Path(__file__).resolve().parents[3] / "configs" / "gateway.yaml"


def flatten_metrics(log_history: list[dict]) -> list[tuple[str, float, int]]:
    """(name, value, step) for every numeric entry in a HF Trainer log history."""
    out = []
    for entry in log_history:
        step = int(entry.get("step", 0))
        for key, value in entry.items():
            if key != "step" and isinstance(value, int | float) and not isinstance(value, bool):
                out.append((key, float(value), step))
    return out


def log_training_run(summary_path: Path, ollama_tag: str, hf_repo: str | None) -> str:
    """Create an MLflow run from a training summary and register a model version."""
    import mlflow
    from mlflow import MlflowClient

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    mlflow.set_tracking_uri(get_settings().mlflow_tracking_uri)
    mlflow.set_experiment("loglens-finetune")
    with mlflow.start_run(run_name=f"qlora-{ollama_tag}") as run:
        mlflow.log_params({f"cfg.{k}": v for k, v in summary["config"].items()})
        mlflow.log_params(
            {k: summary[k] for k in ("n_train", "n_val", "trainable_params", "gpu") if k in summary}
        )
        for name, value, step in flatten_metrics(summary.get("log_history", [])):
            mlflow.log_metric(name, value, step=step)
        for key in (
            "train_seconds",
            "best_eval_loss",
            "peak_gpu_mem_gb",
            "sanity_valid_json_rate",
            "sanity_severity_acc",
            "sanity_category_acc",
        ):
            if summary.get(key) is not None:
                mlflow.log_metric(key, float(summary[key]))
        mlflow.log_artifact(str(summary_path))
        curve = summary_path.parent / "loss_curve.png"
        if curve.exists():
            mlflow.log_artifact(str(curve))
        ref = {
            "ollama_tag": ollama_tag,
            "hf_repo": hf_repo,
            "base_model": summary["config"]["base_model"],
        }
        with tempfile.TemporaryDirectory() as tmp:
            ref_path = Path(tmp) / "model_ref.json"
            ref_path.write_text(json.dumps(ref, indent=2))
            mlflow.log_artifact(str(ref_path), artifact_path="model_ref")
        run_id = run.info.run_id
        artifact_uri = run.info.artifact_uri

    client = MlflowClient()
    with contextlib.suppress(mlflow.exceptions.MlflowException):  # already exists
        client.create_registered_model(MODEL_NAME)
    mv = client.create_model_version(
        name=MODEL_NAME,
        source=f"{artifact_uri}/model_ref",
        run_id=run_id,
        tags={"ollama_tag": ollama_tag, "hf_repo": hf_repo or ""},
    )
    print(f"logged run {run_id}; registered {MODEL_NAME} version {mv.version}")
    return mv.version


def update_gateway_config(path: Path, slot: str, ollama_tag: str, version: str) -> dict:
    """Write the model for a routing slot ("stable" or "canary") into gateway.yaml."""
    cfg = yaml.safe_load(path.read_text(encoding="utf-8")) if path.exists() else {}
    cfg.setdefault("models", {})[slot] = {
        "ollama_tag": ollama_tag,
        "registry_version": str(version),
    }
    path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    return cfg


def promote(version: str, alias: str) -> None:
    """Set an MLflow alias and mirror it into the gateway config.

    champion -> gateway "stable" slot, challenger -> gateway "canary" slot.
    """
    import mlflow
    from mlflow import MlflowClient

    mlflow.set_tracking_uri(get_settings().mlflow_tracking_uri)
    client = MlflowClient()
    mv = client.get_model_version(MODEL_NAME, version)
    client.set_registered_model_alias(MODEL_NAME, alias, version)
    slot = {"champion": "stable", "challenger": "canary"}[alias]
    update_gateway_config(GATEWAY_CONFIG, slot, mv.tags["ollama_tag"], version)
    print(f"{MODEL_NAME} v{version} is now '{alias}' -> gateway slot '{slot}'")


def main() -> None:
    p = argparse.ArgumentParser(description="MLflow model registry helpers")
    sub = p.add_subparsers(dest="cmd", required=True)
    lt = sub.add_parser("log-training")
    lt.add_argument("summary", type=Path)
    lt.add_argument("--ollama-tag", required=True)
    lt.add_argument("--hf-repo")
    pr = sub.add_parser("promote")
    pr.add_argument("--version", required=True)
    pr.add_argument("--alias", choices=["champion", "challenger"], required=True)
    args = p.parse_args()
    if args.cmd == "log-training":
        log_training_run(args.summary, args.ollama_tag, args.hf_repo)
    else:
        promote(args.version, args.alias)


if __name__ == "__main__":
    main()
