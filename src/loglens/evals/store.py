"""Persist eval results to PostgreSQL and MLflow. Both are optional per run."""

import asyncio
from pathlib import Path


def to_postgres(
    database_url: str, run_id: str, results: dict[str, dict], meta: dict[str, dict]
) -> int:
    from loglens.db import EvalMetric, create_tables, make_engine, sessionmaker

    async def _write() -> int:
        engine = make_engine(database_url)
        try:
            await create_tables(engine)
            rows = []
            for system, metrics in results.items():
                for name, value in metrics.items():
                    if isinstance(value, int | float) and value is not None and name != "n":
                        rows.append(
                            EvalMetric(
                                run_id=run_id,
                                system=system,
                                model_version=meta[system].get("model"),
                                prompt_version=meta[system].get("prompt_version"),
                                metric=name,
                                value=float(value),
                                n=int(metrics["n"]),
                            )
                        )
            async with sessionmaker(engine)() as s, s.begin():
                s.add_all(rows)
            return len(rows)
        finally:
            await engine.dispose()

    return asyncio.run(_write())


def to_mlflow(
    tracking_uri: str, run_id: str, results: dict[str, dict], meta: dict[str, dict], run_dir: Path
) -> None:
    import mlflow

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment("loglens-eval")
    for system, metrics in results.items():
        with mlflow.start_run(run_name=f"{run_id}-{system}"):
            mlflow.set_tags(
                {
                    "eval_run": run_id,
                    "system": system,
                    **{k: str(v) for k, v in meta[system].items()},
                }
            )
            mlflow.log_metrics(
                {
                    k: float(v)
                    for k, v in metrics.items()
                    if isinstance(v, int | float) and v is not None
                }
            )
            mlflow.log_artifact(str(run_dir / "report.md"))
            for png in run_dir.glob(f"cm_{system}_*.png"):
                mlflow.log_artifact(str(png))
