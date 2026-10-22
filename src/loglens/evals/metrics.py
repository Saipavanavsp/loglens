"""Metrics for one system's predictions on the golden set.

An invalid answer (no schema-valid JSON) counts as wrong for accuracy and F1,
using the placeholder class "invalid". Otherwise a model that fails to answer
on hard lines would look better than one that answers them wrongly.
"""

import statistics

from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

from loglens.schema import CATEGORIES, SEVERITIES

INVALID = "invalid"


def _labels(preds: list[dict], field: str) -> tuple[list[str], list[str]]:
    y_true = [p["gold"][field] for p in preds]
    y_pred = [p["pred"][field] if p.get("pred") else INVALID for p in preds]
    return y_true, y_pred


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    if len(values) == 1:
        return values[0]
    return statistics.quantiles(values, n=100, method="inclusive")[int(q) - 1]


def compute(preds: list[dict]) -> dict:
    """Headline metrics. `preds` rows: gold, pred (dict|None), latency_ms, cost_usd."""
    n = len(preds)
    if n == 0:
        return {"n": 0}
    out: dict = {"n": n, "valid_json_rate": sum(1 for p in preds if p.get("pred")) / n}
    for field, classes in (("severity", SEVERITIES), ("category", CATEGORIES)):
        y_true, y_pred = _labels(preds, field)
        out[f"{field}_acc"] = accuracy_score(y_true, y_pred)
        # Macro F1 over the real classes present in the gold labels; "invalid" counts as a miss.
        present = [c for c in classes if c in set(y_true)]
        out[f"{field}_macro_f1"] = f1_score(
            y_true, y_pred, labels=present, average="macro", zero_division=0
        )
    lat = [p["latency_ms"] for p in preds if p.get("latency_ms") is not None and not p.get("error")]
    out["latency_p50_ms"] = percentile(lat, 50)
    out["latency_p95_ms"] = percentile(lat, 95)
    out["cost_per_1k_usd"] = 1000 * sum(p.get("cost_usd", 0.0) for p in preds) / n
    out["errors"] = sum(1 for p in preds if p.get("error"))
    return out


def compute_by_group(preds: list[dict], key: str) -> dict[str, dict]:
    groups: dict[str, list[dict]] = {}
    for p in preds:
        groups.setdefault(str(p.get(key)), []).append(p)
    return {g: compute(rows) for g, rows in sorted(groups.items())}


def confusion(preds: list[dict], field: str) -> tuple[list[str], list[list[int]]]:
    classes = SEVERITIES if field == "severity" else CATEGORIES
    y_true, y_pred = _labels(preds, field)
    cols = [*classes, INVALID]
    cm = confusion_matrix(y_true, y_pred, labels=cols)
    return cols, cm[: len(classes)].tolist()  # rows: gold classes only
