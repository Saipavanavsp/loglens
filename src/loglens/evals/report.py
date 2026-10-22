"""Markdown comparison table and confusion-matrix images for an eval run."""

from pathlib import Path

from loglens.evals.metrics import confusion

COLUMNS = [
    ("severity_acc", "Severity acc", "pct"),
    ("severity_macro_f1", "Severity macro F1", "f3"),
    ("category_acc", "Category acc", "pct"),
    ("category_macro_f1", "Category macro F1", "f3"),
    ("valid_json_rate", "Valid JSON", "pct"),
    ("latency_p50_ms", "p50 latency", "ms"),
    ("latency_p95_ms", "p95 latency", "ms"),
    ("cost_per_1k_usd", "API cost / 1k req", "usd"),
    ("judge_mean", "Judge score (1-5)", "f2"),
]


def fmt(value, kind: str) -> str:
    if value is None:
        return "n/a"
    if kind == "pct":
        return f"{100 * value:.1f}%"
    if kind == "f3":
        return f"{value:.3f}"
    if kind == "f2":
        return f"{value:.2f}"
    if kind == "ms":
        return f"{value / 1000:.2f} s" if value >= 1000 else f"{value:.0f} ms"
    if kind == "usd":
        return f"${value:.4f}"
    return str(value)


def comparison_table(results: dict[str, dict]) -> str:
    head = "| System | n | " + " | ".join(c[1] for c in COLUMNS) + " |"
    sep = "|---|---|" + "---|" * len(COLUMNS)
    rows = [
        f"| {name} | {m['n']} | "
        + " | ".join(fmt(m.get(key), kind) for key, _, kind in COLUMNS)
        + " |"
        for name, m in results.items()
    ]
    return "\n".join([head, sep, *rows])


def group_table(by_group: dict[str, dict[str, dict]], title: str) -> str:
    lines = [
        f"| System | {title} | n | Severity acc | Category acc | Valid JSON |",
        "|---|---|---|---|---|---|",
    ]
    for system, groups in by_group.items():
        for g, m in groups.items():
            lines.append(
                f"| {system} | {g} | {m['n']} | {fmt(m['severity_acc'], 'pct')} | "
                f"{fmt(m['category_acc'], 'pct')} | {fmt(m['valid_json_rate'], 'pct')} |"
            )
    return "\n".join(lines)


def save_confusion_png(preds: list[dict], field: str, path: Path, title: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cols, cm = confusion(preds, field)
    rows = cols[:-1]
    fig, ax = plt.subplots(figsize=(1 + 0.8 * len(cols), 1 + 0.6 * len(rows)))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(cols)), cols, rotation=45, ha="right")
    ax.set_yticks(range(len(rows)), rows)
    ax.set_xlabel("predicted")
    ax.set_ylabel("gold")
    ax.set_title(title)
    peak = max((max(r) for r in cm), default=0) or 1
    for i, row in enumerate(cm):
        for j, v in enumerate(row):
            if v:
                ax.text(
                    j,
                    i,
                    str(v),
                    ha="center",
                    va="center",
                    color="white" if v > peak / 2 else "black",
                )
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def write_report(
    run_dir: Path,
    run_id: str,
    results: dict[str, dict],
    by_seen: dict[str, dict[str, dict]],
    by_source: dict[str, dict[str, dict]],
    preds_by_system: dict[str, list[dict]],
    notes: list[str],
) -> Path:
    images = []
    for system, preds in preds_by_system.items():
        for field in ("severity", "category"):
            name = f"cm_{system}_{field}.png"
            save_confusion_png(preds, field, run_dir / name, f"{system}: {field}")
            images.append(f"![{system} {field}]({name})")
    md = [
        f"# Eval run {run_id}",
        "",
        "Golden set: human-reviewed lines never used for training. An invalid answer counts as wrong.",
        "Latency is measured per request from the eval client; local models run one request at a time on CPU.",
        "API cost covers paid providers only (the local model's cost is the hardware it runs on).",
        "",
        "## Summary",
        "",
        comparison_table(results),
        "",
        "## Seen vs unseen event templates",
        "",
        "`seen=True` means the line's Drain template also occurs in the training set.",
        "",
        group_table(by_seen, "seen template"),
        "",
        "## By source",
        "",
        group_table(by_source, "source"),
        "",
        "## Confusion matrices",
        "",
        *images,
        "",
        "## Notes",
        "",
        *[f"- {n}" for n in notes],
        "",
    ]
    out = run_dir / "report.md"
    out.write_text("\n".join(md), encoding="utf-8")
    return out
