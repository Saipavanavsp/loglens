"""Draw the README figures from measured data in reports/.

    python scripts/make_figures.py

Inputs (all produced by real runs): reports/label_free/*.jsonl, reports/loadtest/*.csv,
reports/benchmarks/threads.json. Output: docs/images/*.png
"""

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "images"

# Reference palette (fixed order) and chart ink.
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, MUTED, GRID = "#0b0b0b", "#898781", "#e1e0d9"


def style(ax, title: str) -> None:
    ax.set_title(title, loc="left", fontsize=12, color=INK, pad=10)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelcolor=INK)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def confidence_histogram() -> None:
    path = ROOT / "reports/label_free/qwen2.5_1.5b-instruct.jsonl"
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    conf = [r["confidence"] for r in rows if r.get("confidence") is not None]
    below = sum(c < 0.7 for c in conf) / len(conf)
    fig, ax = plt.subplots(figsize=(7, 3.6))
    ax.hist(conf, bins=[i / 20 for i in range(21)], color=BLUE, edgecolor="white", linewidth=2)
    ax.axvline(0.7, color=ORANGE, linewidth=2)
    ax.text(
        0.71,
        ax.get_ylim()[1] * 0.9,
        f"threshold 0.70\n{below:.0%} of lines below -> Claude",
        color=INK,
        fontsize=9,
    )
    ax.set_xlabel("confidence (min p of severity / category decision token)", color=INK)
    ax.set_ylabel("log lines", color=INK)
    style(ax, f"Base Qwen2.5-1.5B confidence on {len(conf)} sample lines (before fine-tuning)")
    fig.tight_layout()
    fig.savefig(OUT / "base_confidence.png", dpi=140)
    plt.close(fig)


def load_test_latency() -> None:
    def read(path):
        with open(path, newline="") as f:
            return {r["Name"]: r for r in csv.DictReader(f)}

    run = read(ROOT / "reports/loadtest/run2_no_timeout_retry_stats.csv")
    # Per Locust task. Cold-pool lines repeat, so after first use they are cached too
    # (p50 ~9 ms); the tail of every task is cache misses queued behind the model.
    names = [
        ("triage:hot-exact", "hot set, exact repeats"),
        ("triage:hot-variant", "hot templates,\nnew IPs/users"),
        ("triage:cold", "cold pool\n(cached after first use)"),
    ]
    pcts = [("50%", "p50", BLUE), ("95%", "p95", ORANGE), ("99%", "p99", AQUA)]
    fig, ax = plt.subplots(figsize=(7.5, 4.3))
    width = 0.26
    for i, (col, label, color) in enumerate(pcts):
        vals = [float(run[n][col]) / 1000 for n, _ in names]
        xs = [j + (i - 1) * width for j in range(len(names))]
        bars = ax.bar(xs, vals, width - 0.03, color=color, label=label)
        for b, v in zip(bars, vals, strict=True):
            ax.text(
                b.get_x() + b.get_width() / 2,
                v * 1.15 if v > 0 else 0.01,
                f"{v:.3g}s",
                ha="center",
                fontsize=8,
                color=INK,
            )
    ax.set_yscale("log")
    ax.set_xticks(range(len(names)), [lbl for _, lbl in names])
    ax.set_ylabel("latency (seconds, log scale)", color=INK)
    ax.set_ylim(top=80)
    ax.legend(frameon=False, ncols=3, loc="upper center", bbox_to_anchor=(0.5, -0.2))
    style(ax, "Load test: 8 users, 5 min, base model on CPU (latency by task)")
    fig.tight_layout()
    fig.savefig(OUT / "loadtest_latency.png", dpi=140)
    plt.close(fig)


def thread_benchmark() -> None:
    data = json.loads((ROOT / "reports/benchmarks/threads.json").read_text())["runs"]
    labels = [
        f"{r['num_thread']} threads" + (" (default)" if r["num_thread"] == 16 else "") for r in data
    ]
    vals = [r["gen_tokens_per_s"] for r in data]
    fig, ax = plt.subplots(figsize=(6.5, 3.2))
    bars = ax.barh(labels, vals, color=[ORANGE, BLUE, BLUE], height=0.55)
    for b, v in zip(bars, vals, strict=True):
        ax.text(
            v + 0.5,
            b.get_y() + b.get_height() / 2,
            f"{v:g} tok/s",
            va="center",
            fontsize=9,
            color=INK,
        )
    ax.invert_yaxis()
    ax.set_xlabel("generation speed (tokens/s)", color=INK)
    ax.grid(axis="x", color=GRID)
    style(ax, "llama.cpp thread count on an 8-core / 16-vCPU laptop")
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    fig.savefig(OUT / "thread_benchmark.png", dpi=140)
    plt.close(fig)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    confidence_histogram()
    load_test_latency()
    thread_benchmark()
    print("wrote", sorted(p.name for p in OUT.glob("*.png")))
