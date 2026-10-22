"""Capture README screenshots from a running stack with headless Chromium.

    pip install playwright && python -m playwright install chromium
    python scripts/capture_screenshots.py

Expects: compose stack on 127.0.0.1 (gateway 8000, dashboard 8501, MLflow 5000,
Jaeger 16686) and the Space image on 127.0.0.1:7860 for the demo page.
"""

import json
import sys
import urllib.request
from pathlib import Path

from playwright.sync_api import sync_playwright

OUT = Path(__file__).resolve().parents[1] / "docs" / "images"
DEMO_LINES = [
    "Out of Memory: Killed process 2592 (httpd).",
    "Failed password for invalid user admin from 198.51.100.23 port 52211 ssh2",
    "EXT3-fs error (device sda3): ext3_find_entry: reading directory #2 offset 0",
]


def model_trace_id() -> str | None:
    """Most recent trace that includes a local model call."""
    url = "http://127.0.0.1:16686/api/traces?service=loglens-gateway&limit=50&lookback=2h"
    data = json.load(urllib.request.urlopen(url, timeout=10))["data"]  # noqa: S310 - local URL
    for t in sorted(data, key=lambda t: -max(s["startTime"] for s in t["spans"])):
        if any(s["operationName"] == "provider.local" for s in t["spans"]):
            return t["traceID"]
    return None


def main(only: set[str]) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport={"width": 1400, "height": 900}, device_scale_factor=1)

        if not only or "docs" in only:
            page.goto("http://127.0.0.1:8000/docs")
            page.wait_for_selector(".opblock", timeout=30_000)
            page.screenshot(path=OUT / "api_docs.png")

        if not only or "demo" in only:
            page.goto("http://127.0.0.1:7860/")
            page.fill("#logs", "\n".join(DEMO_LINES))
            page.click("#go")
            page.wait_for_selector("#out .item", timeout=240_000)
            page.wait_for_function(
                "document.querySelector('#status').textContent.startsWith('Done')", timeout=240_000
            )
            page.screenshot(path=OUT / "demo_page.png", full_page=True)

        if not only or "dashboard" in only:
            page.goto("http://127.0.0.1:8501/")
            page.get_by_text("Requests by route").wait_for(timeout=60_000)
            page.wait_for_timeout(6_000)  # let the Vega charts finish rendering
            page.screenshot(path=OUT / "dashboard.png", full_page=True)

        if not only or "jaeger" in only:
            trace = model_trace_id()
            if trace:
                page.goto(f"http://127.0.0.1:16686/trace/{trace}")
                page.wait_for_timeout(5_000)
                page.screenshot(path=OUT / "jaeger_trace.png")

        if not only or "mlflow" in only:
            exp = json.load(
                urllib.request.urlopen(  # noqa: S310 - local URL
                    "http://127.0.0.1:5000/api/2.0/mlflow/experiments/get-by-name"
                    "?experiment_name=loglens-benchmarks",
                    timeout=10,
                )
            )["experiment"]["experiment_id"]
            page.goto(f"http://127.0.0.1:5000/#/experiments/{exp}/runs")
            page.wait_for_timeout(6_000)
            # Crop off the assistant side panel.
            page.screenshot(
                path=OUT / "mlflow_runs.png", clip={"x": 0, "y": 0, "width": 1035, "height": 420}
            )

        if not only or "ci" in only:
            page.goto("https://github.com/Saipavanavsp/loglens/actions", wait_until="networkidle")
            page.wait_for_timeout(3_000)
            page.screenshot(
                path=OUT / "github_ci.png", clip={"x": 320, "y": 70, "width": 1080, "height": 500}
            )

        browser.close()
    print("saved", sorted(f.name for f in OUT.glob("*.png")))


if __name__ == "__main__":
    main(set(sys.argv[1:]))
