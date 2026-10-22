"""LogLens operations dashboard.

Run: streamlit run dashboard/app.py
Reads request logs from PostgreSQL (DATABASE_URL) and eval reports from reports/.
"""

from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st
from sqlalchemy import create_engine

from loglens.config import get_settings
from loglens.dashboard_data import (
    PROVIDER_COLORS,
    fallback_reasons,
    kpis,
    label_free_results,
    latest_eval,
    load_requests,
    over_time,
)

REPORTS = Path(__file__).resolve().parents[1] / "reports"
INK_MUTED = "#898781"

st.set_page_config(page_title="LogLens dashboard", layout="wide")
st.title("LogLens")

window = st.radio("Time range", ["1h", "24h", "7d", "30d"], index=1, horizontal=True)
hours = {"1h": 1, "24h": 24, "7d": 168, "30d": 720}[window]
freq = {"1h": "5min", "24h": "1h", "7d": "6h", "30d": "1D"}[window]


@st.cache_resource
def engine():
    # The async driver URL works for sync use too with psycopg 3.
    return create_engine(get_settings().database_url, pool_pre_ping=True)


@st.cache_data(ttl=30)
def requests(h: float) -> pd.DataFrame:
    return load_requests(engine(), h)


def color_scale(domain: list[str]) -> alt.Scale:
    return alt.Scale(domain=domain, range=[PROVIDER_COLORS.get(d, INK_MUTED) for d in domain])


try:
    df = requests(hours)
    db_error = None
except Exception as exc:  # show the rest of the page even if the DB is down
    df, db_error = pd.DataFrame(), exc

st.subheader("Traffic")
if db_error is not None:
    st.warning(f"Request log unavailable: {type(db_error).__name__}. Is PostgreSQL running?")
elif df.empty:
    st.info("No requests in this window yet. Send some with the load test or the demo.")
else:
    k = kpis(df)
    c = st.columns(6)
    c[0].metric("Requests", f"{k['requests']:,}")
    c[1].metric("API cost", f"${k['cost_usd']:.4f}")
    c[2].metric(
        "p95 latency", f"{k['latency_p95_ms'] / 1000:.2f} s" if k["latency_p95_ms"] else "n/a"
    )
    c[3].metric(
        "Cache hit rate",
        f"{100 * k['cache_hit_rate']:.1f}%" if k["cache_hit_rate"] is not None else "n/a",
    )
    c[4].metric(
        "Fallback rate",
        f"{100 * k['fallback_rate']:.1f}%" if k["fallback_rate"] is not None else "n/a",
    )
    c[5].metric("Errors", f"{k['errors']:,}")

    t = over_time(df, freq)
    routes = sorted(t["route"].unique())
    left, right = st.columns(2)
    with left:
        st.caption("Requests by route (who served the answer)")
        chart = (
            alt.Chart(t)
            .mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4, stroke="white", strokeWidth=2)
            .encode(
                x=alt.X("bucket:T", title=None),
                y=alt.Y("requests:Q", title="requests", stack=True),
                color=alt.Color(
                    "route:N",
                    scale=color_scale(routes),
                    legend=alt.Legend(title=None, orient="top"),
                ),
                tooltip=["bucket:T", "route:N", "requests:Q"],
            )
        )
        st.altair_chart(chart, use_container_width=True)
    with right:
        st.caption("p95 latency by route (seconds)")
        lat = t.assign(p95_s=t["p95_ms"] / 1000)
        chart = (
            alt.Chart(lat)
            .mark_line(strokeWidth=2, point=alt.OverlayMarkDef(size=64))
            .encode(
                x=alt.X("bucket:T", title=None),
                y=alt.Y("p95_s:Q", title="seconds"),
                color=alt.Color(
                    "route:N",
                    scale=color_scale(routes),
                    legend=alt.Legend(title=None, orient="top"),
                ),
                tooltip=["bucket:T", "route:N", alt.Tooltip("p95_s:Q", format=".2f")],
            )
        )
        st.altair_chart(chart, use_container_width=True)

    left, right = st.columns(2)
    with left:
        st.caption("Fallback reasons (cache misses only)")
        fr = fallback_reasons(df)
        if fr.empty:
            st.write("No fallbacks in this window.")
        else:
            chart = (
                alt.Chart(fr)
                .mark_bar(
                    cornerRadiusTopRight=4,
                    cornerRadiusBottomRight=4,
                    color=PROVIDER_COLORS["claude"],
                )
                .encode(
                    y=alt.Y("reason:N", sort="-x", title=None),
                    x=alt.X("requests:Q", title="requests"),
                    tooltip=["reason:N", "requests:Q"],
                )
            )
            st.altair_chart(chart, use_container_width=True)
    with right:
        st.caption("Cumulative API cost (USD)")
        cost = df.sort_values("ts").assign(cumulative=lambda d: d["cost_usd"].cumsum())
        chart = (
            alt.Chart(cost)
            .mark_line(strokeWidth=2, color=PROVIDER_COLORS["claude"])
            .encode(
                x=alt.X("ts:T", title=None),
                y=alt.Y("cumulative:Q", title="USD"),
                tooltip=["ts:T", alt.Tooltip("cumulative:Q", format="$.4f")],
            )
        )
        st.altair_chart(chart, use_container_width=True)

    with st.expander("Table view"):
        st.dataframe(t, use_container_width=True)

st.subheader("Latest eval")
ev = latest_eval(REPORTS)
if ev is None:
    st.info("No eval run yet. Run `python -m loglens.evals.run` after building the golden set.")
else:
    rows = [
        {"system": name, **{k: v for k, v in m.items() if v is not None}}
        for name, m in ev["results"].items()
    ]
    st.caption(f"Run {ev['run_id']} (full report: reports/eval/{ev['run_id']}/report.md)")
    st.dataframe(pd.DataFrame(rows).set_index("system"), use_container_width=True)

lf = label_free_results(REPORTS)
if lf:
    st.subheader("Local model benchmark (label-free)")
    st.caption(
        "Valid-JSON rate, latency and confidence on sample lines. Does not measure accuracy."
    )
    st.dataframe(pd.DataFrame(lf).set_index("model"), use_container_width=True)
