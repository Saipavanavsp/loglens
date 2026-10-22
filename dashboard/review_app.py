"""Streamlit tool for reviewing teacher labels on the golden candidates.

Run: streamlit run dashboard/review_app.py

Each save appends a review to data/interim/reviewed.jsonl. Re-reviewing an item
overrides the earlier review.
"""

import streamlit as st

from loglens.data.io import read_jsonl
from loglens.data.review import GOLDEN_CANDIDATES, load_reviews, save_review
from loglens.schema import CATEGORIES, SEVERITIES

st.set_page_config(page_title="LogLens label review", layout="wide")
st.title("Golden set review")

candidates = list(read_jsonl(GOLDEN_CANDIDATES))
if not candidates:
    st.warning("No golden candidates. Run `python -m loglens.data.review select --n 200` first.")
    st.stop()

reviews = load_reviews()
done = sum(c["id"] in reviews for c in candidates)
st.progress(done / len(candidates), text=f"{done}/{len(candidates)} reviewed")

show = st.radio("Show", ["unreviewed", "all"], horizontal=True)
queue = [c for c in candidates if show == "all" or c["id"] not in reviews]
if not queue:
    st.success("Everything is reviewed. Next: python -m loglens.data.split")
    st.stop()

idx = st.number_input("Item", min_value=0, max_value=len(queue) - 1, value=0, step=1)
row = queue[int(idx)]
current = reviews.get(row["id"], {}).get("label") or row["label"]

st.caption(f"id {row['id']} | source {row['source']} | template {row.get('template', '-')}")
st.code(row["raw"], language="text")
if row.get("alert") is not None:
    st.caption(f"Loghub alert flag: {row['alert']}")

with st.form(key=f"f-{row['id']}"):
    c1, c2 = st.columns(2)
    severity = c1.selectbox("Severity", SEVERITIES, index=SEVERITIES.index(current["severity"]))
    category = c2.selectbox("Category", CATEGORIES, index=CATEGORIES.index(current["category"]))
    root_cause = st.text_area("Root cause", current["root_cause"], height=80)
    action = st.text_area("Suggested action", current["suggested_action"], height=80)
    note = st.text_input("Note (optional)")
    b1, b2 = st.columns(2)
    save = b1.form_submit_button("Save", type="primary")
    reject = b2.form_submit_button("Reject line (unusable)")

if save or reject:
    label = {
        "severity": severity,
        "category": category,
        "root_cause": root_cause.strip(),
        "suggested_action": action.strip(),
    }
    if reject:
        status = "rejected"
    elif label == {k: row["label"][k] for k in label}:
        status = "accepted"
    else:
        status = "corrected"
    try:
        save_review(row, status, None if reject else label, note)
        st.success(f"Saved as {status}")
        st.rerun()
    except ValueError as exc:
        st.error(str(exc))

with st.expander("Teacher label (original)"):
    st.json(row["label"])
