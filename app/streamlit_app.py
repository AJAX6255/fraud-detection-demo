"""Fraud Detection AI Demo — Streamlit UI, designed to run on Google Cloud Run.

Run locally:   streamlit run app/streamlit_app.py   (from the repo root)
Cloud Run:     see README.md (Dockerfile included).
"""
import json
import os
import re

import duckdb
import pandas as pd
import streamlit as st

from src.llm_analyst import generate_report, nl_to_sql
from src.scorer import load_bundle, score, shap_explain

st.set_page_config(page_title="Fraud Detection AI Demo", page_icon="🛡️", layout="wide")

DATA_PATH = os.environ.get("TRANSACTIONS_PARQUET", "data/processed/transactions.parquet")
MODEL_PATH = os.environ.get("MODEL_ARTIFACT", "artifacts/model.joblib")
EXPLAINER_PATH = os.environ.get("SHAP_EXPLAINER", "artifacts/shap_explainer.joblib")
METRICS_PATH = os.environ.get("METRICS_JSON", "artifacts/metrics.json")

st.title("🛡️ Fraud Detection: ML scores, the LLM investigates")
st.caption("XGBoost flags suspicious transactions from an open dataset (Sparkov simulation) — "
           "an LLM analyst then explains *why* and answers questions in plain English.")


@st.cache_resource
def get_data():
    return pd.read_parquet(DATA_PATH)


@st.cache_resource
def get_bundle():
    return load_bundle(MODEL_PATH)


def history_summary(full: pd.DataFrame, tx: pd.Series) -> str:
    h = full[full["cc_num"] == tx["cc_num"]]
    if h.empty:
        return "No history available for this customer."
    hrs = pd.to_datetime(h["trans_date_trans_time"]).dt.hour
    top_cats = h["category"].value_counts().head(3).index.tolist()
    return (f"{len(h)} historical transactions; avg ${h['amt'].mean():,.2f}, "
            f"max ${h['amt'].max():,.2f}; typical categories: {', '.join(map(str, top_cats))}; "
            f"usual hours {int(hrs.min())}:00–{int(hrs.max())}:00; "
            f"home city {h['city'].iloc[0]}, {h['state'].iloc[0]}.")


def fallback_sql(question: str) -> str:
    """Tiny offline NL->SQL for the most common demo questions."""
    q = question.lower()
    clauses = ["1=1"]
    m = re.search(r"(?:over|above|more than)\s*\$?\s*([\d,]+(?:\.\d+)?)", q)
    if m:
        clauses.append(f"amt > {m.group(1).replace(',', '')}")
    m = re.search(r"category\s+([\w_]+)", q)
    if m:
        clauses.append(f"category = '{m.group(1)}'")
    m = re.search(r"merchant\s+['\"]?([\w\s&\.]+?)['\"]?$", q)
    if m:
        clauses.append(f"lower(merchant) LIKE '%{m.group(1).lower().strip()}%'")
    if "night" in q:
        clauses.append("(hour(trans_date_trans_time) >= 22 OR hour(trans_date_trans_time) <= 4)")
    if "fraud" in q:
        clauses.append("is_fraud = 1")
    return "SELECT * FROM transactions WHERE " + " AND ".join(clauses) + " LIMIT 500"


missing = [p for p in (DATA_PATH, MODEL_PATH) if not os.path.exists(p)]
if missing:
    st.error("Missing: " + ", ".join(missing) +
             " — run the training pipeline first (see README.md).")
    st.stop()

df = get_data()
bundle = get_bundle()
if "feed" not in st.session_state:
    st.session_state.feed = None

tab_feed, tab_inv, tab_sql, tab_perf = st.tabs(
    ["📡 Live feed", "🔍 LLM investigation", "💬 Natural-language query", "📊 Model performance"])

with tab_feed:
    col1, _ = st.columns([1, 3])
    n = col1.slider("Batch size", 1, 50, 10)
    if col1.button("▶️ Score next batch", type="primary"):
        batch = df.sample(n).copy()
        proba, pred = score(batch, bundle)
        batch["fraud_probability"] = proba.round(3)
        batch["flagged"] = pred
        st.session_state.feed = batch.reset_index(drop=True)
    if st.session_state.feed is not None:
        feed = st.session_state.feed

        def highlight(row):
            return (["background-color: #8b000033"] * len(row)) if row["flagged"] else [""] * len(row)

        cols = ["trans_date_trans_time", "first", "last", "merchant", "category",
                "amt", "city", "fraud_probability", "flagged"]
        st.dataframe(feed[cols].style.apply(highlight, axis=1),
                     use_container_width=True, height=400)
        st.caption(f"{int(feed['flagged'].sum())} of {len(feed)} flagged "
                   f"(threshold {bundle.get('threshold', 0.5):.2f})")

with tab_inv:
    if st.session_state.feed is None:
        st.info("Score a batch in the **Live feed** tab first.")
    else:
        feed = st.session_state.feed
        idx = st.selectbox(
            "Pick a transaction", feed.index,
            format_func=lambda i: (f"#{i} — ${feed.loc[i, 'amt']:,.2f} @ "
                                   f"{feed.loc[i, 'merchant']} (p={feed.loc[i, 'fraud_probability']})"))
        if st.button("🤖 Generate investigation report"):
            single = feed.loc[[idx]].drop(columns=["fraud_probability", "flagged"], errors="ignore")
            p, _ = score(single, bundle)
            with st.spinner("LLM analyst writing report…"):
                sf = shap_explain(single, bundle, EXPLAINER_PATH)[0]
                report = generate_report(single.iloc[0].to_dict(),
                                         history_summary(df, single.iloc[0]),
                                         sf, float(p[0]))
            st.markdown(report)

with tab_sql:
    q = st.text_input("Ask about the data",
                      placeholder="e.g. Show fraud transactions over $200 at night")
    if q:
        sql = nl_to_sql(q)
        used_llm = sql is not None
        if sql is None:
            sql = fallback_sql(q)
        st.code(sql, language="sql")
        try:
            con = duckdb.connect()
            con.register("transactions", df)
            res = con.execute(sql).df()
            st.dataframe(res, use_container_width=True)
            st.caption(f"{len(res)} rows · " +
                       ("LLM-generated SQL" if used_llm else
                        "offline pattern-matched SQL (set LLM_PROVIDER for full NL support)"))
        except Exception as e:
            st.error(f"Query failed: {e}")

with tab_perf:
    if os.path.exists(METRICS_PATH):
        m = json.load(open(METRICS_PATH))
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("ROC AUC", f"{m['roc_auc']:.3f}")
        c2.metric("PR AUC", f"{m['pr_auc']:.3f}")
        c3.metric("Recall @ thr", f"{m['recall_at_threshold']:.2f}")
        c4.metric("Precision @ thr", f"{m['precision_at_threshold']:.2f}")
        with st.expander("Full metrics"):
            st.json(m)
    png = os.path.join(os.path.dirname(MODEL_PATH), "shap_summary.png")
    if os.path.exists(png):
        st.image(png, caption="SHAP global feature importance")
