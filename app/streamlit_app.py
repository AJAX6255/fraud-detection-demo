"""Fraud Detection AI Demo — Streamlit UI, designed to run on Google Cloud Run.

Run locally:   streamlit run app/streamlit_app.py   (from the repo root)
Cloud Run:     see README.md (Dockerfile included).
"""
import json
import os
import re

import duckdb
import joblib
import pandas as pd
import streamlit as st

from src.llm_analyst import generate_report, nl_to_sql
from src.scorer import load_bundle, score, shap_explain

st.set_page_config(page_title="Fraud Detection AI Demo", page_icon="🛡️", layout="wide")

DATA_PATH = os.environ.get("TRANSACTIONS_PARQUET", "data/processed/transactions.parquet")
MODEL_PATH = os.environ.get("MODEL_ARTIFACT", "artifacts/model.joblib")
EXPLAINER_PATH = os.environ.get("SHAP_EXPLAINER", "artifacts/shap_explainer.joblib")
METRICS_PATH = os.environ.get("METRICS_JSON", "artifacts/metrics.json")

st.title("🛡️ Fraud Detection: ML Scores, the LLM Investigates")
st.caption("XGBoost flags suspicious transactions from 1.85M open-source transactions (Sparkov simulation) — "
           "an LLM analyst then explains *why* and answers questions in plain English.")


def _mtime(path: str) -> float:
    return os.path.getmtime(path) if os.path.exists(path) else 0.0


@st.cache_resource
def get_data(mtime: float):
    return pd.read_parquet(DATA_PATH)


@st.cache_resource
def get_bundle(mtime: float):
    return load_bundle(MODEL_PATH)


@st.cache_resource
def get_explainer(mtime: float):
    return joblib.load(EXPLAINER_PATH)


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
        clauses.append("(hour(cast(trans_date_trans_time as timestamp)) >= 22 OR hour(cast(trans_date_trans_time as timestamp)) <= 4)")
    if "fraud" in q:
        clauses.append("is_fraud = 1")
    return "SELECT * FROM transactions WHERE " + " AND ".join(clauses) + " LIMIT 500"


missing = [p for p in (DATA_PATH, MODEL_PATH, EXPLAINER_PATH) if not os.path.exists(p)]
if missing:
    st.error("Missing: " + ", ".join(missing) +
             " — run the training pipeline first (see README.md).")
    st.stop()

# Synchronized resource loading based on file modification timestamps
df = get_data(_mtime(DATA_PATH))
bundle = get_bundle(_mtime(MODEL_PATH))
explainer = get_explainer(_mtime(EXPLAINER_PATH))

# Reset feed if model feature set or threshold changed
if "feed_model_mtime" not in st.session_state or st.session_state.feed_model_mtime != _mtime(MODEL_PATH):
    st.session_state.feed = None
    st.session_state.feed_model_mtime = _mtime(MODEL_PATH)

tab_slides, tab_feed, tab_inv, tab_sql, tab_perf = st.tabs(
    ["📽️ Overview & Slides", "📡 Live feed", "🔍 LLM investigation", "💬 Natural-language query", "📊 Model performance"])

with tab_slides:
    st.subheader("Presentation: The Hybrid AI Fraud Architecture")
    
    slide = st.radio(
        "Select Slide",
        ["1. The Business Challenge",
         "2. The Hybrid AI Solution",
         "3. Open-Source Data & DuckDB",
         "4. Production Serving & Cloud Run",
         "5. Live Demo Guide"],
        horizontal=True,
    )
    
    if slide.startswith("1"):
        st.markdown("""
        ### Slide 1: The Business Challenge in Card Fraud
        - **Extreme Class Imbalance**: Fraud accounts for only **~0.2%–0.5%** of transactions.
        - **The Cost Dilemma**:
          - *False Negatives (Missed Fraud)* $\rightarrow$ Direct financial chargebacks & customer churn.
          - *False Positives (Blocked Customers)* $\rightarrow$ Cardholder embarrassment & lost revenue.
        - **Speed Requirements**: Decisions must occur in **sub-100ms** at point-of-sale.
        - **The Black-Box Bottleneck**: Traditional ML models output a raw score (e.g., `0.94`), but compliance officers and investigators need to know **why** before freezing an account.
        """)
        
    elif slide.startswith("2"):
        st.markdown("""
        ### Slide 2: The Hybrid AI Architecture
        Combining three specialized layers:
        
        1. **Fast-Path Machine Learning (XGBoost)**:
           - Evaluates 30 behavioral signals (velocity, haversine distance, z-scores) in **<15ms**.
        2. **Explainable AI (TreeSHAP)**:
           - Computes exact mathematical Shapley contributions showing which specific features drove the risk score.
        3. **Generative AI Analyst (Gemini / OpenAI)**:
           - Synthesizes transaction facts, customer history, and SHAP drivers into a **structured investigation brief** (Verdict, Evidence, Behavior comparison, Recommended next step).
        """)
        
    elif slide.startswith("3"):
        st.markdown("""
        ### Slide 3: Open-Source Benchmark & Zero-Database Footprint
        - **Benchmark Dataset**: Sparkov Credit-Card Transaction Simulation ([Kaggle: `kartik2112/fraud-detection`](https://www.kaggle.com/datasets/kartik2112/fraud-detection)).
        - **1.85 Million Transactions**: Covers diverse demographics, merchants, night-time fraud patterns, and geographic distances.
        - **DuckDB Embedded Analytics**:
          - Zero external database servers required.
          - Queries compressed Parquet directly in-memory with sub-10ms query execution.
          - Instant natural-language-to-SQL translation for ad-hoc fraud hunting.
        """)

    elif slide.startswith("4"):
        st.markdown("""
        ### Slide 4: Dual-Engine Serving & Cloud Run
        - **High-Throughput FastAPI Microservice (`server/`)**:
          - Vectorized batch scoring: **>5,600 transactions / second**.
          - Asynchronous SHAP worker thread pool prevents event-loop blocking.
          - Auto-generated Swagger documentation at `/docs`.
        - **Google Cloud Run Serverless Container**:
          - One-command deploy via `deploy_cloudrun.ps1` / `deploy_cloudrun.sh`.
          - Scales to zero or holds `--min-instances 1` for zero-latency live Zoom presentations.
        """)

    elif slide.startswith("5"):
        st.markdown("""
        ### Slide 5: Live Demo Walkthrough
        Navigate to the tabs above to see each capability in action:
        - **📡 Tab 2 — Live Feed**: Click *'Score next batch'* to simulate streaming transactions; fraud lights up in red.
        - **🔍 Tab 3 — LLM Investigation**: Select any flagged transaction to produce an automated analyst brief grounded in SHAP drivers.
        - **💬 Tab 4 — Natural-Language Query**: Ask questions in plain English (e.g., *"Show fraud over $200 at night"*).
        - **📊 Tab 5 — Model Performance**: Review ROC-AUC (0.998), PR-AUC (0.878), and global SHAP feature importance.
        """)

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
            try:
                p, _ = score(single, bundle)
                with st.spinner("LLM analyst writing report…"):
                    sf = shap_explain(single, bundle, explainer)[0]
                    report = generate_report(single.iloc[0].to_dict(),
                                             history_summary(df, single.iloc[0]),
                                             sf, float(p[0]))
                st.markdown(report)
            except Exception as e:
                st.error(f"Error generating explanation: {e}")
                st.info("Tip: Click '▶️ Score next batch' in the Live feed tab to refresh transaction records.")

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
