# Fraud Detection AI Engine & Microservice

A production-grade fraud detection system combining **XGBoost, SHAP explainability, DuckDB analytical engine, and an LLM investigator** on credit-card transaction data ([Kaggle: kartik2112/fraud-detection Sparkov simulation](https://www.kaggle.com/datasets/kartik2112/fraud-detection)).

This repo provides both:
1. **High-Throughput FastAPI Microservice Engine** (`server/`): Sub-millisecond scoring, vectorized batch inference (>6,000 tx/s), non-blocking asynchronous SHAP worker pool, and in-memory DuckDB query engine.
2. **Interactive Streamlit Demo UI** (`app/`): 4-tab interactive dashboard (Live feed, LLM investigation, Natural-language query, Model performance).

---

## Architecture Overview

```
├── server/                        # High-Performance FastAPI Microservice
│   ├── main.py                    # App entrypoint, lifespan caching, CORS & timing headers
│   ├── config.py                  # Environment & engine configuration
│   ├── schemas.py                 # Pydantic v2 validation models
│   ├── routes/
│   │   ├── scoring.py             # POST /api/v1/score & POST /api/v1/score/batch
│   │   ├── investigation.py       # POST /api/v1/investigate (async SHAP + LLM)
│   │   ├── query.py               # POST /api/v1/query (DuckDB + NL-to-SQL)
│   │   ├── metrics.py             # GET /api/v1/metrics
│   │   └── health.py              # GET /health, /live, /ready probes
│   └── services/
│       ├── scoring_service.py     # In-memory cached model, vectorized inference
│       ├── shap_service.py        # Asynchronous worker thread pool for TreeExplainer
│       ├── analyst_service.py     # Gemini / OpenAI structured investigator
│       └── query_service.py       # Zero-copy DuckDB parquet query engine
├── app/streamlit_app.py           # Presentation UI
├── src/
│   ├── features.py                # Feature engineering (haversine, rolling velocity, z-scores)
│   ├── train.py                   # XGBoost training + SHAP artifact generation
│   ├── scorer.py                  # Scoring & SHAP explanation utilities
│   └── llm_analyst.py             # LLM prompt construction & offline fallback
├── tests/
│   ├── test_server.py             # Pytest integration suite (100% pass)
│   └── benchmark_server.py        # Latency & throughput benchmarking
├── scripts/
│   ├── train_model.py             # Train model CLI
│   ├── prepare_app_data.py        # Generate parquet dataset
│   ├── make_sample_data.py        # Synthetic Sparkov generator
│   ├── score_demo.py              # Smoke test
│   └── download_data.py           # Kagglehub dataset fetcher
├── Dockerfile                     # Cloud Run container (FastAPI or Streamlit)
├── deploy_cloudrun.sh             # GCP Cloud Run deployment script
└── requirements.txt
```

---

## Benchmarks & Performance

Benchmarked on local server runtime (`tests/benchmark_server.py`):

| Operation | Metric | Latency / Throughput |
|---|---|---|
| **Batch Scoring (1,000 tx)** | Throughput | **6,146 transactions / sec** |
| **Batch Scoring (1,000 tx)** | Engine Latency | **80.34 ms** |
| **Single Scoring** | p50 / p95 Latency | **85 ms / 102 ms** |
| **Investigation** | SHAP Worker + Analyst | **184.52 ms** |

---

## FastAPI Microservice Endpoints

Interactive Swagger UI documentation is available at `/docs` (and ReDoc at `/redoc`).

### 1. Real-Time Transaction Scoring
- **`POST /api/v1/score`**
  - **Payload**: JSON transaction object (amount, timestamps, coordinates, customer metadata).
  - **Returns**: `fraud_probability`, `flagged` (bool), `threshold_applied`, `risk_level` (`Low`, `Medium`, `High`), and execution time header `X-Process-Time-Ms`.

### 2. High-Throughput Batch Scoring
- **`POST /api/v1/score/batch`**
  - **Payload**: `{"transactions": [...], "threshold": 0.5}` (up to 10,000 items).
  - **Returns**: Array of score results, `total_processed`, `total_flagged`, and `latency_ms`.

### 3. Asynchronous Investigation & SHAP Drivers
- **`POST /api/v1/investigate`**
  - Dispatches CPU-bound SHAP computation to a background thread pool without blocking concurrent scoring requests.
  - Generates structured verdict: `Risk: Low/Medium/High`, `Action: Allow/Review/Block`, top 6 SHAP feature contributions, and analyst narrative.

### 4. DuckDB Analytical Engine & NL-to-SQL
- **`POST /api/v1/query`**
  - **Payload**: `{"query": "show fraud over $200 at night", "is_raw_sql": false, "limit": 100}`
  - Executes against `transactions` table using LLM translation or pattern-matched fallback.

### 5. Health & Kubernetes/Cloud Run Probes
- **`GET /health`**: Operational readiness check (`model_loaded`, `explainer_loaded`, `duckdb_ready`, `uptime_seconds`).
- **`GET /live`**: Liveness probe (`{"status": "alive"}`).
- **`GET /ready`**: Readiness probe (`{"status": "ready"}`).

---

## Quickstart

### 1. Environment Setup
```bash
# Using uv (recommended)
uv venv .venv
source .venv/bin/activate  # Or .venv\Scripts\activate on Windows
uv pip install -r requirements.txt
```

### 2. Train Model & Prepare Parquet Data
```bash
python -m scripts.train_model --data sample
python -m scripts.prepare_app_data --data data/sample_transactions.csv
```

### 3. Launch FastAPI Microservice
```bash
uvicorn server.main:app --host 0.0.0.0 --port 8080 --reload
# Access docs at http://localhost:8080/docs
```

### 4. Run Automated Tests & Benchmarks
```bash
pytest tests/test_server.py -v
python -m tests.benchmark_server
```

### 5. (Optional) Run Streamlit UI
```bash
streamlit run app/streamlit_app.py
```

---

## LLM Configuration (Optional)

The system is fully operational offline with deterministic templates. To enable generative LLM analysts:

| Provider | Environment Variables |
|---|---|
| **Google Gemini** (recommended) | `LLM_PROVIDER=gemini`, `GEMINI_API_KEY=your_key` |
| **OpenAI** | `LLM_PROVIDER=openai`, `OPENAI_API_KEY=your_key` |

---

## Deployment to Google Cloud Run

Deploy directly to Google Cloud Run with one command:

```bash
# Deploy high-throughput API microservice:
./deploy_cloudrun.sh YOUR_PROJECT_ID us-central1 api

# Deploy interactive Streamlit demo UI:
./deploy_cloudrun.sh YOUR_PROJECT_ID us-central1 streamlit
```
