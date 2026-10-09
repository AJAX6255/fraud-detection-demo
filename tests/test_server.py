"""Unit and Integration Tests for the Fraud Detection FastAPI Microservice."""
import json
import pytest
from starlette.testclient import TestClient

from server.main import app

SAMPLE_TX = {
    "trans_date_trans_time": "2020-04-13 02:39:00",
    "cc_num": 4200000000000001,
    "merchant": "fraud_Jones Inc",
    "category": "entertainment",
    "amt": 421.12,
    "first": "John",
    "last": "Doe",
    "gender": "M",
    "street": "123 Main St",
    "city": "Miami",
    "state": "FL",
    "zip": "33101",
    "lat": 25.76,
    "long": -80.19,
    "city_pop": 440000.0,
    "job": "Developer",
    "dob": "1985-05-12",
    "trans_num": "tx_test_12345",
    "unix_time": 1586745540,
    "merch_lat": 30.50,
    "merch_long": -75.20,
    "is_fraud": 1,
}

SAMPLE_LEGIT_TX = {
    "trans_date_trans_time": "2020-04-13 14:15:00",
    "cc_num": 4200000000000001,
    "merchant": "Publix Supermarket",
    "category": "grocery_pos",
    "amt": 42.50,
    "first": "John",
    "last": "Doe",
    "gender": "M",
    "street": "123 Main St",
    "city": "Miami",
    "state": "FL",
    "zip": "33101",
    "lat": 25.76,
    "long": -80.19,
    "city_pop": 440000.0,
    "job": "Developer",
    "dob": "1985-05-12",
    "trans_num": "tx_legit_54321",
    "unix_time": 1586787300,
    "merch_lat": 25.77,
    "merch_long": -80.20,
    "is_fraud": 0,
}


def test_health_endpoints():
    with TestClient(app) as client:
        r = client.get("/health")
        assert r.status_code == 200
        data = r.json()
        assert data["status"] == "healthy"
        assert data["model_loaded"] is True
        assert data["explainer_loaded"] is True
        assert data["duckdb_ready"] is True

        r_live = client.get("/live")
        assert r_live.status_code == 200
        assert r_live.json() == {"status": "alive"}

        r_ready = client.get("/ready")
        assert r_ready.status_code == 200
        assert r_ready.json() == {"status": "ready"}


def test_metrics_endpoint():
    with TestClient(app) as client:
        r = client.get("/api/v1/metrics")
        assert r.status_code == 200
        data = r.json()
        assert "roc_auc" in data
        assert "pr_auc" in data
        assert "threshold_f1_max" in data


def test_single_scoring():
    with TestClient(app) as client:
        # Score suspicious transaction
        r = client.post("/api/v1/score", json=SAMPLE_TX)
        assert r.status_code == 200
        data = r.json()
        assert data["transaction_id"] == "tx_test_12345"
        assert 0.0 <= data["fraud_probability"] <= 1.0
        assert isinstance(data["flagged"], bool)
        assert data["risk_level"] in ["Low", "Medium", "High"]
        assert "X-Process-Time-Ms" in r.headers

        # Score legit transaction
        r_legit = client.post("/api/v1/score", json=SAMPLE_LEGIT_TX)
        assert r_legit.status_code == 200
        legit_data = r_legit.json()
        assert legit_data["fraud_probability"] < data["fraud_probability"]


def test_batch_scoring():
    with TestClient(app) as client:
        batch = [SAMPLE_TX, SAMPLE_LEGIT_TX] * 10  # 20 transactions
        r = client.post("/api/v1/score/batch", json={"transactions": batch})
        assert r.status_code == 200
        data = r.json()
        assert data["total_processed"] == 20
        assert len(data["results"]) == 20
        assert data["latency_ms"] > 0


def test_investigation_endpoint():
    with TestClient(app) as client:
        r = client.post(
            "/api/v1/investigate",
            json={
                "transaction": SAMPLE_TX,
                "top_k_shap": 5,
            },
        )
        assert r.status_code == 200
        data = r.json()
        assert data["verdict_risk"] in ["Low", "Medium", "High"]
        assert data["verdict_action"] in ["Allow", "Review", "Block"]
        assert len(data["top_shap_drivers"]) == 5
        assert len(data["report_markdown"]) > 0


def test_analytical_query_endpoint():
    with TestClient(app) as client:
        # 1. Direct SQL
        r_sql = client.post(
            "/api/v1/query",
            json={
                "query": "SELECT count(*) as total, sum(is_fraud) as fraud_count FROM transactions",
                "is_raw_sql": True,
            },
        )
        assert r_sql.status_code == 200
        data_sql = r_sql.json()
        assert data_sql["row_count"] == 1
        assert "total" in data_sql["columns"]

        # 2. Natural language pattern query
        r_nl = client.post(
            "/api/v1/query",
            json={
                "query": "show fraud over $200 at night",
                "is_raw_sql": False,
                "limit": 10,
            },
        )
        assert r_nl.status_code == 200
        data_nl = r_nl.json()
        assert data_nl["row_count"] >= 0
        assert "sql_executed" in data_nl
