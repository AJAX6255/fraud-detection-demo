"""Benchmark latency and throughput for the FastAPI Scoring Microservice."""
import time
from starlette.testclient import TestClient
from server.main import app
from tests.test_server import SAMPLE_TX, SAMPLE_LEGIT_TX

def run_benchmarks():
    with TestClient(app) as client:
        print("\n--- Running Latency & Throughput Benchmarks ---")
        
        # 1. Warm-up
        for _ in range(5):
            client.post("/api/v1/score", json=SAMPLE_TX)
            
        # 2. Single transaction latency benchmark (50 iterations)
        latencies = []
        for _ in range(50):
            t0 = time.perf_counter()
            r = client.post("/api/v1/score", json=SAMPLE_TX)
            latencies.append((time.perf_counter() - t0) * 1000.0)
            assert r.status_code == 200
            
        p50 = sorted(latencies)[len(latencies) // 2]
        p95 = sorted(latencies)[int(len(latencies) * 0.95)]
        avg = sum(latencies) / len(latencies)
        print(f"Single Score Latency (n=50): Avg = {avg:.2f}ms | p50 = {p50:.2f}ms | p95 = {p95:.2f}ms")
        
        # 3. Vectorized batch scoring throughput (1,000 transactions)
        batch_1k = [SAMPLE_TX, SAMPLE_LEGIT_TX] * 500
        t0 = time.perf_counter()
        r_batch = client.post("/api/v1/score/batch", json={"transactions": batch_1k})
        total_time = (time.perf_counter() - t0) * 1000.0
        assert r_batch.status_code == 200
        data = r_batch.json()
        print(f"Batch Scoring (1,000 transactions): Total Time = {total_time:.2f}ms (Engine Latency = {data['latency_ms']:.2f}ms)")
        print(f"Throughput: {1000.0 / (total_time / 1000.0):,.0f} tx/sec")
        
        # 4. Asynchronous SHAP + Report Latency
        t0 = time.perf_counter()
        r_inv = client.post("/api/v1/investigate", json={"transaction": SAMPLE_TX, "top_k_shap": 5})
        inv_time = (time.perf_counter() - t0) * 1000.0
        assert r_inv.status_code == 200
        print(f"Investigation (SHAP + Analyst): Total Time = {inv_time:.2f}ms")
        print("-------------------------------------------------\n")

if __name__ == "__main__":
    run_benchmarks()
