"""SHAP Worker Service for on-demand explainability without blocking the event loop."""
import asyncio
from concurrent.futures import ThreadPoolExecutor
import time
from typing import Dict, List, Tuple
import joblib
import numpy as np
import pandas as pd

from server.services.scoring_service import ScoringService


class ShapService:
    def __init__(self, explainer_path: str = "artifacts/shap_explainer.joblib", scoring_service: ScoringService = None):
        self.explainer_path = explainer_path
        self.scoring_service = scoring_service
        self.explainer = None
        self._executor = ThreadPoolExecutor(max_workers=4)
        self.load_explainer()

    def load_explainer(self):
        """Loads the pre-computed TreeExplainer artifact into memory."""
        self.explainer = joblib.load(self.explainer_path)

    def _sync_explain(self, df: pd.DataFrame, top_k: int = 6) -> List[List[Tuple[str, float, float]]]:
        """Synchronous CPU-bound SHAP computation."""
        X = self.scoring_service._prepare_features(df)
        sv = self.explainer.shap_values(X)
        
        results = []
        for i in range(len(X)):
            order = np.argsort(-np.abs(sv[i]))[:top_k]
            row_drivers = [
                (str(X.columns[j]), float(sv[i][j]), float(X.iloc[i, j]))
                for j in order
            ]
            results.append(row_drivers)
        return results

    async def explain_async(self, tx_data: Dict, top_k: int = 6) -> Tuple[List[Tuple[str, float, float]], float]:
        """Asynchronously dispatches SHAP explanation to thread pool."""
        t0 = time.perf_counter()
        df = pd.DataFrame([tx_data])
        
        loop = asyncio.get_running_loop()
        drivers_list = await loop.run_in_executor(
            self._executor, self._sync_explain, df, top_k
        )
        latency_ms = (time.perf_counter() - t0) * 1000.0
        return drivers_list[0], latency_ms
