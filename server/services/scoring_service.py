"""High-performance Scoring Service with cached model in memory."""
import time
from typing import Dict, List, Optional, Tuple
import joblib
import numpy as np
import pandas as pd

from src.features import engineer


class ScoringService:
    def __init__(self, model_path: str = "artifacts/model.joblib"):
        self.model_path = model_path
        self.bundle: Optional[Dict] = None
        self.load_model()

    def load_model(self):
        """Loads and pre-warms the XGBoost bundle in memory."""
        self.bundle = joblib.load(self.model_path)
        self.feature_cols = self.bundle["feature_cols"]
        self.amt_stats = self.bundle["amt_stats"]
        self.default_threshold = float(self.bundle.get("threshold", 0.5))

    def _prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Runs fast feature extraction and aligns columns."""
        fe, _ = engineer(df, amt_stats=self.amt_stats, fit=False)
        for c in self.feature_cols:
            if c not in fe.columns:
                fe[c] = 0
        return fe[self.feature_cols]

    def score_df(self, df: pd.DataFrame, threshold: Optional[float] = None) -> Tuple[np.ndarray, np.ndarray, float]:
        """Vectorized scoring for pandas DataFrames."""
        if self.bundle is None:
            self.load_model()
        thr = self.default_threshold if threshold is None else threshold
        X = self._prepare_features(df)
        proba = self.bundle["model"].predict_proba(X)[:, 1]
        flags = (proba >= thr).astype(bool)
        return proba, flags, thr

    def score_single(self, tx_data: Dict, threshold: Optional[float] = None) -> Tuple[float, bool, float, str, float]:
        """Scores a single transaction dictionary with execution timer."""
        t0 = time.perf_counter()
        df = pd.DataFrame([tx_data])
        proba, flags, thr = self.score_df(df, threshold=threshold)
        p = float(proba[0])
        flag = bool(flags[0])
        risk = "High" if p >= 0.8 else "Medium" if p >= thr else "Low"
        latency_ms = (time.perf_counter() - t0) * 1000.0
        return p, flag, thr, risk, latency_ms

    def score_batch(self, tx_list: List[Dict], threshold: Optional[float] = None) -> Tuple[List[Tuple[float, bool, str]], float, float]:
        """Vectorized batch scoring for hundreds or thousands of transactions."""
        t0 = time.perf_counter()
        df = pd.DataFrame(tx_list)
        proba, flags, thr = self.score_df(df, threshold=threshold)
        
        results = []
        for p, flag in zip(proba, flags):
            p_val = float(p)
            risk = "High" if p_val >= 0.8 else "Medium" if p_val >= thr else "Low"
            results.append((p_val, bool(flag), risk))
            
        latency_ms = (time.perf_counter() - t0) * 1000.0
        return results, thr, latency_ms
