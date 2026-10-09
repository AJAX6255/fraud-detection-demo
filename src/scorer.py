"""Load trained artifacts, score transactions, and produce SHAP explanations."""
import joblib
import numpy as np

from src.features import engineer


def load_bundle(path="artifacts/model.joblib"):
    return joblib.load(path)


def _prepare(df, bundle):
    fe, _ = engineer(df, amt_stats=bundle["amt_stats"], fit=False)
    for c in bundle["feature_cols"]:          # unseen categories at score time
        if c not in fe.columns:
            fe[c] = 0
    return fe[bundle["feature_cols"]]


def score(df, bundle, threshold=None):
    X = _prepare(df, bundle)
    proba = bundle["model"].predict_proba(X)[:, 1]
    thr = bundle.get("threshold", 0.5) if threshold is None else threshold
    return proba, (proba >= thr).astype(int)


def shap_explain(df, bundle, explainer="artifacts/shap_explainer.joblib", top_n=6):
    """Per-row list of (feature, shap_value, feature_value) for the top drivers."""
    X = _prepare(df, bundle)
    if isinstance(explainer, str):
        explainer = joblib.load(explainer)
    sv = explainer.shap_values(X)
    out = []
    for i in range(len(X)):
        order = np.argsort(-np.abs(sv[i]))[:top_n]
        out.append([(X.columns[j], float(sv[i][j]), float(X.iloc[i, j])) for j in order])
    return out
