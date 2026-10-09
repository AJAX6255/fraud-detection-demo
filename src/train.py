"""Train the XGBoost fraud classifier and persist model + metrics + SHAP artifacts."""
import json
import os

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import xgboost as xgb
from sklearn.metrics import (average_precision_score, confusion_matrix,
                             precision_recall_curve, roc_auc_score)

from src.features import BASE_FEATURES, engineer


def train(df, out_dir="artifacts", test_frac=0.2):
    """Time-based split (last `test_frac` of the timeline = held-out test)."""
    os.makedirs(out_dir, exist_ok=True)
    df = df.sort_values("trans_date_trans_time").reset_index(drop=True)

    fe, amt_stats = engineer(df, fit=True)
    cat_cols = [c for c in fe.columns if c.startswith("cat_")]
    feats = BASE_FEATURES + cat_cols

    cut = fe["trans_date_trans_time"].quantile(1 - test_frac)
    tr = fe[fe["trans_date_trans_time"] < cut]
    te = fe[fe["trans_date_trans_time"] >= cut]
    Xtr, ytr = tr[feats], tr["is_fraud"].astype(int)
    Xte, yte = te[feats], te["is_fraud"].astype(int)

    pos = max(int(ytr.sum()), 1)
    model = xgb.XGBClassifier(
        n_estimators=300, max_depth=6, learning_rate=0.08,
        subsample=0.8, colsample_bytree=0.8,
        scale_pos_weight=(len(ytr) - pos) / pos,   # class imbalance
        eval_metric="aucpr", tree_method="hist", random_state=42,
    )
    model.fit(Xtr, ytr, eval_set=[(Xte, yte)], verbose=False)

    proba = model.predict_proba(Xte)[:, 1]
    prec, rec, thr = precision_recall_curve(yte, proba)
    f1 = 2 * prec * rec / np.clip(prec + rec, 1e-9, None)
    best_i = int(np.nanargmax(f1[:-1])) if len(thr) else 0
    threshold = float(thr[best_i]) if len(thr) else 0.5
    cm = confusion_matrix(yte, (proba >= threshold).astype(int)).tolist()

    metrics = {
        "roc_auc": float(roc_auc_score(yte, proba)),
        "pr_auc": float(average_precision_score(yte, proba)),
        "threshold_f1_max": threshold,
        "precision_at_threshold": float(prec[best_i]),
        "recall_at_threshold": float(rec[best_i]),
        "confusion_matrix": cm,
        "train_rows": int(len(tr)), "test_rows": int(len(te)),
        "fraud_rate_train": float(ytr.mean()),
    }
    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    joblib.dump({"model": model, "feature_cols": feats,
                 "amt_stats": amt_stats, "threshold": threshold},
                os.path.join(out_dir, "model.joblib"))

    sample = Xte.sample(min(2000, len(Xte)), random_state=42)
    explainer = shap.TreeExplainer(model)
    sv = explainer.shap_values(sample)
    plt.figure(figsize=(8, 6))
    shap.summary_plot(sv, sample, show=False, max_display=15)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "shap_summary.png"), dpi=120)
    plt.close()
    joblib.dump(explainer, os.path.join(out_dir, "shap_explainer.joblib"))
    return metrics
