"""Smoke test: score sample rows and generate one (offline-template) report."""
import pandas as pd

from src.llm_analyst import generate_report
from src.scorer import load_bundle, score, shap_explain

df = pd.read_csv("data/sample_transactions.csv")
bundle = load_bundle("artifacts/model.joblib")

sample = df.sample(2000, random_state=7)
proba, pred = score(sample, bundle)
print(f"scored {len(sample):,} rows -> {int(pred.sum())} flagged "
      f"(true frauds in sample: {int(sample['is_fraud'].sum())})")

flagged = sample[pred == 1]
if len(flagged):
    single = pd.DataFrame([flagged.iloc[0]])
    p, _ = score(single, bundle)
    sf = shap_explain(single, bundle)[0]
    hist = df[df["cc_num"] == single.iloc[0]["cc_num"]]
    summary = (f"{len(hist)} transactions; avg ${hist['amt'].mean():,.2f}, "
               f"max ${hist['amt'].max():,.2f}")
    print("\n" + generate_report(single.iloc[0].to_dict(), summary, sf, float(p[0])))
print("\nSMOKE TEST OK")
