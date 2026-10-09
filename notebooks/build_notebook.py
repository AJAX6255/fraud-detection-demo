"""Generate notebooks/01_training.ipynb via nbformat.

Usage: python -m notebooks.build_notebook
"""
import nbformat as nbf

nb = nbf.v4.new_notebook()

md = lambda s: nbf.v4.new_markdown_cell(s)
code = lambda s: nbf.v4.new_code_cell(s)

nb.cells = [
    md("# Fraud Detection Demo — Training Notebook\n\n"
       "**Dataset:** Sparkov credit-card transaction simulation "
       "([Kaggle: kartik2112/fraud-detection](https://www.kaggle.com/datasets/kartik2112/fraud-detection)) "
       "— ~1.9M human-readable transactions, ~0.6% fraud.\n\n"
       "**Pipeline:** load → EDA → feature engineering → XGBoost → evaluation → SHAP → LLM-style report.\n\n"
       "Run from the **repo root** so `src` imports resolve."),

    code("# If running in Colab / a fresh venv, uncomment:\n"
         "# !pip install -q pandas numpy scikit-learn xgboost shap duckdb pyarrow matplotlib joblib kagglehub"),

    code("import os, sys, json\n"
         "# Resolve repo root robustly (works from repo root AND from notebooks/)\n"
         "root = os.getcwd()\n"
         "while root != os.path.dirname(root) and not os.path.isdir(os.path.join(root, 'src')):\n"
         "    root = os.path.dirname(root)\n"
         "os.chdir(root)\n"
         "sys.path.insert(0, root)\n\n"
         "import pandas as pd\n"
         "import matplotlib.pyplot as plt\n\n"
         "# --- Load data -------------------------------------------------------------\n"
         "# Real dataset:  python -m scripts.download_data   (then concatenate train+test)\n"
         "# Fallback:      synthetic Sparkov-schema sample generated on the fly\n"
         "path = 'data/raw/all.csv'\n"
         "if not os.path.exists(path):\n"
         "    print('real dataset not found — generating synthetic sample instead')\n"
         "    from scripts.make_sample_data import generate\n"
         "    path = generate(n=30000)\n"
         "df = pd.read_csv(path)\n"
         "print(f'{len(df):,} rows | fraud rate {df.is_fraud.mean():.4%}')\n"
         "df.head(3)"),

    md("## 1. Quick EDA — why fraud detection is hard (class imbalance)"),

    code("fig, axes = plt.subplots(1, 3, figsize=(16, 4))\n\n"
         "df['amt_log'] = __import__('numpy').log1p(df['amt'])\n"
         "for label, sub in df.groupby('is_fraud'):\n"
         "    axes[0].hist(sub['amt_log'], bins=60, alpha=0.6, density=True,\n"
         "                 label='fraud' if label else 'legit')\n"
         "axes[0].set_title('Amount distribution (log)'); axes[0].legend()\n\n"
         "hr = pd.to_datetime(df['trans_date_trans_time']).dt.hour\n"
         "df.assign(hour=hr).groupby(['hour','is_fraud']).size().unstack(fill_value=0)\\\n"
         "  .pipe(lambda d: (d[1]/d.sum(axis=1))).plot(ax=axes[1], color='crimson')\n"
         "axes[1].set_title('Fraud rate by hour of day')\n\n"
         "df.groupby('category')['is_fraud'].mean().sort_values().plot.barh(ax=axes[2])\n"
         "axes[2].set_title('Fraud rate by merchant category')\n"
         "plt.tight_layout(); plt.show()"),

    md("## 2. Feature engineering\n\n"
       "Behavioural signals: amount z-score vs. the customer's own history, "
       "distance from home to merchant (haversine), night-time flag, "
       "24h transaction velocity, plus category one-hots."),

    code("from src.features import engineer, BASE_FEATURES\n"
         "fe, _ = engineer(df.sample(min(5000, len(df)), random_state=1), fit=True)\n"
         "fe[BASE_FEATURES].describe().T[['mean','min','max']]"),

    md("## 3. Train XGBoost (time-based split — last 20% of the timeline is held out)"),

    code("from src.train import train\n"
         "metrics = train(df, out_dir='artifacts')\n"
         "print(json.dumps(metrics, indent=2))"),

    md("## 4. Global explainability — SHAP feature importance"),

    code("from IPython.display import Image\n"
         "Image('artifacts/shap_summary.png')"),

    md("## 5. Score new transactions + per-flag SHAP drivers"),

    code("from src.scorer import load_bundle, score, shap_explain\n\n"
         "bundle = load_bundle('artifacts/model.joblib')\n"
         "sample = df.sample(2000, random_state=7)\n"
         "proba, pred = score(sample, bundle)\n"
         "flagged = sample[pred == 1].copy()\n"
         "flagged['fraud_probability'] = proba[pred == 1]\n"
         "print(f'{len(flagged)} flagged of {len(sample)} '\n"
         "      f'(true frauds caught: {int(flagged.is_fraud.sum())}/{int(sample.is_fraud.sum())})')\n"
         "flagged[['trans_date_trans_time','first','last','merchant','amt','fraud_probability','is_fraud']].head(10)"),

    md("## 6. LLM-style investigation report\n\n"
       "Without an API key this renders the **offline template** (same data path). "
       "Set `LLM_PROVIDER=gemini` + `GEMINI_API_KEY` (or `openai` + `OPENAI_API_KEY`) "
       "for the real narrative report."),

    code("from src.llm_analyst import generate_report\n\n"
         "single = pd.DataFrame([flagged.iloc[0]]).drop(\n"
         "    columns=['fraud_probability'], errors='ignore')\n"
         "p, _ = score(single, bundle)\n"
         "sf = shap_explain(single, bundle)[0]\n"
         "hist = df[df.cc_num == single.iloc[0].cc_num]\n"
         "summary = (f'{len(hist)} transactions; avg ${hist.amt.mean():,.2f}, '\n"
         "           f'max ${hist.amt.max():,.2f}')\n\n"
         "from IPython.display import Markdown\n"
         "Markdown(generate_report(single.iloc[0].to_dict(), summary, sf, float(p[0])))"),

    md("---\n## Next steps\n"
       "1. `python -m scripts.prepare_app_data --data <your csv>` → parquet for the app\n"
       "2. `streamlit run app/streamlit_app.py` — local check\n"
       "3. Deploy to **Cloud Run**: see README.md (Dockerfile + `deploy_cloudrun.sh` included)\n"
       "4. Wire up Gemini (`LLM_PROVIDER=gemini`) for live LLM reports & NL→SQL"),
]

nbf.write(nb, "notebooks/01_training.ipynb")
print("wrote notebooks/01_training.ipynb")
