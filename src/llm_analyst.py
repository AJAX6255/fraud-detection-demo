"""LLM 'fraud analyst' — investigation reports and NL->SQL, with an offline fallback.

Provider selection via environment:
  LLM_PROVIDER=gemini  + GEMINI_API_KEY   (recommended on Google Cloud)
  LLM_PROVIDER=openai  + OPENAI_API_KEY
  anything else / unset -> deterministic template output (no API calls),
                           which keeps the demo fully functional offline.
"""
import json
import os

SYSTEM_PROMPT = (
    "You are a senior fraud analyst at a card-issuing bank. You explain ML fraud "
    "flags to human investigators. RULES: cite ONLY evidence present in the data "
    "supplied to you — never invent facts. Always end with a structured verdict: "
    "Risk (Low/Medium/High) and Action (Allow/Review/Block)."
)

SQL_SCHEMA = (
    "Table `transactions` columns: trans_date_trans_time (TIMESTAMP), cc_num (VARCHAR), "
    "merchant (VARCHAR), category (VARCHAR), amt (DOUBLE), first, last, gender, city, "
    "state, lat, long, city_pop, job, dob, merch_lat, merch_long, is_fraud (INTEGER 0/1)."
)


def build_report_prompt(tx: dict, history_summary: str, shap_features: list, score: float) -> str:
    shap_lines = "\n".join(
        f"  - {name}: value={val:.4g}, shap={sv:+.3f}" for name, sv, val in shap_features)
    return f"""A machine-learning model flagged this transaction with fraud probability {score:.1%}.

TRANSACTION:
{json.dumps(tx, indent=2, default=str)}

CUSTOMER HISTORY SUMMARY:
{history_summary}

TOP MODEL DRIVERS (SHAP — positive pushes toward fraud, negative toward legitimate):
{shap_lines}

Write an investigation report with:
1. **Verdict** — Risk: Low/Medium/High, Action: Allow/Review/Block (one line).
2. **Key evidence** — bullet points citing only the data above.
3. **Behaviour comparison** — how this transaction differs from the customer's normal pattern.
4. **Recommended next step** — one concrete action for the investigator.
Keep it under 250 words."""


def template_report(tx, history_summary, shap_features, score):
    """Deterministic offline report — proves the pipeline works without an LLM."""
    drivers = "\n".join(f"- `{n}` = {v:.4g} (contribution {s:+.3f})" for n, s, v in shap_features[:5])
    risk = "High" if score >= 0.8 else "Medium" if score >= 0.5 else "Low"
    action = {"High": "Block", "Medium": "Review", "Low": "Allow"}[risk]
    return f"""### Fraud Investigation Report *(offline template — no LLM configured)*

**Verdict — Risk: {risk} | Action: {action}** (model score {score:.1%})

**Transaction:** ${float(tx.get('amt', 0)):,.2f} at `{tx.get('merchant')}` \
({tx.get('category')}), {tx.get('city')}, {tx.get('state')} — {tx.get('trans_date_trans_time')}

**Top model drivers (SHAP):**
{drivers}

**Customer context:** {history_summary}

*Set `LLM_PROVIDER=gemini` (or `openai`) with an API key to generate the full narrative report.*"""


def _llm_text(prompt: str) -> str:
    provider = os.environ.get("LLM_PROVIDER", "").lower()
    if provider == "gemini":
        from google import genai
        client = genai.Client()
        return client.models.generate_content(
            model="gemini-2.5-flash", contents=prompt,
            config={"system_instruction": SYSTEM_PROMPT}).text
    if provider == "openai":
        from openai import OpenAI
        r = OpenAI().chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "system", "content": SYSTEM_PROMPT},
                      {"role": "user", "content": prompt}])
        return r.choices[0].message.content
    raise RuntimeError("no LLM provider configured")


def generate_report(tx, history_summary, shap_features, score) -> str:
    try:
        return _llm_text(build_report_prompt(tx, history_summary, shap_features, score))
    except Exception:
        return template_report(tx, history_summary, shap_features, score)


def nl_to_sql(question: str):
    """LLM NL->SQL over the transactions table. Returns None when no provider is set."""
    prompt = f"""{SQL_SCHEMA}
Write a single DuckDB SELECT query (LIMIT 500) answering the question.
Output ONLY the SQL, no markdown fences, no explanation.
Question: {question}"""
    try:
        sql = _llm_text(prompt).strip().strip("`")
        return sql if sql.lower().startswith("select") else None
    except Exception:
        return None
