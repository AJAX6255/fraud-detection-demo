"""LLM Analyst Service for structured fraud investigation reports."""
import json
import os
import re
from typing import Dict, List, Tuple
from server.config import settings

SYSTEM_PROMPT = (
    "You are a senior fraud analyst at a card-issuing bank. You explain ML fraud "
    "flags to human investigators. RULES: cite ONLY evidence present in the data "
    "supplied to you — never invent facts. Always start with a structured verdict line: "
    "Verdict — Risk: <Low|Medium|High> | Action: <Allow|Review|Block>."
)


class AnalystService:
    def __init__(self):
        self.provider = settings.llm_provider

    def build_report_prompt(self, tx: Dict, history_summary: str, shap_features: List[Tuple[str, float, float]], score: float) -> str:
        shap_lines = "\n".join(
            f"  - {name}: value={val:.4g}, shap={sv:+.3f}" for name, sv, val in shap_features
        )
        return f"""A machine-learning model flagged this transaction with fraud probability {score:.1%}.

TRANSACTION:
{json.dumps(tx, indent=2, default=str)}

CUSTOMER HISTORY SUMMARY:
{history_summary}

TOP MODEL DRIVERS (SHAP — positive pushes toward fraud, negative toward legitimate):
{shap_lines}

Write an investigation report with:
1. **Verdict** — Risk: Low/Medium/High | Action: Allow/Review/Block (first line).
2. **Key evidence** — bullet points citing only the data above.
3. **Behaviour comparison** — how this transaction differs from the customer's normal pattern.
4. **Recommended next step** — one concrete action for the investigator.
Keep it under 250 words."""

    def template_report(self, tx: Dict, history_summary: str, shap_features: List[Tuple[str, float, float]], score: float) -> Tuple[str, str, str]:
        """Deterministic offline fallback report."""
        drivers = "\n".join(f"- `{n}` = {v:.4g} (contribution {s:+.3f})" for n, s, v in shap_features[:5])
        risk = "High" if score >= 0.8 else "Medium" if score >= 0.5 else "Low"
        action = {"High": "Block", "Medium": "Review", "Low": "Allow"}[risk]
        
        report = f"""### Fraud Investigation Report *(offline template — no LLM configured)*

**Verdict — Risk: {risk} | Action: {action}** (model score {score:.1%})

**Transaction:** ${float(tx.get('amt', 0)):,.2f} at `{tx.get('merchant')}` ({tx.get('category')}), {tx.get('city')}, {tx.get('state')} — {tx.get('trans_date_trans_time')}

**Top model drivers (SHAP):**
{drivers}

**Customer context:** {history_summary}

*Configured LLM provider: {self.provider or 'offline template'}*"""
        return risk, action, report

    def _call_llm(self, prompt: str) -> str:
        provider = self.provider or os.environ.get("LLM_PROVIDER", "").lower()
        if provider == "gemini":
            from google import genai
            client = genai.Client()
            resp = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt,
                config={"system_instruction": SYSTEM_PROMPT}
            )
            return resp.text
        if provider == "openai":
            from openai import OpenAI
            r = OpenAI().chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt}
                ]
            )
            return r.choices[0].message.content
        raise RuntimeError("No LLM provider configured")

    def generate_report(self, tx: Dict, history_summary: str, shap_features: List[Tuple[str, float, float]], score: float) -> Tuple[str, str, str]:
        """Generates narrative report and extracts verdict (risk, action, markdown)."""
        try:
            prompt = self.build_report_prompt(tx, history_summary, shap_features, score)
            narrative = self._call_llm(prompt)
            
            # Parse verdict from response
            risk = "High" if score >= 0.8 else "Medium" if score >= 0.5 else "Low"
            action = {"High": "Block", "Medium": "Review", "Low": "Allow"}[risk]
            
            m_risk = re.search(r"Risk:\s*(Low|Medium|High)", narrative, re.IGNORECASE)
            if m_risk:
                risk = m_risk.group(1).capitalize()
            m_act = re.search(r"Action:\s*(Allow|Review|Block)", narrative, re.IGNORECASE)
            if m_act:
                action = m_act.group(1).capitalize()
                
            return risk, action, narrative
        except Exception:
            return self.template_report(tx, history_summary, shap_features, score)
