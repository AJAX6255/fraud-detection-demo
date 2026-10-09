"""DuckDB Analytical Service for high-speed queries and NL->SQL translation."""
import os
import re
import time
from typing import Any, Dict, List, Optional, Tuple
import duckdb
from server.config import settings

SQL_SCHEMA = (
    "Table `transactions` columns: trans_date_trans_time (TIMESTAMP), cc_num (VARCHAR), "
    "merchant (VARCHAR), category (VARCHAR), amt (DOUBLE), first, last, gender, city, "
    "state, lat, long, city_pop, job, dob, merch_lat, merch_long, is_fraud (INTEGER 0/1)."
)


class QueryService:
    def __init__(self, data_path: str = "data/processed/transactions.parquet"):
        self.data_path = data_path
        self.con = duckdb.connect(database=":memory:", read_only=False)
        self.init_data()

    def init_data(self):
        """Registers the parquet file as the `transactions` table/view."""
        if os.path.exists(self.data_path):
            self.con.execute(f"CREATE OR REPLACE VIEW transactions AS SELECT * FROM read_parquet('{self.data_path}')")
        else:
            # Fallback table if parquet is not ready
            self.con.execute("CREATE OR REPLACE TABLE transactions (amt DOUBLE, is_fraud INTEGER)")

    def fallback_sql(self, question: str, limit: int = 100) -> str:
        """Offline pattern-matched NL->SQL for common fraud analytics questions."""
        q = question.lower()
        clauses = ["1=1"]
        m = re.search(r"(?:over|above|more than)\s*\$?\s*([\d,]+(?:\.\d+)?)", q)
        if m:
            clauses.append(f"amt > {m.group(1).replace(',', '')}")
        m = re.search(r"category\s+([\w_]+)", q)
        if m:
            clauses.append(f"category = '{m.group(1)}'")
        m = re.search(r"merchant\s+['\"]?([\w\s&\.]+?)['\"]?$", q)
        if m:
            clauses.append(f"lower(merchant) LIKE '%{m.group(1).lower().strip()}%'")
        if "night" in q:
            clauses.append("(hour(cast(trans_date_trans_time as timestamp)) >= 22 OR hour(cast(trans_date_trans_time as timestamp)) <= 4)")
        if "fraud" in q:
            clauses.append("is_fraud = 1")
        return f"SELECT * FROM transactions WHERE {' AND '.join(clauses)} LIMIT {limit}"

    def nl_to_sql_llm(self, question: str, limit: int = 100) -> Optional[str]:
        """Translates natural language to SQL using configured LLM provider."""
        provider = settings.llm_provider or os.environ.get("LLM_PROVIDER", "").lower()
        if not provider:
            return None
            
        prompt = f"""{SQL_SCHEMA}
Write a single DuckDB SELECT query (LIMIT {limit}) answering the question.
Output ONLY the SQL query, no markdown formatting or commentary.
Question: {question}"""
        try:
            if provider == "gemini":
                from google import genai
                client = genai.Client()
                resp = client.models.generate_content(
                    model="gemini-2.5-flash",
                    contents=prompt
                )
                sql = resp.text.strip().strip("`").strip()
            elif provider == "openai":
                from openai import OpenAI
                r = OpenAI().chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "user", "content": prompt}]
                )
                sql = r.choices[0].message.content.strip().strip("`").strip()
            else:
                return None

            if sql.lower().startswith("select"):
                return sql
            return None
        except Exception:
            return None

    def execute_query(self, query: str, is_raw_sql: bool = False, limit: int = 100) -> Tuple[str, List[str], List[Dict[str, Any]], bool, float]:
        """Executes a query and returns (sql, columns, rows, used_llm, latency_ms)."""
        t0 = time.perf_counter()
        used_llm = False
        
        if is_raw_sql or query.strip().lower().startswith("select"):
            sql = query.strip()
        else:
            llm_sql = self.nl_to_sql_llm(query, limit)
            if llm_sql:
                sql = llm_sql
                used_llm = True
            else:
                sql = self.fallback_sql(query, limit)

        # Enforce safety check: read-only SELECT queries
        if not sql.lower().strip().startswith("select"):
            raise ValueError("Only SELECT queries are permitted.")

        df = self.con.execute(sql).df()
        columns = df.columns.tolist()
        rows = df.to_dict(orient="records")
        latency_ms = (time.perf_counter() - t0) * 1000.0
        return sql, columns, rows, used_llm, latency_ms

    def get_customer_history_summary(self, cc_num: int) -> str:
        """High-speed aggregation of customer behavioral context via DuckDB."""
        try:
            res = self.con.execute(
                f"""SELECT 
                    count(*) as tx_count, 
                    avg(amt) as avg_amt, 
                    max(amt) as max_amt,
                    mode(category) as top_cat,
                    first(city) as city,
                    first(state) as state
                FROM transactions 
                WHERE cc_num = {cc_num}"""
            ).fetchone()
            if not res or res[0] == 0:
                return "No prior transaction history found for this customer."
            tx_count, avg_amt, max_amt, top_cat, city, state = res
            return (
                f"{tx_count} historical transactions; avg ${avg_amt or 0:,.2f}, "
                f"max ${max_amt or 0:,.2f}; most frequent category: {top_cat or 'N/A'}; "
                f"registered location: {city or 'Unknown'}, {state or 'Unknown'}."
            )
        except Exception as e:
            return f"Historical context unavailable: {str(e)}"
