"""Pydantic schemas for request and response validation."""
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class TransactionInput(BaseModel):
    trans_date_trans_time: str = Field(..., description="Transaction timestamp (YYYY-MM-DD HH:MM:SS)")
    cc_num: int = Field(..., description="Masked/Full credit card number")
    merchant: str = Field(..., description="Merchant name or ID")
    category: str = Field(..., description="Merchant spending category")
    amt: float = Field(..., gt=0, description="Transaction dollar amount")
    lat: float = Field(..., description="Customer home latitude")
    long: float = Field(..., description="Customer home longitude")
    merch_lat: float = Field(..., description="Merchant latitude")
    merch_long: float = Field(..., description="Merchant longitude")
    
    # Contextual fields (sensible defaults if omitted)
    first: str = Field(default="John", description="Customer first name")
    last: str = Field(default="Doe", description="Customer last name")
    gender: str = Field(default="M", description="Customer gender (M/F)")
    street: str = Field(default="", description="Customer street address")
    city: str = Field(default="", description="Customer city")
    state: str = Field(default="", description="Customer state")
    zip: str = Field(default="", description="Customer zip code")
    city_pop: float = Field(default=100000.0, description="City population")
    job: str = Field(default="Professional", description="Customer occupation")
    dob: str = Field(default="1980-01-01", description="Customer date of birth")
    trans_num: Optional[str] = Field(default=None, description="Unique transaction ID")
    unix_time: Optional[int] = Field(default=None, description="Unix timestamp")
    is_fraud: Optional[int] = Field(default=None, description="Ground truth if known")


class BatchTransactionInput(BaseModel):
    transactions: List[TransactionInput] = Field(..., min_length=1, description="List of transactions to score")
    threshold: Optional[float] = Field(default=None, ge=0.0, le=1.0, description="Optional override for fraud classification threshold")


class ScoreResult(BaseModel):
    transaction_id: Optional[str] = None
    fraud_probability: float = Field(..., description="Model predicted fraud probability [0.0 - 1.0]")
    flagged: bool = Field(..., description="True if fraud_probability >= threshold")
    threshold_applied: float = Field(..., description="Decision threshold applied")
    risk_level: str = Field(..., description="Risk tier: Low, Medium, High")


class BatchScoreResponse(BaseModel):
    total_processed: int
    total_flagged: int
    threshold: float
    results: List[ScoreResult]
    latency_ms: float


class ShapDriver(BaseModel):
    feature: str
    shap_value: float
    feature_value: float


class InvestigateRequest(BaseModel):
    transaction: TransactionInput
    history_summary: Optional[str] = Field(default=None, description="Optional pre-computed customer history summary")
    top_k_shap: int = Field(default=6, ge=1, le=20, description="Number of top SHAP drivers to return")


class InvestigateResponse(BaseModel):
    verdict_risk: str = Field(..., description="Low, Medium, or High")
    verdict_action: str = Field(..., description="Allow, Review, or Block")
    model_score: float = Field(..., description="Model predicted fraud probability")
    report_markdown: str = Field(..., description="Analyst narrative report")
    top_shap_drivers: List[ShapDriver]
    latency_ms: float


class QueryRequest(BaseModel):
    query: str = Field(..., description="Natural language question or raw SQL query")
    is_raw_sql: bool = Field(default=False, description="If true, execute as raw SQL directly")
    limit: int = Field(default=100, ge=1, le=1000, description="Maximum rows to return")


class QueryResponse(BaseModel):
    sql_executed: str
    row_count: int
    columns: List[str]
    rows: List[Dict[str, Any]]
    used_llm: bool
    latency_ms: float


class MetricsResponse(BaseModel):
    roc_auc: float
    pr_auc: float
    threshold_f1_max: float
    precision_at_threshold: float
    recall_at_threshold: float
    confusion_matrix: List[List[int]]
    train_rows: int
    test_rows: int
    fraud_rate_train: float


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    explainer_loaded: bool
    duckdb_ready: bool
    uptime_seconds: float
