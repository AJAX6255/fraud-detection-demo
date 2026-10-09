"""Configuration settings for the Fraud Detection Microservice."""
import os
from pydantic import BaseModel, Field


class Settings(BaseModel):
    app_name: str = "Fraud Detection AI Engine"
    app_version: str = "1.0.0"
    debug: bool = Field(default_factory=lambda: os.getenv("DEBUG", "false").lower() == "true")
    
    # Paths
    data_path: str = Field(default_factory=lambda: os.getenv("TRANSACTIONS_PARQUET", "data/processed/transactions.parquet"))
    model_path: str = Field(default_factory=lambda: os.getenv("MODEL_ARTIFACT", "artifacts/model.joblib"))
    explainer_path: str = Field(default_factory=lambda: os.getenv("SHAP_EXPLAINER", "artifacts/shap_explainer.joblib"))
    metrics_path: str = Field(default_factory=lambda: os.getenv("METRICS_JSON", "artifacts/metrics.json"))
    
    # LLM Settings
    llm_provider: str = Field(default_factory=lambda: os.getenv("LLM_PROVIDER", "").lower())
    gemini_api_key: str = Field(default_factory=lambda: os.getenv("GEMINI_API_KEY", ""))
    openai_api_key: str = Field(default_factory=lambda: os.getenv("OPENAI_API_KEY", ""))
    
    # Server configuration
    host: str = Field(default_factory=lambda: os.getenv("HOST", "0.0.0.0"))
    port: int = Field(default_factory=lambda: int(os.getenv("PORT", "8080")))
    max_batch_size: int = 10_000
    shap_top_k: int = 6


settings = Settings()
