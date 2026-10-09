"""FastAPI Entry Point for the High-Throughput Fraud Detection Engine."""
from contextlib import asynccontextmanager
import time
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from server.config import settings
from server.routes import health, investigation, metrics, query, scoring
from server.services.analyst_service import AnalystService
from server.services.query_service import QueryService
from server.services.scoring_service import ScoringService
from server.services.shap_service import ShapService


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initializes and caches model bundles, SHAP workers, and analytical database."""
    app.state.start_time = time.perf_counter()
    print("==> Initializing Fraud Detection Engine services...")
    
    # 1. Scoring service
    app.state.scoring_service = ScoringService(model_path=settings.model_path)
    
    # 2. SHAP explainer service
    app.state.shap_service = ShapService(
        explainer_path=settings.explainer_path,
        scoring_service=app.state.scoring_service,
    )
    
    # 3. LLM analyst service
    app.state.analyst_service = AnalystService()
    
    # 4. DuckDB analytical query service
    app.state.query_service = QueryService(data_path=settings.data_path)
    
    print("==> Engine initialized successfully. Ready for inference.")
    yield
    print("==> Shutting down Fraud Detection Engine...")


app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="High-throughput ML scoring, asynchronous SHAP interpretability, and DuckDB analytics microservice.",
    lifespan=lifespan,
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    """Adds precision request latency header to all HTTP responses."""
    start_time = time.perf_counter()
    response = await call_next(request)
    process_time = (time.perf_counter() - start_time) * 1000.0
    response.headers["X-Process-Time-Ms"] = f"{process_time:.2f}"
    return response


# Include Routers
app.include_router(health.router)
app.include_router(scoring.router)
app.include_router(investigation.router)
app.include_router(query.router)
app.include_router(metrics.router)


@app.get("/")
async def root():
    return {
        "service": settings.app_name,
        "version": settings.app_version,
        "docs_url": "/docs",
        "health_url": "/health",
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server.main:app", host=settings.host, port=settings.port, reload=settings.debug)
