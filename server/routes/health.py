"""FastAPI health check and orchestration probes."""
import time
from fastapi import APIRouter, Request, status
from fastapi.responses import JSONResponse
from server.schemas import HealthResponse

router = APIRouter(tags=["Health"])


@router.get("/health", response_model=HealthResponse)
async def health_check(request: Request):
    """Full operational health check of model, explainer, and analytical database."""
    uptime = time.perf_counter() - request.app.state.start_time
    model_loaded = hasattr(request.app.state, "scoring_service") and request.app.state.scoring_service.bundle is not None
    explainer_loaded = hasattr(request.app.state, "shap_service") and request.app.state.shap_service.explainer is not None
    duckdb_ready = hasattr(request.app.state, "query_service") and request.app.state.query_service.con is not None
    
    is_healthy = model_loaded and explainer_loaded and duckdb_ready
    return HealthResponse(
        status="healthy" if is_healthy else "degraded",
        model_loaded=model_loaded,
        explainer_loaded=explainer_loaded,
        duckdb_ready=duckdb_ready,
        uptime_seconds=round(uptime, 2),
    )


@router.get("/live")
async def liveness_probe():
    """Kubernetes / Cloud Run liveness probe."""
    return {"status": "alive"}


@router.get("/ready")
async def readiness_probe(request: Request):
    """Kubernetes / Cloud Run readiness probe."""
    if hasattr(request.app.state, "scoring_service") and request.app.state.scoring_service.bundle is not None:
        return {"status": "ready"}
    return JSONResponse(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, content={"status": "initializing"})
