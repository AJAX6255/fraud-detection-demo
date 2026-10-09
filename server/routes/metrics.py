"""FastAPI model metrics endpoint."""
import json
import os
from fastapi import APIRouter, HTTPException
from server.config import settings
from server.schemas import MetricsResponse

router = APIRouter(prefix="/api/v1", tags=["Model Operations"])


@router.get("/metrics", response_model=MetricsResponse)
async def get_model_metrics():
    """Returns stored offline evaluation metrics and performance benchmarks."""
    if not os.path.exists(settings.metrics_path):
        raise HTTPException(status_code=404, detail="Metrics artifact not found. Please train model.")
        
    try:
        with open(settings.metrics_path, "r") as f:
            data = json.load(f)
        return MetricsResponse(**data)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load metrics: {str(e)}")
