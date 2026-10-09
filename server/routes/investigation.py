"""FastAPI investigation endpoint: combines ML score, async SHAP, and LLM analyst."""
import time
from fastapi import APIRouter, Depends, HTTPException, Request
from server.schemas import InvestigateRequest, InvestigateResponse, ShapDriver
from server.services.analyst_service import AnalystService
from server.services.query_service import QueryService
from server.services.scoring_service import ScoringService
from server.services.shap_service import ShapService

router = APIRouter(prefix="/api/v1", tags=["Investigation"])


def get_services(request: Request):
    return (
        request.app.state.scoring_service,
        request.app.state.shap_service,
        request.app.state.analyst_service,
        request.app.state.query_service,
    )


@router.post("/investigate", response_model=InvestigateResponse)
async def investigate_transaction(
    payload: InvestigateRequest,
    services=Depends(get_services),
):
    """
    On-demand fraud investigation.
    Computes ML score, runs non-blocking SHAP driver explanation,
    and produces structured LLM analyst report.
    """
    scoring_svc, shap_svc, analyst_svc, query_svc = services
    t0 = time.perf_counter()
    
    try:
        tx_data = payload.transaction.model_dump()
        
        # 1. Score transaction
        score, _, _, _, _ = scoring_svc.score_single(tx_data)
        
        # 2. Asynchronously compute SHAP drivers on worker thread pool
        shap_drivers_raw, _ = await shap_svc.explain_async(tx_data, top_k=payload.top_k_shap)
        
        # 3. Customer behavioral summary
        history_summary = payload.history_summary
        if not history_summary:
            history_summary = query_svc.get_customer_history_summary(payload.transaction.cc_num)
            
        # 4. Generate structured report
        risk, action, report_md = analyst_svc.generate_report(
            tx_data, history_summary, shap_drivers_raw, score
        )
        
        total_latency_ms = (time.perf_counter() - t0) * 1000.0
        
        formatted_drivers = [
            ShapDriver(feature=name, shap_value=round(sv, 4), feature_value=round(val, 4))
            for name, sv, val in shap_drivers_raw
        ]
        
        return InvestigateResponse(
            verdict_risk=risk,
            verdict_action=action,
            model_score=round(score, 4),
            report_markdown=report_md,
            top_shap_drivers=formatted_drivers,
            latency_ms=round(total_latency_ms, 2),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Investigation failed: {str(e)}")
