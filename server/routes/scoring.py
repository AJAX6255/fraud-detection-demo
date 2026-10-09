"""FastAPI scoring endpoints for real-time and batch fraud inference."""
from fastapi import APIRouter, Depends, HTTPException, Request
from server.schemas import (
    BatchScoreResponse,
    BatchTransactionInput,
    ScoreResult,
    TransactionInput,
)
from server.services.scoring_service import ScoringService

router = APIRouter(prefix="/api/v1", tags=["Scoring"])


def get_scoring_service(request: Request) -> ScoringService:
    return request.app.state.scoring_service


@router.post("/score", response_model=ScoreResult)
async def score_transaction(
    payload: TransactionInput,
    threshold: float = None,
    service: ScoringService = Depends(get_scoring_service),
):
    """Real-time scoring for a single transaction (low-latency)."""
    try:
        tx_data = payload.model_dump()
        prob, flag, thr, risk, latency = service.score_single(tx_data, threshold=threshold)
        return ScoreResult(
            transaction_id=payload.trans_num,
            fraud_probability=round(prob, 4),
            flagged=flag,
            threshold_applied=round(thr, 4),
            risk_level=risk,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")


@router.post("/score/batch", response_model=BatchScoreResponse)
async def score_batch_transactions(
    payload: BatchTransactionInput,
    service: ScoringService = Depends(get_scoring_service),
):
    """High-throughput vectorized scoring for batches of transactions."""
    try:
        tx_list = [tx.model_dump() for tx in payload.transactions]
        scored_rows, thr, latency = service.score_batch(tx_list, threshold=payload.threshold)
        
        results = []
        flagged_count = 0
        for tx, (p, flag, risk) in zip(payload.transactions, scored_rows):
            if flag:
                flagged_count += 1
            results.append(
                ScoreResult(
                    transaction_id=tx.trans_num,
                    fraud_probability=round(p, 4),
                    flagged=flag,
                    threshold_applied=round(thr, 4),
                    risk_level=risk,
                )
            )
            
        return BatchScoreResponse(
            total_processed=len(results),
            total_flagged=flagged_count,
            threshold=round(thr, 4),
            results=results,
            latency_ms=round(latency, 2),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch inference error: {str(e)}")
