"""FastAPI DuckDB analytical query and NL-to-SQL endpoint."""
from fastapi import APIRouter, Depends, HTTPException, Request
from server.schemas import QueryRequest, QueryResponse
from server.services.query_service import QueryService

router = APIRouter(prefix="/api/v1", tags=["Analytics"])


def get_query_service(request: Request) -> QueryService:
    return request.app.state.query_service


@router.post("/query", response_model=QueryResponse)
async def query_transactions(
    payload: QueryRequest,
    service: QueryService = Depends(get_query_service),
):
    """
    Executes analytical queries against the DuckDB transaction repository.
    Supports either plain natural language questions or direct SQL SELECT queries.
    """
    try:
        sql, cols, rows, used_llm, latency = service.execute_query(
            payload.query,
            is_raw_sql=payload.is_raw_sql,
            limit=payload.limit,
        )
        return QueryResponse(
            sql_executed=sql,
            row_count=len(rows),
            columns=cols,
            rows=rows,
            used_llm=used_llm,
            latency_ms=round(latency, 2),
        )
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Query execution error: {str(e)}")
