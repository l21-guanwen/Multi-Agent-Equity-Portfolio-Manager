"""
Portfolio router.

Provides endpoints for portfolio data access.
"""

from typing import Any, Optional

from fastapi import APIRouter, Depends, HTTPException, Query

from app.core.dependencies import (
    get_benchmark_repository,
    get_alpha_repository,
    get_data_service,
)
from app.repositories.interfaces.benchmark_repository import IBenchmarkRepository
from app.repositories.interfaces.alpha_repository import IAlphaRepository
from app.services.interfaces.data_service import IDataService
from app.schemas.portfolio_schema import (
    BenchmarkResponse,
    DataSummaryResponse,
    PortfolioSummaryResponse,
)

router = APIRouter(prefix="/portfolio", tags=["Portfolio"])


@router.get("/benchmark", response_model=BenchmarkResponse)
async def get_benchmark(
    benchmark_id: str = Query(default="SPX", description="Benchmark identifier"),
    benchmark_repo: IBenchmarkRepository = Depends(get_benchmark_repository),
) -> BenchmarkResponse:
    """
    Get benchmark data.
    
    Returns S&P 500 benchmark constituents and weights.
    """
    benchmark = await benchmark_repo.get_benchmark(benchmark_id=benchmark_id)
    
    if not benchmark:
        raise HTTPException(status_code=404, detail="Benchmark not found")
    
    # Get sector weights
    sector_weights = await benchmark_repo.get_sector_weights()
    
    # Get top constituents
    top = await benchmark_repo.get_top_constituents(n=10)
    top_constituents = [
        {
            "ticker": c.ticker,
            "name": c.security_name,
            "weight_pct": c.benchmark_weight_pct,
            "sector": c.gics_sector,
        }
        for c in top
    ]
    
    return BenchmarkResponse(
        benchmark_id=benchmark.benchmark_id,
        benchmark_name=benchmark.benchmark_name,
        as_of_date=benchmark.as_of_date.isoformat(),
        total_securities=benchmark.security_count,
        sector_weights=sector_weights,
        top_constituents=top_constituents,
    )


@router.get("/alpha/top")
async def get_top_alpha_securities(
    n: int = Query(default=25, ge=1, le=100, description="Number of securities"),
    quintile: Optional[int] = Query(default=None, ge=1, le=5, description="Filter by quintile"),
    alpha_repo: IAlphaRepository = Depends(get_alpha_repository),
) -> dict[str, Any]:
    """
    Get top securities by alpha score.
    
    Returns top N securities ranked by alpha score.
    """
    if quintile:
        scores = await alpha_repo.get_scores_by_quintile(quintile)
        scores = sorted(scores, key=lambda s: s.alpha_score, reverse=True)[:n]
    else:
        scores = await alpha_repo.get_top_scores(n=n)
    
    return {
        "count": len(scores),
        "securities": [
            {
                "ticker": s.ticker,
                "name": s.security_name,
                "alpha_score": s.alpha_score,
                "alpha_quintile": s.alpha_quintile,
                "sector": s.gics_sector,
                "signal_strength": s.signal_strength,
            }
            for s in scores
        ],
    }


@router.get("/alpha/distribution")
async def get_alpha_distribution(
    alpha_repo: IAlphaRepository = Depends(get_alpha_repository),
) -> dict[str, Any]:
    """
    Get alpha score distribution.
    
    Returns quintile distribution and sector breakdown.
    """
    alpha_model = await alpha_repo.get_alpha_model()
    
    if not alpha_model:
        raise HTTPException(status_code=404, detail="Alpha model not found")
    
    # Quintile distribution
    quintile_dist = await alpha_repo.get_quintile_distribution()
    
    # Sector breakdown
    sector_scores: dict[str, list[float]] = {}
    for score in alpha_model.scores:
        sector = score.gics_sector
        if sector not in sector_scores:
            sector_scores[sector] = []
        sector_scores[sector].append(score.alpha_score)
    
    sector_averages = {
        sector: sum(scores) / len(scores)
        for sector, scores in sector_scores.items()
    }
    
    return {
        "model_id": alpha_model.model_id,
        "as_of_date": alpha_model.as_of_date,
        "total_securities": alpha_model.security_count,
        "quintile_distribution": quintile_dist,
        "sector_average_scores": sector_averages,
        "average_alpha_score": sum(s.alpha_score for s in alpha_model.scores) / len(alpha_model.scores),
    }


@router.get("/data/summary", response_model=DataSummaryResponse)
async def get_data_summary(
    data_service: IDataService = Depends(get_data_service),
) -> DataSummaryResponse:
    """
    Get summary of available data.
    
    Returns counts and validation status for all data sources.
    """
    summary = await data_service.get_data_summary()
    
    return DataSummaryResponse(
        benchmark_count=summary.benchmark_count,
        universe_count=summary.universe_count,
        alpha_count=summary.alpha_count,
        factor_loadings_count=summary.factor_loadings_count,
        constraints_count=summary.constraints_count,
        transaction_costs_count=summary.transaction_costs_count,
        as_of_date=summary.as_of_date.isoformat() if summary.as_of_date else None,
        is_valid=summary.validation_result.is_valid if summary.validation_result else False,
        data_quality_score=summary.validation_result.data_quality_score if summary.validation_result else 0.0,
        issues=summary.validation_result.issues if summary.validation_result else [],
    )


@router.get("/data/availability")
async def check_data_availability(
    data_service: IDataService = Depends(get_data_service),
) -> dict[str, bool]:
    """
    Check availability of data sources.
    
    Returns availability status for each data source.
    """
    return await data_service.check_data_availability()

