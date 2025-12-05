"""
Health check router.

Provides health and status endpoints for the API.
"""

from datetime import datetime
from typing import Any

from fastapi import APIRouter, Depends

from app.core.config import Settings, get_settings

router = APIRouter(prefix="/health", tags=["Health"])


@router.get("")
async def health_check() -> dict[str, Any]:
    """
    Basic health check endpoint.
    
    Returns:
        Health status with timestamp
    """
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
    }


@router.get("/ready")
async def readiness_check(
    settings: Settings = Depends(get_settings),
) -> dict[str, Any]:
    """
    Readiness check endpoint.
    
    Verifies that the application is ready to serve requests.
    
    Returns:
        Readiness status with configuration info
    """
    return {
        "status": "ready",
        "timestamp": datetime.utcnow().isoformat(),
        "config": {
            "app_name": settings.app_name,
            "llm_provider": settings.llm_provider,
            "llm_model": settings.llm_model,
            "data_source": settings.data_source,
            "solver": settings.solver,
        },
    }


@router.get("/live")
async def liveness_check() -> dict[str, str]:
    """
    Liveness check endpoint.
    
    Simple endpoint to verify the service is alive.
    
    Returns:
        Liveness status
    """
    return {"status": "alive"}


@router.get("/info")
async def app_info(
    settings: Settings = Depends(get_settings),
) -> dict[str, Any]:
    """
    Application information endpoint.
    
    Returns:
        Application metadata and configuration
    """
    return {
        "app_name": settings.app_name,
        "version": "1.0.0",
        "description": "Multi-Agent Equity Portfolio Manager",
        "configuration": {
            "llm_provider": settings.llm_provider,
            "llm_model": settings.llm_model,
            "data_source": settings.data_source,
            "csv_data_path": settings.csv_data_path,
            "solver": settings.solver,
            "portfolio_size": settings.portfolio_size,
            "risk_aversion": settings.risk_aversion,
            "stock_active_weight_limit": settings.stock_active_weight_limit,
            "sector_active_weight_limit": settings.sector_active_weight_limit,
        },
        "agents": [
            "DataAgent",
            "AlphaAgent",
            "RiskAgent",
            "OptimizationAgent",
            "ComplianceAgent",
        ],
    }

