"""API module for FastAPI routers."""

from app.api.health_router import router as health_router
from app.api.portfolio_router import router as portfolio_router
from app.api.optimization_router import router as optimization_router

__all__ = [
    "health_router",
    "portfolio_router",
    "optimization_router",
]

