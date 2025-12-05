"""
Portfolio API schemas.

Defines request/response models for portfolio endpoints.
"""

from datetime import date
from typing import Optional

from pydantic import BaseModel, Field


class PortfolioHoldingResponse(BaseModel):
    """Response model for a single portfolio holding."""
    
    ticker: str = Field(..., description="Stock ticker symbol")
    security_name: Optional[str] = Field(None, description="Security name")
    sector: str = Field(..., description="GICS sector")
    weight_pct: float = Field(..., description="Portfolio weight (%)")
    benchmark_weight_pct: float = Field(..., description="Benchmark weight (%)")
    active_weight_pct: float = Field(..., description="Active weight (%)")
    alpha_score: Optional[float] = Field(None, description="Alpha score (0-1)")
    alpha_quintile: Optional[int] = Field(None, description="Alpha quintile (1-5)")
    shares: Optional[int] = Field(None, description="Number of shares")
    market_value_usd: Optional[float] = Field(None, description="Market value (USD)")
    price: Optional[float] = Field(None, description="Current price (USD)")

    class Config:
        json_schema_extra = {
            "example": {
                "ticker": "AAPL",
                "security_name": "Apple Inc.",
                "sector": "Information Technology",
                "weight_pct": 5.5,
                "benchmark_weight_pct": 6.7,
                "active_weight_pct": -1.2,
                "alpha_score": 0.85,
                "alpha_quintile": 1,
            }
        }


class PortfolioResponse(BaseModel):
    """Response model for full portfolio."""
    
    portfolio_id: str = Field(..., description="Portfolio identifier")
    portfolio_name: str = Field(..., description="Portfolio display name")
    as_of_date: str = Field(..., description="Data as-of date")
    strategy: Optional[str] = Field(None, description="Strategy name")
    
    # Holdings
    holdings: list[PortfolioHoldingResponse] = Field(
        default_factory=list,
        description="List of portfolio holdings"
    )
    
    # Summary metrics
    total_holdings: int = Field(..., description="Number of holdings")
    total_market_value_usd: Optional[float] = Field(None, description="Total market value")
    
    # Risk/Return
    expected_alpha: Optional[float] = Field(None, description="Expected portfolio alpha")
    expected_risk_pct: Optional[float] = Field(None, description="Expected volatility (%)")
    portfolio_beta: Optional[float] = Field(None, description="Portfolio beta")
    
    # Compliance
    is_compliant: bool = Field(default=True, description="Compliance status")
    
    class Config:
        json_schema_extra = {
            "example": {
                "portfolio_id": "ALPHA_GROWTH_25",
                "portfolio_name": "Alpha Growth Portfolio",
                "as_of_date": "2025-12-05",
                "strategy": "Idiosyncratic AI-Based Equity",
                "total_holdings": 25,
                "expected_alpha": 0.92,
                "expected_risk_pct": 18.5,
                "is_compliant": True,
            }
        }


class PortfolioSummaryResponse(BaseModel):
    """Response model for portfolio summary."""
    
    portfolio_id: str = Field(..., description="Portfolio identifier")
    portfolio_name: str = Field(..., description="Portfolio display name")
    as_of_date: str = Field(..., description="Data as-of date")
    total_holdings: int = Field(..., description="Number of holdings")
    
    # Sector allocation
    sector_weights: dict[str, float] = Field(
        default_factory=dict,
        description="Sector -> weight (%) mapping"
    )
    
    # Top holdings
    top_holdings: list[dict] = Field(
        default_factory=list,
        description="Top 10 holdings by weight"
    )
    
    # Risk metrics
    expected_alpha: Optional[float] = Field(None)
    expected_risk_pct: Optional[float] = Field(None)
    tracking_error_pct: Optional[float] = Field(None)
    
    # Factor exposures
    factor_exposures: Optional[dict[str, float]] = Field(None)


class BenchmarkResponse(BaseModel):
    """Response model for benchmark data."""
    
    benchmark_id: str = Field(..., description="Benchmark identifier")
    benchmark_name: str = Field(..., description="Benchmark name")
    as_of_date: str = Field(..., description="Data as-of date")
    total_securities: int = Field(..., description="Number of constituents")
    
    # Sector breakdown
    sector_weights: dict[str, float] = Field(
        default_factory=dict,
        description="Sector -> weight (%) mapping"
    )
    
    # Top constituents
    top_constituents: list[dict] = Field(
        default_factory=list,
        description="Top 10 constituents by weight"
    )


class DataSummaryResponse(BaseModel):
    """Response model for data summary."""
    
    benchmark_count: int = Field(default=0)
    universe_count: int = Field(default=0)
    alpha_count: int = Field(default=0)
    factor_loadings_count: int = Field(default=0)
    constraints_count: int = Field(default=0)
    transaction_costs_count: int = Field(default=0)
    as_of_date: Optional[str] = Field(None)
    
    # Validation
    is_valid: bool = Field(default=False)
    data_quality_score: float = Field(default=0.0)
    issues: list[str] = Field(default_factory=list)

