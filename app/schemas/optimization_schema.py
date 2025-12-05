"""
Optimization API schemas.

Defines request/response models for optimization endpoints.
"""

from typing import Optional

from pydantic import BaseModel, Field


class OptimizationRequest(BaseModel):
    """Request model for portfolio optimization."""
    
    portfolio_id: str = Field(
        default="ALPHA_GROWTH_25",
        description="Portfolio identifier"
    )
    as_of_date: Optional[str] = Field(
        default=None,
        description="Data as-of date (YYYY-MM-DD). Uses latest if not specified."
    )
    portfolio_size: int = Field(
        default=25,
        ge=5,
        le=100,
        description="Target number of holdings"
    )
    
    # Optimization parameters
    risk_aversion: float = Field(
        default=0.01,
        ge=0.001,
        le=1.0,
        description="Risk aversion parameter (lambda)"
    )
    max_iterations: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Maximum optimization iterations"
    )
    
    # Constraint overrides
    stock_active_weight_limit: Optional[float] = Field(
        default=None,
        description="Stock active weight limit (%). Default: 1%"
    )
    sector_active_weight_limit: Optional[float] = Field(
        default=None,
        description="Sector active weight limit (%). Default: 2%"
    )
    
    # LLM options
    use_llm_analysis: bool = Field(
        default=True,
        description="Whether to generate LLM-powered analysis"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "portfolio_id": "ALPHA_GROWTH_25",
                "as_of_date": "2025-12-05",
                "portfolio_size": 25,
                "risk_aversion": 0.01,
                "max_iterations": 5,
                "use_llm_analysis": True,
            }
        }


class OptimizationResponse(BaseModel):
    """Response model for optimization result."""
    
    # Status
    status: str = Field(..., description="Optimization status")
    is_compliant: bool = Field(..., description="Compliance status")
    
    # Portfolio
    portfolio_id: str = Field(..., description="Portfolio identifier")
    as_of_date: str = Field(..., description="Data as-of date")
    
    # Results
    total_holdings: int = Field(..., description="Number of optimized holdings")
    weights: dict[str, float] = Field(
        default_factory=dict,
        description="Ticker -> weight (decimal) mapping"
    )
    
    # Metrics
    expected_alpha: float = Field(default=0.0, description="Expected portfolio alpha")
    expected_risk_pct: float = Field(default=0.0, description="Expected volatility (%)")
    objective_value: float = Field(default=0.0, description="Optimization objective value")
    
    # Execution
    iterations: int = Field(default=0, description="Number of iterations")
    solve_time_seconds: float = Field(default=0.0, description="Solve time")
    
    # Compliance
    violations: list[dict] = Field(
        default_factory=list,
        description="List of constraint violations"
    )
    
    # Analysis
    data_analysis: Optional[str] = Field(None, description="LLM data analysis")
    alpha_analysis: Optional[str] = Field(None, description="LLM alpha analysis")
    risk_analysis: Optional[str] = Field(None, description="LLM risk analysis")
    optimization_analysis: Optional[str] = Field(None, description="LLM optimization analysis")
    compliance_analysis: Optional[str] = Field(None, description="LLM compliance analysis")
    
    # Execution log
    execution_log: list[str] = Field(
        default_factory=list,
        description="Agent execution log"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "status": "optimal",
                "is_compliant": True,
                "portfolio_id": "ALPHA_GROWTH_25",
                "as_of_date": "2025-12-05",
                "total_holdings": 25,
                "expected_alpha": 0.92,
                "expected_risk_pct": 18.5,
                "iterations": 1,
                "solve_time_seconds": 0.15,
            }
        }


class OptimizationStatusResponse(BaseModel):
    """Response model for optimization job status."""
    
    job_id: str = Field(..., description="Job identifier")
    status: str = Field(..., description="Job status")
    progress: float = Field(default=0.0, description="Progress (0-100)")
    current_step: str = Field(default="", description="Current step")
    
    # Timing
    started_at: Optional[str] = Field(None, description="Start timestamp")
    completed_at: Optional[str] = Field(None, description="Completion timestamp")
    
    # Result reference
    result_available: bool = Field(default=False)
    error_message: Optional[str] = Field(None)


class ConstraintCheckRequest(BaseModel):
    """Request model for constraint check."""
    
    weights: dict[str, float] = Field(
        ...,
        description="Ticker -> weight (%) mapping to check"
    )
    benchmark_id: str = Field(
        default="SPX",
        description="Benchmark identifier"
    )


class ConstraintCheckResponse(BaseModel):
    """Response model for constraint check."""
    
    is_compliant: bool = Field(..., description="Overall compliance status")
    total_violations: int = Field(default=0)
    stock_violations: int = Field(default=0)
    sector_violations: int = Field(default=0)
    violations: list[dict] = Field(default_factory=list)
    max_stock_breach: float = Field(default=0.0)
    max_sector_breach: float = Field(default=0.0)


class EfficientFrontierRequest(BaseModel):
    """Request model for efficient frontier calculation."""
    
    tickers: list[str] = Field(
        default_factory=list,
        description="Tickers to include. Empty = use top alpha."
    )
    n_points: int = Field(
        default=20,
        ge=5,
        le=50,
        description="Number of frontier points"
    )


class EfficientFrontierResponse(BaseModel):
    """Response model for efficient frontier."""
    
    points: list[dict] = Field(
        default_factory=list,
        description="List of (risk, return, weights) points"
    )
    min_risk_portfolio: Optional[dict] = Field(None)
    max_return_portfolio: Optional[dict] = Field(None)

