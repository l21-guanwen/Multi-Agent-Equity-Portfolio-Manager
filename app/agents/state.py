"""
Portfolio state definition for LangGraph workflow.

Defines the shared state that flows between agents in the graph.
"""

from datetime import date
from typing import Annotated, Any, Optional

from pydantic import BaseModel, Field


def add_to_list(existing: list, new: list) -> list:
    """Reducer to append new items to existing list."""
    return existing + new


class PortfolioState(BaseModel):
    """
    State passed between agents in the LangGraph workflow.
    
    This state contains all data and intermediate results
    needed for portfolio construction.
    """

    # ===========================================
    # Configuration
    # ===========================================
    as_of_date: str = Field(default="", description="Data as-of date (YYYY-MM-DD)")
    portfolio_id: str = Field(default="ALPHA_GROWTH_25", description="Portfolio identifier")
    portfolio_size: int = Field(default=25, description="Target number of holdings")
    
    # ===========================================
    # Input Data (loaded by Data Agent)
    # ===========================================
    benchmark_data: Optional[dict[str, Any]] = Field(
        default=None, 
        description="Serialized benchmark data"
    )
    universe_tickers: list[str] = Field(
        default_factory=list,
        description="List of tickers in investment universe"
    )
    alpha_scores: dict[str, float] = Field(
        default_factory=dict,
        description="Ticker -> alpha score mapping"
    )
    alpha_quintiles: dict[str, int] = Field(
        default_factory=dict,
        description="Ticker -> quintile mapping"
    )
    benchmark_weights: dict[str, float] = Field(
        default_factory=dict,
        description="Ticker -> benchmark weight (%) mapping"
    )
    sector_mapping: dict[str, str] = Field(
        default_factory=dict,
        description="Ticker -> GICS sector mapping"
    )
    current_weights: dict[str, float] = Field(
        default_factory=dict,
        description="Current portfolio weights (decimal) for rebalancing"
    )
    
    # ===========================================
    # Data Validation
    # ===========================================
    data_validation_passed: bool = Field(default=False, description="Data validation status")
    data_validation_issues: list[str] = Field(
        default_factory=list,
        description="List of data validation issues"
    )
    data_summary: dict[str, Any] = Field(
        default_factory=dict,
        description="Summary of loaded data"
    )
    
    # ===========================================
    # Alpha Agent Outputs
    # ===========================================
    selected_tickers: list[str] = Field(
        default_factory=list,
        description="Top tickers selected for portfolio"
    )
    alpha_analysis: str = Field(
        default="",
        description="LLM-generated alpha analysis"
    )
    
    # ===========================================
    # Risk Agent Outputs
    # ===========================================
    factor_exposures: dict[str, float] = Field(
        default_factory=dict,
        description="Portfolio factor exposures"
    )
    portfolio_risk_pct: float = Field(
        default=0.0,
        description="Expected portfolio volatility (%)"
    )
    risk_analysis: str = Field(
        default="",
        description="LLM-generated risk analysis"
    )
    
    # ===========================================
    # Optimization Agent Outputs
    # ===========================================
    optimal_weights: dict[str, float] = Field(
        default_factory=dict,
        description="Ticker -> optimal weight (decimal) mapping"
    )
    optimization_status: str = Field(
        default="",
        description="Optimization solver status"
    )
    expected_alpha: float = Field(
        default=0.0,
        description="Expected portfolio alpha"
    )
    optimization_analysis: str = Field(
        default="",
        description="LLM-generated optimization analysis"
    )
    
    # ===========================================
    # Compliance Agent Outputs
    # ===========================================
    is_compliant: bool = Field(
        default=False,
        description="Portfolio compliance status"
    )
    compliance_violations: list[dict[str, Any]] = Field(
        default_factory=list,
        description="List of constraint violations"
    )
    compliance_analysis: str = Field(
        default="",
        description="LLM-generated compliance analysis"
    )
    
    # ===========================================
    # Workflow Control
    # ===========================================
    iteration_count: int = Field(
        default=0,
        description="Number of optimization iterations"
    )
    max_iterations: int = Field(
        default=5,
        description="Maximum optimization iterations"
    )
    error_message: Optional[str] = Field(
        default=None,
        description="Error message if workflow failed"
    )
    current_agent: str = Field(
        default="",
        description="Name of currently executing agent"
    )
    
    # ===========================================
    # Final Output
    # ===========================================
    final_portfolio: dict[str, Any] = Field(
        default_factory=dict,
        description="Final portfolio holdings"
    )
    
    # Execution log (accumulates across agents)
    execution_log: Annotated[list[str], add_to_list] = Field(
        default_factory=list,
        description="Execution log messages"
    )

    class Config:
        """Pydantic model configuration."""
        arbitrary_types_allowed = True

    def add_log(self, message: str) -> None:
        """Add a message to the execution log."""
        self.execution_log.append(message)

    def to_summary(self) -> dict[str, Any]:
        """Get a summary of the current state."""
        return {
            "as_of_date": self.as_of_date,
            "portfolio_id": self.portfolio_id,
            "data_validated": self.data_validation_passed,
            "selected_securities": len(self.selected_tickers),
            "optimized_positions": len(self.optimal_weights),
            "is_compliant": self.is_compliant,
            "portfolio_risk": self.portfolio_risk_pct,
            "expected_alpha": self.expected_alpha,
            "iteration_count": self.iteration_count,
            "error": self.error_message,
        }

