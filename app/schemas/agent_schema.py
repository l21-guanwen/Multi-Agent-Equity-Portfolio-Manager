"""
Agent API schemas.

Defines request/response models for agent execution endpoints.
"""

from typing import Any, Optional

from pydantic import BaseModel, Field


class AgentExecutionRequest(BaseModel):
    """Request model for executing the agent workflow."""
    
    portfolio_id: str = Field(
        default="ALPHA_GROWTH_25",
        description="Portfolio identifier"
    )
    as_of_date: Optional[str] = Field(
        default=None,
        description="Data as-of date (YYYY-MM-DD)"
    )
    portfolio_size: int = Field(
        default=25,
        ge=5,
        le=100,
        description="Target portfolio size"
    )
    max_iterations: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Maximum optimization iterations"
    )
    use_llm: bool = Field(
        default=True,
        description="Whether to use LLM for analysis"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "portfolio_id": "ALPHA_GROWTH_25",
                "as_of_date": "2025-12-05",
                "portfolio_size": 25,
                "max_iterations": 5,
                "use_llm": True,
            }
        }


class AgentExecutionResponse(BaseModel):
    """Response model for agent workflow execution."""
    
    # Status
    success: bool = Field(..., description="Whether workflow completed successfully")
    error_message: Optional[str] = Field(None, description="Error message if failed")
    
    # Portfolio result
    portfolio_id: str = Field(..., description="Portfolio identifier")
    as_of_date: str = Field(..., description="Data as-of date")
    is_compliant: bool = Field(default=False, description="Compliance status")
    
    # Holdings
    total_holdings: int = Field(default=0, description="Number of holdings")
    holdings: list[dict] = Field(
        default_factory=list,
        description="Portfolio holdings"
    )
    
    # Metrics
    expected_alpha: float = Field(default=0.0)
    expected_risk_pct: float = Field(default=0.0)
    
    # Agent outputs
    data_summary: dict[str, Any] = Field(default_factory=dict)
    selected_tickers: list[str] = Field(default_factory=list)
    factor_exposures: dict[str, float] = Field(default_factory=dict)
    optimal_weights: dict[str, float] = Field(default_factory=dict)
    compliance_violations: list[dict] = Field(default_factory=list)
    
    # LLM Analysis
    alpha_analysis: Optional[str] = Field(None)
    risk_analysis: Optional[str] = Field(None)
    optimization_analysis: Optional[str] = Field(None)
    compliance_analysis: Optional[str] = Field(None)
    
    # Execution details
    iteration_count: int = Field(default=0)
    execution_log: list[str] = Field(default_factory=list)


class WorkflowStatusResponse(BaseModel):
    """Response model for workflow status."""
    
    workflow_id: str = Field(..., description="Workflow identifier")
    status: str = Field(..., description="Current status")
    current_agent: str = Field(default="", description="Currently executing agent")
    progress_pct: float = Field(default=0.0, description="Progress percentage")
    
    # Timing
    started_at: Optional[str] = Field(None)
    updated_at: Optional[str] = Field(None)
    completed_at: Optional[str] = Field(None)
    
    # Agent status
    agents_completed: list[str] = Field(default_factory=list)
    agents_pending: list[str] = Field(default_factory=list)
    
    # Result
    result_available: bool = Field(default=False)
    error_message: Optional[str] = Field(None)


class AgentInfoResponse(BaseModel):
    """Response model for agent information."""
    
    name: str = Field(..., description="Agent name")
    description: str = Field(..., description="Agent description")
    inputs: list[str] = Field(default_factory=list, description="Required inputs")
    outputs: list[str] = Field(default_factory=list, description="Produced outputs")


class WorkflowInfoResponse(BaseModel):
    """Response model for workflow information."""
    
    name: str = Field(default="Portfolio Construction Workflow")
    description: str = Field(
        default="Multi-agent workflow for equity portfolio construction"
    )
    agents: list[AgentInfoResponse] = Field(default_factory=list)
    
    # Workflow structure
    entry_point: str = Field(default="data_agent")
    exit_points: list[str] = Field(default_factory=list)
    
    # Configuration
    max_iterations: int = Field(default=5)
    retry_on_compliance_failure: bool = Field(default=True)

