"""
Optimization router.

Provides endpoints for portfolio optimization and agent execution.
"""

from typing import Any

from fastapi import APIRouter, Depends, HTTPException

from app.core.dependencies import get_portfolio_graph
from app.agents.graph import PortfolioGraph
from app.schemas.optimization_schema import (
    OptimizationRequest,
    OptimizationResponse,
)
from app.schemas.agent_schema import (
    AgentExecutionRequest,
    AgentExecutionResponse,
    WorkflowInfoResponse,
    AgentInfoResponse,
)

router = APIRouter(prefix="/optimization", tags=["Optimization"])


@router.post("/run", response_model=OptimizationResponse)
async def run_optimization(
    request: OptimizationRequest,
    graph: PortfolioGraph = Depends(get_portfolio_graph),
) -> OptimizationResponse:
    """
    Run portfolio optimization workflow.
    
    Executes the full multi-agent workflow:
    1. Data Agent: Load and validate data
    2. Alpha Agent: Select top securities
    3. Risk Agent: Analyze risk
    4. Optimization Agent: Construct portfolio
    5. Compliance Agent: Validate constraints
    
    Returns optimized portfolio with analysis.
    """
    try:
        # Run the workflow
        result = await graph.run(
            portfolio_id=request.portfolio_id,
            as_of_date=request.as_of_date or "",
            portfolio_size=request.portfolio_size,
            max_iterations=request.max_iterations,
            use_llm=request.use_llm_analysis,  # Controls LLM-based vs mathematical optimization
        )
        
        return OptimizationResponse(
            status=result.optimization_status or "completed",
            is_compliant=result.is_compliant,
            portfolio_id=result.portfolio_id,
            as_of_date=result.as_of_date,
            total_holdings=len([w for w in result.optimal_weights.values() if w > 0.001]),
            weights=result.optimal_weights,
            expected_alpha=result.expected_alpha,
            expected_risk_pct=result.portfolio_risk_pct,
            objective_value=0.0,  # Could extract from state
            iterations=result.iteration_count,
            solve_time_seconds=0.0,  # Could track timing
            violations=result.compliance_violations,
            alpha_analysis=result.alpha_analysis if request.use_llm_analysis else None,
            risk_analysis=result.risk_analysis if request.use_llm_analysis else None,
            optimization_analysis=result.optimization_analysis if request.use_llm_analysis else None,
            compliance_analysis=result.compliance_analysis if request.use_llm_analysis else None,
            execution_log=result.execution_log,
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Optimization failed: {str(e)}"
        )


@router.post("/execute", response_model=AgentExecutionResponse)
async def execute_agent_workflow(
    request: AgentExecutionRequest,
    graph: PortfolioGraph = Depends(get_portfolio_graph),
) -> AgentExecutionResponse:
    """
    Execute the full agent workflow.
    
    Similar to /run but returns more detailed agent outputs.
    """
    try:
        result = await graph.run(
            portfolio_id=request.portfolio_id,
            as_of_date=request.as_of_date or "",
            portfolio_size=request.portfolio_size,
            max_iterations=request.max_iterations,
            use_llm=request.use_llm,  # Controls LLM-based vs mathematical optimization
        )
        
        # Build holdings list
        holdings = []
        for ticker, weight in result.optimal_weights.items():
            if weight > 0.001:
                holdings.append({
                    "ticker": ticker,
                    "weight_pct": weight * 100,
                    "alpha_score": result.alpha_scores.get(ticker, 0.0),
                    "sector": result.sector_mapping.get(ticker, "Unknown"),
                })
        
        # Sort by weight
        holdings.sort(key=lambda x: x["weight_pct"], reverse=True)
        
        return AgentExecutionResponse(
            success=result.error_message is None,
            error_message=result.error_message,
            portfolio_id=result.portfolio_id,
            as_of_date=result.as_of_date,
            is_compliant=result.is_compliant,
            total_holdings=len(holdings),
            holdings=holdings,
            expected_alpha=result.expected_alpha,
            expected_risk_pct=result.portfolio_risk_pct,
            data_summary=result.data_summary,
            selected_tickers=result.selected_tickers,
            factor_exposures=result.factor_exposures,
            optimal_weights=result.optimal_weights,
            compliance_violations=result.compliance_violations,
            alpha_analysis=result.alpha_analysis if request.use_llm else None,
            risk_analysis=result.risk_analysis if request.use_llm else None,
            optimization_analysis=result.optimization_analysis if request.use_llm else None,
            compliance_analysis=result.compliance_analysis if request.use_llm else None,
            iteration_count=result.iteration_count,
            execution_log=result.execution_log,
        )
        
    except Exception as e:
        return AgentExecutionResponse(
            success=False,
            error_message=str(e),
            portfolio_id=request.portfolio_id,
            as_of_date=request.as_of_date or "",
            is_compliant=False,
        )


@router.get("/workflow/info", response_model=WorkflowInfoResponse)
async def get_workflow_info() -> WorkflowInfoResponse:
    """
    Get information about the agent workflow.
    
    Returns workflow structure and agent descriptions.
    """
    agents = [
        AgentInfoResponse(
            name="DataAgent",
            description="Loads and validates market data from repositories",
            inputs=["portfolio_id", "as_of_date"],
            outputs=["universe_tickers", "alpha_scores", "benchmark_weights", "sector_mapping"],
        ),
        AgentInfoResponse(
            name="AlphaAgent",
            description="Analyzes alpha scores and selects top securities",
            inputs=["alpha_scores", "alpha_quintiles", "portfolio_size"],
            outputs=["selected_tickers", "alpha_analysis"],
        ),
        AgentInfoResponse(
            name="RiskAgent",
            description="Calculates factor exposures and portfolio risk",
            inputs=["selected_tickers"],
            outputs=["factor_exposures", "portfolio_risk_pct", "risk_analysis"],
        ),
        AgentInfoResponse(
            name="OptimizationAgent",
            description="Constructs optimal portfolio using mean-variance optimization",
            inputs=["selected_tickers", "alpha_scores", "benchmark_weights"],
            outputs=["optimal_weights", "expected_alpha", "optimization_analysis"],
        ),
        AgentInfoResponse(
            name="ComplianceAgent",
            description="Validates portfolio against stock and sector constraints",
            inputs=["optimal_weights", "benchmark_weights", "sector_mapping"],
            outputs=["is_compliant", "compliance_violations", "compliance_analysis"],
        ),
    ]
    
    return WorkflowInfoResponse(
        name="Portfolio Construction Workflow",
        description="Multi-agent workflow for AI-driven equity portfolio construction",
        agents=agents,
        entry_point="DataAgent",
        exit_points=["ComplianceAgent"],
        max_iterations=5,
        retry_on_compliance_failure=True,
    )

