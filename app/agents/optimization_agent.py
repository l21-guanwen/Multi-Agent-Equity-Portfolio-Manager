"""
Optimization Agent for the multi-agent portfolio management system.

Responsible for portfolio optimization using configured solver.
"""

from typing import Any, Optional

from app.agents.prompts import OPTIMIZATION_AGENT_SYSTEM_PROMPT
from app.agents.state import PortfolioState
from app.llm.interfaces.llm_provider import ILLMProvider
from app.repositories.interfaces.constraint_repository import IConstraintRepository
from app.repositories.interfaces.risk_repository import IRiskRepository
from app.services.interfaces.optimization_service import (
    IOptimizationService,
    OptimizationInput,
    OptimizationParameters,
)


class OptimizationAgent:
    """
    Optimization Agent responsible for portfolio construction.
    
    This agent:
    1. Prepares optimization inputs (alpha, risk, constraints)
    2. Runs portfolio optimization
    3. Uses LLM to explain optimization results
    """

    def __init__(
        self,
        optimization_service: IOptimizationService,
        risk_repository: IRiskRepository,
        constraint_repository: IConstraintRepository,
        llm_provider: Optional[ILLMProvider] = None,
    ):
        """
        Initialize the Optimization Agent.
        
        Args:
            optimization_service: Service for portfolio optimization
            risk_repository: Repository for risk model data
            constraint_repository: Repository for constraint data
            llm_provider: Optional LLM provider for analysis
        """
        self._optimization_service = optimization_service
        self._risk_repo = risk_repository
        self._constraint_repo = constraint_repository
        self._llm = llm_provider

    async def __call__(self, state: PortfolioState) -> dict[str, Any]:
        """
        Execute the optimization agent.
        
        Args:
            state: Current portfolio state
            
        Returns:
            Updated state fields
        """
        execution_log = [f"[OptimizationAgent] Starting portfolio optimization..."]
        
        try:
            # Check prerequisites
            if not state.selected_tickers:
                execution_log.append("[OptimizationAgent] Skipped - no securities selected")
                return {
                    "execution_log": execution_log,
                    "current_agent": "optimization_agent",
                }
            
            # Load risk model
            risk_model = await self._risk_repo.get_risk_model()
            if not risk_model:
                execution_log.append("[OptimizationAgent] ERROR: No risk model")
                return {
                    "error_message": "Risk model not available for optimization",
                    "execution_log": execution_log,
                    "current_agent": "optimization_agent",
                }
            
            # Load constraints
            constraint_set = await self._constraint_repo.get_constraint_set()
            if not constraint_set:
                execution_log.append("[OptimizationAgent] WARNING: No constraints loaded")
            
            # Prepare optimization input
            optimization_input = OptimizationInput(
                eligible_tickers=state.selected_tickers,
                alpha_scores={
                    t: state.alpha_scores.get(t, 0.0) 
                    for t in state.selected_tickers
                },
                benchmark_weights=state.benchmark_weights,
            )
            
            # Set optimization parameters
            parameters = OptimizationParameters(
                risk_aversion=0.01,
                alpha_coefficient=1.0,
                transaction_cost_penalty=0.0,
                max_portfolio_size=state.portfolio_size,
                long_only=True,
            )
            
            execution_log.append(
                f"[OptimizationAgent] Optimizing {len(state.selected_tickers)} securities"
            )
            
            # Run optimization
            result = await self._optimization_service.optimize_portfolio(
                optimization_input=optimization_input,
                risk_model=risk_model,
                constraint_set=constraint_set,
                parameters=parameters,
            )
            
            execution_log.append(
                f"[OptimizationAgent] Optimization status: {result.status}"
            )
            
            # Generate LLM analysis
            if self._llm:
                llm_analysis = await self._generate_analysis(result, state)
                execution_log.append(f"[OptimizationAgent] LLM analysis generated")
            else:
                llm_analysis = self._generate_basic_analysis(result, state)
            
            execution_log.append(f"[OptimizationAgent] Completed")
            
            # Increment iteration count
            new_iteration = state.iteration_count + 1
            
            return {
                "optimal_weights": result.weights,
                "optimization_status": result.status,
                "expected_alpha": result.expected_alpha,
                "portfolio_risk_pct": result.expected_risk,
                "optimization_analysis": llm_analysis,
                "iteration_count": new_iteration,
                "execution_log": execution_log,
                "current_agent": "optimization_agent",
            }
            
        except Exception as e:
            execution_log.append(f"[OptimizationAgent] ERROR: {str(e)}")
            return {
                "optimization_status": "error",
                "error_message": f"Optimization failed: {str(e)}",
                "execution_log": execution_log,
                "current_agent": "optimization_agent",
            }

    async def _generate_analysis(
        self,
        result: Any,
        state: PortfolioState,
    ) -> str:
        """Generate LLM-powered optimization analysis."""
        # Get top positions
        sorted_weights = sorted(
            result.weights.items(), 
            key=lambda x: x[1], 
            reverse=True
        )[:10]
        
        # Calculate active weights for top positions
        active_positions = []
        for ticker, weight in sorted_weights:
            benchmark_weight = state.benchmark_weights.get(ticker, 0.0) / 100
            active = weight - benchmark_weight
            active_positions.append((ticker, weight * 100, active * 100))
        
        prompt = f"""Analyze the following portfolio optimization results:

Optimization Status: {result.status}
Objective Value: {result.objective_value:.6f}
Expected Alpha: {result.expected_alpha:.4f}
Expected Risk: {result.expected_risk:.2f}%
Solve Time: {result.solve_time_seconds:.3f}s

Top 10 Positions (Weight %, Active Weight %):
{chr(10).join(f'- {t}: {w:.2f}% (active: {a:+.2f}%)' for t, w, a in active_positions)}

Total Positions: {len([w for w in result.weights.values() if w > 0.001])}
Constraints Applied: Stock ±1%, Sector ±2%

Explain the optimization trade-offs and notable position allocations."""

        try:
            response = await self._llm.generate(
                prompt=prompt,
                system_prompt=OPTIMIZATION_AGENT_SYSTEM_PROMPT,
                temperature=0.5,
                max_tokens=600,
            )
            return response.content
        except Exception:
            return self._generate_basic_analysis(result, state)

    def _generate_basic_analysis(
        self,
        result: Any,
        state: PortfolioState,
    ) -> str:
        """Generate basic optimization analysis without LLM."""
        sorted_weights = sorted(
            result.weights.items(),
            key=lambda x: x[1],
            reverse=True
        )[:5]
        
        total_positions = len([w for w in result.weights.values() if w > 0.001])
        
        return f"""Optimization Analysis

Status: {result.status}
Positions: {total_positions}
Expected Alpha: {result.expected_alpha:.4f}
Expected Risk: {result.expected_risk:.2f}%

Top 5 Positions:
{chr(10).join(f'- {t}: {w*100:.2f}%' for t, w in sorted_weights)}

Optimization completed in {result.solve_time_seconds:.3f} seconds."""

