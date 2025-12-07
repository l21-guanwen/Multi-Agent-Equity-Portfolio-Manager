"""
Optimization Agent for the multi-agent portfolio management system.

Responsible for portfolio optimization using configured solver.
"""

from typing import Any, Optional

from app.agents.prompts import OPTIMIZATION_AGENT_SYSTEM_PROMPT
from app.agents.state import PortfolioState
from app.llm.interfaces.llm_provider import ILLMProvider
from app.core.config import get_settings
from app.repositories.interfaces.constraint_repository import IConstraintRepository
from app.repositories.interfaces.risk_repository import IRiskRepository
from app.repositories.interfaces.transaction_cost_repository import ITransactionCostRepository
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
        transaction_cost_repository: ITransactionCostRepository,
        llm_provider: Optional[ILLMProvider] = None,
    ):
        """
        Initialize the Optimization Agent.
        
        Args:
            optimization_service: Service for portfolio optimization
            risk_repository: Repository for risk model data
            constraint_repository: Repository for constraint data
            transaction_cost_repository: Repository for transaction cost data
            llm_provider: Optional LLM provider for analysis
        """
        self._optimization_service = optimization_service
        self._risk_repo = risk_repository
        self._constraint_repo = constraint_repository
        self._transaction_cost_repo = transaction_cost_repository
        self._llm = llm_provider
        self._settings = get_settings()

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
                from app.models.constraint import ConstraintSet
                constraint_set = ConstraintSet(stock_constraints=[], sector_constraints=[])
            
            # Filter selected tickers to only those with risk model coverage
            covered_tickers = {fl.ticker for fl in risk_model.factor_loadings}
            original_count = len(state.selected_tickers)
            filtered_tickers = [t for t in state.selected_tickers if t in covered_tickers]
            
            if len(filtered_tickers) < original_count:
                missing = original_count - len(filtered_tickers)
                execution_log.append(
                    f"[OptimizationAgent] WARNING: {missing} tickers excluded (no risk model coverage)"
                )
            
            if not filtered_tickers:
                execution_log.append("[OptimizationAgent] ERROR: No tickers with risk model coverage")
                return {
                    "error_message": "No selected tickers have risk model coverage",
                    "execution_log": execution_log,
                    "current_agent": "optimization_agent",
                }
            
            # Use filtered tickers for optimization
            tickers_to_optimize = filtered_tickers
            execution_log.append(f"[OptimizationAgent] {len(tickers_to_optimize)} tickers with risk coverage")
            
            # Load transaction cost model
            transaction_cost_model = None
            if self._settings.transaction_cost_penalty > 0:
                transaction_cost_model = await self._transaction_cost_repo.get_transaction_cost_model()
                if transaction_cost_model:
                    execution_log.append(
                        f"[OptimizationAgent] Transaction costs loaded: {transaction_cost_model.security_count} securities"
                    )
            
            # Prepare optimization input with current weights for rebalancing
            current_weights_decimal = None
            if state.current_weights:
                # Convert percentage to decimal if needed
                current_weights_decimal = {
                    t: w / 100.0 if w > 1.0 else w 
                    for t, w in state.current_weights.items()
                }
                execution_log.append(
                    f"[OptimizationAgent] Current portfolio weights loaded: {len(current_weights_decimal)} positions"
                )
            
            optimization_input = OptimizationInput(
                eligible_tickers=tickers_to_optimize,
                alpha_scores={
                    t: state.alpha_scores.get(t, 0.0) 
                    for t in tickers_to_optimize
                },
                benchmark_weights=state.benchmark_weights,
                current_weights=current_weights_decimal,
            )
            
            # Set optimization parameters from config
            parameters = OptimizationParameters(
                risk_aversion=self._settings.risk_aversion,
                alpha_coefficient=1.0,
                transaction_cost_penalty=self._settings.transaction_cost_penalty,
                max_portfolio_size=state.portfolio_size,
                long_only=True,
            )
            
            execution_log.append(
                f"[OptimizationAgent] Optimizing {len(tickers_to_optimize)} securities"
            )
            
            # Run optimization with transaction cost model
            result = await self._optimization_service.optimize_portfolio(
                optimization_input=optimization_input,
                risk_model=risk_model,
                constraint_set=constraint_set,
                parameters=parameters,
                transaction_cost_model=transaction_cost_model,
            )
            
            execution_log.append(
                f"[OptimizationAgent] Optimization status: {result.status}"
            )
            
            # Check for solver errors
            if result.status not in ["optimal", "optimal_inaccurate"]:
                error_msg = f"Optimization failed with status: {result.status}"
                if result.status == "infeasible":
                    error_msg += " - Constraints may be too restrictive"
                elif result.status == "solver_error":
                    error_msg += " - Solver encountered an error"
                execution_log.append(f"[OptimizationAgent] ERROR: {error_msg}")
                return {
                    "optimization_status": result.status,
                    "error_message": error_msg,
                    "optimal_weights": {},
                    "expected_alpha": 0.0,
                    "portfolio_risk_pct": 0.0,
                    "optimization_analysis": f"Optimization failed: {error_msg}",
                    "execution_log": execution_log,
                    "current_agent": "optimization_agent",
                }
            
            # Log optimization results
            if result.weights:
                non_zero_positions = len([w for w in result.weights.values() if w > 0.001])
                execution_log.append(
                    f"[OptimizationAgent] Optimized portfolio: {non_zero_positions} positions, "
                    f"alpha={result.expected_alpha:.4f}, risk={result.expected_risk:.2f}%"
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

