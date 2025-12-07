"""
Chain-of-Thought (CoT) Optimization Agent.

This agent uses step-by-step reasoning to solve the portfolio optimization problem.
Unlike ReAct agents that choose tools, this agent thinks through the math problem
and determines optimal portfolio weights.

When use_llm=False, falls back to CVXPY mathematical solver.
"""

import json
import re
from typing import Any, Optional

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.language_models import BaseChatModel

from app.agents.state import PortfolioState
from app.core.config import get_settings
from app.models.constraint import ConstraintSet
from app.models.risk import RiskModel
from app.services.interfaces.optimization_service import OptimizationInput, OptimizationParameters
from app.models.transaction_cost import TransactionCostModel
from app.repositories.interfaces.constraint_repository import IConstraintRepository
from app.repositories.interfaces.risk_repository import IRiskRepository
from app.repositories.interfaces.transaction_cost_repository import ITransactionCostRepository
from app.services.interfaces.optimization_service import IOptimizationService


COT_OPTIMIZATION_SYSTEM_PROMPT = """You are a Quantitative Portfolio Optimizer. Your task is to determine optimal portfolio weights by reasoning through the problem step-by-step.

## OPTIMIZATION OBJECTIVE
Maximize: α'w - λ × w'Σw - τ × |TC × Δw|

Where:
- α = Alpha scores (expected excess returns)
- w = Portfolio weights (to be determined)
- Σ = Covariance matrix (risk)
- λ = Risk aversion parameter
- TC = Transaction costs
- Δw = Change from current weights
- τ = Transaction cost penalty

## CONSTRAINTS
1. Sum of weights = 100% (fully invested)
2. All weights >= 0 (long only)
3. Single stock active weight: ±1% vs benchmark
4. Sector active weight: ±2% vs benchmark

## YOUR APPROACH
Think through this step-by-step:

1. **Understand the Inputs**: Review alpha scores, benchmark weights, constraints
2. **Identify Top Alpha Securities**: Start with highest alpha Q1 securities
3. **Apply Constraints**: Ensure weights respect position limits
4. **Balance Alpha vs Risk**: Higher alpha should get more weight, but diversify
5. **Calculate Sector Weights**: Ensure sector constraints are met
6. **Finalize Weights**: Adjust to sum to 100%

## OUTPUT FORMAT
After your reasoning, provide the final weights as JSON:

```json
{
  "weights": {
    "TICKER1": 0.05,
    "TICKER2": 0.04,
    ...
  },
  "reasoning_summary": "Brief explanation of key decisions"
}
```

Weights should be decimals that sum to 1.0 (e.g., 0.05 = 5%)."""


class ChainOfThoughtOptimizationAgent:
    """
    Chain-of-Thought agent for portfolio optimization.
    
    This agent uses LLM reasoning to determine portfolio weights.
    It prompts the LLM with:
    - Objective function and constraints
    - Alpha scores for eligible securities
    - Benchmark weights and sector information
    - Current holdings (for transaction cost consideration)
    
    When LLM is unavailable or use_llm=False, it falls back to CVXPY solver.
    """
    
    def __init__(
        self,
        optimization_service: IOptimizationService,
        risk_repository: IRiskRepository,
        constraint_repository: IConstraintRepository,
        transaction_cost_repository: ITransactionCostRepository,
        llm: Optional[BaseChatModel] = None,
    ):
        """Initialize the CoT Optimization Agent."""
        self._optimization_service = optimization_service
        self._risk_repo = risk_repository
        self._constraint_repo = constraint_repository
        self._transaction_cost_repo = transaction_cost_repository
        self._llm = llm
        self._settings = get_settings()
    
    async def __call__(self, state: PortfolioState) -> dict[str, Any]:
        """Execute the optimization agent."""
        execution_log = [f"[OptimizationAgent] Starting portfolio optimization..."]
        
        try:
            # Check prerequisites
            if not state.selected_tickers:
                execution_log.append("[OptimizationAgent] ERROR: No securities selected")
                return {
                    "error_message": "No securities selected for optimization",
                    "execution_log": execution_log,
                    "current_agent": "optimization_agent",
                    "iteration_count": state.iteration_count + 1,
                }
            
            # Load risk model and constraints
            risk_model = await self._risk_repo.get_risk_model()
            constraint_set = await self._constraint_repo.get_constraint_set()
            transaction_cost_model = await self._transaction_cost_repo.get_transaction_cost_model()
            
            if not risk_model:
                return {
                    "error_message": "Risk model not available",
                    "execution_log": execution_log,
                    "current_agent": "optimization_agent",
                    "iteration_count": state.iteration_count + 1,
                }
            
            # Filter to securities with risk model coverage
            eligible_tickers = [
                t for t in state.selected_tickers 
                if risk_model.get_loadings(t) is not None
            ]
            
            if not eligible_tickers:
                return {
                    "error_message": "No eligible securities with risk model coverage",
                    "execution_log": execution_log,
                    "current_agent": "optimization_agent",
                    "iteration_count": state.iteration_count + 1,
                }
            
            execution_log.append(f"[OptimizationAgent] {len(eligible_tickers)} eligible securities")
            
            # Decide: LLM Chain-of-Thought or Mathematical Solver
            if self._llm and state.use_llm:
                execution_log.append("[OptimizationAgent] Using Chain-of-Thought LLM optimization")
                result = await self._optimize_with_cot(
                    state, eligible_tickers, risk_model, constraint_set, 
                    transaction_cost_model, execution_log
                )
            else:
                execution_log.append("[OptimizationAgent] Using CVXPY mathematical optimization")
                result = await self._optimize_with_solver(
                    state, eligible_tickers, risk_model, constraint_set,
                    transaction_cost_model, execution_log
                )
            
            return result
            
        except Exception as e:
            execution_log.append(f"[OptimizationAgent] ERROR: {str(e)}")
            return {
                "error_message": f"Optimization failed: {str(e)}",
                "execution_log": execution_log,
                "current_agent": "optimization_agent",
                "iteration_count": state.iteration_count + 1,  # Increment even on error
            }
    
    async def _optimize_with_cot(
        self,
        state: PortfolioState,
        eligible_tickers: list[str],
        risk_model: RiskModel,
        constraint_set: Optional[ConstraintSet],
        transaction_cost_model: Optional[TransactionCostModel],
        execution_log: list[str],
    ) -> dict[str, Any]:
        """Use Chain-of-Thought reasoning to determine portfolio weights."""
        
        # Build the problem description for the LLM
        problem_prompt = self._build_cot_prompt(
            state, eligible_tickers, risk_model, constraint_set, transaction_cost_model
        )
        
        # Get LLM response
        messages = [
            SystemMessage(content=COT_OPTIMIZATION_SYSTEM_PROMPT),
            HumanMessage(content=problem_prompt),
        ]
        
        response = await self._llm.ainvoke(messages)
        execution_log.append("[OptimizationAgent] LLM reasoning complete")
        
        # Parse weights from response
        weights = self._parse_weights_from_response(response.content, eligible_tickers)
        
        if not weights:
            execution_log.append("[OptimizationAgent] Failed to parse LLM weights, falling back to solver")
            return await self._optimize_with_solver(
                state, eligible_tickers, risk_model, constraint_set,
                transaction_cost_model, execution_log
            )
        
        # Calculate metrics
        expected_alpha = sum(
            state.alpha_scores.get(t, 0) * w 
            for t, w in weights.items()
        )
        
        execution_log.append(f"[OptimizationAgent] LLM determined {len(weights)} positions")
        execution_log.append(f"[OptimizationAgent] Expected alpha: {expected_alpha:.4f}")
        
        return {
            "optimal_weights": weights,
            "optimization_status": "optimal",
            "expected_alpha": expected_alpha,
            "portfolio_risk_pct": state.portfolio_risk_pct or 15.0,
            "optimization_analysis": response.content[:500],
            "execution_log": execution_log,
            "current_agent": "optimization_agent",
            "iteration_count": state.iteration_count + 1,  # Increment iteration counter
        }
    
    def _build_cot_prompt(
        self,
        state: PortfolioState,
        eligible_tickers: list[str],
        risk_model: RiskModel,
        constraint_set: Optional[ConstraintSet],
        transaction_cost_model: Optional[TransactionCostModel],
    ) -> str:
        """Build the problem prompt for Chain-of-Thought reasoning."""
        
        # Prepare security data
        securities_data = []
        for ticker in eligible_tickers[:50]:  # Limit to top 50 for prompt size
            alpha = state.alpha_scores.get(ticker, 0)
            bm_weight = state.benchmark_weights.get(ticker, 0)
            sector = state.sector_mapping.get(ticker, "Unknown")
            
            securities_data.append({
                "ticker": ticker,
                "alpha_score": round(alpha, 4),
                "benchmark_weight_pct": round(bm_weight, 2),
                "sector": sector,
            })
        
        # Sort by alpha
        securities_data.sort(key=lambda x: x["alpha_score"], reverse=True)
        
        # Get sector benchmark weights
        sector_weights = {}
        for ticker in eligible_tickers:
            sector = state.sector_mapping.get(ticker, "Unknown")
            bm_weight = state.benchmark_weights.get(ticker, 0)
            sector_weights[sector] = sector_weights.get(sector, 0) + bm_weight
        
        # Build constraints description
        constraints_desc = """
Constraints:
- Single stock: Portfolio weight must be within ±1% of benchmark weight
- Sector: Total sector weight must be within ±2% of benchmark sector weight
- Long only: All weights >= 0
- Fully invested: Weights sum to 100%"""
        
        prompt = f"""## PORTFOLIO OPTIMIZATION PROBLEM

### Parameters
- Target portfolio size: {state.portfolio_size} securities
- Risk aversion (λ): {self._settings.risk_aversion}
- Transaction cost penalty (τ): {self._settings.transaction_cost_penalty}

### Eligible Securities (sorted by alpha score)
{json.dumps(securities_data[:30], indent=2)}

### Benchmark Sector Weights
{json.dumps({k: round(v, 2) for k, v in sector_weights.items()}, indent=2)}

{constraints_desc}

### Your Task
Think step-by-step to determine the optimal portfolio weights:

1. Start with the highest alpha securities
2. Assign weights based on alpha (higher alpha = more weight)
3. Ensure no single stock exceeds benchmark ± 1%
4. Check sector totals are within benchmark ± 2%
5. Normalize weights to sum to 1.0

Provide your reasoning and final weights as JSON."""

        return prompt
    
    def _parse_weights_from_response(
        self, response: str, eligible_tickers: list[str]
    ) -> Optional[dict[str, float]]:
        """Parse portfolio weights from LLM response."""
        
        # Try to find JSON block in response
        json_match = re.search(r'\{[^{}]*"weights"[^{}]*\{[^{}]+\}[^{}]*\}', response, re.DOTALL)
        
        if json_match:
            try:
                data = json.loads(json_match.group())
                weights = data.get("weights", {})
                
                # Validate and normalize
                valid_weights = {}
                for ticker, weight in weights.items():
                    if ticker in eligible_tickers and isinstance(weight, (int, float)) and weight > 0:
                        valid_weights[ticker] = float(weight)
                
                # Normalize to sum to 1
                total = sum(valid_weights.values())
                if total > 0:
                    return {t: w / total for t, w in valid_weights.items()}
            except json.JSONDecodeError:
                pass
        
        # Try simpler pattern
        weights_match = re.search(r'"weights"\s*:\s*\{([^}]+)\}', response)
        if weights_match:
            try:
                weights_str = "{" + weights_match.group(1) + "}"
                weights = json.loads(weights_str)
                
                valid_weights = {
                    t: float(w) for t, w in weights.items()
                    if t in eligible_tickers and float(w) > 0
                }
                
                total = sum(valid_weights.values())
                if total > 0:
                    return {t: w / total for t, w in valid_weights.items()}
            except:
                pass
        
        return None
    
    async def _optimize_with_solver(
        self,
        state: PortfolioState,
        eligible_tickers: list[str],
        risk_model: RiskModel,
        constraint_set: Optional[ConstraintSet],
        transaction_cost_model: Optional[TransactionCostModel],
        execution_log: list[str],
    ) -> dict[str, Any]:
        """Use CVXPY mathematical solver for optimization."""
        
        # Prepare inputs
        optimization_input = OptimizationInput(
            eligible_tickers=eligible_tickers,
            alpha_scores={t: state.alpha_scores.get(t, 0) for t in eligible_tickers},
            benchmark_weights=state.benchmark_weights,
            current_weights=state.current_weights,
        )
        
        parameters = OptimizationParameters(
            risk_aversion=self._settings.risk_aversion,
            alpha_coefficient=1.0,
            transaction_cost_penalty=self._settings.transaction_cost_penalty,
            max_portfolio_size=state.portfolio_size,
            long_only=True,
        )
        
        # Run optimization
        result = await self._optimization_service.optimize_portfolio(
            optimization_input=optimization_input,
            risk_model=risk_model,
            constraint_set=constraint_set,
            parameters=parameters,
            transaction_cost_model=transaction_cost_model,
        )
        
        execution_log.append(f"[OptimizationAgent] Solver status: {result.status}")
        
        if result.status not in ["optimal", "optimal_inaccurate"]:
            return {
                "optimization_status": result.status,
                "error_message": f"Optimization failed: {result.status}",
                "execution_log": execution_log,
                "current_agent": "optimization_agent",
                "iteration_count": state.iteration_count + 1,  # Increment even on failure
            }
        
        execution_log.append(f"[OptimizationAgent] {len([w for w in result.weights.values() if w > 0.001])} positions")
        execution_log.append(f"[OptimizationAgent] Expected alpha: {result.expected_alpha:.4f}")
        
        return {
            "optimal_weights": result.weights,
            "optimization_status": result.status,
            "expected_alpha": result.expected_alpha,
            "portfolio_risk_pct": result.expected_risk * 100 if result.expected_risk else state.portfolio_risk_pct,
            "optimization_analysis": f"CVXPY solver found optimal solution with {len(result.weights)} positions",
            "execution_log": execution_log,
            "current_agent": "optimization_agent",
            "iteration_count": state.iteration_count + 1,  # Increment iteration counter
        }

