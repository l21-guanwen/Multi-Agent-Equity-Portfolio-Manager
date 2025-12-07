"""
Optimization service implementation.

Handles portfolio optimization using configured solver.
"""

from typing import Any, Optional

import numpy as np

from app.models.alpha import AlphaModel
from app.models.constraint import ConstraintSet
from app.models.risk import RiskModel
from app.models.transaction_cost import TransactionCostModel
from app.services.interfaces.optimization_service import (
    IOptimizationService,
    OptimizationInput,
    OptimizationParameters,
    OptimizationResult,
)
from app.services.interfaces.risk_service import IRiskService
from app.solvers.interfaces.solver import ISolver, SolverStatus


class OptimizationService(IOptimizationService):
    """
    Service for portfolio optimization.
    
    Coordinates between alpha scores, risk model, constraints,
    and optimization solver to construct optimal portfolios.
    """

    def __init__(
        self,
        solver: ISolver,
        risk_service: IRiskService,
    ):
        """
        Initialize the optimization service.
        
        Args:
            solver: Optimization solver instance
            risk_service: Risk service for calculations
        """
        self._solver = solver
        self._risk_service = risk_service

    async def optimize_portfolio(
        self,
        optimization_input: OptimizationInput,
        risk_model: RiskModel,
        constraint_set: ConstraintSet,
        parameters: OptimizationParameters,
        transaction_cost_model: Optional[TransactionCostModel] = None,
    ) -> OptimizationResult:
        """Run portfolio optimization."""
        tickers = optimization_input.eligible_tickers
        n = len(tickers)
        
        if n == 0:
            return OptimizationResult(
                weights={},
                objective_value=0.0,
                status="no_securities",
                solver_name=self._solver.solver_name,
                expected_alpha=0.0,
                expected_risk=0.0,
            )
        
        # Build alpha vector
        alpha_vector = np.array([
            optimization_input.alpha_scores.get(t, 0.0) for t in tickers
        ])
        
        # Build covariance matrix
        cov_matrix = await self._risk_service.calculate_covariance_matrix(tickers, risk_model)
        
        # Build bounds
        lower_bounds = np.zeros(n) if parameters.long_only else np.full(n, -1.0)
        upper_bounds = np.full(n, parameters.max_weight)
        
        # Apply stock-level constraints from constraint_set
        for constraint in constraint_set.stock_constraints:
            if constraint.constraint_name in tickers:
                idx = tickers.index(constraint.constraint_name)
                benchmark_weight = optimization_input.benchmark_weights.get(
                    constraint.constraint_name, 0.0
                ) / 100.0  # Convert to decimal
                
                if constraint.is_relative:
                    new_lower = max(0, benchmark_weight + constraint.lower_bound_pct / 100.0)
                    new_upper = benchmark_weight + constraint.upper_bound_pct / 100.0
                    
                    # Ensure lower <= upper to avoid infeasibility
                    if new_lower <= new_upper:
                        lower_bounds[idx] = new_lower
                        upper_bounds[idx] = max(new_upper, new_lower)  # Ensure upper >= lower
        
        # Build sector constraints as linear constraints
        inequality_constraints = []
        sector_to_tickers: dict[str, list[int]] = {}
        
        # Get sector mapping
        for i, ticker in enumerate(tickers):
            # Get sector from risk model or constraint set
            loading = risk_model.get_loadings(ticker)
            if loading:
                sector = loading.gics_sector
                if sector not in sector_to_tickers:
                    sector_to_tickers[sector] = []
                sector_to_tickers[sector].append(i)
        
        # Add sector constraints (only if we have tickers in that sector)
        from app.solvers.interfaces.solver import ConstraintSpec
        for constraint in constraint_set.sector_constraints:
            sector = constraint.constraint_name
            if sector in sector_to_tickers and sector_to_tickers[sector]:
                indices = sector_to_tickers[sector]
                
                # Build constraint coefficients
                coeffs = np.zeros(n)
                for idx in indices:
                    coeffs[idx] = 1.0
                
                benchmark_sector_weight = max(0, constraint.benchmark_weight_pct / 100.0)
                
                if constraint.is_relative:
                    # Calculate bounds with safety checks
                    lower_pct = constraint.lower_bound_pct / 100.0
                    upper_pct = constraint.upper_bound_pct / 100.0
                    
                    lower_bound_val = max(0, benchmark_sector_weight + lower_pct)
                    upper_bound_val = benchmark_sector_weight + upper_pct
                    
                    # Only add constraints if they are feasible (upper >= lower)
                    if upper_bound_val >= lower_bound_val and upper_bound_val >= 0:
                        # Upper bound: sum(w) <= upper_bound_val
                        upper_constraint = ConstraintSpec(
                            name=f"{sector}_upper",
                            constraint_type="ineq",
                            coefficients=coeffs.tolist(),
                            bound=upper_bound_val,
                        )
                        inequality_constraints.append(upper_constraint)
                        
                        # Lower bound: -sum(w) <= -lower_bound_val (equivalent to sum(w) >= lower)
                        # Only add if lower > 0 to avoid redundant constraints
                        if lower_bound_val > 0:
                            lower_constraint = ConstraintSpec(
                                name=f"{sector}_lower",
                                constraint_type="ineq",
                                coefficients=(-coeffs).tolist(),
                                bound=-lower_bound_val,
                            )
                            inequality_constraints.append(lower_constraint)
        
        # Get initial weights for transaction costs
        initial_weights = None
        transaction_costs = None
        if optimization_input.current_weights and parameters.transaction_cost_penalty > 0:
            initial_weights = np.array([
                optimization_input.current_weights.get(t, 0.0) for t in tickers
            ])
            if transaction_cost_model:
                transaction_costs = np.array([
                    transaction_cost_model.get_cost(t).cost_medium_urgency_bps 
                    if transaction_cost_model.get_cost(t) else 10.0
                    for t in tickers
                ])
        
        # Run optimization
        solver_result = self._solver.solve(
            alpha_vector=alpha_vector,
            covariance_matrix=cov_matrix,
            tickers=tickers,
            risk_aversion=parameters.risk_aversion,
            lower_bounds=lower_bounds,
            upper_bounds=upper_bounds,
            inequality_constraints=inequality_constraints if inequality_constraints else None,
            initial_weights=initial_weights,
            transaction_costs=transaction_costs,
            transaction_cost_penalty=parameters.transaction_cost_penalty,
        )
        
        # Calculate portfolio metrics
        if solver_result.status == SolverStatus.OPTIMAL:
            weights = solver_result.weights
            
            # Expected alpha
            expected_alpha = sum(
                w * optimization_input.alpha_scores.get(t, 0.0)
                for t, w in weights.items()
            )
            
            # Expected risk
            risk_metrics = await self._risk_service.calculate_portfolio_risk(
                weights, risk_model
            )
            expected_risk = risk_metrics.total_risk_pct
        else:
            expected_alpha = 0.0
            expected_risk = 0.0
        
        return OptimizationResult(
            weights=solver_result.weights,
            objective_value=solver_result.objective_value,
            status=str(solver_result.status),  # Convert SolverStatus enum to string
            solver_name=solver_result.solver_name,
            expected_alpha=expected_alpha,
            expected_risk=expected_risk,
            iterations=solver_result.iterations,
            solve_time_seconds=solver_result.solve_time_seconds,
        )

    async def optimize_with_target_risk(
        self,
        optimization_input: OptimizationInput,
        risk_model: RiskModel,
        constraint_set: ConstraintSet,
        target_risk: float,
        parameters: OptimizationParameters,
    ) -> OptimizationResult:
        """Optimize portfolio with a target risk level."""
        # Binary search for risk aversion that achieves target risk
        low_lambda = 0.001
        high_lambda = 1.0
        tolerance = 0.5  # % tolerance
        
        best_result = None
        
        for _ in range(20):  # Max iterations
            mid_lambda = (low_lambda + high_lambda) / 2
            
            test_params = OptimizationParameters(
                risk_aversion=mid_lambda,
                alpha_coefficient=parameters.alpha_coefficient,
                transaction_cost_penalty=parameters.transaction_cost_penalty,
                max_portfolio_size=parameters.max_portfolio_size,
                min_weight=parameters.min_weight,
                max_weight=parameters.max_weight,
                long_only=parameters.long_only,
            )
            
            result = await self.optimize_portfolio(
                optimization_input, risk_model, constraint_set, test_params
            )
            
            if result.status != "optimal":
                break
            
            best_result = result
            
            if abs(result.expected_risk - target_risk) < tolerance:
                break
            elif result.expected_risk > target_risk:
                low_lambda = mid_lambda  # Increase risk aversion
            else:
                high_lambda = mid_lambda  # Decrease risk aversion
        
        return best_result or OptimizationResult(
            weights={},
            objective_value=0.0,
            status="failed",
            solver_name=self._solver.solver_name,
            expected_alpha=0.0,
            expected_risk=0.0,
        )

    async def optimize_tracking_error(
        self,
        optimization_input: OptimizationInput,
        risk_model: RiskModel,
        constraint_set: ConstraintSet,
        max_tracking_error: float,
        parameters: OptimizationParameters,
    ) -> OptimizationResult:
        """Optimize portfolio with tracking error constraint."""
        # This is a simplified implementation
        # A full implementation would add TE as a constraint
        return await self.optimize_portfolio(
            optimization_input, risk_model, constraint_set, parameters
        )

    async def calculate_efficient_frontier(
        self,
        optimization_input: OptimizationInput,
        risk_model: RiskModel,
        constraint_set: ConstraintSet,
        n_points: int = 20,
    ) -> list[tuple[float, float, dict[str, float]]]:
        """Calculate efficient frontier points."""
        frontier_points = []
        
        # Range of risk aversion values
        lambdas = np.logspace(-3, 1, n_points)
        
        for risk_aversion in lambdas:
            params = OptimizationParameters(
                risk_aversion=float(risk_aversion),
                long_only=True,
            )
            
            result = await self.optimize_portfolio(
                optimization_input, risk_model, constraint_set, params
            )
            
            if result.status == "optimal":
                frontier_points.append((
                    result.expected_risk,
                    result.expected_alpha,
                    result.weights,
                ))
        
        return frontier_points

    async def validate_weights(
        self,
        weights: dict[str, float],
        constraint_set: ConstraintSet,
        benchmark_weights: dict[str, float],
    ) -> dict[str, Any]:
        """Validate weights against constraints."""
        violations = []
        
        # Check sum to 1
        total = sum(weights.values())
        if abs(total - 1.0) > 0.01:
            violations.append({
                "type": "budget",
                "message": f"Weights sum to {total:.4f}, not 1.0",
            })
        
        # Check stock constraints
        for constraint in constraint_set.stock_constraints:
            ticker = constraint.constraint_name
            weight = weights.get(ticker, 0.0) * 100  # Convert to %
            benchmark = benchmark_weights.get(ticker, 0.0)
            
            min_allowed = benchmark + constraint.lower_bound_pct
            max_allowed = benchmark + constraint.upper_bound_pct
            
            if weight < min_allowed - 0.01 or weight > max_allowed + 0.01:
                violations.append({
                    "type": "stock",
                    "ticker": ticker,
                    "weight": weight,
                    "allowed_range": (min_allowed, max_allowed),
                })
        
        return {
            "is_valid": len(violations) == 0,
            "violations": violations,
        }

