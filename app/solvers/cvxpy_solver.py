"""
CVXPY-based optimization solver implementation.

Uses CVXPY for convex portfolio optimization with constraints.
"""

import time
from typing import Optional

import cvxpy as cp
import numpy as np

from app.solvers.interfaces.solver import (
    ConstraintSpec,
    ISolver,
    SolverResult,
    SolverStatus,
)


class CvxpySolver(ISolver):
    """
    CVXPY implementation of the portfolio optimization solver.
    
    Solves mean-variance optimization problems of the form:
    
    Maximize: alpha' * w - (lambda/2) * w' * Sigma * w - tau * |dw|
    
    Subject to:
    - Sum of weights = 1
    - Lower <= w <= Upper (box constraints)
    - Linear constraints (sector limits, etc.)
    
    Example:
        solver = CvxpySolver()
        result = solver.solve(
            alpha_vector=alphas,
            covariance_matrix=cov,
            tickers=tickers,
            risk_aversion=0.01,
        )
    """

    def __init__(self, solver_backend: Optional[str] = None):
        """
        Initialize CVXPY solver.
        
        Args:
            solver_backend: Optional CVXPY solver backend (ECOS, OSQP, SCS, etc.)
                          Uses CVXPY default if not specified.
        """
        self._solver_backend = solver_backend

    @property
    def solver_name(self) -> str:
        """Get the name of the solver."""
        return "cvxpy"

    def solve(
        self,
        alpha_vector: np.ndarray,
        covariance_matrix: np.ndarray,
        tickers: list[str],
        risk_aversion: float = 0.01,
        lower_bounds: Optional[np.ndarray] = None,
        upper_bounds: Optional[np.ndarray] = None,
        equality_constraints: Optional[list[ConstraintSpec]] = None,
        inequality_constraints: Optional[list[ConstraintSpec]] = None,
        initial_weights: Optional[np.ndarray] = None,
        transaction_costs: Optional[np.ndarray] = None,
        transaction_cost_penalty: float = 0.0,
    ) -> SolverResult:
        """
        Solve the portfolio optimization problem.
        
        Objective: Maximize alpha'w - (lambda/2) * w'Σw - tau * |w - w0|
        """
        start_time = time.time()
        n = len(tickers)
        
        # Validate inputs
        if len(alpha_vector) != n:
            raise ValueError(f"Alpha vector length {len(alpha_vector)} != {n} tickers")
        if covariance_matrix.shape != (n, n):
            raise ValueError(f"Covariance matrix shape {covariance_matrix.shape} != ({n}, {n})")
        
        # Set default bounds (long-only)
        if lower_bounds is None:
            lower_bounds = np.zeros(n)
        if upper_bounds is None:
            upper_bounds = np.ones(n)
        
        # Decision variable
        w = cp.Variable(n)
        
        # Objective function components
        # Alpha term (maximize)
        alpha_term = alpha_vector @ w
        
        # Risk term (minimize variance)
        risk_term = cp.quad_form(w, covariance_matrix)
        
        # Transaction cost term (if rebalancing)
        tcost_term = 0
        if initial_weights is not None and transaction_cost_penalty > 0:
            if transaction_costs is None:
                transaction_costs = np.ones(n) * 10  # Default 10 bps
            # Convert transaction costs from bps to decimal (100 bps = 1% = 0.01)
            # Then scale by penalty parameter
            transaction_costs_decimal = transaction_costs / 10000.0  # bps to decimal
            # Approximate |dw| with linear cost: penalty * sum(|cost_i * dw_i|)
            dw = w - initial_weights
            tcost_term = transaction_cost_penalty * cp.norm1(cp.multiply(transaction_costs_decimal, dw))
        
        # Full objective: maximize alpha - risk - tcost
        objective = cp.Maximize(alpha_term - (risk_aversion / 2) * risk_term - tcost_term)
        
        # Constraints
        constraints = []
        
        # Budget constraint: weights sum to 1
        constraints.append(cp.sum(w) == 1)
        
        # Box constraints
        constraints.append(w >= lower_bounds)
        constraints.append(w <= upper_bounds)
        
        # Add equality constraints
        if equality_constraints:
            for constraint in equality_constraints:
                if constraint.coefficients is not None and constraint.bound is not None:
                    coeffs = np.array(constraint.coefficients)
                    constraints.append(coeffs @ w == constraint.bound)
        
        # Add inequality constraints
        if inequality_constraints:
            for constraint in inequality_constraints:
                if constraint.coefficients is not None and constraint.bound is not None:
                    coeffs = np.array(constraint.coefficients)
                    constraints.append(coeffs @ w <= constraint.bound)
        
        # Create and solve problem
        problem = cp.Problem(objective, constraints)
        
        try:
            if self._solver_backend:
                problem.solve(solver=self._solver_backend)
            else:
                problem.solve()
            
            solve_time = time.time() - start_time
            
            # Map CVXPY status to our status
            status = self._map_status(problem.status)
            
            # Extract weights
            if status == SolverStatus.OPTIMAL and w.value is not None:
                weights_array = w.value
                # Clean up small values
                weights_array = np.where(np.abs(weights_array) < 1e-6, 0, weights_array)
                # Normalize to sum to 1
                weights_array = weights_array / np.sum(weights_array)
                weights_dict = dict(zip(tickers, weights_array.tolist()))
            else:
                weights_dict = {t: 0.0 for t in tickers}
            
            return SolverResult(
                weights=weights_dict,
                objective_value=problem.value if problem.value is not None else 0.0,
                status=status,
                solver_name=self.solver_name,
                iterations=problem.solver_stats.num_iters if problem.solver_stats else 0,
                solve_time_seconds=solve_time,
                raw_result={
                    "cvxpy_status": problem.status,
                    "solver_stats": str(problem.solver_stats) if problem.solver_stats else None,
                },
            )
            
        except Exception as e:
            return SolverResult(
                weights={t: 0.0 for t in tickers},
                objective_value=0.0,
                status=SolverStatus.SOLVER_ERROR,
                solver_name=self.solver_name,
                solve_time_seconds=time.time() - start_time,
                raw_result={"error": str(e)},
            )

    def solve_minimum_variance(
        self,
        covariance_matrix: np.ndarray,
        tickers: list[str],
        lower_bounds: Optional[np.ndarray] = None,
        upper_bounds: Optional[np.ndarray] = None,
        equality_constraints: Optional[list[ConstraintSpec]] = None,
        inequality_constraints: Optional[list[ConstraintSpec]] = None,
    ) -> SolverResult:
        """Solve for minimum variance portfolio."""
        # Minimum variance = no alpha, high risk aversion
        n = len(tickers)
        alpha_vector = np.zeros(n)
        
        return self.solve(
            alpha_vector=alpha_vector,
            covariance_matrix=covariance_matrix,
            tickers=tickers,
            risk_aversion=1.0,  # Pure variance minimization
            lower_bounds=lower_bounds,
            upper_bounds=upper_bounds,
            equality_constraints=equality_constraints,
            inequality_constraints=inequality_constraints,
        )

    def solve_maximum_sharpe(
        self,
        alpha_vector: np.ndarray,
        covariance_matrix: np.ndarray,
        tickers: list[str],
        risk_free_rate: float = 0.0,
        lower_bounds: Optional[np.ndarray] = None,
        upper_bounds: Optional[np.ndarray] = None,
        equality_constraints: Optional[list[ConstraintSpec]] = None,
        inequality_constraints: Optional[list[ConstraintSpec]] = None,
    ) -> SolverResult:
        """
        Solve for maximum Sharpe ratio portfolio.
        
        Uses the transformation method: maximize (alpha - rf)'w / sqrt(w'Σw)
        Reformulated as a convex problem.
        """
        start_time = time.time()
        n = len(tickers)
        
        # Adjust alpha for risk-free rate
        excess_alpha = alpha_vector - risk_free_rate
        
        # Set default bounds
        if lower_bounds is None:
            lower_bounds = np.zeros(n)
        if upper_bounds is None:
            upper_bounds = np.ones(n)
        
        # Use transformation: y = w/k where k = 1/(alpha-rf)'w
        # Minimize y'Σy subject to (alpha-rf)'y = 1, sum(y) >= 0
        y = cp.Variable(n)
        
        objective = cp.Minimize(cp.quad_form(y, covariance_matrix))
        
        constraints = [
            excess_alpha @ y == 1,  # Normalization
            y >= 0,  # Long only (scaled)
        ]
        
        problem = cp.Problem(objective, constraints)
        
        try:
            problem.solve()
            
            solve_time = time.time() - start_time
            status = self._map_status(problem.status)
            
            if status == SolverStatus.OPTIMAL and y.value is not None:
                # Transform back to weights
                y_val = y.value
                k = np.sum(y_val)
                if k > 1e-8:
                    weights_array = y_val / k
                    # Apply bounds
                    weights_array = np.clip(weights_array, lower_bounds, upper_bounds)
                    # Renormalize
                    weights_array = weights_array / np.sum(weights_array)
                    weights_dict = dict(zip(tickers, weights_array.tolist()))
                else:
                    weights_dict = {t: 1.0/n for t in tickers}  # Equal weight fallback
            else:
                weights_dict = {t: 1.0/n for t in tickers}
            
            return SolverResult(
                weights=weights_dict,
                objective_value=problem.value if problem.value is not None else 0.0,
                status=status,
                solver_name=self.solver_name,
                iterations=problem.solver_stats.num_iters if problem.solver_stats else 0,
                solve_time_seconds=solve_time,
            )
            
        except Exception as e:
            return SolverResult(
                weights={t: 1.0/n for t in tickers},
                objective_value=0.0,
                status=SolverStatus.SOLVER_ERROR,
                solver_name=self.solver_name,
                solve_time_seconds=time.time() - start_time,
                raw_result={"error": str(e)},
            )

    def _map_status(self, cvxpy_status: str) -> SolverStatus:
        """Map CVXPY status to our status enum."""
        status_map = {
            cp.OPTIMAL: SolverStatus.OPTIMAL,
            cp.OPTIMAL_INACCURATE: SolverStatus.OPTIMAL,
            cp.INFEASIBLE: SolverStatus.INFEASIBLE,
            cp.INFEASIBLE_INACCURATE: SolverStatus.INFEASIBLE,
            cp.UNBOUNDED: SolverStatus.UNBOUNDED,
            cp.UNBOUNDED_INACCURATE: SolverStatus.UNBOUNDED,
            cp.SOLVER_ERROR: SolverStatus.SOLVER_ERROR,
        }
        return status_map.get(cvxpy_status, SolverStatus.UNKNOWN)

    def is_available(self) -> bool:
        """Check if CVXPY is available."""
        try:
            import cvxpy
            return True
        except ImportError:
            return False

