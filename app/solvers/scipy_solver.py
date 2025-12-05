"""
SciPy-based optimization solver implementation.

Uses scipy.optimize for portfolio optimization as an alternative to CVXPY.
"""

import time
from typing import Optional

import numpy as np
from scipy.optimize import minimize, Bounds, LinearConstraint

from app.solvers.interfaces.solver import (
    ConstraintSpec,
    ISolver,
    SolverResult,
    SolverStatus,
)


class ScipySolver(ISolver):
    """
    SciPy implementation of the portfolio optimization solver.
    
    Uses scipy.optimize.minimize with SLSQP method for constrained optimization.
    Provides an alternative to CVXPY for environments where CVXPY is not available.
    
    Example:
        solver = ScipySolver()
        result = solver.solve(
            alpha_vector=alphas,
            covariance_matrix=cov,
            tickers=tickers,
            risk_aversion=0.01,
        )
    """

    def __init__(self, method: str = "SLSQP", max_iterations: int = 1000):
        """
        Initialize SciPy solver.
        
        Args:
            method: Optimization method ('SLSQP', 'trust-constr')
            max_iterations: Maximum number of iterations
        """
        self._method = method
        self._max_iterations = max_iterations

    @property
    def solver_name(self) -> str:
        """Get the name of the solver."""
        return "scipy"

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
        
        Objective: Minimize -alpha'w + (lambda/2) * w'Σw + tau * |w - w0|
        (Negated for minimization)
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
        
        # Initial guess (equal weight or provided)
        if initial_weights is None:
            x0 = np.ones(n) / n
        else:
            x0 = initial_weights.copy()
        
        # Objective function (minimization, so negate alpha term)
        def objective(w):
            alpha_term = -np.dot(alpha_vector, w)  # Negate for minimization
            risk_term = (risk_aversion / 2) * np.dot(w, np.dot(covariance_matrix, w))
            
            # Transaction cost term
            tcost_term = 0
            if initial_weights is not None and transaction_cost_penalty > 0:
                if transaction_costs is not None:
                    tcost_term = transaction_cost_penalty * np.sum(
                        np.abs(transaction_costs * (w - initial_weights))
                    )
                else:
                    tcost_term = transaction_cost_penalty * np.sum(np.abs(w - initial_weights))
            
            return alpha_term + risk_term + tcost_term
        
        # Gradient of objective
        def gradient(w):
            grad = -alpha_vector + risk_aversion * np.dot(covariance_matrix, w)
            return grad
        
        # Constraints
        constraints = []
        
        # Budget constraint: sum(w) = 1
        constraints.append({
            'type': 'eq',
            'fun': lambda w: np.sum(w) - 1,
            'jac': lambda w: np.ones(n),
        })
        
        # Add custom equality constraints
        if equality_constraints:
            for constraint in equality_constraints:
                if constraint.coefficients is not None and constraint.bound is not None:
                    coeffs = np.array(constraint.coefficients)
                    bound = constraint.bound
                    constraints.append({
                        'type': 'eq',
                        'fun': lambda w, c=coeffs, b=bound: np.dot(c, w) - b,
                        'jac': lambda w, c=coeffs: c,
                    })
        
        # Add custom inequality constraints (<=)
        if inequality_constraints:
            for constraint in inequality_constraints:
                if constraint.coefficients is not None and constraint.bound is not None:
                    coeffs = np.array(constraint.coefficients)
                    bound = constraint.bound
                    # scipy uses >= 0, so we need: bound - coeffs @ w >= 0
                    constraints.append({
                        'type': 'ineq',
                        'fun': lambda w, c=coeffs, b=bound: b - np.dot(c, w),
                        'jac': lambda w, c=coeffs: -c,
                    })
        
        # Bounds
        bounds = Bounds(lower_bounds, upper_bounds)
        
        # Solve
        try:
            result = minimize(
                objective,
                x0,
                method=self._method,
                jac=gradient,
                bounds=bounds,
                constraints=constraints,
                options={'maxiter': self._max_iterations, 'disp': False},
            )
            
            solve_time = time.time() - start_time
            
            # Map status
            if result.success:
                status = SolverStatus.OPTIMAL
            elif "iteration" in result.message.lower():
                status = SolverStatus.MAX_ITERATIONS
            else:
                status = SolverStatus.INFEASIBLE
            
            # Extract weights
            weights_array = result.x
            # Clean up small values
            weights_array = np.where(np.abs(weights_array) < 1e-6, 0, weights_array)
            # Normalize
            if np.sum(weights_array) > 1e-8:
                weights_array = weights_array / np.sum(weights_array)
            weights_dict = dict(zip(tickers, weights_array.tolist()))
            
            return SolverResult(
                weights=weights_dict,
                objective_value=-result.fun,  # Negate back for maximization
                status=status,
                solver_name=self.solver_name,
                iterations=result.nit,
                solve_time_seconds=solve_time,
                raw_result={
                    "scipy_success": result.success,
                    "scipy_message": result.message,
                    "scipy_nfev": result.nfev,
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
        n = len(tickers)
        alpha_vector = np.zeros(n)
        
        return self.solve(
            alpha_vector=alpha_vector,
            covariance_matrix=covariance_matrix,
            tickers=tickers,
            risk_aversion=1.0,
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
        """Solve for maximum Sharpe ratio portfolio."""
        start_time = time.time()
        n = len(tickers)
        
        # Adjust alpha for risk-free rate
        excess_alpha = alpha_vector - risk_free_rate
        
        # Set default bounds
        if lower_bounds is None:
            lower_bounds = np.zeros(n)
        if upper_bounds is None:
            upper_bounds = np.ones(n)
        
        # Initial guess
        x0 = np.ones(n) / n
        
        # Objective: minimize -Sharpe = -(alpha-rf)'w / sqrt(w'Σw)
        # Use negative Sharpe for minimization
        def neg_sharpe(w):
            ret = np.dot(excess_alpha, w)
            vol = np.sqrt(np.dot(w, np.dot(covariance_matrix, w)))
            if vol < 1e-10:
                return 1e10  # Large penalty for zero volatility
            return -ret / vol
        
        # Constraints
        constraints = [{
            'type': 'eq',
            'fun': lambda w: np.sum(w) - 1,
        }]
        
        bounds = Bounds(lower_bounds, upper_bounds)
        
        try:
            result = minimize(
                neg_sharpe,
                x0,
                method=self._method,
                bounds=bounds,
                constraints=constraints,
                options={'maxiter': self._max_iterations},
            )
            
            solve_time = time.time() - start_time
            status = SolverStatus.OPTIMAL if result.success else SolverStatus.INFEASIBLE
            
            weights_array = result.x
            weights_array = np.where(np.abs(weights_array) < 1e-6, 0, weights_array)
            if np.sum(weights_array) > 1e-8:
                weights_array = weights_array / np.sum(weights_array)
            weights_dict = dict(zip(tickers, weights_array.tolist()))
            
            return SolverResult(
                weights=weights_dict,
                objective_value=-result.fun,  # Return positive Sharpe
                status=status,
                solver_name=self.solver_name,
                iterations=result.nit,
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

    def is_available(self) -> bool:
        """Check if SciPy is available."""
        try:
            from scipy.optimize import minimize
            return True
        except ImportError:
            return False

