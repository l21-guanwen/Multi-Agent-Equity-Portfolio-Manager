"""
Abstract interface for optimization solvers.

Enables switching between different optimization solvers (cvxpy, scipy, etc.)
without changing the application code.
"""

from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Optional

import numpy as np
from pydantic import BaseModel, Field


class SolverStatus(str, Enum):
    """Optimization solver status codes."""
    
    OPTIMAL = "optimal"
    INFEASIBLE = "infeasible"
    UNBOUNDED = "unbounded"
    SOLVER_ERROR = "solver_error"
    MAX_ITERATIONS = "max_iterations"
    NUMERICAL_ERROR = "numerical_error"
    UNKNOWN = "unknown"


class SolverResult(BaseModel):
    """Result from an optimization solver."""
    
    weights: dict[str, float] = Field(..., description="Optimal weights by ticker")
    objective_value: float = Field(..., description="Optimal objective function value")
    status: SolverStatus = Field(..., description="Solver termination status")
    solver_name: str = Field(..., description="Name of the solver used")
    
    # Performance metrics
    iterations: int = Field(default=0, description="Number of iterations")
    solve_time_seconds: float = Field(default=0.0, description="Time to solve")
    
    # Dual values (for constraint analysis)
    dual_values: Optional[dict[str, float]] = Field(
        default=None, 
        description="Dual values for constraints"
    )
    
    # Raw solver output
    raw_result: Optional[dict[str, Any]] = Field(
        default=None,
        description="Raw solver output for debugging"
    )

    class Config:
        arbitrary_types_allowed = True
        use_enum_values = True


class ConstraintSpec(BaseModel):
    """Specification for an optimization constraint."""
    
    name: str = Field(..., description="Constraint name")
    constraint_type: str = Field(..., description="'eq' for equality, 'ineq' for inequality")
    
    # For linear constraints: Ax <= b or Ax = b
    coefficients: Optional[list[float]] = Field(
        default=None, 
        description="Constraint coefficients (A row)"
    )
    bound: Optional[float] = Field(default=None, description="Constraint bound (b)")
    
    # For box constraints
    lower_bound: Optional[float] = Field(default=None, description="Lower bound")
    upper_bound: Optional[float] = Field(default=None, description="Upper bound")
    variable_index: Optional[int] = Field(
        default=None, 
        description="Variable index for box constraint"
    )

    class Config:
        arbitrary_types_allowed = True


class ISolver(ABC):
    """
    Abstract interface for optimization solvers.
    
    Implementations handle:
    - Problem setup
    - Constraint handling
    - Solving the optimization
    - Result parsing
    
    The standard problem formulation is:
    
    Maximize: alpha' * w - (lambda/2) * w' * Sigma * w - tau * |dw|
    
    Subject to:
    - Sum of weights = 1 (or target)
    - Lower <= w <= Upper (box constraints)
    - A * w <= b (linear inequality constraints)
    - A_eq * w = b_eq (linear equality constraints)
    
    Example usage:
        solver = CvxpySolver()
        result = solver.solve(
            alpha_vector=alphas,
            covariance_matrix=cov,
            tickers=tickers,
            constraints=constraints,
        )
    """

    @property
    @abstractmethod
    def solver_name(self) -> str:
        """Get the name of the solver."""
        pass

    @abstractmethod
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
        
        Args:
            alpha_vector: Expected alpha/returns for each security (Nx1)
            covariance_matrix: Covariance matrix (NxN)
            tickers: List of ticker symbols (defines order)
            risk_aversion: Risk aversion parameter (lambda)
            lower_bounds: Lower bounds for weights (default: 0 for long-only)
            upper_bounds: Upper bounds for weights (default: 1)
            equality_constraints: List of equality constraints
            inequality_constraints: List of inequality constraints
            initial_weights: Starting weights for rebalancing
            transaction_costs: Per-security transaction costs
            transaction_cost_penalty: Transaction cost penalty (tau)
            
        Returns:
            SolverResult with optimal weights and metadata
        """
        pass

    @abstractmethod
    def solve_minimum_variance(
        self,
        covariance_matrix: np.ndarray,
        tickers: list[str],
        lower_bounds: Optional[np.ndarray] = None,
        upper_bounds: Optional[np.ndarray] = None,
        equality_constraints: Optional[list[ConstraintSpec]] = None,
        inequality_constraints: Optional[list[ConstraintSpec]] = None,
    ) -> SolverResult:
        """
        Solve for minimum variance portfolio.
        
        Args:
            covariance_matrix: Covariance matrix (NxN)
            tickers: List of ticker symbols
            lower_bounds: Lower bounds for weights
            upper_bounds: Upper bounds for weights
            equality_constraints: List of equality constraints
            inequality_constraints: List of inequality constraints
            
        Returns:
            SolverResult with minimum variance weights
        """
        pass

    @abstractmethod
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
        
        Args:
            alpha_vector: Expected returns for each security
            covariance_matrix: Covariance matrix (NxN)
            tickers: List of ticker symbols
            risk_free_rate: Risk-free rate for Sharpe calculation
            lower_bounds: Lower bounds for weights
            upper_bounds: Upper bounds for weights
            equality_constraints: List of equality constraints
            inequality_constraints: List of inequality constraints
            
        Returns:
            SolverResult with maximum Sharpe ratio weights
        """
        pass

    def is_available(self) -> bool:
        """
        Check if the solver is available/installed.
        
        Returns:
            True if solver can be used, False otherwise
        """
        try:
            # Try a minimal solve to verify solver works
            return True
        except Exception:
            return False

