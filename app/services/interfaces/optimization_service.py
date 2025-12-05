"""
Optimization service interface.

Defines operations for portfolio optimization.
"""

from abc import ABC, abstractmethod
from typing import Any, Optional

import numpy as np
from pydantic import BaseModel

from app.models.alpha import AlphaModel
from app.models.constraint import ConstraintSet
from app.models.risk import RiskModel
from app.models.transaction_cost import TransactionCostModel


class OptimizationParameters(BaseModel):
    """Parameters for portfolio optimization."""
    
    risk_aversion: float = 0.01  # Lambda parameter
    alpha_coefficient: float = 1.0  # Alpha term weight
    transaction_cost_penalty: float = 0.0  # Tau parameter
    max_portfolio_size: int = 25
    min_weight: float = 0.0  # Minimum position weight
    max_weight: float = 1.0  # Maximum position weight
    long_only: bool = True


class OptimizationResult(BaseModel):
    """Result of portfolio optimization."""
    
    weights: dict[str, float]  # Ticker -> optimal weight
    objective_value: float
    status: str  # 'optimal', 'infeasible', 'unbounded', etc.
    solver_name: str
    
    # Portfolio characteristics
    expected_alpha: float
    expected_risk: float
    expected_return: Optional[float] = None
    
    # Constraint satisfaction
    active_constraints: list[str] = []
    constraint_violations: list[dict[str, Any]] = []
    
    # Metadata
    iterations: int = 0
    solve_time_seconds: float = 0.0

    class Config:
        arbitrary_types_allowed = True


class OptimizationInput(BaseModel):
    """Input data for optimization."""
    
    eligible_tickers: list[str]
    alpha_scores: dict[str, float]  # Ticker -> alpha score
    benchmark_weights: dict[str, float]  # Ticker -> benchmark weight
    current_weights: Optional[dict[str, float]] = None  # For rebalancing
    
    class Config:
        arbitrary_types_allowed = True


class IOptimizationService(ABC):
    """
    Service interface for portfolio optimization.
    
    Handles mean-variance optimization with constraints
    for portfolio construction.
    """

    @abstractmethod
    async def optimize_portfolio(
        self,
        optimization_input: OptimizationInput,
        risk_model: RiskModel,
        constraint_set: ConstraintSet,
        parameters: OptimizationParameters,
        transaction_cost_model: Optional[TransactionCostModel] = None,
    ) -> OptimizationResult:
        """
        Run portfolio optimization.
        
        Objective: Maximize α×Alpha - λ×Risk - τ×TCost
        
        Args:
            optimization_input: Securities and alpha scores
            risk_model: RiskModel for risk calculation
            constraint_set: Stock and sector constraints
            parameters: Optimization parameters
            transaction_cost_model: Optional tcost model for rebalancing
            
        Returns:
            OptimizationResult with optimal weights and metadata
        """
        pass

    @abstractmethod
    async def optimize_with_target_risk(
        self,
        optimization_input: OptimizationInput,
        risk_model: RiskModel,
        constraint_set: ConstraintSet,
        target_risk: float,
        parameters: OptimizationParameters,
    ) -> OptimizationResult:
        """
        Optimize portfolio with a target risk level.
        
        Args:
            optimization_input: Securities and alpha scores
            risk_model: RiskModel for risk calculation
            constraint_set: Stock and sector constraints
            target_risk: Target portfolio volatility (%)
            parameters: Optimization parameters
            
        Returns:
            OptimizationResult with optimal weights
        """
        pass

    @abstractmethod
    async def optimize_tracking_error(
        self,
        optimization_input: OptimizationInput,
        risk_model: RiskModel,
        constraint_set: ConstraintSet,
        max_tracking_error: float,
        parameters: OptimizationParameters,
    ) -> OptimizationResult:
        """
        Optimize portfolio with tracking error constraint.
        
        Args:
            optimization_input: Securities and alpha scores
            risk_model: RiskModel for risk calculation
            constraint_set: Stock and sector constraints
            max_tracking_error: Maximum tracking error vs benchmark (%)
            parameters: Optimization parameters
            
        Returns:
            OptimizationResult with optimal weights
        """
        pass

    @abstractmethod
    async def calculate_efficient_frontier(
        self,
        optimization_input: OptimizationInput,
        risk_model: RiskModel,
        constraint_set: ConstraintSet,
        n_points: int = 20,
    ) -> list[tuple[float, float, dict[str, float]]]:
        """
        Calculate efficient frontier points.
        
        Args:
            optimization_input: Securities and alpha scores
            risk_model: RiskModel for risk calculation
            constraint_set: Stock and sector constraints
            n_points: Number of points on the frontier
            
        Returns:
            List of (risk, return, weights) tuples
        """
        pass

    @abstractmethod
    async def validate_weights(
        self,
        weights: dict[str, float],
        constraint_set: ConstraintSet,
        benchmark_weights: dict[str, float],
    ) -> dict[str, Any]:
        """
        Validate weights against constraints.
        
        Args:
            weights: Proposed portfolio weights
            constraint_set: Stock and sector constraints
            benchmark_weights: Benchmark weights for relative constraints
            
        Returns:
            Validation result with any violations
        """
        pass

