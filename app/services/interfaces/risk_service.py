"""
Risk service interface.

Defines operations for risk analysis and calculations.
"""

from abc import ABC, abstractmethod
from typing import Optional

import numpy as np
from pydantic import BaseModel

from app.models.risk import FactorCovariance, FactorLoading, RiskModel


class PortfolioRiskMetrics(BaseModel):
    """Risk metrics for a portfolio."""
    
    total_risk_pct: float  # Total portfolio volatility
    systematic_risk_pct: float  # Factor-driven risk
    specific_risk_pct: float  # Idiosyncratic risk
    tracking_error_pct: Optional[float] = None  # vs benchmark
    beta: float  # Portfolio beta to market
    
    # Factor contributions
    factor_risk_contributions: dict[str, float]
    
    class Config:
        arbitrary_types_allowed = True


class FactorExposure(BaseModel):
    """Portfolio-level factor exposures."""
    
    market: float
    size: float
    value: float
    momentum: float
    quality: float
    volatility: float
    growth: float
    dividend_yield: float
    
    def to_dict(self) -> dict[str, float]:
        return {
            "Market": self.market,
            "Size": self.size,
            "Value": self.value,
            "Momentum": self.momentum,
            "Quality": self.quality,
            "Volatility": self.volatility,
            "Growth": self.growth,
            "Dividend_Yield": self.dividend_yield,
        }
    
    def to_array(self) -> np.ndarray:
        return np.array([
            self.market, self.size, self.value, self.momentum,
            self.quality, self.volatility, self.growth, self.dividend_yield
        ])

    class Config:
        arbitrary_types_allowed = True


class IRiskService(ABC):
    """
    Service interface for risk model operations.
    
    Handles factor exposure calculations, portfolio risk decomposition,
    and risk-related analytics for portfolio construction.
    """

    @abstractmethod
    async def calculate_portfolio_risk(
        self,
        weights: dict[str, float],
        risk_model: RiskModel,
    ) -> PortfolioRiskMetrics:
        """
        Calculate comprehensive risk metrics for a portfolio.
        
        Args:
            weights: Dictionary of ticker -> weight (0-1)
            risk_model: RiskModel with loadings and covariance
            
        Returns:
            PortfolioRiskMetrics with total, systematic, and specific risk
        """
        pass

    @abstractmethod
    async def calculate_factor_exposure(
        self,
        weights: dict[str, float],
        risk_model: RiskModel,
    ) -> FactorExposure:
        """
        Calculate portfolio-level factor exposures.
        
        Args:
            weights: Dictionary of ticker -> weight (0-1)
            risk_model: RiskModel with factor loadings
            
        Returns:
            FactorExposure with weighted average factor loadings
        """
        pass

    @abstractmethod
    async def calculate_tracking_error(
        self,
        portfolio_weights: dict[str, float],
        benchmark_weights: dict[str, float],
        risk_model: RiskModel,
    ) -> float:
        """
        Calculate tracking error vs benchmark.
        
        Args:
            portfolio_weights: Dictionary of ticker -> portfolio weight
            benchmark_weights: Dictionary of ticker -> benchmark weight
            risk_model: RiskModel with loadings and covariance
            
        Returns:
            Tracking error as percentage
        """
        pass

    @abstractmethod
    async def calculate_covariance_matrix(
        self,
        tickers: list[str],
        risk_model: RiskModel,
    ) -> np.ndarray:
        """
        Calculate security-level covariance matrix.
        
        Uses factor model: Cov = B * F * B' + D
        where B = factor loadings, F = factor covariance, D = specific variance
        
        Args:
            tickers: List of tickers (defines matrix order)
            risk_model: RiskModel with loadings and covariance
            
        Returns:
            NxN covariance matrix as numpy array
        """
        pass

    @abstractmethod
    async def decompose_risk(
        self,
        weights: dict[str, float],
        risk_model: RiskModel,
    ) -> dict[str, float]:
        """
        Decompose portfolio risk by factor.
        
        Args:
            weights: Dictionary of ticker -> weight (0-1)
            risk_model: RiskModel with loadings and covariance
            
        Returns:
            Dictionary of factor -> risk contribution percentage
        """
        pass

    @abstractmethod
    async def get_risk_contributors(
        self,
        weights: dict[str, float],
        risk_model: RiskModel,
        top_n: int = 10,
    ) -> list[tuple[str, float]]:
        """
        Get top risk-contributing securities.
        
        Args:
            weights: Dictionary of ticker -> weight (0-1)
            risk_model: RiskModel with loadings and covariance
            top_n: Number of top contributors to return
            
        Returns:
            List of (ticker, risk_contribution) tuples
        """
        pass

    @abstractmethod
    async def calculate_marginal_risk(
        self,
        ticker: str,
        current_weights: dict[str, float],
        risk_model: RiskModel,
    ) -> float:
        """
        Calculate marginal risk contribution of adding a security.
        
        Args:
            ticker: Ticker to calculate marginal risk for
            current_weights: Current portfolio weights
            risk_model: RiskModel with loadings and covariance
            
        Returns:
            Marginal risk contribution
        """
        pass

