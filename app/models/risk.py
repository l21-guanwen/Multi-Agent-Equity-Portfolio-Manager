"""Risk model domain models."""

from datetime import date
from typing import Optional

import numpy as np
from pydantic import BaseModel, Field, computed_field


class FactorLoading(BaseModel):
    """
    Factor loadings for a single security.
    
    Based on 05_Risk_Model_Factor_Loadings.csv schema.
    Represents exposure to 8 Barra-style risk factors.
    """

    # Security Info
    ticker: str = Field(..., description="Stock ticker symbol")
    security_name: str = Field(..., description="Full security name")
    gics_sector: str = Field(..., description="GICS Level 1 sector")
    
    # Specific Risk
    specific_risk_pct: float = Field(..., ge=0, description="Idiosyncratic risk (%)")
    
    # Factor Loadings
    market_loading: float = Field(..., description="Market factor loading")
    size_loading: float = Field(..., description="Size factor loading")
    value_loading: float = Field(..., description="Value factor loading")
    momentum_loading: float = Field(..., description="Momentum factor loading")
    quality_loading: float = Field(..., description="Quality factor loading")
    volatility_loading: float = Field(..., description="Volatility factor loading")
    growth_loading: float = Field(..., description="Growth factor loading")
    dividend_yield_loading: float = Field(..., description="Dividend yield factor loading")
    
    # Model Info
    risk_model_id: str = Field(..., description="Risk model identifier")
    as_of_date: date = Field(..., description="Data as-of date")

    def get_loadings_array(self) -> np.ndarray:
        """Get factor loadings as numpy array."""
        return np.array([
            self.market_loading,
            self.size_loading,
            self.value_loading,
            self.momentum_loading,
            self.quality_loading,
            self.volatility_loading,
            self.growth_loading,
            self.dividend_yield_loading,
        ])

    def get_loadings_dict(self) -> dict[str, float]:
        """Get factor loadings as dictionary."""
        return {
            "Market": self.market_loading,
            "Size": self.size_loading,
            "Value": self.value_loading,
            "Momentum": self.momentum_loading,
            "Quality": self.quality_loading,
            "Volatility": self.volatility_loading,
            "Growth": self.growth_loading,
            "Dividend_Yield": self.dividend_yield_loading,
        }

    class Config:
        """Pydantic model configuration."""
        
        frozen = False
        str_strip_whitespace = True
        arbitrary_types_allowed = True


class FactorReturn(BaseModel):
    """
    Historical returns for a single risk factor.
    
    Based on 06_Risk_Model_Factor_Returns.csv schema.
    """

    factor_name: str = Field(..., description="Factor name")
    factor_return_mtd_pct: Optional[float] = Field(None, description="Month-to-date return (%)")
    factor_return_qtd_pct: Optional[float] = Field(None, description="Quarter-to-date return (%)")
    factor_return_ytd_pct: Optional[float] = Field(None, description="Year-to-date return (%)")
    factor_return_1y_pct: Optional[float] = Field(None, description="1-year return (%)")
    factor_volatility_pct: Optional[float] = Field(None, ge=0, description="Factor volatility (%)")
    factor_sharpe: Optional[float] = Field(None, description="Factor Sharpe ratio")
    
    # Model Info
    risk_model_id: str = Field(..., description="Risk model identifier")
    as_of_date: date = Field(..., description="Data as-of date")


class FactorCovariance(BaseModel):
    """
    Factor covariance matrix.
    
    Based on 07_Risk_Model_Factor_Covariance.csv schema.
    8x8 symmetric positive semi-definite matrix.
    """

    # Factor names in order
    factors: list[str] = Field(
        default=[
            "Market", "Size", "Value", "Momentum",
            "Quality", "Volatility", "Growth", "Dividend_Yield"
        ],
        description="Factor names in matrix order"
    )
    
    # Covariance values (8x8 = 64 values, stored as nested list)
    matrix: list[list[float]] = Field(..., description="Covariance matrix values")
    
    # Model Info
    risk_model_id: str = Field(..., description="Risk model identifier")
    as_of_date: date = Field(..., description="Data as-of date")

    def get_matrix_array(self) -> np.ndarray:
        """Get covariance matrix as numpy array."""
        return np.array(self.matrix)

    def get_factor_variance(self, factor: str) -> float:
        """Get variance for a specific factor."""
        if factor not in self.factors:
            raise ValueError(f"Unknown factor: {factor}")
        idx = self.factors.index(factor)
        return self.matrix[idx][idx]

    def get_factor_covariance(self, factor1: str, factor2: str) -> float:
        """Get covariance between two factors."""
        if factor1 not in self.factors:
            raise ValueError(f"Unknown factor: {factor1}")
        if factor2 not in self.factors:
            raise ValueError(f"Unknown factor: {factor2}")
        idx1 = self.factors.index(factor1)
        idx2 = self.factors.index(factor2)
        return self.matrix[idx1][idx2]

    class Config:
        """Pydantic model configuration."""
        
        frozen = False
        arbitrary_types_allowed = True


class RiskModel(BaseModel):
    """
    Complete risk model containing factor loadings, returns, and covariance.
    
    Represents a Barra-style multi-factor risk model.
    """

    risk_model_id: str = Field(..., description="Risk model identifier")
    risk_model_name: str = Field(default="Barra-Style Multi-Factor Model", description="Model name")
    
    # Components
    factor_loadings: list[FactorLoading] = Field(default_factory=list)
    factor_returns: list[FactorReturn] = Field(default_factory=list)
    factor_covariance: Optional[FactorCovariance] = Field(None)
    
    as_of_date: date = Field(..., description="Model date")

    @computed_field
    @property
    def security_count(self) -> int:
        """Number of securities with factor loadings."""
        return len(self.factor_loadings)

    @computed_field
    @property
    def factor_count(self) -> int:
        """Number of factors in the model."""
        return len(self.factor_returns)

    def get_loadings(self, ticker: str) -> Optional[FactorLoading]:
        """Get factor loadings for a security."""
        for loading in self.factor_loadings:
            if loading.ticker == ticker:
                return loading
        return None

    def get_loadings_matrix(self, tickers: list[str]) -> np.ndarray:
        """
        Get factor loadings matrix for a list of tickers.
        
        Returns NxK matrix where N = number of tickers, K = number of factors.
        """
        loadings = []
        for ticker in tickers:
            loading = self.get_loadings(ticker)
            if loading is None:
                raise ValueError(f"No factor loadings found for ticker: {ticker}")
            loadings.append(loading.get_loadings_array())
        return np.array(loadings)

    def get_specific_risk_dict(self) -> dict[str, float]:
        """Get dictionary of ticker -> specific risk."""
        return {fl.ticker: fl.specific_risk_pct for fl in self.factor_loadings}

