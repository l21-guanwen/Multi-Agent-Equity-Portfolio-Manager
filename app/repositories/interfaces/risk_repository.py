"""
Risk repository interface.

Defines operations for accessing risk model data.
"""

from abc import abstractmethod
from datetime import date
from typing import Optional

import numpy as np

from app.models.risk import (
    FactorCovariance,
    FactorLoading,
    FactorReturn,
    RiskModel,
)
from app.repositories.interfaces.base_repository import IBaseRepository


class IRiskRepository(IBaseRepository[FactorLoading]):
    """
    Repository interface for risk model data access.
    
    Provides methods to retrieve factor loadings, factor returns,
    and factor covariance matrix for the Barra-style risk model.
    """

    @abstractmethod
    async def get_risk_model(
        self,
        model_id: str = "BARRA_STYLE_V1",
        as_of_date: Optional[date] = None,
    ) -> Optional[RiskModel]:
        """
        Get the complete risk model with all components.
        
        Args:
            model_id: Risk model identifier
            as_of_date: Optional date filter
            
        Returns:
            RiskModel with loadings, returns, and covariance
        """
        pass

    @abstractmethod
    async def get_factor_loadings(
        self,
        ticker: str,
        as_of_date: Optional[date] = None,
    ) -> Optional[FactorLoading]:
        """
        Get factor loadings for a single security.
        
        Args:
            ticker: Stock ticker symbol
            as_of_date: Optional date filter
            
        Returns:
            FactorLoading if found, None otherwise
        """
        pass

    @abstractmethod
    async def get_factor_loadings_for_tickers(
        self,
        tickers: list[str],
        as_of_date: Optional[date] = None,
    ) -> list[FactorLoading]:
        """
        Get factor loadings for multiple securities.
        
        Args:
            tickers: List of stock ticker symbols
            as_of_date: Optional date filter
            
        Returns:
            List of FactorLoading objects (may be fewer than requested)
        """
        pass

    @abstractmethod
    async def get_factor_loadings_matrix(
        self,
        tickers: list[str],
        as_of_date: Optional[date] = None,
    ) -> np.ndarray:
        """
        Get factor loadings as a matrix.
        
        Args:
            tickers: List of stock ticker symbols (defines row order)
            as_of_date: Optional date filter
            
        Returns:
            NxK numpy array where N=number of tickers, K=number of factors
        """
        pass

    @abstractmethod
    async def get_factor_returns(
        self,
        as_of_date: Optional[date] = None,
    ) -> list[FactorReturn]:
        """
        Get historical returns for all factors.
        
        Args:
            as_of_date: Optional date filter
            
        Returns:
            List of FactorReturn objects for each factor
        """
        pass

    @abstractmethod
    async def get_factor_covariance(
        self,
        as_of_date: Optional[date] = None,
    ) -> Optional[FactorCovariance]:
        """
        Get the factor covariance matrix.
        
        Args:
            as_of_date: Optional date filter
            
        Returns:
            FactorCovariance containing 8x8 covariance matrix
        """
        pass

    @abstractmethod
    async def get_factor_covariance_matrix(
        self,
        as_of_date: Optional[date] = None,
    ) -> np.ndarray:
        """
        Get the factor covariance matrix as numpy array.
        
        Args:
            as_of_date: Optional date filter
            
        Returns:
            8x8 numpy array of factor covariances
        """
        pass

    @abstractmethod
    async def get_specific_risk_dict(
        self,
        as_of_date: Optional[date] = None,
    ) -> dict[str, float]:
        """
        Get dictionary of ticker to specific (idiosyncratic) risk.
        
        Args:
            as_of_date: Optional date filter
            
        Returns:
            Dictionary of ticker -> specific risk percentage
        """
        pass

