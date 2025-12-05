"""
Universe repository interface.

Defines operations for accessing investment universe data.
"""

from abc import abstractmethod
from datetime import date
from typing import Optional

from app.models.benchmark import BenchmarkConstituent
from app.repositories.interfaces.base_repository import IBaseRepository


class IUniverseRepository(IBaseRepository[BenchmarkConstituent]):
    """
    Repository interface for investment universe data access.
    
    The investment universe is typically the same as the benchmark
    but may have additional investibility and liquidity filters.
    """

    @abstractmethod
    async def get_investible_securities(
        self,
        as_of_date: Optional[date] = None,
    ) -> list[BenchmarkConstituent]:
        """
        Get all investible securities in the universe.
        
        Args:
            as_of_date: Optional date filter
            
        Returns:
            List of investible securities
        """
        pass

    @abstractmethod
    async def get_securities_by_sector(
        self,
        sector: str,
        as_of_date: Optional[date] = None,
    ) -> list[BenchmarkConstituent]:
        """
        Get all securities in a specific GICS sector.
        
        Args:
            sector: GICS sector name
            as_of_date: Optional date filter
            
        Returns:
            List of securities in the sector
        """
        pass

    @abstractmethod
    async def get_securities_by_liquidity(
        self,
        min_liquidity_score: float = 0.0,
        as_of_date: Optional[date] = None,
    ) -> list[BenchmarkConstituent]:
        """
        Get securities filtered by minimum liquidity score.
        
        Args:
            min_liquidity_score: Minimum liquidity score (0-1)
            as_of_date: Optional date filter
            
        Returns:
            List of securities meeting liquidity threshold
        """
        pass

    @abstractmethod
    async def get_tickers(
        self,
        as_of_date: Optional[date] = None,
    ) -> list[str]:
        """
        Get list of all tickers in the universe.
        
        Args:
            as_of_date: Optional date filter
            
        Returns:
            List of ticker symbols
        """
        pass

