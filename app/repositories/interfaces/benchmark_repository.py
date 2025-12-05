"""
Benchmark repository interface.

Defines operations for accessing benchmark constituency data.
"""

from abc import abstractmethod
from datetime import date
from typing import Optional

from app.models.benchmark import Benchmark, BenchmarkConstituent
from app.repositories.interfaces.base_repository import IBaseRepository


class IBenchmarkRepository(IBaseRepository[BenchmarkConstituent]):
    """
    Repository interface for benchmark data access.
    
    Provides methods to retrieve S&P 500 benchmark constituents
    and their weights.
    """

    @abstractmethod
    async def get_benchmark(
        self,
        benchmark_id: str = "SPX",
        as_of_date: Optional[date] = None,
    ) -> Optional[Benchmark]:
        """
        Get the full benchmark with all constituents.
        
        Args:
            benchmark_id: Benchmark identifier (default: SPX)
            as_of_date: Optional date filter
            
        Returns:
            Benchmark object with all constituents, or None if not found
        """
        pass

    @abstractmethod
    async def get_constituent(
        self,
        ticker: str,
        as_of_date: Optional[date] = None,
    ) -> Optional[BenchmarkConstituent]:
        """
        Get a single benchmark constituent by ticker.
        
        Args:
            ticker: Stock ticker symbol
            as_of_date: Optional date filter
            
        Returns:
            BenchmarkConstituent if found, None otherwise
        """
        pass

    @abstractmethod
    async def get_constituents_by_sector(
        self,
        sector: str,
        as_of_date: Optional[date] = None,
    ) -> list[BenchmarkConstituent]:
        """
        Get all constituents in a specific GICS sector.
        
        Args:
            sector: GICS sector name
            as_of_date: Optional date filter
            
        Returns:
            List of constituents in the sector
        """
        pass

    @abstractmethod
    async def get_sector_weights(
        self,
        as_of_date: Optional[date] = None,
    ) -> dict[str, float]:
        """
        Get total benchmark weight by sector.
        
        Args:
            as_of_date: Optional date filter
            
        Returns:
            Dictionary of sector -> total weight percentage
        """
        pass

    @abstractmethod
    async def get_weight_dict(
        self,
        as_of_date: Optional[date] = None,
    ) -> dict[str, float]:
        """
        Get dictionary of ticker to benchmark weight.
        
        Args:
            as_of_date: Optional date filter
            
        Returns:
            Dictionary of ticker -> weight percentage
        """
        pass

    @abstractmethod
    async def get_top_constituents(
        self,
        n: int = 10,
        as_of_date: Optional[date] = None,
    ) -> list[BenchmarkConstituent]:
        """
        Get top N constituents by benchmark weight.
        
        Args:
            n: Number of top constituents to return
            as_of_date: Optional date filter
            
        Returns:
            List of top N constituents sorted by weight descending
        """
        pass

