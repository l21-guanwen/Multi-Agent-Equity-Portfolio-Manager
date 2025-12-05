"""
Base repository interface.

Provides abstract base class for all repository implementations,
enabling switching between CSV, database, or API data sources.
"""

from abc import ABC, abstractmethod
from datetime import date
from typing import Generic, Optional, TypeVar

T = TypeVar("T")


class IBaseRepository(ABC, Generic[T]):
    """
    Abstract base repository interface.
    
    Defines standard CRUD operations that all repositories must implement.
    Generic type T represents the domain model type.
    
    Implementations:
    - CSVRepository: Loads data from CSV files
    - DatabaseRepository: Loads data from SQL database (future)
    - APIRepository: Loads data from external API (future)
    """

    @abstractmethod
    async def get_all(self, as_of_date: Optional[date] = None) -> list[T]:
        """
        Retrieve all records.
        
        Args:
            as_of_date: Optional date filter for time-series data
            
        Returns:
            List of all records of type T
        """
        pass

    @abstractmethod
    async def get_by_id(self, id: str) -> Optional[T]:
        """
        Retrieve a single record by its identifier.
        
        Args:
            id: The unique identifier (e.g., ticker symbol)
            
        Returns:
            The record if found, None otherwise
        """
        pass

    @abstractmethod
    async def get_by_ids(self, ids: list[str]) -> list[T]:
        """
        Retrieve multiple records by their identifiers.
        
        Args:
            ids: List of unique identifiers
            
        Returns:
            List of found records (may be fewer than requested if some not found)
        """
        pass

    async def exists(self, id: str) -> bool:
        """
        Check if a record exists.
        
        Args:
            id: The unique identifier to check
            
        Returns:
            True if the record exists, False otherwise
        """
        result = await self.get_by_id(id)
        return result is not None

    async def count(self, as_of_date: Optional[date] = None) -> int:
        """
        Get the total count of records.
        
        Args:
            as_of_date: Optional date filter
            
        Returns:
            Number of records
        """
        records = await self.get_all(as_of_date)
        return len(records)

