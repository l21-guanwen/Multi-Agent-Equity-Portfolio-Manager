"""
Constraint repository interface.

Defines operations for accessing optimization constraint data.
"""

from abc import abstractmethod
from datetime import date
from typing import Optional

from app.models.constraint import Constraint, ConstraintSet
from app.repositories.interfaces.base_repository import IBaseRepository


class IConstraintRepository(IBaseRepository[Constraint]):
    """
    Repository interface for optimization constraint data access.
    
    Provides methods to retrieve stock-level and sector-level
    constraints for portfolio optimization.
    """

    @abstractmethod
    async def get_constraint_set(
        self,
        optimization_id: str = "OPT_DEF_SPX_ALPHA_001",
        as_of_date: Optional[date] = None,
    ) -> Optional[ConstraintSet]:
        """
        Get the full constraint set for an optimization.
        
        Args:
            optimization_id: Optimization definition identifier
            as_of_date: Optional date filter
            
        Returns:
            ConstraintSet with all constraints, or None if not found
        """
        pass

    @abstractmethod
    async def get_sector_constraints(
        self,
        as_of_date: Optional[date] = None,
    ) -> list[Constraint]:
        """
        Get all sector-level constraints.
        
        Args:
            as_of_date: Optional date filter
            
        Returns:
            List of sector constraints
        """
        pass

    @abstractmethod
    async def get_stock_constraints(
        self,
        as_of_date: Optional[date] = None,
    ) -> list[Constraint]:
        """
        Get all stock-level constraints.
        
        Args:
            as_of_date: Optional date filter
            
        Returns:
            List of stock constraints
        """
        pass

    @abstractmethod
    async def get_constraint_for_sector(
        self,
        sector: str,
        as_of_date: Optional[date] = None,
    ) -> Optional[Constraint]:
        """
        Get constraint for a specific sector.
        
        Args:
            sector: GICS sector name
            as_of_date: Optional date filter
            
        Returns:
            Constraint if found, None otherwise
        """
        pass

    @abstractmethod
    async def get_constraint_for_stock(
        self,
        ticker: str,
        as_of_date: Optional[date] = None,
    ) -> Optional[Constraint]:
        """
        Get constraint for a specific stock.
        
        Args:
            ticker: Stock ticker symbol
            as_of_date: Optional date filter
            
        Returns:
            Constraint if found, None otherwise
        """
        pass

