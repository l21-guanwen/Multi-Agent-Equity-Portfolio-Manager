"""
Transaction cost repository interface.

Defines operations for accessing transaction cost model data.
"""

from abc import abstractmethod
from datetime import date
from typing import Optional

from app.models.transaction_cost import TransactionCost, TransactionCostModel
from app.repositories.interfaces.base_repository import IBaseRepository


class ITransactionCostRepository(IBaseRepository[TransactionCost]):
    """
    Repository interface for transaction cost data access.
    
    Provides methods to retrieve transaction cost estimates
    including bid-ask spread, commission, and market impact.
    """

    @abstractmethod
    async def get_transaction_cost_model(
        self,
        model_id: str = "TCOST_MARKET_IMPACT_V1",
        as_of_date: Optional[date] = None,
    ) -> Optional[TransactionCostModel]:
        """
        Get the full transaction cost model.
        
        Args:
            model_id: Transaction cost model identifier
            as_of_date: Optional date filter
            
        Returns:
            TransactionCostModel with all costs, or None if not found
        """
        pass

    @abstractmethod
    async def get_cost(
        self,
        ticker: str,
        as_of_date: Optional[date] = None,
    ) -> Optional[TransactionCost]:
        """
        Get transaction cost for a single security.
        
        Args:
            ticker: Stock ticker symbol
            as_of_date: Optional date filter
            
        Returns:
            TransactionCost if found, None otherwise
        """
        pass

    @abstractmethod
    async def get_costs_by_liquidity_bucket(
        self,
        bucket: int,
        as_of_date: Optional[date] = None,
    ) -> list[TransactionCost]:
        """
        Get all securities in a specific liquidity bucket.
        
        Args:
            bucket: Liquidity bucket (1=most liquid, 5=least liquid)
            as_of_date: Optional date filter
            
        Returns:
            List of transaction costs in the bucket
        """
        pass

    @abstractmethod
    async def get_cost_dict(
        self,
        urgency: str = "medium",
        as_of_date: Optional[date] = None,
    ) -> dict[str, float]:
        """
        Get dictionary of ticker to transaction cost.
        
        Args:
            urgency: Cost urgency level ('low', 'medium', 'high')
            as_of_date: Optional date filter
            
        Returns:
            Dictionary of ticker -> cost in basis points
        """
        pass

    @abstractmethod
    async def get_liquid_securities(
        self,
        max_bucket: int = 2,
        as_of_date: Optional[date] = None,
    ) -> list[TransactionCost]:
        """
        Get securities in liquid buckets.
        
        Args:
            max_bucket: Maximum liquidity bucket (1-5)
            as_of_date: Optional date filter
            
        Returns:
            List of transaction costs for liquid securities
        """
        pass

    @abstractmethod
    async def get_average_cost_by_sector(
        self,
        urgency: str = "medium",
        as_of_date: Optional[date] = None,
    ) -> dict[str, float]:
        """
        Get average transaction cost by sector.
        
        Args:
            urgency: Cost urgency level ('low', 'medium', 'high')
            as_of_date: Optional date filter
            
        Returns:
            Dictionary of sector -> average cost in basis points
        """
        pass

