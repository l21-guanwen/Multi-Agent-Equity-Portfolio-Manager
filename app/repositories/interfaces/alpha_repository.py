"""
Alpha repository interface.

Defines operations for accessing alpha model data.
"""

from abc import abstractmethod
from datetime import date
from typing import Optional

from app.models.alpha import AlphaModel, AlphaScore
from app.repositories.interfaces.base_repository import IBaseRepository


class IAlphaRepository(IBaseRepository[AlphaScore]):
    """
    Repository interface for alpha model data access.
    
    Provides methods to retrieve alpha scores and quintile rankings
    for securities in the investment universe.
    """

    @abstractmethod
    async def get_alpha_model(
        self,
        model_id: str = "AI_ALPHA_MODEL_V1",
        as_of_date: Optional[date] = None,
    ) -> Optional[AlphaModel]:
        """
        Get the full alpha model with all scores.
        
        Args:
            model_id: Alpha model identifier
            as_of_date: Optional date filter
            
        Returns:
            AlphaModel with all scores, or None if not found
        """
        pass

    @abstractmethod
    async def get_score(
        self,
        ticker: str,
        as_of_date: Optional[date] = None,
    ) -> Optional[AlphaScore]:
        """
        Get alpha score for a single security.
        
        Args:
            ticker: Stock ticker symbol
            as_of_date: Optional date filter
            
        Returns:
            AlphaScore if found, None otherwise
        """
        pass

    @abstractmethod
    async def get_scores_by_quintile(
        self,
        quintile: int,
        as_of_date: Optional[date] = None,
    ) -> list[AlphaScore]:
        """
        Get all securities in a specific alpha quintile.
        
        Args:
            quintile: Quintile number (1=best, 5=worst)
            as_of_date: Optional date filter
            
        Returns:
            List of alpha scores in the quintile
        """
        pass

    @abstractmethod
    async def get_top_scores(
        self,
        n: int = 25,
        as_of_date: Optional[date] = None,
    ) -> list[AlphaScore]:
        """
        Get top N securities by alpha score.
        
        Args:
            n: Number of top scores to return
            as_of_date: Optional date filter
            
        Returns:
            List of top N alpha scores sorted by score descending
        """
        pass

    @abstractmethod
    async def get_scores_by_sector(
        self,
        sector: str,
        as_of_date: Optional[date] = None,
    ) -> list[AlphaScore]:
        """
        Get all alpha scores for a specific sector.
        
        Args:
            sector: GICS sector name
            as_of_date: Optional date filter
            
        Returns:
            List of alpha scores in the sector
        """
        pass

    @abstractmethod
    async def get_score_dict(
        self,
        as_of_date: Optional[date] = None,
    ) -> dict[str, float]:
        """
        Get dictionary of ticker to alpha score.
        
        Args:
            as_of_date: Optional date filter
            
        Returns:
            Dictionary of ticker -> alpha score (0-1)
        """
        pass

    @abstractmethod
    async def get_quintile_distribution(
        self,
        as_of_date: Optional[date] = None,
    ) -> dict[int, int]:
        """
        Get count of securities in each quintile.
        
        Args:
            as_of_date: Optional date filter
            
        Returns:
            Dictionary of quintile -> count
        """
        pass

