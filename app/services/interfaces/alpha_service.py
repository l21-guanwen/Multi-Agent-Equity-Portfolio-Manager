"""
Alpha service interface.

Defines operations for alpha signal processing and analysis.
"""

from abc import ABC, abstractmethod
from datetime import date
from typing import Optional

from pydantic import BaseModel

from app.models.alpha import AlphaModel, AlphaScore


class AlphaAnalysis(BaseModel):
    """Analysis results for alpha model."""
    
    total_securities: int
    quintile_distribution: dict[int, int]
    average_alpha_score: float
    top_sector: str
    bottom_sector: str
    sector_average_scores: dict[str, float]
    signal_strength_distribution: dict[str, int]


class SecuritySelection(BaseModel):
    """Result of security selection process."""
    
    selected_tickers: list[str]
    selection_count: int
    average_alpha_score: float
    sector_distribution: dict[str, int]
    selection_criteria: str


class IAlphaService(ABC):
    """
    Service interface for alpha model operations.
    
    Handles alpha score processing, quintile filtering,
    and security selection for portfolio construction.
    """

    @abstractmethod
    async def get_top_quintile_securities(
        self,
        alpha_model: AlphaModel,
        top_n: int = 25,
    ) -> list[AlphaScore]:
        """
        Get top N securities from quintile 1.
        
        Args:
            alpha_model: AlphaModel containing all scores
            top_n: Number of securities to select
            
        Returns:
            List of top N AlphaScore objects from Q1
        """
        pass

    @abstractmethod
    async def filter_by_quintile(
        self,
        alpha_model: AlphaModel,
        quintiles: list[int] = [1],
    ) -> list[AlphaScore]:
        """
        Filter securities by quintile membership.
        
        Args:
            alpha_model: AlphaModel containing all scores
            quintiles: List of quintiles to include (1=best)
            
        Returns:
            List of AlphaScore objects in specified quintiles
        """
        pass

    @abstractmethod
    async def analyze_alpha_model(
        self,
        alpha_model: AlphaModel,
    ) -> AlphaAnalysis:
        """
        Perform comprehensive analysis of alpha model.
        
        Args:
            alpha_model: AlphaModel to analyze
            
        Returns:
            AlphaAnalysis with distribution and sector breakdowns
        """
        pass

    @abstractmethod
    async def rank_securities(
        self,
        alpha_model: AlphaModel,
        ascending: bool = False,
    ) -> list[AlphaScore]:
        """
        Rank all securities by alpha score.
        
        Args:
            alpha_model: AlphaModel containing all scores
            ascending: If True, lowest scores first
            
        Returns:
            Sorted list of AlphaScore objects
        """
        pass

    @abstractmethod
    async def select_securities(
        self,
        alpha_model: AlphaModel,
        count: int = 25,
        min_quintile: int = 1,
        max_quintile: int = 2,
        sector_constraints: Optional[dict[str, int]] = None,
    ) -> SecuritySelection:
        """
        Select securities for portfolio based on criteria.
        
        Args:
            alpha_model: AlphaModel containing all scores
            count: Target number of securities to select
            min_quintile: Minimum (best) quintile to consider
            max_quintile: Maximum (worst) quintile to consider
            sector_constraints: Optional max count per sector
            
        Returns:
            SecuritySelection with selected tickers and metadata
        """
        pass

    @abstractmethod
    async def get_sector_scores(
        self,
        alpha_model: AlphaModel,
    ) -> dict[str, float]:
        """
        Get average alpha score by sector.
        
        Args:
            alpha_model: AlphaModel containing all scores
            
        Returns:
            Dictionary of sector -> average alpha score
        """
        pass

    @abstractmethod
    async def get_alpha_weights(
        self,
        alpha_scores: list[AlphaScore],
        normalize: bool = True,
    ) -> dict[str, float]:
        """
        Calculate alpha-weighted portfolio weights.
        
        Args:
            alpha_scores: List of AlphaScore objects
            normalize: If True, weights sum to 1.0
            
        Returns:
            Dictionary of ticker -> weight
        """
        pass

