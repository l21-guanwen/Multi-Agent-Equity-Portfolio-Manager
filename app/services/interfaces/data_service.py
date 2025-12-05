"""
Data service interface.

Defines operations for data validation and aggregation.
"""

from abc import ABC, abstractmethod
from datetime import date
from typing import Any, Optional

from pydantic import BaseModel


class DataValidationResult(BaseModel):
    """Result of data validation check."""
    
    is_valid: bool
    issues: list[str] = []
    warnings: list[str] = []
    record_count: int = 0
    missing_fields: list[str] = []
    data_quality_score: float = 1.0  # 0-1 score


class DataSummary(BaseModel):
    """Summary of loaded data."""
    
    benchmark_count: int = 0
    universe_count: int = 0
    alpha_count: int = 0
    factor_loadings_count: int = 0
    constraints_count: int = 0
    transaction_costs_count: int = 0
    as_of_date: Optional[date] = None
    validation_result: Optional[DataValidationResult] = None


class IDataService(ABC):
    """
    Service interface for data management operations.
    
    Handles loading, validating, and aggregating data from
    various repositories. Acts as the data ingestion layer
    for the portfolio management workflow.
    """

    @abstractmethod
    async def load_all_data(
        self,
        as_of_date: Optional[date] = None,
    ) -> dict[str, Any]:
        """
        Load all required data for portfolio construction.
        
        Args:
            as_of_date: Optional date filter for all data
            
        Returns:
            Dictionary containing all loaded data:
            - benchmark: Benchmark object
            - universe: List of securities
            - alpha_model: AlphaModel object
            - risk_model: RiskModel object
            - constraints: ConstraintSet object
            - transaction_costs: TransactionCostModel object
        """
        pass

    @abstractmethod
    async def validate_data(
        self,
        data: dict[str, Any],
    ) -> DataValidationResult:
        """
        Validate loaded data for completeness and consistency.
        
        Checks:
        - All required datasets are present
        - No missing critical fields
        - Data consistency across datasets (e.g., tickers match)
        - Date alignment
        
        Args:
            data: Dictionary of loaded data from load_all_data()
            
        Returns:
            DataValidationResult with validation status and issues
        """
        pass

    @abstractmethod
    async def get_data_summary(
        self,
        as_of_date: Optional[date] = None,
    ) -> DataSummary:
        """
        Get a summary of available data.
        
        Args:
            as_of_date: Optional date filter
            
        Returns:
            DataSummary with record counts and validation status
        """
        pass

    @abstractmethod
    async def check_data_availability(
        self,
        as_of_date: Optional[date] = None,
    ) -> dict[str, bool]:
        """
        Check which data sources are available.
        
        Args:
            as_of_date: Optional date filter
            
        Returns:
            Dictionary of data source -> availability status
        """
        pass

    @abstractmethod
    async def get_common_tickers(
        self,
        as_of_date: Optional[date] = None,
    ) -> list[str]:
        """
        Get tickers that exist across all data sources.
        
        Args:
            as_of_date: Optional date filter
            
        Returns:
            List of ticker symbols present in all datasets
        """
        pass

