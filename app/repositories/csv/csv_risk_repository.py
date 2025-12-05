"""
CSV-based risk repository implementation.

Loads risk model data from CSV files.
"""

from datetime import date
from typing import Optional

import numpy as np

from app.core.constants import DataFileName, RISK_FACTORS
from app.models.risk import (
    FactorCovariance,
    FactorLoading,
    FactorReturn,
    RiskModel,
)
from app.repositories.interfaces.risk_repository import IRiskRepository
from app.utils.csv_loader import CSVLoader


class CSVRiskRepository(IRiskRepository):
    """
    CSV implementation of the risk repository.
    
    Loads risk model data from:
    - 05_Risk_Model_Factor_Loadings.csv
    - 06_Risk_Model_Factor_Returns.csv
    - 07_Risk_Model_Factor_Covariance.csv
    """

    LOADINGS_COLUMN_MAPPING = {
        "Ticker": "ticker",
        "Security_Name": "security_name",
        "GICS_Sector": "gics_sector",
        "Specific_Risk_Pct": "specific_risk_pct",
        "Market_Loading": "market_loading",
        "Size_Loading": "size_loading",
        "Value_Loading": "value_loading",
        "Momentum_Loading": "momentum_loading",
        "Quality_Loading": "quality_loading",
        "Volatility_Loading": "volatility_loading",
        "Growth_Loading": "growth_loading",
        "Dividend_Yield_Loading": "dividend_yield_loading",
        "Risk_Model_ID": "risk_model_id",
        "As_Of_Date": "as_of_date",
    }

    def __init__(self, csv_loader: Optional[CSVLoader] = None):
        """Initialize the repository."""
        self._loader = csv_loader or CSVLoader()
        self._loadings_cache: Optional[list[FactorLoading]] = None
        self._returns_cache: Optional[list[FactorReturn]] = None
        self._covariance_cache: Optional[FactorCovariance] = None

    async def _load_loadings(self) -> list[FactorLoading]:
        """Load and cache factor loadings."""
        if self._loadings_cache is None:
            self._loadings_cache = self._loader.load_as_models(
                DataFileName.FACTOR_LOADINGS,
                FactorLoading,
                self.LOADINGS_COLUMN_MAPPING,
            )
        return self._loadings_cache

    async def _load_returns(self) -> list[FactorReturn]:
        """Load and cache factor returns."""
        if self._returns_cache is None:
            df = self._loader.load_dataframe(DataFileName.FACTOR_RETURNS)
            
            returns = []
            for _, row in df.iterrows():
                factor_return = FactorReturn(
                    factor_name=row.get("Factor_Name", row.get("Factor", "")),
                    factor_return_mtd_pct=row.get("Factor_Return_MTD_Pct"),
                    factor_return_qtd_pct=row.get("Factor_Return_QTD_Pct"),
                    factor_return_ytd_pct=row.get("Factor_Return_YTD_Pct"),
                    factor_return_1y_pct=row.get("Factor_Return_1Y_Pct"),
                    factor_volatility_pct=row.get("Factor_Volatility_Pct"),
                    factor_sharpe=row.get("Factor_Sharpe"),
                    risk_model_id=row.get("Risk_Model_ID", "BARRA_STYLE_V1"),
                    as_of_date=row.get("As_Of_Date", date.today()),
                )
                returns.append(factor_return)
            
            self._returns_cache = returns
        return self._returns_cache

    async def _load_covariance(self) -> FactorCovariance:
        """Load and cache factor covariance matrix."""
        if self._covariance_cache is None:
            df = self._loader.load_dataframe(DataFileName.FACTOR_COVARIANCE)
            
            # Get factor names from first column or index
            if "Factor" in df.columns:
                factors = df["Factor"].tolist()
                # Get numeric columns for matrix
                numeric_cols = [c for c in df.columns if c not in ["Factor", "Risk_Model_ID", "As_Of_Date"]]
                matrix_df = df[numeric_cols]
            else:
                factors = RISK_FACTORS
                matrix_df = df.select_dtypes(include=[np.number])
            
            # Convert to nested list
            matrix = matrix_df.values.tolist()
            
            # Get metadata
            risk_model_id = df["Risk_Model_ID"].iloc[0] if "Risk_Model_ID" in df.columns else "BARRA_STYLE_V1"
            as_of_date_val = df["As_Of_Date"].iloc[0] if "As_Of_Date" in df.columns else date.today()
            
            self._covariance_cache = FactorCovariance(
                factors=factors,
                matrix=matrix,
                risk_model_id=risk_model_id,
                as_of_date=as_of_date_val,
            )
        
        return self._covariance_cache

    def clear_cache(self):
        """Clear all caches."""
        self._loadings_cache = None
        self._returns_cache = None
        self._covariance_cache = None

    async def get_all(self, as_of_date: Optional[date] = None) -> list[FactorLoading]:
        """Get all factor loadings."""
        data = await self._load_loadings()
        if as_of_date:
            return [l for l in data if l.as_of_date == as_of_date]
        return data

    async def get_by_id(self, id: str) -> Optional[FactorLoading]:
        """Get factor loadings by ticker."""
        return await self.get_factor_loadings(id)

    async def get_by_ids(self, ids: list[str]) -> list[FactorLoading]:
        """Get factor loadings by multiple tickers."""
        data = await self._load_loadings()
        return [l for l in data if l.ticker in ids]

    async def get_risk_model(
        self,
        model_id: str = "BARRA_STYLE_V1",
        as_of_date: Optional[date] = None,
    ) -> Optional[RiskModel]:
        """Get the complete risk model with all components."""
        loadings = await self.get_all(as_of_date)
        returns = await self.get_factor_returns(as_of_date)
        covariance = await self.get_factor_covariance(as_of_date)
        
        if not loadings:
            return None
        
        actual_date = as_of_date or loadings[0].as_of_date
        
        return RiskModel(
            risk_model_id=model_id,
            risk_model_name="Barra-Style Multi-Factor Model",
            factor_loadings=loadings,
            factor_returns=returns,
            factor_covariance=covariance,
            as_of_date=actual_date,
        )

    async def get_factor_loadings(
        self,
        ticker: str,
        as_of_date: Optional[date] = None,
    ) -> Optional[FactorLoading]:
        """Get factor loadings for a single security."""
        data = await self.get_all(as_of_date)
        for loading in data:
            if loading.ticker == ticker:
                return loading
        return None

    async def get_factor_loadings_for_tickers(
        self,
        tickers: list[str],
        as_of_date: Optional[date] = None,
    ) -> list[FactorLoading]:
        """Get factor loadings for multiple securities."""
        return await self.get_by_ids(tickers)

    async def get_factor_loadings_matrix(
        self,
        tickers: list[str],
        as_of_date: Optional[date] = None,
    ) -> np.ndarray:
        """Get factor loadings as a matrix."""
        loadings = await self.get_factor_loadings_for_tickers(tickers, as_of_date)
        
        # Create mapping for ordering
        ticker_to_loading = {l.ticker: l for l in loadings}
        
        # Build matrix in ticker order
        matrix = []
        for ticker in tickers:
            if ticker in ticker_to_loading:
                matrix.append(ticker_to_loading[ticker].get_loadings_array())
            else:
                # Missing ticker - use zeros
                matrix.append(np.zeros(8))
        
        return np.array(matrix)

    async def get_factor_returns(
        self,
        as_of_date: Optional[date] = None,
    ) -> list[FactorReturn]:
        """Get historical returns for all factors."""
        returns = await self._load_returns()
        if as_of_date:
            return [r for r in returns if r.as_of_date == as_of_date]
        return returns

    async def get_factor_covariance(
        self,
        as_of_date: Optional[date] = None,
    ) -> Optional[FactorCovariance]:
        """Get the factor covariance matrix."""
        return await self._load_covariance()

    async def get_factor_covariance_matrix(
        self,
        as_of_date: Optional[date] = None,
    ) -> np.ndarray:
        """Get the factor covariance matrix as numpy array."""
        covariance = await self.get_factor_covariance(as_of_date)
        if covariance is None:
            return np.eye(8)  # Return identity if not available
        return covariance.get_matrix_array()

    async def get_specific_risk_dict(
        self,
        as_of_date: Optional[date] = None,
    ) -> dict[str, float]:
        """Get dictionary of ticker to specific risk."""
        loadings = await self.get_all(as_of_date)
        return {l.ticker: l.specific_risk_pct for l in loadings}

