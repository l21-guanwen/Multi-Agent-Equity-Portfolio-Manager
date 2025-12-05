"""
CSV-based universe repository implementation.

Loads investment universe data from CSV files.
"""

from datetime import date
from typing import Optional

from app.core.constants import DataFileName
from app.models.benchmark import BenchmarkConstituent
from app.repositories.interfaces.universe_repository import IUniverseRepository
from app.utils.csv_loader import CSVLoader


class CSVUniverseRepository(IUniverseRepository):
    """
    CSV implementation of the universe repository.
    
    Loads investment universe from 02_SP500_Universe.csv.
    The universe is typically the same as benchmark with additional filters.
    """

    COLUMN_MAPPING = {
        "Ticker": "ticker",
        "Security_Name": "security_name",
        "Security_ID": "security_id",
        "ISIN": "isin",
        "CUSIP": "cusip",
        "GICS_Sector": "gics_sector",
        "GICS_Industry": "gics_industry",
        "Benchmark_ID": "benchmark_id",
        "Benchmark_Name": "benchmark_name",
        "Benchmark_Weight_Pct": "benchmark_weight_pct",
        "Price": "price",
        "Market_Cap_USD_B": "market_cap_usd_b",
        "Shares_Outstanding_M": "shares_outstanding_m",
        "Dividend_Yield_Pct": "dividend_yield_pct",
        "PE_Ratio": "pe_ratio",
        "Beta": "beta",
        "Exchange": "exchange",
        "Currency": "currency",
        "As_Of_Date": "as_of_date",
        # Universe-specific fields
        "Universe_ID": "universe_id",
        "Is_Investible": "is_investible",
        "Min_Position_Pct": "min_position_pct",
        "Max_Position_Pct": "max_position_pct",
        "Liquidity_Score": "liquidity_score",
    }

    def __init__(self, csv_loader: Optional[CSVLoader] = None):
        """Initialize the repository."""
        self._loader = csv_loader or CSVLoader()
        self._cache: Optional[list[BenchmarkConstituent]] = None

    async def _load_data(self) -> list[BenchmarkConstituent]:
        """Load and cache universe data."""
        if self._cache is None:
            # Try universe file first, fall back to benchmark
            try:
                self._cache = self._loader.load_as_models(
                    DataFileName.UNIVERSE,
                    BenchmarkConstituent,
                    self.COLUMN_MAPPING,
                )
            except FileNotFoundError:
                # Fall back to benchmark file
                self._cache = self._loader.load_as_models(
                    DataFileName.BENCHMARK,
                    BenchmarkConstituent,
                    self.COLUMN_MAPPING,
                )
        return self._cache

    def clear_cache(self):
        """Clear the data cache."""
        self._cache = None

    async def get_all(self, as_of_date: Optional[date] = None) -> list[BenchmarkConstituent]:
        """Get all securities in the universe."""
        data = await self._load_data()
        if as_of_date:
            return [s for s in data if s.as_of_date == as_of_date]
        return data

    async def get_by_id(self, id: str) -> Optional[BenchmarkConstituent]:
        """Get security by ticker."""
        data = await self._load_data()
        for security in data:
            if security.ticker == id:
                return security
        return None

    async def get_by_ids(self, ids: list[str]) -> list[BenchmarkConstituent]:
        """Get securities by multiple tickers."""
        data = await self._load_data()
        return [s for s in data if s.ticker in ids]

    async def get_investible_securities(
        self,
        as_of_date: Optional[date] = None,
    ) -> list[BenchmarkConstituent]:
        """Get all investible securities."""
        # For now, all securities are investible
        # In future, filter by is_investible flag
        return await self.get_all(as_of_date)

    async def get_securities_by_sector(
        self,
        sector: str,
        as_of_date: Optional[date] = None,
    ) -> list[BenchmarkConstituent]:
        """Get all securities in a specific sector."""
        data = await self.get_all(as_of_date)
        return [s for s in data if s.gics_sector == sector]

    async def get_securities_by_liquidity(
        self,
        min_liquidity_score: float = 0.0,
        as_of_date: Optional[date] = None,
    ) -> list[BenchmarkConstituent]:
        """Get securities filtered by minimum liquidity score."""
        # For now, return all securities
        # In future, filter by liquidity_score field
        return await self.get_all(as_of_date)

    async def get_tickers(
        self,
        as_of_date: Optional[date] = None,
    ) -> list[str]:
        """Get list of all tickers in the universe."""
        data = await self.get_all(as_of_date)
        return [s.ticker for s in data]

