"""
CSV-based benchmark repository implementation.

Loads benchmark constituency data from CSV files.
"""

from datetime import date
from typing import Optional

from app.core.constants import DataFileName
from app.models.benchmark import Benchmark, BenchmarkConstituent
from app.repositories.interfaces.benchmark_repository import IBenchmarkRepository
from app.utils.csv_loader import CSVLoader


class CSVBenchmarkRepository(IBenchmarkRepository):
    """
    CSV implementation of the benchmark repository.
    
    Loads S&P 500 benchmark data from 01_SP500_Benchmark_Constituency.csv.
    """

    # Column mapping from CSV to model fields
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
        "Alpha_Score": "alpha_score",
        "Alpha_Quintile": "alpha_quintile",
        "Exchange": "exchange",
        "Currency": "currency",
        "As_Of_Date": "as_of_date",
    }

    def __init__(self, csv_loader: Optional[CSVLoader] = None):
        """
        Initialize the repository.
        
        Args:
            csv_loader: Optional CSVLoader instance. Creates new one if not provided.
        """
        self._loader = csv_loader or CSVLoader()
        self._cache: Optional[list[BenchmarkConstituent]] = None

    async def _load_data(self) -> list[BenchmarkConstituent]:
        """Load and cache benchmark data."""
        if self._cache is None:
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
        """Get all benchmark constituents."""
        data = await self._load_data()
        if as_of_date:
            return [c for c in data if c.as_of_date == as_of_date]
        return data

    async def get_by_id(self, id: str) -> Optional[BenchmarkConstituent]:
        """Get constituent by ticker."""
        return await self.get_constituent(id)

    async def get_by_ids(self, ids: list[str]) -> list[BenchmarkConstituent]:
        """Get constituents by multiple tickers."""
        data = await self._load_data()
        return [c for c in data if c.ticker in ids]

    async def get_benchmark(
        self,
        benchmark_id: str = "SPX",
        as_of_date: Optional[date] = None,
    ) -> Optional[Benchmark]:
        """Get the full benchmark with all constituents."""
        constituents = await self.get_all(as_of_date)
        
        if not constituents:
            return None
        
        # Get as_of_date from first constituent if not specified
        actual_date = as_of_date or constituents[0].as_of_date
        
        return Benchmark(
            benchmark_id=benchmark_id,
            benchmark_name="S&P 500",
            constituents=constituents,
            as_of_date=actual_date,
        )

    async def get_constituent(
        self,
        ticker: str,
        as_of_date: Optional[date] = None,
    ) -> Optional[BenchmarkConstituent]:
        """Get a single constituent by ticker."""
        data = await self.get_all(as_of_date)
        for constituent in data:
            if constituent.ticker == ticker:
                return constituent
        return None

    async def get_constituents_by_sector(
        self,
        sector: str,
        as_of_date: Optional[date] = None,
    ) -> list[BenchmarkConstituent]:
        """Get all constituents in a specific sector."""
        data = await self.get_all(as_of_date)
        return [c for c in data if c.gics_sector == sector]

    async def get_sector_weights(
        self,
        as_of_date: Optional[date] = None,
    ) -> dict[str, float]:
        """Get total benchmark weight by sector."""
        data = await self.get_all(as_of_date)
        sector_weights: dict[str, float] = {}
        for constituent in data:
            sector = constituent.gics_sector
            sector_weights[sector] = sector_weights.get(sector, 0.0) + constituent.benchmark_weight_pct
        return sector_weights

    async def get_weight_dict(
        self,
        as_of_date: Optional[date] = None,
    ) -> dict[str, float]:
        """Get dictionary of ticker to benchmark weight."""
        data = await self.get_all(as_of_date)
        return {c.ticker: c.benchmark_weight_pct for c in data}

    async def get_top_constituents(
        self,
        n: int = 10,
        as_of_date: Optional[date] = None,
    ) -> list[BenchmarkConstituent]:
        """Get top N constituents by benchmark weight."""
        data = await self.get_all(as_of_date)
        sorted_data = sorted(data, key=lambda c: c.benchmark_weight_pct, reverse=True)
        return sorted_data[:n]

