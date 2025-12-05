"""Benchmark domain models."""

from datetime import date
from typing import Any, Optional

from pydantic import BaseModel, Field, computed_field, field_validator


class BenchmarkConstituent(BaseModel):
    """
    A single constituent of a benchmark index.
    
    Extends security information with benchmark-specific weight data.
    Based on 01_SP500_Benchmark_Constituency.csv schema.
    """

    # Identifiers
    ticker: str = Field(..., description="Stock ticker symbol")
    security_name: str = Field(..., description="Full security name")
    security_id: Optional[str] = Field(None, description="Unique security identifier")
    isin: Optional[str] = Field(None, description="International Security ID (ISIN)")
    cusip: Optional[str] = Field(None, description="CUSIP identifier")

    @field_validator("security_id", "isin", "cusip", mode="before")
    @classmethod
    def coerce_to_string(cls, v: Any) -> Optional[str]:
        """Coerce numeric identifiers to strings."""
        if v is None:
            return None
        return str(v)
    
    # Benchmark Info
    benchmark_id: str = Field(default="SPX", description="Benchmark identifier")
    benchmark_name: str = Field(default="S&P 500", description="Benchmark name")
    benchmark_weight_pct: float = Field(..., ge=0, le=100, description="Weight in benchmark (%)")
    
    # Classification
    gics_sector: str = Field(..., description="GICS Level 1 sector")
    gics_industry: Optional[str] = Field(None, description="GICS industry group")
    
    # Market Data
    price: float = Field(..., ge=0, description="Last price (USD)")
    market_cap_usd_b: Optional[float] = Field(None, ge=0, description="Market cap in billions USD")
    shares_outstanding_m: Optional[float] = Field(None, ge=0, description="Shares outstanding in millions")
    
    # Fundamentals
    dividend_yield_pct: Optional[float] = Field(None, ge=0, description="Dividend yield (%)")
    pe_ratio: Optional[float] = Field(None, description="Price to earnings ratio")
    beta: Optional[float] = Field(None, description="Market beta")
    
    # Alpha (if included in benchmark file)
    alpha_score: Optional[float] = Field(None, ge=0, le=1, description="Alpha score (0-1)")
    alpha_quintile: Optional[int] = Field(None, ge=1, le=5, description="Alpha quintile (1=best)")
    
    # Metadata
    exchange: Optional[str] = Field(None, description="Exchange")
    currency: str = Field(default="USD", description="Trading currency")
    as_of_date: date = Field(..., description="Data as-of date")

    @computed_field
    @property
    def benchmark_weight_decimal(self) -> float:
        """Benchmark weight as decimal (0-1)."""
        return self.benchmark_weight_pct / 100.0

    class Config:
        """Pydantic model configuration."""
        
        frozen = False
        str_strip_whitespace = True


class Benchmark(BaseModel):
    """
    Benchmark index containing multiple constituents.
    
    Represents the full S&P 500 benchmark.
    """

    benchmark_id: str = Field(default="SPX", description="Benchmark identifier")
    benchmark_name: str = Field(default="S&P 500", description="Benchmark name")
    constituents: list[BenchmarkConstituent] = Field(default_factory=list)
    as_of_date: date = Field(..., description="Data as-of date")

    @computed_field
    @property
    def total_weight_pct(self) -> float:
        """Total weight of all constituents (should be ~100%)."""
        return sum(c.benchmark_weight_pct for c in self.constituents)

    @computed_field
    @property
    def security_count(self) -> int:
        """Number of constituents."""
        return len(self.constituents)

    def get_constituent(self, ticker: str) -> Optional[BenchmarkConstituent]:
        """Get a constituent by ticker."""
        for constituent in self.constituents:
            if constituent.ticker == ticker:
                return constituent
        return None

    def get_sector_weights(self) -> dict[str, float]:
        """Get total weight by GICS sector."""
        sector_weights: dict[str, float] = {}
        for constituent in self.constituents:
            sector = constituent.gics_sector
            sector_weights[sector] = sector_weights.get(sector, 0.0) + constituent.benchmark_weight_pct
        return sector_weights

    def get_weight_dict(self) -> dict[str, float]:
        """Get dictionary of ticker -> weight percentage."""
        return {c.ticker: c.benchmark_weight_pct for c in self.constituents}

