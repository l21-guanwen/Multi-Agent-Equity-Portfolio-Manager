"""Portfolio domain models."""

from datetime import date
from typing import Optional

from pydantic import BaseModel, Field, computed_field


class PortfolioHolding(BaseModel):
    """
    A single holding in a portfolio.
    
    Based on 03_Portfolio_25_Holdings.csv schema.
    """

    # Portfolio Info
    portfolio_id: str = Field(..., description="Portfolio identifier")
    portfolio_name: str = Field(..., description="Portfolio display name")
    
    # Security Info
    ticker: str = Field(..., description="Stock ticker symbol")
    security_name: str = Field(..., description="Full security name")
    gics_sector: str = Field(..., description="GICS Level 1 sector")
    gics_industry: Optional[str] = Field(None, description="GICS industry group")
    
    # Position Data
    shares: int = Field(..., ge=0, description="Number of shares held")
    price: float = Field(..., ge=0, description="Current price (USD)")
    market_value_usd: float = Field(..., ge=0, description="Position market value (USD)")
    
    # Weights
    portfolio_weight_pct: float = Field(..., ge=0, le=100, description="Weight in portfolio (%)")
    benchmark_weight_pct: float = Field(..., ge=0, le=100, description="Weight in benchmark (%)")
    active_weight_pct: float = Field(..., description="Active weight vs benchmark (%)")
    
    # Alpha Info
    alpha_score: float = Field(..., ge=0, le=1, description="Alpha model score (0-1)")
    alpha_quintile: int = Field(..., ge=1, le=5, description="Alpha quintile (1=best)")
    
    # Metadata
    strategy: Optional[str] = Field(None, description="Strategy name")
    as_of_date: date = Field(..., description="Data as-of date")

    @computed_field
    @property
    def portfolio_weight_decimal(self) -> float:
        """Portfolio weight as decimal (0-1)."""
        return self.portfolio_weight_pct / 100.0

    @computed_field
    @property
    def benchmark_weight_decimal(self) -> float:
        """Benchmark weight as decimal (0-1)."""
        return self.benchmark_weight_pct / 100.0

    @computed_field
    @property
    def active_weight_decimal(self) -> float:
        """Active weight as decimal."""
        return self.active_weight_pct / 100.0

    class Config:
        """Pydantic model configuration."""
        
        frozen = False
        str_strip_whitespace = True


class Portfolio(BaseModel):
    """
    Portfolio containing multiple holdings.
    
    Represents a concentrated equity portfolio.
    """

    portfolio_id: str = Field(..., description="Portfolio identifier")
    portfolio_name: str = Field(..., description="Portfolio display name")
    holdings: list[PortfolioHolding] = Field(default_factory=list)
    as_of_date: date = Field(..., description="Data as-of date")
    strategy: Optional[str] = Field(None, description="Strategy name")

    @computed_field
    @property
    def total_market_value(self) -> float:
        """Total market value of all holdings."""
        return sum(h.market_value_usd for h in self.holdings)

    @computed_field
    @property
    def total_weight_pct(self) -> float:
        """Total weight of all holdings (should be ~100%)."""
        return sum(h.portfolio_weight_pct for h in self.holdings)

    @computed_field
    @property
    def holding_count(self) -> int:
        """Number of holdings."""
        return len(self.holdings)

    @computed_field
    @property
    def average_alpha_score(self) -> float:
        """Average alpha score across holdings."""
        if not self.holdings:
            return 0.0
        return sum(h.alpha_score for h in self.holdings) / len(self.holdings)

    def get_holding(self, ticker: str) -> Optional[PortfolioHolding]:
        """Get a holding by ticker."""
        for holding in self.holdings:
            if holding.ticker == ticker:
                return holding
        return None

    def get_sector_weights(self) -> dict[str, float]:
        """Get total portfolio weight by GICS sector."""
        sector_weights: dict[str, float] = {}
        for holding in self.holdings:
            sector = holding.gics_sector
            sector_weights[sector] = sector_weights.get(sector, 0.0) + holding.portfolio_weight_pct
        return sector_weights

    def get_weight_dict(self) -> dict[str, float]:
        """Get dictionary of ticker -> weight percentage."""
        return {h.ticker: h.portfolio_weight_pct for h in self.holdings}

    def get_active_weight_dict(self) -> dict[str, float]:
        """Get dictionary of ticker -> active weight percentage."""
        return {h.ticker: h.active_weight_pct for h in self.holdings}

