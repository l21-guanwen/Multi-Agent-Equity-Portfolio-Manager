"""Security domain model."""

from datetime import date
from typing import Optional

from pydantic import BaseModel, Field


class Security(BaseModel):
    """
    Base security model representing a tradeable equity.
    
    This is the foundational entity that other models reference.
    """

    ticker: str = Field(..., description="Stock ticker symbol")
    security_name: str = Field(..., description="Full security name")
    security_id: Optional[str] = Field(None, description="Unique security identifier")
    isin: Optional[str] = Field(None, description="International Security ID (ISIN)")
    cusip: Optional[str] = Field(None, description="CUSIP identifier")
    
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
    
    # Metadata
    exchange: Optional[str] = Field(None, description="Exchange (e.g., NYSE, NASDAQ)")
    currency: str = Field(default="USD", description="Trading currency")
    as_of_date: date = Field(..., description="Data as-of date")

    class Config:
        """Pydantic model configuration."""
        
        frozen = False
        str_strip_whitespace = True

    def __hash__(self) -> int:
        """Make security hashable by ticker."""
        return hash(self.ticker)

    def __eq__(self, other: object) -> bool:
        """Equality based on ticker."""
        if isinstance(other, Security):
            return self.ticker == other.ticker
        return False

