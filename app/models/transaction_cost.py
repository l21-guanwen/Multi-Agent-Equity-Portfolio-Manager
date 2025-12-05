"""Transaction cost model domain models."""

from datetime import date
from typing import Optional

from pydantic import BaseModel, Field, computed_field


class TransactionCost(BaseModel):
    """
    Transaction cost model for a single security.
    
    Based on 09_Transaction_Cost_Model.csv schema.
    Includes bid-ask spread, commission, and market impact components.
    """

    # Security Info
    ticker: str = Field(..., description="Stock ticker symbol")
    security_name: str = Field(..., description="Full security name")
    gics_sector: str = Field(..., description="GICS Level 1 sector")
    
    # Cost Components (in basis points)
    bid_ask_spread_bps: float = Field(..., ge=0, description="Full bid-ask spread in basis points")
    commission_bps: float = Field(..., ge=0, description="Commission cost in basis points")
    market_impact_bps_per_1m: float = Field(..., ge=0, description="Market impact per $1M traded")
    
    # Liquidity Metrics
    avg_daily_volume_m: float = Field(..., ge=0, description="Average daily volume (millions of shares)")
    avg_daily_dollar_volume_m: float = Field(..., ge=0, description="Average daily dollar volume (millions USD)")
    liquidity_bucket: int = Field(..., ge=1, le=5, description="Liquidity tier (1=Most Liquid, 5=Least)")
    
    # Total Costs (in basis points)
    total_oneway_cost_bps: float = Field(..., ge=0, description="Total one-way transaction cost")
    total_roundtrip_cost_bps: float = Field(..., ge=0, description="Total round-trip (buy+sell) cost")
    
    # Urgency-Based Costs
    cost_low_urgency_bps: float = Field(..., ge=0, description="Cost estimate for patient trading")
    cost_medium_urgency_bps: float = Field(..., ge=0, description="Cost estimate for normal trading")
    cost_high_urgency_bps: float = Field(..., ge=0, description="Cost estimate for urgent trading")
    
    # Trading Constraints
    max_participation_rate_pct: float = Field(..., ge=0, le=100, description="Max % of ADV for low impact")
    days_to_liquidate: float = Field(..., ge=0, description="Days to liquidate a 4% position")
    
    # Model Info
    tcost_model_id: str = Field(..., description="Transaction cost model identifier")
    model_name: Optional[str] = Field(None, description="Model name")
    as_of_date: date = Field(..., description="Data as-of date")

    @computed_field
    @property
    def half_spread_bps(self) -> float:
        """Half spread cost (cost to cross the market)."""
        return self.bid_ask_spread_bps / 2.0

    @computed_field
    @property
    def is_liquid(self) -> bool:
        """Check if security is in liquid buckets (1-2)."""
        return self.liquidity_bucket <= 2

    @computed_field
    @property
    def is_illiquid(self) -> bool:
        """Check if security is in illiquid buckets (4-5)."""
        return self.liquidity_bucket >= 4

    def estimate_cost(
        self,
        trade_value_usd: float,
        urgency: str = "medium",
    ) -> float:
        """
        Estimate transaction cost for a given trade.
        
        Args:
            trade_value_usd: Trade value in USD
            urgency: 'low', 'medium', or 'high'
            
        Returns:
            Estimated cost in USD
        """
        if urgency == "low":
            cost_bps = self.cost_low_urgency_bps
        elif urgency == "high":
            cost_bps = self.cost_high_urgency_bps
        else:
            cost_bps = self.cost_medium_urgency_bps
        
        return trade_value_usd * cost_bps / 10000.0

    def estimate_market_impact(self, trade_value_usd: float) -> float:
        """
        Estimate market impact for a given trade size.
        
        Args:
            trade_value_usd: Trade value in USD
            
        Returns:
            Estimated market impact in basis points
        """
        trade_value_m = trade_value_usd / 1_000_000.0
        return self.market_impact_bps_per_1m * trade_value_m

    class Config:
        """Pydantic model configuration."""
        
        frozen = False
        str_strip_whitespace = True


class TransactionCostModel(BaseModel):
    """
    Collection of transaction costs for the full universe.
    
    Represents a complete transaction cost model.
    """

    tcost_model_id: str = Field(..., description="Transaction cost model identifier")
    model_name: str = Field(default="Market Impact Transaction Cost Model", description="Model name")
    costs: list[TransactionCost] = Field(default_factory=list)
    as_of_date: date = Field(..., description="Model date")

    @computed_field
    @property
    def security_count(self) -> int:
        """Number of securities with cost estimates."""
        return len(self.costs)

    def get_cost(self, ticker: str) -> Optional[TransactionCost]:
        """Get transaction cost by ticker."""
        for cost in self.costs:
            if cost.ticker == ticker:
                return cost
        return None

    def get_cost_dict(self, urgency: str = "medium") -> dict[str, float]:
        """
        Get dictionary of ticker -> transaction cost (bps).
        
        Args:
            urgency: 'low', 'medium', or 'high'
        """
        if urgency == "low":
            return {c.ticker: c.cost_low_urgency_bps for c in self.costs}
        elif urgency == "high":
            return {c.ticker: c.cost_high_urgency_bps for c in self.costs}
        else:
            return {c.ticker: c.cost_medium_urgency_bps for c in self.costs}

    def get_liquidity_buckets(self) -> dict[int, list[str]]:
        """Get tickers grouped by liquidity bucket."""
        buckets: dict[int, list[str]] = {1: [], 2: [], 3: [], 4: [], 5: []}
        for cost in self.costs:
            buckets[cost.liquidity_bucket].append(cost.ticker)
        return buckets

    def get_average_cost_by_sector(self, urgency: str = "medium") -> dict[str, float]:
        """Get average transaction cost by sector."""
        sector_costs: dict[str, list[float]] = {}
        
        for cost in self.costs:
            sector = cost.gics_sector
            if sector not in sector_costs:
                sector_costs[sector] = []
            
            if urgency == "low":
                sector_costs[sector].append(cost.cost_low_urgency_bps)
            elif urgency == "high":
                sector_costs[sector].append(cost.cost_high_urgency_bps)
            else:
                sector_costs[sector].append(cost.cost_medium_urgency_bps)
        
        return {
            sector: sum(costs) / len(costs)
            for sector, costs in sector_costs.items()
            if costs
        }

