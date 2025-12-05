"""
CSV-based transaction cost repository implementation.

Loads transaction cost model data from CSV files.
"""

from datetime import date
from typing import Optional

from app.core.constants import DataFileName
from app.models.transaction_cost import TransactionCost, TransactionCostModel
from app.repositories.interfaces.transaction_cost_repository import ITransactionCostRepository
from app.utils.csv_loader import CSVLoader


class CSVTransactionCostRepository(ITransactionCostRepository):
    """
    CSV implementation of the transaction cost repository.
    
    Loads transaction costs from 09_Transaction_Cost_Model.csv.
    """

    COLUMN_MAPPING = {
        "Ticker": "ticker",
        "Security_Name": "security_name",
        "GICS_Sector": "gics_sector",
        "Bid_Ask_Spread_Bps": "bid_ask_spread_bps",
        "Commission_Bps": "commission_bps",
        "Market_Impact_Bps_Per_1M": "market_impact_bps_per_1m",
        "Avg_Daily_Volume_M": "avg_daily_volume_m",
        "Avg_Daily_Dollar_Volume_M": "avg_daily_dollar_volume_m",
        "Liquidity_Bucket": "liquidity_bucket",
        "Total_Oneway_Cost_Bps": "total_oneway_cost_bps",
        "Total_Roundtrip_Cost_Bps": "total_roundtrip_cost_bps",
        "Cost_Low_Urgency_Bps": "cost_low_urgency_bps",
        "Cost_Medium_Urgency_Bps": "cost_medium_urgency_bps",
        "Cost_High_Urgency_Bps": "cost_high_urgency_bps",
        "Max_Participation_Rate_Pct": "max_participation_rate_pct",
        "Days_To_Liquidate": "days_to_liquidate",
        "TCost_Model_ID": "tcost_model_id",
        "Model_Name": "model_name",
        "As_Of_Date": "as_of_date",
    }

    def __init__(self, csv_loader: Optional[CSVLoader] = None):
        """Initialize the repository."""
        self._loader = csv_loader or CSVLoader()
        self._cache: Optional[list[TransactionCost]] = None

    async def _load_data(self) -> list[TransactionCost]:
        """Load and cache transaction cost data."""
        if self._cache is None:
            self._cache = self._loader.load_as_models(
                DataFileName.TRANSACTION_COSTS,
                TransactionCost,
                self.COLUMN_MAPPING,
            )
        return self._cache

    def clear_cache(self):
        """Clear the data cache."""
        self._cache = None

    async def get_all(self, as_of_date: Optional[date] = None) -> list[TransactionCost]:
        """Get all transaction costs."""
        data = await self._load_data()
        if as_of_date:
            return [c for c in data if c.as_of_date == as_of_date]
        return data

    async def get_by_id(self, id: str) -> Optional[TransactionCost]:
        """Get transaction cost by ticker."""
        return await self.get_cost(id)

    async def get_by_ids(self, ids: list[str]) -> list[TransactionCost]:
        """Get transaction costs by multiple tickers."""
        data = await self._load_data()
        return [c for c in data if c.ticker in ids]

    async def get_transaction_cost_model(
        self,
        model_id: str = "TCOST_MARKET_IMPACT_V1",
        as_of_date: Optional[date] = None,
    ) -> Optional[TransactionCostModel]:
        """Get the full transaction cost model."""
        costs = await self.get_all(as_of_date)
        
        if not costs:
            return None
        
        # Get model info from first cost
        actual_model_id = costs[0].tcost_model_id if costs else model_id
        model_name = costs[0].model_name if costs and costs[0].model_name else "Market Impact Transaction Cost Model"
        actual_date = as_of_date or costs[0].as_of_date
        
        return TransactionCostModel(
            tcost_model_id=actual_model_id,
            model_name=model_name,
            costs=costs,
            as_of_date=actual_date,
        )

    async def get_cost(
        self,
        ticker: str,
        as_of_date: Optional[date] = None,
    ) -> Optional[TransactionCost]:
        """Get transaction cost for a single security."""
        data = await self.get_all(as_of_date)
        for cost in data:
            if cost.ticker == ticker:
                return cost
        return None

    async def get_costs_by_liquidity_bucket(
        self,
        bucket: int,
        as_of_date: Optional[date] = None,
    ) -> list[TransactionCost]:
        """Get all securities in a specific liquidity bucket."""
        data = await self.get_all(as_of_date)
        return [c for c in data if c.liquidity_bucket == bucket]

    async def get_cost_dict(
        self,
        urgency: str = "medium",
        as_of_date: Optional[date] = None,
    ) -> dict[str, float]:
        """Get dictionary of ticker to transaction cost."""
        data = await self.get_all(as_of_date)
        
        if urgency == "low":
            return {c.ticker: c.cost_low_urgency_bps for c in data}
        elif urgency == "high":
            return {c.ticker: c.cost_high_urgency_bps for c in data}
        else:
            return {c.ticker: c.cost_medium_urgency_bps for c in data}

    async def get_liquid_securities(
        self,
        max_bucket: int = 2,
        as_of_date: Optional[date] = None,
    ) -> list[TransactionCost]:
        """Get securities in liquid buckets."""
        data = await self.get_all(as_of_date)
        return [c for c in data if c.liquidity_bucket <= max_bucket]

    async def get_average_cost_by_sector(
        self,
        urgency: str = "medium",
        as_of_date: Optional[date] = None,
    ) -> dict[str, float]:
        """Get average transaction cost by sector."""
        data = await self.get_all(as_of_date)
        
        sector_costs: dict[str, list[float]] = {}
        
        for cost in data:
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

