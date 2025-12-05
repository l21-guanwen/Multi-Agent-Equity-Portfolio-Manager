"""
Tests for transaction cost repository.
"""

import pytest

from app.repositories.csv.csv_transaction_cost_repository import (
    CSVTransactionCostRepository,
)


class TestCSVTransactionCostRepository:
    """Tests for CSVTransactionCostRepository."""

    @pytest.mark.asyncio
    async def test_get_all_returns_costs(
        self,
        transaction_cost_repository: CSVTransactionCostRepository,
    ):
        """Test that get_all returns transaction costs."""
        costs = await transaction_cost_repository.get_all()
        
        assert len(costs) > 0
        assert len(costs) == 500

    @pytest.mark.asyncio
    async def test_get_cost_by_ticker(
        self,
        transaction_cost_repository: CSVTransactionCostRepository,
    ):
        """Test getting transaction cost by ticker."""
        cost = await transaction_cost_repository.get_cost("AAPL")
        
        assert cost is not None
        assert cost.ticker == "AAPL"
        assert cost.bid_ask_spread_bps >= 0
        assert cost.commission_bps >= 0
        assert cost.market_impact_bps >= 0
        assert cost.total_cost_bps > 0

    @pytest.mark.asyncio
    async def test_get_cost_model(
        self,
        transaction_cost_repository: CSVTransactionCostRepository,
    ):
        """Test getting full cost model."""
        model = await transaction_cost_repository.get_cost_model()
        
        assert model is not None
        assert model.security_count == 500
        assert model.avg_bid_ask_spread_bps > 0
        assert model.avg_total_cost_bps > 0

    @pytest.mark.asyncio
    async def test_get_total_cost_dict(
        self,
        transaction_cost_repository: CSVTransactionCostRepository,
    ):
        """Test getting total cost dictionary."""
        costs = await transaction_cost_repository.get_total_cost_dict()
        
        assert len(costs) == 500
        for ticker, cost in costs.items():
            assert cost > 0

    @pytest.mark.asyncio
    async def test_get_costs_sorted_by_total(
        self,
        transaction_cost_repository: CSVTransactionCostRepository,
    ):
        """Test getting costs sorted by total cost."""
        costs = await transaction_cost_repository.get_costs_sorted_by_total()
        
        assert len(costs) == 500
        # Should be sorted by total_cost_bps ascending
        total_costs = [c.total_cost_bps for c in costs]
        assert total_costs == sorted(total_costs)

    @pytest.mark.asyncio
    async def test_cost_components_sum_to_total(
        self,
        transaction_cost_repository: CSVTransactionCostRepository,
    ):
        """Test that cost components sum to total."""
        costs = await transaction_cost_repository.get_all()
        
        for cost in costs:
            expected_total = (
                cost.bid_ask_spread_bps +
                cost.commission_bps +
                cost.market_impact_bps
            )
            assert abs(cost.total_cost_bps - expected_total) < 0.01

    @pytest.mark.asyncio
    async def test_avg_costs(
        self,
        transaction_cost_repository: CSVTransactionCostRepository,
    ):
        """Test average cost calculations."""
        avg_total = await transaction_cost_repository.get_avg_total_cost()
        avg_spread = await transaction_cost_repository.get_avg_bid_ask_spread()
        
        assert avg_total > 0
        assert avg_spread > 0
        assert avg_spread < avg_total  # Spread is just one component

