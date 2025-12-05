"""
Tests for data service.
"""

import pytest

from app.services.data_service import DataService


class TestDataService:
    """Tests for DataService."""

    @pytest.mark.asyncio
    async def test_load_benchmark(
        self,
        data_service: DataService,
    ):
        """Test loading benchmark data."""
        benchmark = await data_service.load_benchmark()
        
        assert benchmark is not None
        assert benchmark.benchmark_id == "SPX"
        assert benchmark.security_count == 500

    @pytest.mark.asyncio
    async def test_load_universe(
        self,
        data_service: DataService,
    ):
        """Test loading universe data."""
        universe = await data_service.load_universe()
        
        assert len(universe) == 500
        # Check sample security
        aapl = next((s for s in universe if s.ticker == "AAPL"), None)
        assert aapl is not None
        assert aapl.security_name == "Apple Inc."

    @pytest.mark.asyncio
    async def test_load_alpha_scores(
        self,
        data_service: DataService,
    ):
        """Test loading alpha scores."""
        alpha_model = await data_service.load_alpha_scores()
        
        assert alpha_model is not None
        assert alpha_model.security_count == 500
        assert len(alpha_model.scores) == 500

    @pytest.mark.asyncio
    async def test_load_risk_model(
        self,
        data_service: DataService,
    ):
        """Test loading risk model."""
        risk_model = await data_service.load_risk_model()
        
        assert risk_model is not None
        assert len(risk_model.factor_loadings) == 500
        assert len(risk_model.factor_returns) == 8
        assert risk_model.factor_covariance.matrix.shape == (8, 8)

    @pytest.mark.asyncio
    async def test_load_constraints(
        self,
        data_service: DataService,
    ):
        """Test loading constraints."""
        constraints = await data_service.load_constraints()
        
        assert len(constraints) > 0
        # Should have both stock and sector constraints
        types = {c.constraint_type for c in constraints}
        assert "single_stock_active" in types
        assert "sector_active" in types

    @pytest.mark.asyncio
    async def test_load_transaction_costs(
        self,
        data_service: DataService,
    ):
        """Test loading transaction costs."""
        cost_model = await data_service.load_transaction_costs()
        
        assert cost_model is not None
        assert cost_model.security_count == 500
        assert cost_model.avg_total_cost_bps > 0

    @pytest.mark.asyncio
    async def test_load_all_data(
        self,
        data_service: DataService,
    ):
        """Test loading all data at once."""
        data = await data_service.load_all_data()
        
        assert "benchmark" in data
        assert "universe" in data
        assert "alpha_model" in data
        assert "risk_model" in data
        assert "constraints" in data
        assert "transaction_costs" in data

    @pytest.mark.asyncio
    async def test_get_sector_mapping(
        self,
        data_service: DataService,
    ):
        """Test getting sector mapping."""
        mapping = await data_service.get_sector_mapping()
        
        assert len(mapping) == 500
        assert mapping["AAPL"] == "Information Technology"
        assert mapping["JPM"] == "Financials"

    @pytest.mark.asyncio
    async def test_get_benchmark_weights(
        self,
        data_service: DataService,
    ):
        """Test getting benchmark weights."""
        weights = await data_service.get_benchmark_weights()
        
        assert len(weights) == 500
        # Total should be approximately 100%
        total = sum(weights.values())
        assert 99.0 < total < 101.0

