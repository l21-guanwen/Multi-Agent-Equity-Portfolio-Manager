"""
Tests for benchmark repository.
"""

import pytest

from app.repositories.csv.csv_benchmark_repository import CSVBenchmarkRepository


class TestCSVBenchmarkRepository:
    """Tests for CSVBenchmarkRepository."""

    @pytest.mark.asyncio
    async def test_get_all_returns_constituents(
        self,
        benchmark_repository: CSVBenchmarkRepository,
    ):
        """Test that get_all returns benchmark constituents."""
        constituents = await benchmark_repository.get_all()
        
        assert len(constituents) > 0
        assert len(constituents) == 500  # S&P 500

    @pytest.mark.asyncio
    async def test_get_benchmark_returns_full_benchmark(
        self,
        benchmark_repository: CSVBenchmarkRepository,
    ):
        """Test that get_benchmark returns full benchmark object."""
        benchmark = await benchmark_repository.get_benchmark()
        
        assert benchmark is not None
        assert benchmark.benchmark_id == "SPX"
        assert benchmark.security_count == 500
        # Total weight should be approximately 100%
        assert 99.0 < benchmark.total_weight_pct < 101.0

    @pytest.mark.asyncio
    async def test_get_constituent_by_ticker(
        self,
        benchmark_repository: CSVBenchmarkRepository,
    ):
        """Test getting a single constituent by ticker."""
        constituent = await benchmark_repository.get_constituent("AAPL")
        
        assert constituent is not None
        assert constituent.ticker == "AAPL"
        assert constituent.gics_sector == "Information Technology"
        assert constituent.benchmark_weight_pct > 0

    @pytest.mark.asyncio
    async def test_get_constituent_not_found(
        self,
        benchmark_repository: CSVBenchmarkRepository,
    ):
        """Test getting a non-existent constituent."""
        constituent = await benchmark_repository.get_constituent("INVALID")
        
        assert constituent is None

    @pytest.mark.asyncio
    async def test_get_sector_weights(
        self,
        benchmark_repository: CSVBenchmarkRepository,
    ):
        """Test getting sector weights."""
        sector_weights = await benchmark_repository.get_sector_weights()
        
        assert len(sector_weights) == 11  # 11 GICS sectors
        assert "Information Technology" in sector_weights
        # Total should be approximately 100%
        total = sum(sector_weights.values())
        assert 99.0 < total < 101.0

    @pytest.mark.asyncio
    async def test_get_top_constituents(
        self,
        benchmark_repository: CSVBenchmarkRepository,
    ):
        """Test getting top constituents by weight."""
        top_10 = await benchmark_repository.get_top_constituents(n=10)
        
        assert len(top_10) == 10
        # Should be sorted by weight descending
        weights = [c.benchmark_weight_pct for c in top_10]
        assert weights == sorted(weights, reverse=True)

    @pytest.mark.asyncio
    async def test_get_constituents_by_sector(
        self,
        benchmark_repository: CSVBenchmarkRepository,
    ):
        """Test getting constituents by sector."""
        tech_stocks = await benchmark_repository.get_constituents_by_sector(
            "Information Technology"
        )
        
        assert len(tech_stocks) > 0
        for stock in tech_stocks:
            assert stock.gics_sector == "Information Technology"

    @pytest.mark.asyncio
    async def test_get_weight_dict(
        self,
        benchmark_repository: CSVBenchmarkRepository,
    ):
        """Test getting weight dictionary."""
        weights = await benchmark_repository.get_weight_dict()
        
        assert len(weights) == 500
        assert "AAPL" in weights
        assert weights["AAPL"] > 0

