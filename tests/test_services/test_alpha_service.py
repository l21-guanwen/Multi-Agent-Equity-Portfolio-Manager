"""
Tests for alpha service.
"""

import pytest

from app.services.alpha_service import AlphaService
from app.models.alpha import AlphaScore, AlphaModel


class TestAlphaService:
    """Tests for AlphaService."""

    @pytest.fixture
    def sample_scores(self) -> list[AlphaScore]:
        """Create sample alpha scores."""
        return [
            AlphaScore(ticker="AAPL", alpha_score=0.92, alpha_quintile=1),
            AlphaScore(ticker="MSFT", alpha_score=0.88, alpha_quintile=1),
            AlphaScore(ticker="GOOGL", alpha_score=0.75, alpha_quintile=2),
            AlphaScore(ticker="AMZN", alpha_score=0.65, alpha_quintile=2),
            AlphaScore(ticker="META", alpha_score=0.55, alpha_quintile=3),
            AlphaScore(ticker="TSLA", alpha_score=0.45, alpha_quintile=3),
            AlphaScore(ticker="NFLX", alpha_score=0.35, alpha_quintile=4),
            AlphaScore(ticker="NVDA", alpha_score=0.95, alpha_quintile=1),
            AlphaScore(ticker="AMD", alpha_score=0.25, alpha_quintile=4),
            AlphaScore(ticker="INTC", alpha_score=0.15, alpha_quintile=5),
        ]

    @pytest.fixture
    def sample_model(self, sample_scores: list[AlphaScore]) -> AlphaModel:
        """Create sample alpha model."""
        return AlphaModel(scores=sample_scores)

    def test_rank_by_alpha(
        self,
        alpha_service: AlphaService,
        sample_model: AlphaModel,
    ):
        """Test ranking securities by alpha."""
        ranked = alpha_service.rank_by_alpha(sample_model)
        
        # Should be sorted by alpha_score descending
        scores = [s.alpha_score for s in ranked]
        assert scores == sorted(scores, reverse=True)
        assert ranked[0].ticker == "NVDA"  # Highest alpha

    def test_select_top_n(
        self,
        alpha_service: AlphaService,
        sample_model: AlphaModel,
    ):
        """Test selecting top N securities."""
        top_5 = alpha_service.select_top_n(sample_model, n=5)
        
        assert len(top_5) == 5
        # Should include highest alpha securities
        tickers = [s.ticker for s in top_5]
        assert "NVDA" in tickers
        assert "AAPL" in tickers
        assert "MSFT" in tickers

    def test_filter_by_quintile(
        self,
        alpha_service: AlphaService,
        sample_model: AlphaModel,
    ):
        """Test filtering by quintile."""
        q1 = alpha_service.filter_by_quintile(sample_model, quintile=1)
        
        for score in q1:
            assert score.alpha_quintile == 1

    def test_calculate_alpha_weights(
        self,
        alpha_service: AlphaService,
        sample_model: AlphaModel,
    ):
        """Test calculating alpha-weighted portfolio weights."""
        top_5 = alpha_service.select_top_n(sample_model, n=5)
        weights = alpha_service.calculate_alpha_weights(top_5)
        
        # Weights should sum to 1.0 (100%)
        assert abs(sum(weights.values()) - 1.0) < 0.001
        # Higher alpha should have higher weight
        assert weights["NVDA"] > weights["GOOGL"]

    def test_get_quintile_stats(
        self,
        alpha_service: AlphaService,
        sample_model: AlphaModel,
    ):
        """Test getting quintile statistics."""
        stats = alpha_service.get_quintile_stats(sample_model)
        
        assert 1 in stats
        assert stats[1]["count"] == 3  # NVDA, AAPL, MSFT
        assert stats[1]["avg_score"] > stats[5]["avg_score"]

    def test_normalize_scores(
        self,
        alpha_service: AlphaService,
        sample_scores: list[AlphaScore],
    ):
        """Test normalizing alpha scores."""
        normalized = alpha_service.normalize_scores(sample_scores)
        
        # Normalized scores should be between 0 and 1
        for score in normalized:
            assert 0 <= score.alpha_score <= 1

    def test_calculate_score_spread(
        self,
        alpha_service: AlphaService,
        sample_model: AlphaModel,
    ):
        """Test calculating alpha score spread."""
        spread = alpha_service.calculate_score_spread(sample_model)
        
        # Spread is max - min
        assert spread == 0.95 - 0.15

    def test_get_alpha_vector(
        self,
        alpha_service: AlphaService,
        sample_model: AlphaModel,
    ):
        """Test getting alpha vector for optimization."""
        tickers = ["AAPL", "MSFT", "GOOGL"]
        vector = alpha_service.get_alpha_vector(sample_model, tickers)
        
        assert len(vector) == 3
        # Should match order of tickers
        assert vector[0] == 0.92  # AAPL
        assert vector[1] == 0.88  # MSFT
        assert vector[2] == 0.75  # GOOGL

