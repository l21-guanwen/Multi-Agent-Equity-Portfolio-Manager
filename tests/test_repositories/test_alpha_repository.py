"""
Tests for alpha repository.
"""

import pytest

from app.repositories.csv.csv_alpha_repository import CSVAlphaRepository


class TestCSVAlphaRepository:
    """Tests for CSVAlphaRepository."""

    @pytest.mark.asyncio
    async def test_get_all_returns_scores(
        self,
        alpha_repository: CSVAlphaRepository,
    ):
        """Test that get_all returns alpha scores."""
        scores = await alpha_repository.get_all()
        
        assert len(scores) > 0
        assert len(scores) == 500

    @pytest.mark.asyncio
    async def test_get_alpha_model(
        self,
        alpha_repository: CSVAlphaRepository,
    ):
        """Test getting full alpha model."""
        model = await alpha_repository.get_alpha_model()
        
        assert model is not None
        assert model.security_count == 500

    @pytest.mark.asyncio
    async def test_get_score_by_ticker(
        self,
        alpha_repository: CSVAlphaRepository,
    ):
        """Test getting score by ticker."""
        score = await alpha_repository.get_score("AAPL")
        
        assert score is not None
        assert score.ticker == "AAPL"
        assert 0 <= score.alpha_score <= 1
        assert 1 <= score.alpha_quintile <= 5

    @pytest.mark.asyncio
    async def test_get_top_scores(
        self,
        alpha_repository: CSVAlphaRepository,
    ):
        """Test getting top scores."""
        top_25 = await alpha_repository.get_top_scores(n=25)
        
        assert len(top_25) == 25
        # Should be sorted by alpha score descending
        scores = [s.alpha_score for s in top_25]
        assert scores == sorted(scores, reverse=True)

    @pytest.mark.asyncio
    async def test_get_scores_by_quintile(
        self,
        alpha_repository: CSVAlphaRepository,
    ):
        """Test getting scores by quintile."""
        q1_scores = await alpha_repository.get_scores_by_quintile(1)
        
        assert len(q1_scores) > 0
        for score in q1_scores:
            assert score.alpha_quintile == 1
            assert score.alpha_score >= 0.8  # Q1 is 0.8-1.0

    @pytest.mark.asyncio
    async def test_get_quintile_distribution(
        self,
        alpha_repository: CSVAlphaRepository,
    ):
        """Test getting quintile distribution."""
        distribution = await alpha_repository.get_quintile_distribution()
        
        assert len(distribution) == 5
        # Each quintile should have approximately 100 securities (20% of 500)
        for q in range(1, 6):
            assert distribution[q] > 0
        # Total should be 500
        assert sum(distribution.values()) == 500

    @pytest.mark.asyncio
    async def test_get_score_dict(
        self,
        alpha_repository: CSVAlphaRepository,
    ):
        """Test getting score dictionary."""
        scores = await alpha_repository.get_score_dict()
        
        assert len(scores) == 500
        for ticker, score in scores.items():
            assert 0 <= score <= 1

