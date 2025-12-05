"""
Tests for alpha agent.
"""

import pytest
from datetime import date
from unittest.mock import MagicMock

from app.agents.alpha_agent import AlphaAgent
from app.agents.state import create_initial_state
from app.services.alpha_service import AlphaService
from app.models.alpha import AlphaScore, AlphaModel


class TestAlphaAgent:
    """Tests for AlphaAgent."""

    @pytest.fixture
    def sample_alpha_model(self) -> AlphaModel:
        """Create sample alpha model."""
        scores = [
            AlphaScore(ticker=f"STOCK{i}", alpha_score=0.9 - i * 0.02, alpha_quintile=1)
            for i in range(30)
        ]
        return AlphaModel(scores=scores)

    @pytest.fixture
    def alpha_agent(self) -> AlphaAgent:
        """Create alpha agent."""
        return AlphaAgent(
            alpha_service=AlphaService(),
            llm_provider=None,
        )

    @pytest.mark.asyncio
    async def test_process_selects_top_securities(
        self,
        alpha_agent: AlphaAgent,
        sample_alpha_model: AlphaModel,
    ):
        """Test that process selects top N securities."""
        state = create_initial_state(
            portfolio_id="test",
            as_of_date=date(2024, 1, 1),
        )
        state["alpha_model"] = sample_alpha_model
        state["current_step"] = "data_loaded"
        
        result = await alpha_agent.process(state, n_securities=25)
        
        assert result["selected_securities"] is not None
        assert len(result["selected_securities"]) == 25
        assert result["current_step"] == "alpha_analyzed"

    @pytest.mark.asyncio
    async def test_selected_securities_sorted_by_alpha(
        self,
        alpha_agent: AlphaAgent,
        sample_alpha_model: AlphaModel,
    ):
        """Test that selected securities are sorted by alpha."""
        state = create_initial_state(
            portfolio_id="test",
            as_of_date=date(2024, 1, 1),
        )
        state["alpha_model"] = sample_alpha_model
        state["current_step"] = "data_loaded"
        
        result = await alpha_agent.process(state, n_securities=10)
        
        securities = result["selected_securities"]
        scores = [s.alpha_score for s in securities]
        assert scores == sorted(scores, reverse=True)

    @pytest.mark.asyncio
    async def test_process_without_alpha_model(
        self,
        alpha_agent: AlphaAgent,
    ):
        """Test that process handles missing alpha model."""
        state = create_initial_state(
            portfolio_id="test",
            as_of_date=date(2024, 1, 1),
        )
        state["alpha_model"] = None
        state["current_step"] = "data_loaded"
        
        result = await alpha_agent.process(state)
        
        assert len(result["errors"]) > 0
        assert "alpha" in result["errors"][0].lower()

    @pytest.mark.asyncio
    async def test_process_calculates_initial_weights(
        self,
        alpha_agent: AlphaAgent,
        sample_alpha_model: AlphaModel,
    ):
        """Test that process calculates initial alpha-weighted weights."""
        state = create_initial_state(
            portfolio_id="test",
            as_of_date=date(2024, 1, 1),
        )
        state["alpha_model"] = sample_alpha_model
        state["current_step"] = "data_loaded"
        
        result = await alpha_agent.process(state, n_securities=10)
        
        # Check that initial weights are stored in llm_insights
        assert "initial_weights" in result.get("llm_insights", {})
        weights = result["llm_insights"]["initial_weights"]
        
        # Weights should sum to 1
        assert abs(sum(weights.values()) - 1.0) < 0.01

