"""
Tests for Alpha Agent (LangGraph ReAct-based).
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from app.agents.alpha_agent import AlphaAgent
from app.agents.state import PortfolioState
from app.services.alpha_service import AlphaService


class TestAlphaAgent:
    """Tests for AlphaAgent with LangGraph ReAct pattern."""

    @pytest.fixture
    def alpha_agent(self) -> AlphaAgent:
        """Create alpha agent without LLM."""
        return AlphaAgent(
            alpha_service=AlphaService(),
            llm_provider=None,
        )

    @pytest.fixture
    def initial_state(self) -> PortfolioState:
        """Create initial portfolio state with alpha data."""
        return PortfolioState(
            portfolio_id="TEST",
            portfolio_size=25,
            data_validation_passed=True,
            use_llm=False,
            alpha_scores={f"STOCK{i:02d}": 0.95 - i * 0.02 for i in range(50)},
            alpha_quintiles={f"STOCK{i:02d}": 1 if i < 30 else 2 for i in range(50)},
            sector_mapping={f"STOCK{i:02d}": "Technology" if i < 20 else "Financials" for i in range(50)},
        )

    @pytest.mark.asyncio
    async def test_call_selects_top_securities(
        self,
        alpha_agent: AlphaAgent,
        initial_state: PortfolioState,
    ):
        """Test that __call__ selects top securities by alpha."""
        result = await alpha_agent(initial_state)
        
        assert "selected_tickers" in result
        assert len(result["selected_tickers"]) == 25
        assert result["current_agent"] == "alpha_agent"

    @pytest.mark.asyncio
    async def test_selected_tickers_are_from_q1(
        self,
        alpha_agent: AlphaAgent,
        initial_state: PortfolioState,
    ):
        """Test that selected tickers are from Quintile 1."""
        result = await alpha_agent(initial_state)
        
        selected = result["selected_tickers"]
        # All selected should be Q1 (quintile == 1)
        for ticker in selected:
            assert initial_state.alpha_quintiles.get(ticker) == 1

    @pytest.mark.asyncio
    async def test_skips_when_data_validation_failed(
        self,
        alpha_agent: AlphaAgent,
    ):
        """Test that agent skips when data validation failed."""
        state = PortfolioState(
            portfolio_id="TEST",
            data_validation_passed=False,
            use_llm=False,
        )
        
        result = await alpha_agent(state)
        
        assert "selected_tickers" not in result or not result.get("selected_tickers")
        assert "Skipped" in str(result.get("execution_log", []))

    @pytest.mark.asyncio
    async def test_generates_analysis(
        self,
        alpha_agent: AlphaAgent,
        initial_state: PortfolioState,
    ):
        """Test that agent generates alpha analysis."""
        result = await alpha_agent(initial_state)
        
        assert "alpha_analysis" in result
        assert len(result["alpha_analysis"]) > 0

    @pytest.mark.asyncio
    async def test_execution_log_populated(
        self,
        alpha_agent: AlphaAgent,
        initial_state: PortfolioState,
    ):
        """Test that execution log is populated."""
        result = await alpha_agent(initial_state)
        
        assert "execution_log" in result
        assert len(result["execution_log"]) > 0
        assert any("AlphaAgent" in log for log in result["execution_log"])

    @pytest.mark.asyncio
    async def test_respects_portfolio_size(
        self,
        alpha_agent: AlphaAgent,
    ):
        """Test that agent respects portfolio size from state."""
        state = PortfolioState(
            portfolio_id="TEST",
            portfolio_size=10,
            data_validation_passed=True,
            use_llm=False,
            alpha_scores={f"STOCK{i:02d}": 0.95 - i * 0.02 for i in range(50)},
            alpha_quintiles={f"STOCK{i:02d}": 1 for i in range(50)},
            sector_mapping={f"STOCK{i:02d}": "Tech" for i in range(50)},
        )
        
        result = await alpha_agent(state)
        
        assert len(result["selected_tickers"]) == 10


class TestAlphaAgentTools:
    """Tests for AlphaAgent tool handling."""

    def test_agent_has_tools(self):
        """Test that agent has the alpha loading tools."""
        agent = AlphaAgent(
            alpha_service=AlphaService(),
            llm_provider=None,
        )
        
        assert len(agent._tools) >= 1
        tool_names = [t.name for t in agent._tools]
        assert "load_alpha_scores" in tool_names

    def test_get_langchain_llm_returns_none_without_provider(self):
        """Test that _get_langchain_llm returns None without provider."""
        agent = AlphaAgent(
            alpha_service=AlphaService(),
            llm_provider=None,
        )
        
        assert agent._get_langchain_llm() is None
