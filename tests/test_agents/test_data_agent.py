"""
Tests for Data Agent (ReAct-based).
"""

import pytest
from unittest.mock import AsyncMock, MagicMock

from app.agents.data_agent import DataAgent
from app.agents.state import PortfolioState
from app.services.data_service import DataService
from app.services.interfaces.data_service import DataValidationResult


class TestDataAgent:
    """Tests for DataAgent ReAct implementation."""

    @pytest.fixture
    def mock_data_service(self) -> MagicMock:
        """Create mock data service."""
        service = MagicMock(spec=DataService)
        
        # Mock load_all_data
        benchmark_mock = MagicMock()
        benchmark_mock.security_count = 500
        benchmark_mock.as_of_date = MagicMock()
        benchmark_mock.as_of_date.isoformat.return_value = "2025-01-01"
        benchmark_mock.constituents = [
            MagicMock(
                ticker=f"STOCK{i:03d}",
                benchmark_weight_pct=0.2,
                gics_sector="Technology",
            )
            for i in range(500)
        ]
        
        alpha_mock = MagicMock()
        alpha_mock.security_count = 500
        alpha_mock.scores = [
            MagicMock(ticker=f"STOCK{i:03d}", alpha_score=0.5, alpha_quintile=3)
            for i in range(500)
        ]
        
        risk_mock = MagicMock()
        risk_mock.security_count = 500
        
        service.load_all_data = AsyncMock(return_value={
            "benchmark": benchmark_mock,
            "alpha_model": alpha_mock,
            "risk_model": risk_mock,
            "constraints": MagicMock(),
            "transaction_costs": MagicMock(security_count=500),
        })
        
        service.validate_data = AsyncMock(return_value=DataValidationResult(
            is_valid=True,
            issues=[],
            warnings=[],
            data_quality_score=0.95,
        ))
        
        return service

    @pytest.fixture
    def data_agent(self, mock_data_service: MagicMock) -> DataAgent:
        """Create data agent with mock service."""
        return DataAgent(
            data_service=mock_data_service,
            llm_provider=None,
        )

    @pytest.fixture
    def initial_state(self) -> PortfolioState:
        """Create initial portfolio state."""
        return PortfolioState(
            portfolio_id="TEST",
            use_llm=False,
        )

    @pytest.mark.asyncio
    async def test_call_loads_data(
        self,
        data_agent: DataAgent,
        initial_state: PortfolioState,
        mock_data_service: MagicMock,
    ):
        """Test that __call__ loads all required data."""
        result = await data_agent(initial_state)
        
        mock_data_service.load_all_data.assert_called_once()
        assert result["current_agent"] == "data_agent"

    @pytest.mark.asyncio
    async def test_populates_universe_tickers(
        self,
        data_agent: DataAgent,
        initial_state: PortfolioState,
    ):
        """Test that universe tickers are populated."""
        result = await data_agent(initial_state)
        
        assert "universe_tickers" in result
        assert len(result["universe_tickers"]) == 500

    @pytest.mark.asyncio
    async def test_populates_benchmark_weights(
        self,
        data_agent: DataAgent,
        initial_state: PortfolioState,
    ):
        """Test that benchmark weights are populated."""
        result = await data_agent(initial_state)
        
        assert "benchmark_weights" in result
        assert len(result["benchmark_weights"]) == 500

    @pytest.mark.asyncio
    async def test_populates_alpha_scores(
        self,
        data_agent: DataAgent,
        initial_state: PortfolioState,
    ):
        """Test that alpha scores are populated."""
        result = await data_agent(initial_state)
        
        assert "alpha_scores" in result
        assert len(result["alpha_scores"]) == 500

    @pytest.mark.asyncio
    async def test_populates_sector_mapping(
        self,
        data_agent: DataAgent,
        initial_state: PortfolioState,
    ):
        """Test that sector mapping is populated."""
        result = await data_agent(initial_state)
        
        assert "sector_mapping" in result
        assert len(result["sector_mapping"]) == 500

    @pytest.mark.asyncio
    async def test_sets_validation_status(
        self,
        data_agent: DataAgent,
        initial_state: PortfolioState,
    ):
        """Test that validation status is set."""
        result = await data_agent(initial_state)
        
        assert "data_validation_passed" in result
        assert result["data_validation_passed"] is True

    @pytest.mark.asyncio
    async def test_handles_validation_failure(
        self,
        mock_data_service: MagicMock,
    ):
        """Test handling of validation failure."""
        mock_data_service.validate_data = AsyncMock(return_value=DataValidationResult(
            is_valid=False,
            issues=["Missing benchmark data"],
            warnings=[],
            data_quality_score=0.3,
        ))
        
        agent = DataAgent(
            data_service=mock_data_service,
            llm_provider=None,
        )
        
        state = PortfolioState(portfolio_id="TEST", use_llm=False)
        result = await agent(state)
        
        assert result["data_validation_passed"] is False
        assert len(result["data_validation_issues"]) > 0

    @pytest.mark.asyncio
    async def test_execution_log_populated(
        self,
        data_agent: DataAgent,
        initial_state: PortfolioState,
    ):
        """Test that execution log is populated."""
        result = await data_agent(initial_state)
        
        assert "execution_log" in result
        assert len(result["execution_log"]) > 0
        assert any("DataAgent" in log for log in result["execution_log"])

    @pytest.mark.asyncio
    async def test_data_summary_created(
        self,
        data_agent: DataAgent,
        initial_state: PortfolioState,
    ):
        """Test that data summary is created."""
        result = await data_agent(initial_state)
        
        assert "data_summary" in result
        assert result["data_summary"]["benchmark_count"] == 500


class TestDataAgentTools:
    """Tests for DataAgent tool handling."""

    def test_agent_has_all_data_tools(self):
        """Test that agent has all required data tools."""
        mock_service = MagicMock(spec=DataService)
        agent = DataAgent(
            data_service=mock_service,
            llm_provider=None,
        )
        
        tools = agent.get_tools()
        tool_names = [t.name for t in tools]
        
        assert "load_benchmark" in tool_names
        assert "load_alpha_scores" in tool_names
        assert "load_risk_model" in tool_names
        assert "load_constraints" in tool_names

    def test_tools_have_descriptions(self):
        """Test that all tools have descriptions."""
        mock_service = MagicMock(spec=DataService)
        agent = DataAgent(
            data_service=mock_service,
            llm_provider=None,
        )
        
        for tool in agent.get_tools():
            assert len(tool.description) > 20
