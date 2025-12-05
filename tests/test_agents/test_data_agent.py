"""
Tests for data agent.
"""

import pytest
from datetime import date
from unittest.mock import AsyncMock, MagicMock

from app.agents.data_agent import DataAgent
from app.agents.state import create_initial_state
from app.services.data_service import DataService


class TestDataAgent:
    """Tests for DataAgent."""

    @pytest.fixture
    def mock_data_service(self) -> MagicMock:
        """Create mock data service."""
        service = MagicMock(spec=DataService)
        
        # Mock async methods
        service.load_benchmark = AsyncMock(return_value=MagicMock(
            benchmark_id="SPX",
            security_count=500,
        ))
        service.load_universe = AsyncMock(return_value=[
            MagicMock(ticker="AAPL"),
            MagicMock(ticker="MSFT"),
        ])
        service.load_alpha_scores = AsyncMock(return_value=MagicMock(
            security_count=500,
        ))
        service.load_risk_model = AsyncMock(return_value=MagicMock(
            factor_loadings=[],
        ))
        service.load_constraints = AsyncMock(return_value=[])
        service.load_transaction_costs = AsyncMock(return_value=MagicMock(
            security_count=500,
        ))
        
        return service

    @pytest.fixture
    def data_agent(self, mock_data_service: MagicMock) -> DataAgent:
        """Create data agent with mock service."""
        return DataAgent(
            data_service=mock_data_service,
            llm_provider=None,  # No LLM for unit tests
        )

    @pytest.mark.asyncio
    async def test_process_loads_all_data(
        self,
        data_agent: DataAgent,
        mock_data_service: MagicMock,
    ):
        """Test that process loads all required data."""
        state = create_initial_state(
            portfolio_id="test",
            as_of_date=date(2024, 1, 1),
        )
        
        result = await data_agent.process(state)
        
        # Verify all data was loaded
        mock_data_service.load_benchmark.assert_called_once()
        mock_data_service.load_universe.assert_called_once()
        mock_data_service.load_alpha_scores.assert_called_once()
        mock_data_service.load_risk_model.assert_called_once()
        mock_data_service.load_constraints.assert_called_once()
        mock_data_service.load_transaction_costs.assert_called_once()

    @pytest.mark.asyncio
    async def test_process_updates_state(
        self,
        data_agent: DataAgent,
    ):
        """Test that process updates state correctly."""
        state = create_initial_state(
            portfolio_id="test",
            as_of_date=date(2024, 1, 1),
        )
        
        result = await data_agent.process(state)
        
        assert result["benchmark"] is not None
        assert result["universe"] is not None
        assert result["alpha_model"] is not None
        assert result["risk_model"] is not None
        assert result["constraints"] is not None
        assert result["transaction_costs"] is not None
        assert result["current_step"] == "data_loaded"

    @pytest.mark.asyncio
    async def test_process_handles_errors(
        self,
        mock_data_service: MagicMock,
    ):
        """Test that process handles errors gracefully."""
        mock_data_service.load_benchmark = AsyncMock(
            side_effect=Exception("Failed to load benchmark")
        )
        
        agent = DataAgent(
            data_service=mock_data_service,
            llm_provider=None,
        )
        
        state = create_initial_state(
            portfolio_id="test",
            as_of_date=date(2024, 1, 1),
        )
        
        result = await agent.process(state)
        
        assert len(result["errors"]) > 0
        assert "Failed to load benchmark" in result["errors"][0]

