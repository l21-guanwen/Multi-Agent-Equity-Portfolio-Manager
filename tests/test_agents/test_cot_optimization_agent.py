"""
Tests for Chain-of-Thought Optimization Agent.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock

from app.agents.cot_optimization_agent import ChainOfThoughtOptimizationAgent
from app.agents.state import PortfolioState
from app.models.optimization import OptimizationResult


class TestChainOfThoughtOptimizationAgent:
    """Tests for CoT Optimization Agent."""

    @pytest.fixture
    def mock_services(self):
        """Create mock services."""
        opt_service = MagicMock()
        opt_service.optimize_portfolio = AsyncMock(return_value=OptimizationResult(
            weights={"STOCK00": 0.1, "STOCK01": 0.1, "STOCK02": 0.1},
            objective_value=0.5,
            status="optimal",
            solver_name="cvxpy",
            expected_alpha=0.85,
            expected_risk=0.15,
        ))
        
        risk_repo = MagicMock()
        risk_repo.get_risk_model = AsyncMock(return_value=MagicMock(
            security_count=500,
            get_loadings=MagicMock(return_value=MagicMock()),
        ))
        
        constraint_repo = MagicMock()
        constraint_repo.get_constraint_set = AsyncMock(return_value=MagicMock())
        
        tcost_repo = MagicMock()
        tcost_repo.get_transaction_cost_model = AsyncMock(return_value=MagicMock())
        
        return {
            "optimization_service": opt_service,
            "risk_repository": risk_repo,
            "constraint_repository": constraint_repo,
            "transaction_cost_repository": tcost_repo,
        }

    @pytest.fixture
    def agent(self, mock_services) -> ChainOfThoughtOptimizationAgent:
        """Create agent without LLM."""
        return ChainOfThoughtOptimizationAgent(
            optimization_service=mock_services["optimization_service"],
            risk_repository=mock_services["risk_repository"],
            constraint_repository=mock_services["constraint_repository"],
            transaction_cost_repository=mock_services["transaction_cost_repository"],
            llm=None,
        )

    @pytest.fixture
    def state_with_selection(self) -> PortfolioState:
        """Create state with selected securities."""
        return PortfolioState(
            portfolio_id="TEST",
            portfolio_size=25,
            use_llm=False,
            selected_tickers=[f"STOCK{i:02d}" for i in range(30)],
            alpha_scores={f"STOCK{i:02d}": 0.9 - i * 0.01 for i in range(30)},
            benchmark_weights={f"STOCK{i:02d}": 0.2 for i in range(30)},
            sector_mapping={f"STOCK{i:02d}": "Technology" for i in range(30)},
        )

    @pytest.mark.asyncio
    async def test_call_returns_optimal_weights(
        self,
        agent: ChainOfThoughtOptimizationAgent,
        state_with_selection: PortfolioState,
    ):
        """Test that __call__ returns optimal weights."""
        result = await agent(state_with_selection)
        
        assert "optimal_weights" in result
        assert len(result["optimal_weights"]) > 0
        assert result["current_agent"] == "optimization_agent"

    @pytest.mark.asyncio
    async def test_returns_error_without_selected_tickers(
        self,
        agent: ChainOfThoughtOptimizationAgent,
    ):
        """Test that agent returns error without selected tickers."""
        state = PortfolioState(
            portfolio_id="TEST",
            selected_tickers=[],
            use_llm=False,
        )
        
        result = await agent(state)
        
        assert "error_message" in result
        assert "No securities selected" in result["error_message"]

    @pytest.mark.asyncio
    async def test_uses_cvxpy_when_llm_disabled(
        self,
        agent: ChainOfThoughtOptimizationAgent,
        state_with_selection: PortfolioState,
        mock_services,
    ):
        """Test that agent uses CVXPY solver when LLM is disabled."""
        result = await agent(state_with_selection)
        
        # Should have called the optimization service
        mock_services["optimization_service"].optimize_portfolio.assert_called_once()
        assert result["optimization_status"] == "optimal"

    @pytest.mark.asyncio
    async def test_execution_log_shows_solver_usage(
        self,
        agent: ChainOfThoughtOptimizationAgent,
        state_with_selection: PortfolioState,
    ):
        """Test that execution log shows CVXPY usage."""
        result = await agent(state_with_selection)
        
        assert "execution_log" in result
        assert any("CVXPY" in log for log in result["execution_log"])


class TestCoTPromptBuilding:
    """Tests for Chain-of-Thought prompt building."""

    @pytest.fixture
    def agent(self) -> ChainOfThoughtOptimizationAgent:
        """Create agent for prompt testing."""
        return ChainOfThoughtOptimizationAgent(
            optimization_service=MagicMock(),
            risk_repository=MagicMock(),
            constraint_repository=MagicMock(),
            transaction_cost_repository=MagicMock(),
            llm=None,
        )

    def test_build_cot_prompt_includes_securities(self, agent):
        """Test that prompt includes securities data."""
        state = PortfolioState(
            portfolio_id="TEST",
            portfolio_size=25,
            selected_tickers=["AAPL", "MSFT", "GOOGL"],
            alpha_scores={"AAPL": 0.9, "MSFT": 0.85, "GOOGL": 0.8},
            benchmark_weights={"AAPL": 0.067, "MSFT": 0.055, "GOOGL": 0.045},
            sector_mapping={"AAPL": "Technology", "MSFT": "Technology", "GOOGL": "Communication"},
        )
        
        prompt = agent._build_cot_prompt(
            state=state,
            eligible_tickers=["AAPL", "MSFT", "GOOGL"],
            risk_model=None,
            constraint_set=None,
            transaction_cost_model=None,
        )
        
        assert "AAPL" in prompt
        assert "MSFT" in prompt
        assert "portfolio_size" in prompt.lower() or "25" in prompt

    def test_parse_weights_extracts_json(self, agent):
        """Test that weights are parsed from JSON response."""
        response = '''Let me analyze this...
        
```json
{
  "weights": {
    "AAPL": 0.15,
    "MSFT": 0.12,
    "GOOGL": 0.10
  },
  "reasoning_summary": "Selected top alpha stocks"
}
```'''
        
        weights = agent._parse_weights_from_response(
            response, 
            eligible_tickers=["AAPL", "MSFT", "GOOGL", "AMZN"]
        )
        
        assert weights is not None
        assert "AAPL" in weights
        assert abs(sum(weights.values()) - 1.0) < 0.01  # Normalized

