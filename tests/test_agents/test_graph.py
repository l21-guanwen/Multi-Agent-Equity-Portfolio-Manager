"""
Tests for LangGraph workflow orchestration.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock

from app.agents.graph import PortfolioGraph, create_portfolio_graph
from app.agents.state import PortfolioState


class TestPortfolioGraph:
    """Tests for PortfolioGraph orchestration."""

    @pytest.fixture
    def mock_services(self):
        """Create mock services for graph."""
        data_service = MagicMock()
        data_service.load_all_data = AsyncMock(return_value={
            "benchmark": MagicMock(
                security_count=500,
                as_of_date=MagicMock(isoformat=lambda: "2025-01-01"),
                constituents=[],
            ),
            "alpha_model": MagicMock(security_count=500, scores=[]),
            "risk_model": MagicMock(security_count=500, factor_loadings=[]),
            "constraints": MagicMock(),
            "transaction_costs": MagicMock(security_count=500),
        })
        data_service.validate_data = AsyncMock(return_value=MagicMock(
            is_valid=True,
            issues=[],
            warnings=[],
            data_quality_score=0.95,
        ))
        
        return {
            "data_service": data_service,
            "alpha_service": MagicMock(),
            "risk_service": MagicMock(),
            "optimization_service": MagicMock(),
            "compliance_service": MagicMock(),
            "risk_repository": MagicMock(),
            "constraint_repository": MagicMock(),
            "transaction_cost_repository": MagicMock(),
        }

    def test_create_portfolio_graph_returns_graph(self, mock_services):
        """Test that create_portfolio_graph returns a PortfolioGraph."""
        graph = create_portfolio_graph(
            data_service=mock_services["data_service"],
            alpha_service=mock_services["alpha_service"],
            risk_service=mock_services["risk_service"],
            optimization_service=mock_services["optimization_service"],
            compliance_service=mock_services["compliance_service"],
            risk_repository=mock_services["risk_repository"],
            constraint_repository=mock_services["constraint_repository"],
            transaction_cost_repository=mock_services["transaction_cost_repository"],
            llm_provider=None,
        )
        
        assert isinstance(graph, PortfolioGraph)

    def test_graph_can_compile(self, mock_services):
        """Test that graph can be compiled."""
        graph = create_portfolio_graph(
            data_service=mock_services["data_service"],
            alpha_service=mock_services["alpha_service"],
            risk_service=mock_services["risk_service"],
            optimization_service=mock_services["optimization_service"],
            compliance_service=mock_services["compliance_service"],
            risk_repository=mock_services["risk_repository"],
            constraint_repository=mock_services["constraint_repository"],
            transaction_cost_repository=mock_services["transaction_cost_repository"],
            llm_provider=None,
        )
        
        compiled = graph.compile()
        assert compiled is not None


class TestPortfolioState:
    """Tests for PortfolioState."""

    def test_create_initial_state(self):
        """Test creating initial state."""
        state = PortfolioState(
            portfolio_id="TEST",
            portfolio_size=25,
        )
        
        assert state.portfolio_id == "TEST"
        assert state.portfolio_size == 25
        assert state.iteration_count == 0
        assert state.is_compliant is False

    def test_state_with_all_fields(self):
        """Test state with all fields populated."""
        state = PortfolioState(
            portfolio_id="TEST",
            as_of_date="2025-01-01",
            portfolio_size=25,
            use_llm=False,
            max_iterations=5,
        )
        
        assert state.portfolio_id == "TEST"
        assert state.as_of_date == "2025-01-01"
        assert state.use_llm is False
        assert state.max_iterations == 5

    def test_state_defaults(self):
        """Test state default values."""
        state = PortfolioState()
        
        assert state.portfolio_id == "ALPHA_GROWTH_25"
        assert state.portfolio_size == 25
        assert state.use_llm is True
        assert state.data_validation_passed is False
        assert state.is_compliant is False

    def test_state_to_summary(self):
        """Test state summary method."""
        state = PortfolioState(
            portfolio_id="TEST",
            data_validation_passed=True,
            selected_tickers=["AAPL", "MSFT", "GOOGL"],
            optimal_weights={"AAPL": 0.4, "MSFT": 0.35, "GOOGL": 0.25},
            is_compliant=True,
            portfolio_risk_pct=15.5,
            expected_alpha=0.92,
        )
        
        summary = state.to_summary()
        
        assert summary["portfolio_id"] == "TEST"
        assert summary["data_validated"] is True
        assert summary["selected_securities"] == 3
        assert summary["optimized_positions"] == 3
        assert summary["is_compliant"] is True


class TestGraphRouting:
    """Tests for graph routing logic."""

    def test_route_after_data_continues_on_success(self):
        """Test routing continues after successful data loading."""
        from app.agents.graph import PortfolioGraph
        
        state = PortfolioState(
            data_validation_passed=True,
            error_message=None,
        )
        
        # Access the routing method via an instance
        mock_services = {
            "data_service": MagicMock(),
            "alpha_service": MagicMock(),
            "risk_service": MagicMock(),
            "optimization_service": MagicMock(),
            "compliance_service": MagicMock(),
            "risk_repository": MagicMock(),
            "constraint_repository": MagicMock(),
            "transaction_cost_repository": MagicMock(),
        }
        
        graph = PortfolioGraph(**mock_services, llm_provider=None)
        result = graph._route_after_data(state)
        
        assert result == "continue"

    def test_route_after_data_errors_on_failure(self):
        """Test routing errors after failed data loading."""
        from app.agents.graph import PortfolioGraph
        
        state = PortfolioState(
            data_validation_passed=False,
            error_message="Data validation failed",
        )
        
        mock_services = {
            "data_service": MagicMock(),
            "alpha_service": MagicMock(),
            "risk_service": MagicMock(),
            "optimization_service": MagicMock(),
            "compliance_service": MagicMock(),
            "risk_repository": MagicMock(),
            "constraint_repository": MagicMock(),
            "transaction_cost_repository": MagicMock(),
        }
        
        graph = PortfolioGraph(**mock_services, llm_provider=None)
        result = graph._route_after_data(state)
        
        assert result == "error"

    def test_route_after_compliance_compliant(self):
        """Test routing after compliant portfolio."""
        from app.agents.graph import PortfolioGraph
        
        state = PortfolioState(
            is_compliant=True,
            iteration_count=1,
        )
        
        mock_services = {
            "data_service": MagicMock(),
            "alpha_service": MagicMock(),
            "risk_service": MagicMock(),
            "optimization_service": MagicMock(),
            "compliance_service": MagicMock(),
            "risk_repository": MagicMock(),
            "constraint_repository": MagicMock(),
            "transaction_cost_repository": MagicMock(),
        }
        
        graph = PortfolioGraph(**mock_services, llm_provider=None)
        result = graph._route_after_compliance(state)
        
        assert result == "compliant"

    def test_route_after_compliance_retry(self):
        """Test routing retries when not compliant."""
        from app.agents.graph import PortfolioGraph
        
        state = PortfolioState(
            is_compliant=False,
            iteration_count=1,
            max_iterations=5,
        )
        
        mock_services = {
            "data_service": MagicMock(),
            "alpha_service": MagicMock(),
            "risk_service": MagicMock(),
            "optimization_service": MagicMock(),
            "compliance_service": MagicMock(),
            "risk_repository": MagicMock(),
            "constraint_repository": MagicMock(),
            "transaction_cost_repository": MagicMock(),
        }
        
        graph = PortfolioGraph(**mock_services, llm_provider=None)
        result = graph._route_after_compliance(state)
        
        assert result == "retry"

    def test_route_after_compliance_max_iterations(self):
        """Test routing stops at max iterations."""
        from app.agents.graph import PortfolioGraph
        
        state = PortfolioState(
            is_compliant=False,
            iteration_count=5,
            max_iterations=5,
        )
        
        mock_services = {
            "data_service": MagicMock(),
            "alpha_service": MagicMock(),
            "risk_service": MagicMock(),
            "optimization_service": MagicMock(),
            "compliance_service": MagicMock(),
            "risk_repository": MagicMock(),
            "constraint_repository": MagicMock(),
            "transaction_cost_repository": MagicMock(),
        }
        
        graph = PortfolioGraph(**mock_services, llm_provider=None)
        result = graph._route_after_compliance(state)
        
        assert result == "max_iterations"
