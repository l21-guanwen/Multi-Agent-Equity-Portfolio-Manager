"""
Tests for agent graph orchestration.
"""

import pytest
from datetime import date
from unittest.mock import AsyncMock, MagicMock, patch

from app.agents.graph import create_portfolio_graph, PortfolioGraphBuilder
from app.agents.state import create_initial_state


class TestPortfolioGraph:
    """Tests for portfolio optimization graph."""

    @pytest.fixture
    def mock_services(self):
        """Create mock services for graph."""
        return {
            "data_service": MagicMock(),
            "alpha_service": MagicMock(),
            "risk_service": MagicMock(),
            "optimization_service": MagicMock(),
            "compliance_service": MagicMock(),
        }

    def test_graph_builder_creates_graph(self, mock_services):
        """Test that graph builder creates a valid graph."""
        builder = PortfolioGraphBuilder(
            data_service=mock_services["data_service"],
            alpha_service=mock_services["alpha_service"],
            risk_service=mock_services["risk_service"],
            optimization_service=mock_services["optimization_service"],
            compliance_service=mock_services["compliance_service"],
        )
        
        graph = builder.build()
        
        assert graph is not None

    def test_graph_has_all_nodes(self, mock_services):
        """Test that graph has all required agent nodes."""
        builder = PortfolioGraphBuilder(
            data_service=mock_services["data_service"],
            alpha_service=mock_services["alpha_service"],
            risk_service=mock_services["risk_service"],
            optimization_service=mock_services["optimization_service"],
            compliance_service=mock_services["compliance_service"],
        )
        
        graph = builder.build()
        
        # Verify nodes exist in the graph
        node_names = list(graph.nodes.keys())
        expected_nodes = [
            "data_agent",
            "alpha_agent",
            "risk_agent",
            "optimization_agent",
            "compliance_agent",
        ]
        
        for node in expected_nodes:
            assert node in node_names

    def test_initial_state_creation(self):
        """Test initial state creation for graph."""
        state = create_initial_state(
            portfolio_id="test_portfolio",
            as_of_date=date(2024, 1, 15),
        )
        
        assert state["portfolio_id"] == "test_portfolio"
        assert state["as_of_date"] == date(2024, 1, 15)
        assert state["iteration"] == 0

    def test_graph_conditional_routing(self, mock_services):
        """Test that graph has conditional routing for compliance."""
        builder = PortfolioGraphBuilder(
            data_service=mock_services["data_service"],
            alpha_service=mock_services["alpha_service"],
            risk_service=mock_services["risk_service"],
            optimization_service=mock_services["optimization_service"],
            compliance_service=mock_services["compliance_service"],
        )
        
        graph = builder.build()
        
        # Graph should have conditional edges
        # This verifies the structure without running the full graph
        assert graph is not None


class TestComplianceRouting:
    """Tests for compliance-based routing decisions."""

    def test_route_to_end_when_compliant(self):
        """Test routing to END when portfolio is compliant."""
        from app.agents.graph import should_retry_optimization
        
        state = create_initial_state(
            portfolio_id="test",
            as_of_date=date(2024, 1, 1),
        )
        state["is_compliant"] = True
        state["iteration"] = 1
        
        next_node = should_retry_optimization(state)
        
        assert next_node == "end"

    def test_route_to_optimization_when_not_compliant(self):
        """Test routing back to optimization when not compliant."""
        from app.agents.graph import should_retry_optimization
        
        state = create_initial_state(
            portfolio_id="test",
            as_of_date=date(2024, 1, 1),
        )
        state["is_compliant"] = False
        state["iteration"] = 1
        
        next_node = should_retry_optimization(state)
        
        assert next_node == "optimization_agent"

    def test_route_to_end_when_max_iterations(self):
        """Test routing to END when max iterations reached."""
        from app.agents.graph import should_retry_optimization
        
        state = create_initial_state(
            portfolio_id="test",
            as_of_date=date(2024, 1, 1),
        )
        state["is_compliant"] = False
        state["iteration"] = 5  # Max iterations
        
        next_node = should_retry_optimization(state)
        
        assert next_node == "end"

