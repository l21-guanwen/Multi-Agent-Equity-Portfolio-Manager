"""
Tests for agent state.
"""

import pytest
from datetime import date

from app.agents.state import PortfolioState, create_initial_state


class TestPortfolioState:
    """Tests for PortfolioState."""

    def test_create_initial_state(self):
        """Test creating initial state."""
        state = create_initial_state(
            portfolio_id="test_portfolio",
            as_of_date=date(2024, 1, 15),
        )
        
        assert state["portfolio_id"] == "test_portfolio"
        assert state["as_of_date"] == date(2024, 1, 15)
        assert state["current_step"] == "initialized"
        assert state["iteration"] == 0
        assert state["is_compliant"] is False
        assert state["errors"] == []

    def test_state_has_required_keys(self):
        """Test that state has all required keys."""
        state = create_initial_state(
            portfolio_id="test",
            as_of_date=date(2024, 1, 1),
        )
        
        required_keys = [
            "portfolio_id",
            "as_of_date",
            "current_step",
            "iteration",
            "benchmark",
            "universe",
            "alpha_model",
            "risk_model",
            "constraints",
            "transaction_costs",
            "selected_securities",
            "optimal_weights",
            "compliance_result",
            "is_compliant",
            "final_portfolio",
            "errors",
            "llm_insights",
        ]
        
        for key in required_keys:
            assert key in state

    def test_state_default_values(self):
        """Test default values in initial state."""
        state = create_initial_state(
            portfolio_id="test",
            as_of_date=date(2024, 1, 1),
        )
        
        assert state["benchmark"] is None
        assert state["universe"] is None
        assert state["alpha_model"] is None
        assert state["risk_model"] is None
        assert state["constraints"] is None
        assert state["transaction_costs"] is None
        assert state["selected_securities"] is None
        assert state["optimal_weights"] is None
        assert state["compliance_result"] is None
        assert state["final_portfolio"] is None
        assert state["llm_insights"] == {}

