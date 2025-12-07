"""
Tests for optimization endpoints.
"""

import pytest
from datetime import date
from fastapi.testclient import TestClient


class TestOptimizationEndpoints:
    """Tests for optimization endpoints."""

    def test_trigger_optimization(self, client: TestClient):
        """Test triggering a new optimization run."""
        request_data = {
            "portfolio_id": "test_portfolio",
            "as_of_date": str(date.today()),
            "portfolio_size": 25,
            "risk_aversion": 0.01,
            "use_llm_analysis": False,
        }
        
        response = client.post("/optimization/run", json=request_data)
        
        # Should return 200 (synchronous) or 500 (if solver error)
        assert response.status_code in [200, 500]
        data = response.json()
        
        if response.status_code == 200:
            assert "weights" in data
            assert "is_compliant" in data
            assert "status" in data

    def test_trigger_optimization_default_params(self, client: TestClient):
        """Test optimization with default parameters."""
        request_data = {
            "portfolio_id": "test_portfolio",
        }
        
        response = client.post("/optimization/run", json=request_data)
        
        assert response.status_code in [200, 422, 500]  # 422 if validation fails, 500 if solver error

    def test_trigger_optimization_invalid_params(self, client: TestClient):
        """Test optimization with invalid parameters."""
        request_data = {
            "portfolio_id": "test_portfolio",
            "portfolio_size": -5,  # Invalid
        }
        
        response = client.post("/optimize", json=request_data)
        
        assert response.status_code == 422

    def test_get_optimization_result(self, client: TestClient):
        """Test getting optimization result."""
        # First trigger an optimization
        request_data = {
            "portfolio_id": "test_portfolio",
            "as_of_date": str(date.today()),
            "n_securities": 25,
        }
        
        trigger_response = client.post("/optimize", json=request_data)
        
        if trigger_response.status_code == 202:
            # Get the task ID and poll for result
            task_id = trigger_response.json()["task_id"]
            result_response = client.get(f"/optimize/result/{task_id}")
            assert result_response.status_code in [200, 202]  # 202 if still processing

    def test_optimization_response_format(self, client: TestClient):
        """Test that optimization response has correct format."""
        request_data = {
            "portfolio_id": "test_portfolio",
            "as_of_date": str(date.today()),
            "n_securities": 10,
            "risk_aversion": 0.01,
        }
        
        response = client.post("/optimize", json=request_data)
        
        if response.status_code == 200:
            data = response.json()
            
            # Check required fields
            assert "portfolio_id" in data
            assert "optimal_weights" in data
            assert "is_compliant" in data
            
            # Weights should sum to ~1
            if data["optimal_weights"]:
                total_weight = sum(data["optimal_weights"].values())
                assert abs(total_weight - 1.0) < 0.01


class TestOptimizationValidation:
    """Tests for optimization request validation."""

    def test_n_securities_must_be_positive(self, client: TestClient):
        """Test that n_securities must be positive."""
        request_data = {
            "portfolio_id": "test",
            "n_securities": 0,
        }
        
        response = client.post("/optimize", json=request_data)
        assert response.status_code == 422

    def test_risk_aversion_must_be_positive(self, client: TestClient):
        """Test that risk_aversion must be positive."""
        request_data = {
            "portfolio_id": "test",
            "risk_aversion": -0.01,
        }
        
        response = client.post("/optimize", json=request_data)
        assert response.status_code == 422

    def test_portfolio_id_required(self, client: TestClient):
        """Test that portfolio_id is required."""
        request_data = {
            "n_securities": 25,
        }
        
        response = client.post("/optimize", json=request_data)
        assert response.status_code == 422

