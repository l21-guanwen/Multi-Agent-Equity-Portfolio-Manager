"""
Tests for portfolio endpoints.
"""

import pytest
from fastapi.testclient import TestClient


class TestPortfolioEndpoints:
    """Tests for portfolio endpoints."""

    def test_get_benchmark(self, client: TestClient):
        """Test getting benchmark data."""
        response = client.get("/portfolio/benchmark")
        
        assert response.status_code == 200
        data = response.json()
        assert "benchmark_id" in data
        assert data["benchmark_id"] == "SPX"
        assert "security_count" in data

    def test_get_benchmark_constituents(self, client: TestClient):
        """Test getting benchmark constituents."""
        response = client.get("/portfolio/benchmark/constituents")
        
        assert response.status_code == 200
        data = response.json()
        assert "constituents" in data
        assert len(data["constituents"]) == 500

    def test_get_benchmark_constituent_by_ticker(self, client: TestClient):
        """Test getting a specific benchmark constituent."""
        response = client.get("/portfolio/benchmark/constituents/AAPL")
        
        assert response.status_code == 200
        data = response.json()
        assert data["ticker"] == "AAPL"
        assert "benchmark_weight_pct" in data
        assert "gics_sector" in data

    def test_get_benchmark_constituent_not_found(self, client: TestClient):
        """Test getting a non-existent constituent."""
        response = client.get("/portfolio/benchmark/constituents/INVALID")
        
        assert response.status_code == 404

    def test_get_sector_weights(self, client: TestClient):
        """Test getting sector weights."""
        response = client.get("/portfolio/benchmark/sectors")
        
        assert response.status_code == 200
        data = response.json()
        assert "sectors" in data
        assert len(data["sectors"]) == 11

    def test_get_universe(self, client: TestClient):
        """Test getting investment universe."""
        response = client.get("/portfolio/universe")
        
        assert response.status_code == 200
        data = response.json()
        assert "securities" in data
        assert len(data["securities"]) == 500

    def test_get_alpha_scores(self, client: TestClient):
        """Test getting alpha scores."""
        response = client.get("/portfolio/alpha")
        
        assert response.status_code == 200
        data = response.json()
        assert "scores" in data
        assert len(data["scores"]) == 500

    def test_get_top_alpha_scores(self, client: TestClient):
        """Test getting top alpha scores."""
        response = client.get("/portfolio/alpha/top?n=25")
        
        assert response.status_code == 200
        data = response.json()
        assert "scores" in data
        assert len(data["scores"]) == 25
        
        # Verify sorted by alpha descending
        scores = [s["alpha_score"] for s in data["scores"]]
        assert scores == sorted(scores, reverse=True)

    def test_get_constraints(self, client: TestClient):
        """Test getting portfolio constraints."""
        response = client.get("/portfolio/constraints")
        
        assert response.status_code == 200
        data = response.json()
        assert "constraints" in data
        assert len(data["constraints"]) > 0

