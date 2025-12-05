"""
Tests for health endpoints.
"""

import pytest
from fastapi.testclient import TestClient


class TestHealthEndpoints:
    """Tests for health check endpoints."""

    def test_health_check(self, client: TestClient):
        """Test basic health check endpoint."""
        response = client.get("/health")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"

    def test_health_check_returns_app_info(self, client: TestClient):
        """Test that health check returns app information."""
        response = client.get("/health")
        
        assert response.status_code == 200
        data = response.json()
        assert "app_name" in data
        assert "version" in data

    def test_readiness_check(self, client: TestClient):
        """Test readiness check endpoint."""
        response = client.get("/health/ready")
        
        assert response.status_code == 200
        data = response.json()
        assert "ready" in data

    def test_liveness_check(self, client: TestClient):
        """Test liveness check endpoint."""
        response = client.get("/health/live")
        
        assert response.status_code == 200
        data = response.json()
        assert data["alive"] is True

