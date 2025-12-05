"""
Tests for risk repository.
"""

import pytest
import numpy as np

from app.repositories.csv.csv_risk_repository import CSVRiskRepository


class TestCSVRiskRepository:
    """Tests for CSVRiskRepository."""

    @pytest.mark.asyncio
    async def test_get_factor_loadings(
        self,
        risk_repository: CSVRiskRepository,
    ):
        """Test getting factor loadings."""
        loadings = await risk_repository.get_factor_loadings()
        
        assert len(loadings) > 0
        assert len(loadings) == 500

    @pytest.mark.asyncio
    async def test_get_factor_loading_by_ticker(
        self,
        risk_repository: CSVRiskRepository,
    ):
        """Test getting factor loading by ticker."""
        loading = await risk_repository.get_factor_loading("AAPL")
        
        assert loading is not None
        assert loading.ticker == "AAPL"
        # Should have loadings for all factors
        assert hasattr(loading, "momentum")
        assert hasattr(loading, "value")
        assert hasattr(loading, "size")
        assert hasattr(loading, "volatility")
        assert hasattr(loading, "quality")
        assert hasattr(loading, "growth")
        assert hasattr(loading, "liquidity")
        assert hasattr(loading, "dividend_yield")
        assert loading.idiosyncratic_risk > 0

    @pytest.mark.asyncio
    async def test_get_factor_returns(
        self,
        risk_repository: CSVRiskRepository,
    ):
        """Test getting factor returns."""
        returns = await risk_repository.get_factor_returns()
        
        assert len(returns) == 8  # 8 factors

    @pytest.mark.asyncio
    async def test_get_factor_covariance(
        self,
        risk_repository: CSVRiskRepository,
    ):
        """Test getting factor covariance matrix."""
        cov = await risk_repository.get_factor_covariance()
        
        assert cov is not None
        assert cov.matrix.shape == (8, 8)
        assert len(cov.factor_names) == 8
        # Matrix should be symmetric
        assert np.allclose(cov.matrix, cov.matrix.T)
        # Diagonal should be positive (variances)
        assert all(cov.matrix[i, i] > 0 for i in range(8))

    @pytest.mark.asyncio
    async def test_get_full_risk_model(
        self,
        risk_repository: CSVRiskRepository,
    ):
        """Test getting full risk model."""
        model = await risk_repository.get_full_risk_model()
        
        assert model is not None
        assert len(model.factor_loadings) == 500
        assert len(model.factor_returns) == 8
        assert model.factor_covariance.matrix.shape == (8, 8)

    @pytest.mark.asyncio
    async def test_get_loading_matrix(
        self,
        risk_repository: CSVRiskRepository,
    ):
        """Test getting loading matrix as numpy array."""
        matrix, tickers = await risk_repository.get_loading_matrix()
        
        assert matrix.shape == (500, 8)
        assert len(tickers) == 500
        assert "AAPL" in tickers

    @pytest.mark.asyncio
    async def test_get_idiosyncratic_risk_dict(
        self,
        risk_repository: CSVRiskRepository,
    ):
        """Test getting idiosyncratic risk dictionary."""
        risks = await risk_repository.get_idiosyncratic_risk_dict()
        
        assert len(risks) == 500
        for ticker, risk in risks.items():
            assert risk > 0

