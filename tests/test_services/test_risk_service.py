"""
Tests for risk service.
"""

import pytest
import numpy as np

from app.services.risk_service import RiskService
from app.models.risk import (
    FactorLoading,
    FactorReturns,
    FactorCovariance,
    RiskModel,
)


class TestRiskService:
    """Tests for RiskService."""

    @pytest.fixture
    def sample_loadings(self) -> list[FactorLoading]:
        """Create sample factor loadings."""
        return [
            FactorLoading(
                ticker="AAPL",
                momentum=0.8, value=-0.3, size=-0.5,
                volatility=-0.2, quality=0.6, growth=0.7,
                liquidity=-0.4, dividend_yield=-0.1,
                idiosyncratic_risk=0.15,
            ),
            FactorLoading(
                ticker="MSFT",
                momentum=0.6, value=-0.2, size=-0.6,
                volatility=-0.3, quality=0.7, growth=0.5,
                liquidity=-0.3, dividend_yield=0.1,
                idiosyncratic_risk=0.12,
            ),
            FactorLoading(
                ticker="GOOGL",
                momentum=0.4, value=-0.4, size=-0.7,
                volatility=-0.1, quality=0.5, growth=0.8,
                liquidity=-0.2, dividend_yield=-0.3,
                idiosyncratic_risk=0.18,
            ),
        ]

    @pytest.fixture
    def sample_factor_returns(self) -> FactorReturns:
        """Create sample factor returns."""
        return FactorReturns(
            momentum=0.05,
            value=0.02,
            size=-0.01,
            volatility=-0.03,
            quality=0.04,
            growth=0.06,
            liquidity=0.01,
            dividend_yield=0.02,
        )

    @pytest.fixture
    def sample_covariance(self) -> FactorCovariance:
        """Create sample factor covariance matrix."""
        # Simple diagonal covariance for testing
        matrix = np.diag([0.04, 0.03, 0.02, 0.05, 0.03, 0.04, 0.02, 0.02])
        return FactorCovariance(
            matrix=matrix,
            factor_names=[
                "momentum", "value", "size", "volatility",
                "quality", "growth", "liquidity", "dividend_yield"
            ],
        )

    @pytest.fixture
    def sample_risk_model(
        self,
        sample_loadings: list[FactorLoading],
        sample_factor_returns: FactorReturns,
        sample_covariance: FactorCovariance,
    ) -> RiskModel:
        """Create sample risk model."""
        return RiskModel(
            factor_loadings=sample_loadings,
            factor_returns=sample_factor_returns,
            factor_covariance=sample_covariance,
        )

    def test_calculate_portfolio_variance(
        self,
        risk_service: RiskService,
        sample_risk_model: RiskModel,
    ):
        """Test calculating portfolio variance."""
        weights = {"AAPL": 0.4, "MSFT": 0.35, "GOOGL": 0.25}
        
        variance = risk_service.calculate_portfolio_variance(
            weights=weights,
            risk_model=sample_risk_model,
        )
        
        assert variance > 0
        assert isinstance(variance, float)

    def test_calculate_portfolio_volatility(
        self,
        risk_service: RiskService,
        sample_risk_model: RiskModel,
    ):
        """Test calculating portfolio volatility."""
        weights = {"AAPL": 0.4, "MSFT": 0.35, "GOOGL": 0.25}
        
        volatility = risk_service.calculate_portfolio_volatility(
            weights=weights,
            risk_model=sample_risk_model,
        )
        
        assert volatility > 0
        # Volatility is sqrt of variance
        variance = risk_service.calculate_portfolio_variance(weights, sample_risk_model)
        assert abs(volatility - np.sqrt(variance)) < 0.0001

    def test_calculate_factor_exposures(
        self,
        risk_service: RiskService,
        sample_risk_model: RiskModel,
    ):
        """Test calculating portfolio factor exposures."""
        weights = {"AAPL": 0.4, "MSFT": 0.35, "GOOGL": 0.25}
        
        exposures = risk_service.calculate_factor_exposures(
            weights=weights,
            risk_model=sample_risk_model,
        )
        
        assert len(exposures) == 8  # 8 factors
        assert "momentum" in exposures
        # Weighted average of loadings
        expected_momentum = 0.4 * 0.8 + 0.35 * 0.6 + 0.25 * 0.4
        assert abs(exposures["momentum"] - expected_momentum) < 0.001

    def test_calculate_systematic_risk(
        self,
        risk_service: RiskService,
        sample_risk_model: RiskModel,
    ):
        """Test calculating systematic (factor) risk."""
        weights = {"AAPL": 0.4, "MSFT": 0.35, "GOOGL": 0.25}
        
        systematic = risk_service.calculate_systematic_risk(
            weights=weights,
            risk_model=sample_risk_model,
        )
        
        assert systematic > 0

    def test_calculate_idiosyncratic_risk(
        self,
        risk_service: RiskService,
        sample_risk_model: RiskModel,
    ):
        """Test calculating idiosyncratic risk."""
        weights = {"AAPL": 0.4, "MSFT": 0.35, "GOOGL": 0.25}
        
        idio = risk_service.calculate_idiosyncratic_risk(
            weights=weights,
            risk_model=sample_risk_model,
        )
        
        assert idio > 0
        # Idiosyncratic risk should be less than total risk for diversified portfolio
        total = risk_service.calculate_portfolio_volatility(weights, sample_risk_model)
        assert idio < total

    def test_get_covariance_matrix(
        self,
        risk_service: RiskService,
        sample_risk_model: RiskModel,
    ):
        """Test building full covariance matrix."""
        tickers = ["AAPL", "MSFT", "GOOGL"]
        
        cov = risk_service.get_covariance_matrix(
            tickers=tickers,
            risk_model=sample_risk_model,
        )
        
        assert cov.shape == (3, 3)
        # Should be symmetric
        assert np.allclose(cov, cov.T)
        # Diagonal should be positive
        assert all(cov[i, i] > 0 for i in range(3))

    def test_decompose_risk(
        self,
        risk_service: RiskService,
        sample_risk_model: RiskModel,
    ):
        """Test risk decomposition."""
        weights = {"AAPL": 0.4, "MSFT": 0.35, "GOOGL": 0.25}
        
        decomposition = risk_service.decompose_risk(
            weights=weights,
            risk_model=sample_risk_model,
        )
        
        assert "total_risk" in decomposition
        assert "systematic_risk" in decomposition
        assert "idiosyncratic_risk" in decomposition
        assert "factor_contributions" in decomposition
        
        # Total risk should equal sqrt(systematic^2 + idiosyncratic^2)
        total = decomposition["total_risk"]
        systematic = decomposition["systematic_risk"]
        idio = decomposition["idiosyncratic_risk"]
        assert abs(total - np.sqrt(systematic**2 + idio**2)) < 0.001

