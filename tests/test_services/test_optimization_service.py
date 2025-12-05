"""
Tests for optimization service.
"""

import pytest
import numpy as np

from app.services.optimization_service import OptimizationService
from app.solvers.cvxpy_solver import CvxpySolver
from app.services.risk_service import RiskService
from app.models.alpha import AlphaScore, AlphaModel
from app.models.risk import (
    FactorLoading,
    FactorReturns,
    FactorCovariance,
    RiskModel,
)
from app.models.constraint import Constraint


class TestOptimizationService:
    """Tests for OptimizationService."""

    @pytest.fixture
    def tickers(self) -> list[str]:
        """List of test tickers."""
        return ["AAPL", "MSFT", "GOOGL", "AMZN", "META"]

    @pytest.fixture
    def sample_alpha_model(self, tickers: list[str]) -> AlphaModel:
        """Create sample alpha model."""
        scores = [
            AlphaScore(ticker="AAPL", alpha_score=0.92, alpha_quintile=1),
            AlphaScore(ticker="MSFT", alpha_score=0.88, alpha_quintile=1),
            AlphaScore(ticker="GOOGL", alpha_score=0.75, alpha_quintile=2),
            AlphaScore(ticker="AMZN", alpha_score=0.70, alpha_quintile=2),
            AlphaScore(ticker="META", alpha_score=0.65, alpha_quintile=2),
        ]
        return AlphaModel(scores=scores)

    @pytest.fixture
    def sample_risk_model(self, tickers: list[str]) -> RiskModel:
        """Create sample risk model."""
        loadings = []
        for i, ticker in enumerate(tickers):
            loadings.append(FactorLoading(
                ticker=ticker,
                momentum=0.5 + i * 0.1,
                value=-0.2 - i * 0.05,
                size=-0.3,
                volatility=-0.1,
                quality=0.4,
                growth=0.5,
                liquidity=-0.2,
                dividend_yield=0.1,
                idiosyncratic_risk=0.12 + i * 0.01,
            ))
        
        factor_returns = FactorReturns(
            momentum=0.05, value=0.02, size=-0.01, volatility=-0.02,
            quality=0.03, growth=0.04, liquidity=0.01, dividend_yield=0.02,
        )
        
        # Simple diagonal covariance
        matrix = np.diag([0.04, 0.03, 0.02, 0.05, 0.03, 0.04, 0.02, 0.02])
        factor_cov = FactorCovariance(
            matrix=matrix,
            factor_names=[
                "momentum", "value", "size", "volatility",
                "quality", "growth", "liquidity", "dividend_yield"
            ],
        )
        
        return RiskModel(
            factor_loadings=loadings,
            factor_returns=factor_returns,
            factor_covariance=factor_cov,
        )

    @pytest.fixture
    def sample_constraints(self, tickers: list[str]) -> list[Constraint]:
        """Create sample constraints."""
        constraints = []
        for ticker in tickers:
            constraints.append(Constraint(
                constraint_id=f"single_stock_{ticker}",
                constraint_type="single_stock_active",
                target=ticker,
                min_value=-1.0,
                max_value=1.0,
                is_enabled=True,
            ))
        return constraints

    @pytest.fixture
    def sample_benchmark_weights(self, tickers: list[str]) -> dict[str, float]:
        """Sample benchmark weights."""
        return {
            "AAPL": 6.70,
            "MSFT": 5.77,
            "GOOGL": 3.21,
            "AMZN": 3.96,
            "META": 2.50,
        }

    def test_prepare_optimization_inputs(
        self,
        optimization_service: OptimizationService,
        tickers: list[str],
        sample_alpha_model: AlphaModel,
        sample_risk_model: RiskModel,
    ):
        """Test preparing optimization inputs."""
        inputs = optimization_service.prepare_inputs(
            tickers=tickers,
            alpha_model=sample_alpha_model,
            risk_model=sample_risk_model,
        )
        
        assert "alpha_vector" in inputs
        assert "covariance_matrix" in inputs
        assert len(inputs["alpha_vector"]) == 5
        assert inputs["covariance_matrix"].shape == (5, 5)

    def test_run_optimization(
        self,
        optimization_service: OptimizationService,
        tickers: list[str],
        sample_alpha_model: AlphaModel,
        sample_risk_model: RiskModel,
    ):
        """Test running optimization."""
        result = optimization_service.optimize(
            tickers=tickers,
            alpha_model=sample_alpha_model,
            risk_model=sample_risk_model,
            risk_aversion=0.01,
        )
        
        assert result is not None
        assert result.status == "optimal"
        assert len(result.weights) == 5
        
        # Weights should be between 0 and 1 for long-only
        for weight in result.weights.values():
            assert 0 <= weight <= 1
        
        # Weights should sum to 1 (fully invested)
        assert abs(sum(result.weights.values()) - 1.0) < 0.01

    def test_optimize_with_constraints(
        self,
        optimization_service: OptimizationService,
        tickers: list[str],
        sample_alpha_model: AlphaModel,
        sample_risk_model: RiskModel,
        sample_constraints: list[Constraint],
        sample_benchmark_weights: dict[str, float],
    ):
        """Test optimization with constraints."""
        result = optimization_service.optimize_with_constraints(
            tickers=tickers,
            alpha_model=sample_alpha_model,
            risk_model=sample_risk_model,
            constraints=sample_constraints,
            benchmark_weights=sample_benchmark_weights,
            risk_aversion=0.01,
        )
        
        assert result is not None
        # Check that active weights are within bounds
        for ticker, weight in result.weights.items():
            portfolio_pct = weight * 100
            benchmark_pct = sample_benchmark_weights.get(ticker, 0)
            active = portfolio_pct - benchmark_pct
            # Active weight should be within ±1% (approximately)
            # Some slack allowed due to solver tolerance
            assert -1.5 <= active <= 1.5

    def test_calculate_expected_return(
        self,
        optimization_service: OptimizationService,
        tickers: list[str],
        sample_alpha_model: AlphaModel,
    ):
        """Test calculating expected portfolio return."""
        weights = {
            "AAPL": 0.25,
            "MSFT": 0.25,
            "GOOGL": 0.20,
            "AMZN": 0.15,
            "META": 0.15,
        }
        
        expected_return = optimization_service.calculate_expected_return(
            weights=weights,
            alpha_model=sample_alpha_model,
        )
        
        # Weighted average of alpha scores
        expected = (0.25 * 0.92 + 0.25 * 0.88 + 0.20 * 0.75 + 
                   0.15 * 0.70 + 0.15 * 0.65)
        assert abs(expected_return - expected) < 0.001

    def test_calculate_tracking_error(
        self,
        optimization_service: OptimizationService,
        sample_risk_model: RiskModel,
        sample_benchmark_weights: dict[str, float],
    ):
        """Test calculating tracking error vs benchmark."""
        portfolio_weights = {
            "AAPL": 0.30,
            "MSFT": 0.25,
            "GOOGL": 0.20,
            "AMZN": 0.15,
            "META": 0.10,
        }
        
        # Convert benchmark to decimals
        benchmark_decimal = {
            k: v / 100 for k, v in sample_benchmark_weights.items()
        }
        
        tracking_error = optimization_service.calculate_tracking_error(
            portfolio_weights=portfolio_weights,
            benchmark_weights=benchmark_decimal,
            risk_model=sample_risk_model,
        )
        
        assert tracking_error > 0

    def test_validate_optimization_result(
        self,
        optimization_service: OptimizationService,
    ):
        """Test validating optimization result."""
        # Valid result
        valid_weights = {
            "AAPL": 0.25,
            "MSFT": 0.25,
            "GOOGL": 0.25,
            "AMZN": 0.25,
        }
        
        is_valid, errors = optimization_service.validate_result(valid_weights)
        assert is_valid is True
        assert len(errors) == 0
        
        # Invalid result - doesn't sum to 1
        invalid_weights = {
            "AAPL": 0.50,
            "MSFT": 0.30,
        }
        
        is_valid, errors = optimization_service.validate_result(invalid_weights)
        assert is_valid is False
        assert len(errors) > 0

