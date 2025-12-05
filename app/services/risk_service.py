"""
Risk service implementation.

Handles risk analysis, factor exposure calculations, and portfolio risk metrics.
"""

from typing import Optional

import numpy as np

from app.models.risk import RiskModel
from app.services.interfaces.risk_service import (
    FactorExposure,
    IRiskService,
    PortfolioRiskMetrics,
)


class RiskService(IRiskService):
    """
    Service for risk model operations.
    
    Handles factor exposure calculations, portfolio risk decomposition,
    and risk-related analytics for portfolio construction.
    """

    async def calculate_portfolio_risk(
        self,
        weights: dict[str, float],
        risk_model: RiskModel,
    ) -> PortfolioRiskMetrics:
        """Calculate comprehensive risk metrics for a portfolio."""
        if not weights or not risk_model.factor_loadings:
            return PortfolioRiskMetrics(
                total_risk_pct=0.0,
                systematic_risk_pct=0.0,
                specific_risk_pct=0.0,
                tracking_error_pct=None,
                beta=0.0,
                factor_risk_contributions={},
            )
        
        tickers = list(weights.keys())
        weight_array = np.array([weights[t] for t in tickers])
        
        # Get factor loadings matrix (N x K)
        loadings_matrix = risk_model.get_loadings_matrix(tickers)
        
        # Get factor covariance (K x K)
        if risk_model.factor_covariance:
            factor_cov = risk_model.factor_covariance.get_matrix_array()
        else:
            factor_cov = np.eye(8) * 100  # Default identity scaled
        
        # Get specific risk
        specific_risks = np.array([
            risk_model.get_loadings(t).specific_risk_pct if risk_model.get_loadings(t) else 20.0
            for t in tickers
        ])
        
        # Calculate portfolio factor exposure (K x 1)
        portfolio_loadings = loadings_matrix.T @ weight_array
        
        # Systematic variance: B'FB where B = portfolio loadings, F = factor cov
        systematic_variance = portfolio_loadings @ factor_cov @ portfolio_loadings
        
        # Specific variance: sum of (w_i^2 * sigma_i^2)
        specific_variance = np.sum((weight_array ** 2) * (specific_risks ** 2))
        
        # Total variance
        total_variance = systematic_variance + specific_variance
        
        # Convert to percentage (volatility)
        total_risk = np.sqrt(total_variance)
        systematic_risk = np.sqrt(systematic_variance)
        specific_risk = np.sqrt(specific_variance)
        
        # Calculate beta (market factor exposure)
        beta = portfolio_loadings[0] if len(portfolio_loadings) > 0 else 1.0
        
        # Calculate factor risk contributions
        factor_contributions = await self.decompose_risk(weights, risk_model)
        
        return PortfolioRiskMetrics(
            total_risk_pct=float(total_risk),
            systematic_risk_pct=float(systematic_risk),
            specific_risk_pct=float(specific_risk),
            tracking_error_pct=None,
            beta=float(beta),
            factor_risk_contributions=factor_contributions,
        )

    async def calculate_factor_exposure(
        self,
        weights: dict[str, float],
        risk_model: RiskModel,
    ) -> FactorExposure:
        """Calculate portfolio-level factor exposures."""
        if not weights or not risk_model.factor_loadings:
            return FactorExposure(
                market=0.0, size=0.0, value=0.0, momentum=0.0,
                quality=0.0, volatility=0.0, growth=0.0, dividend_yield=0.0,
            )
        
        tickers = list(weights.keys())
        weight_array = np.array([weights[t] for t in tickers])
        
        # Get factor loadings matrix
        loadings_matrix = risk_model.get_loadings_matrix(tickers)
        
        # Calculate weighted average exposures
        portfolio_loadings = loadings_matrix.T @ weight_array
        
        return FactorExposure(
            market=float(portfolio_loadings[0]),
            size=float(portfolio_loadings[1]),
            value=float(portfolio_loadings[2]),
            momentum=float(portfolio_loadings[3]),
            quality=float(portfolio_loadings[4]),
            volatility=float(portfolio_loadings[5]),
            growth=float(portfolio_loadings[6]),
            dividend_yield=float(portfolio_loadings[7]),
        )

    async def calculate_tracking_error(
        self,
        portfolio_weights: dict[str, float],
        benchmark_weights: dict[str, float],
        risk_model: RiskModel,
    ) -> float:
        """Calculate tracking error vs benchmark."""
        # Calculate active weights
        all_tickers = set(portfolio_weights.keys()) | set(benchmark_weights.keys())
        active_weights = {
            t: portfolio_weights.get(t, 0.0) - benchmark_weights.get(t, 0.0)
            for t in all_tickers
        }
        
        # Use active weights to calculate risk
        risk_metrics = await self.calculate_portfolio_risk(active_weights, risk_model)
        
        return risk_metrics.total_risk_pct

    async def calculate_covariance_matrix(
        self,
        tickers: list[str],
        risk_model: RiskModel,
    ) -> np.ndarray:
        """
        Calculate security-level covariance matrix.
        
        Uses factor model: Cov = B * F * B' + D
        """
        n = len(tickers)
        
        # Get factor loadings (N x K)
        loadings_matrix = risk_model.get_loadings_matrix(tickers)
        
        # Get factor covariance (K x K)
        if risk_model.factor_covariance:
            factor_cov = risk_model.factor_covariance.get_matrix_array()
        else:
            factor_cov = np.eye(8) * 100
        
        # Get specific risks (diagonal)
        specific_risks = np.array([
            risk_model.get_loadings(t).specific_risk_pct if risk_model.get_loadings(t) else 20.0
            for t in tickers
        ])
        
        # Calculate: Cov = B * F * B' + D
        systematic_cov = loadings_matrix @ factor_cov @ loadings_matrix.T
        specific_cov = np.diag(specific_risks ** 2)
        
        return systematic_cov + specific_cov

    async def decompose_risk(
        self,
        weights: dict[str, float],
        risk_model: RiskModel,
    ) -> dict[str, float]:
        """Decompose portfolio risk by factor."""
        if not weights or not risk_model.factor_loadings:
            return {}
        
        tickers = list(weights.keys())
        weight_array = np.array([weights[t] for t in tickers])
        
        # Get factor loadings matrix
        loadings_matrix = risk_model.get_loadings_matrix(tickers)
        
        # Get factor covariance
        if risk_model.factor_covariance:
            factor_cov = risk_model.factor_covariance.get_matrix_array()
        else:
            factor_cov = np.eye(8) * 100
        
        # Portfolio factor exposures
        portfolio_loadings = loadings_matrix.T @ weight_array
        
        # Factor variances (diagonal of factor cov)
        factor_variances = np.diag(factor_cov)
        
        # Risk contribution from each factor
        factor_names = [
            "Market", "Size", "Value", "Momentum",
            "Quality", "Volatility", "Growth", "Dividend_Yield"
        ]
        
        contributions = {}
        for i, name in enumerate(factor_names):
            # Marginal contribution: exposure^2 * factor_variance
            contribution = (portfolio_loadings[i] ** 2) * factor_variances[i]
            contributions[name] = float(np.sqrt(contribution))
        
        return contributions

    async def get_risk_contributors(
        self,
        weights: dict[str, float],
        risk_model: RiskModel,
        top_n: int = 10,
    ) -> list[tuple[str, float]]:
        """Get top risk-contributing securities."""
        if not weights:
            return []
        
        tickers = list(weights.keys())
        
        # Calculate covariance matrix
        cov_matrix = await self.calculate_covariance_matrix(tickers, risk_model)
        weight_array = np.array([weights[t] for t in tickers])
        
        # Portfolio variance
        portfolio_variance = weight_array @ cov_matrix @ weight_array
        
        # Marginal contribution to risk for each security
        marginal_contrib = cov_matrix @ weight_array
        
        # Component risk contribution: w_i * marginal_i / sqrt(portfolio_variance)
        if portfolio_variance > 0:
            component_risk = (weight_array * marginal_contrib) / np.sqrt(portfolio_variance)
        else:
            component_risk = np.zeros(len(tickers))
        
        # Create list of (ticker, contribution)
        contributions = list(zip(tickers, component_risk.tolist()))
        
        # Sort by absolute contribution descending
        contributions.sort(key=lambda x: abs(x[1]), reverse=True)
        
        return contributions[:top_n]

    async def calculate_marginal_risk(
        self,
        ticker: str,
        current_weights: dict[str, float],
        risk_model: RiskModel,
    ) -> float:
        """Calculate marginal risk contribution of adding a security."""
        if ticker not in current_weights:
            # Add with small weight
            test_weights = current_weights.copy()
            test_weights[ticker] = 0.01
        else:
            test_weights = current_weights
        
        tickers = list(test_weights.keys())
        ticker_idx = tickers.index(ticker)
        
        # Calculate covariance matrix
        cov_matrix = await self.calculate_covariance_matrix(tickers, risk_model)
        weight_array = np.array([test_weights[t] for t in tickers])
        
        # Marginal risk = (Cov @ w)[i] / sqrt(w' @ Cov @ w)
        cov_w = cov_matrix @ weight_array
        portfolio_vol = np.sqrt(weight_array @ cov_w)
        
        if portfolio_vol > 0:
            marginal = cov_w[ticker_idx] / portfolio_vol
        else:
            marginal = 0.0
        
        return float(marginal)

