"""
Risk Agent for the multi-agent portfolio management system.

Responsible for risk analysis and factor exposure calculation.
"""

from typing import Any, Optional

from app.agents.prompts import RISK_AGENT_SYSTEM_PROMPT
from app.agents.state import PortfolioState
from app.llm.interfaces.llm_provider import ILLMProvider
from app.models.risk import RiskModel
from app.repositories.interfaces.risk_repository import IRiskRepository
from app.services.interfaces.risk_service import IRiskService


class RiskAgent:
    """
    Risk Agent responsible for risk analysis.
    
    This agent:
    1. Calculates factor exposures for selected securities
    2. Computes portfolio-level risk metrics
    3. Uses LLM to interpret risk characteristics
    """

    def __init__(
        self,
        risk_service: IRiskService,
        risk_repository: IRiskRepository,
        llm_provider: Optional[ILLMProvider] = None,
    ):
        """
        Initialize the Risk Agent.
        
        Args:
            risk_service: Service for risk calculations
            risk_repository: Repository for risk model data
            llm_provider: Optional LLM provider for analysis
        """
        self._risk_service = risk_service
        self._risk_repo = risk_repository
        self._llm = llm_provider

    async def __call__(self, state: PortfolioState) -> dict[str, Any]:
        """
        Execute the risk agent.
        
        Args:
            state: Current portfolio state
            
        Returns:
            Updated state fields
        """
        execution_log = [f"[RiskAgent] Starting risk analysis..."]
        
        try:
            # Check prerequisites
            if not state.selected_tickers:
                execution_log.append("[RiskAgent] Skipped - no securities selected")
                return {
                    "execution_log": execution_log,
                    "current_agent": "risk_agent",
                }
            
            # Load risk model
            risk_model = await self._risk_repo.get_risk_model()
            
            if not risk_model:
                execution_log.append("[RiskAgent] WARNING: No risk model available")
                return {
                    "execution_log": execution_log,
                    "current_agent": "risk_agent",
                }
            
            # Calculate equal-weight exposures for selected securities
            # (Will be refined after optimization)
            n = len(state.selected_tickers)
            equal_weights = {t: 1.0 / n for t in state.selected_tickers}
            
            # Calculate factor exposures
            factor_exposure = await self._risk_service.calculate_factor_exposure(
                equal_weights, risk_model
            )
            execution_log.append(f"[RiskAgent] Factor exposures calculated")
            
            # Calculate portfolio risk
            risk_metrics = await self._risk_service.calculate_portfolio_risk(
                equal_weights, risk_model
            )
            execution_log.append(
                f"[RiskAgent] Portfolio risk: {risk_metrics.total_risk_pct:.2f}%"
            )
            
            # Generate LLM analysis
            if self._llm:
                llm_analysis = await self._generate_analysis(
                    factor_exposure, risk_metrics, state
                )
                execution_log.append(f"[RiskAgent] LLM analysis generated")
            else:
                llm_analysis = self._generate_basic_analysis(
                    factor_exposure, risk_metrics, state
                )
            
            execution_log.append(f"[RiskAgent] Completed")
            
            return {
                "factor_exposures": factor_exposure.to_dict(),
                "portfolio_risk_pct": risk_metrics.total_risk_pct,
                "risk_analysis": llm_analysis,
                "execution_log": execution_log,
                "current_agent": "risk_agent",
            }
            
        except Exception as e:
            execution_log.append(f"[RiskAgent] ERROR: {str(e)}")
            return {
                "error_message": f"Risk analysis failed: {str(e)}",
                "execution_log": execution_log,
                "current_agent": "risk_agent",
            }

    async def _generate_analysis(
        self,
        factor_exposure: Any,
        risk_metrics: Any,
        state: PortfolioState,
    ) -> str:
        """Generate LLM-powered risk analysis."""
        exposures = factor_exposure.to_dict()
        
        prompt = f"""Analyze the following portfolio risk characteristics:

Risk Metrics:
- Total Risk: {risk_metrics.total_risk_pct:.2f}%
- Systematic Risk: {risk_metrics.systematic_risk_pct:.2f}%
- Specific Risk: {risk_metrics.specific_risk_pct:.2f}%
- Portfolio Beta: {risk_metrics.beta:.2f}

Factor Exposures:
{chr(10).join(f'- {factor}: {exp:.3f}' for factor, exp in exposures.items())}

Portfolio Context:
- Number of holdings: {len(state.selected_tickers)}
- This is a pre-optimization analysis (equal-weighted)

Identify key risk drivers and any factor concentrations of concern."""

        try:
            response = await self._llm.generate(
                prompt=prompt,
                system_prompt=RISK_AGENT_SYSTEM_PROMPT,
                temperature=0.5,
                max_tokens=600,
            )
            return response.content
        except Exception:
            return self._generate_basic_analysis(factor_exposure, risk_metrics, state)

    def _generate_basic_analysis(
        self,
        factor_exposure: Any,
        risk_metrics: Any,
        state: PortfolioState,
    ) -> str:
        """Generate basic risk analysis without LLM."""
        exposures = factor_exposure.to_dict()
        
        # Find highest and lowest exposures
        sorted_exp = sorted(exposures.items(), key=lambda x: abs(x[1]), reverse=True)
        top_factor = sorted_exp[0] if sorted_exp else ("N/A", 0)
        
        return f"""Risk Analysis Summary

Portfolio Risk Metrics:
- Total Risk: {risk_metrics.total_risk_pct:.2f}%
- Systematic Risk: {risk_metrics.systematic_risk_pct:.2f}%
- Specific Risk: {risk_metrics.specific_risk_pct:.2f}%
- Beta: {risk_metrics.beta:.2f}

Dominant Factor Exposure: {top_factor[0]} ({top_factor[1]:.3f})

Factor Profile:
{chr(10).join(f'- {f}: {e:.3f}' for f, e in sorted_exp[:4])}

Risk assessment based on {len(state.selected_tickers)} equal-weighted positions."""

