"""
Alpha Agent for the multi-agent portfolio management system.

Responsible for alpha signal analysis and security selection.
"""

from typing import Any, Optional

from app.agents.prompts import ALPHA_AGENT_SYSTEM_PROMPT
from app.agents.state import PortfolioState
from app.llm.interfaces.llm_provider import ILLMProvider
from app.models.alpha import AlphaModel, AlphaScore
from app.services.interfaces.alpha_service import IAlphaService


class AlphaAgent:
    """
    Alpha Agent responsible for security selection.
    
    This agent:
    1. Analyzes alpha scores across the universe
    2. Selects top securities for portfolio inclusion
    3. Uses LLM to explain selection rationale
    """

    def __init__(
        self,
        alpha_service: IAlphaService,
        llm_provider: Optional[ILLMProvider] = None,
    ):
        """
        Initialize the Alpha Agent.
        
        Args:
            alpha_service: Service for alpha analysis
            llm_provider: Optional LLM provider for analysis
        """
        self._alpha_service = alpha_service
        self._llm = llm_provider

    async def __call__(self, state: PortfolioState) -> dict[str, Any]:
        """
        Execute the alpha agent.
        
        Args:
            state: Current portfolio state
            
        Returns:
            Updated state fields
        """
        execution_log = [f"[AlphaAgent] Starting security selection..."]
        
        try:
            # Check prerequisites
            if not state.data_validation_passed:
                execution_log.append("[AlphaAgent] Skipped - data validation failed")
                return {
                    "execution_log": execution_log,
                    "current_agent": "alpha_agent",
                }
            
            if not state.alpha_scores:
                execution_log.append("[AlphaAgent] ERROR: No alpha scores available")
                return {
                    "error_message": "No alpha scores available for selection",
                    "execution_log": execution_log,
                    "current_agent": "alpha_agent",
                }
            
            # Build AlphaModel from state data
            alpha_scores = [
                AlphaScore(
                    ticker=ticker,
                    security_name=ticker,  # Simplified
                    gics_sector=state.sector_mapping.get(ticker, "Unknown"),
                    alpha_score=score,
                    alpha_quintile=state.alpha_quintiles.get(ticker, 3),
                    model_id="STATE_MODEL",
                    as_of_date=state.as_of_date or "2025-01-01",
                )
                for ticker, score in state.alpha_scores.items()
            ]
            
            alpha_model = AlphaModel(
                model_id="STATE_MODEL",
                scores=alpha_scores,
                as_of_date=state.as_of_date or "2025-01-01",
            )
            
            # Select top securities from Q1
            top_securities = await self._alpha_service.get_top_quintile_securities(
                alpha_model,
                top_n=state.portfolio_size,
            )
            
            selected_tickers = [s.ticker for s in top_securities]
            execution_log.append(
                f"[AlphaAgent] Selected {len(selected_tickers)} securities from Q1"
            )
            
            # Analyze alpha model
            analysis = await self._alpha_service.analyze_alpha_model(alpha_model)
            execution_log.append(f"[AlphaAgent] Alpha analysis complete")
            
            # Generate LLM analysis
            if self._llm:
                llm_analysis = await self._generate_analysis(
                    top_securities, analysis, state
                )
                execution_log.append(f"[AlphaAgent] LLM analysis generated")
            else:
                llm_analysis = self._generate_basic_analysis(
                    top_securities, analysis, state
                )
            
            execution_log.append(f"[AlphaAgent] Completed")
            
            return {
                "selected_tickers": selected_tickers,
                "alpha_analysis": llm_analysis,
                "execution_log": execution_log,
                "current_agent": "alpha_agent",
            }
            
        except Exception as e:
            execution_log.append(f"[AlphaAgent] ERROR: {str(e)}")
            return {
                "error_message": f"Alpha selection failed: {str(e)}",
                "execution_log": execution_log,
                "current_agent": "alpha_agent",
            }

    async def _generate_analysis(
        self,
        selected: list[AlphaScore],
        analysis: Any,
        state: PortfolioState,
    ) -> str:
        """Generate LLM-powered alpha analysis."""
        # Build selection summary
        sector_counts: dict[str, int] = {}
        for s in selected:
            sector_counts[s.gics_sector] = sector_counts.get(s.gics_sector, 0) + 1
        
        avg_alpha = sum(s.alpha_score for s in selected) / len(selected) if selected else 0
        
        prompt = f"""Analyze the following security selection for a concentrated equity portfolio:

Selection Summary:
- Portfolio size: {len(selected)} securities
- Average alpha score: {avg_alpha:.4f}
- All securities from Quintile 1 (top 20%)

Sector Distribution:
{chr(10).join(f'- {sector}: {count} securities' for sector, count in sorted(sector_counts.items(), key=lambda x: -x[1]))}

Top 5 Selected Securities:
{chr(10).join(f'- {s.ticker}: alpha={s.alpha_score:.4f}, sector={s.gics_sector}' for s in selected[:5])}

Universe Statistics:
- Total securities: {len(state.alpha_scores)}
- Quintile distribution: {analysis.quintile_distribution}
- Top sector by alpha: {analysis.top_sector}

Provide insights on the selection quality and any concentration concerns."""

        try:
            response = await self._llm.generate(
                prompt=prompt,
                system_prompt=ALPHA_AGENT_SYSTEM_PROMPT,
                temperature=0.5,
                max_tokens=600,
            )
            return response.content
        except Exception:
            return self._generate_basic_analysis(selected, analysis, state)

    def _generate_basic_analysis(
        self,
        selected: list[AlphaScore],
        analysis: Any,
        state: PortfolioState,
    ) -> str:
        """Generate basic alpha analysis without LLM."""
        sector_counts: dict[str, int] = {}
        for s in selected:
            sector_counts[s.gics_sector] = sector_counts.get(s.gics_sector, 0) + 1
        
        avg_alpha = sum(s.alpha_score for s in selected) / len(selected) if selected else 0
        
        top_sector = max(sector_counts.items(), key=lambda x: x[1])[0] if sector_counts else "N/A"
        
        return f"""Alpha Selection Analysis

Selected {len(selected)} securities from Q1 (top quintile)
Average Alpha Score: {avg_alpha:.4f}

Sector Distribution:
{chr(10).join(f'- {sector}: {count}' for sector, count in sorted(sector_counts.items(), key=lambda x: -x[1])[:5])}

Top Concentration: {top_sector} ({sector_counts.get(top_sector, 0)} securities)

Selection is {'well diversified' if len(sector_counts) >= 5 else 'concentrated'} across {len(sector_counts)} sectors."""

