"""
Risk Agent for the multi-agent portfolio management system.

Uses LangGraph's create_react_agent for tool-based reasoning.
When LLM is enabled, the agent decides which tools to call.
When LLM is disabled, falls back to direct calculation.
"""

from typing import Any, Optional

from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.language_models import BaseChatModel
from langgraph.prebuilt import create_react_agent as create_langgraph_react_agent

from app.agents.state import PortfolioState
from app.llm.interfaces.llm_provider import ILLMProvider
from app.repositories.interfaces.risk_repository import IRiskRepository
from app.services.interfaces.risk_service import IRiskService
from app.tools.langchain_tools import load_risk_model


RISK_AGENT_SYSTEM_PROMPT = """You are a Risk Analysis Agent for an institutional equity portfolio manager.

Your task is to analyze portfolio risk using a Barra-style multi-factor model.

## Risk Model
The model has 8 factors:
1. Market - Broad market beta (systematic risk)
2. Size - Market cap factor (small vs large)
3. Value - Value vs growth characteristics
4. Momentum - Price momentum
5. Quality - Earnings quality
6. Volatility - Stock volatility
7. Growth - Revenue/earnings growth
8. Dividend Yield - Dividend characteristics

## Risk Calculation
Portfolio Risk = √(Systematic Risk² + Specific Risk²)

Where:
- Systematic Risk comes from factor exposures
- Specific Risk is idiosyncratic (stock-specific)

## Your Approach
1. Use load_risk_model tool to get factor data
2. Analyze factor exposures for selected securities
3. Calculate estimated portfolio volatility
4. Identify dominant risk factors"""


class RiskAgent:
    """
    Risk Agent for portfolio risk analysis.
    
    When LLM is enabled (use_llm=True):
        - Creates a ReAct agent using LangGraph's create_react_agent
        - Agent decides to call load_risk_model tool
        - LLM analyzes factor exposures
    
    When LLM is disabled (use_llm=False):
        - Directly loads risk model
        - Programmatically calculates risk metrics
    """

    def __init__(
        self,
        risk_service: IRiskService,
        risk_repository: IRiskRepository,
        llm_provider: Optional[ILLMProvider] = None,
    ):
        """Initialize the Risk Agent."""
        self._risk_service = risk_service
        self._risk_repo = risk_repository
        self._llm_provider = llm_provider
        self._tools = [load_risk_model]
    
    def _get_langchain_llm(self) -> Optional[BaseChatModel]:
        """Get LangChain-compatible LLM from provider."""
        if not self._llm_provider:
            return None
        return getattr(self._llm_provider, 'langchain_model', None)
    
    async def __call__(self, state: PortfolioState) -> dict[str, Any]:
        """Execute the risk agent."""
        execution_log = [f"[RiskAgent] Starting risk analysis..."]
        
        try:
            # Check prerequisites
            if not state.selected_tickers:
                execution_log.append("[RiskAgent] Skipped - no securities selected")
                return {
                    "execution_log": execution_log,
                    "current_agent": "risk_agent",
                }
            
            llm = self._get_langchain_llm()
            
            # Use ReAct agent if LLM available and enabled
            if llm and state.use_llm:
                execution_log.append("[RiskAgent] Using LangGraph ReAct agent")
                result = await self._execute_with_react(state, llm, execution_log)
            else:
                execution_log.append("[RiskAgent] Using direct calculation (no LLM)")
                result = await self._execute_direct(state, execution_log)
            
            return result
            
        except Exception as e:
            execution_log.append(f"[RiskAgent] ERROR: {str(e)}")
            return {
                "error_message": f"Risk analysis failed: {str(e)}",
                "execution_log": execution_log,
                "current_agent": "risk_agent",
            }

    async def _execute_with_react(
        self, 
        state: PortfolioState, 
        llm: BaseChatModel,
        execution_log: list[str],
    ) -> dict[str, Any]:
        """Execute using LangGraph ReAct agent."""
        
        # Create ReAct agent using LangGraph's prebuilt agent
        from langchain_core.messages import SystemMessage
        
        react_agent = create_langgraph_react_agent(
            model=llm,
            tools=self._tools,
            prompt=SystemMessage(content=RISK_AGENT_SYSTEM_PROMPT),
        )
        
        # Prepare task
        tickers_str = ", ".join(state.selected_tickers[:10])
        task = f"""Analyze risk for the selected portfolio.

Selected securities: {tickers_str}{'...' if len(state.selected_tickers) > 10 else ''}
Total: {len(state.selected_tickers)} securities

Steps:
1. Use load_risk_model tool to get factor loadings
2. Calculate portfolio factor exposures (assume equal weights)
3. Estimate portfolio volatility

Provide:
- Factor exposures for each of the 8 factors
- Estimated portfolio volatility (%)
- Key risk insights"""

        # Invoke agent
        result = await react_agent.ainvoke({
            "messages": [HumanMessage(content=task)]
        })
        
        # Get analysis from last AI message
        analysis = ""
        for msg in reversed(result["messages"]):
            if isinstance(msg, AIMessage):
                analysis = msg.content[:500]
                break
        
        # Calculate actual risk metrics
        risk_metrics = await self._calculate_risk_metrics(state, execution_log)
        
        execution_log.append(f"[RiskAgent] Risk analysis complete")
        
        return {
            **risk_metrics,
            "risk_analysis": analysis,
            "execution_log": execution_log,
            "current_agent": "risk_agent",
        }

    async def _execute_direct(
        self, 
        state: PortfolioState, 
        execution_log: list[str],
    ) -> dict[str, Any]:
        """Execute risk calculation directly without LLM."""
        
        risk_metrics = await self._calculate_risk_metrics(state, execution_log)
        
        # Generate analysis
        exposures = risk_metrics.get("factor_exposures", {})
        risk_pct = risk_metrics.get("portfolio_risk_pct", 15.0)
        
        dominant_factors = sorted(
            exposures.items(), 
            key=lambda x: abs(x[1]), 
            reverse=True
        )[:3]
        
        analysis = f"""Risk Analysis Complete

Portfolio Volatility: {risk_pct:.2f}%

Dominant Factor Exposures:
{chr(10).join(f'- {f}: {e:.3f}' for f, e in dominant_factors)}

Market Beta: {exposures.get('Market', 1.0):.2f}

Risk Profile: {'High' if risk_pct > 20 else 'Moderate' if risk_pct > 15 else 'Low'} volatility portfolio."""
        
        return {
            **risk_metrics,
            "risk_analysis": analysis,
            "execution_log": execution_log,
            "current_agent": "risk_agent",
        }

    async def _calculate_risk_metrics(
        self, 
        state: PortfolioState,
        execution_log: list[str],
    ) -> dict[str, Any]:
        """Calculate risk metrics for the portfolio."""
        
        # Load risk model
        risk_model = await self._risk_repo.get_risk_model()
        
        if not risk_model:
            execution_log.append("[RiskAgent] Risk model not available")
            return {
                "factor_exposures": {},
                "portfolio_risk_pct": 15.0,  # Default estimate
            }
        
        execution_log.append(f"[RiskAgent] Loaded risk model: {risk_model.security_count} securities")
        
        # Create equal-weight portfolio
        n = len(state.selected_tickers)
        equal_weights = {t: 1.0 / n for t in state.selected_tickers}
        
        # Calculate factor exposures
        factor_exposure = await self._risk_service.calculate_factor_exposure(
            equal_weights, risk_model
        )
        
        # Calculate portfolio risk
        risk_metrics = await self._risk_service.calculate_portfolio_risk(
            equal_weights, risk_model
        )
        
        execution_log.append(
            f"[RiskAgent] Portfolio risk: {risk_metrics.total_risk_pct:.2f}%"
        )
        
        return {
            "factor_exposures": {
                "Market": factor_exposure.market,
                "Size": factor_exposure.size,
                "Value": factor_exposure.value,
                "Momentum": factor_exposure.momentum,
                "Quality": factor_exposure.quality,
                "Volatility": factor_exposure.volatility,
                "Growth": factor_exposure.growth,
                "Dividend_Yield": factor_exposure.dividend_yield,
            },
            "portfolio_risk_pct": risk_metrics.total_risk_pct,
        }
