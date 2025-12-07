"""
Alpha Agent for the multi-agent portfolio management system.

Uses LangGraph's create_react_agent for tool-based reasoning.
When LLM is enabled, the agent decides which tools to call.
When LLM is disabled, falls back to direct execution.
"""

import json
from typing import Any, Optional

from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.language_models import BaseChatModel
from langgraph.prebuilt import create_react_agent as create_langgraph_react_agent

from app.agents.state import PortfolioState
from app.llm.interfaces.llm_provider import ILLMProvider
from app.models.alpha import AlphaModel, AlphaScore
from app.services.interfaces.alpha_service import IAlphaService
from app.tools.langchain_tools import load_alpha_scores, load_benchmark


ALPHA_AGENT_SYSTEM_PROMPT = """You are an Alpha Analysis Agent for an institutional equity portfolio manager.

Your task is to select securities for portfolio inclusion based on alpha signals.

## What is Alpha?
Alpha scores (0-1) represent expected excess returns:
- 0.80-1.00 = Quintile 1 (Top 20%, strongest buy signal)
- 0.60-0.80 = Quintile 2
- 0.40-0.60 = Quintile 3 (Neutral)
- 0.20-0.40 = Quintile 4  
- 0.00-0.20 = Quintile 5 (Bottom 20%, weakest)

## Selection Criteria
1. Focus on Quintile 1 (Q1) securities only
2. Higher alpha score = higher expected return
3. Consider sector diversification
4. Select exactly the number of securities specified

## Your Approach
1. Use the load_alpha_scores tool to get alpha data
2. Identify Q1 securities
3. Select the top N by alpha score
4. Report your selection with analysis"""


class AlphaAgent:
    """
    Alpha Agent for security selection using alpha signals.
    
    When LLM is enabled (use_llm=True):
        - Creates a ReAct agent using LangGraph's create_react_agent
        - Agent decides to call load_alpha_scores tool
        - LLM analyzes results and selects top securities
    
    When LLM is disabled (use_llm=False):
        - Directly loads alpha scores
        - Programmatically selects top N from Q1
    """

    def __init__(
        self,
        alpha_service: IAlphaService,
        llm_provider: Optional[ILLMProvider] = None,
    ):
        """Initialize the Alpha Agent."""
        self._alpha_service = alpha_service
        self._llm_provider = llm_provider
        self._tools = [load_alpha_scores, load_benchmark]
    
    def _get_langchain_llm(self) -> Optional[BaseChatModel]:
        """Get LangChain-compatible LLM from provider."""
        if not self._llm_provider:
            return None
        
        # LLM provider should expose a LangChain-compatible model
        return getattr(self._llm_provider, 'langchain_model', None)
    
    async def __call__(self, state: PortfolioState) -> dict[str, Any]:
        """Execute the alpha agent."""
        execution_log = [f"[AlphaAgent] Starting alpha analysis..."]
        
        try:
            # Check prerequisites
            if not state.data_validation_passed:
                execution_log.append("[AlphaAgent] Skipped - data validation failed")
                return {
                    "execution_log": execution_log,
                    "current_agent": "alpha_agent",
                }
            
            llm = self._get_langchain_llm()
            
            # Use ReAct agent if LLM available and enabled
            if llm and state.use_llm:
                execution_log.append("[AlphaAgent] Using LangGraph ReAct agent")
                result = await self._execute_with_react(state, llm, execution_log)
            else:
                execution_log.append("[AlphaAgent] Using direct execution (no LLM)")
                result = await self._execute_direct(state, execution_log)
            
            return result
            
        except Exception as e:
            execution_log.append(f"[AlphaAgent] ERROR: {str(e)}")
            return {
                "error_message": f"Alpha analysis failed: {str(e)}",
                "execution_log": execution_log,
                "current_agent": "alpha_agent",
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
            prompt=SystemMessage(content=ALPHA_AGENT_SYSTEM_PROMPT),
        )
        
        # Prepare task
        task = f"""Select the top {state.portfolio_size} securities for the portfolio.

Steps:
1. Use the load_alpha_scores tool to get alpha data
2. Identify securities in Quintile 1 (Q1)
3. Select the top {state.portfolio_size} by alpha score
4. Provide the list of selected tickers

Respond with:
- selected_tickers: List of {state.portfolio_size} tickers
- Average alpha of selection
- Sector distribution"""

        # Invoke agent
        result = await react_agent.ainvoke({
            "messages": [HumanMessage(content=task)]
        })
        
        # Extract selected tickers from agent messages
        selected_tickers = self._extract_tickers_from_messages(
            result["messages"], state.portfolio_size, state
        )
        
        # Get analysis from last AI message
        analysis = ""
        for msg in reversed(result["messages"]):
            if isinstance(msg, AIMessage):
                analysis = msg.content[:500]
                break
        
        execution_log.append(f"[AlphaAgent] ReAct agent selected {len(selected_tickers)} securities")
        
        return {
            "selected_tickers": selected_tickers,
            "alpha_analysis": analysis,
            "execution_log": execution_log,
            "current_agent": "alpha_agent",
        }
    
    def _extract_tickers_from_messages(
        self, 
        messages: list, 
        target_count: int,
        state: PortfolioState,
    ) -> list[str]:
        """Extract ticker list from agent messages."""
        
        # Try to find tickers in the last AI message
        for msg in reversed(messages):
            if isinstance(msg, AIMessage):
                content = msg.content
                
                # Try JSON extraction
                try:
                    if "selected_tickers" in content:
                        import re
                        match = re.search(r'"selected_tickers"\s*:\s*\[([^\]]+)\]', content)
                        if match:
                            tickers_str = match.group(1)
                            tickers = [t.strip().strip('"\'') for t in tickers_str.split(',')]
                            return [t for t in tickers if t in state.alpha_scores][:target_count]
                except:
                    pass
                
                # Try to find ticker patterns
                import re
                ticker_pattern = r'\b([A-Z]{1,5})\b'
                potential_tickers = re.findall(ticker_pattern, content)
                valid_tickers = [t for t in potential_tickers if t in state.alpha_scores]
                
                if len(valid_tickers) >= target_count // 2:
                    return valid_tickers[:target_count]
        
        # Fallback: select from state's alpha scores
        return self._select_top_by_alpha(state, target_count)
    
    def _select_top_by_alpha(self, state: PortfolioState, n: int) -> list[str]:
        """Select top N securities by alpha score."""
        if not state.alpha_scores:
            return []
        
        # Filter to Q1 if quintiles available
        if state.alpha_quintiles:
            q1_tickers = [t for t, q in state.alpha_quintiles.items() if q == 1]
            scores = {t: state.alpha_scores.get(t, 0) for t in q1_tickers}
        else:
            scores = state.alpha_scores
        
        sorted_tickers = sorted(scores.keys(), key=lambda t: scores[t], reverse=True)
        return sorted_tickers[:n]

    async def _execute_direct(
        self, 
        state: PortfolioState, 
        execution_log: list[str],
    ) -> dict[str, Any]:
        """Execute alpha selection directly without LLM."""
        
        # Select top securities by alpha
        selected_tickers = self._select_top_by_alpha(state, state.portfolio_size)
        
        if not selected_tickers:
            # Load alpha scores if not in state
            execution_log.append("[AlphaAgent] Loading alpha scores...")
            result = await load_alpha_scores.ainvoke({})
            
            if "error" in result:
                return {
                    "error_message": result["error"],
                    "execution_log": execution_log,
                    "current_agent": "alpha_agent",
                }
            
            # Select from loaded data
            scores = result.get("scores", {})
            quintiles = result.get("quintiles", {})
            
            q1_tickers = [t for t, q in quintiles.items() if q == 1]
            sorted_q1 = sorted(q1_tickers, key=lambda t: scores.get(t, 0), reverse=True)
            selected_tickers = sorted_q1[:state.portfolio_size]
        
        execution_log.append(f"[AlphaAgent] Selected {len(selected_tickers)} securities from Q1")
        
        # Generate analysis
        avg_alpha = sum(state.alpha_scores.get(t, 0) for t in selected_tickers) / len(selected_tickers) if selected_tickers else 0
        
        sector_counts = {}
        for t in selected_tickers:
            sector = state.sector_mapping.get(t, "Unknown")
            sector_counts[sector] = sector_counts.get(sector, 0) + 1
        
        analysis = f"""Alpha Selection Complete

Selected {len(selected_tickers)} securities from Quintile 1 (top 20% by alpha).
Average Alpha Score: {avg_alpha:.4f}

Sector Distribution:
{chr(10).join(f'- {s}: {c}' for s, c in sorted(sector_counts.items(), key=lambda x: -x[1])[:5])}

Top 5 by Alpha:
{chr(10).join(f'- {t}: {state.alpha_scores.get(t, 0):.4f}' for t in selected_tickers[:5])}"""
        
        return {
            "selected_tickers": selected_tickers,
            "alpha_analysis": analysis,
            "execution_log": execution_log,
            "current_agent": "alpha_agent",
        }
                              