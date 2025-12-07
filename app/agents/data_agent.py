"""
Data Agent for the multi-agent portfolio management system.

Uses LangGraph's create_react_agent for tool-based reasoning.
When LLM is enabled, the agent decides which tools to call.
When LLM is disabled, falls back to loading all data directly.
"""

from datetime import datetime
from typing import Any, Optional

from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.language_models import BaseChatModel
from langgraph.prebuilt import create_react_agent as create_langgraph_react_agent

from app.agents.state import PortfolioState
from app.llm.interfaces.llm_provider import ILLMProvider
from app.services.interfaces.data_service import IDataService
from app.tools.langchain_tools import (
    load_benchmark,
    load_alpha_scores,
    load_risk_model,
    load_constraints,
    load_transaction_costs,
)


DATA_AGENT_SYSTEM_PROMPT = """You are a Data Ingestion Agent for an institutional equity portfolio manager.

Your role is to:
1. Load all required market data using available tools
2. Validate data completeness and consistency
3. Report any data quality issues

Required data sources:
1. Benchmark constituency (S&P 500 weights and securities)
2. Alpha model scores (expected returns)
3. Risk model (factor loadings and covariance)
4. Optimization constraints (position limits)
5. Transaction costs (for rebalancing)

For each dataset, use the appropriate tool to load it."""


class DataAgent:
    """
    Data Agent responsible for data ingestion and validation.
    
    When LLM is enabled (use_llm=True):
        - Creates a ReAct agent using LangGraph's create_react_agent
        - Agent decides which data tools to call
        - LLM validates data completeness
    
    When LLM is disabled (use_llm=False):
        - Directly loads all required data via data service
    """

    def __init__(
        self,
        data_service: IDataService,
        llm_provider: Optional[ILLMProvider] = None,
    ):
        """Initialize the Data Agent."""
        self._data_service = data_service
        self._llm_provider = llm_provider
        self._tools = [
            load_benchmark,
            load_alpha_scores,
            load_risk_model,
            load_constraints,
            load_transaction_costs,
        ]
    
    def _get_langchain_llm(self) -> Optional[BaseChatModel]:
        """Get LangChain-compatible LLM from provider."""
        if not self._llm_provider:
            return None
        return getattr(self._llm_provider, 'langchain_model', None)

    async def __call__(self, state: PortfolioState) -> dict[str, Any]:
        """Execute the data agent."""
        execution_log = [f"[DataAgent] Starting data loading..."]
        
        try:
            llm = self._get_langchain_llm()
            
            # Decide execution mode
            if llm and state.use_llm:
                execution_log.append("[DataAgent] Using LangGraph ReAct agent")
                result = await self._execute_with_react(state, llm, execution_log)
            else:
                execution_log.append("[DataAgent] Using direct data loading (no LLM)")
                result = await self._execute_direct(state, execution_log)
            
            return result
            
        except Exception as e:
            execution_log.append(f"[DataAgent] ERROR: {str(e)}")
            return {
                "data_validation_passed": False,
                "data_validation_issues": [str(e)],
                "error_message": f"Data loading failed: {str(e)}",
                "execution_log": execution_log,
                "current_agent": "data_agent",
            }

    async def _execute_with_react(
        self,
        state: PortfolioState,
        llm: BaseChatModel,
        execution_log: list[str],
    ) -> dict[str, Any]:
        """Execute data loading using LangGraph ReAct agent."""
        
        # Create ReAct agent with system prompt
        from langchain_core.messages import SystemMessage
        
        react_agent = create_langgraph_react_agent(
            model=llm,
            tools=self._tools,
            prompt=SystemMessage(content=DATA_AGENT_SYSTEM_PROMPT),
        )
        
        date_str = state.as_of_date or "latest available"
        task = f"""Load all required market data for portfolio construction (as of {date_str}).

Required datasets:
1. Benchmark constituency (S&P 500 weights)
2. Alpha model scores (expected returns)
3. Risk model (factor loadings)
4. Optimization constraints
5. Transaction costs

Use the appropriate tool for each dataset and report what was loaded."""
        
        # Invoke agent
        result = await react_agent.ainvoke({
            "messages": [HumanMessage(content=task)]
        })
        
        # Get summary from last AI message
        summary = ""
        for msg in reversed(result["messages"]):
            if isinstance(msg, AIMessage):
                summary = msg.content[:500]
                break
        
        execution_log.append(f"[DataAgent] ReAct agent completed data loading")
        
        # Still load data via service for state updates
        return await self._execute_direct(state, execution_log)

    async def _execute_direct(
        self, state: PortfolioState, execution_log: list[str]
    ) -> dict[str, Any]:
        """Execute data loading directly without LLM."""
        
        # Load all data using data service
        as_of_date = None
        if state.as_of_date:
            as_of_date = datetime.strptime(state.as_of_date, "%Y-%m-%d").date()
        
        data = await self._data_service.load_all_data(as_of_date)
        execution_log.append(f"[DataAgent] Loaded all data via service")
        
        # Validate
        validation = await self._data_service.validate_data(data)
        execution_log.append(f"[DataAgent] Validation: {'PASSED' if validation.is_valid else 'FAILED'}")
        
        # Build state updates
        benchmark = data.get("benchmark")
        alpha_model = data.get("alpha_model")
        risk_model = data.get("risk_model")
        transaction_costs = data.get("transaction_costs")
        
        updates: dict[str, Any] = {
            "data_validation_passed": validation.is_valid,
            "data_validation_issues": validation.issues + validation.warnings,
            "execution_log": execution_log,
            "current_agent": "data_agent",
        }
        
        if benchmark:
            updates["universe_tickers"] = [c.ticker for c in benchmark.constituents]
            updates["benchmark_weights"] = {
                c.ticker: c.benchmark_weight_pct for c in benchmark.constituents
            }
            updates["sector_mapping"] = {
                c.ticker: c.gics_sector for c in benchmark.constituents
            }
            updates["as_of_date"] = benchmark.as_of_date.isoformat()
        
        if alpha_model:
            updates["alpha_scores"] = {s.ticker: s.alpha_score for s in alpha_model.scores}
            updates["alpha_quintiles"] = {s.ticker: s.alpha_quintile for s in alpha_model.scores}
        
        if transaction_costs:
            execution_log.append(f"[DataAgent] Transaction costs: {transaction_costs.security_count} securities")
        
        updates["data_summary"] = {
            "benchmark_count": benchmark.security_count if benchmark else 0,
            "alpha_count": alpha_model.security_count if alpha_model else 0,
            "risk_count": risk_model.security_count if risk_model else 0,
            "validation_score": validation.data_quality_score,
        }
        
        execution_log.append("[DataAgent] Data loading complete")
        
        return updates
