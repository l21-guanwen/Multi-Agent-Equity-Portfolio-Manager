"""
LangGraph workflow definition for the multi-agent portfolio manager.

Defines the state graph connecting all agents.
"""

from typing import Any, Literal, Optional

from langgraph.graph import END, StateGraph

from app.agents.state import PortfolioState
from app.agents.data_agent import DataAgent
from app.agents.alpha_agent import AlphaAgent
from app.agents.risk_agent import RiskAgent
from app.agents.optimization_agent import OptimizationAgent
from app.agents.compliance_agent import ComplianceAgent

from app.llm.interfaces.llm_provider import ILLMProvider
from app.repositories.interfaces.constraint_repository import IConstraintRepository
from app.repositories.interfaces.risk_repository import IRiskRepository
from app.repositories.interfaces.transaction_cost_repository import ITransactionCostRepository
from app.services.interfaces.alpha_service import IAlphaService
from app.services.interfaces.compliance_service import IComplianceService
from app.services.interfaces.data_service import IDataService
from app.services.interfaces.optimization_service import IOptimizationService
from app.services.interfaces.risk_service import IRiskService


class PortfolioGraph:
    """
    LangGraph-based multi-agent portfolio management workflow.
    
    Workflow:
    1. Data Agent: Load and validate data
    2. Alpha Agent: Select top securities
    3. Risk Agent: Analyze risk characteristics
    4. Optimization Agent: Construct optimal portfolio
    5. Compliance Agent: Validate constraints
    
    If compliance fails and iterations remain, loop back to optimization.
    """

    def __init__(
        self,
        data_service: IDataService,
        alpha_service: IAlphaService,
        risk_service: IRiskService,
        optimization_service: IOptimizationService,
        compliance_service: IComplianceService,
        risk_repository: IRiskRepository,
        constraint_repository: IConstraintRepository,
        transaction_cost_repository: ITransactionCostRepository,
        llm_provider: Optional[ILLMProvider] = None,
    ):
        """
        Initialize the portfolio graph.
        
        Args:
            data_service: Service for data loading
            alpha_service: Service for alpha analysis
            risk_service: Service for risk calculations
            optimization_service: Service for portfolio optimization
            compliance_service: Service for compliance checking
            risk_repository: Repository for risk model data
            constraint_repository: Repository for constraint data
            transaction_cost_repository: Repository for transaction cost data
            llm_provider: Optional LLM provider for agent analysis
        """
        # Create agents
        self._data_agent = DataAgent(data_service, llm_provider)
        self._alpha_agent = AlphaAgent(alpha_service, llm_provider)
        self._risk_agent = RiskAgent(risk_service, risk_repository, llm_provider)
        self._optimization_agent = OptimizationAgent(
            optimization_service, risk_repository, constraint_repository, transaction_cost_repository, llm_provider
        )
        self._compliance_agent = ComplianceAgent(
            compliance_service, constraint_repository, llm_provider
        )
        
        # Build graph
        self._graph = self._build_graph()

    def _build_graph(self) -> StateGraph:
        """Build the LangGraph state graph."""
        # Create graph with PortfolioState
        graph = StateGraph(PortfolioState)
        
        # Add nodes (agents)
        graph.add_node("data_agent", self._data_agent)
        graph.add_node("alpha_agent", self._alpha_agent)
        graph.add_node("risk_agent", self._risk_agent)
        graph.add_node("optimization_agent", self._optimization_agent)
        graph.add_node("compliance_agent", self._compliance_agent)
        
        # Set entry point
        graph.set_entry_point("data_agent")
        
        # Add edges
        # Data -> check if valid
        graph.add_conditional_edges(
            "data_agent",
            self._route_after_data,
            {
                "continue": "alpha_agent",
                "error": END,
            }
        )
        
        # Alpha -> Risk
        graph.add_edge("alpha_agent", "risk_agent")
        
        # Risk -> Optimization
        graph.add_edge("risk_agent", "optimization_agent")
        
        # Optimization -> Compliance
        graph.add_edge("optimization_agent", "compliance_agent")
        
        # Compliance -> check if compliant or need to retry
        graph.add_conditional_edges(
            "compliance_agent",
            self._route_after_compliance,
            {
                "compliant": END,
                "retry": "optimization_agent",
                "max_iterations": END,
            }
        )
        
        return graph

    def _route_after_data(self, state: PortfolioState) -> Literal["continue", "error"]:
        """Route after data agent based on validation result."""
        if state.error_message:
            return "error"
        if not state.data_validation_passed:
            return "error"
        return "continue"

    def _route_after_compliance(
        self, state: PortfolioState
    ) -> Literal["compliant", "retry", "max_iterations"]:
        """Route after compliance agent based on compliance result."""
        if state.is_compliant:
            return "compliant"
        
        if state.iteration_count >= state.max_iterations:
            return "max_iterations"
        
        return "retry"

    def compile(self):
        """Compile the graph for execution."""
        return self._graph.compile()

    async def run(
        self,
        portfolio_id: str = "ALPHA_GROWTH_25",
        as_of_date: str = "",
        portfolio_size: int = 25,
        max_iterations: int = 5,
    ) -> PortfolioState:
        """
        Run the portfolio construction workflow.
        
        Args:
            portfolio_id: Portfolio identifier
            as_of_date: Data as-of date (YYYY-MM-DD)
            portfolio_size: Target number of holdings
            max_iterations: Maximum optimization iterations
            
        Returns:
            Final PortfolioState after workflow completion
        """
        # Create initial state
        initial_state = PortfolioState(
            portfolio_id=portfolio_id,
            as_of_date=as_of_date,
            portfolio_size=portfolio_size,
            max_iterations=max_iterations,
        )
        
        # Compile and run graph
        compiled = self.compile()
        
        # Run the graph
        final_state = await compiled.ainvoke(initial_state)
        
        return PortfolioState.model_validate(final_state)


def create_portfolio_graph(
    data_service: IDataService,
    alpha_service: IAlphaService,
    risk_service: IRiskService,
    optimization_service: IOptimizationService,
    compliance_service: IComplianceService,
    risk_repository: IRiskRepository,
    constraint_repository: IConstraintRepository,
    transaction_cost_repository: ITransactionCostRepository,
    llm_provider: Optional[ILLMProvider] = None,
) -> PortfolioGraph:
    """
    Factory function to create a PortfolioGraph instance.
    
    Args:
        data_service: Service for data loading
        alpha_service: Service for alpha analysis
        risk_service: Service for risk calculations
        optimization_service: Service for portfolio optimization
        compliance_service: Service for compliance checking
        risk_repository: Repository for risk model data
        constraint_repository: Repository for constraint data
        transaction_cost_repository: Repository for transaction cost data
        llm_provider: Optional LLM provider for agent analysis
        
    Returns:
        Configured PortfolioGraph instance
    """
    return PortfolioGraph(
        data_service=data_service,
        alpha_service=alpha_service,
        risk_service=risk_service,
        optimization_service=optimization_service,
        compliance_service=compliance_service,
        risk_repository=risk_repository,
        constraint_repository=constraint_repository,
        transaction_cost_repository=transaction_cost_repository,
        llm_provider=llm_provider,
    )

