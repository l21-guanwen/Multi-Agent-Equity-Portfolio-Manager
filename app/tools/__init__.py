"""
Tools module for ReAct agents.

Two types of tools are available:
1. LangChain tools (@tool decorator) - Used with create_react_agent
2. Class-based tools (BaseTool) - For custom implementations

LLM agents read tool docstrings to decide which tool to use.
"""

# Class-based tools (legacy/custom)
from app.tools.base import BaseTool, ToolResult
from app.tools.data_tools import (
    LoadBenchmarkTool,
    LoadUniverseTool,
    LoadAlphaScoresTool,
    LoadRiskModelTool,
    LoadConstraintsTool,
    LoadTransactionCostsTool,
    get_data_tools,
    get_tool_by_name,
)

# LangChain-compatible tools (@tool decorator)
from app.tools.langchain_tools import (
    load_benchmark,
    load_alpha_scores,
    load_risk_model,
    load_constraints,
    load_transaction_costs,
    DATA_AGENT_TOOLS,
    ALPHA_AGENT_TOOLS,
    RISK_AGENT_TOOLS,
)

__all__ = [
    # Base classes
    "BaseTool",
    "ToolResult",
    # Class-based tools
    "LoadBenchmarkTool",
    "LoadUniverseTool",
    "LoadAlphaScoresTool",
    "LoadRiskModelTool",
    "LoadConstraintsTool",
    "LoadTransactionCostsTool",
    "get_data_tools",
    "get_tool_by_name",
    # LangChain tools
    "load_benchmark",
    "load_alpha_scores",
    "load_risk_model",
    "load_constraints",
    "load_transaction_costs",
    "DATA_AGENT_TOOLS",
    "ALPHA_AGENT_TOOLS",
    "RISK_AGENT_TOOLS",
]

