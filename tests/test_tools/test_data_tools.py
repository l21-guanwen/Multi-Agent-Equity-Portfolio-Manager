"""
Tests for data loading tools.
"""

import pytest
from pathlib import Path

from app.tools.base import BaseTool, ToolResult
from app.tools.data_tools import (
    LoadBenchmarkTool,
    LoadAlphaScoresTool,
    LoadRiskModelTool,
    LoadConstraintsTool,
    LoadTransactionCostsTool,
    get_data_tools,
    get_tool_by_name,
)


class TestBaseTool:
    """Tests for BaseTool interface."""

    def test_tool_result_success(self):
        """Test successful ToolResult."""
        result = ToolResult(
            success=True,
            data={"key": "value"},
            message="Success"
        )
        
        assert result.success is True
        assert result.data == {"key": "value"}
        assert "Success" in result.to_prompt_string()

    def test_tool_result_failure(self):
        """Test failed ToolResult."""
        result = ToolResult(
            success=False,
            error="Something went wrong"
        )
        
        assert result.success is False
        assert "failed" in result.to_prompt_string().lower()


class TestLoadBenchmarkTool:
    """Tests for LoadBenchmarkTool."""

    def test_tool_has_name(self):
        """Test tool has proper name."""
        tool = LoadBenchmarkTool()
        assert tool.name == "load_benchmark"

    def test_tool_has_description(self):
        """Test tool has description for LLM."""
        tool = LoadBenchmarkTool()
        assert len(tool.description) > 50
        assert "benchmark" in tool.description.lower()
        assert "S&P 500" in tool.description or "500" in tool.description

    def test_tool_can_convert_to_openai_function(self):
        """Test tool converts to OpenAI function format."""
        tool = LoadBenchmarkTool()
        func = tool.to_openai_function()
        
        assert func["type"] == "function"
        assert func["function"]["name"] == "load_benchmark"
        assert "description" in func["function"]

    @pytest.mark.asyncio
    async def test_execute_loads_benchmark(self, test_data_path: Path):
        """Test tool loads benchmark data."""
        tool = LoadBenchmarkTool()
        result = await tool.execute()
        
        assert result.success is True
        assert result.data is not None
        assert result.data.security_count > 0


class TestLoadAlphaScoresTool:
    """Tests for LoadAlphaScoresTool."""

    def test_tool_has_name(self):
        """Test tool has proper name."""
        tool = LoadAlphaScoresTool()
        assert tool.name == "load_alpha_scores"

    def test_tool_description_explains_quintiles(self):
        """Test tool description explains alpha quintiles."""
        tool = LoadAlphaScoresTool()
        assert "quintile" in tool.description.lower()
        assert "Q1" in tool.description or "0.80" in tool.description

    @pytest.mark.asyncio
    async def test_execute_loads_alpha(self, test_data_path: Path):
        """Test tool loads alpha scores."""
        tool = LoadAlphaScoresTool()
        result = await tool.execute()
        
        assert result.success is True
        assert result.data is not None
        assert result.data.security_count > 0


class TestLoadRiskModelTool:
    """Tests for LoadRiskModelTool."""

    def test_tool_has_name(self):
        """Test tool has proper name."""
        tool = LoadRiskModelTool()
        assert tool.name == "load_risk_model"

    def test_tool_description_lists_factors(self):
        """Test tool description lists risk factors."""
        tool = LoadRiskModelTool()
        desc_lower = tool.description.lower()
        assert "factor" in desc_lower
        # Should mention at least some factors
        assert any(f in desc_lower for f in ["market", "size", "value", "momentum"])

    @pytest.mark.asyncio
    async def test_execute_loads_risk_model(self, test_data_path: Path):
        """Test tool loads risk model."""
        tool = LoadRiskModelTool()
        result = await tool.execute()
        
        assert result.success is True
        assert result.data is not None


class TestLoadConstraintsTool:
    """Tests for LoadConstraintsTool."""

    def test_tool_has_name(self):
        """Test tool has proper name."""
        tool = LoadConstraintsTool()
        assert tool.name == "load_constraints"

    def test_tool_description_explains_limits(self):
        """Test tool description explains constraint limits."""
        tool = LoadConstraintsTool()
        assert "±1%" in tool.description or "1%" in tool.description
        assert "±2%" in tool.description or "2%" in tool.description

    @pytest.mark.asyncio
    async def test_execute_loads_constraints(self, test_data_path: Path):
        """Test tool loads constraints."""
        tool = LoadConstraintsTool()
        result = await tool.execute()
        
        assert result.success is True
        assert result.data is not None


class TestToolRegistry:
    """Tests for tool registry functions."""

    def test_get_data_tools_returns_all_tools(self):
        """Test get_data_tools returns all data tools."""
        tools = get_data_tools()
        
        assert len(tools) >= 5
        tool_names = [t.name for t in tools]
        assert "load_benchmark" in tool_names
        assert "load_alpha_scores" in tool_names
        assert "load_risk_model" in tool_names

    def test_get_tool_by_name_finds_tool(self):
        """Test get_tool_by_name finds existing tool."""
        tool = get_tool_by_name("load_benchmark")
        
        assert tool is not None
        assert tool.name == "load_benchmark"

    def test_get_tool_by_name_returns_none_for_unknown(self):
        """Test get_tool_by_name returns None for unknown tool."""
        tool = get_tool_by_name("nonexistent_tool")
        
        assert tool is None

    def test_all_tools_are_base_tool_instances(self):
        """Test all tools inherit from BaseTool."""
        tools = get_data_tools()
        
        for tool in tools:
            assert isinstance(tool, BaseTool)

