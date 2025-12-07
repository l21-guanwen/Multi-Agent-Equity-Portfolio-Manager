"""
Base class for tools that ReAct agents can use.

Tools are actions that agents can invoke. The LLM reads the tool's
name and description to decide which tool to use for a given task.
"""

from abc import ABC, abstractmethod
from typing import Any, Optional
from pydantic import BaseModel, Field


class ToolResult(BaseModel):
    """Result from executing a tool."""
    
    success: bool = Field(..., description="Whether the tool executed successfully")
    data: Any = Field(default=None, description="The data returned by the tool")
    error: Optional[str] = Field(default=None, description="Error message if failed")
    message: str = Field(default="", description="Human-readable result message")
    
    def to_prompt_string(self) -> str:
        """Convert result to a string for LLM prompt."""
        if not self.success:
            return f"Tool execution failed: {self.error}"
        return self.message if self.message else str(self.data)


class BaseTool(ABC):
    """
    Base class for all tools.
    
    Tools are actions that agents can invoke. Each tool has:
    - name: Unique identifier for the tool
    - description: What the tool does (LLM reads this to decide usage)
    - execute(): The actual implementation
    
    Subclasses must implement:
    - name property
    - description property  
    - execute() method
    """
    
    @property
    @abstractmethod
    def name(self) -> str:
        """
        Unique name of the tool.
        
        This is used by the LLM to identify which tool to call.
        Example: "load_alpha_scores", "get_risk_model"
        """
        pass
    
    @property
    @abstractmethod
    def description(self) -> str:
        """
        Description of what the tool does.
        
        The LLM reads this to understand when to use this tool.
        Be specific about:
        - What data the tool returns
        - When to use it
        - Any required parameters
        """
        pass
    
    @property
    def parameters(self) -> dict[str, Any]:
        """
        Parameters the tool accepts.
        
        Returns a JSON schema describing the parameters.
        Override in subclasses if the tool needs parameters.
        """
        return {}
    
    @abstractmethod
    async def execute(self, **kwargs) -> ToolResult:
        """
        Execute the tool and return results.
        
        Args:
            **kwargs: Tool-specific parameters
            
        Returns:
            ToolResult with success status and data
        """
        pass
    
    def to_openai_function(self) -> dict:
        """Convert tool to OpenAI function calling format."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters or {"type": "object", "properties": {}},
            }
        }
    
    def __str__(self) -> str:
        return f"Tool({self.name}): {self.description}"

