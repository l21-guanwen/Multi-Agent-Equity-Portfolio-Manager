"""
Abstract interface for LLM providers.

Enables switching between different LLM providers (OpenAI, DeepSeek, Anthropic)
without changing the application code.
"""

from abc import ABC, abstractmethod
from typing import Any, Optional, Type

from pydantic import BaseModel, Field


class LLMResponse(BaseModel):
    """Standard response from an LLM provider."""

    content: str = Field(..., description="The generated text content")
    model: str = Field(..., description="The model that generated the response")
    provider: str = Field(..., description="The LLM provider name")
    usage: dict[str, int] = Field(
        default_factory=dict,
        description="Token usage statistics (prompt_tokens, completion_tokens, total_tokens)"
    )
    finish_reason: Optional[str] = Field(None, description="Reason for completion")
    raw_response: Optional[dict[str, Any]] = Field(
        None, 
        description="Raw response from the provider (for debugging)"
    )

    class Config:
        """Pydantic model configuration."""
        
        extra = "allow"


class LLMMessage(BaseModel):
    """A single message in a conversation."""

    role: str = Field(..., description="Message role: 'system', 'user', or 'assistant'")
    content: str = Field(..., description="Message content")


class ILLMProvider(ABC):
    """
    Abstract interface for LLM providers.
    
    Implementations should handle:
    - API authentication
    - Request formatting
    - Response parsing
    - Error handling
    - Rate limiting (optional)
    
    Example usage:
        provider = OpenAIProvider(api_key="sk-...", model="gpt-4")
        response = await provider.generate("What is portfolio optimization?")
    """

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Get the name of the LLM provider."""
        pass

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Get the name of the model being used."""
        pass

    @abstractmethod
    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 2000,
        stop: Optional[list[str]] = None,
    ) -> LLMResponse:
        """
        Generate a response from the LLM.
        
        Args:
            prompt: The user prompt/question
            system_prompt: Optional system prompt to set context
            temperature: Sampling temperature (0.0 = deterministic, 1.0 = creative)
            max_tokens: Maximum tokens in the response
            stop: Optional list of stop sequences
            
        Returns:
            LLMResponse containing the generated text and metadata
        """
        pass

    @abstractmethod
    async def generate_with_history(
        self,
        messages: list[LLMMessage],
        temperature: float = 0.7,
        max_tokens: int = 2000,
        stop: Optional[list[str]] = None,
    ) -> LLMResponse:
        """
        Generate a response given a conversation history.
        
        Args:
            messages: List of messages in the conversation
            temperature: Sampling temperature
            max_tokens: Maximum tokens in the response
            stop: Optional list of stop sequences
            
        Returns:
            LLMResponse containing the generated text and metadata
        """
        pass

    @abstractmethod
    async def generate_structured(
        self,
        prompt: str,
        response_schema: Type[BaseModel],
        system_prompt: Optional[str] = None,
        temperature: float = 0.3,
    ) -> BaseModel:
        """
        Generate a structured response matching a Pydantic schema.
        
        Uses JSON mode or function calling to ensure the response
        matches the expected schema.
        
        Args:
            prompt: The user prompt/question
            response_schema: Pydantic model class defining the expected response structure
            system_prompt: Optional system prompt to set context
            temperature: Sampling temperature (lower for more consistent structured output)
            
        Returns:
            Instance of response_schema populated with the generated data
            
        Raises:
            ValueError: If the response cannot be parsed into the schema
        """
        pass

    async def health_check(self) -> bool:
        """
        Check if the LLM provider is accessible and working.
        
        Returns:
            True if the provider is healthy, False otherwise
        """
        try:
            response = await self.generate(
                prompt="Say 'ok'",
                max_tokens=10,
                temperature=0.0,
            )
            return len(response.content) > 0
        except Exception:
            return False

