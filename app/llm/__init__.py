"""LLM Provider module for flexible LLM integration."""

from app.llm.interfaces.llm_provider import ILLMProvider, LLMResponse

__all__ = [
    "ILLMProvider",
    "LLMResponse",
]

