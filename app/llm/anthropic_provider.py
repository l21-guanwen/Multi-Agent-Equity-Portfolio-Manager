"""
Anthropic LLM Provider implementation.

Supports Claude 3 Opus, Sonnet, Haiku and other Anthropic models.
"""

import json
from typing import Optional, Type

from anthropic import AsyncAnthropic
from pydantic import BaseModel

from app.llm.interfaces.llm_provider import (
    ILLMProvider,
    LLMMessage,
    LLMResponse,
)


class AnthropicProvider(ILLMProvider):
    """
    Anthropic LLM provider implementation.
    
    Uses the Anthropic Python SDK for API communication.
    
    Example:
        provider = AnthropicProvider(
            api_key="sk-ant-...",
            model="claude-3-opus-20240229"
        )
        response = await provider.generate("What is alpha?")
    """

    def __init__(
        self,
        api_key: str,
        model: str = "claude-3-opus-20240229",
        base_url: Optional[str] = None,
    ):
        """
        Initialize Anthropic provider.
        
        Args:
            api_key: Anthropic API key
            model: Model name (claude-3-opus-20240229, claude-3-sonnet-20240229, etc.)
            base_url: Optional custom base URL
        """
        self._model = model
        self._client = AsyncAnthropic(
            api_key=api_key,
            base_url=base_url,
        )

    @property
    def provider_name(self) -> str:
        """Get the name of the LLM provider."""
        return "anthropic"

    @property
    def model_name(self) -> str:
        """Get the name of the model being used."""
        return self._model

    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 2000,
        stop: Optional[list[str]] = None,
    ) -> LLMResponse:
        """
        Generate a response from Anthropic Claude.
        
        Args:
            prompt: The user prompt/question
            system_prompt: Optional system prompt to set context
            temperature: Sampling temperature (0.0-1.0)
            max_tokens: Maximum tokens in response
            stop: Optional stop sequences
            
        Returns:
            LLMResponse with generated text and metadata
        """
        kwargs = {
            "model": self._model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": [{"role": "user", "content": prompt}],
        }
        
        if system_prompt:
            kwargs["system"] = system_prompt
            
        if stop:
            kwargs["stop_sequences"] = stop
        
        response = await self._client.messages.create(**kwargs)
        
        # Extract content from response
        content = ""
        if response.content:
            content = response.content[0].text if response.content else ""
        
        return LLMResponse(
            content=content,
            model=response.model,
            provider=self.provider_name,
            usage={
                "prompt_tokens": response.usage.input_tokens,
                "completion_tokens": response.usage.output_tokens,
                "total_tokens": response.usage.input_tokens + response.usage.output_tokens,
            },
            finish_reason=response.stop_reason,
            raw_response=response.model_dump() if response else None,
        )

    async def generate_with_history(
        self,
        messages: list[LLMMessage],
        temperature: float = 0.7,
        max_tokens: int = 2000,
        stop: Optional[list[str]] = None,
    ) -> LLMResponse:
        """
        Generate a response given conversation history.
        
        Args:
            messages: List of conversation messages
            temperature: Sampling temperature
            max_tokens: Maximum tokens in response
            stop: Optional stop sequences
            
        Returns:
            LLMResponse with generated text and metadata
        """
        # Separate system message from others
        system_prompt = None
        formatted_messages = []
        
        for msg in messages:
            if msg.role == "system":
                system_prompt = msg.content
            else:
                formatted_messages.append({
                    "role": msg.role,
                    "content": msg.content,
                })
        
        kwargs = {
            "model": self._model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": formatted_messages,
        }
        
        if system_prompt:
            kwargs["system"] = system_prompt
            
        if stop:
            kwargs["stop_sequences"] = stop
        
        response = await self._client.messages.create(**kwargs)
        
        content = ""
        if response.content:
            content = response.content[0].text if response.content else ""
        
        return LLMResponse(
            content=content,
            model=response.model,
            provider=self.provider_name,
            usage={
                "prompt_tokens": response.usage.input_tokens,
                "completion_tokens": response.usage.output_tokens,
                "total_tokens": response.usage.input_tokens + response.usage.output_tokens,
            },
            finish_reason=response.stop_reason,
            raw_response=response.model_dump() if response else None,
        )

    async def generate_structured(
        self,
        prompt: str,
        response_schema: Type[BaseModel],
        system_prompt: Optional[str] = None,
        temperature: float = 0.3,
    ) -> BaseModel:
        """
        Generate a structured response matching a Pydantic schema.
        
        Uses explicit JSON formatting instructions.
        
        Args:
            prompt: The user prompt/question
            response_schema: Pydantic model class for response structure
            system_prompt: Optional system prompt
            temperature: Sampling temperature (lower for consistency)
            
        Returns:
            Instance of response_schema with generated data
            
        Raises:
            ValueError: If response cannot be parsed into schema
        """
        # Build schema instruction
        schema_json = response_schema.model_json_schema()
        schema_instruction = (
            f"You must respond with valid JSON that matches this schema:\n"
            f"{json.dumps(schema_json, indent=2)}\n\n"
            f"Respond ONLY with the JSON object, no additional text, "
            f"no markdown code blocks, just the raw JSON."
        )
        
        # Combine system prompts
        full_system_prompt = schema_instruction
        if system_prompt:
            full_system_prompt = f"{system_prompt}\n\n{schema_instruction}"
        
        response = await self._client.messages.create(
            model=self._model,
            max_tokens=4096,
            temperature=temperature,
            system=full_system_prompt,
            messages=[{"role": "user", "content": prompt}],
        )
        
        content = ""
        if response.content:
            content = response.content[0].text if response.content else "{}"
        
        # Clean up potential markdown formatting
        content = content.strip()
        if content.startswith("```json"):
            content = content[7:]
        if content.startswith("```"):
            content = content[3:]
        if content.endswith("```"):
            content = content[:-3]
        content = content.strip()
        
        try:
            data = json.loads(content)
            return response_schema.model_validate(data)
        except (json.JSONDecodeError, Exception) as e:
            raise ValueError(
                f"Failed to parse response into {response_schema.__name__}: {e}\n"
                f"Raw response: {content}"
            )

