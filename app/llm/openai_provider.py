"""
OpenAI LLM Provider implementation.

Supports GPT-4, GPT-4-turbo, GPT-3.5-turbo and other OpenAI models.
"""

import json
from typing import Optional, Type

from openai import AsyncOpenAI
from pydantic import BaseModel

from app.llm.interfaces.llm_provider import (
    ILLMProvider,
    LLMMessage,
    LLMResponse,
)


class OpenAIProvider(ILLMProvider):
    """
    OpenAI LLM provider implementation.
    
    Uses the OpenAI Python SDK for API communication.
    Supports structured output via JSON mode.
    
    Example:
        provider = OpenAIProvider(
            api_key="sk-...",
            model="gpt-4"
        )
        response = await provider.generate("What is alpha?")
    """

    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4",
        organization: Optional[str] = None,
        base_url: Optional[str] = None,
    ):
        """
        Initialize OpenAI provider.
        
        Args:
            api_key: OpenAI API key
            model: Model name (gpt-4, gpt-4-turbo, gpt-3.5-turbo)
            organization: Optional organization ID
            base_url: Optional custom base URL
        """
        self._model = model
        self._client = AsyncOpenAI(
            api_key=api_key,
            organization=organization,
            base_url=base_url,
        )

    @property
    def provider_name(self) -> str:
        """Get the name of the LLM provider."""
        return "openai"

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
        Generate a response from OpenAI.
        
        Args:
            prompt: The user prompt/question
            system_prompt: Optional system prompt to set context
            temperature: Sampling temperature (0.0-2.0)
            max_tokens: Maximum tokens in response
            stop: Optional stop sequences
            
        Returns:
            LLMResponse with generated text and metadata
        """
        messages = []
        
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        
        messages.append({"role": "user", "content": prompt})
        
        response = await self._client.chat.completions.create(
            model=self._model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stop=stop,
        )
        
        choice = response.choices[0]
        usage = response.usage
        
        return LLMResponse(
            content=choice.message.content or "",
            model=response.model,
            provider=self.provider_name,
            usage={
                "prompt_tokens": usage.prompt_tokens if usage else 0,
                "completion_tokens": usage.completion_tokens if usage else 0,
                "total_tokens": usage.total_tokens if usage else 0,
            },
            finish_reason=choice.finish_reason,
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
        formatted_messages = [
            {"role": msg.role, "content": msg.content}
            for msg in messages
        ]
        
        response = await self._client.chat.completions.create(
            model=self._model,
            messages=formatted_messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stop=stop,
        )
        
        choice = response.choices[0]
        usage = response.usage
        
        return LLMResponse(
            content=choice.message.content or "",
            model=response.model,
            provider=self.provider_name,
            usage={
                "prompt_tokens": usage.prompt_tokens if usage else 0,
                "completion_tokens": usage.completion_tokens if usage else 0,
                "total_tokens": usage.total_tokens if usage else 0,
            },
            finish_reason=choice.finish_reason,
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
        
        Uses OpenAI's JSON mode with schema instructions.
        
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
            f"Respond ONLY with the JSON object, no additional text."
        )
        
        # Combine system prompts
        full_system_prompt = schema_instruction
        if system_prompt:
            full_system_prompt = f"{system_prompt}\n\n{schema_instruction}"
        
        messages = [
            {"role": "system", "content": full_system_prompt},
            {"role": "user", "content": prompt},
        ]
        
        response = await self._client.chat.completions.create(
            model=self._model,
            messages=messages,
            temperature=temperature,
            response_format={"type": "json_object"},
        )
        
        content = response.choices[0].message.content or "{}"
        
        try:
            data = json.loads(content)
            return response_schema.model_validate(data)
        except (json.JSONDecodeError, Exception) as e:
            raise ValueError(
                f"Failed to parse response into {response_schema.__name__}: {e}\n"
                f"Raw response: {content}"
            )

