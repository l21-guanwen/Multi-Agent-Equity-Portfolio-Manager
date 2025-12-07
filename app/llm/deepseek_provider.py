"""
DeepSeek LLM Provider implementation.

Supports DeepSeek-Chat, DeepSeek-Coder and other DeepSeek models.
Uses OpenAI-compatible API.
"""

import json
from typing import Optional, Type

import httpx
from pydantic import BaseModel
from langchain_openai import ChatOpenAI

from app.llm.interfaces.llm_provider import (
    ILLMProvider,
    LLMMessage,
    LLMResponse,
)


class DeepSeekProvider(ILLMProvider):
    """
    DeepSeek LLM provider implementation.
    
    DeepSeek uses an OpenAI-compatible API, so we use httpx
    for direct API calls to maintain flexibility.
    
    Example:
        provider = DeepSeekProvider(
            api_key="sk-...",
            model="deepseek-chat"
        )
        response = await provider.generate("What is alpha?")
    """

    DEFAULT_BASE_URL = "https://api.deepseek.com"

    def __init__(
        self,
        api_key: str,
        model: str = "deepseek-chat",
        base_url: Optional[str] = None,
    ):
        """
        Initialize DeepSeek provider.
        
        Args:
            api_key: DeepSeek API key
            model: Model name (deepseek-chat, deepseek-coder)
            base_url: Optional custom base URL
        """
        self._model = model
        self._api_key = api_key
        self._base_url = base_url or self.DEFAULT_BASE_URL
        self._client = httpx.AsyncClient(
            base_url=self._base_url,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            timeout=120.0,
        )
        
        # LangChain-compatible model for ReAct agents
        self._langchain_model = ChatOpenAI(
            model=model,
            api_key=api_key,
            base_url=f"{self._base_url}/v1",
            temperature=0.7,
        )

    @property
    def langchain_model(self) -> ChatOpenAI:
        """Get LangChain-compatible chat model for use with LangGraph agents."""
        return self._langchain_model

    @property
    def provider_name(self) -> str:
        """Get the name of the LLM provider."""
        return "deepseek"

    @property
    def model_name(self) -> str:
        """Get the name of the model being used."""
        return self._model

    async def _make_request(
        self,
        messages: list[dict],
        temperature: float = 0.7,
        max_tokens: int = 2000,
        stop: Optional[list[str]] = None,
        response_format: Optional[dict] = None,
    ) -> dict:
        """Make API request to DeepSeek."""
        payload = {
            "model": self._model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        
        if stop:
            payload["stop"] = stop
            
        if response_format:
            payload["response_format"] = response_format
        
        response = await self._client.post(
            "/v1/chat/completions",
            json=payload,
        )
        response.raise_for_status()
        return response.json()

    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 2000,
        stop: Optional[list[str]] = None,
    ) -> LLMResponse:
        """
        Generate a response from DeepSeek.
        
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
        
        response_data = await self._make_request(
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stop=stop,
        )
        
        choice = response_data["choices"][0]
        usage = response_data.get("usage", {})
        
        return LLMResponse(
            content=choice["message"]["content"] or "",
            model=response_data.get("model", self._model),
            provider=self.provider_name,
            usage={
                "prompt_tokens": usage.get("prompt_tokens", 0),
                "completion_tokens": usage.get("completion_tokens", 0),
                "total_tokens": usage.get("total_tokens", 0),
            },
            finish_reason=choice.get("finish_reason"),
            raw_response=response_data,
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
        
        response_data = await self._make_request(
            messages=formatted_messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stop=stop,
        )
        
        choice = response_data["choices"][0]
        usage = response_data.get("usage", {})
        
        return LLMResponse(
            content=choice["message"]["content"] or "",
            model=response_data.get("model", self._model),
            provider=self.provider_name,
            usage={
                "prompt_tokens": usage.get("prompt_tokens", 0),
                "completion_tokens": usage.get("completion_tokens", 0),
                "total_tokens": usage.get("total_tokens", 0),
            },
            finish_reason=choice.get("finish_reason"),
            raw_response=response_data,
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
        
        Uses JSON mode with schema instructions.
        
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
        
        response_data = await self._make_request(
            messages=messages,
            temperature=temperature,
            response_format={"type": "json_object"},
        )
        
        content = response_data["choices"][0]["message"]["content"] or "{}"
        
        try:
            data = json.loads(content)
            return response_schema.model_validate(data)
        except (json.JSONDecodeError, Exception) as e:
            raise ValueError(
                f"Failed to parse response into {response_schema.__name__}: {e}\n"
                f"Raw response: {content}"
            )

    async def close(self):
        """Close the HTTP client."""
        await self._client.aclose()

