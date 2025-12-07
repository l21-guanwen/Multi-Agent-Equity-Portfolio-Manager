"""
OpenRouter LLM Provider implementation.

OpenRouter provides a unified API to access multiple LLM providers
(OpenAI, Anthropic, Meta, Google, etc.) through a single API endpoint.

See https://openrouter.ai/docs for available models and pricing.
"""

import json
from typing import Optional, Type

from openai import AsyncOpenAI
from pydantic import BaseModel
from langchain_openai import ChatOpenAI

from app.llm.interfaces.llm_provider import (
    ILLMProvider,
    LLMMessage,
    LLMResponse,
)


# Popular models available on OpenRouter
OPENROUTER_MODELS = {
    # OpenAI
    "openai/gpt-4o": "OpenAI GPT-4o",
    "openai/gpt-4o-mini": "OpenAI GPT-4o Mini",
    "openai/gpt-4-turbo": "OpenAI GPT-4 Turbo",
    "openai/o1-preview": "OpenAI o1 Preview",
    "openai/o1-mini": "OpenAI o1 Mini",
    # Anthropic
    "anthropic/claude-3.5-sonnet": "Anthropic Claude 3.5 Sonnet",
    "anthropic/claude-3-opus": "Anthropic Claude 3 Opus",
    "anthropic/claude-3-sonnet": "Anthropic Claude 3 Sonnet",
    "anthropic/claude-3-haiku": "Anthropic Claude 3 Haiku",
    # Meta Llama
    "meta-llama/llama-3.1-405b-instruct": "Meta Llama 3.1 405B",
    "meta-llama/llama-3.1-70b-instruct": "Meta Llama 3.1 70B",
    "meta-llama/llama-3.1-8b-instruct": "Meta Llama 3.1 8B",
    # Google
    "google/gemini-pro-1.5": "Google Gemini Pro 1.5",
    "google/gemini-flash-1.5": "Google Gemini Flash 1.5",
    # Mistral
    "mistralai/mistral-large": "Mistral Large",
    "mistralai/mixtral-8x22b-instruct": "Mixtral 8x22B",
    # DeepSeek
    "deepseek/deepseek-chat": "DeepSeek Chat",
    "deepseek/deepseek-coder": "DeepSeek Coder",
    # Qwen
    "qwen/qwen-2.5-72b-instruct": "Qwen 2.5 72B",
    # Cohere
    "cohere/command-r-plus": "Cohere Command R+",
}


class OpenRouterProvider(ILLMProvider):
    """
    OpenRouter LLM provider implementation.
    
    OpenRouter provides access to 100+ models through a unified API.
    Uses OpenAI-compatible API format.
    
    Example:
        provider = OpenRouterProvider(
            api_key="sk-or-...",
            model="anthropic/claude-3.5-sonnet"
        )
        response = await provider.generate("What is alpha?")
        
    Available models include:
    - OpenAI: openai/gpt-4o, openai/gpt-4-turbo, openai/o1-preview
    - Anthropic: anthropic/claude-3.5-sonnet, anthropic/claude-3-opus
    - Meta: meta-llama/llama-3.1-405b-instruct
    - Google: google/gemini-pro-1.5
    - And many more: https://openrouter.ai/models
    """

    OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

    def __init__(
        self,
        api_key: str,
        model: str = "openai/gpt-4o",
        site_url: Optional[str] = None,
        site_name: Optional[str] = None,
    ):
        """
        Initialize OpenRouter provider.
        
        Args:
            api_key: OpenRouter API key (starts with sk-or-)
            model: Model identifier (e.g., "anthropic/claude-3.5-sonnet")
            site_url: Optional URL for rankings/analytics
            site_name: Optional site name for rankings/analytics
        """
        self._model = model
        self._api_key = api_key
        self._site_url = site_url
        self._site_name = site_name
        
        # Build headers for OpenRouter
        default_headers = {
            "HTTP-Referer": site_url or "https://github.com/multi-agent-portfolio",
            "X-Title": site_name or "Multi-Agent Portfolio Manager",
        }
        
        # Create OpenAI client with OpenRouter base URL
        self._client = AsyncOpenAI(
            api_key=api_key,
            base_url=self.OPENROUTER_BASE_URL,
            default_headers=default_headers,
        )
        
        # LangChain-compatible model for ReAct agents
        self._langchain_model = ChatOpenAI(
            model=model,
            api_key=api_key,
            base_url=self.OPENROUTER_BASE_URL,
            default_headers=default_headers,
            temperature=0.7,
        )

    @property
    def langchain_model(self) -> ChatOpenAI:
        """Get LangChain-compatible chat model for use with LangGraph agents."""
        return self._langchain_model

    @property
    def provider_name(self) -> str:
        """Get the name of the LLM provider."""
        return "openrouter"

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
        Generate a response using OpenRouter.
        
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
        
        # Note: JSON mode may not be supported by all models on OpenRouter
        # Fall back to prompt-based approach
        try:
            response = await self._client.chat.completions.create(
                model=self._model,
                messages=messages,
                temperature=temperature,
                response_format={"type": "json_object"},
            )
        except Exception:
            # Fallback without response_format for models that don't support it
            response = await self._client.chat.completions.create(
                model=self._model,
                messages=messages,
                temperature=temperature,
            )
        
        content = response.choices[0].message.content or "{}"
        
        # Extract JSON from response (handle markdown code blocks)
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0].strip()
        elif "```" in content:
            content = content.split("```")[1].split("```")[0].strip()
        
        try:
            data = json.loads(content)
            return response_schema.model_validate(data)
        except (json.JSONDecodeError, Exception) as e:
            raise ValueError(
                f"Failed to parse response into {response_schema.__name__}: {e}\n"
                f"Raw response: {content}"
            )

    @classmethod
    def get_available_models(cls) -> dict[str, str]:
        """
        Get dictionary of popular models available on OpenRouter.
        
        Returns:
            Dict of model_id -> description
        """
        return OPENROUTER_MODELS.copy()

