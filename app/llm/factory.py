"""
LLM Provider Factory.

Creates LLM provider instances based on configuration.
Enables easy switching between providers.
"""

from typing import Optional

from app.core.config import Settings
from app.llm.interfaces.llm_provider import ILLMProvider
from app.llm.openai_provider import OpenAIProvider
from app.llm.deepseek_provider import DeepSeekProvider
from app.llm.anthropic_provider import AnthropicProvider


class LLMProviderFactory:
    """
    Factory for creating LLM provider instances.
    
    Supports:
    - OpenAI (GPT-4, GPT-3.5-turbo, etc.)
    - DeepSeek (DeepSeek-Chat, DeepSeek-Coder)
    - Anthropic (Claude 3 Opus, Sonnet, Haiku)
    
    Example:
        # From settings
        settings = get_settings()
        provider = LLMProviderFactory.create_from_settings(settings)
        
        # Direct creation
        provider = LLMProviderFactory.create(
            provider_name="openai",
            api_key="sk-...",
            model="gpt-4"
        )
    """

    SUPPORTED_PROVIDERS = ["openai", "deepseek", "anthropic"]

    @classmethod
    def create(
        cls,
        provider_name: str,
        api_key: str,
        model: Optional[str] = None,
        base_url: Optional[str] = None,
        **kwargs,
    ) -> ILLMProvider:
        """
        Create an LLM provider instance.
        
        Args:
            provider_name: Provider name ('openai', 'deepseek', 'anthropic')
            api_key: API key for the provider
            model: Optional model name override
            base_url: Optional custom base URL
            **kwargs: Additional provider-specific arguments
            
        Returns:
            ILLMProvider instance
            
        Raises:
            ValueError: If provider_name is not supported
        """
        provider_name = provider_name.lower()
        
        if provider_name not in cls.SUPPORTED_PROVIDERS:
            raise ValueError(
                f"Unsupported LLM provider: {provider_name}. "
                f"Supported providers: {cls.SUPPORTED_PROVIDERS}"
            )
        
        if provider_name == "openai":
            return cls._create_openai(api_key, model, base_url, **kwargs)
        elif provider_name == "deepseek":
            return cls._create_deepseek(api_key, model, base_url)
        elif provider_name == "anthropic":
            return cls._create_anthropic(api_key, model, base_url)
        else:
            raise ValueError(f"Unsupported provider: {provider_name}")

    @classmethod
    def create_from_settings(cls, settings: Settings) -> ILLMProvider:
        """
        Create an LLM provider from application settings.
        
        Reads provider configuration from Settings and creates
        the appropriate provider instance.
        
        Args:
            settings: Application settings instance
            
        Returns:
            ILLMProvider instance configured from settings
            
        Raises:
            ValueError: If required API key is not configured
        """
        provider_name = settings.llm_provider
        model = settings.llm_model
        
        if provider_name == "openai":
            if not settings.openai_api_key:
                raise ValueError("OPENAI_API_KEY is required for OpenAI provider")
            return cls._create_openai(
                api_key=settings.openai_api_key,
                model=model,
            )
        
        elif provider_name == "deepseek":
            if not settings.deepseek_api_key:
                raise ValueError("DEEPSEEK_API_KEY is required for DeepSeek provider")
            return cls._create_deepseek(
                api_key=settings.deepseek_api_key,
                model=model,
                base_url=settings.deepseek_base_url,
            )
        
        elif provider_name == "anthropic":
            if not settings.anthropic_api_key:
                raise ValueError("ANTHROPIC_API_KEY is required for Anthropic provider")
            return cls._create_anthropic(
                api_key=settings.anthropic_api_key,
                model=model,
            )
        
        else:
            raise ValueError(f"Unsupported LLM provider: {provider_name}")

    @classmethod
    def _create_openai(
        cls,
        api_key: str,
        model: Optional[str] = None,
        base_url: Optional[str] = None,
        **kwargs,
    ) -> OpenAIProvider:
        """Create OpenAI provider instance."""
        return OpenAIProvider(
            api_key=api_key,
            model=model or "gpt-4",
            base_url=base_url,
            **kwargs,
        )

    @classmethod
    def _create_deepseek(
        cls,
        api_key: str,
        model: Optional[str] = None,
        base_url: Optional[str] = None,
    ) -> DeepSeekProvider:
        """Create DeepSeek provider instance."""
        return DeepSeekProvider(
            api_key=api_key,
            model=model or "deepseek-chat",
            base_url=base_url,
        )

    @classmethod
    def _create_anthropic(
        cls,
        api_key: str,
        model: Optional[str] = None,
        base_url: Optional[str] = None,
    ) -> AnthropicProvider:
        """Create Anthropic provider instance."""
        return AnthropicProvider(
            api_key=api_key,
            model=model or "claude-3-opus-20240229",
            base_url=base_url,
        )

    @classmethod
    def get_default_model(cls, provider_name: str) -> str:
        """
        Get the default model for a provider.
        
        Args:
            provider_name: Provider name
            
        Returns:
            Default model name for the provider
        """
        defaults = {
            "openai": "gpt-4",
            "deepseek": "deepseek-chat",
            "anthropic": "claude-3-opus-20240229",
        }
        return defaults.get(provider_name.lower(), "gpt-4")

    @classmethod
    def get_available_models(cls, provider_name: str) -> list[str]:
        """
        Get list of known models for a provider.
        
        Args:
            provider_name: Provider name
            
        Returns:
            List of known model names
        """
        models = {
            "openai": [
                "gpt-4",
                "gpt-4-turbo",
                "gpt-4-turbo-preview",
                "gpt-4o",
                "gpt-4o-mini",
                "gpt-3.5-turbo",
            ],
            "deepseek": [
                "deepseek-chat",
                "deepseek-coder",
            ],
            "anthropic": [
                "claude-3-opus-20240229",
                "claude-3-sonnet-20240229",
                "claude-3-haiku-20240307",
                "claude-3-5-sonnet-20241022",
            ],
        }
        return models.get(provider_name.lower(), [])


def create_llm_provider(settings: Optional[Settings] = None) -> ILLMProvider:
    """
    Convenience function to create an LLM provider.
    
    Args:
        settings: Optional settings instance (uses get_settings() if not provided)
        
    Returns:
        Configured ILLMProvider instance
    """
    if settings is None:
        from app.core.config import get_settings
        settings = get_settings()
    
    return LLMProviderFactory.create_from_settings(settings)

