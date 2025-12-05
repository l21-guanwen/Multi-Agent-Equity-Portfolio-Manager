"""
Application configuration using Pydantic Settings.

Supports environment variables and .env file for configuration.
"""

from functools import lru_cache
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings with environment variable support."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ===========================================
    # Application Settings
    # ===========================================
    app_name: str = "Multi-Agent Equity Portfolio Manager"
    debug: bool = False

    # ===========================================
    # LLM Provider Configuration
    # ===========================================
    llm_provider: Literal["openai", "deepseek", "anthropic"] = "openai"
    llm_model: str = "gpt-4"
    llm_temperature: float = 0.7
    llm_max_tokens: int = 2000

    # API Keys
    openai_api_key: str = ""
    deepseek_api_key: str = ""
    deepseek_base_url: str = "https://api.deepseek.com"
    anthropic_api_key: str = ""

    # ===========================================
    # Data Source Configuration
    # ===========================================
    data_source: Literal["csv", "database", "api"] = "csv"
    csv_data_path: str = "./tests/testdata"
    database_url: str = ""  # For future DB support

    # ===========================================
    # Solver Configuration
    # ===========================================
    solver: Literal["cvxpy", "scipy"] = "cvxpy"

    # ===========================================
    # Optimization Parameters
    # ===========================================
    risk_aversion: float = 0.01
    transaction_cost_penalty: float = 0.0
    max_iterations: int = 5

    # ===========================================
    # Portfolio Parameters
    # ===========================================
    portfolio_size: int = 25
    stock_active_weight_limit: float = 0.01  # ±1%
    sector_active_weight_limit: float = 0.02  # ±2%

    def get_llm_api_key(self) -> str:
        """Get the API key for the configured LLM provider."""
        if self.llm_provider == "openai":
            return self.openai_api_key
        elif self.llm_provider == "deepseek":
            return self.deepseek_api_key
        elif self.llm_provider == "anthropic":
            return self.anthropic_api_key
        else:
            raise ValueError(f"Unknown LLM provider: {self.llm_provider}")


@lru_cache
def get_settings() -> Settings:
    """
    Get cached settings instance.
    
    Uses lru_cache to ensure settings are only loaded once.
    """
    return Settings()

