"""
Multi-Agent Equity Portfolio Manager

Main FastAPI application entry point.
"""

from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.core.config import get_settings
from app.api import health_router, portfolio_router, optimization_router


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator:
    """
    Application lifespan handler.
    
    Handles startup and shutdown events.
    """
    # Startup
    settings = get_settings()
    print(f"Starting {settings.app_name}...")
    print(f"LLM Provider: {settings.llm_provider} ({settings.llm_model})")
    print(f"Data Source: {settings.data_source} ({settings.csv_data_path})")
    print(f"Solver: {settings.solver}")
    
    yield
    
    # Shutdown
    print("Shutting down...")


def create_app() -> FastAPI:
    """
    Create and configure the FastAPI application.
    
    Returns:
        Configured FastAPI application instance
    """
    settings = get_settings()
    
    app = FastAPI(
        title=settings.app_name,
        description="""
## Multi-Agent Equity Portfolio Manager

An AI-powered portfolio construction system using LangGraph multi-agent orchestration.

### Features

- **Multi-Agent Architecture**: 5 specialized agents (Data, Alpha, Risk, Optimization, Compliance)
- **Flexible LLM Integration**: Supports OpenAI, DeepSeek, and Anthropic
- **Mean-Variance Optimization**: Constrained portfolio construction
- **Constraint Management**: Stock (±1%) and sector (±2%) active weight limits

### Workflow

1. **Data Agent**: Load and validate market data
2. **Alpha Agent**: Select top securities from Q1 (top quintile)
3. **Risk Agent**: Calculate factor exposures and portfolio risk
4. **Optimization Agent**: Construct optimal portfolio
5. **Compliance Agent**: Validate constraints, retry if needed

### Getting Started

1. Configure `.env` with your LLM API key
2. Place data files in the configured data path
3. Call `POST /optimization/run` to execute the workflow
        """,
        version="1.0.0",
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
    )
    
    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Configure appropriately for production
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Include routers
    app.include_router(health_router)
    app.include_router(portfolio_router)
    app.include_router(optimization_router)
    
    return app


# Create application instance
app = create_app()


@app.get("/", tags=["Root"])
async def root():
    """
    Root endpoint.
    
    Returns welcome message and API information.
    """
    settings = get_settings()
    return {
        "message": f"Welcome to {settings.app_name}",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health",
        "endpoints": {
            "optimization": "/optimization/run",
            "portfolio": "/portfolio/benchmark",
            "alpha": "/portfolio/alpha/top",
        },
    }


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )

