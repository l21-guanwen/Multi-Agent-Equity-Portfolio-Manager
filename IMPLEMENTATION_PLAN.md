# Multi-Agent Equity Portfolio Manager - Implementation Plan

**Version:** 1.0  
**Created:** December 5, 2025  
**Status:** Ready for Implementation

---

## Table of Contents

1. [Overview](#overview)
2. [Key Design Decisions](#key-design-decisions)
3. [Project Structure](#project-structure)
4. [Abstraction Layers](#abstraction-layers)
5. [Configuration](#configuration)
6. [Implementation Phases](#implementation-phases)
7. [Dependencies](#dependencies)
8. [Environment Variables](#environment-variables)
9. [LLM Usage in Agents](#llm-usage-in-agents)
10. [LangGraph Flow](#langgraph-flow)
11. [Data Files Reference](#data-files-reference)
12. [Implementation Progress](#implementation-progress)

---

## Overview

This document defines the implementation plan for a **Multi-Agent Equity Portfolio Manager** built on:

- **FastAPI** - REST API framework
- **LangGraph** - Multi-agent orchestration
- **LLM Integration** - Flexible provider switching (GPT-4, DeepSeek, Claude)
- **cvxpy** - Portfolio optimization (with abstraction for other solvers)

### Core Specifications (from `Multi_Agent_Equity_PM_Specification.md`)

| Parameter | Value |
|-----------|-------|
| Benchmark | S&P 500 (500 securities) |
| Portfolio Holdings | 25 concentrated positions |
| Risk Factors | 8 (Barra-style) |
| Stock Constraint | ±1% active weight vs benchmark |
| Sector Constraint | ±2% active weight vs benchmark |
| Alpha Scoring | 0-1 (Quintile 1-5) |

### Agent Architecture

| Agent | Role | Outputs |
|-------|------|---------|
| Data Agent | Market data ingestion & validation | Clean benchmark/universe data |
| Alpha Agent | Generate/process stock-level alpha signals | Alpha scores (0-1), quintiles |
| Risk Agent | Factor decomposition & risk metrics | Factor loadings, covariance |
| Optimization Agent | Portfolio construction with constraints | Optimal weights |
| Compliance Agent | Constraint validation & monitoring | Compliance reports |

---

## Key Design Decisions

| Requirement | Design Approach |
|-------------|-----------------|
| **LLM Flexibility** | Abstract `ILLMProvider` interface with implementations for DeepSeek, GPT-4, Claude |
| **Data Source Flexibility** | Abstract repository interfaces with CSV (default) and DB implementations |
| **Future Auth Support** | Design middleware-ready architecture, no implementation now |
| **Future API Integration** | Repository interface allows adding API-based data sources later |
| **Solver Flexibility** | Abstract `ISolver` interface with cvxpy (default) and scipy implementations |

---

## Project Structure

Following `Linvest21_object_oriented_python_project_structure.md` conventions:

```
project_root/
├── main.py                                    # FastAPI app entry point
├── requirements.txt                           # Dependencies
├── .env.example                               # Environment variables template
├── IMPLEMENTATION_PLAN.md                     # This file
├── tests/testdata/                            # Test CSV files (9 files)
│
├── app/
│   ├── __init__.py
│   │
│   ├── api/                                   # API Routing Layer
│   │   ├── __init__.py
│   │   ├── portfolio_router.py                # Portfolio endpoints
│   │   ├── optimization_router.py             # Optimization trigger endpoints
│   │   └── health_router.py                   # Health check endpoints
│   │
│   ├── core/                                  # Configuration Layer
│   │   ├── __init__.py
│   │   ├── config.py                          # App settings (Pydantic BaseSettings)
│   │   ├── constants.py                       # GICS sectors, factors, etc.
│   │   └── dependencies.py                    # Dependency injection setup
│   │
│   ├── models/                                # Domain Models (Pydantic)
│   │   ├── __init__.py
│   │   ├── security.py                        # Security entity
│   │   ├── benchmark.py                       # Benchmark constituency
│   │   ├── portfolio.py                       # Portfolio holdings
│   │   ├── alpha.py                           # Alpha model scores
│   │   ├── risk.py                            # Factor loadings, covariance
│   │   ├── constraint.py                      # Optimization constraints
│   │   └── transaction_cost.py                # Transaction cost model
│   │
│   ├── schemas/                               # API Request/Response Schemas
│   │   ├── __init__.py
│   │   ├── portfolio_schema.py                # Portfolio API schemas
│   │   ├── optimization_schema.py             # Optimization request/response
│   │   └── agent_schema.py                    # Agent state schemas
│   │
│   ├── repositories/                          # Data Access Layer
│   │   ├── __init__.py
│   │   ├── interfaces/                        # Abstract Interfaces
│   │   │   ├── __init__.py
│   │   │   ├── base_repository.py             # Abstract base for all repos
│   │   │   ├── benchmark_repository.py        # Benchmark interface
│   │   │   ├── universe_repository.py         # Universe interface
│   │   │   ├── alpha_repository.py            # Alpha interface
│   │   │   ├── risk_repository.py             # Risk interface
│   │   │   ├── constraint_repository.py       # Constraint interface
│   │   │   └── transaction_cost_repository.py # Transaction cost interface
│   │   │
│   │   ├── csv/                               # CSV Implementations (DEFAULT)
│   │   │   ├── __init__.py
│   │   │   ├── csv_benchmark_repository.py
│   │   │   ├── csv_universe_repository.py
│   │   │   ├── csv_alpha_repository.py
│   │   │   ├── csv_risk_repository.py
│   │   │   ├── csv_constraint_repository.py
│   │   │   └── csv_transaction_cost_repository.py
│   │   │
│   │   └── db/                                # DB Implementations (FUTURE)
│   │       ├── __init__.py
│   │       └── .gitkeep
│   │
│   ├── services/                              # Business Logic Layer
│   │   ├── __init__.py
│   │   ├── interfaces/                        # Abstract Interfaces
│   │   │   ├── __init__.py
│   │   │   ├── data_service.py
│   │   │   ├── alpha_service.py
│   │   │   ├── risk_service.py
│   │   │   ├── optimization_service.py
│   │   │   └── compliance_service.py
│   │   │
│   │   ├── data_service.py                    # Data validation & aggregation
│   │   ├── alpha_service.py                   # Alpha score processing
│   │   ├── risk_service.py                    # Risk calculations
│   │   ├── optimization_service.py            # Portfolio optimization
│   │   └── compliance_service.py              # Constraint validation
│   │
│   ├── llm/                                   # LLM Provider Layer
│   │   ├── __init__.py
│   │   ├── interfaces/
│   │   │   ├── __init__.py
│   │   │   └── llm_provider.py                # Abstract LLM interface
│   │   │
│   │   ├── openai_provider.py                 # GPT-4 implementation
│   │   ├── deepseek_provider.py               # DeepSeek implementation
│   │   ├── anthropic_provider.py              # Claude implementation
│   │   └── factory.py                         # LLM factory for switching
│   │
│   ├── solvers/                               # Optimization Solver Layer
│   │   ├── __init__.py
│   │   ├── interfaces/
│   │   │   ├── __init__.py
│   │   │   └── solver.py                      # Abstract solver interface
│   │   │
│   │   ├── cvxpy_solver.py                    # cvxpy implementation (DEFAULT)
│   │   ├── scipy_solver.py                    # scipy.optimize alternative
│   │   └── factory.py                         # Solver factory for switching
│   │
│   ├── agents/                                # LangGraph Agents Layer
│   │   ├── __init__.py
│   │   ├── state.py                           # PortfolioState TypedDict
│   │   ├── graph.py                           # LangGraph StateGraph definition
│   │   ├── data_agent.py                      # Data ingestion agent node
│   │   ├── alpha_agent.py                     # Alpha generation agent node
│   │   ├── risk_agent.py                      # Risk analysis agent node
│   │   ├── optimization_agent.py              # Portfolio optimization agent node
│   │   └── compliance_agent.py                # Compliance validation agent node
│   │
│   └── utils/                                 # Utility Functions
│       ├── __init__.py
│       ├── csv_loader.py                      # CSV parsing utilities
│       └── math_utils.py                      # Matrix operations
│
└── tests/                                     # Testing Layer
    ├── __init__.py
    ├── conftest.py                            # Pytest fixtures
    ├── test_repositories/
    ├── test_services/
    ├── test_agents/
    ├── test_llm/
    └── test_api/
```

---

## Abstraction Layers

### 1. LLM Provider Interface

```python
# app/llm/interfaces/llm_provider.py
from abc import ABC, abstractmethod
from typing import Any
from pydantic import BaseModel

class LLMResponse(BaseModel):
    content: str
    model: str
    usage: dict[str, int]

class ILLMProvider(ABC):
    """Abstract interface for LLM providers - enables switching between models."""
    
    @abstractmethod
    async def generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 2000,
    ) -> LLMResponse:
        """Generate a response from the LLM."""
        pass
    
    @abstractmethod
    async def generate_structured(
        self,
        prompt: str,
        response_schema: type[BaseModel],
        system_prompt: str | None = None,
    ) -> BaseModel:
        """Generate a structured response matching a Pydantic schema."""
        pass
```

### 2. Repository Interface (Data Source Agnostic)

```python
# app/repositories/interfaces/base_repository.py
from abc import ABC, abstractmethod
from typing import Generic, TypeVar
from datetime import date

T = TypeVar('T')

class IBaseRepository(ABC, Generic[T]):
    """Abstract base repository - enables switching between CSV/DB/API sources."""
    
    @abstractmethod
    async def get_all(self, as_of_date: date | None = None) -> list[T]:
        """Retrieve all records."""
        pass
    
    @abstractmethod
    async def get_by_id(self, id: str) -> T | None:
        """Retrieve a record by ID."""
        pass
```

### 3. Solver Interface

```python
# app/solvers/interfaces/solver.py
from abc import ABC, abstractmethod
from typing import Any
import numpy as np
from pydantic import BaseModel

class OptimizationResult(BaseModel):
    weights: dict[str, float]
    objective_value: float
    status: str
    solver_name: str

class ISolver(ABC):
    """Abstract solver interface - enables switching between optimization solvers."""
    
    @abstractmethod
    def solve(
        self,
        alpha_vector: np.ndarray,
        covariance_matrix: np.ndarray,
        constraints: list[dict[str, Any]],
        bounds: tuple[float, float],
        risk_aversion: float,
    ) -> OptimizationResult:
        """Solve the portfolio optimization problem."""
        pass
```

### 4. Service Interfaces

```python
# app/services/interfaces/data_service.py
from abc import ABC, abstractmethod
from datetime import date

class IDataService(ABC):
    @abstractmethod
    async def load_and_validate_data(self, as_of_date: date) -> dict:
        """Load and validate all required data for portfolio construction."""
        pass

# app/services/interfaces/alpha_service.py
class IAlphaService(ABC):
    @abstractmethod
    async def get_top_quintile_securities(self, alpha_data: list, top_n: int = 25) -> list:
        """Filter and return top N securities from quintile 1."""
        pass

# app/services/interfaces/risk_service.py
class IRiskService(ABC):
    @abstractmethod
    async def calculate_portfolio_risk(self, weights: dict, factor_loadings: list, covariance: list) -> dict:
        """Calculate portfolio risk metrics."""
        pass

# app/services/interfaces/optimization_service.py
class IOptimizationService(ABC):
    @abstractmethod
    async def optimize_portfolio(self, alpha_scores: dict, risk_data: dict, constraints: list) -> dict:
        """Run portfolio optimization."""
        pass

# app/services/interfaces/compliance_service.py
class IComplianceService(ABC):
    @abstractmethod
    async def check_constraints(self, weights: dict, constraints: list, benchmark_weights: dict) -> dict:
        """Check portfolio weights against constraints."""
        pass
```

---

## Configuration

```python
# app/core/config.py
from pydantic_settings import BaseSettings
from typing import Literal

class Settings(BaseSettings):
    # Application
    app_name: str = "Multi-Agent Equity Portfolio Manager"
    debug: bool = False
    
    # LLM Configuration
    llm_provider: Literal["openai", "deepseek", "anthropic"] = "openai"
    openai_api_key: str = ""
    deepseek_api_key: str = ""
    anthropic_api_key: str = ""
    llm_model: str = "gpt-4"  # or "deepseek-chat", "claude-3-opus"
    llm_temperature: float = 0.7
    
    # Data Source Configuration
    data_source: Literal["csv", "database", "api"] = "csv"
    csv_data_path: str = "./testdata"
    database_url: str = ""  # For future DB support
    
    # Solver Configuration
    solver: Literal["cvxpy", "scipy"] = "cvxpy"
    
    # Optimization Parameters
    risk_aversion: float = 0.01
    transaction_cost_penalty: float = 0.0
    max_iterations: int = 5
    
    # Portfolio Parameters
    portfolio_size: int = 25
    stock_active_weight_limit: float = 0.01  # ±1%
    sector_active_weight_limit: float = 0.02  # ±2%
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
```

---

## Implementation Phases

### Phase 1: Foundation (Core & Models) ✅ COMPLETE

| # | Task | File(s) | Status |
|---|------|---------|--------|
| 1.1 | Project setup, dependencies | `requirements.txt` | ✅ Done |
| 1.2 | Environment template | `env.example` | ✅ Done |
| 1.3 | App configuration | `app/core/config.py` | ✅ Done |
| 1.4 | Constants (GICS, factors) | `app/core/constants.py` | ✅ Done |
| 1.5 | Security model | `app/models/security.py` | ✅ Done |
| 1.6 | Benchmark model | `app/models/benchmark.py` | ✅ Done |
| 1.7 | Portfolio model | `app/models/portfolio.py` | ✅ Done |
| 1.8 | Alpha model | `app/models/alpha.py` | ✅ Done |
| 1.9 | Risk model | `app/models/risk.py` | ✅ Done |
| 1.10 | Constraint model | `app/models/constraint.py` | ✅ Done |
| 1.11 | Transaction cost model | `app/models/transaction_cost.py` | ✅ Done |

### Phase 2: Abstraction Interfaces ✅ COMPLETE

| # | Task | File(s) | Status |
|---|------|---------|--------|
| 2.1 | LLM provider interface | `app/llm/interfaces/llm_provider.py` | ✅ Done |
| 2.2 | Base repository interface | `app/repositories/interfaces/base_repository.py` | ✅ Done |
| 2.3 | All repository interfaces | `app/repositories/interfaces/*.py` | ✅ Done |
| 2.4 | All service interfaces | `app/services/interfaces/*.py` | ✅ Done |
| 2.5 | Solver interface | `app/solvers/interfaces/solver.py` | ✅ Done |

### Phase 3: LLM Providers ✅ COMPLETE

| # | Task | File(s) | Status |
|---|------|---------|--------|
| 3.1 | OpenAI provider (GPT-4) | `app/llm/openai_provider.py` | ✅ Done |
| 3.2 | DeepSeek provider | `app/llm/deepseek_provider.py` | ✅ Done |
| 3.3 | Anthropic provider | `app/llm/anthropic_provider.py` | ✅ Done |
| 3.4 | LLM factory | `app/llm/factory.py` | ✅ Done |

### Phase 4: Data Layer (CSV Repositories) ✅ COMPLETE

| # | Task | File(s) | Status |
|---|------|---------|--------|
| 4.1 | CSV utilities | `app/utils/csv_loader.py` | ✅ Done |
| 4.2 | Benchmark repository | `app/repositories/csv/csv_benchmark_repository.py` | ✅ Done |
| 4.3 | Universe repository | `app/repositories/csv/csv_universe_repository.py` | ✅ Done |
| 4.4 | Alpha repository | `app/repositories/csv/csv_alpha_repository.py` | ✅ Done |
| 4.5 | Risk repository | `app/repositories/csv/csv_risk_repository.py` | ✅ Done |
| 4.6 | Constraint repository | `app/repositories/csv/csv_constraint_repository.py` | ✅ Done |
| 4.7 | Transaction cost repository | `app/repositories/csv/csv_transaction_cost_repository.py` | ✅ Done |

### Phase 5: Solvers ✅ COMPLETE

| # | Task | File(s) | Status |
|---|------|---------|--------|
| 5.1 | cvxpy solver (default) | `app/solvers/cvxpy_solver.py` | ✅ Done |
| 5.2 | scipy solver (alternative) | `app/solvers/scipy_solver.py` | ✅ Done |
| 5.3 | Solver factory | `app/solvers/factory.py` | ✅ Done |

### Phase 6: Business Services ✅ COMPLETE

| # | Task | File(s) | Status |
|---|------|---------|--------|
| 6.1 | Data service | `app/services/data_service.py` | ✅ Done |
| 6.2 | Alpha service | `app/services/alpha_service.py` | ✅ Done |
| 6.3 | Risk service | `app/services/risk_service.py` | ✅ Done |
| 6.4 | Optimization service | `app/services/optimization_service.py` | ✅ Done |
| 6.5 | Compliance service | `app/services/compliance_service.py` | ✅ Done |

### Phase 7: LangGraph Agents ✅ COMPLETE

| # | Task | File(s) | Status |
|---|------|---------|--------|
| 7.1 | Portfolio state definition | `app/agents/state.py` | ✅ Done |
| 7.2 | Data agent | `app/agents/data_agent.py` | ✅ Done |
| 7.3 | Alpha agent | `app/agents/alpha_agent.py` | ✅ Done |
| 7.4 | Risk agent | `app/agents/risk_agent.py` | ✅ Done |
| 7.5 | Optimization agent | `app/agents/optimization_agent.py` | ✅ Done |
| 7.6 | Compliance agent | `app/agents/compliance_agent.py` | ✅ Done |
| 7.7 | LangGraph StateGraph | `app/agents/graph.py` | ✅ Done |

### Phase 8: API Layer ✅ COMPLETE

| # | Task | File(s) | Status |
|---|------|---------|--------|
| 8.1 | Portfolio schemas | `app/schemas/portfolio_schema.py` | ✅ Done |
| 8.2 | Optimization schemas | `app/schemas/optimization_schema.py` | ✅ Done |
| 8.3 | Agent schemas | `app/schemas/agent_schema.py` | ✅ Done |
| 8.4 | Health router | `app/api/health_router.py` | ✅ Done |
| 8.5 | Portfolio router | `app/api/portfolio_router.py` | ✅ Done |
| 8.6 | Optimization router | `app/api/optimization_router.py` | ✅ Done |
| 8.7 | Dependency injection | `app/core/dependencies.py` | ✅ Done |
| 8.8 | Main application | `main.py` | ✅ Done |

### Phase 9: Testing

| # | Task | File(s) | Status |
|---|------|---------|--------|
| 9.1 | Pytest fixtures | `tests/conftest.py` | ✅ Complete |
| 9.2 | Repository tests | `tests/test_repositories/` | ✅ Complete |
| 9.3 | Service tests | `tests/test_services/` | ✅ Complete |
| 9.4 | Agent tests | `tests/test_agents/` | ✅ Complete |
| 9.5 | API tests | `tests/test_api/` | ✅ Complete |

---

## Dependencies

```
# requirements.txt

# FastAPI & Server
fastapi>=0.104.0
uvicorn[standard]>=0.24.0
python-multipart>=0.0.6

# Pydantic
pydantic>=2.5.0
pydantic-settings>=2.1.0

# LangGraph & LangChain
langgraph>=0.0.40
langchain>=0.1.0
langchain-openai>=0.0.5
langchain-community>=0.0.10

# LLM Providers
openai>=1.6.0
anthropic>=0.8.0
httpx>=0.25.0  # For DeepSeek API calls

# Data Processing
pandas>=2.1.0
numpy>=1.26.0

# Optimization
cvxpy>=1.4.0
scipy>=1.11.0

# Environment
python-dotenv>=1.0.0

# Testing
pytest>=7.4.0
pytest-asyncio>=0.21.0
pytest-cov>=4.1.0

# Future DB Support (uncomment when needed)
# sqlalchemy>=2.0.0
# asyncpg>=0.29.0
```

---

## Environment Variables

```bash
# .env.example

# ===========================================
# Application Settings
# ===========================================
DEBUG=false

# ===========================================
# LLM Provider Configuration
# ===========================================
# Provider: "openai" | "deepseek" | "anthropic"
LLM_PROVIDER=openai

# Model name (depends on provider)
# OpenAI: gpt-4, gpt-4-turbo, gpt-3.5-turbo
# DeepSeek: deepseek-chat, deepseek-coder
# Anthropic: claude-3-opus-20240229, claude-3-sonnet-20240229
LLM_MODEL=gpt-4

LLM_TEMPERATURE=0.7

# API Keys (only need the one for your selected provider)
OPENAI_API_KEY=sk-...
DEEPSEEK_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...

# ===========================================
# Data Source Configuration
# ===========================================
# Source: "csv" | "database" | "api"
DATA_SOURCE=csv
CSV_DATA_PATH=./testdata

# Future: Database URL (when DATA_SOURCE=database)
# DATABASE_URL=postgresql+asyncpg://user:pass@localhost/portfolio_db

# ===========================================
# Solver Configuration
# ===========================================
# Solver: "cvxpy" | "scipy"
SOLVER=cvxpy

# ===========================================
# Optimization Parameters
# ===========================================
RISK_AVERSION=0.01
TRANSACTION_COST_PENALTY=0.0
MAX_ITERATIONS=5

# ===========================================
# Portfolio Parameters
# ===========================================
PORTFOLIO_SIZE=25
STOCK_ACTIVE_WEIGHT_LIMIT=0.01
SECTOR_ACTIVE_WEIGHT_LIMIT=0.02
```

---

## LLM Usage in Agents

Each agent uses the LLM for intelligent decision-making and reasoning:

| Agent | LLM Responsibilities |
|-------|---------------------|
| **Data Agent** | Validate data quality, identify anomalies, explain data issues, suggest data fixes |
| **Alpha Agent** | Explain alpha signal selection rationale, justify quintile filtering, interpret signal strength |
| **Risk Agent** | Interpret factor exposures, explain portfolio risk characteristics, identify risk concentrations |
| **Optimization Agent** | Explain optimization trade-offs, justify weight allocations, interpret solver results |
| **Compliance Agent** | Analyze constraint violations, explain breach severity, suggest remediation strategies |

### Agent Prompt Templates (Examples)

```python
# Data Agent System Prompt
DATA_AGENT_SYSTEM_PROMPT = """
You are a Data Validation Agent for an equity portfolio management system.
Your role is to:
1. Validate the quality and completeness of market data
2. Identify any anomalies or missing values
3. Ensure data consistency across benchmark, universe, and holdings
4. Report any issues that could affect portfolio construction

Always provide structured analysis with specific findings.
"""

# Alpha Agent System Prompt  
ALPHA_AGENT_SYSTEM_PROMPT = """
You are an Alpha Analysis Agent for an equity portfolio management system.
Your role is to:
1. Analyze alpha scores and their distribution across the universe
2. Select top quintile securities for portfolio inclusion
3. Explain the rationale for security selection
4. Assess signal strength and confidence levels

Focus on idiosyncratic alpha opportunities.
"""

# Risk Agent System Prompt
RISK_AGENT_SYSTEM_PROMPT = """
You are a Risk Analysis Agent for an equity portfolio management system.
Your role is to:
1. Analyze factor exposures for selected securities
2. Calculate portfolio-level risk metrics
3. Identify risk concentrations and factor tilts
4. Assess diversification across the 8 style factors

Use the Barra-style factor model for analysis.
"""

# Optimization Agent System Prompt
OPTIMIZATION_AGENT_SYSTEM_PROMPT = """
You are a Portfolio Optimization Agent for an equity portfolio management system.
Your role is to:
1. Construct optimal portfolios maximizing risk-adjusted alpha
2. Apply stock-level (±1%) and sector-level (±2%) constraints
3. Balance alpha capture with risk management
4. Explain weight allocation decisions

Objective: Maximize α×Alpha - λ×Risk - τ×TransactionCosts
"""

# Compliance Agent System Prompt
COMPLIANCE_AGENT_SYSTEM_PROMPT = """
You are a Compliance Agent for an equity portfolio management system.
Your role is to:
1. Verify portfolio weights comply with all constraints
2. Check stock active weights are within ±1% vs benchmark
3. Check sector active weights are within ±2% vs benchmark
4. Report any violations and suggest remediation

Be thorough and flag all compliance issues.
"""
```

---

## LangGraph Flow

```
                         ┌─────────────┐
                         │   START     │
                         └──────┬──────┘
                                │
                                ▼
                         ┌─────────────┐
                         │ Data Agent  │  ← Load & validate CSV data
                         └──────┬──────┘
                                │
                         ┌──────┴──────┐
                         │ Data Valid? │
                         └──────┬──────┘
                           yes  │  no → ERROR
                                ▼
                         ┌─────────────┐
                         │ Alpha Agent │  ← Filter Q1, rank by alpha
                         └──────┬──────┘
                                │
                                ▼
                         ┌─────────────┐
                         │ Risk Agent  │  ← Calculate factor exposures
                         └──────┬──────┘
                                │
                                ▼
                    ┌──────────────────────┐
                    │  Optimization Agent  │  ← Solve for optimal weights
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
              ┌─────│  Compliance Agent    │
              │     └──────────┬───────────┘
              │                │
     violations?               │ compliant
     (retry < max)             │
              │                ▼
              │         ┌─────────────┐
              └────────►│    END      │  ← Return optimized portfolio
                        └─────────────┘
```

### LangGraph State Definition

```python
# app/agents/state.py
from typing import TypedDict, Optional, Annotated
from operator import add
import pandas as pd

class PortfolioState(TypedDict):
    """State passed between agents in the LangGraph workflow."""
    
    # Configuration
    as_of_date: str
    portfolio_id: str
    
    # Input Data (loaded by Data Agent)
    benchmark_data: Optional[pd.DataFrame]
    universe_data: Optional[pd.DataFrame]
    alpha_data: Optional[pd.DataFrame]
    factor_loadings: Optional[pd.DataFrame]
    factor_covariance: Optional[pd.DataFrame]
    constraints: Optional[pd.DataFrame]
    transaction_costs: Optional[pd.DataFrame]
    
    # Data Validation
    data_validation_passed: bool
    data_validation_issues: list[str]
    
    # Alpha Agent Outputs
    top_securities: list[str]  # Top 25 tickers from Q1
    alpha_analysis: str  # LLM explanation
    
    # Risk Agent Outputs
    portfolio_factor_exposures: Optional[dict]
    risk_analysis: str  # LLM explanation
    
    # Optimization Agent Outputs
    optimal_weights: Optional[dict[str, float]]
    optimization_status: str
    optimization_analysis: str  # LLM explanation
    
    # Compliance Agent Outputs
    is_compliant: bool
    compliance_violations: list[dict]
    compliance_analysis: str  # LLM explanation
    
    # Workflow Control
    iteration_count: int
    max_iterations: int
    error_message: Optional[str]
    
    # Final Output
    final_portfolio: Optional[pd.DataFrame]
    execution_log: Annotated[list[str], add]  # Accumulates log messages
```

---

## Data Files Reference

Test data files in `tests/testdata/` folder:

| # | File | Description | Records | Key Columns |
|---|------|-------------|---------|-------------|
| 1 | `01_SP500_Benchmark_Constituency.csv` | Benchmark holdings | 500 | Ticker, Benchmark_Weight_Pct, GICS_Sector |
| 2 | `02_SP500_Universe.csv` | Investible universe | 500 | Ticker, Is_Investible, Liquidity_Score |
| 3 | `03_Portfolio_25_Holdings.csv` | Current portfolio | 25 | Ticker, Shares, Portfolio_Weight_Pct |
| 4 | `04_Alpha_Model_SP500.csv` | Alpha scores | 500 | Ticker, Alpha_Score, Alpha_Quintile |
| 5 | `05_Risk_Model_Factor_Loadings.csv` | Factor loadings | 500 | Ticker, Market_Loading, Size_Loading, ... |
| 6 | `06_Risk_Model_Factor_Returns.csv` | Factor returns | 8 | Factor_Name, Factor_Return_YTD_Pct |
| 7 | `07_Risk_Model_Factor_Covariance.csv` | Covariance matrix | 8×8 | Factor, Market, Size, Value, ... |
| 8 | `08_Optimization_Constraints.csv` | Constraints | 36 | Constraint_Type, Lower_Bound_Pct, Upper_Bound_Pct |
| 9 | `09_Transaction_Cost_Model.csv` | Transaction costs | 500 | Ticker, Bid_Ask_Spread_Bps, Market_Impact_Bps |

---

## Implementation Progress

**Last Updated:** December 5, 2025

| Phase | Description | Status | Progress |
|-------|-------------|--------|----------|
| 1 | Foundation (Core & Models) | ✅ Complete | 11/11 |
| 2 | Abstraction Interfaces | ✅ Complete | 5/5 |
| 3 | LLM Providers | ✅ Complete | 4/4 |
| 4 | Data Layer (CSV) | ✅ Complete | 7/7 |
| 5 | Solvers | ✅ Complete | 3/3 |
| 6 | Business Services | ✅ Complete | 5/5 |
| 7 | LangGraph Agents | ✅ Complete | 7/7 |
| 8 | API Layer | ✅ Complete | 8/8 |
| 9 | Testing | ✅ Complete | 5/5 |

**Overall Progress:** 55/55 tasks completed ✅ ALL PHASES COMPLETE

---

## Notes for Continuation

If starting a new session, follow these steps:

1. **Read this file** to understand the full implementation plan
2. **Check Implementation Progress** section to see what's completed
3. **Follow the phase order** (Phase 1 → 2 → 3 → etc.)
4. **Update progress** after completing each task (change ⬜ to ✅)
5. **Follow coding conventions** in `Linvest21_object_oriented_python_project_structure.md`
6. **Reference test data** in `tests/testdata/` folder for schema validation
7. **Reference specification** in `Multi_Agent_Equity_PM_Specification.md` for business rules

---

*© 2025 Linvest21 | AlphaCopilot - Agentic CIO*

