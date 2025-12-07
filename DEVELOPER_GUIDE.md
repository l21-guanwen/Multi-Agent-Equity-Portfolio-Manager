# Developer Guide: Multi-Agent Equity Portfolio Manager

Technical documentation for developers working on or extending the Multi-Agent Equity Portfolio Manager.

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Agent System](#agent-system)
3. [Data Models](#data-models)
4. [Service Layer](#service-layer)
5. [Adding New Features](#adding-new-features)
6. [Testing](#testing)
7. [Code Structure](#code-structure)

---

## Architecture Overview

### System Components

```
┌─────────────────────────────────────────────────────────────┐
│                    FastAPI Application                      │
│  (main.py, app/api/*)                                       │
└───────────────────────┬─────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────┐
│              LangGraph Multi-Agent Orchestration            │
│  (app/agents/graph.py)                                      │
│                                                             │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐  │
│  │   Data   │→ │  Alpha   │→ │   Risk   │→ │Optimize  │  │
│  │  Agent   │  │  Agent   │  │  Agent   │  │  Agent   │  │
│  └──────────┘  └──────────┘  └──────────┘  └────┬─────┘  │
│                                                  │         │
│                                            ┌─────▼─────┐   │
│                                            │Compliance │   │
│                                            │  Agent    │   │
│                                            └─────┬─────┘   │
│                                                  │ (retry) │
└──────────────────────────────────────────────────┼─────────┘
                                                   │
                        ┌──────────────────────────┴──────────┐
                        │                                       │
                        ▼                                       ▼
            ┌──────────────────┐              ┌──────────────────┐
            │   Service Layer   │              │  Repository Layer│
            │  (Business Logic) │              │  (Data Access)   │
            └──────────────────┘              └──────────────────┘
                        │                               │
                        └───────────┬───────────────────┘
                                    ▼
                        ┌──────────────────┐
                        │   Data Sources   │
                        │  (CSV Files)    │
                        └──────────────────┘
```

### Key Technologies

- **LangGraph**: Multi-agent orchestration and state management
- **FastAPI**: REST API framework
- **Pydantic**: Data validation and models
- **CVXPY/SCIPY**: Mathematical optimization solvers
- **LangChain**: LLM integration and tool system

---

## Agent System

### Agent Types

#### 1. ReAct Agents (Data, Alpha, Risk)

**Pattern**: Reasoning + Acting
- LLM decides which tools to call
- Tools execute and return observations
- LLM reasons about results and decides next action

**Implementation**:
```python
from langgraph.prebuilt import create_react_agent as create_langgraph_react_agent
from langchain_core.messages import SystemMessage

react_agent = create_langgraph_react_agent(
    model=llm,
    tools=[load_benchmark, load_alpha_scores],
    prompt=SystemMessage(content=SYSTEM_PROMPT),
)
```

**Flow**:
```
LLM receives task
  ↓
LLM outputs: THOUGHT → ACTION → ACTION_INPUT
  ↓
Agent executes tool
  ↓
Tool returns OBSERVATION
  ↓
LLM processes observation, decides next action
  ↓
Repeat until FINISH
```

#### 2. Chain-of-Thought Agent (Optimization)

**Pattern**: Step-by-step reasoning
- LLM receives full problem description
- LLM reasons through optimization step-by-step
- LLM outputs final portfolio weights

**Implementation**:
```python
messages = [
    SystemMessage(content=COT_OPTIMIZATION_SYSTEM_PROMPT),
    HumanMessage(content=problem_prompt),
]
response = await llm.ainvoke(messages)
weights = parse_weights_from_response(response.content)
```

#### 3. Rule-Based Agent (Compliance)

**Pattern**: Deterministic validation
- No LLM required
- Validates constraints programmatically
- Returns compliance status and violations

### State Management

**PortfolioState** (`app/agents/state.py`):
- Shared state passed between agents
- Pydantic model for validation
- Reducers for list accumulation (`execution_log`)

**State Updates**:
```python
# Agent returns dict of updates
return {
    "selected_tickers": ["AAPL", "MSFT"],
    "alpha_analysis": "...",
    "execution_log": ["[AlphaAgent] Selected 2 securities"],
}

# LangGraph merges into state
state.selected_tickers = ["AAPL", "MSFT"]
state.execution_log.append("[AlphaAgent] Selected 2 securities")
```

### Graph Routing

**Conditional Edges**:
```python
graph.add_conditional_edges(
    "data_agent",
    self._route_after_data,
    {
        "continue": "alpha_agent",
        "error": END,
    }
)
```

**Routing Logic**:
```python
def _route_after_data(self, state: PortfolioState) -> Literal["continue", "error"]:
    if not state.data_validation_passed:
        return "error"
    return "continue"
```

---

## Data Models

### Core Models

#### FactorLoading (`app/models/risk.py`)
```python
class FactorLoading(BaseModel):
    ticker: str
    market_loading: float
    size_loading: float
    value_loading: float
    momentum_loading: float
    quality_loading: float
    volatility_loading: float
    growth_loading: float
    dividend_yield_loading: float
    specific_risk_pct: float
```

#### Constraint (`app/models/constraint.py`)
```python
class Constraint(BaseModel):
    constraint_type: str  # "Stock" or "Sector"
    constraint_name: str  # Ticker or sector name
    benchmark_weight_pct: float
    lower_bound_pct: float
    upper_bound_pct: float
    is_relative: bool
```

#### AlphaScore (`app/models/alpha.py`)
```python
class AlphaScore(BaseModel):
    ticker: str
    alpha_score: float  # 0.0 to 1.0
    alpha_quintile: int  # 1 (top) to 5 (bottom)
```

### Model Validation

Pydantic validators ensure data integrity:
```python
@field_validator('alpha_score')
@classmethod
def validate_alpha_score(cls, v: float) -> float:
    if not 0.0 <= v <= 1.0:
        raise ValueError("Alpha score must be between 0 and 1")
    return v
```

---

## Service Layer

### Service Interfaces

All services implement interfaces (`app/services/interfaces/`):

```python
class IOptimizationService(ABC):
    @abstractmethod
    async def optimize_portfolio(
        self,
        optimization_input: OptimizationInput,
        risk_model: RiskModel,
        constraint_set: ConstraintSet,
        parameters: OptimizationParameters,
        transaction_cost_model: Optional[TransactionCostModel] = None,
    ) -> OptimizationResult:
        """Optimize portfolio weights."""
        pass
```

### Service Responsibilities

| Service | Responsibility |
|---------|---------------|
| `DataService` | Load and validate all input data |
| `AlphaService` | Analyze alpha scores, select securities |
| `RiskService` | Calculate factor exposures, portfolio risk |
| `OptimizationService` | Solve optimization problem |
| `ComplianceService` | Validate constraints, check violations |

### Dependency Injection

Services are injected via FastAPI dependencies:

```python
# app/core/dependencies.py
def get_optimization_service() -> IOptimizationService:
    solver = get_solver()
    risk_service = get_risk_service()
    return OptimizationService(solver, risk_service)
```

---

## Adding New Features

### Adding a New Tool

1. **Create Tool** (`app/tools/langchain_tools.py`):
```python
@tool
async def load_custom_data(param: str) -> dict:
    """Load custom data source.
    
    Args:
        param: Parameter description
        
    Returns:
        Dictionary with data
    """
    # Implementation
    return {"data": "..."}
```

2. **Add to Agent** (`app/agents/data_agent.py`):
```python
self._tools = [
    load_benchmark,
    load_alpha_scores,
    load_custom_data,  # New tool
]
```

3. **Test**:
```python
# tests/test_tools/test_custom_tool.py
def test_load_custom_data():
    result = await load_custom_data.ainvoke({"param": "value"})
    assert "data" in result
```

### Replacing CSV with API Data Sources

The system currently loads data from CSV files, but you can easily replace this with API calls. Here's how to add API-based data sources for alpha and risk agents:

#### Option 1: Create API-Based Tools (Recommended for ReAct Agents)

**For Alpha Agent - API-Based Alpha Scores:**

1. **Create API Tool** (`app/tools/langchain_tools.py`):
```python
import httpx
from langchain_core.tools import tool

@tool
async def load_alpha_scores_from_api(as_of_date: Optional[str] = None) -> dict:
    """Load alpha scores from external API.
    
    This tool calls your alpha generation API to get latest scores.
    The LLM agent will choose this tool when it needs alpha data.
    
    Args:
        as_of_date: Optional date string (YYYY-MM-DD)
        
    Returns:
        Dictionary with alpha model data including:
        - security_count: Number of securities
        - scores: Dict of ticker -> alpha_score
        - quintiles: Dict of ticker -> quintile
    """
    from app.core.config import get_settings
    
    settings = get_settings()
    api_url = settings.alpha_api_url  # Configure in .env
    
    async with httpx.AsyncClient() as client:
        params = {}
        if as_of_date:
            params["as_of_date"] = as_of_date
            
        response = await client.get(
            f"{api_url}/alpha/scores",
            params=params,
            headers={"Authorization": f"Bearer {settings.alpha_api_key}"},
            timeout=30.0
        )
        response.raise_for_status()
        data = response.json()
    
    # Transform API response to expected format
    return {
        "security_count": len(data["scores"]),
        "scores": {s["ticker"]: s["alpha_score"] for s in data["scores"]},
        "quintiles": {s["ticker"]: s["quintile"] for s in data["scores"]},
        "top_alpha_securities": sorted(
            data["scores"], 
            key=lambda x: x["alpha_score"], 
            reverse=True
        )[:20],
    }
```

2. **Update Alpha Agent** (`app/agents/alpha_agent.py`):
```python
from app.tools.langchain_tools import (
    load_alpha_scores,  # Original CSV tool
    load_alpha_scores_from_api,  # New API tool
    load_benchmark,
)

class AlphaAgent:
    def __init__(self, ...):
        # Agent can choose between CSV or API tool
        self._tools = [
            load_alpha_scores_from_api,  # Prefer API
            load_alpha_scores,  # Fallback to CSV
            load_benchmark,
        ]
```

**For Risk Agent - API-Based Risk Model:**

1. **Create API Tool** (`app/tools/langchain_tools.py`):
```python
@tool
async def load_risk_model_from_api(as_of_date: Optional[str] = None) -> dict:
    """Load risk model from external API.
    
    Calls your risk model API to get factor loadings and covariance.
    
    Args:
        as_of_date: Optional date string (YYYY-MM-DD)
        
    Returns:
        Dictionary with risk model data
    """
    from app.core.config import get_settings
    
    settings = get_settings()
    
    async with httpx.AsyncClient() as client:
        response = await client.get(
            f"{settings.risk_api_url}/risk-model",
            params={"as_of_date": as_of_date} if as_of_date else {},
            headers={"Authorization": f"Bearer {settings.risk_api_key}"},
            timeout=30.0
        )
        response.raise_for_status()
        data = response.json()
    
    return {
        "security_count": data["security_count"],
        "factors": data["factors"],
        "has_covariance": "covariance_matrix" in data,
        "sample_loadings": data["loadings"][:5],  # First 5 for preview
    }
```

2. **Update Risk Agent** (`app/agents/risk_agent.py`):
```python
from app.tools.langchain_tools import (
    load_risk_model,  # Original CSV tool
    load_risk_model_from_api,  # New API tool
)

class RiskAgent:
    def __init__(self, ...):
        self._tools = [
            load_risk_model_from_api,  # Prefer API
            load_risk_model,  # Fallback to CSV
        ]
```

#### Option 2: Create API-Based Repository (Recommended for Direct Access)

If you want to replace CSV repositories entirely with API-based ones:

1. **Create API Repository** (`app/repositories/api_repositories/alpha_api_repository.py`):
```python
import httpx
from app.models.alpha import AlphaModel, AlphaScore
from app.repositories.interfaces.alpha_repository import IAlphaRepository
from app.core.config import get_settings

class AlphaAPIRepository(IAlphaRepository):
    """Repository that loads alpha data from external API."""
    
    def __init__(self):
        self._settings = get_settings()
        self._base_url = self._settings.alpha_api_url
        self._api_key = self._settings.alpha_api_key
    
    async def get_alpha_model(self, as_of_date: Optional[date] = None) -> Optional[AlphaModel]:
        """Load alpha model from API."""
        async with httpx.AsyncClient() as client:
            params = {}
            if as_of_date:
                params["as_of_date"] = as_of_date.isoformat()
            
            response = await client.get(
                f"{self._base_url}/alpha/model",
                params=params,
                headers={"Authorization": f"Bearer {self._api_key}"},
                timeout=30.0
            )
            response.raise_for_status()
            data = response.json()
        
        # Transform API response to AlphaModel
        scores = [
            AlphaScore(
                ticker=s["ticker"],
                security_name=s.get("security_name", s["ticker"]),
                alpha_score=s["alpha_score"],
                alpha_quintile=s["quintile"],
                gics_sector=s.get("sector", "Unknown"),
            )
            for s in data["scores"]
        ]
        
        return AlphaModel(
            model_id=data.get("model_id", "api_alpha_model"),
            as_of_date=date.fromisoformat(data["as_of_date"]),
            scores=scores,
        )
```

2. **Create API Repository for Risk** (`app/repositories/api_repositories/risk_api_repository.py`):
```python
import httpx
import numpy as np
from app.models.risk import RiskModel, FactorLoading
from app.repositories.interfaces.risk_repository import IRiskRepository
from app.core.config import get_settings

class RiskAPIRepository(IRiskRepository):
    """Repository that loads risk model from external API."""
    
    def __init__(self):
        self._settings = get_settings()
        self._base_url = self._settings.risk_api_url
        self._api_key = self._settings.risk_api_key
    
    async def get_risk_model(self, as_of_date: Optional[date] = None) -> Optional[RiskModel]:
        """Load risk model from API."""
        async with httpx.AsyncClient() as client:
            params = {}
            if as_of_date:
                params["as_of_date"] = as_of_date.isoformat()
            
            response = await client.get(
                f"{self._base_url}/risk-model",
                headers={"Authorization": f"Bearer {self._api_key}"},
                params=params,
                timeout=60.0  # Risk model may be larger
            )
            response.raise_for_status()
            data = response.json()
        
        # Transform API response to RiskModel
        factor_loadings = [
            FactorLoading(
                ticker=loading["ticker"],
                security_name=loading.get("security_name", loading["ticker"]),
                gics_sector=loading.get("sector", "Unknown"),
                market_loading=loading["market"],
                size_loading=loading["size"],
                value_loading=loading["value"],
                momentum_loading=loading["momentum"],
                quality_loading=loading["quality"],
                volatility_loading=loading["volatility"],
                growth_loading=loading["growth"],
                dividend_yield_loading=loading["dividend_yield"],
                specific_risk_pct=loading["specific_risk"],
            )
            for loading in data["factor_loadings"]
        ]
        
        # Parse covariance matrix if provided
        factor_covariance = None
        if "covariance_matrix" in data:
            factor_covariance = np.array(data["covariance_matrix"])
        
        return RiskModel(
            model_id=data.get("model_id", "api_risk_model"),
            as_of_date=date.fromisoformat(data["as_of_date"]),
            factor_loadings=factor_loadings,
            factor_covariance=factor_covariance,
        )
```

3. **Update Dependencies** (`app/core/dependencies.py`):
```python
from app.core.config import get_settings
from app.repositories.api_repositories.alpha_api_repository import AlphaAPIRepository
from app.repositories.api_repositories.risk_api_repository import RiskAPIRepository
from app.repositories.csv_repositories.alpha_csv_repository import AlphaCSVRepository
from app.repositories.csv_repositories.risk_csv_repository import RiskCSVRepository

def get_alpha_repository() -> IAlphaRepository:
    """Get alpha repository (API or CSV based on config)."""
    settings = get_settings()
    
    if settings.use_alpha_api:
        return AlphaAPIRepository()
    else:
        return AlphaCSVRepository()

def get_risk_repository() -> IRiskRepository:
    """Get risk repository (API or CSV based on config)."""
    settings = get_settings()
    
    if settings.use_risk_api:
        return RiskAPIRepository()
    else:
        return RiskCSVRepository()
```

4. **Add Configuration** (`app/core/config.py`):
```python
class Settings(BaseSettings):
    # ... existing settings ...
    
    # API Configuration
    use_alpha_api: bool = Field(default=False, description="Use API for alpha data")
    alpha_api_url: Optional[str] = Field(default=None, description="Alpha API base URL")
    alpha_api_key: Optional[str] = Field(default=None, description="Alpha API key")
    
    use_risk_api: bool = Field(default=False, description="Use API for risk data")
    risk_api_url: Optional[str] = Field(default=None, description="Risk API base URL")
    risk_api_key: Optional[str] = Field(default=None, description="Risk API key")
```

5. **Update Environment** (`.env`):
```bash
# Enable API-based data sources
USE_ALPHA_API=true
ALPHA_API_URL=https://api.example.com/v1
ALPHA_API_KEY=your-api-key-here

USE_RISK_API=true
RISK_API_URL=https://api.example.com/v1
RISK_API_KEY=your-api-key-here
```

#### Option 3: Hybrid Approach (API with CSV Fallback)

You can implement a repository that tries API first, falls back to CSV:

```python
class HybridAlphaRepository(IAlphaRepository):
    """Repository that tries API first, falls back to CSV."""
    
    def __init__(self):
        self._api_repo = AlphaAPIRepository()
        self._csv_repo = AlphaCSVRepository()
    
    async def get_alpha_model(self, as_of_date: Optional[date] = None) -> Optional[AlphaModel]:
        """Try API first, fall back to CSV."""
        try:
            return await self._api_repo.get_alpha_model(as_of_date)
        except Exception as e:
            logger.warning(f"API failed, falling back to CSV: {e}")
            return await self._csv_repo.get_alpha_model(as_of_date)
```

### Best Practices for API Integration

1. **Error Handling**: Always handle API failures gracefully
2. **Caching**: Consider caching API responses to reduce calls
3. **Rate Limiting**: Respect API rate limits
4. **Timeout**: Set appropriate timeouts (30-60 seconds)
5. **Retries**: Implement retry logic for transient failures
6. **Logging**: Log API calls for debugging
7. **Testing**: Mock API calls in tests

### Example: Complete API Integration

See `examples/api_integration_example.py` for a complete example of:
- Setting up API repositories
- Handling errors
- Caching responses
- Testing with mocks

### Adding a New Agent

1. **Create Agent Class** (`app/agents/new_agent.py`):
```python
class NewAgent:
    async def __call__(self, state: PortfolioState) -> dict[str, Any]:
        # Agent logic
        return {
            "new_field": value,
            "execution_log": ["[NewAgent] Completed"],
        }
```

2. **Add to Graph** (`app/agents/graph.py`):
```python
self._new_agent = NewAgent(...)
graph.add_node("new_agent", self._new_agent)
graph.add_edge("risk_agent", "new_agent")
```

3. **Update State** (`app/agents/state.py`):
```python
class PortfolioState(BaseModel):
    new_field: str = Field(default="", description="...")
```

### Adding a New LLM Provider

1. **Create Provider** (`app/llm/new_provider.py`):
```python
class NewLLMProvider(ILLMProvider):
    def __init__(self, api_key: str, model: str):
        self._client = NewClient(api_key)
        self._model = model
    
    @property
    def langchain_model(self) -> BaseChatModel:
        return NewLangChainModel(...)
    
    async def generate(self, prompt: str, **kwargs) -> LLMResponse:
        # Implementation
        pass
```

2. **Register in Factory** (`app/llm/factory.py`):
```python
def create_llm_provider(settings: Settings) -> ILLMProvider:
    if settings.llm_provider == "new":
        return NewLLMProvider(settings.new_api_key, settings.llm_model)
    # ...
```

3. **Add Config** (`app/core/config.py`):
```python
class Settings(BaseSettings):
    new_api_key: Optional[str] = None
```

---

## Testing

### Test Structure

```
tests/
├── conftest.py              # Shared fixtures
├── test_agents/             # Agent tests
│   ├── test_data_agent.py
│   ├── test_alpha_agent.py
│   └── test_cot_optimization_agent.py
├── test_services/           # Service tests
├── test_repositories/       # Repository tests
├── test_api/                # API endpoint tests
└── test_tools/              # Tool tests
```

### Running Tests

```bash
# All tests
pytest tests/ -v

# Specific test file
pytest tests/test_agents/test_data_agent.py -v

# With coverage
pytest tests/ --cov=app --cov-report=html

# Parallel execution
pytest tests/ -n auto
```

### Writing Tests

**Example: Agent Test**
```python
@pytest.mark.asyncio
async def test_agent_execution():
    # Arrange
    agent = DataAgent(mock_service, llm_provider=None)
    state = PortfolioState(use_llm=False)
    
    # Act
    result = await agent(state)
    
    # Assert
    assert "data_validation_passed" in result
    assert result["current_agent"] == "data_agent"
```

**Example: Service Test**
```python
def test_optimization_service():
    # Arrange
    solver = MockSolver()
    service = OptimizationService(solver, mock_risk_service)
    
    # Act
    result = await service.optimize_portfolio(...)
    
    # Assert
    assert result.status == "optimal"
    assert len(result.weights) > 0
```

### Mocking

**Mock LLM Provider**:
```python
mock_llm = MagicMock(spec=ILLMProvider)
mock_llm.generate = AsyncMock(return_value=LLMResponse(content="..."))
```

**Mock Repository**:
```python
mock_repo = MagicMock(spec=IRiskRepository)
mock_repo.get_risk_model = AsyncMock(return_value=mock_risk_model)
```

---

## Code Structure

### Directory Layout

```
app/
├── agents/              # LangGraph agents
│   ├── __init__.py
│   ├── data_agent.py   # ReAct agent
│   ├── alpha_agent.py  # ReAct agent
│   ├── risk_agent.py   # ReAct agent
│   ├── cot_optimization_agent.py  # CoT agent
│   ├── compliance_agent.py  # Rule-based
│   ├── graph.py        # LangGraph orchestration
│   ├── state.py        # PortfolioState definition
│   └── prompts.py      # System prompts
│
├── api/                # FastAPI routers
│   ├── health_router.py
│   ├── portfolio_router.py
│   └── optimization_router.py
│
├── core/               # Configuration & dependencies
│   ├── config.py       # Settings management
│   └── dependencies.py  # Dependency injection
│
├── llm/                # LLM providers
│   ├── factory.py      # Provider factory
│   ├── interfaces/
│   ├── openai_provider.py
│   ├── deepseek_provider.py
│   └── anthropic_provider.py
│
├── models/             # Pydantic domain models
│   ├── alpha.py
│   ├── benchmark.py
│   ├── constraint.py
│   ├── risk.py
│   └── transaction_cost.py
│
├── repositories/       # Data access layer
│   ├── interfaces/
│   └── csv_repositories/
│
├── services/           # Business logic
│   ├── interfaces/
│   └── implementations/
│
├── solvers/            # Optimization solvers
│   ├── interfaces/
│   ├── cvxpy_solver.py
│   └── scipy_solver.py
│
└── tools/              # LangChain tools
    ├── base.py
    ├── data_tools.py   # Class-based tools
    └── langchain_tools.py  # @tool decorator tools
```

### Key Design Patterns

#### 1. Dependency Injection
- Services injected via FastAPI dependencies
- Enables easy testing and mocking
- Centralized in `app/core/dependencies.py`

#### 2. Interface Segregation
- All services/repositories have interfaces
- Enables swapping implementations
- Defined in `app/*/interfaces/`

#### 3. Repository Pattern
- Data access abstracted behind repositories
- CSV implementation can be swapped for API/DB
- Defined in `app/repositories/`

#### 4. Strategy Pattern
- Multiple solvers (CVXPY, SCIPY)
- Multiple LLM providers (OpenAI, DeepSeek, Anthropic)
- Selected via configuration

---

## Common Development Tasks

### Debugging Agent Execution

**Enable Debug Logging**:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

**Check Execution Log**:
```python
result = await graph.run(...)
for log in result.execution_log:
    print(log)
```

**Inspect State**:
```python
state = PortfolioState(...)
print(state.model_dump_json(indent=2))
```

### Performance Optimization

**Profile Code**:
```python
import cProfile
cProfile.run('await graph.run(...)')
```

**Cache LLM Calls**:
```python
# In development, cache responses
from functools import lru_cache
@lru_cache(maxsize=100)
def cached_llm_call(prompt: str):
    return llm.generate(prompt)
```

### Error Handling

**Agent Error Handling**:
```python
async def __call__(self, state: PortfolioState) -> dict[str, Any]:
    try:
        # Agent logic
        return {"success": True, ...}
    except Exception as e:
        return {
            "error_message": str(e),
            "execution_log": [f"[Agent] ERROR: {e}"],
        }
```

**Graph Error Handling**:
```python
# Errors propagate via state.error_message
if state.error_message:
    return "error"  # Route to END
```

---

## Best Practices

### Code Style
- Follow PEP 8
- Use type hints
- Document public APIs
- Keep functions focused (single responsibility)

### Testing
- Write tests for new features
- Aim for >80% coverage
- Test edge cases
- Mock external dependencies

### Error Handling
- Use specific exceptions
- Provide helpful error messages
- Log errors appropriately
- Don't swallow exceptions silently

### Performance
- Profile before optimizing
- Cache expensive operations
- Use async/await properly
- Avoid blocking I/O in agents

---

## Contributing

### Development Workflow

1. **Create Feature Branch**:
   ```bash
   git checkout -b feature/new-feature
   ```

2. **Make Changes**:
   - Write code
   - Add tests
   - Update documentation

3. **Run Tests**:
   ```bash
   pytest tests/ -v
   ```

4. **Check Linting**:
   ```bash
   ruff check app/
   ```

5. **Submit PR**:
   - Include description
   - Reference issues
   - Ensure CI passes

---

*For user-facing documentation, see [USER_GUIDE.md](USER_GUIDE.md)*

