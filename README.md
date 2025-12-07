# Multi-Agent Equity Portfolio Manager

An AI-powered, institutional-grade portfolio construction system using LangGraph multi-agent orchestration. The system implements a concentrated long-only equity strategy benchmarked against the S&P 500.

## 🎯 Overview

This system uses five specialized AI agents orchestrated by LangGraph to:
1. **Load and validate** market data (benchmark, universe, risk model)
2. **Analyze alpha signals** and select top securities
3. **Calculate risk exposures** using a multi-factor model
4. **Optimize portfolio weights** with constraints
5. **Ensure compliance** with investment guidelines

## ✨ Features

- **ReAct Agent Architecture**: Agents use LLM to reason about and select tools (Reasoning + Acting)
- **Multi-Agent Orchestration**: 5 specialized agents orchestrated by LangGraph
- **Tool-Based Data Loading**: Extensible tool system for data sources (CSV now, APIs supported - see [Developer Guide](DEVELOPER_GUIDE.md#replacing-csv-with-api-data-sources))
- **LLM-Driven Optimization**: LLM can directly determine portfolio weights based on objectives
- **Flexible LLM Integration**: Supports OpenAI (GPT-4), DeepSeek, and Anthropic (Claude)
- **Fallback Mode**: When `use_llm=False`, uses mathematical optimization (CVXPY)
- **Barra-Style Risk Model**: 8-factor model with idiosyncratic risk
- **Constraint Management**: Single stock (±1%) and sector (±2%) active weight limits
- **Customizable Data**: Users can provide their own benchmark, universe, alpha scores, and constraints

## 🏗️ Architecture

### Agent Workflow
```
┌─────────────┐
│ Data Agent  │  ← Load benchmark, universe, alpha, risk model
└──────┬──────┘
       │
       ▼
┌─────────────┐
│ Alpha Agent │  ← Select top 25 securities from Q1 (highest alpha)
└──────┬──────┘
       │
       ▼
┌─────────────┐
│ Risk Agent  │  ← Calculate factor exposures & portfolio risk
└──────┬──────┘
       │
       ▼
┌──────────────────┐
│ Optimization     │  ← Solve for optimal weights (max alpha, min risk)
│ Agent            │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ Compliance Agent │  ← Validate constraints, retry if violations
└────────┬─────────┘
         │ (retry if not compliant)
         ▼
    [Final Portfolio]
```

### ReAct Agent Pattern

Each agent follows the **ReAct (Reasoning + Acting)** pattern:

```
┌─────────────────────────────────────────────────────────────┐
│                     ReAct Agent Loop                        │
├─────────────────────────────────────────────────────────────┤
│ 1. LLM receives task + available tools                      │
│ 2. LLM outputs: THOUGHT → ACTION → ACTION_INPUT             │
│ 3. Agent executes the chosen tool                           │
│ 4. Tool returns OBSERVATION                                 │
│ 5. Repeat until LLM outputs ACTION: FINISH                  │
└─────────────────────────────────────────────────────────────┘
```

### Available Tools

| Tool | Description |
|------|-------------|
| `load_benchmark` | Load S&P 500 benchmark constituents and weights |
| `load_alpha_scores` | Load AI-generated alpha scores and quintiles |
| `load_risk_model` | Load Barra-style factor loadings and covariance |
| `load_constraints` | Load stock (±1%) and sector (±2%) constraints |
| `load_transaction_costs` | Load trading cost estimates |

Tools are extensible - add new tools (e.g., API-based data) by implementing `BaseTool`. See [Developer Guide](DEVELOPER_GUIDE.md#replacing-csv-with-api-data-sources) for examples of integrating external APIs for alpha and risk data.

## 📦 Installation

### Prerequisites
- Python 3.12+
- [uv](https://github.com/astral-sh/uv) (recommended) or pip

### Setup

```bash
# Clone the repository
git clone https://github.com/your-org/Multi-Agent-Equity-Portfolio-Manager.git
cd Multi-Agent-Equity-Portfolio-Manager

# Using uv (recommended)
uv sync

# Copy environment file and configure
cp env.example .env
# Edit .env with your API keys and settings
```

## 🐳 Docker

The easiest way to run the server is using Docker.

### Quick Start with Docker

```bash
# Build and run
docker compose up -d

# Check logs
docker compose logs -f

# Stop
docker compose down
```

### Using Docker with Custom Configuration

```bash
# Run with environment variables
docker compose up -d \
  -e LLM_PROVIDER=deepseek \
  -e DEEPSEEK_API_KEY=your-key-here \
  -e PORTFOLIO_SIZE=30

# Or create a .env file first, then run
cp env.example .env
# Edit .env with your settings
docker compose up -d
```

### Development Mode (Hot Reload)

```bash
# Start with hot reload (code changes reflect immediately)
docker compose --profile dev up portfolio-manager-dev
```

### Docker Configuration Options

| Environment Variable | Description | Default |
|---------------------|-------------|---------|
| `HOST_PORT` | Port to expose on host | 8000 |
| `LLM_PROVIDER` | LLM provider (openai/deepseek/anthropic) | openai |
| `LLM_MODEL` | Model name | gpt-4o |
| `PORTFOLIO_SIZE` | Number of holdings | 25 |
| `SOLVER` | Optimization solver | cvxpy |
| `RISK_AVERSION` | Risk penalty (higher = more conservative) | 0.01 |

### Custom Data with Docker

Mount your own data folder:

```bash
# Use a custom data directory
docker run -d \
  -p 8000:8000 \
  -v /path/to/your/data:/app/data:ro \
  -e DEEPSEEK_API_KEY=your-key \
  equity-portfolio-manager
```

### Build Manually

```bash
# Build the image
docker build -t equity-portfolio-manager .

# Run the container
docker run -d \
  --name portfolio-manager \
  -p 8000:8000 \
  -e LLM_PROVIDER=deepseek \
  -e DEEPSEEK_API_KEY=your-key-here \
  -v $(pwd)/data:/app/data:ro \
  equity-portfolio-manager

# Access the API
curl http://localhost:8000/health
curl http://localhost:8000/docs  # Interactive API documentation
```

## ⚙️ Configuration

### Environment Variables (`.env`)

```bash
# LLM Provider: "openai" | "deepseek" | "anthropic"
LLM_PROVIDER=deepseek
LLM_MODEL=deepseek-chat

# API Keys (only need the one for your selected provider)
OPENAI_API_KEY=sk-...
DEEPSEEK_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...

# Data source path
CSV_DATA_PATH=./data

# Optimization solver: "cvxpy" | "scipy"
SOLVER=cvxpy

# Optimization parameters
RISK_AVERSION=0.01              # Higher = more conservative
TRANSACTION_COST_PENALTY=0.0    # Penalty for trading
MAX_ITERATIONS=5                # Max compliance retry loops

# Portfolio parameters
PORTFOLIO_SIZE=25               # Number of holdings
STOCK_ACTIVE_WEIGHT_LIMIT=0.01  # ±1% vs benchmark per stock
SECTOR_ACTIVE_WEIGHT_LIMIT=0.02 # ±2% vs benchmark per sector
```

### Key Configuration Options

| Parameter | Description | Default |
|-----------|-------------|---------|
| `PORTFOLIO_SIZE` | Number of securities in the portfolio | 25 |
| `RISK_AVERSION` | Risk penalty in optimization (higher = less risk) | 0.01 |
| `STOCK_ACTIVE_WEIGHT_LIMIT` | Max active weight per stock vs benchmark | ±1% |
| `SECTOR_ACTIVE_WEIGHT_LIMIT` | Max active weight per sector vs benchmark | ±2% |
| `SOLVER` | Optimization solver (`cvxpy` or `scipy`) | cvxpy |
| `LLM_PROVIDER` | LLM for agent reasoning | openai |

## 📊 Data Files

The system reads input data from CSV files in the `data/` folder. **You can customize these files to change the investment universe, benchmark, alpha model, or constraints.** 

**Future**: The system supports API-based data sources - see [Developer Guide](DEVELOPER_GUIDE.md#replacing-csv-with-api-data-sources) for integrating external APIs for alpha scores and risk models.

### Required Files

| File | Description | Key Columns |
|------|-------------|-------------|
| `01_SP500_Benchmark_Constituency.csv` | Benchmark constituents & weights | `Ticker`, `Benchmark_Weight_Pct`, `GICS_Sector` |
| `02_SP500_Universe.csv` | Investable universe | `Ticker`, `Is_Investible`, `Liquidity_Score` |
| `03_Portfolio_25_Holdings.csv` | Current portfolio (optional) | `Ticker`, `Shares`, `Portfolio_Weight_Pct` |
| `04_Alpha_Model_SP500.csv` | Alpha scores (0-1) | `Ticker`, `Alpha_Score`, `Alpha_Quintile` |
| `05_Risk_Model_Factor_Loadings.csv` | Factor exposures | `Ticker`, `Momentum`, `Value`, `Size`, ... |
| `06_Risk_Model_Factor_Returns.csv` | Factor returns | `Factor_Name`, `Factor_Return_YTD_Pct` |
| `07_Risk_Model_Factor_Covariance.csv` | Factor covariance matrix | 8×8 matrix |
| `08_Optimization_Constraints.csv` | Investment constraints | `Constraint_Type`, `Min_Value`, `Max_Value` |
| `09_Transaction_Cost_Model.csv` | Transaction costs | `Ticker`, `Bid_Ask_Spread_Bps`, `Market_Impact_Bps` |

### Customizing Data

All 9 CSV files in the `data/` folder can be replaced with your own data. Below are examples for each:

---

#### 1. Benchmark (`01_SP500_Benchmark_Constituency.csv`)
Define your benchmark index (S&P 500, Russell 1000, custom):
```csv
Ticker,Security_Name,GICS_Sector,GICS_Industry,Benchmark_Weight_Pct,Price,As_Of_Date
AAPL,Apple Inc.,Information Technology,Technology Hardware,6.5,180.00,2025-01-15
MSFT,Microsoft,Information Technology,Software,5.8,400.00,2025-01-15
JPM,JPMorgan Chase,Financials,Banks,2.1,180.00,2025-01-15
```

---

#### 2. Universe (`02_SP500_Universe.csv`)
Define which securities are eligible for investment:
```csv
Ticker,Security_Name,GICS_Sector,Is_Investible,Liquidity_Score,Market_Cap_USD_B
AAPL,Apple Inc.,Information Technology,TRUE,0.95,3000
MSFT,Microsoft,Information Technology,TRUE,0.92,2800
PENNY,Penny Stock Co,Financials,FALSE,0.10,0.5
```
- Set `Is_Investible=FALSE` to exclude securities from selection

---

#### 3. Current Holdings (`03_Portfolio_25_Holdings.csv`)
Your existing portfolio (used for transaction cost calculation):
```csv
Ticker,Security_Name,Shares,Price,Market_Value,Portfolio_Weight_Pct
AAPL,Apple Inc.,1000,180.00,180000,7.2
MSFT,Microsoft,500,400.00,200000,8.0
```

---

#### 4. Alpha Model (`04_Alpha_Model_SP500.csv`)
Your alpha signals/scores:
```csv
Ticker,Security_Name,Alpha_Score,Alpha_Quintile,Signal_Date
AAPL,Apple Inc.,0.85,1,2025-01-15
MSFT,Microsoft,0.72,2,2025-01-15
XOM,Exxon Mobil,0.35,4,2025-01-15
```
- `Alpha_Score`: 0.0 (bearish) to 1.0 (bullish)
- `Alpha_Quintile`: 1 (top 20%) to 5 (bottom 20%)

---

#### 5. Factor Loadings (`05_Risk_Model_Factor_Loadings.csv`)
Factor exposures for each security (Barra-style):
```csv
Ticker,Momentum,Value,Size,Volatility,Quality,Growth,Liquidity,Dividend_Yield,Idiosyncratic_Risk
AAPL,0.8,-0.3,-0.5,-0.2,0.6,0.7,-0.4,-0.1,0.15
MSFT,0.6,-0.2,-0.6,-0.3,0.7,0.5,-0.3,0.1,0.12
```

---

#### 6. Factor Returns (`06_Risk_Model_Factor_Returns.csv`)
Historical or expected returns for each factor:
```csv
Factor_Name,Factor_Return_Pct,Factor_Return_YTD_Pct
Momentum,0.5,2.1
Value,0.3,1.5
Size,-0.2,-0.8
Volatility,-0.4,-1.2
Quality,0.4,1.8
```

---

#### 7. Factor Covariance (`07_Risk_Model_Factor_Covariance.csv`)
Covariance matrix between factors (8×8):
```csv
Factor,Momentum,Value,Size,Volatility,Quality,Growth,Liquidity,Dividend_Yield
Momentum,0.04,0.01,-0.005,0.008,-0.002,0.015,0.003,-0.001
Value,0.01,0.03,0.012,-0.006,0.008,-0.01,0.005,0.01
...
```

---

#### 8. Constraints (`08_Optimization_Constraints.csv`)
Investment guidelines and limits:
```csv
Constraint_ID,Constraint_Type,Target,Min_Value,Max_Value,Is_Enabled
single_stock_AAPL,single_stock_active,AAPL,-1.0,1.0,TRUE
sector_IT,sector_active,Information Technology,-2.0,2.0,TRUE
```
- `single_stock_active`: Max deviation from benchmark per stock (%)
- `sector_active`: Max deviation from benchmark per sector (%)
- Set `Is_Enabled=FALSE` to disable

---

#### 9. Transaction Costs (`09_Transaction_Cost_Model.csv`)
Trading costs per security:
```csv
Ticker,Security_Name,Bid_Ask_Spread_Bps,Commission_Bps,Market_Impact_Bps,Total_Cost_Bps
AAPL,Apple Inc.,1.5,0.5,2.0,4.0
MSFT,Microsoft,1.8,0.5,2.2,4.5
SMALL,Small Cap Stock,8.0,0.5,12.0,20.5
```
- Costs in basis points (100 bps = 1%)

---

### Quickstart: Custom Data

**Example: Switch from S&P 500 to Russell 2000**

1. Replace `01_SP500_Benchmark_Constituency.csv` with Russell 2000 constituents
2. Replace `02_SP500_Universe.csv` with Russell 2000 universe
3. Update `04_Alpha_Model_SP500.csv` with your alpha signals for those stocks
4. Update `05_Risk_Model_Factor_Loadings.csv` with factor loadings
5. Run optimization - the system will now build a Russell 2000 portfolio!

**Example: Use Your Own Alpha Model**

Replace only `04_Alpha_Model_SP500.csv`:
```csv
Ticker,Alpha_Score,Alpha_Quintile
AAPL,0.85,1
MSFT,0.72,2
...
```
- Your own model's output, ML predictions, analyst ratings, etc.
- System will select top N securities by `Alpha_Quintile` (quintile 1 first)

## 🚀 Usage

### Start the Server

```bash
# Using uv
uv run python main.py

# Or directly
python main.py

# Or with uvicorn
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

### API Endpoints

Once running, access the interactive API docs at: **http://localhost:8000/docs**

#### Key Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Welcome message and API info |
| `/health` | GET | Health check |
| `/portfolio/benchmark` | GET | Get benchmark data |
| `/portfolio/alpha/top?n=25` | GET | Get top N alpha scores |
| `/portfolio/constraints` | GET | Get optimization constraints |
| `/optimization/run` | POST | Run portfolio optimization |

### Run Optimization

```bash
# Using curl
curl -X POST http://localhost:8000/optimization/run \
  -H "Content-Type: application/json" \
  -d '{
    "portfolio_id": "my_portfolio",
    "portfolio_size": 25,
    "risk_aversion": 0.01,
    "use_llm_analysis": false
  }'

# Using Python
import requests

response = requests.post(
    "http://localhost:8000/optimization/run",
    json={
        "portfolio_id": "my_portfolio",
        "portfolio_size": 25,
        "risk_aversion": 0.01,
        "use_llm_analysis": False
    }
)
result = response.json()
print(result["weights"])  # Note: field is "weights", not "optimal_weights"
```

### Example Response

```json
{
  "status": "completed",
  "is_compliant": true,
  "portfolio_id": "my_portfolio",
  "as_of_date": "2025-12-05",
  "total_holdings": 25,
  "weights": {
    "AAPL": 0.072,
    "MSFT": 0.065,
    "GOOGL": 0.045
  },
  "expected_alpha": 0.85,
  "expected_risk_pct": 14.5,
  "iterations": 1,
  "violations": [],
  "execution_log": [...]
}
```

## 🧪 Testing

```bash
# Run all tests
uv run pytest tests/ -v

# Run specific test file
uv run pytest tests/test_services/test_alpha_service.py -v

# Run with coverage
uv run pytest tests/ --cov=app --cov-report=html
```

## 📁 Project Structure

```
Multi-Agent-Equity-Portfolio-Manager/
├── app/
│   ├── agents/                    # LangGraph agents
│   │   ├── data_agent.py          # ReAct agent for data loading
│   │   ├── alpha_agent.py          # ReAct agent for alpha selection
│   │   ├── risk_agent.py           # ReAct agent for risk analysis
│   │   ├── cot_optimization_agent.py  # Chain-of-Thought optimization agent
│   │   ├── compliance_agent.py    # Rule-based compliance validation
│   │   ├── graph.py                # LangGraph orchestration
│   │   ├── state.py                # Shared PortfolioState definition
│   │   └── prompts.py              # System prompts for agents
│   ├── api/                        # FastAPI routers
│   │   ├── health_router.py
│   │   ├── portfolio_router.py
│   │   └── optimization_router.py
│   ├── core/                       # Configuration & dependencies
│   │   ├── config.py               # Settings management
│   │   ├── constants.py            # Application constants
│   │   └── dependencies.py        # Dependency injection
│   ├── llm/                        # LLM provider implementations
│   │   ├── factory.py              # LLM provider factory
│   │   ├── openai_provider.py
│   │   ├── deepseek_provider.py
│   │   ├── anthropic_provider.py
│   │   └── interfaces/
│   ├── models/                     # Pydantic domain models
│   │   ├── alpha.py
│   │   ├── benchmark.py
│   │   ├── constraint.py
│   │   ├── risk.py
│   │   ├── transaction_cost.py
│   │   └── ...
│   ├── repositories/               # Data access layer
│   │   ├── csv/                    # CSV-based repositories
│   │   │   ├── csv_alpha_repository.py
│   │   │   ├── csv_benchmark_repository.py
│   │   │   ├── csv_risk_repository.py
│   │   │   └── ...
│   │   └── interfaces/             # Repository interfaces
│   ├── schemas/                    # API request/response schemas
│   │   ├── agent_schema.py
│   │   ├── optimization_schema.py
│   │   └── portfolio_schema.py
│   ├── services/                   # Business logic layer
│   │   ├── data_service.py
│   │   ├── alpha_service.py
│   │   ├── risk_service.py
│   │   ├── optimization_service.py
│   │   ├── compliance_service.py
│   │   └── interfaces/
│   ├── solvers/                    # Optimization solvers
│   │   ├── cvxpy_solver.py
│   │   ├── scipy_solver.py
│   │   ├── factory.py
│   │   └── interfaces/
│   ├── tools/                      # LangChain tools for agents
│   │   ├── base.py                 # BaseTool class
│   │   ├── data_tools.py           # Class-based tools
│   │   └── langchain_tools.py     # @tool decorator tools
│   └── utils/                      # Utilities
│       └── csv_loader.py           # CSV loading utilities
├── data/                           # Input CSV files (customizable)
│   ├── 01_SP500_Benchmark_Constituency.csv
│   ├── 04_Alpha_Model_SP500.csv
│   ├── 05_Risk_Model_Factor_Loadings.csv
│   └── ...
├── tests/                          # Test suite
│   ├── test_agents/
│   ├── test_services/
│   ├── test_repositories/
│   ├── test_api/
│   ├── test_tools/
│   └── conftest.py
├── main.py                         # Application entry point
├── .env                            # Environment configuration
├── env.example                     # Example environment file
├── pyproject.toml                  # Dependencies & project config
├── Dockerfile                      # Docker image definition
├── docker-compose.yml              # Docker Compose configuration
├── README.md                       # This file
├── USER_GUIDE.md                   # User documentation
├── DEVELOPER_GUIDE.md              # Developer documentation
└── IMPLEMENTATION_PLAN.md          # Implementation tracking
```

## 🔧 Advanced Configuration

### Using Different LLM Providers

**OpenAI (GPT-4)**
```bash
LLM_PROVIDER=openai
LLM_MODEL=gpt-4
OPENAI_API_KEY=sk-...
```

**DeepSeek**
```bash
LLM_PROVIDER=deepseek
LLM_MODEL=deepseek-chat
DEEPSEEK_API_KEY=sk-...
```

**Anthropic (Claude)**
```bash
LLM_PROVIDER=anthropic
LLM_MODEL=claude-3-opus-20240229
ANTHROPIC_API_KEY=sk-ant-...
```

### Switching Optimization Solvers

**CVXPY (Convex Optimization - Default)**
```bash
SOLVER=cvxpy
```

**SciPy (General Purpose)**
```bash
SOLVER=scipy
```

### Adjusting Risk Parameters

```bash
# More aggressive (higher alpha, higher risk)
RISK_AVERSION=0.001
PORTFOLIO_SIZE=50

# More conservative (lower alpha, lower risk)
RISK_AVERSION=0.1
PORTFOLIO_SIZE=15
```

## 📚 Documentation

- **[User Guide](USER_GUIDE.md)** - Complete guide for using the system, including data preparation, API usage, and troubleshooting
- **[Developer Guide](DEVELOPER_GUIDE.md)** - Technical documentation for developers, including architecture, extending the system, and testing

## 📝 License

MIT License - see [LICENSE](LICENSE) for details.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

*Built with ❤️ using LangGraph, FastAPI, and modern Python*
