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

- **Multi-Agent Architecture**: 5 specialized agents (Data, Alpha, Risk, Optimization, Compliance)
- **Flexible LLM Integration**: Supports OpenAI (GPT-4), DeepSeek, and Anthropic (Claude)
- **Mean-Variance Optimization**: Constrained portfolio construction using `cvxpy` or `scipy`
- **Barra-Style Risk Model**: 8-factor model with idiosyncratic risk
- **Constraint Management**: Single stock (±1%) and sector (±2%) active weight limits
- **Customizable Data**: Users can provide their own benchmark, universe, alpha scores, and constraints

## 🏗️ Architecture

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

All CSV files in the `data/` folder can be replaced with your own data. Below are examples for each file:

#### 1. Benchmark Constituents (`01_SP500_Benchmark_Constituency.csv`)
Define your benchmark index (e.g., S&P 500, Russell 1000, custom index):
```csv
Ticker,Security_Name,GICS_Sector,GICS_Industry,Benchmark_Weight_Pct,Price,As_Of_Date,Benchmark_ID
AAPL,Apple Inc.,Information Technology,Technology Hardware,6.5,180.00,2025-01-15,SPX
MSFT,Microsoft,Information Technology,Software,5.8,400.00,2025-01-15,SPX
JPM,JPMorgan Chase,Financials,Banks,2.1,180.00,2025-01-15,SPX
```

#### 2. Investment Universe (`02_SP500_Universe.csv`)
Define which securities are eligible for investment:
```csv
Ticker,Security_Name,GICS_Sector,Is_Investible,Liquidity_Score,Market_Cap_USD_B
AAPL,Apple Inc.,Information Technology,TRUE,0.95,3000
MSFT,Microsoft,Information Technology,TRUE,0.92,2800
PENNY,Penny Stock Co,Financials,FALSE,0.10,0.5
```
- Set `Is_Investible=FALSE` to exclude securities from selection

#### 3. Current Portfolio (`03_Portfolio_25_Holdings.csv`)
Your existing holdings (used for transaction cost calculation):
```csv
Ticker,Security_Name,Shares,Price,Market_Value,Portfolio_Weight_Pct
AAPL,Apple Inc.,1000,180.00,180000,7.2
MSFT,Microsoft,500,400.00,200000,8.0
```

#### 4. Alpha Model (`04_Alpha_Model_SP500.csv`)
Your alpha signals/scores for each security:
```csv
Ticker,Security_Name,Alpha_Score,Alpha_Quintile,Signal_Date
AAPL,Apple Inc.,0.85,1,2025-01-15
MSFT,Microsoft,0.72,2,2025-01-15
XOM,Exxon Mobil,0.35,4,2025-01-15
```
- `Alpha_Score`: 0.0 (most bearish) to 1.0 (most bullish)
- `Alpha_Quintile`: 1 (top 20%) to 5 (bottom 20%)

#### 5. Factor Loadings (`05_Risk_Model_Factor_Loadings.csv`)
Factor exposures for each security (Barra-style risk model):
```csv
Ticker,Momentum,Value,Size,Volatility,Quality,Growth,Liquidity,Dividend_Yield,Idiosyncratic_Risk
AAPL,0.8,-0.3,-0.5,-0.2,0.6,0.7,-0.4,-0.1,0.15
MSFT,0.6,-0.2,-0.6,-0.3,0.7,0.5,-0.3,0.1,0.12
```
- Factor loadings are typically standardized (z-scores)
- `Idiosyncratic_Risk`: Security-specific volatility not explained by factors

#### 6. Factor Returns (`06_Risk_Model_Factor_Returns.csv`)
Historical or expected returns for each factor:
```csv
Factor_Name,Factor_Return_Pct,Factor_Return_YTD_Pct
Momentum,0.5,2.1
Value,0.3,1.5
Size,-0.2,-0.8
Volatility,-0.4,-1.2
Quality,0.4,1.8
Growth,0.6,2.5
Liquidity,0.1,0.4
Dividend_Yield,0.2,0.9
```

#### 7. Factor Covariance Matrix (`07_Risk_Model_Factor_Covariance.csv`)
Covariance between factors (8×8 matrix):
```csv
Factor,Momentum,Value,Size,Volatility,Quality,Growth,Liquidity,Dividend_Yield
Momentum,0.04,0.01,-0.005,0.008,-0.002,0.015,0.003,-0.001
Value,0.01,0.03,0.012,-0.006,0.008,-0.01,0.005,0.01
Size,-0.005,0.012,0.02,0.01,-0.003,-0.008,0.015,0.002
...
```

#### 8. Optimization Constraints (`08_Optimization_Constraints.csv`)
Investment guidelines and limits:
```csv
Constraint_ID,Constraint_Type,Target,Min_Value,Max_Value,Is_Enabled
single_stock_AAPL,single_stock_active,AAPL,-1.0,1.0,TRUE
single_stock_MSFT,single_stock_active,MSFT,-1.0,1.0,TRUE
sector_IT,sector_active,Information Technology,-2.0,2.0,TRUE
sector_Financials,sector_active,Financials,-2.0,2.0,TRUE
```
- `single_stock_active`: Max deviation from benchmark weight per stock
- `sector_active`: Max deviation from benchmark weight per sector
- Set `Is_Enabled=FALSE` to disable a constraint

#### 9. Transaction Costs (`09_Transaction_Cost_Model.csv`)
Trading costs for each security:
```csv
Ticker,Security_Name,Bid_Ask_Spread_Bps,Commission_Bps,Market_Impact_Bps,Total_Cost_Bps
AAPL,Apple Inc.,1.5,0.5,2.0,4.0
MSFT,Microsoft,1.8,0.5,2.2,4.5
SMALL,Small Cap Stock,8.0,0.5,12.0,20.5
```
- Costs in basis points (bps), where 100 bps = 1%

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
  -d '{"portfolio_id": "my_portfolio", "n_securities": 25}'

# Using Python
import requests

response = requests.post(
    "http://localhost:8000/optimization/run",
    json={
        "portfolio_id": "my_portfolio",
        "n_securities": 25,
        "risk_aversion": 0.01
    }
)
result = response.json()
print(result["optimal_weights"])
```

### Example Response

```json
{
  "portfolio_id": "my_portfolio",
  "optimal_weights": {
    "AAPL": 0.072,
    "MSFT": 0.065,
    "GOOGL": 0.045,
    ...
  },
  "is_compliant": true,
  "iterations": 1,
  "risk_metrics": {
    "portfolio_volatility": 0.15,
    "tracking_error": 0.02
  }
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
│   ├── agents/           # LangGraph agents
│   │   ├── data_agent.py
│   │   ├── alpha_agent.py
│   │   ├── risk_agent.py
│   │   ├── optimization_agent.py
│   │   ├── compliance_agent.py
│   │   ├── graph.py      # LangGraph orchestration
│   │   └── state.py      # Shared state definition
│   ├── api/              # FastAPI routers
│   ├── core/             # Config & constants
│   ├── llm/              # LLM provider implementations
│   ├── models/           # Pydantic domain models
│   ├── repositories/     # Data access layer
│   ├── schemas/          # API request/response schemas
│   ├── services/         # Business logic
│   ├── solvers/          # Optimization solvers
│   └── utils/            # Utilities
├── data/                 # Input CSV files (customizable)
├── tests/                # Test suite
├── main.py               # Application entry point
├── .env                  # Environment configuration
└── pyproject.toml        # Dependencies
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
