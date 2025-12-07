# User Guide: Multi-Agent Equity Portfolio Manager

A practical guide for using the Multi-Agent Equity Portfolio Manager to construct optimized equity portfolios.

## Table of Contents

1. [Quick Start](#quick-start)
2. [Understanding the System](#understanding-the-system)
3. [Preparing Your Data](#preparing-your-data)
4. [Running Portfolio Optimization](#running-portfolio-optimization)
5. [Interpreting Results](#interpreting-results)
6. [Common Use Cases](#common-use-cases)
7. [Troubleshooting](#troubleshooting)

---

## Quick Start

### 1. Install and Configure

```bash
# Clone the repository
git clone <repository-url>
cd Multi-Agent-Equity-Portfolio-Manager

# Install dependencies (using uv)
uv sync

# Copy and configure environment
cp env.example .env
# Edit .env with your LLM API key (OpenAI, DeepSeek, Anthropic, or OpenRouter)
```

### 2. Prepare Your Data

Place your CSV files in the `data/` folder:
- `01_SP500_Benchmark_Constituency.csv` - Benchmark constituents
- `04_Alpha_Model_SP500.csv` - Your alpha scores
- `05_Risk_Model_Factor_Loadings.csv` - Risk factor loadings
- (See [Data Files](#data-files) for complete list)

### 3. Start the Server

```bash
# Using Docker (recommended)
docker compose up -d

# Or directly
python main.py
```

### 4. Run Optimization

```bash
curl -X POST http://localhost:8000/optimization/run \
  -H "Content-Type: application/json" \
  -d '{
    "portfolio_id": "MY_PORTFOLIO",
    "as_of_date": "2025-12-05",
    "portfolio_size": 25,
    "use_llm_analysis": false
  }'
```

---

## Understanding the System

### What This System Does

The Multi-Agent Equity Portfolio Manager constructs optimized equity portfolios by:

1. **Loading Data**: Reads benchmark, universe, alpha scores, risk model, and constraints
2. **Selecting Securities**: Chooses top N securities based on alpha signals
3. **Analyzing Risk**: Calculates portfolio risk using a multi-factor model
4. **Optimizing Weights**: Determines optimal portfolio weights that maximize alpha while controlling risk
5. **Ensuring Compliance**: Validates the portfolio meets all investment constraints

### Two Optimization Modes

#### 1. Mathematical Optimization (`use_llm_analysis: false`)
- Uses CVXPY/SCIPY solvers
- Fast, deterministic results
- Best for production use
- Requires feasible constraints

#### 2. LLM-Driven Optimization (`use_llm_analysis: true`)
- Uses AI reasoning to determine weights
- Can handle complex, nuanced decisions
- Useful for exploratory analysis
- Slower but more flexible

### Key Concepts

**Alpha Score**: Expected excess return (0-1 scale)
- 0.80-1.00 = Quintile 1 (Top 20%, strongest buy signal)
- 0.60-0.80 = Quintile 2
- 0.40-0.60 = Quintile 3 (Neutral)
- 0.20-0.40 = Quintile 4
- 0.00-0.20 = Quintile 5 (Bottom 20%, weakest)

**Active Weight**: Difference between portfolio weight and benchmark weight
- Example: If AAPL is 6.7% of benchmark, portfolio can hold 5.7%-7.7% (±1%)

**Risk Aversion**: Controls trade-off between alpha and risk
- Lower (0.001) = More aggressive, higher alpha, higher risk
- Higher (0.1) = More conservative, lower alpha, lower risk

---

## Preparing Your Data

### Required CSV Files

All files should be placed in the `data/` folder:

| File | Purpose | Required Columns |
|------|---------|------------------|
| `01_SP500_Benchmark_Constituency.csv` | Benchmark definition | `Ticker`, `Benchmark_Weight_Pct`, `GICS_Sector` |
| `04_Alpha_Model_SP500.csv` | Alpha signals | `Ticker`, `Alpha_Score`, `Alpha_Quintile` |
| `05_Risk_Model_Factor_Loadings.csv` | Risk exposures | `Ticker`, `Market_Loading`, `Size_Loading`, etc. |
| `08_Optimization_Constraints.csv` | Investment limits | `Constraint_Type`, `Constraint_Name`, `Lower_Bound_Pct`, `Upper_Bound_Pct` |

### Data File Examples

#### Alpha Model (`04_Alpha_Model_SP500.csv`)

```csv
Ticker,Security_Name,Alpha_Score,Alpha_Quintile,Signal_Date
AAPL,Apple Inc.,0.85,1,2025-12-05
MSFT,Microsoft Corporation,0.78,1,2025-12-05
GOOGL,Alphabet Inc.,0.72,2,2025-12-05
```

**Tips:**
- Alpha scores should be between 0.0 and 1.0
- Quintile 1 = top 20% (best signals)
- Ensure all tickers match your benchmark file

#### Constraints (`08_Optimization_Constraints.csv`)

```csv
Constraint_Type,Constraint_Name,Benchmark_Weight_Pct,Lower_Bound_Pct,Upper_Bound_Pct,Constraint_Type_Code,Is_Hard_Constraint
Stock,AAPL,6.7,-1.0,1.0,REL,True
Sector,Information Technology,32.0,-2.0,2.0,REL,True
```

**Understanding Constraints:**
- **Stock constraint**: Portfolio weight must be within ±1% of benchmark weight
  - If AAPL is 6.7% of benchmark → portfolio can hold 5.7%-7.7%
- **Sector constraint**: Total sector weight must be within ±2% of benchmark
  - If Tech is 32% of benchmark → portfolio can have 30%-34% in Tech

---

## Running Portfolio Optimization

### Using the API

#### Basic Request

```bash
POST http://localhost:8000/optimization/run
Content-Type: application/json

{
  "portfolio_id": "ALPHA_GROWTH_25",
  "as_of_date": "2025-12-05",
  "portfolio_size": 25,
  "risk_aversion": 0.01,
  "use_llm_analysis": false,
  "max_iterations": 5
}
```

#### Request Parameters

| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| `portfolio_id` | string | Portfolio identifier | `"ALPHA_GROWTH_25"` |
| `as_of_date` | string | Data date (YYYY-MM-DD) | Latest available |
| `portfolio_size` | integer | Number of holdings | 25 |
| `risk_aversion` | float | Risk penalty (higher = conservative) | 0.01 |
| `use_llm_analysis` | boolean | Use LLM for optimization | false |
| `max_iterations` | integer | Max compliance retry loops | 5 |

#### Example: Conservative Portfolio

```json
{
  "portfolio_id": "CONSERVATIVE_15",
  "portfolio_size": 15,
  "risk_aversion": 0.1,
  "use_llm_analysis": false
}
```

#### Example: Aggressive Portfolio

```json
{
  "portfolio_id": "AGGRESSIVE_50",
  "portfolio_size": 50,
  "risk_aversion": 0.001,
  "use_llm_analysis": false
}
```

### Using Python

```python
import requests

response = requests.post(
    "http://localhost:8000/optimization/run",
    json={
        "portfolio_id": "MY_PORTFOLIO",
        "as_of_date": "2025-12-05",
        "portfolio_size": 25,
        "risk_aversion": 0.01,
        "use_llm_analysis": False
    }
)

result = response.json()

# Check status
print(f"Status: {result['status']}")
print(f"Is Compliant: {result['is_compliant']}")
print(f"Total Holdings: {result['total_holdings']}")

# View portfolio weights
for ticker, weight_pct in result['weights'].items():
    print(f"{ticker}: {weight_pct:.2f}%")
```

### Using Docker

```bash
# Start server
docker compose up -d

# Run optimization
curl -X POST http://localhost:8000/optimization/run \
  -H "Content-Type: application/json" \
  -d @optimization_request.json

# View logs
docker compose logs -f
```

---

## Interpreting Results

### Response Structure

```json
{
  "status": "completed",
  "is_compliant": true,
  "portfolio_id": "ALPHA_GROWTH_25",
  "total_holdings": 25,
  "weights": {
    "AAPL": 0.072,
    "MSFT": 0.065,
    "GOOGL": 0.045
  },
  "expected_alpha": 0.85,
  "expected_risk_pct": 14.5,
  "violations": [],
  "iterations": 1,
  "execution_log": [...]
}
```

### Key Fields

- **`status`**: `"completed"` (success) or `"solver_error"` (failed)
- **`is_compliant`**: `true` if portfolio meets all constraints
- **`weights`**: Dictionary of ticker → weight (decimal, e.g., 0.072 = 7.2%)
- **`expected_alpha`**: Portfolio's expected alpha score
- **`expected_risk_pct`**: Expected portfolio volatility (%)
- **`violations`**: List of constraint violations (if any)
- **`iterations`**: Number of optimization attempts

### Understanding Violations

If `is_compliant: false`, check the `violations` array:

```json
{
  "violations": [
    {
      "type": "stock",
      "name": "AAPL",
      "current_weight": 8.5,
      "benchmark_weight": 6.7,
      "active_weight": 1.8,
      "max_allowed": 1.0,
      "breach_amount": 0.8,
      "severity": "moderate"
    }
  ]
}
```

**Action**: Adjust constraints or increase `max_iterations` to allow retries.

---

## Common Use Cases

### Use Case 1: Monthly Rebalancing

**Scenario**: Rebalance portfolio monthly using latest alpha signals.

**Steps**:
1. Update `04_Alpha_Model_SP500.csv` with latest alpha scores
2. Update `03_Portfolio_25_Holdings.csv` with current holdings
3. Run optimization:
   ```json
   {
     "portfolio_id": "MONTHLY_REBALANCE",
     "as_of_date": "2025-12-05",
     "portfolio_size": 25,
     "use_llm_analysis": false
   }
   ```
4. Review `weights` and `violations`
5. Execute trades based on weight changes

### Use Case 2: Testing Different Alpha Models

**Scenario**: Compare portfolios built with different alpha models.

**Steps**:
1. Create multiple alpha files:
   - `04_Alpha_Model_SP500_v1.csv`
   - `04_Alpha_Model_SP500_v2.csv`
2. For each version:
   - Copy to `04_Alpha_Model_SP500.csv`
   - Run optimization
   - Save results
3. Compare `expected_alpha` and `expected_risk_pct`

### Use Case 3: Sector Overweight/Underweight

**Scenario**: Overweight Technology sector by +3% vs benchmark.

**Steps**:
1. Edit `08_Optimization_Constraints.csv`:
   ```csv
   Sector,Information Technology,32.0,1.0,5.0,REL,True
   ```
   (Lower bound = +1%, Upper bound = +5%)
2. Run optimization
3. Verify sector weight in results

### Use Case 4: LLM-Driven Exploration

**Scenario**: Use AI reasoning to explore non-standard portfolio construction.

**Steps**:
1. Set `use_llm_analysis: true`
2. Provide detailed prompt context (if extending system)
3. Review `optimization_analysis` field for LLM reasoning
4. Compare with mathematical optimization results

---

## Troubleshooting

### Problem: "Solver Error" or "Infeasible"

**Cause**: Constraints are too tight (e.g., 25 stocks × 1% max each = only 25% total, but need 100%)

**Solutions**:
1. Increase `portfolio_size` to allow more diversification
2. Relax constraints in `08_Optimization_Constraints.csv`
3. Check benchmark weights - very small weights (<0.1%) may cause issues
4. Try `use_llm_analysis: true` for more flexible optimization

### Problem: "No Securities Selected"

**Cause**: Alpha scores missing or all securities filtered out

**Solutions**:
1. Verify `04_Alpha_Model_SP500.csv` has data
2. Check that tickers match benchmark file
3. Ensure `Alpha_Quintile` values are 1-5
4. Check `execution_log` for specific errors

### Problem: "Data Validation Failed"

**Cause**: Missing or invalid CSV files

**Solutions**:
1. Verify all required files exist in `data/` folder
2. Check CSV column names match expected format
3. Ensure dates are in YYYY-MM-DD format
4. Check `data_validation_issues` in response

### Problem: "LLM Timeout"

**Cause**: LLM API call taking too long

**Solutions**:
1. Check API key is valid
2. Reduce `portfolio_size` (fewer securities = faster)
3. Use `use_llm_analysis: false` for faster results
4. Check network connectivity

### Problem: Portfolio Not Compliant After Multiple Iterations

**Cause**: Constraints are conflicting or too restrictive

**Solutions**:
1. Review `violations` to identify problematic constraints
2. Temporarily relax constraints to test feasibility
3. Increase `max_iterations` (default: 5)
4. Consider using `use_llm_analysis: true` for more flexible handling

---

## Best Practices

### 1. Data Quality
- **Validate CSV files** before running optimization
- **Ensure ticker consistency** across all files
- **Check date formats** (YYYY-MM-DD)
- **Verify weights sum to 100%** in benchmark file

### 2. Constraint Design
- **Start with loose constraints** and tighten gradually
- **Test feasibility** with small `portfolio_size` first
- **Document constraint rationale** in comments or separate file

### 3. Optimization Strategy
- **Use mathematical optimization** (`use_llm_analysis: false`) for production
- **Use LLM optimization** for exploratory analysis
- **Monitor `iterations`** - high count indicates constraint issues

### 4. Performance
- **Cache results** for repeated runs with same data
- **Use Docker** for consistent environment
- **Monitor execution logs** for performance insights

---

## Getting Help

### API Documentation
- Interactive docs: `http://localhost:8000/docs`
- OpenAPI spec: `http://localhost:8000/openapi.json`

### Logs
- Server logs: `docker compose logs -f`
- Execution log: Check `execution_log` field in response

### Common Issues
- See [Troubleshooting](#troubleshooting) section above
- Check `execution_log` for detailed error messages

---

*For technical details and development information, see [DEVELOPER_GUIDE.md](DEVELOPER_GUIDE.md)*

