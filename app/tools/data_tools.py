"""
Data loading tools for ReAct agents.

These tools load data from CSV files. In the future, additional tools
can be added to load data from APIs, databases, etc.

The LLM agent reads the tool descriptions to decide which tool to use.
"""

from datetime import date
from typing import Any, Optional

from app.tools.base import BaseTool, ToolResult


def _get_repositories():
    """Lazy import to avoid circular dependency."""
    from app.core.dependencies import (
        get_benchmark_repository,
        get_universe_repository,
        get_alpha_repository,
        get_risk_repository,
        get_constraint_repository,
        get_transaction_cost_repository,
    )
    return {
        'benchmark': get_benchmark_repository,
        'universe': get_universe_repository,
        'alpha': get_alpha_repository,
        'risk': get_risk_repository,
        'constraint': get_constraint_repository,
        'transaction_cost': get_transaction_cost_repository,
    }


class LoadBenchmarkTool(BaseTool):
    """Tool to load benchmark constituency data."""
    
    @property
    def name(self) -> str:
        return "load_benchmark"
    
    @property
    def description(self) -> str:
        return """Load S&P 500 benchmark constituency data from CSV file.

Returns:
- List of 500 securities with their benchmark weights
- Each security includes: ticker, name, GICS sector, benchmark weight (%), price
- Data is sorted by benchmark weight descending

Use this tool when you need:
- Benchmark constituent information
- Benchmark weights for active weight calculations
- Sector breakdown of the benchmark
- Security universe definition
"""

    @property
    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "as_of_date": {
                    "type": "string",
                    "description": "Data as-of date in YYYY-MM-DD format. Optional - uses latest if not provided."
                }
            }
        }

    async def execute(self, as_of_date: Optional[str] = None, **kwargs) -> ToolResult:
        try:
            repos = _get_repositories()
            repo = repos['benchmark']()
            parsed_date = None
            if as_of_date:
                from datetime import datetime
                parsed_date = datetime.strptime(as_of_date, "%Y-%m-%d").date()
            
            benchmark = await repo.get_benchmark(as_of_date=parsed_date)
            
            if not benchmark:
                return ToolResult(
                    success=False,
                    error="No benchmark data found",
                    message="Failed to load benchmark data"
                )
            
            # Format data for LLM
            summary = {
                "benchmark_id": benchmark.benchmark_id,
                "security_count": benchmark.security_count,
                "total_weight": sum(c.benchmark_weight_pct for c in benchmark.constituents),
                "as_of_date": str(benchmark.as_of_date),
                "top_10_holdings": [
                    {
                        "ticker": c.ticker,
                        "name": c.security_name,
                        "sector": c.gics_sector,
                        "weight_pct": round(c.benchmark_weight_pct, 2)
                    }
                    for c in sorted(benchmark.constituents, key=lambda x: x.benchmark_weight_pct, reverse=True)[:10]
                ],
                "sector_weights": benchmark.get_sector_weights(),
            }
            
            return ToolResult(
                success=True,
                data=benchmark,
                message=f"Loaded benchmark with {benchmark.security_count} securities. "
                        f"Top holdings: {', '.join(h['ticker'] for h in summary['top_10_holdings'][:5])}"
            )
        except Exception as e:
            return ToolResult(success=False, error=str(e))


class LoadUniverseTool(BaseTool):
    """Tool to load investable universe data."""
    
    @property
    def name(self) -> str:
        return "load_universe"
    
    @property
    def description(self) -> str:
        return """Load investable universe data from CSV file.

Returns:
- List of securities that are eligible for investment
- Includes investibility flags, liquidity scores, position limits
- Filters out securities that are not investible

Use this tool when you need:
- The list of securities you CAN invest in
- Liquidity information for trading
- Position size limits per security
"""

    async def execute(self, as_of_date: Optional[str] = None, **kwargs) -> ToolResult:
        try:
            repos = _get_repositories()
            repo = repos['universe']()
            securities = await repo.get_all()
            
            investible = [s for s in securities if s.is_investible]
            
            return ToolResult(
                success=True,
                data=investible,
                message=f"Loaded universe with {len(investible)} investible securities out of {len(securities)} total"
            )
        except Exception as e:
            return ToolResult(success=False, error=str(e))


class LoadAlphaScoresTool(BaseTool):
    """Tool to load alpha model scores."""
    
    @property
    def name(self) -> str:
        return "load_alpha_scores"
    
    @property
    def description(self) -> str:
        return """Load AI-generated alpha scores from CSV file.

Returns:
- Alpha scores (0-1) for each security in the universe
- Alpha quintile rankings (Q1=best, Q5=worst)
- Expected returns and signal strength

Alpha Score Interpretation:
- 0.80-1.00 = Quintile 1 (Top 20%, strongest buy signal)
- 0.60-0.80 = Quintile 2
- 0.40-0.60 = Quintile 3 (Neutral)
- 0.20-0.40 = Quintile 4
- 0.00-0.20 = Quintile 5 (Bottom 20%, weakest)

Use this tool when you need:
- Alpha signals for portfolio construction
- To identify top-ranked securities for selection
- Expected return estimates
"""

    async def execute(self, as_of_date: Optional[str] = None, **kwargs) -> ToolResult:
        try:
            repos = _get_repositories()
            repo = repos['alpha']()
            alpha_model = await repo.get_alpha_model()
            
            if not alpha_model:
                return ToolResult(
                    success=False,
                    error="No alpha model data found",
                    message="Failed to load alpha scores"
                )
            
            # Group by quintile
            quintile_counts = {}
            for score in alpha_model.scores:
                q = score.alpha_quintile
                quintile_counts[q] = quintile_counts.get(q, 0) + 1
            
            top_scores = sorted(alpha_model.scores, key=lambda x: x.alpha_score, reverse=True)[:10]
            
            return ToolResult(
                success=True,
                data=alpha_model,
                message=f"Loaded alpha scores for {alpha_model.security_count} securities. "
                        f"Q1 (top) has {quintile_counts.get(1, 0)} securities. "
                        f"Top alpha: {top_scores[0].ticker} ({top_scores[0].alpha_score:.3f})"
            )
        except Exception as e:
            return ToolResult(success=False, error=str(e))


class LoadRiskModelTool(BaseTool):
    """Tool to load risk model data."""
    
    @property
    def name(self) -> str:
        return "load_risk_model"
    
    @property
    def description(self) -> str:
        return """Load Barra-style multi-factor risk model from CSV files.

Returns:
- Factor loadings for each security (8 factors)
- Factor covariance matrix (8x8)
- Specific (idiosyncratic) risk per security

Risk Factors:
1. Market - Broad market beta exposure
2. Size - Market capitalization factor
3. Value - Value vs growth characteristics
4. Momentum - Price momentum
5. Quality - Earnings quality
6. Volatility - Stock volatility
7. Growth - Earnings/revenue growth
8. Dividend Yield - Dividend characteristics

Use this tool when you need:
- Risk exposures for portfolio construction
- Covariance matrix for variance calculation
- Specific risk estimates
"""

    async def execute(self, as_of_date: Optional[str] = None, **kwargs) -> ToolResult:
        try:
            repos = _get_repositories()
            repo = repos['risk']()
            risk_model = await repo.get_risk_model()
            
            if not risk_model:
                return ToolResult(
                    success=False,
                    error="No risk model data found",
                    message="Failed to load risk model"
                )
            
            has_cov = risk_model.factor_covariance is not None
            
            return ToolResult(
                success=True,
                data=risk_model,
                message=f"Loaded risk model with {risk_model.security_count} securities, "
                        f"8 factors, covariance matrix: {'Yes' if has_cov else 'No'}"
            )
        except Exception as e:
            return ToolResult(success=False, error=str(e))


class LoadConstraintsTool(BaseTool):
    """Tool to load optimization constraints."""
    
    @property
    def name(self) -> str:
        return "load_constraints"
    
    @property
    def description(self) -> str:
        return """Load optimization constraints from CSV file.

Returns:
- Single stock active weight limits (default ±1% vs benchmark)
- Sector active weight limits (default ±2% vs benchmark)
- Constraint types: REL (relative to benchmark) or ABS (absolute)

Constraint Rules:
- Single Stock: Portfolio weight must be within ±1% of benchmark weight
  Example: If AAPL is 6.7% of benchmark, portfolio can hold 5.7%-7.7%
- Sector: Total sector weight must be within ±2% of benchmark sector weight
  Example: If Tech is 32% of benchmark, portfolio can have 30%-34% in Tech

Use this tool when you need:
- Position limits for optimization
- Compliance thresholds
- Constraint validation rules
"""

    async def execute(self, as_of_date: Optional[str] = None, **kwargs) -> ToolResult:
        try:
            repos = _get_repositories()
            repo = repos['constraint']()
            constraint_set = await repo.get_constraint_set()
            
            if not constraint_set:
                return ToolResult(
                    success=False,
                    error="No constraints found",
                    message="Failed to load constraints"
                )
            
            return ToolResult(
                success=True,
                data=constraint_set,
                message=f"Loaded {len(constraint_set.stock_constraints)} stock constraints "
                        f"and {len(constraint_set.sector_constraints)} sector constraints"
            )
        except Exception as e:
            return ToolResult(success=False, error=str(e))


class LoadTransactionCostsTool(BaseTool):
    """Tool to load transaction cost model."""
    
    @property
    def name(self) -> str:
        return "load_transaction_costs"
    
    @property
    def description(self) -> str:
        return """Load transaction cost estimates from CSV file.

Returns:
- Per-security transaction costs in basis points (bps)
- Components: bid-ask spread, commission, market impact
- Liquidity buckets (1=most liquid, 5=least liquid)

Cost Calculation:
Total One-Way Cost = Bid_Ask_Spread/2 + Commission + Market_Impact × Trade_Size

Urgency Adjustments:
- Low Urgency: Cost × 0.70 (patient TWAP/VWAP execution)
- Medium Urgency: Cost × 1.00 (normal execution)
- High Urgency: Cost × 1.50 (aggressive, immediate)

Use this tool when you need:
- Transaction cost estimates for rebalancing
- Liquidity information for trade sizing
- Cost-aware portfolio construction
"""

    async def execute(self, as_of_date: Optional[str] = None, **kwargs) -> ToolResult:
        try:
            repos = _get_repositories()
            repo = repos['transaction_cost']()
            tcost_model = await repo.get_transaction_cost_model()
            
            if not tcost_model:
                return ToolResult(
                    success=False,
                    error="No transaction cost data found",
                    message="Failed to load transaction costs"
                )
            
            avg_cost = sum(c.total_oneway_cost_bps for c in tcost_model.costs) / len(tcost_model.costs) if tcost_model.costs else 0
            
            return ToolResult(
                success=True,
                data=tcost_model,
                message=f"Loaded transaction costs for {tcost_model.security_count} securities. "
                        f"Average one-way cost: {avg_cost:.1f} bps"
            )
        except Exception as e:
            return ToolResult(success=False, error=str(e))


# Tool registry - all available data tools
DATA_TOOLS = [
    LoadBenchmarkTool(),
    LoadUniverseTool(),
    LoadAlphaScoresTool(),
    LoadRiskModelTool(),
    LoadConstraintsTool(),
    LoadTransactionCostsTool(),
]


def get_data_tools() -> list[BaseTool]:
    """Get all available data loading tools."""
    return DATA_TOOLS


def get_tool_by_name(name: str) -> Optional[BaseTool]:
    """Get a tool by its name."""
    for tool in DATA_TOOLS:
        if tool.name == name:
            return tool
    return None

