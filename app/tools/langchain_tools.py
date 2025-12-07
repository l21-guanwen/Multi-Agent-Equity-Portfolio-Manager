"""
LangChain-compatible tools using the @tool decorator.

These tools can be used with LangGraph's create_react_agent.
The LLM reads the docstrings to understand when to use each tool.
"""

from typing import Optional
from langchain_core.tools import tool


def _get_repos():
    """Lazy import to avoid circular dependency."""
    from app.core.dependencies import (
        get_benchmark_repository,
        get_alpha_repository,
        get_risk_repository,
        get_constraint_repository,
        get_transaction_cost_repository,
    )
    return {
        'benchmark': get_benchmark_repository,
        'alpha': get_alpha_repository,
        'risk': get_risk_repository,
        'constraint': get_constraint_repository,
        'transaction_cost': get_transaction_cost_repository,
    }


@tool
async def load_benchmark(as_of_date: Optional[str] = None) -> dict:
    """Load S&P 500 benchmark constituency data.
    
    Returns benchmark constituents with their weights, sectors, and prices.
    Use this tool when you need:
    - Benchmark weights for active weight calculations
    - Security universe (all S&P 500 tickers)
    - Sector breakdown of the benchmark
    
    Args:
        as_of_date: Optional date string (YYYY-MM-DD). Uses latest if not provided.
        
    Returns:
        Dictionary with benchmark data including:
        - benchmark_id: Benchmark identifier (e.g., "SPX")
        - security_count: Number of constituents
        - constituents: List of securities with ticker, weight, sector
        - sector_weights: Dict of sector -> total weight
    """
    from datetime import datetime
    
    repo = _get_repos()['benchmark']()
    parsed_date = None
    if as_of_date:
        parsed_date = datetime.strptime(as_of_date, "%Y-%m-%d").date()
    
    benchmark = await repo.get_benchmark(as_of_date=parsed_date)
    
    if not benchmark:
        return {"error": "No benchmark data found"}
    
    return {
        "benchmark_id": benchmark.benchmark_id,
        "security_count": benchmark.security_count,
        "as_of_date": str(benchmark.as_of_date),
        "sector_weights": benchmark.get_sector_weights(),
        "top_holdings": [
            {"ticker": c.ticker, "weight_pct": c.benchmark_weight_pct, "sector": c.gics_sector}
            for c in sorted(benchmark.constituents, key=lambda x: x.benchmark_weight_pct, reverse=True)[:10]
        ],
        "all_tickers": [c.ticker for c in benchmark.constituents],
    }


@tool
async def load_alpha_scores(as_of_date: Optional[str] = None) -> dict:
    """Load AI-generated alpha scores for securities.
    
    Alpha scores represent expected returns (0-1 scale):
    - 0.80-1.00 = Quintile 1 (Top 20%, strongest buy signal)
    - 0.60-0.80 = Quintile 2
    - 0.40-0.60 = Quintile 3 (Neutral)
    - 0.20-0.40 = Quintile 4
    - 0.00-0.20 = Quintile 5 (Bottom 20%, weakest)
    
    Use this tool when you need:
    - Alpha signals for security selection
    - To identify top-ranked securities (Q1)
    - Expected return estimates
    
    Args:
        as_of_date: Optional date string (YYYY-MM-DD).
        
    Returns:
        Dictionary with alpha model data including:
        - security_count: Number of securities with scores
        - quintile_distribution: Count per quintile
        - top_alpha_securities: Top 20 by alpha score
        - scores: Dict of ticker -> alpha_score
    """
    repo = _get_repos()['alpha']()
    alpha_model = await repo.get_alpha_model()
    
    if not alpha_model:
        return {"error": "No alpha model data found"}
    
    # Count by quintile
    quintile_counts = {}
    for score in alpha_model.scores:
        q = score.alpha_quintile
        quintile_counts[q] = quintile_counts.get(q, 0) + 1
    
    # Get top securities
    top_scores = sorted(alpha_model.scores, key=lambda x: x.alpha_score, reverse=True)[:20]
    
    return {
        "security_count": alpha_model.security_count,
        "quintile_distribution": quintile_counts,
        "top_alpha_securities": [
            {"ticker": s.ticker, "alpha_score": s.alpha_score, "quintile": s.alpha_quintile, "sector": s.gics_sector}
            for s in top_scores
        ],
        "scores": {s.ticker: s.alpha_score for s in alpha_model.scores},
        "quintiles": {s.ticker: s.alpha_quintile for s in alpha_model.scores},
    }


@tool
async def load_risk_model(as_of_date: Optional[str] = None) -> dict:
    """Load Barra-style multi-factor risk model.
    
    The risk model contains:
    - Factor loadings for each security (8 factors)
    - Factor covariance matrix (8x8)
    - Specific (idiosyncratic) risk per security
    
    Risk Factors:
    1. Market - Broad market beta
    2. Size - Market cap factor  
    3. Value - Value vs growth
    4. Momentum - Price momentum
    5. Quality - Earnings quality
    6. Volatility - Stock volatility
    7. Growth - Revenue/earnings growth
    8. Dividend Yield - Dividend characteristics
    
    Use this tool when you need:
    - Factor exposures for risk calculation
    - Covariance matrix for portfolio variance
    - Security-specific risk estimates
    
    Args:
        as_of_date: Optional date string (YYYY-MM-DD).
        
    Returns:
        Dictionary with risk model data including:
        - security_count: Number of securities
        - factors: List of factor names
        - has_covariance: Whether covariance matrix is available
        - sample_loadings: Factor loadings for first 5 securities
    """
    repo = _get_repos()['risk']()
    risk_model = await repo.get_risk_model()
    
    if not risk_model:
        return {"error": "No risk model data found"}
    
    factors = ["Market", "Size", "Value", "Momentum", "Quality", "Volatility", "Growth", "Dividend_Yield"]
    
    sample_loadings = {}
    for loading in risk_model.factor_loadings[:5]:
        sample_loadings[loading.ticker] = {
            "market": loading.market_loading,
            "size": loading.size_loading,
            "value": loading.value_loading,
            "momentum": loading.momentum_loading,
            "specific_risk": loading.specific_risk_pct,
        }
    
    return {
        "security_count": risk_model.security_count,
        "factors": factors,
        "has_covariance": risk_model.factor_covariance is not None,
        "sample_loadings": sample_loadings,
    }


@tool
async def load_constraints(as_of_date: Optional[str] = None) -> dict:
    """Load optimization constraints.
    
    Constraints define position limits:
    - Single Stock: ±1% active weight vs benchmark
      (If AAPL is 6.7% of benchmark, portfolio can hold 5.7%-7.7%)
    - Sector: ±2% active weight vs benchmark
      (If Tech is 32% of benchmark, portfolio can have 30%-34%)
    
    Use this tool when you need:
    - Position limits for optimization
    - Compliance thresholds
    
    Args:
        as_of_date: Optional date string (YYYY-MM-DD).
        
    Returns:
        Dictionary with constraints including:
        - stock_constraints: List of stock-level constraints
        - sector_constraints: List of sector-level constraints
    """
    repo = _get_repos()['constraint']()
    constraint_set = await repo.get_constraint_set()
    
    if not constraint_set:
        return {"error": "No constraints found"}
    
    return {
        "stock_constraints": [
            {"type": c.constraint_type, "upper_pct": c.upper_bound_pct, "lower_pct": c.lower_bound_pct}
            for c in constraint_set.stock_constraints
        ],
        "sector_constraints": [
            {"sector": c.constraint_name, "upper_pct": c.upper_bound_pct, "lower_pct": c.lower_bound_pct}
            for c in constraint_set.sector_constraints
        ],
    }


@tool
async def load_transaction_costs(as_of_date: Optional[str] = None) -> dict:
    """Load transaction cost estimates.
    
    Transaction costs include:
    - Bid-ask spread (half spread for one-way)
    - Commission fees
    - Market impact (price movement from trading)
    
    Cost Calculation:
    Total One-Way Cost = Spread/2 + Commission + Impact × Trade_Size
    
    Urgency adjustments:
    - Low urgency: Cost × 0.70 (patient execution)
    - Medium urgency: Cost × 1.00 (normal)
    - High urgency: Cost × 1.50 (aggressive)
    
    Use this tool when you need:
    - Trading cost estimates for rebalancing
    - Liquidity information
    
    Args:
        as_of_date: Optional date string (YYYY-MM-DD).
        
    Returns:
        Dictionary with transaction costs including:
        - security_count: Number of securities
        - average_cost_bps: Average one-way cost in basis points
        - costs_by_ticker: Dict of ticker -> cost in bps
    """
    repo = _get_repos()['transaction_cost']()
    tcost_model = await repo.get_transaction_cost_model()
    
    if not tcost_model:
        return {"error": "No transaction cost data found"}
    
    costs_dict = {c.ticker: c.total_oneway_cost_bps for c in tcost_model.costs}
    avg_cost = sum(costs_dict.values()) / len(costs_dict) if costs_dict else 0
    
    return {
        "security_count": tcost_model.security_count,
        "average_cost_bps": round(avg_cost, 2),
        "costs_by_ticker": costs_dict,
    }


# Grouped tool lists for different agents
DATA_AGENT_TOOLS = [load_benchmark, load_alpha_scores, load_risk_model, load_constraints, load_transaction_costs]
ALPHA_AGENT_TOOLS = [load_alpha_scores, load_benchmark]
RISK_AGENT_TOOLS = [load_risk_model]

