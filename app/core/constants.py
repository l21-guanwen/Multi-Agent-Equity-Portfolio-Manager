"""
Constants used throughout the application.

Includes GICS sectors, risk factors, and other domain constants.
"""

from enum import Enum
from typing import Final


# ===========================================
# GICS Sector Classification
# ===========================================
class GICSSector(str, Enum):
    """GICS Level 1 Sector Classification."""
    
    INFORMATION_TECHNOLOGY = "Information Technology"
    HEALTH_CARE = "Health Care"
    FINANCIALS = "Financials"
    CONSUMER_DISCRETIONARY = "Consumer Discretionary"
    COMMUNICATION_SERVICES = "Communication Services"
    INDUSTRIALS = "Industrials"
    CONSUMER_STAPLES = "Consumer Staples"
    ENERGY = "Energy"
    UTILITIES = "Utilities"
    REAL_ESTATE = "Real Estate"
    MATERIALS = "Materials"


GICS_SECTORS: Final[list[str]] = [sector.value for sector in GICSSector]


# ===========================================
# Risk Factors (Barra-Style)
# ===========================================
class RiskFactor(str, Enum):
    """Barra-style risk factors."""
    
    MARKET = "Market"
    SIZE = "Size"
    VALUE = "Value"
    MOMENTUM = "Momentum"
    QUALITY = "Quality"
    VOLATILITY = "Volatility"
    GROWTH = "Growth"
    DIVIDEND_YIELD = "Dividend_Yield"


RISK_FACTORS: Final[list[str]] = [factor.value for factor in RiskFactor]

# Factor loading column names in CSV
FACTOR_LOADING_COLUMNS: Final[list[str]] = [
    "Market_Loading",
    "Size_Loading",
    "Value_Loading",
    "Momentum_Loading",
    "Quality_Loading",
    "Volatility_Loading",
    "Growth_Loading",
    "Dividend_Yield_Loading",
]


# ===========================================
# Alpha Model Constants
# ===========================================
class AlphaQuintile(int, Enum):
    """Alpha quintile rankings (1 = best, 5 = worst)."""
    
    Q1 = 1  # Top 20% (Best)
    Q2 = 2  # 20-40%
    Q3 = 3  # 40-60%
    Q4 = 4  # 60-80%
    Q5 = 5  # Bottom 20% (Worst)


ALPHA_QUINTILES: Final[dict[int, str]] = {
    1: "Top 20% (Best)",
    2: "20-40%",
    3: "40-60%",
    4: "60-80%",
    5: "Bottom 20% (Worst)",
}

# Alpha score ranges by quintile
ALPHA_SCORE_RANGES: Final[dict[int, tuple[float, float]]] = {
    1: (0.80, 1.00),
    2: (0.60, 0.80),
    3: (0.40, 0.60),
    4: (0.20, 0.40),
    5: (0.00, 0.20),
}


# ===========================================
# Signal Strength Classification
# ===========================================
class SignalStrength(str, Enum):
    """Alpha signal strength categories."""
    
    VERY_STRONG = "Very Strong"
    STRONG = "Strong"
    NEUTRAL = "Neutral"
    WEAK = "Weak"
    VERY_WEAK = "Very Weak"


SIGNAL_STRENGTHS: Final[list[str]] = [strength.value for strength in SignalStrength]


# ===========================================
# Constraint Types
# ===========================================
class ConstraintType(str, Enum):
    """Optimization constraint types."""
    
    SECTOR = "Sector"
    STOCK = "Stock"


class ConstraintTypeCode(str, Enum):
    """Constraint bound types."""
    
    RELATIVE = "REL"  # Relative to benchmark
    ABSOLUTE = "ABS"  # Absolute bounds


# ===========================================
# Liquidity Buckets
# ===========================================
class LiquidityBucket(int, Enum):
    """Liquidity classification buckets."""
    
    MOST_LIQUID = 1
    HIGH_LIQUIDITY = 2
    MEDIUM_LIQUIDITY = 3
    LOW_LIQUIDITY = 4
    LEAST_LIQUID = 5


LIQUIDITY_BUCKETS: Final[dict[int, str]] = {
    1: "Most Liquid",
    2: "High Liquidity",
    3: "Medium Liquidity",
    4: "Low Liquidity",
    5: "Least Liquid",
}


# ===========================================
# Benchmark Constants
# ===========================================
BENCHMARK_ID: Final[str] = "SPX"
BENCHMARK_NAME: Final[str] = "S&P 500"
BENCHMARK_SECURITIES_COUNT: Final[int] = 500


# ===========================================
# Portfolio Constants
# ===========================================
DEFAULT_PORTFOLIO_SIZE: Final[int] = 25
DEFAULT_STOCK_ACTIVE_WEIGHT_LIMIT: Final[float] = 0.01  # ±1%
DEFAULT_SECTOR_ACTIVE_WEIGHT_LIMIT: Final[float] = 0.02  # ±2%


# ===========================================
# Optimization Constants
# ===========================================
DEFAULT_RISK_AVERSION: Final[float] = 0.01
DEFAULT_ALPHA_COEFFICIENT: Final[float] = 1.0
DEFAULT_TCOST_COEFFICIENT: Final[float] = 0.0


# ===========================================
# File Names
# ===========================================
class DataFileName(str, Enum):
    """Standard data file names."""
    
    BENCHMARK = "01_SP500_Benchmark_Constituency.csv"
    UNIVERSE = "02_SP500_Universe.csv"
    PORTFOLIO = "03_Portfolio_25_Holdings.csv"
    ALPHA = "04_Alpha_Model_SP500.csv"
    FACTOR_LOADINGS = "05_Risk_Model_Factor_Loadings.csv"
    FACTOR_RETURNS = "06_Risk_Model_Factor_Returns.csv"
    FACTOR_COVARIANCE = "07_Risk_Model_Factor_Covariance.csv"
    CONSTRAINTS = "08_Optimization_Constraints.csv"
    TRANSACTION_COSTS = "09_Transaction_Cost_Model.csv"

