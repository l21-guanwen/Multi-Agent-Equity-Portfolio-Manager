"""Optimization constraint domain models."""

from datetime import date
from typing import Optional

from pydantic import BaseModel, Field, computed_field


class Constraint(BaseModel):
    """
    A single optimization constraint.
    
    Based on 08_Optimization_Constraints.csv schema.
    Supports both stock-level and sector-level constraints.
    """

    # Constraint Definition
    constraint_type: str = Field(..., description="'Sector' or 'Stock'")
    constraint_name: str = Field(..., description="Sector name or ticker symbol")
    
    # Benchmark Reference
    benchmark_weight_pct: float = Field(..., ge=0, description="Current benchmark weight (%)")
    
    # Bounds
    lower_bound_pct: float = Field(..., description="Minimum relative bound (%)")
    upper_bound_pct: float = Field(..., description="Maximum relative bound (%)")
    
    # Constraint Type
    constraint_type_code: str = Field(default="REL", description="'REL' (relative) or 'ABS' (absolute)")
    is_hard_constraint: bool = Field(default=True, description="Hard vs soft constraint flag")
    penalty_coefficient: float = Field(default=0.0, ge=0, description="Soft constraint penalty")
    
    # Metadata
    optimization_definition_id: Optional[str] = Field(None, description="Optimization definition ID")

    @computed_field
    @property
    def is_sector_constraint(self) -> bool:
        """Check if this is a sector-level constraint."""
        return self.constraint_type.lower() == "sector"

    @computed_field
    @property
    def is_stock_constraint(self) -> bool:
        """Check if this is a stock-level constraint."""
        return self.constraint_type.lower() == "stock"

    @computed_field
    @property
    def is_relative(self) -> bool:
        """Check if this is a relative constraint."""
        return self.constraint_type_code.upper() == "REL"

    @computed_field
    @property
    def min_weight_pct(self) -> float:
        """Calculate minimum allowed weight (%)."""
        if self.is_relative:
            return self.benchmark_weight_pct + self.lower_bound_pct
        return self.lower_bound_pct

    @computed_field
    @property
    def max_weight_pct(self) -> float:
        """Calculate maximum allowed weight (%)."""
        if self.is_relative:
            return self.benchmark_weight_pct + self.upper_bound_pct
        return self.upper_bound_pct

    @computed_field
    @property
    def allowed_range_pct(self) -> tuple[float, float]:
        """Get allowed weight range as tuple (min, max)."""
        return (self.min_weight_pct, self.max_weight_pct)

    def check_compliance(self, weight_pct: float) -> tuple[bool, Optional[float]]:
        """
        Check if a weight complies with this constraint.
        
        Returns:
            Tuple of (is_compliant, violation_amount)
            violation_amount is None if compliant, otherwise the amount of breach
        """
        if weight_pct < self.min_weight_pct:
            return (False, self.min_weight_pct - weight_pct)
        elif weight_pct > self.max_weight_pct:
            return (False, weight_pct - self.max_weight_pct)
        return (True, None)

    class Config:
        """Pydantic model configuration."""
        
        frozen = False
        str_strip_whitespace = True


class ConstraintSet(BaseModel):
    """
    Collection of optimization constraints.
    
    Contains both stock-level and sector-level constraints.
    """

    optimization_definition_id: str = Field(..., description="Optimization definition ID")
    constraints: list[Constraint] = Field(default_factory=list)
    as_of_date: Optional[date] = Field(None, description="Constraint set date")

    @computed_field
    @property
    def total_constraints(self) -> int:
        """Total number of constraints."""
        return len(self.constraints)

    @computed_field
    @property
    def sector_constraints(self) -> list[Constraint]:
        """Get all sector-level constraints."""
        return [c for c in self.constraints if c.is_sector_constraint]

    @computed_field
    @property
    def stock_constraints(self) -> list[Constraint]:
        """Get all stock-level constraints."""
        return [c for c in self.constraints if c.is_stock_constraint]

    @computed_field
    @property
    def sector_count(self) -> int:
        """Number of sector constraints."""
        return len(self.sector_constraints)

    @computed_field
    @property
    def stock_count(self) -> int:
        """Number of stock constraints."""
        return len(self.stock_constraints)

    def get_constraint(self, name: str) -> Optional[Constraint]:
        """Get a constraint by name (sector or ticker)."""
        for constraint in self.constraints:
            if constraint.constraint_name == name:
                return constraint
        return None

    def get_sector_constraint(self, sector: str) -> Optional[Constraint]:
        """Get a sector constraint by sector name."""
        for constraint in self.sector_constraints:
            if constraint.constraint_name == sector:
                return constraint
        return None

    def get_stock_constraint(self, ticker: str) -> Optional[Constraint]:
        """Get a stock constraint by ticker."""
        for constraint in self.stock_constraints:
            if constraint.constraint_name == ticker:
                return constraint
        return None

    def check_portfolio_compliance(
        self,
        stock_weights: dict[str, float],
        sector_weights: dict[str, float],
    ) -> dict[str, list[dict]]:
        """
        Check full portfolio compliance against all constraints.
        
        Args:
            stock_weights: Dict of ticker -> weight percentage
            sector_weights: Dict of sector -> weight percentage
            
        Returns:
            Dict with 'violations' list and 'compliant' boolean
        """
        violations: list[dict] = []
        
        # Check sector constraints
        for constraint in self.sector_constraints:
            sector = constraint.constraint_name
            weight = sector_weights.get(sector, 0.0)
            is_compliant, breach_amount = constraint.check_compliance(weight)
            if not is_compliant:
                violations.append({
                    "type": "sector",
                    "name": sector,
                    "weight": weight,
                    "min_allowed": constraint.min_weight_pct,
                    "max_allowed": constraint.max_weight_pct,
                    "breach_amount": breach_amount,
                })
        
        # Check stock constraints
        for constraint in self.stock_constraints:
            ticker = constraint.constraint_name
            weight = stock_weights.get(ticker, 0.0)
            is_compliant, breach_amount = constraint.check_compliance(weight)
            if not is_compliant:
                violations.append({
                    "type": "stock",
                    "name": ticker,
                    "weight": weight,
                    "min_allowed": constraint.min_weight_pct,
                    "max_allowed": constraint.max_weight_pct,
                    "breach_amount": breach_amount,
                })
        
        return {
            "compliant": len(violations) == 0,
            "violations": violations,
            "violation_count": len(violations),
        }

