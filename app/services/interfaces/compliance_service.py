"""
Compliance service interface.

Defines operations for constraint validation and compliance checking.
"""

from abc import ABC, abstractmethod
from typing import Any, Optional

from pydantic import BaseModel

from app.models.constraint import ConstraintSet


class ComplianceViolation(BaseModel):
    """A single compliance violation."""
    
    violation_type: str  # 'stock' or 'sector'
    name: str  # Ticker or sector name
    current_weight: float
    benchmark_weight: float
    active_weight: float
    min_allowed: float
    max_allowed: float
    breach_amount: float
    severity: str  # 'minor', 'moderate', 'severe'


class ComplianceReport(BaseModel):
    """Comprehensive compliance report."""
    
    is_compliant: bool
    total_violations: int
    stock_violations: int
    sector_violations: int
    violations: list[ComplianceViolation]
    
    # Summary metrics
    max_stock_breach: float
    max_sector_breach: float
    total_active_risk: float
    
    # Recommendations
    remediation_actions: list[str]


class ComplianceCheck(BaseModel):
    """Result of a single compliance check."""
    
    check_name: str
    passed: bool
    details: str
    value: Optional[float] = None
    threshold: Optional[float] = None


class IComplianceService(ABC):
    """
    Service interface for compliance operations.
    
    Handles constraint validation, compliance checking,
    and violation analysis for portfolio management.
    """

    @abstractmethod
    async def check_portfolio_compliance(
        self,
        portfolio_weights: dict[str, float],
        benchmark_weights: dict[str, float],
        constraint_set: ConstraintSet,
        sector_mapping: dict[str, str],
    ) -> ComplianceReport:
        """
        Check full portfolio compliance against all constraints.
        
        Args:
            portfolio_weights: Dictionary of ticker -> portfolio weight (%)
            benchmark_weights: Dictionary of ticker -> benchmark weight (%)
            constraint_set: All constraints to check against
            sector_mapping: Dictionary of ticker -> sector
            
        Returns:
            ComplianceReport with all violations and recommendations
        """
        pass

    @abstractmethod
    async def check_stock_constraints(
        self,
        portfolio_weights: dict[str, float],
        benchmark_weights: dict[str, float],
        constraint_set: ConstraintSet,
    ) -> list[ComplianceViolation]:
        """
        Check stock-level constraint compliance.
        
        Args:
            portfolio_weights: Dictionary of ticker -> portfolio weight (%)
            benchmark_weights: Dictionary of ticker -> benchmark weight (%)
            constraint_set: Constraints to check against
            
        Returns:
            List of stock-level violations (empty if compliant)
        """
        pass

    @abstractmethod
    async def check_sector_constraints(
        self,
        portfolio_weights: dict[str, float],
        benchmark_weights: dict[str, float],
        constraint_set: ConstraintSet,
        sector_mapping: dict[str, str],
    ) -> list[ComplianceViolation]:
        """
        Check sector-level constraint compliance.
        
        Args:
            portfolio_weights: Dictionary of ticker -> portfolio weight (%)
            benchmark_weights: Dictionary of ticker -> benchmark weight (%)
            constraint_set: Constraints to check against
            sector_mapping: Dictionary of ticker -> sector
            
        Returns:
            List of sector-level violations (empty if compliant)
        """
        pass

    @abstractmethod
    async def calculate_active_weights(
        self,
        portfolio_weights: dict[str, float],
        benchmark_weights: dict[str, float],
    ) -> dict[str, float]:
        """
        Calculate active weights vs benchmark.
        
        Args:
            portfolio_weights: Dictionary of ticker -> portfolio weight (%)
            benchmark_weights: Dictionary of ticker -> benchmark weight (%)
            
        Returns:
            Dictionary of ticker -> active weight (%)
        """
        pass

    @abstractmethod
    async def calculate_sector_weights(
        self,
        weights: dict[str, float],
        sector_mapping: dict[str, str],
    ) -> dict[str, float]:
        """
        Aggregate weights by sector.
        
        Args:
            weights: Dictionary of ticker -> weight (%)
            sector_mapping: Dictionary of ticker -> sector
            
        Returns:
            Dictionary of sector -> total weight (%)
        """
        pass

    @abstractmethod
    async def suggest_remediation(
        self,
        violations: list[ComplianceViolation],
        portfolio_weights: dict[str, float],
        benchmark_weights: dict[str, float],
    ) -> list[dict[str, Any]]:
        """
        Suggest actions to fix compliance violations.
        
        Args:
            violations: List of current violations
            portfolio_weights: Current portfolio weights
            benchmark_weights: Benchmark weights
            
        Returns:
            List of suggested trades to achieve compliance
        """
        pass

    @abstractmethod
    async def run_compliance_checks(
        self,
        portfolio_weights: dict[str, float],
        benchmark_weights: dict[str, float],
    ) -> list[ComplianceCheck]:
        """
        Run a series of compliance checks.
        
        Checks include:
        - Total weight sums to 100%
        - No negative weights (long-only)
        - No weight exceeds max position size
        - Number of holdings within limits
        
        Args:
            portfolio_weights: Dictionary of ticker -> portfolio weight (%)
            benchmark_weights: Dictionary of ticker -> benchmark weight (%)
            
        Returns:
            List of ComplianceCheck results
        """
        pass

    @abstractmethod
    async def classify_violation_severity(
        self,
        breach_amount: float,
        constraint_type: str,
    ) -> str:
        """
        Classify severity of a constraint violation.
        
        Args:
            breach_amount: Amount of constraint breach (%)
            constraint_type: 'stock' or 'sector'
            
        Returns:
            Severity level: 'minor', 'moderate', or 'severe'
        """
        pass

