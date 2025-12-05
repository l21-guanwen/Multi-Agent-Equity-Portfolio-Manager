"""
Compliance service implementation.

Handles constraint validation and compliance checking for portfolios.
"""

from typing import Any, Optional

from app.models.constraint import ConstraintSet
from app.services.interfaces.compliance_service import (
    ComplianceCheck,
    ComplianceReport,
    ComplianceViolation,
    IComplianceService,
)


class ComplianceService(IComplianceService):
    """
    Service for compliance operations.
    
    Handles constraint validation, compliance checking,
    and violation analysis for portfolio management.
    """

    async def check_portfolio_compliance(
        self,
        portfolio_weights: dict[str, float],
        benchmark_weights: dict[str, float],
        constraint_set: ConstraintSet,
        sector_mapping: dict[str, str],
    ) -> ComplianceReport:
        """Check full portfolio compliance against all constraints."""
        # Check stock constraints
        stock_violations = await self.check_stock_constraints(
            portfolio_weights, benchmark_weights, constraint_set
        )
        
        # Check sector constraints
        sector_violations = await self.check_sector_constraints(
            portfolio_weights, benchmark_weights, constraint_set, sector_mapping
        )
        
        all_violations = stock_violations + sector_violations
        
        # Calculate max breaches
        max_stock_breach = 0.0
        max_sector_breach = 0.0
        
        for v in stock_violations:
            max_stock_breach = max(max_stock_breach, abs(v.breach_amount))
        for v in sector_violations:
            max_sector_breach = max(max_sector_breach, abs(v.breach_amount))
        
        # Calculate total active risk (simplified)
        active_weights = await self.calculate_active_weights(
            portfolio_weights, benchmark_weights
        )
        total_active_risk = sum(abs(w) for w in active_weights.values())
        
        # Generate remediation actions
        remediation = await self.suggest_remediation(
            all_violations, portfolio_weights, benchmark_weights
        )
        remediation_actions = [r["action"] for r in remediation]
        
        return ComplianceReport(
            is_compliant=len(all_violations) == 0,
            total_violations=len(all_violations),
            stock_violations=len(stock_violations),
            sector_violations=len(sector_violations),
            violations=all_violations,
            max_stock_breach=max_stock_breach,
            max_sector_breach=max_sector_breach,
            total_active_risk=total_active_risk,
            remediation_actions=remediation_actions,
        )

    async def check_stock_constraints(
        self,
        portfolio_weights: dict[str, float],
        benchmark_weights: dict[str, float],
        constraint_set: ConstraintSet,
    ) -> list[ComplianceViolation]:
        """Check stock-level constraint compliance."""
        violations = []
        
        for constraint in constraint_set.stock_constraints:
            ticker = constraint.constraint_name
            portfolio_weight = portfolio_weights.get(ticker, 0.0)
            benchmark_weight = benchmark_weights.get(ticker, 0.0)
            active_weight = portfolio_weight - benchmark_weight
            
            min_allowed = constraint.min_weight_pct
            max_allowed = constraint.max_weight_pct
            
            breach_amount = 0.0
            if portfolio_weight < min_allowed:
                breach_amount = min_allowed - portfolio_weight
            elif portfolio_weight > max_allowed:
                breach_amount = portfolio_weight - max_allowed
            
            if breach_amount > 0.001:  # Small tolerance
                severity = await self.classify_violation_severity(breach_amount, "stock")
                
                violations.append(ComplianceViolation(
                    violation_type="stock",
                    name=ticker,
                    current_weight=portfolio_weight,
                    benchmark_weight=benchmark_weight,
                    active_weight=active_weight,
                    min_allowed=min_allowed,
                    max_allowed=max_allowed,
                    breach_amount=breach_amount,
                    severity=severity,
                ))
        
        return violations

    async def check_sector_constraints(
        self,
        portfolio_weights: dict[str, float],
        benchmark_weights: dict[str, float],
        constraint_set: ConstraintSet,
        sector_mapping: dict[str, str],
    ) -> list[ComplianceViolation]:
        """Check sector-level constraint compliance."""
        violations = []
        
        # Calculate sector weights
        portfolio_sector_weights = await self.calculate_sector_weights(
            portfolio_weights, sector_mapping
        )
        benchmark_sector_weights = await self.calculate_sector_weights(
            benchmark_weights, sector_mapping
        )
        
        for constraint in constraint_set.sector_constraints:
            sector = constraint.constraint_name
            portfolio_weight = portfolio_sector_weights.get(sector, 0.0)
            benchmark_weight = benchmark_sector_weights.get(sector, 0.0)
            active_weight = portfolio_weight - benchmark_weight
            
            min_allowed = constraint.min_weight_pct
            max_allowed = constraint.max_weight_pct
            
            breach_amount = 0.0
            if portfolio_weight < min_allowed:
                breach_amount = min_allowed - portfolio_weight
            elif portfolio_weight > max_allowed:
                breach_amount = portfolio_weight - max_allowed
            
            if breach_amount > 0.001:  # Small tolerance
                severity = await self.classify_violation_severity(breach_amount, "sector")
                
                violations.append(ComplianceViolation(
                    violation_type="sector",
                    name=sector,
                    current_weight=portfolio_weight,
                    benchmark_weight=benchmark_weight,
                    active_weight=active_weight,
                    min_allowed=min_allowed,
                    max_allowed=max_allowed,
                    breach_amount=breach_amount,
                    severity=severity,
                ))
        
        return violations

    async def calculate_active_weights(
        self,
        portfolio_weights: dict[str, float],
        benchmark_weights: dict[str, float],
    ) -> dict[str, float]:
        """Calculate active weights vs benchmark."""
        all_tickers = set(portfolio_weights.keys()) | set(benchmark_weights.keys())
        
        return {
            ticker: portfolio_weights.get(ticker, 0.0) - benchmark_weights.get(ticker, 0.0)
            for ticker in all_tickers
        }

    async def calculate_sector_weights(
        self,
        weights: dict[str, float],
        sector_mapping: dict[str, str],
    ) -> dict[str, float]:
        """Aggregate weights by sector."""
        sector_weights: dict[str, float] = {}
        
        for ticker, weight in weights.items():
            sector = sector_mapping.get(ticker, "Unknown")
            sector_weights[sector] = sector_weights.get(sector, 0.0) + weight
        
        return sector_weights

    async def suggest_remediation(
        self,
        violations: list[ComplianceViolation],
        portfolio_weights: dict[str, float],
        benchmark_weights: dict[str, float],
    ) -> list[dict[str, Any]]:
        """Suggest actions to fix compliance violations."""
        suggestions = []
        
        for violation in violations:
            if violation.current_weight > violation.max_allowed:
                # Over-weight - need to sell
                target_weight = (violation.min_allowed + violation.max_allowed) / 2
                trade_amount = violation.current_weight - target_weight
                
                suggestions.append({
                    "violation": violation.name,
                    "action": f"Reduce {violation.name} by {trade_amount:.2f}%",
                    "direction": "sell",
                    "amount_pct": trade_amount,
                    "target_weight": target_weight,
                })
            else:
                # Under-weight - need to buy
                target_weight = (violation.min_allowed + violation.max_allowed) / 2
                trade_amount = target_weight - violation.current_weight
                
                suggestions.append({
                    "violation": violation.name,
                    "action": f"Increase {violation.name} by {trade_amount:.2f}%",
                    "direction": "buy",
                    "amount_pct": trade_amount,
                    "target_weight": target_weight,
                })
        
        return suggestions

    async def run_compliance_checks(
        self,
        portfolio_weights: dict[str, float],
        benchmark_weights: dict[str, float],
    ) -> list[ComplianceCheck]:
        """Run a series of compliance checks."""
        checks = []
        
        # Check 1: Total weight sums to 100%
        total_weight = sum(portfolio_weights.values())
        checks.append(ComplianceCheck(
            check_name="Total Weight",
            passed=abs(total_weight - 100.0) < 0.1,
            details=f"Portfolio weights sum to {total_weight:.2f}%",
            value=total_weight,
            threshold=100.0,
        ))
        
        # Check 2: No negative weights (long-only)
        negative_weights = [t for t, w in portfolio_weights.items() if w < -0.001]
        checks.append(ComplianceCheck(
            check_name="Long Only",
            passed=len(negative_weights) == 0,
            details=f"{len(negative_weights)} securities with negative weights" if negative_weights else "All weights non-negative",
            value=len(negative_weights),
            threshold=0,
        ))
        
        # Check 3: No weight exceeds max position size (e.g., 10%)
        max_position = 10.0
        over_limit = [t for t, w in portfolio_weights.items() if w > max_position]
        checks.append(ComplianceCheck(
            check_name="Max Position Size",
            passed=len(over_limit) == 0,
            details=f"{len(over_limit)} positions exceed {max_position}% limit" if over_limit else f"All positions under {max_position}%",
            value=max(portfolio_weights.values()) if portfolio_weights else 0,
            threshold=max_position,
        ))
        
        # Check 4: Number of holdings within limits
        num_holdings = len([w for w in portfolio_weights.values() if w > 0.01])
        min_holdings = 10
        max_holdings = 50
        checks.append(ComplianceCheck(
            check_name="Number of Holdings",
            passed=min_holdings <= num_holdings <= max_holdings,
            details=f"Portfolio has {num_holdings} holdings",
            value=num_holdings,
            threshold=max_holdings,
        ))
        
        return checks

    async def classify_violation_severity(
        self,
        breach_amount: float,
        constraint_type: str,
    ) -> str:
        """Classify severity of a constraint violation."""
        if constraint_type == "stock":
            if breach_amount < 0.25:
                return "minor"
            elif breach_amount < 0.5:
                return "moderate"
            else:
                return "severe"
        else:  # sector
            if breach_amount < 0.5:
                return "minor"
            elif breach_amount < 1.0:
                return "moderate"
            else:
                return "severe"

