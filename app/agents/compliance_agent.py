"""
Compliance Agent for the multi-agent portfolio management system.

Responsible for constraint validation and compliance checking.
"""

from typing import Any, Optional

from app.agents.prompts import COMPLIANCE_AGENT_SYSTEM_PROMPT
from app.agents.state import PortfolioState
from app.llm.interfaces.llm_provider import ILLMProvider
from app.repositories.interfaces.constraint_repository import IConstraintRepository
from app.services.interfaces.compliance_service import IComplianceService


class ComplianceAgent:
    """
    Compliance Agent responsible for constraint validation.
    
    This agent:
    1. Checks portfolio weights against all constraints
    2. Identifies and classifies violations
    3. Uses LLM to analyze violations and suggest remediation
    """

    def __init__(
        self,
        compliance_service: IComplianceService,
        constraint_repository: IConstraintRepository,
        llm_provider: Optional[ILLMProvider] = None,
    ):
        """
        Initialize the Compliance Agent.
        
        Args:
            compliance_service: Service for compliance checking
            constraint_repository: Repository for constraint data
            llm_provider: Optional LLM provider for analysis
        """
        self._compliance_service = compliance_service
        self._constraint_repo = constraint_repository
        self._llm = llm_provider

    async def __call__(self, state: PortfolioState) -> dict[str, Any]:
        """
        Execute the compliance agent.
        
        Args:
            state: Current portfolio state
            
        Returns:
            Updated state fields
        """
        execution_log = [f"[ComplianceAgent] Starting compliance check..."]
        
        try:
            # Check prerequisites
            if not state.optimal_weights:
                execution_log.append("[ComplianceAgent] Skipped - no weights to check")
                return {
                    "is_compliant": False,
                    "execution_log": execution_log,
                    "current_agent": "compliance_agent",
                }
            
            # Load constraints
            constraint_set = await self._constraint_repo.get_constraint_set()
            
            if not constraint_set:
                execution_log.append("[ComplianceAgent] No constraints - assuming compliant")
                return {
                    "is_compliant": True,
                    "compliance_violations": [],
                    "execution_log": execution_log,
                    "current_agent": "compliance_agent",
                }
            
            # Convert weights to percentage for compliance check
            portfolio_weights_pct = {
                t: w * 100 for t, w in state.optimal_weights.items()
            }
            
            # Run compliance check
            report = await self._compliance_service.check_portfolio_compliance(
                portfolio_weights=portfolio_weights_pct,
                benchmark_weights=state.benchmark_weights,
                constraint_set=constraint_set,
                sector_mapping=state.sector_mapping,
            )
            
            execution_log.append(
                f"[ComplianceAgent] Compliance: {'PASS' if report.is_compliant else 'FAIL'}"
            )
            
            if not report.is_compliant:
                execution_log.append(
                    f"[ComplianceAgent] Violations: {report.total_violations} "
                    f"(Stock: {report.stock_violations}, Sector: {report.sector_violations})"
                )
            
            # Convert violations to dict format
            violations_dict = [
                {
                    "type": v.violation_type,
                    "name": v.name,
                    "current_weight": v.current_weight,
                    "benchmark_weight": v.benchmark_weight,
                    "active_weight": v.active_weight,
                    "min_allowed": v.min_allowed,
                    "max_allowed": v.max_allowed,
                    "breach_amount": v.breach_amount,
                    "severity": v.severity,
                }
                for v in report.violations
            ]
            
            # Generate LLM analysis
            if self._llm:
                llm_analysis = await self._generate_analysis(report, state)
                execution_log.append(f"[ComplianceAgent] LLM analysis generated")
            else:
                llm_analysis = self._generate_basic_analysis(report, state)
            
            execution_log.append(f"[ComplianceAgent] Completed")
            
            # Build final portfolio if compliant
            final_portfolio = {}
            if report.is_compliant:
                final_portfolio = self._build_final_portfolio(state)
            
            return {
                "is_compliant": report.is_compliant,
                "compliance_violations": violations_dict,
                "compliance_analysis": llm_analysis,
                "final_portfolio": final_portfolio,
                "execution_log": execution_log,
                "current_agent": "compliance_agent",
            }
            
        except Exception as e:
            execution_log.append(f"[ComplianceAgent] ERROR: {str(e)}")
            return {
                "is_compliant": False,
                "error_message": f"Compliance check failed: {str(e)}",
                "execution_log": execution_log,
                "current_agent": "compliance_agent",
            }

    def _build_final_portfolio(self, state: PortfolioState) -> dict[str, Any]:
        """Build final portfolio summary."""
        holdings = []
        for ticker, weight in state.optimal_weights.items():
            if weight > 0.001:  # Filter small weights
                holdings.append({
                    "ticker": ticker,
                    "weight_pct": weight * 100,
                    "alpha_score": state.alpha_scores.get(ticker, 0.0),
                    "sector": state.sector_mapping.get(ticker, "Unknown"),
                    "benchmark_weight_pct": state.benchmark_weights.get(ticker, 0.0),
                    "active_weight_pct": (weight * 100) - state.benchmark_weights.get(ticker, 0.0),
                })
        
        # Sort by weight
        holdings.sort(key=lambda x: x["weight_pct"], reverse=True)
        
        return {
            "portfolio_id": state.portfolio_id,
            "as_of_date": state.as_of_date,
            "holdings": holdings,
            "total_holdings": len(holdings),
            "expected_alpha": state.expected_alpha,
            "expected_risk_pct": state.portfolio_risk_pct,
        }

    async def _generate_analysis(
        self,
        report: Any,
        state: PortfolioState,
    ) -> str:
        """Generate LLM-powered compliance analysis."""
        violations_text = ""
        if report.violations:
            violations_text = "Violations:\n" + "\n".join(
                f"- {v.violation_type.upper()} {v.name}: "
                f"weight={v.current_weight:.2f}%, "
                f"allowed=[{v.min_allowed:.2f}%, {v.max_allowed:.2f}%], "
                f"breach={v.breach_amount:.2f}%, "
                f"severity={v.severity}"
                for v in report.violations[:10]
            )
        
        prompt = f"""Analyze the following portfolio compliance check results:

Compliance Status: {'COMPLIANT' if report.is_compliant else 'NON-COMPLIANT'}

Violation Summary:
- Total Violations: {report.total_violations}
- Stock Violations: {report.stock_violations}
- Sector Violations: {report.sector_violations}
- Max Stock Breach: {report.max_stock_breach:.2f}%
- Max Sector Breach: {report.max_sector_breach:.2f}%

{violations_text}

Constraint Limits:
- Stock: ±1% active weight vs benchmark
- Sector: ±2% active weight vs benchmark

{'Provide remediation recommendations.' if not report.is_compliant else 'Confirm portfolio is ready for execution.'}"""

        try:
            response = await self._llm.generate(
                prompt=prompt,
                system_prompt=COMPLIANCE_AGENT_SYSTEM_PROMPT,
                temperature=0.3,
                max_tokens=600,
            )
            return response.content
        except Exception:
            return self._generate_basic_analysis(report, state)

    def _generate_basic_analysis(
        self,
        report: Any,
        state: PortfolioState,
    ) -> str:
        """Generate basic compliance analysis without LLM."""
        status = "COMPLIANT" if report.is_compliant else "NON-COMPLIANT"
        
        analysis = f"""Compliance Analysis

Status: {status}
Total Violations: {report.total_violations}
- Stock Violations: {report.stock_violations}
- Sector Violations: {report.sector_violations}

"""
        
        if report.violations:
            analysis += "Top Violations:\n"
            for v in report.violations[:5]:
                analysis += f"- {v.violation_type} {v.name}: breach {v.breach_amount:.2f}% ({v.severity})\n"
            
            analysis += f"\nRemediation Required: {len(report.remediation_actions)} actions"
        else:
            analysis += "Portfolio meets all constraints and is ready for execution."
        
        return analysis

