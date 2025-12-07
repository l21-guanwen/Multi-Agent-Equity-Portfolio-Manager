"""
Data Agent for the multi-agent portfolio management system.

Responsible for loading and validating market data.
"""

from datetime import date, datetime
from typing import Any, Optional

from app.agents.prompts import DATA_AGENT_SYSTEM_PROMPT
from app.agents.state import PortfolioState
from app.llm.interfaces.llm_provider import ILLMProvider
from app.services.interfaces.data_service import IDataService


class DataAgent:
    """
    Data Agent responsible for data ingestion and validation.
    
    This agent:
    1. Loads all required data from repositories
    2. Validates data completeness and consistency
    3. Prepares data for downstream agents
    4. Uses LLM to analyze data quality
    """

    def __init__(
        self,
        data_service: IDataService,
        llm_provider: Optional[ILLMProvider] = None,
    ):
        """
        Initialize the Data Agent.
        
        Args:
            data_service: Service for data loading and validation
            llm_provider: Optional LLM provider for analysis
        """
        self._data_service = data_service
        self._llm = llm_provider

    async def __call__(self, state: PortfolioState) -> dict[str, Any]:
        """
        Execute the data agent.
        
        Args:
            state: Current portfolio state
            
        Returns:
            Updated state fields
        """
        execution_log = [f"[DataAgent] Starting data loading..."]
        
        try:
            # Parse as_of_date if provided
            as_of_date = None
            if state.as_of_date:
                as_of_date = datetime.strptime(state.as_of_date, "%Y-%m-%d").date()
            
            # Load all data
            data = await self._data_service.load_all_data(as_of_date)
            execution_log.append(f"[DataAgent] Data loaded successfully")
            
            # Validate data
            validation = await self._data_service.validate_data(data)
            execution_log.append(
                f"[DataAgent] Validation: {'PASSED' if validation.is_valid else 'FAILED'}"
            )
            
            # Extract data for state
            benchmark = data.get("benchmark")
            alpha_model = data.get("alpha_model")
            risk_model = data.get("risk_model")
            transaction_costs = data.get("transaction_costs")
            
            # Build state updates
            updates: dict[str, Any] = {
                "data_validation_passed": validation.is_valid,
                "data_validation_issues": validation.issues + validation.warnings,
                "execution_log": execution_log,
                "current_agent": "data_agent",
            }
            
            # Store transaction cost availability flag
            if transaction_costs:
                updates["execution_log"].append(
                    f"[DataAgent] Transaction costs loaded: {transaction_costs.security_count} securities"
                )
            
            # Extract universe tickers
            if benchmark:
                updates["universe_tickers"] = [c.ticker for c in benchmark.constituents]
                updates["benchmark_weights"] = {
                    c.ticker: c.benchmark_weight_pct for c in benchmark.constituents
                }
                updates["sector_mapping"] = {
                    c.ticker: c.gics_sector for c in benchmark.constituents
                }
                updates["as_of_date"] = benchmark.as_of_date.isoformat()
            
            # Extract alpha scores
            if alpha_model:
                updates["alpha_scores"] = {
                    s.ticker: s.alpha_score for s in alpha_model.scores
                }
                updates["alpha_quintiles"] = {
                    s.ticker: s.alpha_quintile for s in alpha_model.scores
                }
            
            # Build data summary
            updates["data_summary"] = {
                "benchmark_count": benchmark.security_count if benchmark else 0,
                "alpha_count": alpha_model.security_count if alpha_model else 0,
                "risk_count": risk_model.security_count if risk_model else 0,
                "validation_score": validation.data_quality_score,
            }
            
            # Generate LLM analysis if available
            if self._llm and validation.is_valid:
                analysis = await self._generate_analysis(updates, validation)
                updates["execution_log"].append(f"[DataAgent] LLM analysis generated")
            else:
                analysis = self._generate_basic_analysis(updates, validation)
            
            updates["execution_log"].append(f"[DataAgent] Completed")
            
            return updates
            
        except Exception as e:
            execution_log.append(f"[DataAgent] ERROR: {str(e)}")
            return {
                "data_validation_passed": False,
                "data_validation_issues": [str(e)],
                "error_message": f"Data loading failed: {str(e)}",
                "execution_log": execution_log,
                "current_agent": "data_agent",
            }

    async def _generate_analysis(
        self,
        data_summary: dict[str, Any],
        validation: Any,
    ) -> str:
        """Generate LLM-powered data analysis."""
        prompt = f"""Analyze the following portfolio data quality:

Data Summary:
- Benchmark securities: {data_summary.get('data_summary', {}).get('benchmark_count', 0)}
- Alpha scores: {data_summary.get('data_summary', {}).get('alpha_count', 0)}
- Risk model coverage: {data_summary.get('data_summary', {}).get('risk_count', 0)}
- Data quality score: {data_summary.get('data_summary', {}).get('validation_score', 0):.2f}

Validation Issues:
{chr(10).join(data_summary.get('data_validation_issues', [])) or 'None'}

Provide a brief assessment of data quality and readiness for portfolio construction."""

        try:
            response = await self._llm.generate(
                prompt=prompt,
                system_prompt=DATA_AGENT_SYSTEM_PROMPT,
                temperature=0.3,
                max_tokens=500,
            )
            return response.content
        except Exception:
            return self._generate_basic_analysis(data_summary, validation)

    def _generate_basic_analysis(
        self,
        data_summary: dict[str, Any],
        validation: Any,
    ) -> str:
        """Generate basic data analysis without LLM."""
        summary = data_summary.get("data_summary", {})
        issues = data_summary.get("data_validation_issues", [])
        
        status = "READY" if not issues else "WARNING"
        
        return f"""Data Quality Assessment: {status}
        
Loaded Data:
- Benchmark: {summary.get('benchmark_count', 0)} securities
- Alpha Model: {summary.get('alpha_count', 0)} scores
- Risk Model: {summary.get('risk_count', 0)} securities

Issues: {len(issues)}
{chr(10).join(f'- {i}' for i in issues[:5]) if issues else '- None'}

Data is {'ready' if not issues else 'available with warnings'} for portfolio construction."""

