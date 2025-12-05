"""
Data service implementation.

Handles data loading, validation, and aggregation from repositories.
"""

from datetime import date
from typing import Any, Optional

from app.models.alpha import AlphaModel
from app.models.benchmark import Benchmark
from app.models.constraint import ConstraintSet
from app.models.risk import RiskModel
from app.models.transaction_cost import TransactionCostModel
from app.repositories.interfaces.alpha_repository import IAlphaRepository
from app.repositories.interfaces.benchmark_repository import IBenchmarkRepository
from app.repositories.interfaces.constraint_repository import IConstraintRepository
from app.repositories.interfaces.risk_repository import IRiskRepository
from app.repositories.interfaces.transaction_cost_repository import ITransactionCostRepository
from app.repositories.interfaces.universe_repository import IUniverseRepository
from app.services.interfaces.data_service import (
    DataSummary,
    DataValidationResult,
    IDataService,
)


class DataService(IDataService):
    """
    Service for data management operations.
    
    Coordinates data loading from multiple repositories,
    validates data consistency, and provides aggregated data
    for portfolio construction.
    """

    def __init__(
        self,
        benchmark_repo: IBenchmarkRepository,
        universe_repo: IUniverseRepository,
        alpha_repo: IAlphaRepository,
        risk_repo: IRiskRepository,
        constraint_repo: IConstraintRepository,
        transaction_cost_repo: ITransactionCostRepository,
    ):
        """
        Initialize the data service.
        
        Args:
            benchmark_repo: Repository for benchmark data
            universe_repo: Repository for universe data
            alpha_repo: Repository for alpha model data
            risk_repo: Repository for risk model data
            constraint_repo: Repository for constraint data
            transaction_cost_repo: Repository for transaction cost data
        """
        self._benchmark_repo = benchmark_repo
        self._universe_repo = universe_repo
        self._alpha_repo = alpha_repo
        self._risk_repo = risk_repo
        self._constraint_repo = constraint_repo
        self._transaction_cost_repo = transaction_cost_repo

    async def load_all_data(
        self,
        as_of_date: Optional[date] = None,
    ) -> dict[str, Any]:
        """
        Load all required data for portfolio construction.
        
        Returns dictionary with:
        - benchmark: Benchmark object
        - universe: List of securities
        - alpha_model: AlphaModel object
        - risk_model: RiskModel object
        - constraints: ConstraintSet object
        - transaction_costs: TransactionCostModel object
        """
        # Load all data in parallel conceptually (async)
        benchmark = await self._benchmark_repo.get_benchmark(as_of_date=as_of_date)
        universe = await self._universe_repo.get_all(as_of_date=as_of_date)
        alpha_model = await self._alpha_repo.get_alpha_model(as_of_date=as_of_date)
        risk_model = await self._risk_repo.get_risk_model(as_of_date=as_of_date)
        constraints = await self._constraint_repo.get_constraint_set(as_of_date=as_of_date)
        transaction_costs = await self._transaction_cost_repo.get_transaction_cost_model(
            as_of_date=as_of_date
        )
        
        return {
            "benchmark": benchmark,
            "universe": universe,
            "alpha_model": alpha_model,
            "risk_model": risk_model,
            "constraints": constraints,
            "transaction_costs": transaction_costs,
            "as_of_date": as_of_date or (benchmark.as_of_date if benchmark else None),
        }

    async def validate_data(
        self,
        data: dict[str, Any],
    ) -> DataValidationResult:
        """
        Validate loaded data for completeness and consistency.
        """
        issues: list[str] = []
        warnings: list[str] = []
        missing_fields: list[str] = []
        
        # Check required datasets
        if data.get("benchmark") is None:
            issues.append("Benchmark data is missing")
            missing_fields.append("benchmark")
        
        if data.get("alpha_model") is None:
            issues.append("Alpha model data is missing")
            missing_fields.append("alpha_model")
        
        if data.get("risk_model") is None:
            issues.append("Risk model data is missing")
            missing_fields.append("risk_model")
        
        if data.get("constraints") is None:
            warnings.append("Constraint data is missing - using defaults")
        
        if data.get("transaction_costs") is None:
            warnings.append("Transaction cost data is missing - using defaults")
        
        # Validate data consistency
        benchmark: Optional[Benchmark] = data.get("benchmark")
        alpha_model: Optional[AlphaModel] = data.get("alpha_model")
        risk_model: Optional[RiskModel] = data.get("risk_model")
        
        if benchmark and alpha_model:
            benchmark_tickers = {c.ticker for c in benchmark.constituents}
            alpha_tickers = {s.ticker for s in alpha_model.scores}
            
            # Check for missing alpha scores
            missing_alpha = benchmark_tickers - alpha_tickers
            if missing_alpha:
                warnings.append(
                    f"{len(missing_alpha)} benchmark securities missing alpha scores"
                )
        
        if benchmark and risk_model:
            benchmark_tickers = {c.ticker for c in benchmark.constituents}
            risk_tickers = {fl.ticker for fl in risk_model.factor_loadings}
            
            # Check for missing risk data
            missing_risk = benchmark_tickers - risk_tickers
            if missing_risk:
                warnings.append(
                    f"{len(missing_risk)} benchmark securities missing risk data"
                )
        
        # Calculate record count
        record_count = 0
        if benchmark:
            record_count = benchmark.security_count
        
        # Calculate data quality score
        quality_score = 1.0
        if issues:
            quality_score -= 0.3 * len(issues)
        if warnings:
            quality_score -= 0.1 * len(warnings)
        quality_score = max(0.0, quality_score)
        
        return DataValidationResult(
            is_valid=len(issues) == 0,
            issues=issues,
            warnings=warnings,
            record_count=record_count,
            missing_fields=missing_fields,
            data_quality_score=quality_score,
        )

    async def get_data_summary(
        self,
        as_of_date: Optional[date] = None,
    ) -> DataSummary:
        """Get a summary of available data."""
        # Get counts from each repository
        benchmark = await self._benchmark_repo.get_benchmark(as_of_date=as_of_date)
        universe = await self._universe_repo.get_all(as_of_date=as_of_date)
        alpha_model = await self._alpha_repo.get_alpha_model(as_of_date=as_of_date)
        risk_model = await self._risk_repo.get_risk_model(as_of_date=as_of_date)
        constraints = await self._constraint_repo.get_constraint_set(as_of_date=as_of_date)
        transaction_costs = await self._transaction_cost_repo.get_transaction_cost_model(
            as_of_date=as_of_date
        )
        
        # Build data dict for validation
        data = {
            "benchmark": benchmark,
            "universe": universe,
            "alpha_model": alpha_model,
            "risk_model": risk_model,
            "constraints": constraints,
            "transaction_costs": transaction_costs,
        }
        
        validation_result = await self.validate_data(data)
        
        return DataSummary(
            benchmark_count=benchmark.security_count if benchmark else 0,
            universe_count=len(universe) if universe else 0,
            alpha_count=alpha_model.security_count if alpha_model else 0,
            factor_loadings_count=risk_model.security_count if risk_model else 0,
            constraints_count=constraints.total_constraints if constraints else 0,
            transaction_costs_count=transaction_costs.security_count if transaction_costs else 0,
            as_of_date=as_of_date or (benchmark.as_of_date if benchmark else None),
            validation_result=validation_result,
        )

    async def check_data_availability(
        self,
        as_of_date: Optional[date] = None,
    ) -> dict[str, bool]:
        """Check which data sources are available."""
        availability = {
            "benchmark": False,
            "universe": False,
            "alpha_model": False,
            "risk_model": False,
            "constraints": False,
            "transaction_costs": False,
        }
        
        try:
            benchmark = await self._benchmark_repo.get_benchmark(as_of_date=as_of_date)
            availability["benchmark"] = benchmark is not None and benchmark.security_count > 0
        except Exception:
            pass
        
        try:
            universe = await self._universe_repo.get_all(as_of_date=as_of_date)
            availability["universe"] = len(universe) > 0
        except Exception:
            pass
        
        try:
            alpha = await self._alpha_repo.get_alpha_model(as_of_date=as_of_date)
            availability["alpha_model"] = alpha is not None and alpha.security_count > 0
        except Exception:
            pass
        
        try:
            risk = await self._risk_repo.get_risk_model(as_of_date=as_of_date)
            availability["risk_model"] = risk is not None and risk.security_count > 0
        except Exception:
            pass
        
        try:
            constraints = await self._constraint_repo.get_constraint_set(as_of_date=as_of_date)
            availability["constraints"] = constraints is not None and constraints.total_constraints > 0
        except Exception:
            pass
        
        try:
            tcosts = await self._transaction_cost_repo.get_transaction_cost_model(as_of_date=as_of_date)
            availability["transaction_costs"] = tcosts is not None and tcosts.security_count > 0
        except Exception:
            pass
        
        return availability

    async def get_common_tickers(
        self,
        as_of_date: Optional[date] = None,
    ) -> list[str]:
        """Get tickers that exist across all data sources."""
        # Get tickers from each source
        benchmark = await self._benchmark_repo.get_benchmark(as_of_date=as_of_date)
        alpha_model = await self._alpha_repo.get_alpha_model(as_of_date=as_of_date)
        risk_model = await self._risk_repo.get_risk_model(as_of_date=as_of_date)
        
        # Start with benchmark tickers
        if benchmark is None:
            return []
        
        common_tickers = {c.ticker for c in benchmark.constituents}
        
        # Intersect with alpha tickers
        if alpha_model:
            alpha_tickers = {s.ticker for s in alpha_model.scores}
            common_tickers = common_tickers.intersection(alpha_tickers)
        
        # Intersect with risk tickers
        if risk_model:
            risk_tickers = {fl.ticker for fl in risk_model.factor_loadings}
            common_tickers = common_tickers.intersection(risk_tickers)
        
        return sorted(list(common_tickers))

