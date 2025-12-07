"""
Dependency injection for FastAPI.

Provides factory functions for creating service and repository instances.
"""

from functools import lru_cache
from typing import Optional

from app.core.config import Settings, get_settings

# Repositories
from app.repositories.csv.csv_benchmark_repository import CSVBenchmarkRepository
from app.repositories.csv.csv_universe_repository import CSVUniverseRepository
from app.repositories.csv.csv_alpha_repository import CSVAlphaRepository
from app.repositories.csv.csv_risk_repository import CSVRiskRepository
from app.repositories.csv.csv_constraint_repository import CSVConstraintRepository
from app.repositories.csv.csv_transaction_cost_repository import CSVTransactionCostRepository
from app.repositories.interfaces.benchmark_repository import IBenchmarkRepository
from app.repositories.interfaces.universe_repository import IUniverseRepository
from app.repositories.interfaces.alpha_repository import IAlphaRepository
from app.repositories.interfaces.risk_repository import IRiskRepository
from app.repositories.interfaces.constraint_repository import IConstraintRepository
from app.repositories.interfaces.transaction_cost_repository import ITransactionCostRepository

# Services
from app.services.data_service import DataService
from app.services.alpha_service import AlphaService
from app.services.risk_service import RiskService
from app.services.optimization_service import OptimizationService
from app.services.compliance_service import ComplianceService
from app.services.interfaces.data_service import IDataService
from app.services.interfaces.alpha_service import IAlphaService
from app.services.interfaces.risk_service import IRiskService
from app.services.interfaces.optimization_service import IOptimizationService
from app.services.interfaces.compliance_service import IComplianceService

# LLM
from app.llm.factory import create_llm_provider
from app.llm.interfaces.llm_provider import ILLMProvider

# Solvers
from app.solvers.factory import create_solver
from app.solvers.interfaces.solver import ISolver

# Graph
from app.agents.graph import PortfolioGraph, create_portfolio_graph

# Utils
from app.utils.csv_loader import CSVLoader


# ===========================================
# CSV Loader
# ===========================================

@lru_cache
def get_csv_loader() -> CSVLoader:
    """Get cached CSV loader instance."""
    settings = get_settings()
    return CSVLoader(data_path=settings.csv_data_path)


# ===========================================
# Repositories
# ===========================================

def get_benchmark_repository() -> IBenchmarkRepository:
    """Get benchmark repository instance."""
    return CSVBenchmarkRepository(csv_loader=get_csv_loader())


def get_universe_repository() -> IUniverseRepository:
    """Get universe repository instance."""
    return CSVUniverseRepository(csv_loader=get_csv_loader())


def get_alpha_repository() -> IAlphaRepository:
    """Get alpha repository instance."""
    return CSVAlphaRepository(csv_loader=get_csv_loader())


def get_risk_repository() -> IRiskRepository:
    """Get risk repository instance."""
    return CSVRiskRepository(csv_loader=get_csv_loader())


def get_constraint_repository() -> IConstraintRepository:
    """Get constraint repository instance."""
    return CSVConstraintRepository(csv_loader=get_csv_loader())


def get_transaction_cost_repository() -> ITransactionCostRepository:
    """Get transaction cost repository instance."""
    return CSVTransactionCostRepository(csv_loader=get_csv_loader())


# ===========================================
# LLM Provider
# ===========================================

def get_llm_provider() -> Optional[ILLMProvider]:
    """
    Get LLM provider instance.
    
    Returns None if API key is not configured.
    """
    settings = get_settings()
    
    try:
        api_key = settings.get_llm_api_key()
        if not api_key or api_key.startswith("sk-..."):
            return None
        return create_llm_provider(settings)
    except Exception:
        return None


# ===========================================
# Solver
# ===========================================

def get_solver() -> ISolver:
    """Get optimization solver instance."""
    settings = get_settings()
    return create_solver(settings)


# ===========================================
# Services
# ===========================================

def get_data_service() -> IDataService:
    """Get data service instance."""
    return DataService(
        benchmark_repo=get_benchmark_repository(),
        universe_repo=get_universe_repository(),
        alpha_repo=get_alpha_repository(),
        risk_repo=get_risk_repository(),
        constraint_repo=get_constraint_repository(),
        transaction_cost_repo=get_transaction_cost_repository(),
    )


def get_alpha_service() -> IAlphaService:
    """Get alpha service instance."""
    return AlphaService()


def get_risk_service() -> IRiskService:
    """Get risk service instance."""
    return RiskService()


def get_optimization_service() -> IOptimizationService:
    """Get optimization service instance."""
    return OptimizationService(
        solver=get_solver(),
        risk_service=get_risk_service(),
    )


def get_compliance_service() -> IComplianceService:
    """Get compliance service instance."""
    return ComplianceService()


# ===========================================
# Portfolio Graph
# ===========================================

def get_portfolio_graph() -> PortfolioGraph:
    """
    Get portfolio graph instance.
    
    Creates a fully configured PortfolioGraph with all
    dependencies injected.
    """
    return create_portfolio_graph(
        data_service=get_data_service(),
        alpha_service=get_alpha_service(),
        risk_service=get_risk_service(),
        optimization_service=get_optimization_service(),
        compliance_service=get_compliance_service(),
        risk_repository=get_risk_repository(),
        constraint_repository=get_constraint_repository(),
        transaction_cost_repository=get_transaction_cost_repository(),
        llm_provider=get_llm_provider(),
    )

