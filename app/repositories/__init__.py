"""Repository module for data access layer."""

# Interfaces
from app.repositories.interfaces.base_repository import IBaseRepository
from app.repositories.interfaces.benchmark_repository import IBenchmarkRepository
from app.repositories.interfaces.universe_repository import IUniverseRepository
from app.repositories.interfaces.alpha_repository import IAlphaRepository
from app.repositories.interfaces.risk_repository import IRiskRepository
from app.repositories.interfaces.constraint_repository import IConstraintRepository
from app.repositories.interfaces.transaction_cost_repository import ITransactionCostRepository

# CSV Implementations
from app.repositories.csv.csv_benchmark_repository import CSVBenchmarkRepository
from app.repositories.csv.csv_universe_repository import CSVUniverseRepository
from app.repositories.csv.csv_alpha_repository import CSVAlphaRepository
from app.repositories.csv.csv_risk_repository import CSVRiskRepository
from app.repositories.csv.csv_constraint_repository import CSVConstraintRepository
from app.repositories.csv.csv_transaction_cost_repository import CSVTransactionCostRepository

__all__ = [
    # Interfaces
    "IBaseRepository",
    "IBenchmarkRepository",
    "IUniverseRepository",
    "IAlphaRepository",
    "IRiskRepository",
    "IConstraintRepository",
    "ITransactionCostRepository",
    # CSV Implementations
    "CSVBenchmarkRepository",
    "CSVUniverseRepository",
    "CSVAlphaRepository",
    "CSVRiskRepository",
    "CSVConstraintRepository",
    "CSVTransactionCostRepository",
]

