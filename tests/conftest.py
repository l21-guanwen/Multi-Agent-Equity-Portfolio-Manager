"""
Pytest configuration and fixtures.

Provides shared fixtures for all tests.
"""

import os
import sys
from datetime import date
from pathlib import Path
from typing import AsyncGenerator, Generator

import pytest
import pytest_asyncio
from fastapi.testclient import TestClient
from httpx import AsyncClient

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.core.config import Settings
from app.utils.csv_loader import CSVLoader

# Repositories
from app.repositories.csv.csv_benchmark_repository import CSVBenchmarkRepository
from app.repositories.csv.csv_universe_repository import CSVUniverseRepository
from app.repositories.csv.csv_alpha_repository import CSVAlphaRepository
from app.repositories.csv.csv_risk_repository import CSVRiskRepository
from app.repositories.csv.csv_constraint_repository import CSVConstraintRepository
from app.repositories.csv.csv_transaction_cost_repository import CSVTransactionCostRepository

# Services
from app.services.data_service import DataService
from app.services.alpha_service import AlphaService
from app.services.risk_service import RiskService
from app.services.optimization_service import OptimizationService
from app.services.compliance_service import ComplianceService

# Solvers
from app.solvers.cvxpy_solver import CvxpySolver

# Main app
from main import app


# ===========================================
# Configuration Fixtures
# ===========================================

@pytest.fixture(scope="session")
def test_data_path() -> str:
    """Get path to data directory."""
    return str(Path(__file__).parent.parent / "data")


@pytest.fixture(scope="session")
def settings(test_data_path: str) -> Settings:
    """Create test settings."""
    return Settings(
        app_name="Test Portfolio Manager",
        debug=True,
        data_source="csv",
        csv_data_path=test_data_path,
        llm_provider="openai",
        llm_model="gpt-4",
        solver="cvxpy",
    )


# ===========================================
# CSV Loader Fixtures
# ===========================================

@pytest.fixture(scope="session")
def csv_loader(test_data_path: str) -> CSVLoader:
    """Create CSV loader with test data path."""
    return CSVLoader(data_path=test_data_path)


# ===========================================
# Repository Fixtures
# ===========================================

@pytest.fixture
def benchmark_repository(csv_loader: CSVLoader) -> CSVBenchmarkRepository:
    """Create benchmark repository."""
    return CSVBenchmarkRepository(csv_loader=csv_loader)


@pytest.fixture
def universe_repository(csv_loader: CSVLoader) -> CSVUniverseRepository:
    """Create universe repository."""
    return CSVUniverseRepository(csv_loader=csv_loader)


@pytest.fixture
def alpha_repository(csv_loader: CSVLoader) -> CSVAlphaRepository:
    """Create alpha repository."""
    return CSVAlphaRepository(csv_loader=csv_loader)


@pytest.fixture
def risk_repository(csv_loader: CSVLoader) -> CSVRiskRepository:
    """Create risk repository."""
    return CSVRiskRepository(csv_loader=csv_loader)


@pytest.fixture
def constraint_repository(csv_loader: CSVLoader) -> CSVConstraintRepository:
    """Create constraint repository."""
    return CSVConstraintRepository(csv_loader=csv_loader)


@pytest.fixture
def transaction_cost_repository(csv_loader: CSVLoader) -> CSVTransactionCostRepository:
    """Create transaction cost repository."""
    return CSVTransactionCostRepository(csv_loader=csv_loader)


# ===========================================
# Service Fixtures
# ===========================================

@pytest.fixture
def data_service(
    benchmark_repository: CSVBenchmarkRepository,
    universe_repository: CSVUniverseRepository,
    alpha_repository: CSVAlphaRepository,
    risk_repository: CSVRiskRepository,
    constraint_repository: CSVConstraintRepository,
    transaction_cost_repository: CSVTransactionCostRepository,
) -> DataService:
    """Create data service."""
    return DataService(
        benchmark_repo=benchmark_repository,
        universe_repo=universe_repository,
        alpha_repo=alpha_repository,
        risk_repo=risk_repository,
        constraint_repo=constraint_repository,
        transaction_cost_repo=transaction_cost_repository,
    )


@pytest.fixture
def alpha_service() -> AlphaService:
    """Create alpha service."""
    return AlphaService()


@pytest.fixture
def risk_service() -> RiskService:
    """Create risk service."""
    return RiskService()


@pytest.fixture
def solver() -> CvxpySolver:
    """Create optimization solver."""
    return CvxpySolver()


@pytest.fixture
def optimization_service(
    solver: CvxpySolver,
    risk_service: RiskService,
) -> OptimizationService:
    """Create optimization service."""
    return OptimizationService(
        solver=solver,
        risk_service=risk_service,
    )


@pytest.fixture
def compliance_service() -> ComplianceService:
    """Create compliance service."""
    return ComplianceService()


# ===========================================
# API Client Fixtures
# ===========================================

@pytest.fixture
def client() -> Generator[TestClient, None, None]:
    """Create synchronous test client."""
    with TestClient(app) as client:
        yield client


@pytest_asyncio.fixture
async def async_client() -> AsyncGenerator[AsyncClient, None]:
    """Create asynchronous test client."""
    async with AsyncClient(app=app, base_url="http://test") as client:
        yield client


# ===========================================
# Sample Data Fixtures
# ===========================================

@pytest.fixture
def sample_weights() -> dict[str, float]:
    """Sample portfolio weights for testing."""
    return {
        "AAPL": 0.05,
        "MSFT": 0.05,
        "GOOGL": 0.04,
        "AMZN": 0.04,
        "NVDA": 0.04,
    }


@pytest.fixture
def sample_benchmark_weights() -> dict[str, float]:
    """Sample benchmark weights for testing."""
    return {
        "AAPL": 6.70,
        "MSFT": 5.77,
        "GOOGL": 3.21,
        "AMZN": 3.96,
        "NVDA": 7.20,
    }


@pytest.fixture
def sample_sector_mapping() -> dict[str, str]:
    """Sample sector mapping for testing."""
    return {
        "AAPL": "Information Technology",
        "MSFT": "Information Technology",
        "GOOGL": "Communication Services",
        "AMZN": "Consumer Discretionary",
        "NVDA": "Information Technology",
    }

