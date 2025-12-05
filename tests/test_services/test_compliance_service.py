"""
Tests for compliance service.
"""

import pytest

from app.services.compliance_service import ComplianceService
from app.models.constraint import Constraint


class TestComplianceService:
    """Tests for ComplianceService."""

    @pytest.fixture
    def sample_constraints(self) -> list[Constraint]:
        """Create sample constraints."""
        return [
            Constraint(
                constraint_id="single_stock_AAPL",
                constraint_type="single_stock_active",
                target="AAPL",
                min_value=-1.0,
                max_value=1.0,
                is_enabled=True,
            ),
            Constraint(
                constraint_id="single_stock_MSFT",
                constraint_type="single_stock_active",
                target="MSFT",
                min_value=-1.0,
                max_value=1.0,
                is_enabled=True,
            ),
            Constraint(
                constraint_id="sector_IT",
                constraint_type="sector_active",
                target="Information Technology",
                min_value=-2.0,
                max_value=2.0,
                is_enabled=True,
            ),
            Constraint(
                constraint_id="sector_Financials",
                constraint_type="sector_active",
                target="Financials",
                min_value=-2.0,
                max_value=2.0,
                is_enabled=True,
            ),
        ]

    @pytest.fixture
    def sample_benchmark_weights(self) -> dict[str, float]:
        """Sample benchmark weights."""
        return {
            "AAPL": 6.70,
            "MSFT": 5.77,
            "GOOGL": 3.21,
            "JPM": 1.50,
        }

    @pytest.fixture
    def sample_sector_mapping(self) -> dict[str, str]:
        """Sample sector mapping."""
        return {
            "AAPL": "Information Technology",
            "MSFT": "Information Technology",
            "GOOGL": "Communication Services",
            "JPM": "Financials",
        }

    def test_check_single_stock_constraints_pass(
        self,
        compliance_service: ComplianceService,
        sample_constraints: list[Constraint],
        sample_benchmark_weights: dict[str, float],
    ):
        """Test single stock constraint check - passing case."""
        # Portfolio weights within ±1% of benchmark
        portfolio_weights = {
            "AAPL": 7.50,  # Active: +0.8%
            "MSFT": 5.00,  # Active: -0.77%
        }
        
        violations = compliance_service.check_single_stock_constraints(
            portfolio_weights=portfolio_weights,
            benchmark_weights=sample_benchmark_weights,
            constraints=sample_constraints,
        )
        
        assert len(violations) == 0

    def test_check_single_stock_constraints_fail(
        self,
        compliance_service: ComplianceService,
        sample_constraints: list[Constraint],
        sample_benchmark_weights: dict[str, float],
    ):
        """Test single stock constraint check - violation case."""
        # Portfolio weights exceeding ±1% of benchmark
        portfolio_weights = {
            "AAPL": 9.00,  # Active: +2.3% (exceeds +1%)
            "MSFT": 4.00,  # Active: -1.77% (exceeds -1%)
        }
        
        violations = compliance_service.check_single_stock_constraints(
            portfolio_weights=portfolio_weights,
            benchmark_weights=sample_benchmark_weights,
            constraints=sample_constraints,
        )
        
        assert len(violations) == 2
        tickers = [v["ticker"] for v in violations]
        assert "AAPL" in tickers
        assert "MSFT" in tickers

    def test_check_sector_constraints_pass(
        self,
        compliance_service: ComplianceService,
        sample_constraints: list[Constraint],
        sample_benchmark_weights: dict[str, float],
        sample_sector_mapping: dict[str, str],
    ):
        """Test sector constraint check - passing case."""
        # Portfolio weights with sector active within ±2%
        portfolio_weights = {
            "AAPL": 7.00,
            "MSFT": 6.00,  # IT total: 13% vs benchmark 12.47%
            "JPM": 1.50,   # Financials: 1.5% vs benchmark 1.5%
        }
        
        violations = compliance_service.check_sector_constraints(
            portfolio_weights=portfolio_weights,
            benchmark_weights=sample_benchmark_weights,
            sector_mapping=sample_sector_mapping,
            constraints=sample_constraints,
        )
        
        assert len(violations) == 0

    def test_check_sector_constraints_fail(
        self,
        compliance_service: ComplianceService,
        sample_constraints: list[Constraint],
        sample_benchmark_weights: dict[str, float],
        sample_sector_mapping: dict[str, str],
    ):
        """Test sector constraint check - violation case."""
        # Portfolio weights with IT sector overweight > 2%
        portfolio_weights = {
            "AAPL": 10.00,
            "MSFT": 8.00,  # IT total: 18% vs benchmark 12.47% = +5.53%
            "JPM": 0.00,   # Financials: 0% vs benchmark 1.5% = -1.5%
        }
        
        violations = compliance_service.check_sector_constraints(
            portfolio_weights=portfolio_weights,
            benchmark_weights=sample_benchmark_weights,
            sector_mapping=sample_sector_mapping,
            constraints=sample_constraints,
        )
        
        assert len(violations) >= 1
        sectors = [v["sector"] for v in violations]
        assert "Information Technology" in sectors

    def test_check_all_constraints(
        self,
        compliance_service: ComplianceService,
        sample_constraints: list[Constraint],
        sample_benchmark_weights: dict[str, float],
        sample_sector_mapping: dict[str, str],
    ):
        """Test checking all constraints at once."""
        portfolio_weights = {
            "AAPL": 7.50,
            "MSFT": 5.00,
            "JPM": 1.50,
        }
        
        result = compliance_service.check_all_constraints(
            portfolio_weights=portfolio_weights,
            benchmark_weights=sample_benchmark_weights,
            sector_mapping=sample_sector_mapping,
            constraints=sample_constraints,
        )
        
        assert "is_compliant" in result
        assert "stock_violations" in result
        assert "sector_violations" in result
        assert result["is_compliant"] is True

    def test_calculate_active_weights(
        self,
        compliance_service: ComplianceService,
        sample_benchmark_weights: dict[str, float],
    ):
        """Test calculating active weights."""
        portfolio_weights = {
            "AAPL": 8.00,
            "MSFT": 4.00,
            "GOOGL": 5.00,
        }
        
        active = compliance_service.calculate_active_weights(
            portfolio_weights=portfolio_weights,
            benchmark_weights=sample_benchmark_weights,
        )
        
        assert active["AAPL"] == 8.00 - 6.70  # +1.30
        assert active["MSFT"] == 4.00 - 5.77  # -1.77
        assert active["GOOGL"] == 5.00 - 3.21  # +1.79

    def test_get_compliance_report(
        self,
        compliance_service: ComplianceService,
        sample_constraints: list[Constraint],
        sample_benchmark_weights: dict[str, float],
        sample_sector_mapping: dict[str, str],
    ):
        """Test generating compliance report."""
        portfolio_weights = {
            "AAPL": 7.50,
            "MSFT": 5.00,
            "JPM": 1.50,
        }
        
        report = compliance_service.get_compliance_report(
            portfolio_weights=portfolio_weights,
            benchmark_weights=sample_benchmark_weights,
            sector_mapping=sample_sector_mapping,
            constraints=sample_constraints,
        )
        
        assert "summary" in report
        assert "details" in report
        assert "recommendations" in report

