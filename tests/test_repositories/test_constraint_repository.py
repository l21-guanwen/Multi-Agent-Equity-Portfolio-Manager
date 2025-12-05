"""
Tests for constraint repository.
"""

import pytest

from app.repositories.csv.csv_constraint_repository import CSVConstraintRepository


class TestCSVConstraintRepository:
    """Tests for CSVConstraintRepository."""

    @pytest.mark.asyncio
    async def test_get_all_returns_constraints(
        self,
        constraint_repository: CSVConstraintRepository,
    ):
        """Test that get_all returns constraints."""
        constraints = await constraint_repository.get_all()
        
        assert len(constraints) > 0

    @pytest.mark.asyncio
    async def test_get_active_constraints(
        self,
        constraint_repository: CSVConstraintRepository,
    ):
        """Test getting active constraints."""
        active = await constraint_repository.get_active_constraints()
        
        # All active constraints should be enabled
        for constraint in active:
            assert constraint.is_enabled is True

    @pytest.mark.asyncio
    async def test_get_constraints_by_type(
        self,
        constraint_repository: CSVConstraintRepository,
    ):
        """Test getting constraints by type."""
        stock_constraints = await constraint_repository.get_constraints_by_type(
            "single_stock_active"
        )
        
        for constraint in stock_constraints:
            assert constraint.constraint_type == "single_stock_active"

    @pytest.mark.asyncio
    async def test_get_stock_constraints(
        self,
        constraint_repository: CSVConstraintRepository,
    ):
        """Test getting single stock constraints."""
        constraints = await constraint_repository.get_stock_constraints()
        
        assert len(constraints) > 0
        for constraint in constraints:
            assert constraint.constraint_type == "single_stock_active"
            # Active weight bounds should be ±1%
            assert constraint.min_value == -1.0
            assert constraint.max_value == 1.0

    @pytest.mark.asyncio
    async def test_get_sector_constraints(
        self,
        constraint_repository: CSVConstraintRepository,
    ):
        """Test getting sector constraints."""
        constraints = await constraint_repository.get_sector_constraints()
        
        # Should have constraints for all 11 GICS sectors
        assert len(constraints) == 11
        sectors = [c.target for c in constraints]
        assert "Information Technology" in sectors
        assert "Health Care" in sectors

    @pytest.mark.asyncio
    async def test_constraint_bounds(
        self,
        constraint_repository: CSVConstraintRepository,
    ):
        """Test constraint bounds are valid."""
        constraints = await constraint_repository.get_all()
        
        for constraint in constraints:
            # Min should be <= max
            assert constraint.min_value <= constraint.max_value

    @pytest.mark.asyncio
    async def test_get_constraint_dict(
        self,
        constraint_repository: CSVConstraintRepository,
    ):
        """Test getting constraint dictionary."""
        constraint_dict = await constraint_repository.get_constraint_dict()
        
        assert "single_stock_active" in constraint_dict
        assert "sector_active" in constraint_dict

