"""
CSV-based constraint repository implementation.

Loads optimization constraint data from CSV files.
"""

from datetime import date
from typing import Optional

from app.core.constants import DataFileName
from app.models.constraint import Constraint, ConstraintSet
from app.repositories.interfaces.constraint_repository import IConstraintRepository
from app.utils.csv_loader import CSVLoader


class CSVConstraintRepository(IConstraintRepository):
    """
    CSV implementation of the constraint repository.
    
    Loads constraints from 08_Optimization_Constraints.csv.
    """

    COLUMN_MAPPING = {
        "Constraint_Type": "constraint_type",
        "Constraint_Name": "constraint_name",
        "Benchmark_Weight_Pct": "benchmark_weight_pct",
        "Lower_Bound_Pct": "lower_bound_pct",
        "Upper_Bound_Pct": "upper_bound_pct",
        "Constraint_Type_Code": "constraint_type_code",
        "Is_Hard_Constraint": "is_hard_constraint",
        "Penalty_Coefficient": "penalty_coefficient",
        "Optimization_Definition_ID": "optimization_definition_id",
    }

    def __init__(self, csv_loader: Optional[CSVLoader] = None):
        """Initialize the repository."""
        self._loader = csv_loader or CSVLoader()
        self._cache: Optional[list[Constraint]] = None

    async def _load_data(self) -> list[Constraint]:
        """Load and cache constraint data."""
        if self._cache is None:
            self._cache = self._loader.load_as_models(
                DataFileName.CONSTRAINTS,
                Constraint,
                self.COLUMN_MAPPING,
            )
        return self._cache

    def clear_cache(self):
        """Clear the data cache."""
        self._cache = None

    async def get_all(self, as_of_date: Optional[date] = None) -> list[Constraint]:
        """Get all constraints."""
        return await self._load_data()

    async def get_by_id(self, id: str) -> Optional[Constraint]:
        """Get constraint by name."""
        data = await self._load_data()
        for constraint in data:
            if constraint.constraint_name == id:
                return constraint
        return None

    async def get_by_ids(self, ids: list[str]) -> list[Constraint]:
        """Get constraints by multiple names."""
        data = await self._load_data()
        return [c for c in data if c.constraint_name in ids]

    async def get_constraint_set(
        self,
        optimization_id: str = "OPT_DEF_SPX_ALPHA_001",
        as_of_date: Optional[date] = None,
    ) -> Optional[ConstraintSet]:
        """Get the full constraint set for an optimization."""
        constraints = await self.get_all(as_of_date)
        
        if not constraints:
            return None
        
        # Filter by optimization ID if specified
        filtered = [
            c for c in constraints
            if c.optimization_definition_id is None or c.optimization_definition_id == optimization_id
        ]
        
        return ConstraintSet(
            optimization_definition_id=optimization_id,
            constraints=filtered,
            as_of_date=as_of_date,
        )

    async def get_sector_constraints(
        self,
        as_of_date: Optional[date] = None,
    ) -> list[Constraint]:
        """Get all sector-level constraints."""
        data = await self.get_all(as_of_date)
        return [c for c in data if c.is_sector_constraint]

    async def get_stock_constraints(
        self,
        as_of_date: Optional[date] = None,
    ) -> list[Constraint]:
        """Get all stock-level constraints."""
        data = await self.get_all(as_of_date)
        return [c for c in data if c.is_stock_constraint]

    async def get_constraint_for_sector(
        self,
        sector: str,
        as_of_date: Optional[date] = None,
    ) -> Optional[Constraint]:
        """Get constraint for a specific sector."""
        sector_constraints = await self.get_sector_constraints(as_of_date)
        for constraint in sector_constraints:
            if constraint.constraint_name == sector:
                return constraint
        return None

    async def get_constraint_for_stock(
        self,
        ticker: str,
        as_of_date: Optional[date] = None,
    ) -> Optional[Constraint]:
        """Get constraint for a specific stock."""
        stock_constraints = await self.get_stock_constraints(as_of_date)
        for constraint in stock_constraints:
            if constraint.constraint_name == ticker:
                return constraint
        return None

