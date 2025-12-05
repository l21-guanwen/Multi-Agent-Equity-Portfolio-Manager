"""
CSV-based alpha repository implementation.

Loads alpha model data from CSV files.
"""

from datetime import date
from typing import Optional

from app.core.constants import DataFileName
from app.models.alpha import AlphaModel, AlphaScore
from app.repositories.interfaces.alpha_repository import IAlphaRepository
from app.utils.csv_loader import CSVLoader


class CSVAlphaRepository(IAlphaRepository):
    """
    CSV implementation of the alpha repository.
    
    Loads alpha scores from 04_Alpha_Model_SP500.csv.
    """

    COLUMN_MAPPING = {
        "Ticker": "ticker",
        "Security_Name": "security_name",
        "GICS_Sector": "gics_sector",
        "GICS_Industry": "gics_industry",
        "Alpha_Score": "alpha_score",
        "Alpha_Quintile": "alpha_quintile",
        "Alpha_Confidence": "alpha_confidence",
        "Expected_Return_1M_Pct": "expected_return_1m_pct",
        "Expected_Return_3M_Pct": "expected_return_3m_pct",
        "Signal_Strength": "signal_strength",
        "Model_ID": "model_id",
        "Model_Name": "model_name",
        "As_Of_Date": "as_of_date",
    }

    def __init__(self, csv_loader: Optional[CSVLoader] = None):
        """Initialize the repository."""
        self._loader = csv_loader or CSVLoader()
        self._cache: Optional[list[AlphaScore]] = None

    async def _load_data(self) -> list[AlphaScore]:
        """Load and cache alpha data."""
        if self._cache is None:
            self._cache = self._loader.load_as_models(
                DataFileName.ALPHA,
                AlphaScore,
                self.COLUMN_MAPPING,
            )
        return self._cache

    def clear_cache(self):
        """Clear the data cache."""
        self._cache = None

    async def get_all(self, as_of_date: Optional[date] = None) -> list[AlphaScore]:
        """Get all alpha scores."""
        data = await self._load_data()
        if as_of_date:
            return [s for s in data if s.as_of_date == as_of_date]
        return data

    async def get_by_id(self, id: str) -> Optional[AlphaScore]:
        """Get alpha score by ticker."""
        return await self.get_score(id)

    async def get_by_ids(self, ids: list[str]) -> list[AlphaScore]:
        """Get alpha scores by multiple tickers."""
        data = await self._load_data()
        return [s for s in data if s.ticker in ids]

    async def get_alpha_model(
        self,
        model_id: str = "AI_ALPHA_MODEL_V1",
        as_of_date: Optional[date] = None,
    ) -> Optional[AlphaModel]:
        """Get the full alpha model with all scores."""
        scores = await self.get_all(as_of_date)
        
        if not scores:
            return None
        
        # Get model info from first score
        actual_model_id = scores[0].model_id if scores else model_id
        model_name = scores[0].model_name if scores and scores[0].model_name else "AI-Based Idiosyncratic Alpha Model"
        actual_date = as_of_date or scores[0].as_of_date
        
        return AlphaModel(
            model_id=actual_model_id,
            model_name=model_name,
            scores=scores,
            as_of_date=actual_date,
        )

    async def get_score(
        self,
        ticker: str,
        as_of_date: Optional[date] = None,
    ) -> Optional[AlphaScore]:
        """Get alpha score for a single security."""
        data = await self.get_all(as_of_date)
        for score in data:
            if score.ticker == ticker:
                return score
        return None

    async def get_scores_by_quintile(
        self,
        quintile: int,
        as_of_date: Optional[date] = None,
    ) -> list[AlphaScore]:
        """Get all securities in a specific alpha quintile."""
        data = await self.get_all(as_of_date)
        return [s for s in data if s.alpha_quintile == quintile]

    async def get_top_scores(
        self,
        n: int = 25,
        as_of_date: Optional[date] = None,
    ) -> list[AlphaScore]:
        """Get top N securities by alpha score."""
        data = await self.get_all(as_of_date)
        sorted_data = sorted(data, key=lambda s: s.alpha_score, reverse=True)
        return sorted_data[:n]

    async def get_scores_by_sector(
        self,
        sector: str,
        as_of_date: Optional[date] = None,
    ) -> list[AlphaScore]:
        """Get all alpha scores for a specific sector."""
        data = await self.get_all(as_of_date)
        return [s for s in data if s.gics_sector == sector]

    async def get_score_dict(
        self,
        as_of_date: Optional[date] = None,
    ) -> dict[str, float]:
        """Get dictionary of ticker to alpha score."""
        data = await self.get_all(as_of_date)
        return {s.ticker: s.alpha_score for s in data}

    async def get_quintile_distribution(
        self,
        as_of_date: Optional[date] = None,
    ) -> dict[int, int]:
        """Get count of securities in each quintile."""
        data = await self.get_all(as_of_date)
        distribution: dict[int, int] = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
        for score in data:
            if score.alpha_quintile in distribution:
                distribution[score.alpha_quintile] += 1
        return distribution

