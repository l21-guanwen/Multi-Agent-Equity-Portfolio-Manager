"""Alpha model domain models."""

from datetime import date
from typing import Optional

from pydantic import BaseModel, Field, computed_field


class AlphaScore(BaseModel):
    """
    Alpha score for a single security.
    
    Based on 04_Alpha_Model_SP500.csv schema.
    """

    # Security Info
    ticker: str = Field(..., description="Stock ticker symbol")
    security_name: str = Field(..., description="Full security name")
    gics_sector: str = Field(..., description="GICS Level 1 sector")
    gics_industry: Optional[str] = Field(None, description="GICS industry group")
    
    # Alpha Scores
    alpha_score: float = Field(..., ge=0, le=1, description="Normalized score (0=worst, 1=best)")
    alpha_quintile: int = Field(..., ge=1, le=5, description="Quintile rank (1=best, 5=worst)")
    alpha_confidence: Optional[float] = Field(None, ge=0, le=1, description="Model confidence (0-1)")
    
    # Expected Returns
    expected_return_1m_pct: Optional[float] = Field(None, description="1-month expected return (%)")
    expected_return_3m_pct: Optional[float] = Field(None, description="3-month expected return (%)")
    
    # Signal Info
    signal_strength: Optional[str] = Field(None, description="Signal category (Very Weak to Very Strong)")
    
    # Model Info
    model_id: str = Field(..., description="Alpha model identifier")
    model_name: Optional[str] = Field(None, description="Alpha model name")
    as_of_date: date = Field(..., description="Score generation date")

    @computed_field
    @property
    def is_top_quintile(self) -> bool:
        """Check if security is in top quintile (Q1)."""
        return self.alpha_quintile == 1

    @computed_field
    @property
    def is_bottom_quintile(self) -> bool:
        """Check if security is in bottom quintile (Q5)."""
        return self.alpha_quintile == 5

    class Config:
        """Pydantic model configuration."""
        
        frozen = False
        str_strip_whitespace = True


class AlphaModel(BaseModel):
    """
    Collection of alpha scores for the full universe.
    
    Represents the output of an alpha model run.
    """

    model_id: str = Field(..., description="Alpha model identifier")
    model_name: str = Field(default="AI-Based Idiosyncratic Alpha Model", description="Model name")
    scores: list[AlphaScore] = Field(default_factory=list)
    as_of_date: date = Field(..., description="Model run date")

    @computed_field
    @property
    def security_count(self) -> int:
        """Number of securities with scores."""
        return len(self.scores)

    def get_score(self, ticker: str) -> Optional[AlphaScore]:
        """Get alpha score by ticker."""
        for score in self.scores:
            if score.ticker == ticker:
                return score
        return None

    def get_quintile(self, quintile: int) -> list[AlphaScore]:
        """Get all securities in a specific quintile."""
        return [s for s in self.scores if s.alpha_quintile == quintile]

    def get_top_quintile(self) -> list[AlphaScore]:
        """Get all securities in top quintile (Q1)."""
        return self.get_quintile(1)

    def get_top_n(self, n: int) -> list[AlphaScore]:
        """Get top N securities by alpha score."""
        sorted_scores = sorted(self.scores, key=lambda s: s.alpha_score, reverse=True)
        return sorted_scores[:n]

    def get_score_dict(self) -> dict[str, float]:
        """Get dictionary of ticker -> alpha score."""
        return {s.ticker: s.alpha_score for s in self.scores}

    def get_quintile_distribution(self) -> dict[int, int]:
        """Get count of securities in each quintile."""
        distribution: dict[int, int] = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
        for score in self.scores:
            distribution[score.alpha_quintile] += 1
        return distribution

