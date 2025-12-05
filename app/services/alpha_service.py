"""
Alpha service implementation.

Handles alpha signal processing, quintile filtering, and security selection.
"""

from typing import Optional

from app.models.alpha import AlphaModel, AlphaScore
from app.services.interfaces.alpha_service import (
    AlphaAnalysis,
    IAlphaService,
    SecuritySelection,
)


class AlphaService(IAlphaService):
    """
    Service for alpha model operations.
    
    Handles alpha score processing, quintile filtering,
    and security selection for portfolio construction.
    """

    async def get_top_quintile_securities(
        self,
        alpha_model: AlphaModel,
        top_n: int = 25,
    ) -> list[AlphaScore]:
        """Get top N securities from quintile 1."""
        # Filter to quintile 1 only
        q1_securities = [s for s in alpha_model.scores if s.alpha_quintile == 1]
        
        # Sort by alpha score descending
        sorted_securities = sorted(q1_securities, key=lambda s: s.alpha_score, reverse=True)
        
        return sorted_securities[:top_n]

    async def filter_by_quintile(
        self,
        alpha_model: AlphaModel,
        quintiles: list[int] = [1],
    ) -> list[AlphaScore]:
        """Filter securities by quintile membership."""
        return [s for s in alpha_model.scores if s.alpha_quintile in quintiles]

    async def analyze_alpha_model(
        self,
        alpha_model: AlphaModel,
    ) -> AlphaAnalysis:
        """Perform comprehensive analysis of alpha model."""
        scores = alpha_model.scores
        
        if not scores:
            return AlphaAnalysis(
                total_securities=0,
                quintile_distribution={1: 0, 2: 0, 3: 0, 4: 0, 5: 0},
                average_alpha_score=0.0,
                top_sector="",
                bottom_sector="",
                sector_average_scores={},
                signal_strength_distribution={},
            )
        
        # Quintile distribution
        quintile_dist = alpha_model.get_quintile_distribution()
        
        # Average alpha score
        avg_score = sum(s.alpha_score for s in scores) / len(scores)
        
        # Sector analysis
        sector_scores = await self.get_sector_scores(alpha_model)
        top_sector = max(sector_scores, key=sector_scores.get) if sector_scores else ""
        bottom_sector = min(sector_scores, key=sector_scores.get) if sector_scores else ""
        
        # Signal strength distribution
        signal_dist: dict[str, int] = {}
        for score in scores:
            strength = score.signal_strength or "Unknown"
            signal_dist[strength] = signal_dist.get(strength, 0) + 1
        
        return AlphaAnalysis(
            total_securities=len(scores),
            quintile_distribution=quintile_dist,
            average_alpha_score=avg_score,
            top_sector=top_sector,
            bottom_sector=bottom_sector,
            sector_average_scores=sector_scores,
            signal_strength_distribution=signal_dist,
        )

    async def rank_securities(
        self,
        alpha_model: AlphaModel,
        ascending: bool = False,
    ) -> list[AlphaScore]:
        """Rank all securities by alpha score."""
        return sorted(
            alpha_model.scores,
            key=lambda s: s.alpha_score,
            reverse=not ascending,
        )

    async def select_securities(
        self,
        alpha_model: AlphaModel,
        count: int = 25,
        min_quintile: int = 1,
        max_quintile: int = 2,
        sector_constraints: Optional[dict[str, int]] = None,
    ) -> SecuritySelection:
        """Select securities for portfolio based on criteria."""
        # Filter by quintile
        eligible = [
            s for s in alpha_model.scores
            if min_quintile <= s.alpha_quintile <= max_quintile
        ]
        
        # Sort by alpha score
        eligible = sorted(eligible, key=lambda s: s.alpha_score, reverse=True)
        
        # Apply sector constraints if provided
        selected: list[AlphaScore] = []
        sector_counts: dict[str, int] = {}
        
        for score in eligible:
            if len(selected) >= count:
                break
            
            sector = score.gics_sector
            current_count = sector_counts.get(sector, 0)
            
            # Check sector constraint
            if sector_constraints and sector in sector_constraints:
                if current_count >= sector_constraints[sector]:
                    continue
            
            selected.append(score)
            sector_counts[sector] = current_count + 1
        
        # Build selection result
        selected_tickers = [s.ticker for s in selected]
        avg_score = sum(s.alpha_score for s in selected) / len(selected) if selected else 0.0
        
        # Sector distribution
        sector_dist: dict[str, int] = {}
        for score in selected:
            sector = score.gics_sector
            sector_dist[sector] = sector_dist.get(sector, 0) + 1
        
        criteria = f"Top {count} from Q{min_quintile}-Q{max_quintile}"
        if sector_constraints:
            criteria += f" with sector limits"
        
        return SecuritySelection(
            selected_tickers=selected_tickers,
            selection_count=len(selected),
            average_alpha_score=avg_score,
            sector_distribution=sector_dist,
            selection_criteria=criteria,
        )

    async def get_sector_scores(
        self,
        alpha_model: AlphaModel,
    ) -> dict[str, float]:
        """Get average alpha score by sector."""
        sector_scores: dict[str, list[float]] = {}
        
        for score in alpha_model.scores:
            sector = score.gics_sector
            if sector not in sector_scores:
                sector_scores[sector] = []
            sector_scores[sector].append(score.alpha_score)
        
        return {
            sector: sum(scores) / len(scores)
            for sector, scores in sector_scores.items()
            if scores
        }

    async def get_alpha_weights(
        self,
        alpha_scores: list[AlphaScore],
        normalize: bool = True,
    ) -> dict[str, float]:
        """Calculate alpha-weighted portfolio weights."""
        if not alpha_scores:
            return {}
        
        # Calculate raw weights based on alpha scores
        weights = {s.ticker: s.alpha_score for s in alpha_scores}
        
        # Normalize to sum to 1 if requested
        if normalize:
            total = sum(weights.values())
            if total > 0:
                weights = {k: v / total for k, v in weights.items()}
        
        return weights

