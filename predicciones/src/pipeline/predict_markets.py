"""
Market Predictions Pipeline.

Integrates all market models (corners, cards, shots, player props)
with availability checking and feature gating.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from ..domain.market_types import (
    AllMarketsOutput,
    CornersMarketPrediction,
    CardsMarketPrediction,
    ShotsMarketPrediction,
    PlayerPropsPrediction,
    PlayerProp,
    MarketType,
    ConfidenceLevel,
    DataSource,
)
from ..models.market_availability import MarketAvailabilityEvaluator
from ..models.market_models import (
    CornersModel,
    CardsModel,
    ShotsModel,
    PlayerPropsModel,
)

logger = logging.getLogger(__name__)


class MarketsPredictor:
    """
    Main predictor for alternative markets.
    
    Orchestrates:
    - Data availability evaluation
    - Feature computation from recent matches
    - Model predictions for each market
    - Output formatting with availability flags
    """
    
    def __init__(self):
        self.availability_evaluator = MarketAvailabilityEvaluator()
        self.corners_model = CornersModel()
        self.cards_model = CardsModel()
        self.shots_model = ShotsModel()
        self.player_props_model = PlayerPropsModel()
    
    def predict_markets(
        self,
        home_recent_matches: List[Dict[str, Any]],
        away_recent_matches: List[Dict[str, Any]],
        home_team_name: str,
        away_team_name: str,
        team_xg_home: float = 1.5,
        team_xg_away: float = 1.5,
    ) -> AllMarketsOutput:
        """
        Generate predictions for all available markets.
        
        Args:
            home_recent_matches: Recent matches for home team
            away_recent_matches: Recent matches for away team
            home_team_name: Home team name
            away_team_name: Away team name
            team_xg_home: Home team expected goals (from main model)
            team_xg_away: Away team expected goals (from main model)
            
        Returns:
            AllMarketsOutput with predictions for available markets
        """
        output = AllMarketsOutput()
        
        # Compute aggregate stats from recent matches
        home_stats = self._compute_team_stats(home_recent_matches)
        away_stats = self._compute_team_stats(away_recent_matches)
        
        # Check event data availability
        has_event_data = self._check_event_data_available(
            home_recent_matches + away_recent_matches
        )
        
        # ===== CORNERS =====
        output.corners = self._predict_corners(
            home_stats, away_stats, has_event_data,
            home_recent_matches, away_recent_matches,
        )
        
        # ===== CARDS =====
        output.cards = self._predict_cards(
            home_stats, away_stats, has_event_data,
            home_recent_matches, away_recent_matches,
        )
        
        # ===== SHOTS ON TARGET =====
        output.shots_on_target = self._predict_sot(
            home_stats, away_stats,
            home_recent_matches, away_recent_matches,
        )
        
        # ===== PLAYER PROPS =====
        output.player_props = self._predict_player_props(
            home_recent_matches, away_recent_matches,
            home_team_name, away_team_name,
            team_xg_home, team_xg_away,
        )
        
        return output
    
    def _compute_team_stats(self, matches: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Compute aggregate stats from list of matches."""
        if not matches:
            return self._empty_team_stats()
        
        total_corners_for = 0
        total_corners_against = 0
        total_cards_for = 0
        total_cards_against = 0
        total_sot_for = 0
        total_sot_against = 0
        count = 0
        
        for match in matches:
            stats = match.get("stats", {})
            
            # Determine if team was home or away
            home_team = match.get("home_team", "")
            is_home = "home" in match.get("_team_context", "home")
            
            if is_home:
                corners_for = stats.get("home_corners")
                corners_against = stats.get("away_corners")
                cards_for = stats.get("home_total_cards")
                cards_against = stats.get("away_total_cards")
                sot_for = stats.get("home_shots_on_target")
                sot_against = stats.get("away_shots_on_target")
            else:
                corners_for = stats.get("away_corners")
                corners_against = stats.get("home_corners")
                cards_for = stats.get("away_total_cards")
                cards_against = stats.get("home_total_cards")
                sot_for = stats.get("away_shots_on_target")
                sot_against = stats.get("home_shots_on_target")
            
            # Accumulate if data available
            if corners_for is not None:
                total_corners_for += corners_for
                total_corners_against += corners_against or 0
                count += 1
            
            if cards_for is not None:
                total_cards_for += cards_for
                total_cards_against += cards_against or 0
            
            if sot_for is not None:
                total_sot_for += sot_for
                total_sot_against += sot_against or 0
        
        n = max(count, 1)
        
        return {
            "avg_corners_for": total_corners_for / n,
            "avg_corners_against": total_corners_against / n,
            "avg_cards_for": total_cards_for / n,
            "avg_cards_against": total_cards_against / n,
            "avg_sot_for": total_sot_for / n,
            "avg_sot_against": total_sot_against / n,
            "matches_with_corners": count,
            "matches_with_cards": len([m for m in matches if m.get("stats", {}).get("home_total_cards") is not None]),
            "matches_with_sot": len([m for m in matches if m.get("stats", {}).get("home_shots_on_target") is not None]),
        }
    
    def _empty_team_stats(self) -> Dict[str, Any]:
        """Return empty stats dict."""
        return {
            "avg_corners_for": 5.0,  # League average fallback
            "avg_corners_against": 5.0,
            "avg_cards_for": 2.0,
            "avg_cards_against": 2.0,
            "avg_sot_for": 4.0,
            "avg_sot_against": 4.0,
            "matches_with_corners": 0,
            "matches_with_cards": 0,
            "matches_with_sot": 0,
        }
    
    def _check_event_data_available(self, matches: List[Dict[str, Any]]) -> bool:
        """Check if any matches have event-level data."""
        for match in matches:
            events = match.get("events", [])
            if events:
                return True
        return False
    
    def _predict_corners(
        self,
        home_stats: Dict[str, Any],
        away_stats: Dict[str, Any],
        has_event_data: bool,
        home_matches: List[Dict[str, Any]],
        away_matches: List[Dict[str, Any]],
    ) -> CornersMarketPrediction:
        """Generate corners market predictions."""
        # Evaluate availability
        total_avail, event_avail, other_avail = self.availability_evaluator.evaluate_corners_availability(
            home_matches, away_matches, has_event_data
        )
        
        if not total_avail.available:
            return CornersMarketPrediction(
                market_type=MarketType.CORNERS_TOTAL,
                availability=total_avail,
            )
        
        # Compute predictions
        home_avg = home_stats["avg_corners_for"]
        away_avg = away_stats["avg_corners_for"]
        home_conceded = home_stats["avg_corners_against"]
        away_conceded = away_stats["avg_corners_against"]
        
        predictions = {}
        
        # Total over/under
        total_pred = self.corners_model.predict_total_corners(
            home_avg, away_avg, home_conceded, away_conceded
        )
        predictions["total_over_under"] = total_pred["lines"]
        predictions["expected_total"] = total_pred["expected_total"]
        
        # Team totals
        predictions["team_totals"] = self.corners_model.predict_team_corners(
            home_avg, away_avg
        )
        
        # More corners
        predictions["more_corners_team"] = self.corners_model.predict_more_corners(
            home_avg, away_avg
        )
        
        # First/last corner (if event data available)
        if event_avail.available:
            # Simple heuristic based on attack strength
            total_attack = home_avg + away_avg
            if total_attack > 0:
                first_corner_home = home_avg / total_attack
                first_corner_away = away_avg / total_attack
            else:
                first_corner_home = 0.5
                first_corner_away = 0.5
            
            predictions["first_corner"] = {
                "home": round(first_corner_home, 4),
                "away": round(first_corner_away, 4),
            }
            predictions["last_corner"] = {
                "home": round(first_corner_home, 4),  # Simplified
                "away": round(first_corner_away, 4),
            }
        
        return CornersMarketPrediction(
            market_type=MarketType.CORNERS_TOTAL,
            availability=total_avail,
            predictions=predictions,
        )
    
    def _predict_cards(
        self,
        home_stats: Dict[str, Any],
        away_stats: Dict[str, Any],
        has_event_data: bool,
        home_matches: List[Dict[str, Any]],
        away_matches: List[Dict[str, Any]],
    ) -> CardsMarketPrediction:
        """Generate cards market predictions."""
        total_avail, event_avail = self.availability_evaluator.evaluate_cards_availability(
            home_matches, away_matches, has_event_data
        )
        
        if not total_avail.available:
            return CardsMarketPrediction(
                market_type=MarketType.CARDS_TOTAL,
                availability=total_avail,
            )
        
        home_avg = home_stats["avg_cards_for"]
        away_avg = away_stats["avg_cards_for"]
        
        predictions = {}
        
        # Total over/under
        total_pred = self.cards_model.predict_total_cards(
            home_avg, away_avg
        )
        predictions["total_over_under"] = total_pred["lines"]
        predictions["expected_total"] = total_pred["expected_total"]
        
        # Team totals
        predictions["team_totals"] = self.cards_model.predict_team_cards(
            home_avg, away_avg
        )
        
        # More cards
        predictions["more_cards_team"] = self.cards_model.predict_more_cards(
            home_avg, away_avg
        )
        
        # First card (if event data available)
        if event_avail.available:
            # Heuristic based on aggression (cards per match)
            total_aggression = home_avg + away_avg
            if total_aggression > 0:
                first_card_home = home_avg / total_aggression
                first_card_away = away_avg / total_aggression
            else:
                first_card_home = 0.5
                first_card_away = 0.5
            
            predictions["first_card"] = {
                "home": round(first_card_home, 4),
                "away": round(first_card_away, 4),
            }
        
        return CardsMarketPrediction(
            market_type=MarketType.CARDS_TOTAL,
            availability=total_avail,
            predictions=predictions,
        )
    
    def _predict_sot(
        self,
        home_stats: Dict[str, Any],
        away_stats: Dict[str, Any],
        home_matches: List[Dict[str, Any]],
        away_matches: List[Dict[str, Any]],
    ) -> ShotsMarketPrediction:
        """Generate shots on target predictions."""
        avail = self.availability_evaluator.evaluate_shots_availability(
            home_matches, away_matches
        )
        
        if not avail.available:
            return ShotsMarketPrediction(
                market_type=MarketType.SHOTS_ON_TARGET_TOTAL,
                availability=avail,
            )
        
        home_avg = home_stats["avg_sot_for"]
        away_avg = away_stats["avg_sot_for"]
        
        predictions = {}
        
        # Total over/under
        total_pred = self.shots_model.predict_total_sot(
            home_avg, away_avg
        )
        predictions["total_over_under"] = total_pred["lines"]
        predictions["expected_total"] = total_pred["expected_total"]
        
        # Team totals
        predictions["team_totals"] = self.shots_model.predict_team_sot(
            home_avg, away_avg
        )
        
        return ShotsMarketPrediction(
            market_type=MarketType.SHOTS_ON_TARGET_TOTAL,
            availability=avail,
            predictions=predictions,
        )
    
    def _predict_player_props(
        self,
        home_matches: List[Dict[str, Any]],
        away_matches: List[Dict[str, Any]],
        home_team_name: str,
        away_team_name: str,
        team_xg_home: float,
        team_xg_away: float,
    ) -> PlayerPropsPrediction:
        """Generate player props predictions."""
        # For now, return unavailable - proper implementation needs lineup data
        avail = self.availability_evaluator.evaluate_player_props_availability(
            home_matches, away_matches,
            home_lineup_available=False,
            away_lineup_available=False,
        )
        
        return PlayerPropsPrediction(
            market_type=MarketType.PLAYER_ANYTIME_SCORER,
            availability=avail,
            players=[],
        )
