"""
Market Availability Evaluator.

Determines whether sufficient data exists to make predictions for each market type.
Implements feature gating based on ESPN data coverage.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

from ..domain.market_types import (
    MarketAvailability,
    ConfidenceLevel,
    DataSource,
)

logger = logging.getLogger(__name__)


class MarketAvailabilityEvaluator:
    """
    Evaluates data availability for different market types.
    
    Each market has minimum data requirements:
    - corners_total: >= 3 matches with team-level corner stats
    - first_corner: >= 3 matches with event-level corner sequence
    - player_props: >= 3 matches with player-level stats and lineup mapping
    - etc.
    """
    
    # Minimum sample sizes for different markets
    MIN_SAMPLES_TOTAL = 3
    MIN_SAMPLES_EVENT_LEVEL = 3
    MIN_SAMPLES_PLAYER = 3
    MIN_SAMPLES_HALF = 4
    
    def __init__(self):
        pass
    
    def evaluate_corners_availability(
        self,
        home_recent_matches: List[Dict[str, Any]],
        away_recent_matches: List[Dict[str, Any]],
        has_event_data: bool = False,
    ) -> Tuple[MarketAvailability, MarketAvailability, MarketAvailability]:
        """
        Evaluate availability for corners markets.
        
        Returns:
            Tuple of (total, first_corner, other_corners) availability
        """
        # Count matches with corner data
        total_matches = home_recent_matches + away_recent_matches
        matches_with_corners = [
            m for m in total_matches
            if self._has_corner_stats(m)
        ]
        corner_sample_size = len(matches_with_corners)
        
        # Total corners availability
        if corner_sample_size >= self.MIN_SAMPLES_TOTAL:
            total_availability = MarketAvailability(
                available=True,
                sample_size=corner_sample_size,
                confidence=self._compute_confidence(corner_sample_size, 10),
                data_source=DataSource.ESPN_RECENT_MATCHES,
            )
        else:
            total_availability = MarketAvailability(
                available=False,
                reason=f"Insufficient matches with corner data ({corner_sample_size} < {self.MIN_SAMPLES_TOTAL})",
                sample_size=corner_sample_size,
                data_source=DataSource.UNAVAILABLE,
            )
        
        # First/last corner availability (requires event-level data)
        if has_event_data and corner_sample_size >= self.MIN_SAMPLES_EVENT_LEVEL:
            event_availability = MarketAvailability(
                available=True,
                sample_size=corner_sample_size,
                confidence=self._compute_confidence(corner_sample_size, 8),
                data_source=DataSource.ESPN_SUMMARY,
            )
        else:
            event_availability = MarketAvailability(
                available=False,
                reason="No event-level corner sequence available from ESPN for sampled matches",
                sample_size=corner_sample_size,
                data_source=DataSource.UNAVAILABLE,
            )
        
        # Other corners (team totals, more corners) use same data as total
        other_availability = MarketAvailability(
            available=total_availability.available,
            reason=total_availability.reason if not total_availability.available else "",
            sample_size=corner_sample_size,
            confidence=total_availability.confidence,
            data_source=total_availability.data_source,
        )
        
        return total_availability, event_availability, other_availability
    
    def evaluate_cards_availability(
        self,
        home_recent_matches: List[Dict[str, Any]],
        away_recent_matches: List[Dict[str, Any]],
        has_event_data: bool = False,
    ) -> Tuple[MarketAvailability, MarketAvailability]:
        """
        Evaluate availability for cards markets.
        
        Returns:
            Tuple of (total, first_card) availability
        """
        total_matches = home_recent_matches + away_recent_matches
        matches_with_cards = [
            m for m in total_matches
            if self._has_card_stats(m)
        ]
        card_sample_size = len(matches_with_cards)
        
        # Total cards availability
        if card_sample_size >= self.MIN_SAMPLES_TOTAL:
            total_availability = MarketAvailability(
                available=True,
                sample_size=card_sample_size,
                confidence=self._compute_confidence(card_sample_size, 10),
                data_source=DataSource.ESPN_RECENT_MATCHES,
            )
        else:
            total_availability = MarketAvailability(
                available=False,
                reason=f"Insufficient matches with card data ({card_sample_size} < {self.MIN_SAMPLES_TOTAL})",
                sample_size=card_sample_size,
                data_source=DataSource.UNAVAILABLE,
            )
        
        # First card availability (requires event-level data)
        if has_event_data and card_sample_size >= self.MIN_SAMPLES_EVENT_LEVEL:
            event_availability = MarketAvailability(
                available=True,
                sample_size=card_sample_size,
                confidence=self._compute_confidence(card_sample_size, 8),
                data_source=DataSource.ESPN_SUMMARY,
            )
        else:
            event_availability = MarketAvailability(
                available=False,
                reason="No event-level card sequence available from ESPN for sampled matches",
                sample_size=card_sample_size,
                data_source=DataSource.UNAVAILABLE,
            )
        
        return total_availability, event_availability
    
    def evaluate_shots_availability(
        self,
        home_recent_matches: List[Dict[str, Any]],
        away_recent_matches: List[Dict[str, Any]],
    ) -> MarketAvailability:
        """
        Evaluate availability for shots on target markets.
        """
        total_matches = home_recent_matches + away_recent_matches
        matches_with_sot = [
            m for m in total_matches
            if self._has_shot_stats(m)
        ]
        sot_sample_size = len(matches_with_sot)
        
        if sot_sample_size >= self.MIN_SAMPLES_TOTAL:
            return MarketAvailability(
                available=True,
                sample_size=sot_sample_size,
                confidence=self._compute_confidence(sot_sample_size, 10),
                data_source=DataSource.ESPN_RECENT_MATCHES,
            )
        else:
            return MarketAvailability(
                available=False,
                reason=f"Insufficient matches with shots on target data ({sot_sample_size} < {self.MIN_SAMPLES_TOTAL})",
                sample_size=sot_sample_size,
                data_source=DataSource.UNAVAILABLE,
            )
    
    def evaluate_player_props_availability(
        self,
        home_recent_matches: List[Dict[str, Any]],
        away_recent_matches: List[Dict[str, Any]],
        home_lineup_available: bool = False,
        away_lineup_available: bool = False,
    ) -> MarketAvailability:
        """
        Evaluate availability for player props markets.
        
        Player props require:
        - Player-level stats in recent matches
        - Lineup information (confirmed or estimated)
        """
        total_matches = home_recent_matches + away_recent_matches
        
        # Count matches with player stats
        matches_with_players = [
            m for m in total_matches
            if self._has_player_stats(m)
        ]
        player_sample_size = len(matches_with_players)
        
        # Need both player stats AND some lineup info
        has_some_lineup = home_lineup_available or away_lineup_available
        
        if player_sample_size >= self.MIN_SAMPLES_PLAYER and has_some_lineup:
            confidence = self._compute_confidence(player_sample_size, 8)
            # Lower confidence if no confirmed lineups
            if not home_lineup_available or not away_lineup_available:
                if confidence == ConfidenceLevel.HIGH:
                    confidence = ConfidenceLevel.MEDIUM
                elif confidence == ConfidenceLevel.MEDIUM:
                    confidence = ConfidenceLevel.LOW
            
            return MarketAvailability(
                available=True,
                sample_size=player_sample_size,
                confidence=confidence,
                data_source=DataSource.HYBRID if has_some_lineup else DataSource.ESPN_RECENT_MATCHES,
            )
        else:
            reasons = []
            if player_sample_size < self.MIN_SAMPLES_PLAYER:
                reasons.append(f"insufficient player stats ({player_sample_size} < {self.MIN_SAMPLES_PLAYER})")
            if not has_some_lineup:
                reasons.append("no lineup information available")
            
            return MarketAvailability(
                available=False,
                reason="; ".join(reasons),
                sample_size=player_sample_size,
                data_source=DataSource.UNAVAILABLE,
            )
    
    def _has_corner_stats(self, match: Dict[str, Any]) -> bool:
        """Check if match has corner statistics."""
        stats = match.get("stats", {})
        return any([
            stats.get("home_corners") is not None,
            stats.get("away_corners") is not None,
        ])
    
    def _has_card_stats(self, match: Dict[str, Any]) -> bool:
        """Check if match has card statistics."""
        stats = match.get("stats", {})
        return any([
            stats.get("home_yellow_cards") is not None,
            stats.get("away_yellow_cards") is not None,
            stats.get("home_red_cards") is not None,
            stats.get("away_red_cards") is not None,
            stats.get("home_total_cards") is not None,
            stats.get("away_total_cards") is not None,
        ])
    
    def _has_shot_stats(self, match: Dict[str, Any]) -> bool:
        """Check if match has shot statistics."""
        stats = match.get("stats", {})
        return any([
            stats.get("home_shots") is not None,
            stats.get("away_shots") is not None,
            stats.get("home_shots_on_target") is not None,
            stats.get("away_shots_on_target") is not None,
        ])
    
    def _has_player_stats(self, match: Dict[str, Any]) -> bool:
        """Check if match has player-level statistics."""
        players = match.get("players", [])
        return len(players) > 0
    
    def _compute_confidence(self, sample_size: int, high_threshold: int = 10) -> ConfidenceLevel:
        """Compute confidence level based on sample size."""
        if sample_size >= high_threshold:
            return ConfidenceLevel.HIGH
        elif sample_size >= self.MIN_SAMPLES_TOTAL * 2:
            return ConfidenceLevel.MEDIUM
        else:
            return ConfidenceLevel.LOW
