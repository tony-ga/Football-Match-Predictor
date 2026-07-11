"""
Market Availability Evaluator.

Determines whether sufficient data exists to make predictions for each market type.
Implements feature gating based on ESPN data coverage.

Supports multi-level player data availability:
- Level 1: roster coverage (players parseable from rosters)
- Level 2: player signal coverage (roster + leaders/offensive signals)
- Level 3: player event coverage (goals, cards, substitutions from keyEvents)
- Level 4: full player stats coverage (complete numerical stats per player)
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
    - player_props_partial: >= 3 matches with roster + offensive signals
    - player_props_full: >= 3 matches with complete player stats
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
    ) -> Dict[str, Any]:
        """
        Evaluate availability for player props markets at multiple levels.
        
        Player props now support partial availability:
        - Level 1: roster coverage
        - Level 2: player signal coverage  
        - Level 3: player event coverage
        - Level 4: full player stats coverage
        
        Returns dict with granular availability info for each prop type.
        
        Args:
            home_recent_matches: Recent matches for home team
            away_recent_matches: Recent matches for away team
            home_lineup_available: Whether home lineup is known
            away_lineup_available: Whether away lineup is known
            
        Returns:
            Dict with availability info for each prop type
        """
        total_matches = home_recent_matches + away_recent_matches
        
        # Count matches at each level
        matches_with_rosters = [m for m in total_matches if self._has_player_roster(m)]
        matches_with_signals = [m for m in total_matches if self._has_player_signals(m)]
        matches_with_events = [m for m in total_matches if self._has_player_events(m)]
        matches_with_full_stats = [m for m in total_matches if self._has_full_player_stats(m)]
        
        roster_count = len(matches_with_rosters)
        signal_count = len(matches_with_signals)
        event_count = len(matches_with_events)
        full_stats_count = len(matches_with_full_stats)
        
        has_some_lineup = home_lineup_available or away_lineup_available
        
        # Determine overall availability
        any_player_data = roster_count > 0 or signal_count > 0 or event_count > 0
        
        # Scorer props (anytime, first) need: roster + (signals OR events with goals)
        scorer_props_available = (
            roster_count >= self.MIN_SAMPLES_PLAYER 
            and (signal_count >= 1 or event_count >= 1)
        )
        
        # SOT and assists need full player stats
        stats_props_available = full_stats_count >= self.MIN_SAMPLES_PLAYER
        
        # Build detailed response
        result = {
            "available": any_player_data,
            "coverage_levels": {
                "matches_with_rosters": roster_count,
                "matches_with_player_signals": signal_count,
                "matches_with_player_events": event_count,
                "matches_with_full_player_stats": full_stats_count,
            },
            "prop_availability": {
                "anytime_scorer": {
                    "available": scorer_props_available,
                    "reason": None if scorer_props_available else "Insufficient roster + offensive signal data for scorer props",
                },
                "first_scorer": {
                    "available": scorer_props_available,
                    "reason": None if scorer_props_available else "Insufficient roster + offensive signal data for scorer props",
                },
                "shots_on_target": {
                    "available": stats_props_available,
                    "reason": None if stats_props_available else "No reliable player-level shots on target history found",
                },
                "assists": {
                    "available": stats_props_available,
                    "reason": None if stats_props_available else "No reliable player-level assist history found",
                },
            },
        }
        
        # Compute overall confidence based on best available level
        if full_stats_count >= self.MIN_SAMPLES_PLAYER:
            confidence = self._compute_confidence(full_stats_count, 8)
            data_source = DataSource.HYBRID if has_some_lineup else DataSource.ESPN_RECENT_MATCHES
        elif event_count >= self.MIN_SAMPLES_PLAYER:
            confidence = ConfidenceLevel.LOW
            data_source = DataSource.ESPN_SUMMARY
        elif signal_count >= self.MIN_SAMPLES_PLAYER:
            confidence = ConfidenceLevel.LOW
            data_source = DataSource.ESPN_SUMMARY
        elif roster_count >= self.MIN_SAMPLES_PLAYER:
            confidence = ConfidenceLevel.LOW
            data_source = DataSource.ESPN_RECENT_MATCHES
        else:
            confidence = ConfidenceLevel.LOW
            data_source = DataSource.UNAVAILABLE
        
        result["sample_size"] = max(roster_count, signal_count, event_count, full_stats_count)
        result["confidence"] = confidence.value
        result["data_source"] = data_source.value
        
        return result
    
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
    
    def _has_player_roster(self, match: Dict[str, Any]) -> bool:
        """Check if match has player roster data."""
        players = match.get("players", [])
        return len(players) > 0
    
    def _has_player_signals(self, match: Dict[str, Any]) -> bool:
        """Check if match has player signals (leaders, offensive indicators)."""
        signals = match.get("player_signals", [])
        return len(signals) > 0
    
    def _has_player_events(self, match: Dict[str, Any]) -> bool:
        """Check if match has player events (goals, cards, subs)."""
        events = match.get("player_events", [])
        # Check for goal events specifically
        goal_events = [e for e in events if e.get("event_type") in ("goal", "own_goal")]
        return len(goal_events) > 0
    
    def _has_full_player_stats(self, match: Dict[str, Any]) -> bool:
        """Check if match has full player-level statistics."""
        players = match.get("players", [])
        if not players:
            return False
        
        # Need at least some players with meaningful stats
        players_with_stats = [
            p for p in players
            if p.get("goals") is not None or p.get("assists") is not None or p.get("shots") is not None
        ]
        return len(players_with_stats) >= 3
    
    def _has_player_stats(self, match: Dict[str, Any]) -> bool:
        """Legacy method - check if match has any player-level data."""
        return (
            self._has_player_roster(match) 
            or self._has_player_signals(match)
            or self._has_player_events(match)
        )
    
    def _compute_confidence(self, sample_size: int, high_threshold: int = 10) -> ConfidenceLevel:
        """Compute confidence level based on sample size."""
        if sample_size >= high_threshold:
            return ConfidenceLevel.HIGH
        elif sample_size >= self.MIN_SAMPLES_TOTAL * 2:
            return ConfidenceLevel.MEDIUM
        else:
            return ConfidenceLevel.LOW
