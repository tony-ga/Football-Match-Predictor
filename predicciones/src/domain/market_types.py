"""
Market types and domain models for prediction markets.

Defines market types, availability status, and output structures
for corners, cards, shots, and player props markets.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class MarketType(str, Enum):
    """Supported market types."""
    CORNERS_TOTAL = "corners_total"
    CORNERS_TEAM = "corners_team"
    CORNERS_HALF = "corners_half"
    CORNERS_RACE = "corners_race"
    CORNERS_FIRST = "corners_first"
    CORNERS_LAST = "corners_last"
    CORNERS_MORE = "corners_more"
    
    CARDS_TOTAL = "cards_total"
    CARDS_TEAM = "cards_team"
    CARDS_HALF = "cards_half"
    CARDS_FIRST = "cards_first"
    CARDS_MORE = "cards_more"
    
    SHOTS_TOTAL = "shots_total"
    SHOTS_TEAM = "shots_team"
    SHOTS_ON_TARGET_TOTAL = "shots_on_target_total"
    SHOTS_ON_TARGET_TEAM = "shots_on_target_team"
    
    PLAYER_ANYTIME_SCORER = "player_anytime_scorer"
    PLAYER_FIRST_SCORER = "player_first_scorer"
    PLAYER_SOT = "player_sot"
    PLAYER_ASSISTS = "player_assists"


class DataSource(str, Enum):
    """Data source types for market predictions."""
    ESPN_SUMMARY = "espn_summary"
    ESPN_RECENT_MATCHES = "espn_recent_matches"
    ESPN_SCOREBOARD = "espn_scoreboard"
    HYBRID = "hybrid"
    STATIC_FALLBACK = "static_fallback"
    UNAVAILABLE = "unavailable"


class ConfidenceLevel(str, Enum):
    """Confidence levels for market predictions."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


@dataclass
class MarketAvailability:
    """
    Represents the availability status of a market.
    
    Attributes:
        available: Whether the market can be predicted
        reason: Explanation if not available
        sample_size: Number of samples used for prediction
        confidence: Confidence level based on data quality
        data_source: Source of the data used
    """
    available: bool = True
    reason: str = ""
    sample_size: int = 0
    confidence: ConfidenceLevel = ConfidenceLevel.LOW
    data_source: DataSource = DataSource.UNAVAILABLE
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "available": self.available,
            "reason": self.reason if not self.available else None,
            "sample_size": self.sample_size,
            "confidence": self.confidence.value,
            "data_source": self.data_source.value,
        }


@dataclass
class MarketPrediction:
    """
    Base class for market predictions.
    
    Attributes:
        market_type: Type of market
        availability: Availability status
        predictions: Dictionary of prediction values
        metadata: Additional metadata about the prediction
    """
    market_type: MarketType
    availability: MarketAvailability
    predictions: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        result = self.availability.to_dict()
        result["predictions"] = self.predictions
        if self.metadata:
            result["metadata"] = self.metadata
        return result


@dataclass
class CornersMarketPrediction(MarketPrediction):
    """
    Prediction output for corners markets.
    
    Supports:
    - total_over_under: Over/under lines for total corners
    - team_totals: Over/under for individual team corners
    - more_corners_team: Which team will have more corners
    - first_corner: Which team will win first corner
    - last_corner: Which team will win last corner
    - race_to_X: Race to X corners probabilities
    - half_totals: First/second half corner totals
    """
    pass


@dataclass
class CardsMarketPrediction(MarketPrediction):
    """
    Prediction output for cards markets.
    
    Supports:
    - total_over_under: Over/under lines for total cards
    - team_totals: Over/under for individual team cards
    - more_cards_team: Which team will have more cards
    - first_card: Which team will receive first card
    - half_totals: First/second half card totals
    """
    pass


@dataclass
class ShotsMarketPrediction(MarketPrediction):
    """
    Prediction output for shots markets.
    
    Supports:
    - total_over_under: Over/under for total shots on target
    - team_totals: Over/under for team shots on target
    """
    pass


@dataclass
class PlayerProp:
    """
    Individual player prop prediction.
    
    Attributes:
        player_id: Unique player identifier
        player_name: Player display name
        team: Team name
        position: Player position if available
        is_starter: Whether player is expected to start
        starter_probability: Probability of starting (0-1)
    """
    player_id: str
    player_name: str
    team: str
    position: Optional[str] = None
    is_starter: bool = False
    starter_probability: float = 0.0
    
    # Prop-specific probabilities
    anytime_scorer_prob: Optional[float] = None
    first_scorer_prob: Optional[float] = None
    sot_over_under: Optional[Dict[str, float]] = None
    assists_over_under: Optional[Dict[str, float]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "player_id": self.player_id,
            "player_name": self.player_name,
            "team": self.team,
            "position": self.position,
            "is_starter": self.is_starter,
            "starter_probability": self.starter_probability,
            "anytime_scorer_prob": self.anytime_scorer_prob,
            "first_scorer_prob": self.first_scorer_prob,
            "sot_over_under": self.sot_over_under,
            "assists_over_under": self.assists_over_under,
        }


@dataclass
class PlayerPropsPrediction(MarketPrediction):
    """
    Prediction output for player props markets.
    
    Contains list of player-specific predictions for:
    - anytime scorer
    - first scorer
    - shots on target
    - assists
    """
    players: List[PlayerProp] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        result = super().to_dict()
        result["players"] = [p.to_dict() for p in self.players]
        return result


@dataclass
class AllMarketsOutput:
    """
    Complete output for all markets.
    
    Attributes:
        corners: Corners market predictions
        cards: Cards market predictions
        shots_on_target: Shots on target predictions
        player_props: Player props predictions
    """
    corners: Optional[CornersMarketPrediction] = None
    cards: Optional[CardsMarketPrediction] = None
    shots_on_target: Optional[ShotsMarketPrediction] = None
    player_props: Optional[PlayerPropsPrediction] = None
    
    def to_dict(self) -> Dict[str, Any]:
        result = {}
        if self.corners:
            result["corners"] = self.corners.to_dict()
        if self.cards:
            result["cards"] = self.cards.to_dict()
        if self.shots_on_target:
            result["shots_on_target"] = self.shots_on_target.to_dict()
        if self.player_props:
            result["player_props"] = self.player_props.to_dict()
        return result
