"""
Domain models for ESPN match data and prediction inputs.

This module defines the core data structures used throughout the application
for representing matches, teams, and prediction inputs.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class EspnTeamRef:
    """
    Reference to a team from ESPN data.
    
    Attributes:
        team_id: ESPN team ID if available
        name: Canonical team name (normalized)
        display_name: Full display name from ESPN
        short_name: Short/abbreviated name
        abbreviation: 3-letter code if available
    """
    team_id: Optional[str] = None
    name: str = ""
    display_name: str = ""
    short_name: Optional[str] = None
    abbreviation: Optional[str] = None


@dataclass
class UpcomingMatch:
    """
    Represents an upcoming match from ESPN scoreboard.
    
    Attributes:
        event_id: Unique ESPN event identifier
        date: ISO 8601 date string
        competition: Competition/tournament name
        stage: Match stage (group, round_of_16, etc.)
        status: Match status (pre, in, post)
        home_team: Home team name
        away_team: Away team name
        neutral_venue: Whether venue is neutral
        venue: Venue name/location
    """
    event_id: str
    date: str
    competition: str
    stage: Optional[str]
    status: str
    home_team: str
    away_team: str
    neutral_venue: Optional[bool] = None
    venue: Optional[str] = None
    
    def to_display_row(self) -> Dict[str, str]:
        """Return a dict suitable for tabular display."""
        return {
            "event_id": self.event_id,
            "date": self.date[:16].replace("T", " ") if self.date else "",
            "competition": self.competition,
            "stage": self.stage or "N/A",
            "status": self.status.upper(),
            "home_team": self.home_team,
            "away_team": self.away_team,
            "venue": self.venue or "N/A",
        }


@dataclass
class EspnMatchContext:
    """
    Complete match context from ESPN data.
    
    This contains all relevant information about a match retrieved from ESPN,
    including team references, competition details, and raw data for debugging.
    
    Attributes:
        event_id: Unique ESPN event identifier
        competition: Competition/tournament name
        date: ISO 8601 date string
        stage: Match stage (normalized)
        status: Match status (pre, in, post)
        home_team: Home team reference
        away_team: Away team reference
        venue: Venue name/location
        neutral_venue: Whether venue is neutral
        raw: Raw ESPN JSON response for debugging
    """
    event_id: str
    competition: str
    date: str
    stage: Optional[str]
    status: str
    home_team: EspnTeamRef
    away_team: EspnTeamRef
    venue: Optional[str] = None
    neutral_venue: Optional[bool] = None
    raw: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TeamNormalizationResult:
    """
    Result of team name normalization attempt.
    
    Attributes:
        found: Whether a matching team was found
        normalized_name: Normalized/canonical team name
        team_id: ESPN team ID if found
        confidence: Confidence score of the match (0.0-1.0)
        alternatives: List of alternative matches if ambiguous
    """
    found: bool
    normalized_name: str = ""
    team_id: Optional[str] = None
    confidence: float = 0.0
    alternatives: List[str] = field(default_factory=list)
