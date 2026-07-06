"""
Match Input Factory.

Converts ESPN match data and team selections to internal MatchInput format
compatible with the prediction pipeline.
"""
from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

from ..ingestion.schemas import (
    MatchInput,
    Metadata,
    TeamData,
    FactoresTacticos,
    FactoresColectivos,
    Contexto,
    FactoresExternos,
    MatchStage,
    CompetitionType,
)
from ..data.espn_client_v2 import EspnClient
from ..data.espn_normalizers import TeamNormalizer
from ..domain.models import UpcomingMatch, EspnMatchContext
from ..domain.exceptions import MatchInputBuildError, TeamNotFoundError

logger = logging.getLogger(__name__)


class MatchInputFactory:
    """
    Factory for building MatchInput objects from various sources.
    
    Supports:
    - ESPN event_id
    - ESPN upcoming matches
    - Manual team names
    - JSON file (legacy)
    """
    
    def __init__(
        self,
        espn_client: Optional[EspnClient] = None,
        normalizer: Optional[TeamNormalizer] = None
    ):
        """
        Initialize factory.
        
        Args:
            espn_client: ESPN client instance
            normalizer: Team name normalizer
        """
        self.client = espn_client or EspnClient()
        self.normalizer = normalizer or TeamNormalizer()
    
    def build_from_event_id(self, event_id: str) -> MatchInput:
        """
        Build MatchInput from ESPN event ID.
        
        Args:
            event_id: ESPN event ID
            
        Returns:
            MatchInput ready for prediction pipeline
            
        Raises:
            MatchInputBuildError: If event not found or parsing fails
        """
        try:
            summary = self.client.get_summary(event_id)
        except Exception as e:
            raise MatchInputBuildError(
                f"Failed to fetch event {event_id} from ESPN: {e}",
                source="espn"
            )
        
        if not summary:
            raise MatchInputBuildError(
                f"No data returned for event {event_id}",
                source="espn"
            )
        
        # Parse summary
        competitions = summary.get("competitions", [])
        if not competitions:
            raise MatchInputBuildError(
                f"No competitions in summary for event {event_id}",
                source="espn"
            )
        
        comp = competitions[0]
        competitors = comp.get("competitors", [])
        
        home_comp = next((c for c in competitors if c.get("homeAway") == "home"), competitors[0] if competitors else {})
        away_comp = next((c for c in competitors if c.get("homeAway") == "away"), competitors[1] if len(competitors) > 1 else {})
        
        home_team_raw = home_comp.get("team", {}).get("displayName", "")
        away_team_raw = away_comp.get("team", {}).get("displayName", "")
        
        if not home_team_raw or not away_team_raw:
            raise MatchInputBuildError(
                f"Could not extract team names from event {event_id}",
                source="espn"
            )
        
        # Normalize team names
        home_team = self.normalizer.normalize(home_team_raw)
        away_team = self.normalizer.normalize(away_team_raw)
        
        # Parse stage
        stage = self._parse_stage(summary, comp)
        
        # Parse venue
        venue_info = comp.get("venue", {})
        venue = venue_info.get("fullName") or venue_info.get("address", {}).get("city")
        neutral_venue = venue_info.get("neutral", False)
        
        # Build metadata
        metadata = Metadata(
            match_id=f"espn_{event_id}",
            home_team=home_team,
            away_team=away_team,
            competition=summary.get("league", {}).get("name", "FIFA World Cup"),
            competition_type=CompetitionType.WORLD_CUP,
            stage=stage,
            date=summary.get("date", datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")),
            neutral_venue=bool(neutral_venue),
            venue=venue,
        )
        
        # Build team data with defaults
        team1 = self._build_team_data(home_team, is_home=True)
        team2 = self._build_team_data(away_team, is_home=False)
        
        return MatchInput(metadata=metadata, team1=team1, team2=team2)
    
    def build_from_upcoming_match(self, match: UpcomingMatch) -> MatchInput:
        """
        Build MatchInput from an UpcomingMatch object.
        
        Args:
            match: UpcomingMatch from match selector
            
        Returns:
            MatchInput ready for prediction pipeline
        """
        # Normalize team names
        home_team = self.normalizer.normalize(match.home_team)
        away_team = self.normalizer.normalize(match.away_team)
        
        # Parse stage
        stage = self._parse_stage_string(match.stage)
        
        # Build metadata
        metadata = Metadata(
            match_id=f"espn_{match.event_id}",
            home_team=home_team,
            away_team=away_team,
            competition=match.competition,
            competition_type=CompetitionType.WORLD_CUP,
            stage=stage,
            date=match.date[:10] if match.date else datetime.utcnow().strftime("%Y-%m-%d"),
            neutral_venue=bool(match.neutral_venue) if match.neutral_venue is not None else False,
            venue=match.venue,
        )
        
        # Build team data
        team1 = self._build_team_data(home_team, is_home=True)
        team2 = self._build_team_data(away_team, is_home=False)
        
        return MatchInput(metadata=metadata, team1=team1, team2=team2)
    
    def build_from_team_names(
        self,
        home_team: str,
        away_team: str,
        competition: Optional[str] = None,
        stage: Optional[str] = None,
        match_date: Optional[str] = None
    ) -> MatchInput:
        """
        Build MatchInput from team names.
        
        Args:
            home_team: Home team name
            away_team: Away team name
            competition: Optional competition name
            stage: Optional stage
            match_date: Optional match date
            
        Returns:
            MatchInput ready for prediction pipeline
            
        Raises:
            MatchInputBuildError: If teams cannot be normalized
        """
        # Normalize team names
        home_normalized = self.normalizer.normalize(home_team)
        away_normalized = self.normalizer.normalize(away_team)
        
        logger.info(f"Normalized teams: {home_team} -> {home_normalized}, {away_team} -> {away_normalized}")
        
        # Build metadata
        metadata = Metadata(
            match_id=f"manual_{home_normalized}_vs_{away_normalized}",
            home_team=home_normalized,
            away_team=away_normalized,
            competition=competition or "Friendly",
            competition_type=CompetitionType.FRIENDLY,
            stage=self._parse_stage_string(stage) if stage else MatchStage.REGULAR,
            date=match_date or datetime.utcnow().strftime("%Y-%m-%d"),
            neutral_venue=False,
        )
        
        # Build team data
        team1 = self._build_team_data(home_normalized, is_home=True)
        team2 = self._build_team_data(away_normalized, is_home=False)
        
        return MatchInput(metadata=metadata, team1=team1, team2=team2)
    
    def _build_team_data(self, team_name: str, is_home: bool) -> TeamData:
        """
        Build TeamData with default values.
        
        The prediction pipeline will enrich this with actual data from
        ESPN, ratings, etc. Here we just provide minimal valid structure.
        
        Args:
            team_name: Normalized team name
            is_home: Whether this is the home team
            
        Returns:
            TeamData with defaults
        """
        # Create minimal valid team data
        # The feature builder will fill in actual stats from ESPN/ratings
        localia = 1.0 if is_home else -1.0
        
        externos = FactoresExternos(
            localía=localia,
        )
        
        return TeamData(
            nombre=team_name,
            FACTORES_TACTICOS=FactoresTacticos(),
            FACTORES_COLECTIVOS=FactoresColectivos(),
            CONTEXTO=Contexto(),
            FACTORES_EXTERNOS=externos,
            JUGADORES=[],
        )
    
    def _parse_stage(self, summary: Dict[str, Any], comp: Dict[str, Any]) -> MatchStage:
        """Parse stage from ESPN summary data."""
        season = summary.get("season", {})
        season_slug = season.get("slug", "")
        week = summary.get("week", "")
        notes = comp.get("notes", "")
        type_info = comp.get("type", {})
        short_detail = type_info.get("shortDetail", "")
        
        stage_lower = f"{season_slug} {week} {notes} {short_detail}".lower()
        
        if "final" in stage_lower and "semi" not in stage_lower and "third" not in stage_lower:
            return MatchStage.FINAL
        elif "semi" in stage_lower:
            return MatchStage.SEMI_FINAL
        elif "quarter" in stage_lower or "cuartos" in stage_lower:
            return MatchStage.QUARTER_FINAL
        elif "round of 16" in stage_lower or "octavos" in stage_lower:
            return MatchStage.ROUND_OF_16
        elif "group" in stage_lower or "fase de grupos" in stage_lower:
            return MatchStage.GROUP
        else:
            return MatchStage.REGULAR
    
    def _parse_stage_string(self, stage: Optional[str]) -> MatchStage:
        """Parse stage from string."""
        if not stage:
            return MatchStage.REGULAR
        
        stage_lower = stage.lower()
        
        if stage_lower == "final":
            return MatchStage.FINAL
        elif "semi" in stage_lower:
            return MatchStage.SEMI_FINAL
        elif "quarter" in stage_lower or "cuartos" in stage_lower:
            return MatchStage.QUARTER_FINAL
        elif "round of 16" in stage_lower or "octavos" in stage_lower:
            return MatchStage.ROUND_OF_16
        elif "group" in stage_lower or "fase de grupos" in stage_lower:
            return MatchStage.GROUP
        else:
            return MatchStage.REGULAR


def build_match_input_from_espn_event(event_id: str) -> MatchInput:
    """Convenience function to build MatchInput from ESPN event ID."""
    factory = MatchInputFactory()
    return factory.build_from_event_id(event_id)


def build_match_input_from_team_names(
    home_team: str,
    away_team: str,
    **kwargs
) -> MatchInput:
    """Convenience function to build MatchInput from team names."""
    factory = MatchInputFactory()
    return factory.build_from_team_names(home_team, away_team, **kwargs)


def build_match_inputs_from_upcoming(limit: int = 10) -> List[UpcomingMatch]:
    """Convenience function to get upcoming matches."""
    from .application.match_selector import MatchSelector
    
    selector = MatchSelector()
    return selector.get_upcoming_matches(limit=limit)
