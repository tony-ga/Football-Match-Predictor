"""
ESPN data parsers.

Converts raw ESPN API responses to structured internal models.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from ..domain.models import UpcomingMatch, EspnMatchContext, EspnTeamRef

logger = logging.getLogger(__name__)


def parse_scoreboard_event(event: Dict[str, Any]) -> Optional[UpcomingMatch]:
    """
    Parse a single event from ESPN scoreboard response.
    
    Args:
        event: Raw event dict from ESPN scoreboard
        
    Returns:
        UpcomingMatch object or None if parsing fails
    """
    try:
        competitions = event.get("competitions", [])
        if not competitions:
            return None
        
        comp = competitions[0]
        competitors = comp.get("competitors", [])
        
        # Identify home/away
        home_comp = None
        away_comp = None
        for c in competitors:
            if c.get("homeAway") == "home":
                home_comp = c
            elif c.get("homeAway") == "away":
                away_comp = c
        
        # Fallback to first two competitors if homeAway not specified
        if not home_comp or not away_comp:
            if len(competitors) >= 2:
                home_comp = competitors[0]
                away_comp = competitors[1]
            else:
                return None
        
        # Extract basic info
        event_id = str(event.get("id", ""))
        event_date = event.get("date", "")
        
        # Status
        status = _parse_status(comp)
        
        # Venue
        venue_info = comp.get("venue", {})
        venue_name = venue_info.get("fullName") or venue_info.get("address", {}).get("city")
        neutral_venue = venue_info.get("neutral", None)
        
        # Competition and stage
        league_info = event.get("league", {})
        competition = league_info.get("name", "FIFA World Cup")
        stage = _parse_stage(event, comp)
        
        # Team names
        home_team_raw = home_comp.get("team", {}).get("displayName", "")
        away_team_raw = away_comp.get("team", {}).get("displayName", "")
        
        if not home_team_raw or not away_team_raw:
            logger.warning(f"Missing team names in event {event_id}")
            return None
        
        return UpcomingMatch(
            event_id=event_id,
            date=event_date,
            competition=competition,
            stage=stage,
            status=status,
            home_team=home_team_raw,
            away_team=away_team_raw,
            neutral_venue=neutral_venue,
            venue=venue_name,
        )
    except Exception as e:
        logger.error(f"Failed to parse scoreboard event: {e}")
        return None


def parse_summary_to_context(
    summary: Dict[str, Any],
    event_id: str
) -> Optional[EspnMatchContext]:
    """
    Parse ESPN summary response to EspnMatchContext.
    
    Args:
        summary: Raw summary dict from ESPN
        event_id: Event ID (used as fallback)
        
    Returns:
        EspnMatchContext or None if parsing fails
    """
    try:
        competitions = summary.get("competitions", [])
        if not competitions:
            # Try to get from header or boxscore
            logger.warning(f"No competitions found in summary for event {event_id}")
            return None
        
        comp = competitions[0]
        competitors = comp.get("competitors", [])
        
        home_comp = None
        away_comp = None
        for c in competitors:
            if c.get("homeAway") == "home":
                home_comp = c
            elif c.get("homeAway") == "away":
                away_comp = c
        
        if not home_comp or not away_comp:
            if len(competitors) >= 2:
                home_comp = competitors[0]
                away_comp = competitors[1]
            else:
                return None
        
        # Build team refs
        home_team = _build_team_ref(home_comp)
        away_team = _build_team_ref(away_comp)
        
        # Extract other fields
        event_date = summary.get("date", "")
        league_info = summary.get("league", {})
        competition = league_info.get("name", "FIFA World Cup")
        stage = _parse_stage(summary, comp)
        status = _parse_status(comp)
        
        venue_info = comp.get("venue", {})
        venue_name = venue_info.get("fullName") or venue_info.get("address", {}).get("city")
        neutral_venue = venue_info.get("neutral", None)
        
        return EspnMatchContext(
            event_id=event_id,
            competition=competition,
            date=event_date,
            stage=stage,
            status=status,
            home_team=home_team,
            away_team=away_team,
            venue=venue_name,
            neutral_venue=neutral_venue,
            raw=summary,
        )
    except Exception as e:
        logger.error(f"Failed to parse summary for event {event_id}: {e}")
        return None


def _parse_status(comp: Dict[str, Any]) -> str:
    """Parse match status from competition block."""
    status = comp.get("status", {})
    status_type = status.get("type", {})
    status_name = status_type.get("name", "").lower()
    status_state = status_type.get("state", "").lower()
    
    if "final" in status_name or "ended" in status_name or status_state == "post":
        return "post"
    elif "progress" in status_name or "live" in status_name or "active" in status_name or status_state == "in":
        return "in"
    elif "pre" in status_name or "scheduled" in status_name or status_state == "pre":
        return "pre"
    else:
        return "pre"  # default


def _parse_stage(event: Dict[str, Any], comp: Dict[str, Any]) -> Optional[str]:
    """
    Parse and normalize stage from ESPN data.
    
    Maps ESPN stage names to internal enum values:
    - group
    - round_of_16
    - quarter_final
    - semi_final
    - final
    - regular (fallback)
    """
    season = event.get("season", {})
    season_slug = season.get("slug", "")
    week = event.get("week", "")
    notes = comp.get("notes", "")
    type_info = comp.get("type", {})
    short_detail = type_info.get("shortDetail", "")
    
    # Combine all potential stage indicators
    stage_lower = f"{season_slug} {week} {notes} {short_detail}".lower()
    
    # Map to normalized stage names
    # Check specific stages first before general ones
    if "third" in stage_lower or "tercer" in stage_lower:
        return "third_place"
    elif "semi" in stage_lower:
        return "semi_final"
    elif "quarter" in stage_lower or "cuartos" in stage_lower:
        return "quarter_final"
    elif "round of 16" in stage_lower or "octavos" in stage_lower:
        return "round_of_16"
    elif "round of 32" in stage_lower:
        return "round_of_32"
    elif "group" in stage_lower or "fase de grupos" in stage_lower:
        return "group"
    elif "final" in stage_lower:
        return "final"
    else:
        return "regular"


def _build_team_ref(team_comp: Dict[str, Any]) -> EspnTeamRef:
    """Build EspnTeamRef from competitor block."""
    team_data = team_comp.get("team", {})
    
    return EspnTeamRef(
        team_id=str(team_data.get("id")) if team_data.get("id") else None,
        name=team_data.get("displayName", ""),
        display_name=team_data.get("displayName", ""),
        short_name=team_data.get("shortDisplayName"),
        abbreviation=team_data.get("abbreviation"),
    )


def parse_teams_list(teams_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Parse teams endpoint response to list of team info.
    
    Args:
        teams_data: Raw teams response from ESPN
        
    Returns:
        List of dicts with team_id, name, display_name, etc.
    """
    teams = []
    if not teams_data:
        return teams
    
    items = teams_data.get("items", [])
    for item in items:
        team = item.get("team", {})
        if team:
            teams.append({
                "team_id": str(team.get("id")),
                "name": team.get("displayName", ""),
                "short_name": team.get("shortDisplayName"),
                "abbreviation": team.get("abbreviation"),
                "location": team.get("location"),
                "color": team.get("color"),
            })
    
    return teams
