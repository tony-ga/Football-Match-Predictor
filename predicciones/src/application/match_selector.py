"""
Match selection logic for ESPN-based match discovery.

Handles listing, filtering, and selecting matches from ESPN data.
"""
from __future__ import annotations

import logging
from typing import List, Optional, Tuple

from ..data.espn_client_v2 import EspnClient
from ..data.espn_parsers import parse_scoreboard_event
from ..domain.models import UpcomingMatch
from ..domain.exceptions import MatchSelectionError, EspnApiError

logger = logging.getLogger(__name__)


class MatchSelector:
    """
    Handles match discovery and selection from ESPN data.
    
    Supports:
    - Listing upcoming matches
    - Filtering by date, competition, status
    - Selecting by index or event_id
    - Finding teams in matches
    """
    
    def __init__(self, espn_client: Optional[EspnClient] = None):
        """
        Initialize match selector.
        
        Args:
            espn_client: ESPN client instance. Creates default if None.
        """
        self.client = espn_client or EspnClient()
    
    def get_upcoming_matches(
        self,
        limit: int = 20,
        dates: Optional[str] = None,
        status_filter: Optional[str] = None
    ) -> List[UpcomingMatch]:
        """
        Get list of upcoming matches from ESPN.
        
        Args:
            limit: Maximum number of matches to return
            dates: Date range filter (YYYYMMDD format)
            status_filter: Filter by status ('pre', 'in', 'post')
            
        Returns:
            List of UpcomingMatch objects
            
        Raises:
            MatchSelectionError: If no matches found or API fails
        """
        try:
            scoreboard = self.client.get_scoreboard(dates=dates, limit=limit)
        except EspnApiError as e:
            raise MatchSelectionError(f"Failed to fetch upcoming matches: {e}")
        
        events = scoreboard.get("events", [])
        if not events:
            logger.warning("No events returned from ESPN scoreboard")
            return []
        
        matches = []
        for event in events:
            match = parse_scoreboard_event(event)
            if match:
                # Apply status filter
                if status_filter is None or match.status == status_filter:
                    matches.append(match)
        
        logger.info(f"Found {len(matches)} matches from ESPN")
        return matches
    
    def select_by_index(
        self,
        matches: List[UpcomingMatch],
        index: int
    ) -> UpcomingMatch:
        """
        Select a match by index from a list.
        
        Args:
            matches: List of available matches
            index: 0-based index
            
        Returns:
            Selected UpcomingMatch
            
        Raises:
            MatchSelectionError: If index is out of range
        """
        if not matches:
            raise MatchSelectionError("No matches available to select from")
        
        if index < 0 or index >= len(matches):
            raise MatchSelectionError(
                f"Invalid index {index}. Must be between 0 and {len(matches) - 1}",
                available_matches=matches
            )
        
        return matches[index]
    
    def select_by_event_id(self, event_id: str) -> UpcomingMatch:
        """
        Select a match by ESPN event ID.
        
        Args:
            event_id: ESPN event ID
            
        Returns:
            Selected UpcomingMatch
            
        Raises:
            MatchSelectionError: If match not found
        """
        # First try to get from summary
        try:
            summary = self.client.get_summary(event_id)
            if summary:
                # Extract match info from summary
                competitions = summary.get("competitions", [])
                if competitions:
                    comp = competitions[0]
                    competitors = comp.get("competitors", [])
                    
                    if len(competitors) >= 2:
                        home_comp = next((c for c in competitors if c.get("homeAway") == "home"), competitors[0])
                        away_comp = next((c for c in competitors if c.get("homeAway") == "away"), competitors[1])
                        
                        home_team = home_comp.get("team", {}).get("displayName", "")
                        away_team = away_comp.get("team", {}).get("displayName", "")
                        
                        return UpcomingMatch(
                            event_id=event_id,
                            date=summary.get("date", ""),
                            competition=summary.get("league", {}).get("name", ""),
                            stage=None,  # Would need more parsing
                            status="pre",  # Default
                            home_team=home_team,
                            away_team=away_team,
                        )
        except EspnApiError:
            pass
        
        # Fallback: search in scoreboard
        matches = self.get_upcoming_matches(limit=100)
        for match in matches:
            if match.event_id == event_id:
                return match
        
        raise MatchSelectionError(f"Match with event_id '{event_id}' not found")
    
    def find_matches_by_team(
        self,
        team_name: str,
        limit: int = 20
    ) -> List[UpcomingMatch]:
        """
        Find matches involving a specific team.
        
        Args:
            team_name: Team name to search for
            limit: Maximum matches to return
            
        Returns:
            List of matches involving the team
        """
        matches = self.get_upcoming_matches(limit=limit)
        team_lower = team_name.lower()
        
        return [
            m for m in matches
            if team_lower in m.home_team.lower() or team_lower in m.away_team.lower()
        ]
    
    def format_for_display(
        self,
        matches: List[UpcomingMatch],
        max_display: int = 20
    ) -> List[dict]:
        """
        Format matches for tabular display.
        
        Args:
            matches: List of matches to format
            max_display: Maximum number to format
            
        Returns:
            List of dicts suitable for table display
        """
        formatted = []
        for i, match in enumerate(matches[:max_display]):
            row = match.to_display_row()
            row["index"] = i
            formatted.append(row)
        return formatted
    
    def get_match_context(self, event_id: str) -> dict:
        """
        Get detailed match context from ESPN summary.
        
        Args:
            event_id: ESPN event ID
            
        Returns:
            Dict with match context including teams, venue, etc.
            
        Raises:
            EspnApiError: If summary fetch fails
        """
        summary = self.client.get_summary(event_id)
        if not summary:
            raise EspnApiError(f"No summary found for event {event_id}")
        
        # Try multiple locations for competitions
        competitions = summary.get("competitions", [])
        if not competitions:
            # Fallback to header.competitions (new ESPN API format)
            header = summary.get("header", {})
            competitions = header.get("competitions", [])
        
        if not competitions:
            raise EspnApiError(f"No competitions in summary for event {event_id}")
        
        comp = competitions[0]
        competitors = comp.get("competitors", [])
        
        home_comp = next((c for c in competitors if c.get("homeAway") == "home"), competitors[0] if competitors else {})
        away_comp = next((c for c in competitors if c.get("homeAway") == "away"), competitors[1] if len(competitors) > 1 else {})
        
        # Get header for fallback league info
        header = summary.get("header", {})
        
        return {
            "event_id": event_id,
            "date": summary.get("date", ""),
            "competition": summary.get("league", {}).get("name", "") or header.get("league", {}).get("name", ""),
            "home_team": home_comp.get("team", {}).get("displayName", ""),
            "away_team": away_comp.get("team", {}).get("displayName", ""),
            "venue": comp.get("venue", {}).get("fullName"),
            "status": comp.get("status", {}).get("type", {}).get("name", ""),
            "raw": summary,
        }
