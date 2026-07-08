#!/usr/bin/env python
"""
ESPN Match Events Module - Minute-by-minute match timeline extraction.

This module provides robust access to chronological match events (goals, cards,
corners, fouls, substitutions, offsides, etc.) using ESPN's hidden API for soccer.

Sources (in priority order):
1. Primary: site.api / site.web.api summary -> commentary
2. Secondary: summary -> keyEvents (highlights)
3. Experimental: sports.core.api -> plays (may not be available for soccer)

The commentary source is the most reliable for soccer minute-by-minute data.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple
from enum import Enum

logger = logging.getLogger(__name__)


class EventSource(Enum):
    """Available sources for match events."""
    COMMENTARY = "commentary"
    KEY_EVENTS = "keyEvents"
    CORE_PLAYS = "core_plays"
    AUTO = "auto"  # Automatically choose best available


# Mapping of ESPN commentary types to normalized event types
COMMENTARY_TYPE_MAPPING = {
    # General match states
    "kickoff": "kickoff",
    "first_half_begins": "kickoff",
    "second_half_begins": "second_half_start",
    "halftime": "halftime",
    "fulltime": "fulltime",
    "match_ends": "fulltime",
    "first_half_ends": "halftime",
    
    # Goals
    "goal": "goal",
    "goal_scored": "goal",
    "own_goal": "own_goal",
    "penalty_goal": "goal",
    "penalty_missed": "shot_blocked",
    
    # Cards
    "yellow_card": "yellow_card",
    "red_card": "red_card",
    "second_yellow": "red_card",
    
    # Substitutions
    "substitution": "substitution",
    "substitution_off": "substitution",
    "substitution_on": "substitution",
    
    # Set pieces
    "corner": "corner",
    "free_kick": "free_kick",
    "penalty_awarded": "penalty",
    
    # Other events
    "foul": "foul",
    "offside": "offside",
    "shot_on_target": "shot_on_target",
    "shot_off_target": "shot_off_target",
    "shot_blocked": "shot_blocked",
    "hit_woodwork": "hit_woodwork",
    
    # VAR
    "var_decision": "var_decision",
    "var_check": "var_decision",
    
    # Time
    "added_time_announced": "added_time_announced",
    "injury_delay": "injury_delay",
    
    # Lineups
    "lineups_announced": "lineups_announced",
}

# Mapping of keyEvents types to normalized event types
KEY_EVENTS_TYPE_MAPPING = {
    "goal": "goal",
    "own-goal": "own_goal",
    "penalty-goal": "goal",
    "penalty-miss": "shot_blocked",
    "yellow-card": "yellow_card",
    "red-card": "red_card",
    "second-yellow-card": "red_card",
    "substitution": "substitution",
    "corner": "corner",
    "free-kick": "free_kick",
    "penalty-awarded": "penalty",
    "foul": "foul",
    "offside": "offside",
    "shot-on-target": "shot_on_target",
    "shot-off-target": "shot_off_target",
    "shot-blocked": "shot_blocked",
    "hit-woodwork": "hit_woodwork",
    "var": "var_decision",
    "kickoff": "kickoff",
    "half-time": "halftime",
    "full-time": "fulltime",
    "start-of-second-half": "second_half_start",
}


def fetch_match_summary(event_id: str, league: str = "fifa.world") -> Dict[str, Any]:
    """
    Fetch match summary from ESPN site API.
    
    Args:
        event_id: The ESPN event ID for the match
        league: League slug (e.g., 'fifa.world')
    
    Returns:
        Raw JSON response from ESPN summary endpoint
    
    Raises:
        ValueError: If event_id is invalid
        Exception: If HTTP request fails
    """
    if not event_id or not event_id.strip():
        raise ValueError("Event ID cannot be empty")
    
    # Import here to avoid circular imports
    from src.data.espn_client_v2 import EspnClient
    
    client = EspnClient(league=league)
    summary = client.get_summary(event_id)
    
    if not summary:
        raise ValueError(f"Empty response from ESPN for event {event_id}")
    
    # Check for error indicators in response
    if isinstance(summary, dict) and summary.get("error"):
        raise ValueError(
            f"ESPN returned error for event {event_id}: {summary.get('message', 'Unknown error')}"
        )
    
    return summary


def fetch_match_core_plays(event_id: str, league: str = "fifa.world") -> Optional[Dict[str, Any]]:
    """
    Attempt to fetch plays from ESPN Core API (experimental).
    
    Note: This endpoint may not be available for all soccer leagues/events.
    It's included as an experimental fallback but commentary is preferred.
    
    Args:
        event_id: The ESPN event ID for the match
        league: League slug (e.g., 'fifa.world')
    
    Returns:
        Raw JSON response if available, None otherwise
    """
    try:
        from src.data.espn_client_v2 import EspnClient
        
        client = EspnClient(league=league)
        
        # Try the core API plays endpoint
        # Format: https://sports.core.api.espn.com/apis/site/v2/sports/{sport}/{league}/events/{event_id}/competitions/{competition_id}/plays
        # We need to get competition_id first from summary
        summary = fetch_match_summary(event_id, league)
        
        competitions = summary.get("competitions", [])
        if not competitions:
            logger.debug(f"No competitions found for event {event_id}")
            return None
        
        competition_id = competitions[0].get("id", "")
        if not competition_id:
            logger.debug(f"No competition ID found for event {event_id}")
            return None
        
        # Build core API URL
        base_core_url = "https://sports.core.api.espn.com/apis/site/v2/sports"
        url = f"{base_core_url}/{client.config['sport']}/{league}/events/{event_id}/competitions/{competition_id}/plays"
        
        # Make request without caching (experimental)
        result = client._make_request(url, {}, use_cache=False)
        
        if result and isinstance(result, dict):
            logger.info(f"Successfully fetched core plays for event {event_id}")
            return result
        else:
            logger.debug(f"Core plays returned empty/null for event {event_id}")
            return None
            
    except Exception as e:
        logger.debug(f"Core plays fetch failed for event {event_id}: {e}")
        return None


def _parse_clock_display(clock_str: Optional[str]) -> int:
    """
    Parse clock display string to minutes.
    
    Examples:
        "0'" -> 0
        "45+2'" -> 47
        "90+4'" -> 94
        "HT" -> 45
        "FT" -> 90
        "90'+2'" -> 92
    
    Args:
        clock_str: Clock display string from ESPN
    
    Returns:
        Minutes as integer
    """
    if not clock_str:
        return 0
    
    # Remove trailing apostrophe and strip
    clock_str = str(clock_str).strip().rstrip("'")
    
    # Handle halftime/fulltime
    if clock_str.upper() == "HT":
        return 45
    if clock_str.upper() == "FT":
        return 90
    
    # Handle added time with apostrophe inside (e.g., "90'+2" or "45'+1")
    # First try pattern like "90'+2" 
    if "'" in clock_str and "+" in clock_str:
        parts = clock_str.split("+")
        try:
            base_part = parts[0].rstrip("'")
            added_part = parts[1].rstrip("'")
            base = int(base_part)
            added = int(added_part) if added_part else 0
            return base + added
        except (ValueError, IndexError):
            pass
    
    # Handle standard added time (e.g., "45+2")
    if "+" in clock_str:
        parts = clock_str.split("+")
        try:
            base = int(parts[0])
            added = int(parts[1]) if len(parts) > 1 else 0
            return base + added
        except (ValueError, IndexError):
            pass
    
    # Regular minutes - try to extract just the number
    try:
        # Remove any non-numeric characters except + which we already handled
        cleaned = ''.join(c for c in clock_str if c.isdigit())
        return int(cleaned) if cleaned else 0
    except ValueError:
        return 0


def _normalize_commentary_event(
    raw_event: Dict[str, Any],
    sequence_index: int,
    event_id: str
) -> Dict[str, Any]:
    """
    Normalize a commentary event to standard schema.
    
    Args:
        raw_event: Raw event dict from ESPN commentary
        sequence_index: Position in timeline
        event_id: Match event ID
    
    Returns:
        Normalized event dict
    """
    # Extract basic info - check both top-level and nested 'play' object
    play_data = raw_event.get("play", {}) or {}
    
    # Get type from play object first, then fall back to top-level
    type_info = play_data.get("type", {}) or raw_event.get("type", {})
    event_type_raw = type_info.get("text", "").lower().replace(" ", "_").replace("-", "_")
    if not event_type_raw:
        event_type_raw = type_info.get("type", "").lower().replace(" ", "_").replace("-", "_")
    if not event_type_raw:
        event_type_raw = type_info.get("name", "").lower().replace(" ", "_").replace("-", "_")
    
    description = raw_event.get("text") or raw_event.get("description", "") or play_data.get("text", "")
    
    # Get timing info - try time.displayValue/time.value first (commentary format), then clock
    time_data = raw_event.get("time", {}) or {}
    clock_data = raw_event.get("clock", {}) or play_data.get("clock", {})
    
    # Prefer displayValue for display, value for calculation
    display_value = time_data.get("displayValue") or clock_data.get("displayValue") or clock_data.get("displayTime", "")
    time_value_seconds = time_data.get("value") or clock_data.get("value")
    
    # Get period
    period_data = raw_event.get("period", {}) or play_data.get("period", {})
    period = period_data.get("number", 1)
    
    # Parse minute from display value or calculate from seconds
    if display_value:
        minute = _parse_clock_display(str(display_value))
        clock_display = str(display_value)
    elif time_value_seconds is not None:
        # Convert seconds to minutes
        minute = int(time_value_seconds // 60)
        clock_display = f"{minute}'"
    else:
        minute = 0
        clock_display = None
    
    # Get team info - check play data first
    team_data = play_data.get("team", {}) or raw_event.get("team", {}) or {}
    team_name = team_data.get("displayName") or team_data.get("name", "")
    team_abbr = team_data.get("abbreviation") or team_data.get("shortName", "")
    
    # Get player info - check participants array first (most common in commentary)
    player_name = ""
    participants = play_data.get("participants", [])
    if participants:
        for p in participants:
            athlete = p.get("athlete", p)
            player_name = athlete.get("fullName") or athlete.get("displayName", "")
            if player_name:
                break
    
    # Also check direct player field
    if not player_name:
        player_data = play_data.get("player", {}) or raw_event.get("player", {}) or {}
        player_name = player_data.get("fullName") or player_data.get("displayName", "")
    
    # Map event type
    event_type = COMMENTARY_TYPE_MAPPING.get(event_type_raw, "unknown")
    
    # Special handling for certain event types based on description - priority over mapping
    desc_lower = description.lower()
    
    # Check for goals first (highest priority) - must have "Goal!" pattern
    if "goal!" in desc_lower:
        if "own goal" in desc_lower or "autogol" in desc_lower:
            event_type = "own_goal"
        else:
            event_type = "goal"
    elif "penalty" in desc_lower and "goal" in desc_lower:
        event_type = "goal"
    elif "var" in desc_lower and ("decision" in desc_lower or "check" in desc_lower):
        event_type = "var_decision"
    elif "yellow card" in desc_lower or "tarjeta amarilla" in desc_lower or "shown the yellow card" in desc_lower:
        event_type = "yellow_card"
    elif "red card" in desc_lower or "tarjeta roja" in desc_lower or "shown the red card" in desc_lower:
        event_type = "red_card"
    elif "substitution" in desc_lower or ("cambio" in desc_lower and "entra" in desc_lower) or "comes on" in desc_lower or "replaces" in desc_lower:
        event_type = "substitution"
    elif "corner" in desc_lower and "wins" not in desc_lower:
        event_type = "corner"
    elif "offside" in desc_lower:
        event_type = "offside"
    elif "foul" in desc_lower and "wins a free kick" in desc_lower:
        event_type = "foul"
    elif "attempt saved" in desc_lower or "shot saved" in desc_lower:
        event_type = "shot_on_target"
    elif "attempt missed" in desc_lower or "shot missed" in desc_lower:
        event_type = "shot_off_target"
    elif "attempt blocked" in desc_lower or "shot blocked" in desc_lower:
        event_type = "shot_blocked"
    elif "hit the bar" in desc_lower or "hit the post" in desc_lower or "woodwork" in desc_lower:
        event_type = "hit_woodwork"
    elif "free kick" in desc_lower and "wins" not in desc_lower:
        event_type = "free_kick"
    elif "injury" in desc_lower or "delay in match" in desc_lower:
        event_type = "injury_delay"
    elif "lineups" in desc_lower:
        event_type = "lineups_announced"
    elif "first half begins" in desc_lower or "kick-off" in desc_lower or "kickoff" in desc_lower:
        event_type = "kickoff"
    elif "first half ends" in desc_lower:
        event_type = "halftime"
    elif "second half begins" in desc_lower:
        event_type = "second_half_start"
    elif "match ends" in desc_lower or "full-time" in desc_lower or "fulltime" in desc_lower:
        event_type = "fulltime"
    elif "second half ends" in desc_lower:
        event_type = "fulltime"
    elif "added time" in desc_lower or "minutes of added time" in desc_lower:
        event_type = "added_time_announced"
    
    return {
        "event_id": event_id,
        "sequence_index": sequence_index,
        "minute": minute,
        "clock_display": clock_display if clock_display else None,
        "period": period,
        "event_type": event_type,
        "team_name": team_name or None,
        "team_abbr": team_abbr or None,
        "player_name": player_name or None,
        "description": description or None,
        "source": "commentary",
        "raw_event": raw_event,
    }


def _normalize_key_event(
    raw_event: Dict[str, Any],
    sequence_index: int,
    event_id: str
) -> Dict[str, Any]:
    """
    Normalize a keyEvent to standard schema.
    
    Args:
        raw_event: Raw event dict from ESPN keyEvents
        sequence_index: Position in timeline
        event_id: Match event ID
    
    Returns:
        Normalized event dict
    """
    # Extract basic info - get type from nested type object
    type_info = raw_event.get("type", {}) or {}
    event_type_raw = type_info.get("text", "").lower().replace(" ", "-").replace("_", "-")
    if not event_type_raw:
        event_type_raw = type_info.get("type", "").lower().replace(" ", "-").replace("_", "-")
    if not event_type_raw:
        # Fallback to typeId if present
        event_type_raw = raw_event.get("typeId", "")
        if isinstance(event_type_raw, int):
            event_type_raw = str(event_type_raw)
        else:
            event_type_raw = str(event_type_raw).lower().replace(" ", "-").replace("_", "-")
    
    description = raw_event.get("text") or raw_event.get("description", "")
    
    # Get timing info - use clock.displayValue and clock.value
    clock_data = raw_event.get("clock", {})
    display_value = clock_data.get("displayValue", "")
    time_value_seconds = clock_data.get("value")
    
    period_data = raw_event.get("period", {})
    period = period_data.get("number", 1) if isinstance(period_data, dict) else 1
    
    # Parse minute from display value or calculate from seconds
    if display_value:
        minute = _parse_clock_display(str(display_value))
        clock_display = str(display_value)
    elif time_value_seconds is not None:
        minute = int(time_value_seconds // 60)
        clock_display = f"{minute}'"
    else:
        # Fallback to minute field
        minute_val = raw_event.get("minute", 0)
        if isinstance(minute_val, dict):
            minute_val = minute_val.get("displayValue", 0) or 0
        minute = int(minute_val) if isinstance(minute_val, (int, float)) else 0
        clock_display = f"{minute}'" if minute else None
    
    # Get team info
    team_data = raw_event.get("team", {}) or {}
    team_name = team_data.get("displayName") or team_data.get("name", "")
    team_abbr = team_data.get("abbreviation") or team_data.get("shortName", "")
    
    # Get player info - keyEvents often have participants
    participants = raw_event.get("participants", [])
    player_name = ""
    for p in participants:
        athlete = p.get("athlete", p)
        if athlete:
            player_name = athlete.get("fullName") or athlete.get("displayName", "")
            if player_name:
                break
    
    if not player_name:
        # Try direct player field
        player_data = raw_event.get("player", {}) or {}
        player_name = player_data.get("fullName") or player_data.get("displayName", "")
    
    # Map event type
    event_type = KEY_EVENTS_TYPE_MAPPING.get(event_type_raw, "unknown")
    
    # Fallback: try to infer from description and type text
    if event_type == "unknown" and description:
        desc_lower = description.lower()
        type_text = type_info.get("text", "").lower()
        
        if "goal" in desc_lower or type_text == "goal":
            event_type = "goal"
        elif "own goal" in desc_lower or type_text == "own-goal":
            event_type = "own_goal"
        elif "yellow card" in desc_lower or "tarjeta amarilla" in desc_lower or type_text == "yellow-card":
            event_type = "yellow_card"
        elif "red card" in desc_lower or "tarjeta roja" in desc_lower or type_text == "red-card":
            event_type = "red_card"
        elif "substitution" in desc_lower or "cambio" in desc_lower or type_text == "substitution":
            event_type = "substitution"
        elif "corner" in desc_lower or type_text == "corner":
            event_type = "corner"
        elif "offside" in desc_lower or type_text == "offside":
            event_type = "offside"
        elif "foul" in desc_lower or type_text == "foul":
            event_type = "foul"
        elif "kickoff" in type_text or "start of first half" in desc_lower:
            event_type = "kickoff"
        elif "half-time" in type_text or "first half ends" in desc_lower:
            event_type = "halftime"
        elif "full-time" in type_text or "match ends" in desc_lower:
            event_type = "fulltime"
        elif "second half" in type_text or "second half begins" in desc_lower:
            event_type = "second_half_start"
    
    return {
        "event_id": event_id,
        "sequence_index": sequence_index,
        "minute": minute,
        "clock_display": clock_display if clock_display else None,
        "period": int(period) if isinstance(period, (int, float)) else 1,
        "event_type": event_type,
        "team_name": team_name or None,
        "team_abbr": team_abbr or None,
        "player_name": player_name or None,
        "description": description or None,
        "source": "keyEvents",
        "raw_event": raw_event,
    }


def extract_commentary_events(summary: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Extract and normalize commentary events from match summary.
    
    Args:
        summary: Raw JSON summary from ESPN
    
    Returns:
        List of normalized event dicts
    """
    commentary = summary.get("commentary", [])
    
    if not commentary or not isinstance(commentary, list):
        logger.debug("No commentary found in summary")
        return []
    
    event_id = summary.get("header", {}).get("id", "") or summary.get("event_id", "")
    
    events = []
    for idx, raw_event in enumerate(commentary):
        if not isinstance(raw_event, dict):
            continue
        
        try:
            normalized = _normalize_commentary_event(raw_event, idx, event_id)
            events.append(normalized)
        except Exception as e:
            logger.warning(f"Failed to normalize commentary event at index {idx}: {e}")
            continue
    
    logger.info(f"Extracted {len(events)} commentary events")
    return events


def extract_key_events(summary: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Extract and normalize keyEvents from match summary.
    
    Args:
        summary: Raw JSON summary from ESPN
    
    Returns:
        List of normalized event dicts
    """
    key_events = summary.get("keyEvents", [])
    
    if not key_events or not isinstance(key_events, list):
        logger.debug("No keyEvents found in summary")
        return []
    
    event_id = summary.get("header", {}).get("id", "") or summary.get("event_id", "")
    
    events = []
    for idx, raw_event in enumerate(key_events):
        if not isinstance(raw_event, dict):
            continue
        
        try:
            normalized = _normalize_key_event(raw_event, idx, event_id)
            events.append(normalized)
        except Exception as e:
            logger.warning(f"Failed to normalize keyEvent at index {idx}: {e}")
            continue
    
    logger.info(f"Extracted {len(events)} keyEvents")
    return events


def extract_core_plays_events(core_plays: Dict[str, Any], event_id: str) -> List[Dict[str, Any]]:
    """
    Extract and normalize events from Core API plays response.
    
    Args:
        core_plays: Raw JSON from Core API plays endpoint
        event_id: Match event ID
    
    Returns:
        List of normalized event dicts
    """
    if not core_plays:
        return []
    
    # Core plays structure varies; typically has a 'plays' array
    plays = core_plays.get("plays", [])
    
    if not plays or not isinstance(plays, list):
        logger.debug("No plays array found in core plays response")
        return []
    
    events = []
    for idx, raw_play in enumerate(plays):
        if not isinstance(raw_play, dict):
            continue
        
        # Extract basic info
        play_type = raw_play.get("type", {}).get("text", "").lower().replace(" ", "_")
        description = raw_play.get("text", "")
        
        # Timing
        period = raw_play.get("period", {}).get("number", 1)
        clock = raw_play.get("clock", {}).get("displayTime", "")
        minute = _parse_clock_display(str(clock)) if clock else 0
        
        # Team/player info
        team = raw_play.get("team", {})
        team_name = team.get("displayName") if isinstance(team, dict) else ""
        
        athletes = raw_play.get("athletes", [])
        player_name = ""
        if athletes and isinstance(athletes, list):
            athlete = athletes[0] if isinstance(athletes[0], dict) else {}
            player_name = athlete.get("fullName", "")
        
        # Map event type (simplified mapping for core plays)
        event_type = "unknown"
        if "goal" in play_type or "goal" in description.lower():
            event_type = "goal"
        elif "card" in play_type or "card" in description.lower():
            event_type = "yellow_card" if "yellow" in description.lower() else "red_card"
        elif "substitution" in play_type or "sub" in description.lower():
            event_type = "substitution"
        elif "corner" in play_type:
            event_type = "corner"
        elif "foul" in play_type:
            event_type = "foul"
        elif "offside" in play_type:
            event_type = "offside"
        
        events.append({
            "event_id": event_id,
            "sequence_index": idx,
            "minute": minute,
            "clock_display": str(clock) if clock else None,
            "period": int(period) if isinstance(period, (int, float)) else 1,
            "event_type": event_type,
            "team_name": team_name or None,
            "team_abbr": None,
            "player_name": player_name or None,
            "description": description or None,
            "source": "core_plays",
            "raw_event": raw_play,
        })
    
    logger.info(f"Extracted {len(events)} core plays events")
    return events


def _merge_events(
    commentary_events: List[Dict[str, Any]],
    key_events: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """
    Merge commentary and keyEvents, avoiding duplicates.
    
    Strategy:
    - Use commentary as primary source
    - Add keyEvents that are missing from commentary (by matching minute+type+team)
    
    Args:
        commentary_events: Events from commentary
        key_events: Events from keyEvents
    
    Returns:
        Merged and deduplicated event list
    """
    if not key_events:
        return commentary_events
    
    if not commentary_events:
        return key_events
    
    # Create signature set for commentary events
    commentary_signatures = set()
    for evt in commentary_events:
        sig = (evt["minute"], evt["event_type"], evt.get("team_abbr"))
        commentary_signatures.add(sig)
    
    # Merge: start with commentary, add unique keyEvents
    merged = list(commentary_events)
    
    for evt in key_events:
        sig = (evt["minute"], evt["event_type"], evt.get("team_abbr"))
        if sig not in commentary_signatures:
            # Add with updated sequence index
            evt_copy = evt.copy()
            evt_copy["sequence_index"] = len(merged)
            merged.append(evt_copy)
            commentary_signatures.add(sig)
    
    # Sort by minute and sequence
    merged.sort(key=lambda x: (x["minute"], x["sequence_index"]))
    
    # Re-index sequence
    for idx, evt in enumerate(merged):
        evt["sequence_index"] = idx
    
    return merged


def normalize_match_events(
    summary: Dict[str, Any],
    prefer_source: str = "commentary",
    include_key_events_fallback: bool = True
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Normalize match events from summary with source preference.
    
    Args:
        summary: Raw JSON summary from ESPN
        prefer_source: Preferred source ('commentary', 'keyEvents', 'auto')
        include_key_events_fallback: Whether to merge keyEvents as fallback
    
    Returns:
        Tuple of (normalized_events_list, source_metadata_dict)
    """
    event_id = summary.get("header", {}).get("id", "")
    
    # Extract from all available sources
    commentary_events = extract_commentary_events(summary)
    key_events = extract_key_events(summary)
    
    # Build source metadata
    source_meta = {
        "commentary_available": len(commentary_events) > 0,
        "commentary_count": len(commentary_events),
        "key_events_available": len(key_events) > 0,
        "key_events_count": len(key_events),
        "core_plays_available": False,
    }
    
    # Determine which source(s) to use
    if prefer_source == "auto":
        # Auto: prefer commentary, fall back to keyEvents
        if commentary_events:
            events = commentary_events
        elif key_events:
            events = key_events
            source_meta["used_source"] = "keyEvents_fallback"
        else:
            events = []
    elif prefer_source == "commentary":
        if commentary_events:
            events = commentary_events
        elif key_events and include_key_events_fallback:
            events = key_events
            source_meta["used_source"] = "keyEvents_fallback"
        else:
            events = []
    elif prefer_source == "keyEvents":
        events = key_events if key_events else commentary_events
        if key_events:
            source_meta["used_source"] = "keyEvents"
    else:
        events = commentary_events
    
    # Merge if both available and requested
    if include_key_events_fallback and commentary_events and key_events and prefer_source != "keyEvents":
        events = _merge_events(commentary_events, key_events)
        source_meta["used_source"] = "merged"
    
    # Sort by minute
    events.sort(key=lambda x: (x["minute"], x["sequence_index"]))
    
    # Re-index after sorting
    for idx, evt in enumerate(events):
        evt["sequence_index"] = idx
    
    source_meta["used_source"] = source_meta.get("used_source", prefer_source)
    source_meta["total_events"] = len(events)
    
    return events, source_meta


def get_match_event_timeline(
    event_id: str,
    league: str = "fifa.world",
    prefer_source: str = "commentary"
) -> Dict[str, Any]:
    """
    Get complete match event timeline with metadata.
    
    This is the main entry point for getting a match timeline.
    
    Args:
        event_id: ESPN event ID
        league: League slug (default: 'fifa.world')
        prefer_source: Preferred source ('commentary', 'keyEvents', 'auto', 'core')
    
    Returns:
        Complete timeline dict with match info, sources metadata, and events
    
    Example:
        {
          "event_id": "760500",
          "league": "fifa.world",
          "match": { ... },
          "sources": { ... },
          "events": [ ... ]
        }
    """
    # Fetch summary (primary source)
    try:
        summary = fetch_match_summary(event_id, league)
    except Exception as e:
        logger.error(f"Failed to fetch summary for event {event_id}: {e}")
        raise
    
    # Extract match context from header
    header = summary.get("header", {})
    competitions = header.get("competitions", [])
    
    match_info = {
        "short_name": "",
        "date": header.get("date"),
        "status": "",
        "home_team": "",
        "away_team": "",
        "home_score": None,
        "away_score": None,
    }
    
    if competitions:
        comp = competitions[0]
        
        # Status
        status_type = comp.get("status", {}).get("type", {})
        match_info["status"] = status_type.get("name") or status_type.get("state", "")
        
        # Teams and scores
        competitors = comp.get("competitors", [])
        for c in competitors:
            home_away = c.get("homeAway", "")
            team_name = c.get("team", {}).get("displayName", "")
            score = c.get("score")
            
            if home_away == "home":
                match_info["home_team"] = team_name
                match_info["home_score"] = float(score) if score else None
            elif home_away == "away":
                match_info["away_team"] = team_name
                match_info["away_score"] = float(score) if score else None
        
        # Short name
        if match_info["home_team"] and match_info["away_team"]:
            match_info["short_name"] = f"{match_info['home_team']} vs {match_info['away_team']}"
    
    # Get events based on source preference
    if prefer_source == "core":
        # Try core plays first
        core_plays = fetch_match_core_plays(event_id, league)
        if core_plays:
            events = extract_core_plays_events(core_plays, event_id)
            source_meta = {
                "commentary_available": len(extract_commentary_events(summary)) > 0,
                "commentary_count": len(extract_commentary_events(summary)),
                "key_events_available": len(extract_key_events(summary)) > 0,
                "key_events_count": len(extract_key_events(summary)),
                "core_plays_available": True,
                "used_source": "core_plays",
                "total_events": len(events),
            }
        else:
            # Fall back to summary sources
            events, source_meta = normalize_match_events(summary, "auto", True)
    else:
        events, source_meta = normalize_match_events(summary, prefer_source, True)
    
    return {
        "event_id": event_id,
        "league": league,
        "match": match_info,
        "sources": source_meta,
        "events": events,
    }


def filter_events_by_type(
    events: List[Dict[str, Any]],
    event_types: List[str]
) -> List[Dict[str, Any]]:
    """
    Filter events by event type.
    
    Args:
        events: List of normalized events
        event_types: List of event types to keep
    
    Returns:
        Filtered list of events
    """
    return [evt for evt in events if evt["event_type"] in event_types]
