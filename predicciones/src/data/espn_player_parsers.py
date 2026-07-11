"""
ESPN Player Data Parsers.

Extracts player-level data from ESPN summary responses including:
- Rosters (lineups, positions, starters)
- Leaders (top performers in various categories)
- Boxscore player stats
- Key events involving players (goals, cards, substitutions)

Supports partial availability - doesn't require perfect stats to consider
player data as available.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


def extract_roster_players(summary: dict) -> list[dict]:
    """
    Extract roster players from ESPN summary.
    
    Looks for player roster information in:
    - boxscore.teams[].roster
    - boxscore.players
    - competitions[0].competitors[].roster
    
    Returns list of player dicts with:
    - athlete_id
    - player_name
    - team_id
    - team_name
    - home_away
    - position
    - jersey
    - is_starter (if inferable)
    - formation_slot (if available)
    
    Args:
        summary: Raw ESPN summary dict
        
    Returns:
        List of player roster dicts
    """
    if not summary:
        return []
    
    players = []
    seen_ids: Set[str] = set()
    
    # Try boxscore first
    boxscore = summary.get("boxscore", {})
    
    # Format 1: boxscore.teams[].roster
    teams_data = boxscore.get("teams", [])
    for team_block in teams_data:
        home_away = team_block.get("homeAway", "")
        team_info = team_block.get("team", {})
        team_id = str(team_info.get("id", ""))
        team_name = team_info.get("displayName", "")
        
        roster = team_block.get("roster", [])
        if isinstance(roster, list):
            for p in roster:
                player_data = _parse_roster_player(p, home_away, team_id, team_name)
                if player_data and player_data["athlete_id"] not in seen_ids:
                    seen_ids.add(player_data["athlete_id"])
                    players.append(player_data)
        
        # Also check direct players array in team block
        team_players = team_block.get("players", [])
        if isinstance(team_players, list):
            for p in team_players:
                player_data = _parse_roster_player(p, home_away, team_id, team_name)
                if player_data and player_data["athlete_id"] not in seen_ids:
                    seen_ids.add(player_data["athlete_id"])
                    players.append(player_data)
    
    # Format 2: boxscore.players (flat list)
    boxscore_players = boxscore.get("players", [])
    if isinstance(boxscore_players, list):
        for p in boxscore_players:
            athlete = p.get("athlete", {})
            team_ref = p.get("team", {}) or athlete.get("team", {})
            home_away = p.get("homeAway", "") or team_ref.get("homeAway", "")
            team_id = str(team_ref.get("id", ""))
            team_name = team_ref.get("displayName", "")
            
            player_data = _parse_roster_player(p, home_away, team_id, team_name)
            if player_data and player_data["athlete_id"] not in seen_ids:
                seen_ids.add(player_data["athlete_id"])
                players.append(player_data)
    
    # Format 3: competitions[0].competitors[].roster
    competitions = summary.get("competitions", [])
    if competitions:
        comp = competitions[0]
        competitors = comp.get("competitors", [])
        for competitor in competitors:
            home_away = competitor.get("homeAway", "")
            team_info = competitor.get("team", {})
            team_id = str(team_info.get("id", ""))
            team_name = team_info.get("displayName", "")
            
            roster = competitor.get("roster", [])
            if isinstance(roster, list):
                for p in roster:
                    player_data = _parse_roster_player(p, home_away, team_id, team_name)
                    if player_data and player_data["athlete_id"] not in seen_ids:
                        seen_ids.add(player_data["athlete_id"])
                        players.append(player_data)
    
    return players


def _parse_roster_player(
    player_block: dict,
    home_away: str,
    team_id: str,
    team_name: str
) -> Optional[dict]:
    """Parse individual roster player block."""
    if not isinstance(player_block, dict):
        return None
    
    # Get athlete info - can be nested or flat
    athlete = player_block.get("athlete", {})
    if not isinstance(athlete, dict):
        athlete = {}
    
    # Athlete ID - check multiple locations
    athlete_id = (
        str(player_block.get("id") or "")
        or str(athlete.get("id") or "")
        or str(player_block.get("athleteId") or "")
    )
    
    if not athlete_id:
        return None
    
    # Player name
    player_name = (
        player_block.get("displayName")
        or athlete.get("displayName")
        or player_block.get("name")
        or athlete.get("name")
        or player_block.get("shortName")
        or athlete.get("shortName")
        or ""
    )
    
    # Position
    position_raw = player_block.get("position") or athlete.get("position", {})
    if isinstance(position_raw, dict):
        position = position_raw.get("abbreviation") or position_raw.get("name") or ""
    else:
        position = str(position_raw) if position_raw else ""
    
    # Jersey number
    jersey = player_block.get("jersey") or athlete.get("jersey") or ""
    if jersey:
        jersey = str(jersey)
    
    # Starter status
    is_starter = (
        player_block.get("starter", False)
        or player_block.get("starterFlag", False)
        or athlete.get("starter", False)
    )
    
    # Formation slot / lineup position
    formation_slot = (
        player_block.get("formationSlot")
        or player_block.get("slot")
        or player_block.get("lineupPosition")
    )
    
    # Captain
    is_captain = player_block.get("captain", False)
    
    return {
        "athlete_id": athlete_id,
        "player_name": player_name,
        "team_id": team_id,
        "team_name": team_name,
        "home_away": home_away,
        "position": position,
        "jersey": jersey,
        "is_starter": bool(is_starter),
        "formation_slot": formation_slot,
        "is_captain": bool(is_captain),
    }


def extract_player_signals(summary: dict) -> list[dict]:
    """
    Extract player signals/leaders from ESPN summary.
    
    Uses leaders, boxscore stats, and other indicators to derive signals like:
    - is_goal_leader
    - is_shot_leader
    - is_key_attacker
    - has_recent_offensive_signal
    
    Args:
        summary: Raw ESPN summary dict
        
    Returns:
        List of player signal dicts
    """
    if not summary:
        return []
    
    signals = []
    seen_ids: Set[str] = set()
    
    # Extract from leaders
    leaders_data = summary.get("leaders", [])
    if isinstance(leaders_data, list):
        for leader_block in leaders_data:
            leader_signals = _parse_leader_block(leader_block)
            for sig in leader_signals:
                if sig["athlete_id"] not in seen_ids:
                    seen_ids.add(sig["athlete_id"])
                    signals.append(sig)
    
    # Also check boxscore for top performers
    boxscore = summary.get("boxscore", {})
    teams_data = boxscore.get("teams", [])
    for team_block in teams_data:
        home_away = team_block.get("homeAway", "")
        team_info = team_block.get("team", {})
        team_id = str(team_info.get("id", ""))
        team_name = team_info.get("displayName", "")
        
        # Check statistics for top performers
        stats_list = team_block.get("statistics", [])
        for stat_block in stats_list:
            if isinstance(stat_block, dict):
                stat_name = (stat_block.get("name") or "").lower()
                display_value = stat_block.get("displayValue")
                
                # Check if this stat has associated athletes
                athletes = stat_block.get("athletes", [])
                if athletes and isinstance(athletes, list):
                    for ath in athletes:
                        athlete_id = str(ath.get("id") or "")
                        if not athlete_id or athlete_id in seen_ids:
                            continue
                        
                        seen_ids.add(athlete_id)
                        athlete_name = ath.get("displayName") or ath.get("name", "")
                        
                        # Determine signal type based on stat name
                        signal_type = _infer_signal_from_stat(stat_name)
                        
                        signals.append({
                            "athlete_id": athlete_id,
                            "player_name": athlete_name,
                            "team_id": team_id,
                            "team_name": team_name,
                            "home_away": home_away,
                            "signal_type": signal_type,
                            "stat_name": stat_name,
                            "stat_value": display_value,
                        })
    
    return signals


def _parse_leader_block(leader_block: dict) -> list[dict]:
    """Parse a leaders block to extract player signals."""
    signals = []
    
    if not isinstance(leader_block, dict):
        return signals
    
    # Leader name/category
    leader_name = (leader_block.get("name") or "").lower()
    display_name = leader_block.get("displayName", "")
    
    # Get top performers
    top_performers = leader_block.get("topPerformers", [])
    if not isinstance(top_performers, list):
        return signals
    
    for performer in top_performers:
        if not isinstance(performer, dict):
            continue
        
        athlete = performer.get("athlete", {})
        if not isinstance(athlete, dict):
            continue
        
        athlete_id = str(athlete.get("id") or "")
        if not athlete_id:
            continue
        
        athlete_name = (
            athlete.get("displayName")
            or athlete.get("name")
            or athlete.get("shortName")
            or ""
        )
        
        team_ref = athlete.get("team", {})
        if isinstance(team_ref, dict):
            team_id = str(team_ref.get("id", ""))
            team_name = team_ref.get("displayName", "")
        else:
            team_id = ""
            team_name = ""
        
        home_away = performer.get("homeAway", "") or team_ref.get("homeAway", "")
        
        # Stat value
        stat_value = performer.get("value") or performer.get("displayValue", "")
        
        # Determine signal type
        signal_type = _infer_signal_from_stat(leader_name)
        
        signals.append({
            "athlete_id": athlete_id,
            "player_name": athlete_name,
            "team_id": team_id,
            "team_name": team_name,
            "home_away": home_away,
            "signal_type": signal_type,
            "stat_name": leader_name,
            "stat_value": stat_value,
            "is_leader": True,
        })
    
    return signals


def _infer_signal_from_stat(stat_name: str) -> str:
    """Infer signal type from stat name."""
    stat_lower = stat_name.lower()
    
    if "goal" in stat_lower and "own" not in stat_lower:
        return "goal_scorer"
    elif "assist" in stat_lower:
        return "playmaker"
    elif "shot" in stat_lower and "on target" in stat_lower:
        return "shot_accuracy"
    elif "shot" in stat_lower:
        return "volume_shooter"
    elif "save" in stat_lower:
        return "goalkeeper"
    elif "tackle" in stat_lower:
        return "defender"
    elif "pass" in stat_lower:
        return "distributor"
    else:
        return "general"


def extract_player_events(summary: dict) -> list[dict]:
    """
    Extract player events from keyEvents/plays.
    
    Parses events like:
    - goals (including own goals)
    - yellow/red cards
    - substitutions
    - assists (if in payload)
    
    Args:
        summary: Raw ESPN summary dict
        
    Returns:
        List of player event dicts
    """
    if not summary:
        return []
    
    events = []
    
    # Look for keyEvents
    key_events = summary.get("keyEvents", [])
    if not key_events:
        # Try plays
        key_events = summary.get("plays", [])
    if not key_events:
        # Try competitions[0].plays
        competitions = summary.get("competitions", [])
        if competitions:
            key_events = competitions[0].get("plays", [])
    
    if not isinstance(key_events, list):
        return events
    
    for event in key_events:
        if not isinstance(event, dict):
            continue
        
        event_type = _classify_event_type(event)
        if not event_type:
            continue
        
        # Get timing
        period = event.get("period", {}).get("number") or event.get("period", 1)
        clock = event.get("clock", {})
        if isinstance(clock, dict):
            time_display = clock.get("displayValue", "")
        else:
            time_display = str(clock) if clock else ""
        
        # Get involved athletes
        participants = event.get("participants", [])
        if not participants:
            # Try to get from text or other fields
            participants = _extract_participants_from_event(event)
        
        for participant in participants:
            if isinstance(participant, dict):
                athlete_id = str(participant.get("id") or "")
                athlete_name = participant.get("displayName") or participant.get("name", "")
                team_ref = participant.get("team", {})
                if isinstance(team_ref, dict):
                    team_id = str(team_ref.get("id", ""))
                else:
                    team_id = ""
                
                events.append({
                    "athlete_id": athlete_id,
                    "player_name": athlete_name,
                    "team_id": team_id,
                    "event_type": event_type,
                    "period": period,
                    "time_display": time_display,
                    "raw_event": event,
                })
    
    return events


def _classify_event_type(event: dict) -> Optional[str]:
    """Classify event type from event dict."""
    text = (event.get("text") or "").lower()
    type_info = event.get("type", {})
    if isinstance(type_info, dict):
        type_id = (type_info.get("id") or type_info.get("text") or "").lower()
    else:
        type_id = str(type_info).lower()
    
    # Goal
    if "goal" in text or type_id == "goal":
        if "own" in text:
            return "own_goal"
        return "goal"
    
    # Cards
    if "yellow card" in text or "tarjeta amarilla" in text:
        return "yellow_card"
    if "red card" in text or "tarjeta roja" in text:
        return "red_card"
    
    # Substitution
    if "substitut" in text or "cambio" in text or type_id == "substitution":
        return "substitution"
    
    # Assist (sometimes marked separately)
    if "assist" in text:
        return "assist"
    
    return None


def _extract_participants_from_event(event: dict) -> list[dict]:
    """Extract participants from event when not in standard format."""
    participants = []
    
    # Some events have athlete directly
    athlete = event.get("athlete", {})
    if isinstance(athlete, dict) and athlete.get("id"):
        participants.append(athlete)
    
    # Some have team and we can infer from context
    team = event.get("team", {})
    
    return participants


def build_player_match_rows(summary: dict) -> list[dict]:
    """
    Combine roster + signals + events into structured player match rows.
    
    Each row represents one player's complete data for a match:
    - Basic info from roster
    - Signals from leaders/boxscore
    - Events from keyEvents
    
    Args:
        summary: Raw ESPN summary dict
        
    Returns:
        List of player match row dicts
    """
    if not summary:
        return []
    
    # Get all components
    roster_players = extract_roster_players(summary)
    signals = extract_player_signals(summary)
    events = extract_player_events(summary)
    stats_by_player: Dict[str, dict] = {}
    try:
        from .espn_stats_parsers import extract_player_stats_from_summary

        for stat_row in extract_player_stats_from_summary(summary):
            player_id = stat_row.get("player_id")
            if player_id:
                stats_by_player[str(player_id)] = stat_row
    except Exception as exc:
        logger.debug("Could not extract roster player stats: %s", exc)
    
    # Index signals and events by athlete_id
    signals_by_player: Dict[str, list] = {}
    for sig in signals:
        aid = sig["athlete_id"]
        if aid not in signals_by_player:
            signals_by_player[aid] = []
        signals_by_player[aid].append(sig)
    
    events_by_player: Dict[str, list] = {}
    for evt in events:
        aid = evt["athlete_id"]
        if aid not in events_by_player:
            events_by_player[aid] = []
        events_by_player[aid].append(evt)
    
    # Build combined rows
    rows = []
    seen_ids: Set[str] = set()
    
    # Start with roster as base
    for player in roster_players:
        aid = player["athlete_id"]
        if aid in seen_ids:
            continue
        seen_ids.add(aid)
        
        player_signals = signals_by_player.get(aid, [])
        player_events = events_by_player.get(aid, [])
        
        # Aggregate signals
        is_goal_leader = any(s.get("signal_type") == "goal_scorer" for s in player_signals)
        is_shot_leader = any(s.get("signal_type") in ("volume_shooter", "shot_accuracy") for s in player_signals)
        is_key_attacker = is_goal_leader or is_shot_leader
        has_offensive_signal = is_key_attacker or any(s.get("signal_type") == "playmaker" for s in player_signals)
        
        # Aggregate stats first; keyEvents are only a fallback to avoid inflation.
        stat_row = stats_by_player.get(aid, {})
        event_goals = sum(1 for e in player_events if e["event_type"] == "goal")
        own_goals = sum(1 for e in player_events if e["event_type"] == "own_goal")
        event_yellow_cards = sum(1 for e in player_events if e["event_type"] == "yellow_card")
        event_red_cards = sum(1 for e in player_events if e["event_type"] == "red_card")
        event_assists = sum(1 for e in player_events if e["event_type"] == "assist")
        goals = stat_row.get("goals") if stat_row.get("goals") is not None else event_goals
        yellow_cards = stat_row.get("yellow_cards") if stat_row.get("yellow_cards") is not None else event_yellow_cards
        red_cards = stat_row.get("red_cards") if stat_row.get("red_cards") is not None else event_red_cards
        assists = stat_row.get("assists") if stat_row.get("assists") is not None else event_assists
        was_substituted = any(e["event_type"] == "substitution" for e in player_events)
        
        rows.append({
            # Base roster info
            "athlete_id": aid,
            "player_name": player["player_name"],
            "team_id": player["team_id"],
            "team_name": player["team_name"],
            "home_away": player["home_away"],
            "position": player["position"],
            "jersey": player["jersey"],
            "is_starter": player["is_starter"],
            "formation_slot": player["formation_slot"],
            "is_captain": player["is_captain"],
            
            # Signal flags
            "is_goal_leader": is_goal_leader,
            "is_shot_leader": is_shot_leader,
            "is_key_attacker": is_key_attacker,
            "has_offensive_signal": has_offensive_signal,
            
            # Event counts
            "goals": goals,
            "own_goals": own_goals,
            "yellow_cards": yellow_cards,
            "red_cards": red_cards,
            "assists": assists,
            "was_substituted": was_substituted,
            
            # Raw data references
            "_signals_count": len(player_signals),
            "_events_count": len(player_events),
        })
    
    # Add players from signals/events that weren't in roster
    all_player_ids = set(signals_by_player.keys()) | set(events_by_player.keys())
    
    for aid in all_player_ids:
        if aid in seen_ids:
            continue
        seen_ids.add(aid)
        
        player_signals = signals_by_player.get(aid, [])
        player_events = events_by_player.get(aid, [])
        
        # Get basic info from first signal or event
        base_info = {}
        if player_signals:
            base_info = player_signals[0]
        elif player_events:
            base_info = player_events[0]
        
        # Aggregate signals
        is_goal_leader = any(s.get("signal_type") == "goal_scorer" for s in player_signals)
        is_shot_leader = any(s.get("signal_type") in ("volume_shooter", "shot_accuracy") for s in player_signals)
        is_key_attacker = is_goal_leader or is_shot_leader
        has_offensive_signal = is_key_attacker or any(s.get("signal_type") == "playmaker" for s in player_signals)
        
        # Aggregate events; there is no roster stat row for these players.
        goals = sum(1 for e in player_events if e["event_type"] == "goal")
        own_goals = sum(1 for e in player_events if e["event_type"] == "own_goal")
        yellow_cards = sum(1 for e in player_events if e["event_type"] == "yellow_card")
        red_cards = sum(1 for e in player_events if e["event_type"] == "red_card")
        assists = sum(1 for e in player_events if e["event_type"] == "assist")
        was_substituted = any(e["event_type"] == "substitution" for e in player_events)
        
        rows.append({
            "athlete_id": aid,
            "player_name": base_info.get("player_name", ""),
            "team_id": base_info.get("team_id", ""),
            "team_name": base_info.get("team_name", ""),
            "home_away": base_info.get("home_away", ""),
            "position": "",
            "jersey": "",
            "is_starter": False,
            "formation_slot": None,
            "is_captain": False,
            
            "is_goal_leader": is_goal_leader,
            "is_shot_leader": is_shot_leader,
            "is_key_attacker": is_key_attacker,
            "has_offensive_signal": has_offensive_signal,
            
            "goals": goals,
            "own_goals": own_goals,
            "yellow_cards": yellow_cards,
            "red_cards": red_cards,
            "assists": assists,
            "was_substituted": was_substituted,
            
            "_signals_count": len(player_signals),
            "_events_count": len(player_events),
        })
    
    return rows


def check_player_data_availability(summary: dict) -> dict:
    """
    Check what level of player data is available in a summary.
    
    Returns availability at multiple levels:
    - player_roster_available: Has parseable rosters
    - player_signal_available: Has leaders or offensive signals
    - player_props_partial: Can support scorer props
    - player_props_full: Has complete player stats
    
    Args:
        summary: Raw ESPN summary dict
        
    Returns:
        Dict with availability flags
    """
    roster = extract_roster_players(summary)
    signals = extract_player_signals(summary)
    events = extract_player_events(summary)
    
    has_roster = len(roster) > 0
    has_signals = len(signals) > 0
    has_events = len(events) > 0
    
    # Check for offensive signals specifically
    offensive_signals = [
        s for s in signals
        if s.get("signal_type") in ("goal_scorer", "volume_shooter", "shot_accuracy", "playmaker")
    ]
    has_offensive_signals = len(offensive_signals) > 0
    
    # Check for goal events
    goal_events = [e for e in events if e["event_type"] in ("goal", "own_goal")]
    has_goal_events = len(goal_events) > 0
    
    # Partial props available if: roster + (offensive signals OR goal events)
    partial_props = has_roster and (has_offensive_signals or has_goal_events)
    
    # Full props would need detailed per-player stats
    # For now, consider full if we have roster with starters + signals
    starters = [p for p in roster if p.get("is_starter")]
    full_props = has_roster and len(starters) >= 7 and has_signals
    
    return {
        "player_roster_available": has_roster,
        "player_signal_available": has_signals,
        "player_event_available": has_events,
        "has_offensive_signals": has_offensive_signals,
        "has_goal_events": has_goal_events,
        "player_props_partial": partial_props,
        "player_props_full": full_props,
        "roster_count": len(roster),
        "signals_count": len(signals),
        "events_count": len(events),
    }
