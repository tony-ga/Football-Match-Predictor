"""
ESPN data parsers extension for team and player statistics.

Extends base ESPN parsing to extract detailed stats for corners, cards,
shots, and player-level data from scoreboard and summary endpoints.

Supports ESPN API shape with:
- boxscore.teams[].statistics with names like: wonCorners, yellowCards, redCards, 
  totalShots, shotsOnTarget, foulsCommitted, etc.
- keyEvents for temporal event extraction (corners, cards, goals)
- rosters and leaders for player-level data
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# Mapeo de nombres de estadísticas ESPN a nombres internos
# ESPN usa nombres como "wonCorners", "yellowCards", "totalShots", etc.
ESPN_TO_INTERNAL_MAP = {
    # Corners
    "woncorners": "corners",
    "cornerkicks": "corners",
    "corners": "corners",
    
    # Cards
    "yellowcards": "yellow_cards",
    "redcards": "red_cards",
    "totalcards": "total_cards",
    
    # Shots - orden importa: primero los específicos, luego generales
    # Nota: "shotsOnTarget" -> "shotsontarget" (sin espacios)
    "shotsontarget": "shots_on_target",
    "shotson target": "shots_on_target",
    "shotson_target": "shots_on_target",
    "shotsongoal": "shots_on_target",
    "ongoal": "shots_on_target",
    "on goal": "shots_on_target",
    "totalshots": "shots",
    "shots": "shots",  # fallback
    "blockedshots": "blocked_shots",
    "penaltykickshots": "penalty_shots",
    "penaltykickgoals": "penalty_goals",
    "shotpct": "shot_percentage",  # no usar para shots count
    
    # Other
    "foulscommitted": "fouls",
    "fouls": "fouls",
    "offsides": "offsides",
    "possessionpct": "possession",
    "saves": "saves",
}


def extract_team_stats_from_summary(summary: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract team-level statistics from ESPN summary response.
    
    Looks for stats in multiple locations:
    - boxscore.teams[].statistics (primary)
    - competitions[0].statistics (fallback)
    
    Returns dict with normalized stat names or None values if not found.
    
    Args:
        summary: Raw ESPN summary JSON
        
    Returns:
        Dict with home_*/away_* prefixed stats
        
    Supported stats:
        - goals, corners, yellow_cards, red_cards, total_cards
        - shots, shots_on_target, possession, fouls, offsides
    """
    stats = {
        # Basic
        "home_goals": None,
        "away_goals": None,
        
        # Corners
        "home_corners": None,
        "away_corners": None,
        
        # Cards
        "home_yellow_cards": None,
        "away_yellow_cards": None,
        "home_red_cards": None,
        "away_red_cards": None,
        "home_total_cards": None,
        "away_total_cards": None,
        
        # Shots
        "home_shots": None,
        "away_shots": None,
        "home_shots_on_target": None,
        "away_shots_on_target": None,
        
        # Other
        "home_possession": None,
        "away_possession": None,
        "home_fouls": None,
        "away_fouls": None,
        "home_offsides": None,
        "away_offsides": None,
    }
    
    if not summary:
        return stats
    
    # Try boxscore first (primary source)
    boxscore = summary.get("boxscore", {})
    teams_data = boxscore.get("teams", [])
    
    home_stats_raw = None
    away_stats_raw = None
    
    for team_block in teams_data:
        home_away = team_block.get("homeAway", "")
        if home_away == "home":
            home_stats_raw = team_block.get("statistics", [])
            # Also get score from here
            stats["home_goals"] = _parse_int(team_block.get("score"))
        elif home_away == "away":
            away_stats_raw = team_block.get("statistics", [])
            stats["away_goals"] = _parse_int(team_block.get("score"))
    
    # Parse statistics arrays using the mapping
    for stat_list, is_home in [(home_stats_raw, True), (away_stats_raw, False)]:
        if not stat_list:
            continue
            
        prefix = "home" if is_home else "away"
        
        for stat in stat_list:
            if not isinstance(stat, dict):
                continue
                
            name = (stat.get("name") or "").lower()
            display_value = stat.get("displayValue") or stat.get("value")
            value = _parse_float(display_value)
            
            # Use mapping first
            mapped_name = ESPN_TO_INTERNAL_MAP.get(name)
            
            if mapped_name:
                stats[f"{prefix}_{mapped_name}"] = value
            else:
                # Fallback to pattern matching for unknown names
                _apply_pattern_matching(stats, prefix, name, value)
    
    # Calculate total cards if not explicitly provided
    for prefix in ["home", "away"]:
        if stats[f"{prefix}_total_cards"] is None:
            yellow = stats[f"{prefix}_yellow_cards"]
            red = stats[f"{prefix}_red_cards"]
            if yellow is not None or red is not None:
                stats[f"{prefix}_total_cards"] = (yellow or 0) + (red or 0)
    
    # Fallback to competitions[0].statistics if boxscore didn't have data
    competitions = summary.get("competitions", [])
    if competitions and len(competitions) > 0:
        comp_stats = _extract_stats_from_competition_block(competitions[0])
        for key, val in comp_stats.items():
            if val is not None and stats.get(key) is None:
                stats[key] = val
    
    return stats


def _extract_stats_from_competition_block(comp: Dict[str, Any]) -> Dict[str, Any]:
    """Extract stats from competition.statistics block."""
    stats = {}
    
    statistics = comp.get("statistics", [])
    if not statistics:
        return stats
    
    # Find home/away indices
    competitors = comp.get("competitors", [])
    home_idx = None
    away_idx = None
    for i, c in enumerate(competitors):
        if c.get("homeAway") == "home":
            home_idx = i
        elif c.get("homeAway") == "away":
            away_idx = i
    
    for stat_block in statistics:
        name = (stat_block.get("name") or "").lower()
        displays = stat_block.get("displayValue", [])
        
        if not isinstance(displays, list) or len(displays) < 2:
            continue
        
        home_val = _parse_float(displays[home_idx]) if home_idx is not None else None
        away_val = _parse_float(displays[away_idx]) if away_idx is not None else None
        
        if "corner" in name:
            stats["home_corners"] = home_val
            stats["away_corners"] = away_val
        elif "yellow" in name and "card" in name:
            stats["home_yellow_cards"] = home_val
            stats["away_yellow_cards"] = away_val
        elif "red" in name and "card" in name:
            stats["home_red_cards"] = home_val
            stats["away_red_cards"] = away_val
        elif "shot" in name and "on target" in name:
            stats["home_shots_on_target"] = home_val
            stats["away_shots_on_target"] = away_val
        elif "shot" in name:
            stats["home_shots"] = home_val
            stats["away_shots"] = away_val
        elif "possession" in name:
            stats["home_possession"] = home_val
            stats["away_possession"] = away_val
        elif "foul" in name:
            stats["home_fouls"] = home_val
            stats["away_fouls"] = away_val
    
    return stats


def extract_player_stats_from_summary(summary: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Extract player-level statistics from ESPN summary response.
    
    Looks for player stats in:
    - rosters[].roster[] (top-level)
    - leaders[].leaders[] with athlete info
    - keyEvents participants (goals, cards)
    - boxscore.players[] or boxscore.teams[].players (fallback)
    
    Returns list of player stat dicts with normalized fields.
    
    Args:
        summary: Raw ESPN summary JSON
        
    Returns:
        List of player stat dictionaries
    """
    players = []
    
    if not summary:
        return players
    
    seen_ids = set()
    
    # Format 1: Top-level rosters (most reliable for FIFA/soccer)
    # Structure: summary['rosters'][].roster[]
    rosters = summary.get("rosters", [])
    if isinstance(rosters, list):
        for roster_entry in rosters:
            if not isinstance(roster_entry, dict):
                continue
            
            home_away = roster_entry.get("homeAway", "")
            team_info = roster_entry.get("team", {})
            team_id = str(team_info.get("id", ""))
            team_name = team_info.get("displayName", "")
            
            roster = roster_entry.get("roster", [])
            if isinstance(roster, list):
                for p in roster:
                    player_data = _parse_roster_player_entry(
                        p, home_away, team_id, team_name
                    )
                    if player_data and player_data["player_id"] not in seen_ids:
                        seen_ids.add(player_data["player_id"])
                        players.append(player_data)
    
    # Format 2: Leaders with athlete stats
    # Structure: summary['leaders'][].leaders[].athlete
    leaders = summary.get("leaders", [])
    if isinstance(leaders, list):
        for leader_category in leaders:
            if not isinstance(leader_category, dict):
                continue
            
            team_info = leader_category.get("team", {})
            team_id = str(team_info.get("id", ""))
            team_name = team_info.get("displayName", "")
            home_away = ""
            
            leader_list = leader_category.get("leaders", [])
            if isinstance(leader_list, list):
                for leader_item in leader_list:
                    if not isinstance(leader_item, dict):
                        continue
                    
                    athlete = leader_item.get("athlete", {})
                    if not isinstance(athlete, dict):
                        continue
                    
                    athlete_id = str(athlete.get("id", ""))
                    if not athlete_id or athlete_id in seen_ids:
                        continue
                    
                    seen_ids.add(athlete_id)
                    
                    display_name = athlete.get("displayName", "")
                    short_name = athlete.get("shortName", "")
                    jersey = athlete.get("jersey", "")
                    position_raw = athlete.get("position", {})
                    if isinstance(position_raw, dict):
                        position = position_raw.get("abbreviation", "")
                    else:
                        position = str(position_raw) if position_raw else ""
                    
                    # Get stat value from leader item
                    stat_name = leader_item.get("name", "") or leader_item.get("displayName", "")
                    stat_value = leader_item.get("displayValue", "") or leader_item.get("value")
                    
                    # Parse numeric value
                    goals = None
                    assists = None
                    shots = None
                    
                    stat_lower = (stat_name or "").lower()
                    if isinstance(stat_value, (int, float)):
                        val = float(stat_value)
                    else:
                        val = _parse_float(stat_value) if stat_value else None
                    
                    if "goal" in stat_lower and "own" not in stat_lower:
                        goals = val
                    elif "assist" in stat_lower:
                        assists = val
                    elif "shot" in stat_lower:
                        shots = val
                    
                    player_data = {
                        "player_id": athlete_id,
                        "display_name": display_name,
                        "short_name": short_name,
                        "position": position,
                        "team_home_away": home_away,
                        "is_starter": False,
                        "minutes": None,
                        "goals": goals,
                        "assists": assists,
                        "shots": shots,
                        "shots_on_target": None,
                        "yellow_cards": None,
                        "red_cards": None,
                        "total_cards": None,
                        "fouls": None,
                        "offsides": None,
                        "saves": None,
                    }
                    players.append(player_data)
    
    # Format 3: Key events for goals/cards
    key_events = summary.get("keyEvents", [])
    if not key_events:
        key_events = summary.get("plays", [])
    
    if isinstance(key_events, list):
        for event in key_events:
            if not isinstance(event, dict):
                continue
            
            event_type = event.get("type", {}).get("text", "").lower()
            event_text = (event.get("text") or "").lower()
            
            # Determine event type
            parsed_event_type = None
            if "goal" in event_type or "goal" in event_text:
                if "own" in event_text:
                    parsed_event_type = "own_goal"
                else:
                    parsed_event_type = "goal"
            elif "yellow" in event_text and "card" in event_text:
                parsed_event_type = "yellow_card"
            elif "red" in event_text and "card" in event_text:
                parsed_event_type = "red_card"
            
            if not parsed_event_type:
                continue
            
            participants = event.get("participants", [])
            if not isinstance(participants, list):
                continue
            
            for participant in participants:
                if not isinstance(participant, dict):
                    continue
                
                athlete = participant.get("athlete", {})
                if not isinstance(athlete, dict):
                    continue
                
                athlete_id = str(athlete.get("id", ""))
                if not athlete_id:
                    continue
                
                # Check if we already have this player
                existing_player = None
                for p in players:
                    if p["player_id"] == athlete_id:
                        existing_player = p
                        break
                
                if existing_player:
                    # Update existing player with event
                    if parsed_event_type == "goal":
                        existing_player["goals"] = (existing_player["goals"] or 0) + 1
                    elif parsed_event_type == "own_goal":
                        # Don't count own goals as regular goals
                        pass
                    elif parsed_event_type == "yellow_card":
                        existing_player["yellow_cards"] = (existing_player["yellow_cards"] or 0) + 1
                        existing_player["total_cards"] = (existing_player["total_cards"] or 0) + 1
                    elif parsed_event_type == "red_card":
                        existing_player["red_cards"] = (existing_player["red_cards"] or 0) + 1
                        existing_player["total_cards"] = (existing_player["total_cards"] or 0) + 1
                else:
                    # Create new player entry from event
                    seen_ids.add(athlete_id)
                    display_name = athlete.get("displayName", "")
                    short_name = athlete.get("shortName", "")
                    team_ref = participant.get("team", {})
                    if isinstance(team_ref, dict):
                        team_id = str(team_ref.get("id", ""))
                        team_name = team_ref.get("displayName", "")
                    else:
                        team_id = ""
                        team_name = ""
                    
                    goals = 1 if parsed_event_type == "goal" else None
                    yellow_cards = 1 if parsed_event_type == "yellow_card" else None
                    red_cards = 1 if parsed_event_type == "red_card" else None
                    
                    player_data = {
                        "player_id": athlete_id,
                        "display_name": display_name,
                        "short_name": short_name,
                        "position": "",
                        "team_home_away": "",
                        "is_starter": False,
                        "minutes": None,
                        "goals": goals,
                        "assists": None,
                        "shots": None,
                        "shots_on_target": None,
                        "yellow_cards": yellow_cards,
                        "red_cards": red_cards,
                        "total_cards": (yellow_cards or 0) + (red_cards or 0),
                        "fouls": None,
                        "offsides": None,
                        "saves": None,
                    }
                    players.append(player_data)
    
    # Format 4: Fallback to boxscore structure
    boxscore = summary.get("boxscore", {})
    
    # Format 4a: boxscore.players is a list
    raw_players = boxscore.get("players", [])
    if isinstance(raw_players, list):
        for p in raw_players:
            if not isinstance(p, dict):
                continue
            player_id = str(p.get("id") or p.get("athlete", {}).get("id") or "")
            if not player_id or player_id in seen_ids:
                continue
            seen_ids.add(player_id)
            # Parse using standard method
            athlete = p.get("athlete", {})
            if isinstance(athlete, dict):
                player_id = str(athlete.get("id") or player_id)
            
            display_name = (
                p.get("displayName") or 
                athlete.get("displayName") or 
                p.get("name") or 
                athlete.get("name") or
                ""
            )
            
            short_name = p.get("shortName") or athlete.get("shortName", "")
            position = p.get("position") or athlete.get("position", {})
            if isinstance(position, dict):
                position = position.get("abbreviation") or position.get("name") or ""
            
            team_home_away = p.get("team", {}).get("homeAway", "")
            
            stats = p.get("stats", [])
            parsed_stats = _parse_player_stat_array(stats)
            
            is_starter = p.get("starter", False) or p.get("starterFlag", False)
            minutes_played = parsed_stats.get("minutes") or p.get("minutes") or 0
            
            if not is_starter and minutes_played > 60:
                is_starter = True
            
            player_data = {
                "player_id": player_id,
                "display_name": display_name,
                "short_name": short_name,
                "position": position,
                "team_home_away": team_home_away,
                "is_starter": is_starter,
                "minutes": minutes_played,
                "goals": parsed_stats.get("goals"),
                "assists": parsed_stats.get("assists"),
                "shots": parsed_stats.get("shots"),
                "shots_on_target": parsed_stats.get("shots_on_target"),
                "yellow_cards": parsed_stats.get("yellow_cards"),
                "red_cards": parsed_stats.get("red_cards"),
                "total_cards": parsed_stats.get("total_cards"),
                "fouls": parsed_stats.get("fouls"),
                "offsides": parsed_stats.get("offsides"),
                "saves": parsed_stats.get("saves"),
            }
            players.append(player_data)
    
    # Format 4b: boxscore.teams[].players or boxscore.teams[].roster
    teams_data = boxscore.get("teams", [])
    for team_block in teams_data:
        home_away = team_block.get("homeAway", "")
        team_info = team_block.get("team", {})
        team_id = str(team_info.get("id", ""))
        team_name = team_info.get("displayName", "")
        
        # Check roster
        roster = team_block.get("roster", [])
        if isinstance(roster, list):
            for p in roster:
                if not isinstance(p, dict):
                    continue
                player_id = str(p.get("id") or p.get("athlete", {}).get("id") or "")
                if not player_id or player_id in seen_ids:
                    continue
                seen_ids.add(player_id)
                
                athlete = p.get("athlete", {})
                if isinstance(athlete, dict):
                    player_id = str(athlete.get("id") or player_id)
                
                display_name = (
                    p.get("displayName") or 
                    athlete.get("displayName") or 
                    p.get("name") or 
                    athlete.get("name") or
                    ""
                )
                
                short_name = p.get("shortName") or athlete.get("shortName", "")
                position = p.get("position") or athlete.get("position", {})
                if isinstance(position, dict):
                    position = position.get("abbreviation") or position.get("name") or ""
                
                stats = p.get("stats", [])
                parsed_stats = _parse_player_stat_array(stats)
                
                is_starter = p.get("starter", False) or p.get("starterFlag", False)
                
                player_data = {
                    "player_id": player_id,
                    "display_name": display_name,
                    "short_name": short_name,
                    "position": position,
                    "team_home_away": home_away,
                    "is_starter": is_starter,
                    "minutes": parsed_stats.get("minutes"),
                    "goals": parsed_stats.get("goals"),
                    "assists": parsed_stats.get("assists"),
                    "shots": parsed_stats.get("shots"),
                    "shots_on_target": parsed_stats.get("shots_on_target"),
                    "yellow_cards": parsed_stats.get("yellow_cards"),
                    "red_cards": parsed_stats.get("red_cards"),
                    "total_cards": parsed_stats.get("total_cards"),
                    "fouls": parsed_stats.get("fouls"),
                    "offsides": parsed_stats.get("offsides"),
                    "saves": parsed_stats.get("saves"),
                }
                players.append(player_data)
        
        # Check players
        team_players = team_block.get("players", [])
        if isinstance(team_players, list):
            for p in team_players:
                if not isinstance(p, dict):
                    continue
                player_id = str(p.get("id") or p.get("athlete", {}).get("id") or "")
                if not player_id or player_id in seen_ids:
                    continue
                seen_ids.add(player_id)
                # Similar parsing as above...
    
    return players


def _parse_roster_player_entry(
    player_block: dict,
    home_away: str,
    team_id: str,
    team_name: str,
    substitutions: Optional[Dict[str, Dict[str, Any]]] = None,
    match_duration: int = 90
) -> Optional[dict]:
    """Parse individual roster player entry from top-level rosters.
    
    Args:
        player_block: Player data from roster
        home_away: 'home' or 'away'
        team_id: Team ID
        team_name: Team display name
        substitutions: Dict mapping athlete_id to substitution info {
            'subbed_out_minute': int or None,
            'subbed_in_minute': int or None
        }
        match_duration: Total match duration in minutes (default 90, +extra time if applicable)
    """
    if not isinstance(player_block, dict):
        return None
    
    # Get athlete info - can be nested
    athlete = player_block.get("athlete", {})
    if not isinstance(athlete, dict):
        athlete = {}
    
    # Athlete ID
    athlete_id = (
        str(player_block.get("id") or "")
        or str(athlete.get("id") or "")
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
    jersey = player_block.get("jersey") or athlete.get("jersey")
    if jersey:
        jersey = str(jersey)
    
    # Starter status
    is_starter = (
        player_block.get("starter", False)
        or player_block.get("starterFlag", False)
        or athlete.get("starter", False)
    )
    
    # Stats from roster entry
    stats = player_block.get("stats", [])
    parsed_stats = _parse_player_stat_array(stats) if isinstance(stats, list) else {}
    
    # Minutes played - DERIVE FROM SUBSTITUTION DATA
    minutes_played = _derive_minutes_played(
        is_starter=bool(is_starter),
        subbed_in=player_block.get("subbedIn", False),
        subbed_out=player_block.get("subbedOut", False),
        athlete_id=athlete_id,
        substitutions=substitutions,
        match_duration=match_duration
    )
    
    return {
        "player_id": athlete_id,
        "display_name": player_name,
        "short_name": "",
        "position": position,
        "team_home_away": home_away,
        "is_starter": bool(is_starter),
        "minutes": minutes_played,
        "goals": parsed_stats.get("goals"),
        "assists": parsed_stats.get("assists"),
        "shots": parsed_stats.get("shots"),
        "shots_on_target": parsed_stats.get("shots_on_target"),
        "yellow_cards": parsed_stats.get("yellow_cards"),
        "red_cards": parsed_stats.get("red_cards"),
        "total_cards": parsed_stats.get("total_cards"),
        "fouls": parsed_stats.get("fouls"),
        "offsides": parsed_stats.get("offsides"),
        "saves": parsed_stats.get("saves"),
    }


def _derive_minutes_played(
    is_starter: bool,
    subbed_in: bool,
    subbed_out: bool,
    athlete_id: str,
    substitutions: Optional[Dict[str, Dict[str, Any]]],
    match_duration: int
) -> Optional[int]:
    """Derive minutes played from starter status and substitution events.
    
    Args:
        is_starter: Whether player started the match
        subbed_in: Whether player was subbed in
        subbed_out: Whether player was subbed out
        athlete_id: Player's athlete ID
        substitutions: Dict with substitution timing info
        match_duration: Total match duration in minutes
        
    Returns:
        Minutes played (int) or None if cannot derive
    """
    if not is_starter and not subbed_in:
        # Player on bench, never entered
        return 0
    
    if substitutions and athlete_id in substitutions:
        sub_info = substitutions[athlete_id]
        subbed_out_minute = sub_info.get('subbed_out_minute')
        subbed_in_minute = sub_info.get('subbed_in_minute')
        
        if is_starter and subbed_out_minute is not None:
            # Starter who was substituted out
            return subbed_out_minute
        elif is_starter and not subbed_out:
            # Starter who played full match
            return match_duration
        elif not is_starter and subbed_in_minute is not None:
            # Sub who came in
            return max(0, match_duration - subbed_in_minute)
    
    # Fallback estimates without precise substitution timing
    if is_starter and not subbed_out:
        return match_duration
    elif is_starter and subbed_out:
        # Estimate: assume subbed out around 60-75'
        return 70
    elif subbed_in:
        # Estimate: assume subbed in around 60-70'
        return 25
    
    return 0


def _parse_player_stat_array(stats: List[Any]) -> Dict[str, Optional[float]]:
    """Parse player stats array to normalized dict."""
    result = {
        "goals": None,
        "assists": None,
        "shots": None,
        "shots_on_target": None,
        "minutes": None,
        "yellow_cards": None,
        "red_cards": None,
        "total_cards": None,
        "fouls": None,
        "offsides": None,
        "saves": None,
    }
    
    if not isinstance(stats, list):
        return result
    
    for stat in stats:
        if isinstance(stat, dict):
            name = (stat.get("name") or "").lower()
            value = _parse_float(stat.get("displayValue") or stat.get("value"))
        elif isinstance(stat, (int, float)):
            # Some APIs just give values in order
            continue
        else:
            continue
        
        if "goal" in name and "own" not in name:
            result["goals"] = value
        elif "assist" in name:
            result["assists"] = value
        elif "minute" in name or "min" == name:
            result["minutes"] = value
        elif "shot" in name and "on target" in name:
            result["shots_on_target"] = value
        elif "shot" in name:
            result["shots"] = value
        elif "yellow" in name and "card" in name:
            result["yellow_cards"] = value
        elif "red" in name and "card" in name:
            result["red_cards"] = value
        elif "card" in name and "total" in name:
            result["total_cards"] = value
        elif "foul" in name:
            result["fouls"] = value
        elif "offside" in name:
            result["offsides"] = value
        elif "save" in name:
            result["saves"] = value
    
    # Calculate total cards
    if result["total_cards"] is None:
        yellow = result["yellow_cards"] or 0
        red = result["red_cards"] or 0
        if yellow > 0 or red > 0:
            result["total_cards"] = yellow + red
    
    return result


def extract_events_from_summary(summary: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Extract match events (corners, cards, goals) with temporal ordering.
    
    This is needed for markets like:
    - first corner
    - last corner
    - first card
    - race to X corners
    
    For soccer, ESPN provides event data in the 'commentary' field,
    not in 'plays' or 'playByPlay'.
    
    Args:
        summary: Raw ESPN summary JSON
        
    Returns:
        List of event dicts sorted by time
    """
    events = []
    
    if not summary:
        return events
    
    # SOCCER: Use commentary as the primary source for event data
    # ESPN soccer API provides detailed play-by-play in 'commentary'
    commentary = summary.get("commentary", [])
    if commentary and isinstance(commentary, list):
        parsed_events, _ = _parse_commentary_events(commentary)
        if parsed_events:
            return parsed_events
    
    # FALLBACK: Try plays/playByPlay for other sports
    plays = summary.get("plays", [])
    if not plays:
        # Try competitions[0].plays
        competitions = summary.get("competitions", [])
        if competitions:
            plays = competitions[0].get("plays", [])
    
    if not isinstance(plays, list):
        return events
    
    for play in plays:
        if not isinstance(play, dict):
            continue
        
        play_type = (play.get("type", {}).get("id") or 
                     play.get("type", {}).get("text") or 
                     play.get("typeId") or
                     "").lower()
        
        # Get timing info
        period = play.get("period", {}).get("number") or play.get("period", 1)
        time_display = play.get("clock", {}).get("displayValue") or play.get("time", {})
        if isinstance(time_display, dict):
            time_display = time_display.get("displayValue", "")
        
        # Normalize event type
        event_type = None
        team_side = None  # "home" or "away"
        
        team_data = play.get("team", {})
        if isinstance(team_data, dict):
            team_side = team_data.get("id")  # Use team id for mapping
        
        # Classify event
        text = (play.get("text") or "").lower()
        
        if "corner" in text or play_type == "corner":
            event_type = "corner"
        elif "yellow card" in text or "tarjeta amarilla" in text:
            event_type = "yellow_card"
        elif "red card" in text or "tarjeta roja" in text:
            event_type = "red_card"
        elif "goal" in text or play_type == "goal":
            event_type = "goal"
        
        if event_type:
            events.append({
                "event_type": event_type,
                "period": period,
                "time_display": time_display,
                "team_id": team_side,
                "raw": play,
            })
    
    # Sort by period then time
    def sort_key(e):
        period = e.get("period", 1)
        time_str = e.get("time_display", "")
        # Try to parse time as minutes
        try:
            if "'" in time_str:
                minutes = int(time_str.split("'")[0])
            else:
                minutes = 0
        except (ValueError, TypeError):
            minutes = 0
        return (period, minutes)
    
    events.sort(key=sort_key)
    
    return events


def parse_commentary_events_with_stats(commentary: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    """
    Public wrapper for _parse_commentary_events.
    
    Parse ESPN soccer commentary into structured match events with statistics.
    
    Args:
        commentary: List of commentary items from ESPN API
        
    Returns:
        Tuple of (list of parsed event dicts, event_type_counts dict)
    """
    return _parse_commentary_events(commentary)


def _get_event_order_priority(event_type: str) -> int:
    """
    Return a priority value for special events to ensure correct temporal ordering.
    
    Lower values = earlier in the match flow.
    This is used as a secondary sort key when minute/period are equal or ambiguous.
    
    Priority order:
    0: lineups_announced (pre-match)
    1: kickoff (start of first half)
    2-98: regular play events (goals, corners, cards, etc.)
    99: halftime (end of first half)
    100: second_half_start
    101-198: second half play events
    199: added_time_announced (late in match)
    200: fulltime (end of match)
    """
    SPECIAL_ORDER = {
        # Pre-match
        "lineups_announced": 0,
        
        # First half start
        "kickoff": 1,
        
        # First half end
        "halftime": 99,
        
        # Second half start
        "second_half_start": 100,
        
        # Late match announcements
        "added_time_announced": 199,
        
        # Match end
        "fulltime": 200,
        
        # Extra time (if applicable)
        "extra_time_first_half_start": 201,
        "extra_time_first_half_end": 250,
        "extra_time_second_half_start": 251,
        "extra_time_second_half_end": 300,
    }
    
    return SPECIAL_ORDER.get(event_type, 50)  # Default to middle for regular events


def _compute_sort_key(event: Dict[str, Any]) -> Tuple[int, int, int, int]:
    """
    Compute a robust sort key for temporal ordering of match events.
    
    This handles edge cases where:
    - fulltime/halftime might have minute=0 or incorrect timestamps
    - kickoff might appear after other events in raw data
    - special events need to be positioned correctly regardless of source_index
    
    Sort key tuple: (period, adjusted_minute, event_priority, source_index)
    
    Args:
        event: Event dict with period, minute, event_type, source_index
        
    Returns:
        Tuple for sorting
    """
    period = event.get("period", 1)
    minute = event.get("minute", 0)
    event_type = event.get("event_type", "unknown")
    source_index = event.get("source_index", 0)
    
    # Adjust minute for special events that should appear at specific positions
    adjusted_minute = minute
    event_priority = _get_event_order_priority(event_type)
    
    # Special handling for events that might have wrong minute values
    if event_type == "kickoff":
        # Kickoff should always be at minute 0 of its period
        adjusted_minute = 0
        # But it should come AFTER lineups_announced if both exist at minute 0
        event_priority = 2  # Just after lineups
    elif event_type == "halftime":
        # Halftime marks end of first half - should come after all period 1 events
        # If minute shows 0 or very low, use a high minute value for period 1
        if minute < 45:
            adjusted_minute = 45
        event_priority = 99
    elif event_type == "second_half_start":
        # Second half start should be at minute 0 of period 2
        adjusted_minute = 0
        event_priority = 1  # Early in period 2, but after any period 2 kickoff
    elif event_type == "fulltime":
        # Fulltime should always be last - use max minute for its period
        # Even if minute=0 in source, it goes to the end
        adjusted_minute = 999  # Effectively \"infinite\" minute
        event_priority = 200
    elif event_type == "lineups_announced":
        # Lineups always first, regardless of period/minute
        period = 0  # Force to \"pre-period\"
        adjusted_minute = 0
        event_priority = 0
    elif event_type == "added_time_announced":
        # Added time announcement comes late in the match
        # Keep original minute but boost priority
        event_priority = 199
    
    return (period, adjusted_minute, event_priority, source_index)


def _parse_commentary_events(commentary: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    """
    Parse ESPN soccer commentary into structured match events.
    
    Commentary items have this structure:
    {
        \"sequence\": int,
        \"time\": {\"value\": float (seconds), \"displayValue\": str (e.g., \"5'\")},
        \"text\": str,
        \"play\": {...}  # optional detailed play info
    }
    
    Args:
        commentary: List of commentary items from ESPN API
        
    Returns:
        Tuple of (list of parsed event dicts with standardized fields, event_type_counts dict)
    """
    events = []
    event_type_counts: Dict[str, int] = {}
    
    for idx, item in enumerate(commentary):
        if not isinstance(item, dict):
            continue
        
        # Extract basic info
        sequence = item.get("sequence", 0)
        time_info = item.get("time", {})
        time_value = time_info.get("value", 0) if isinstance(time_info, dict) else 0
        time_display = time_info.get("displayValue", "") if isinstance(time_info, dict) else ""
        text = item.get("text", "")
        
        # Get nested play data if available
        play = item.get("play", {})
        if not isinstance(play, dict):
            play = {}
        
        # Determine period from play or infer from time
        period = 1
        if play.get("period"):
            period = play.get("period", {}).get("number", 1)
        elif time_value > 2700:  # After 45 minutes (2700 seconds)
            period = 2
        elif time_value > 6300:  # After 75 minutes + extra time
            period = 3  # Extra time first half
        elif time_value > 8100:  # After 105 minutes
            period = 4  # Extra time second half
        
        # Calculate minute from time value (seconds)
        minute = int(time_value / 60) if time_value else 0
        
        # Detect event type from text and play type
        event_type = None
        team_name = None
        player_name = None
        description = text
        
        # Extract team name from play.team.displayName or from text
        team_data = play.get("team", {})
        if isinstance(team_data, dict):
            team_name = team_data.get("displayName")
        
        # Extract participant/player info
        participants = play.get("participants", [])
        if participants and isinstance(participants, list):
            first_participant = participants[0]
            if isinstance(first_participant, dict):
                athlete = first_participant.get("athlete", {})
                if isinstance(athlete, dict):
                    player_name = athlete.get("displayName")
        
        # Get play type for more accurate classification
        play_type_obj = play.get("type", {})
        play_type_text = ""
        play_type_id = ""
        if isinstance(play_type_obj, dict):
            play_type_text = play_type_obj.get("text", "").lower()
            play_type_id = play_type_obj.get("id", "")
        
        text_lower = text.lower()
        
        # Event type detection logic
        # Priority: explicit play type > text keywords
        
        # Goals - check for penalty goal, own goal, regular goal
        if play_type_text.startswith("goal") or "goal" in play_type_text:
            # Check for overturned goals
            if "overturned" in text_lower or "no goal" in text_lower or "cancelled" in play_type_text:
                event_type = "goal_overturned"  # VAR overturned
            # Check for own goal
            elif "own goal" in text_lower or play_type_text == "own goal":
                event_type = "own_goal"
            # Check for penalty goal
            elif "penalty" in text_lower or play_type_text == "penalty goal" or "penalty kick" in play_type_text:
                event_type = "penalty_goal"
            else:
                event_type = "goal"
        # Penalty awarded (not yet taken)
        elif play_type_text == "penalty" or ("penalty" in text_lower and "awarded" in text_lower):
            event_type = "penalty"
        # Yellow cards
        elif play_type_text == "yellow card" or "yellow card" in text_lower:
            event_type = "yellow_card"
        # Red cards  
        elif play_type_text == "red card" or "red card" in text_lower:
            event_type = "red_card"
        # Corners
        elif play_type_text == "corner awarded" or "corner" in text_lower:
            event_type = "corner"
        # Substitutions
        elif play_type_text == "substitution" or "substitution" in text_lower:
            event_type = "substitution"
        # Halftime
        elif play_type_text == "halftime" or "first half ends" in text_lower or "half-time" in text_lower:
            event_type = "halftime"
        # Fulltime
        elif play_type_text == "fulltime" or "second half ends" in text_lower or "match ends" in text_lower:
            event_type = "fulltime"
        # Second half start
        elif "second half begins" in text_lower:
            event_type = "second_half_start"
        # First half start / kickoff
        elif "first half begins" in text_lower or "kickoff" in text_lower:
            event_type = "kickoff"
        # Offside
        elif "offside" in text_lower:
            event_type = "offside"
        # Foul
        elif "foul" in text_lower:
            event_type = "foul"
        # Free kick
        elif "free kick" in text_lower:
            event_type = "free_kick"
        # Injury/delay
        elif "delay" in text_lower or "injury" in text_lower:
            event_type = "injury_delay"
        # Shot on target
        elif "shot on target" in text_lower or "save" in text_lower:
            event_type = "shot_on_target"
        # Shot off target
        elif "shot" in text_lower and ("off target" in text_lower or "blocked" in text_lower):
            event_type = "shot_off_target"
        # Attempt missed (shots that miss)
        elif "attempt missed" in text_lower:
            event_type = "shot_off_target"
        # Hit woodwork (post/bar)
        elif "hits the" in text_lower and ("post" in text_lower or "bar" in text_lower):
            event_type = "hit_woodwork"
        # Added time announcement
        elif "added time" in text_lower or "minutes of added time" in text_lower:
            event_type = "added_time_announced"
        # Lineups announced
        elif "lineups" in text_lower and "announced" in text_lower:
            event_type = "lineups_announced"
        # Extra time periods
        elif "extra time begins" in text_lower:
            if "first half" in text_lower:
                event_type = "extra_time_first_half_start"
            elif "second half" in text_lower:
                event_type = "extra_time_second_half_start"
        elif "extra time ends" in text_lower:
            if "first half" in text_lower:
                event_type = "extra_time_first_half_end"
            elif "second half" in text_lower:
                event_type = "extra_time_second_half_end"  # This is effectively fulltime for extra time matches
        # VAR decisions
        elif "var decision" in text_lower:
            event_type = "var_decision"
        # Handball
        elif "handball" in text_lower:
            event_type = "handball"
        # Attempt blocked
        elif "attempt blocked" in text_lower or "blocked" in text_lower:
            event_type = "shot_blocked"
        
        # If we detected an event type, add it to the list
        if event_type:
            event_record = {
                "event_type": event_type,
                "minute": minute,
                "period": period,
                "team_name": team_name,
                "player_name": player_name,
                "description": description,
                "raw_event": item,
                "source_index": idx,  # Original index in commentary array
            }
            events.append(event_record)
            event_type_counts[event_type] = event_type_counts.get(event_type, 0) + 1
        elif text_lower.strip():
            # Keep unmatched events with raw data for debugging
            # Mark as unknown but preserve all info
            event_record = {
                "event_type": "unknown",
                "minute": minute,
                "period": period,
                "team_name": team_name,
                "player_name": player_name,
                "description": description,
                "raw_event": item,
                "source_index": idx,  # Original index in commentary array
            }
            events.append(event_record)
            event_type_counts["unknown"] = event_type_counts.get("unknown", 0) + 1
    
    # Sort using robust temporal ordering that handles special events correctly
    events.sort(key=_compute_sort_key)
    
    return events, event_type_counts


def _apply_pattern_matching(stats: Dict[str, Any], prefix: str, name: str, value: Optional[float]) -> None:
    """
    Fallback pattern matching for stat names not in the explicit map.
    
    Uses substring matching to identify common stat categories.
    Updates stats dict in place.
    
    IMPORTANT: Skip percentage stats (shotpct, passpct, etc.) to avoid 
    overwriting count values with percentages.
    """
    if value is None:
        return
    
    # Skip percentage stats - they should not be used as counts
    if "pct" in name or "percentage" in name or "%" in name:
        return
    
    # Corners
    if "corner" in name:
        stats[f"{prefix}_corners"] = value
    # Cards
    elif "yellow" in name and "card" in name:
        stats[f"{prefix}_yellow_cards"] = value
    elif "red" in name and "card" in name:
        stats[f"{prefix}_red_cards"] = value
    elif "card" in name and "total" in name:
        stats[f"{prefix}_total_cards"] = value
    # Shots - only match total shots, not percentages
    elif "shot" in name and "on target" in name:
        stats[f"{prefix}_shots_on_target"] = value
    elif name == "shots" or (name.endswith("shots") and "blocked" not in name and "penalty" not in name):
        stats[f"{prefix}_shots"] = value
    # Other
    elif "possession" in name or "posesión" in name:
        stats[f"{prefix}_possession"] = value
    elif "foul" in name:
        stats[f"{prefix}_fouls"] = value
    elif "offside" in name:
        stats[f"{prefix}_offsides"] = value


def _parse_float(value: Any) -> Optional[float]:
    """Parse value to float, returning None if not possible."""
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value.replace("%", "").strip())
        except ValueError:
            return None
    return None


def _parse_int(value: Any) -> Optional[int]:
    """Parse value to int, returning None if not possible."""
    if value is None:
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        try:
            return int(float(value.replace("%", "").strip()))
        except ValueError:
            return None
    return None
