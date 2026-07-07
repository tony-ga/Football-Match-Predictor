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
    - boxscore.players[] or boxscore.rosters[]
    - competitions[0].players[]
    
    Returns list of player stat dicts with normalized fields.
    
    Args:
        summary: Raw ESPN summary JSON
        
    Returns:
        List of player stat dictionaries
    """
    players = []
    
    if not summary:
        return players
    
    # Try boxscore.players or boxscore.rosters
    boxscore = summary.get("boxscore", {})
    
    # Different ESPN formats use different structures
    player_blocks = []
    
    # Format 1: boxscore.players is a list
    raw_players = boxscore.get("players", [])
    if isinstance(raw_players, list):
        player_blocks.extend(raw_players)
    
    # Format 2: boxscore has teams with rosters
    teams_data = boxscore.get("teams", [])
    for team_block in teams_data:
        roster = team_block.get("roster", [])
        if isinstance(roster, list):
            for p in roster:
                p["_team_home_away"] = team_block.get("homeAway", "")
            player_blocks.extend(roster)
        
        # Also check statistics.players
        team_stats = team_block.get("statistics", [])
        for stat_block in team_stats:
            if isinstance(stat_block, dict):
                sub_players = stat_block.get("players", [])
                if isinstance(sub_players, list):
                    for p in sub_players:
                        p["_team_home_away"] = team_block.get("homeAway", "")
                    player_blocks.extend(sub_players)
    
    # Format 3: competitions[0].players
    competitions = summary.get("competitions", [])
    if competitions:
        comp_players = competitions[0].get("players", [])
        if isinstance(comp_players, list):
            player_blocks.extend(comp_players)
    
    # Parse each player block
    seen_ids = set()
    for p in player_blocks:
        if not isinstance(p, dict):
            continue
        
        player_id = str(p.get("id") or p.get("athlete", {}).get("id") or "")
        if not player_id or player_id in seen_ids:
            continue
        seen_ids.add(player_id)
        
        athlete = p.get("athlete", {})
        if isinstance(athlete, dict):
            player_id = str(athlete.get("id") or player_id)
        
        # Extract basic info
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
        
        team_home_away = p.get("_team_home_away") or p.get("team", {}).get("homeAway", "")
        
        # Extract stats
        stats = p.get("stats", [])
        parsed_stats = _parse_player_stat_array(stats)
        
        # Check if starter (some APIs have this directly)
        is_starter = p.get("starter", False) or p.get("starterFlag", False)
        minutes_played = parsed_stats.get("minutes") or p.get("minutes") or 0
        
        # Infer starter if played significant minutes
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
            "saves": parsed_stats.get("saves"),  # for goalkeepers
        }
        
        players.append(player_data)
    
    return players


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
    
    Args:
        summary: Raw ESPN summary JSON
        
    Returns:
        List of event dicts sorted by time
    """
    events = []
    
    if not summary:
        return events
    
    # Look for plays or events
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
