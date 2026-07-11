#!/usr/bin/env python3
"""
Match Player Defensive Stats CLI.

Extracts individual player defensive and goalkeeper metrics from ESPN soccer summary endpoint.

Usage:
    python scripts/match_player_defensive_stats.py --event 760500
    python scripts/match_player_defensive_stats.py --event 760500 --league fifa.world
    python scripts/match_player_defensive_stats.py --event 760500 --json
    python scripts/match_player_defensive_stats.py --event 760500 --save output/player_defensive_stats_760500.json
    python scripts/match_player_defensive_stats.py --event 760500 --team Argentina
    python scripts/match_player_defensive_stats.py --event 760500 --only-nonzero

Metrics extracted:
    Goalkeepers: saves, goalsConceded
    Outfield players: offsides, interceptions, clearances

Note: Some metrics may not be available for all leagues/competitions.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from predicciones.src.data.espn_client_v2 import EspnClient
from predicciones.src.utils.team_normalization import normalize_team_name, get_canonical_team_name

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)


# =============================================================================
# Team Lookup with Normalization and Fallback
# =============================================================================

def _find_team_in_data(team_name: str) -> Optional[str]:
    """
    Find team in available data files using normalized name matching.
    
    Searches through:
    - output/*.json files for match data
    - data/examples/*.json
    
    Args:
        team_name: Team name (may be in Spanish or English)
        
    Returns:
        Canonical team name as found in data, or None if not found
    """
    # First normalize the input team name
    normalized_input = normalize_team_name(team_name)
    
    # Also try direct match with ratings file which has Spanish names
    ratings_paths = [
        Path(__file__).parent.parent / "data" / "ratings_wc2026.json",
        Path(__file__).parent.parent.parent / "data" / "ratings_wc2026.json",
    ]
    
    for ratings_path in ratings_paths:
        if ratings_path.exists():
            try:
                with open(ratings_path, 'r', encoding='utf-8') as f:
                    ratings_data = json.load(f)
                teams = ratings_data.get('teams', {})
                # Check both normalized and original name
                if normalized_input in teams:
                    return normalized_input
                if team_name in teams:
                    return team_name
                # Case-insensitive search
                for team_key in teams.keys():
                    if team_key.lower() == normalized_input.lower():
                        return team_key
                    if team_key.lower() == team_name.lower():
                        return team_key
            except Exception:
                continue
    
    # Search in output JSON files
    output_dirs = [
        Path(__file__).parent.parent / "output",
        Path(__file__).parent.parent.parent / "output",
    ]
    
    for output_dir in output_dirs:
        if not output_dir.exists():
            continue
        for json_file in output_dir.glob("*.json"):
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # Check match info
                match_info = data.get('match', {})
                home_team = match_info.get('home_team', '')
                away_team = match_info.get('away_team', '')
                
                for candidate in [home_team, away_team]:
                    if not candidate:
                        continue
                    # Direct match
                    if candidate == normalized_input or candidate == team_name:
                        return candidate
                    # Case-insensitive match
                    if candidate.lower() == normalized_input.lower():
                        return candidate
                    if candidate.lower() == team_name.lower():
                        return candidate
                
                # Check teams array
                teams_list = data.get('teams', [])
                if isinstance(teams_list, list):
                    for team_entry in teams_list:
                        if isinstance(team_entry, dict):
                            t_name = team_entry.get('team_name', '')
                            if t_name == normalized_input or t_name == team_name:
                                return t_name
                            if t_name.lower() == normalized_input.lower():
                                return t_name
            except Exception:
                continue
    
    return None


def _get_team_matches(team_name: str, max_matches: int = 10) -> List[Dict[str, Any]]:
    """
    Get recent matches for a team from available data files.
    
    Args:
        team_name: Canonical team name
        max_matches: Maximum number of matches to return
        
    Returns:
        List of match data dicts
    """
    matches = []
    
    output_dirs = [
        Path(__file__).parent.parent / "output",
        Path(__file__).parent.parent.parent / "output",
    ]
    
    for output_dir in output_dirs:
        if not output_dir.exists():
            continue
            
        # Look for timeline and advanced stats files
        patterns = ["*timeline*.json", "*stats*.json", "player_defensive*.json"]
        
        for pattern in patterns:
            for json_file in output_dir.glob(pattern):
                if len(matches) >= max_matches:
                    break
                    
                try:
                    with open(json_file, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    
                    match_info = data.get('match', {})
                    home_team = match_info.get('home_team', '')
                    away_team = match_info.get('away_team', '')
                    
                    # Check if this match involves our team
                    if team_name not in [home_team, away_team]:
                        continue
                    
                    # Extract player data if available
                    if 'teams' in data and isinstance(data['teams'], list):
                        matches.append({
                            'file': str(json_file),
                            'data': data,
                            'home_team': home_team,
                            'away_team': away_team,
                        })
                except Exception as e:
                    logger.debug(f"Error reading {json_file}: {e}")
                    continue
    
    return matches[:max_matches]


def fetch_team_players(team: str, max_matches: int = 10) -> Optional[Dict[str, Any]]:
    """
    Fetch player data for a team from available sources.
    
    This function implements a multi-tier lookup strategy:
    1. Normalize team name using centralized team_normalization module
    2. Search in local data files (output/*.json, data/examples/*.json)
    3. Try ESPN API if connected and team found
    4. Provide fallback with suggestions if team not found
    
    Args:
        team: Team name (can be in Spanish or English, e.g., "Francia" or "France")
        max_matches: Maximum number of matches to consider
        
    Returns:
        Team data dict with players or None on error
    """
    # Step 1: Normalize the team name
    canonical_name = normalize_team_name(team)
    logger.info(f"Looking up team: '{team}' -> normalized: '{canonical_name}'")
    
    # Step 2: Find the actual team name in available data
    actual_team_name = _find_team_in_data(team)
    
    if actual_team_name:
        logger.info(f"Found team in data: '{actual_team_name}'")
    else:
        # Try with canonical name
        actual_team_name = _find_team_in_data(canonical_name)
        if actual_team_name:
            logger.info(f"Found team using canonical name: '{actual_team_name}'")
    
    if not actual_team_name:
        # Provide helpful suggestion
        suggestions = _get_team_suggestions(team)
        if suggestions:
            logger.warning(f"Team '{team}' not found. Did you mean: {', '.join(suggestions[:3])}?")
        else:
            logger.warning(f"Team '{team}' not found in available data.")
        return None
    
    # Step 3: Get matches for this team
    matches = _get_team_matches(actual_team_name, max_matches)
    
    if not matches:
        logger.warning(f"No match data found for team: {actual_team_name}")
        return None
    
    # Step 4: Aggregate player data from all matches
    all_goalkeepers = {}
    all_outfield = {}
    
    for match in matches:
        match_data = match['data']
        teams_data = match_data.get('teams', [])
        
        for team_entry in teams_data:
            if not isinstance(team_entry, dict):
                continue
            
            t_name = team_entry.get('team_name', '')
            if t_name != actual_team_name:
                # Try normalized comparison
                if normalize_team_name(t_name) != canonical_name:
                    continue
            
            # Process goalkeepers
            for gk in team_entry.get('goalkeepers', []):
                player_id = gk.get('player_id', gk.get('player_name', ''))
                if player_id:
                    if player_id not in all_goalkeepers:
                        all_goalkeepers[player_id] = {
                            'player_name': gk.get('player_name', 'Unknown'),
                            'player_id': player_id,
                            'position': 'GK',
                            'matches': 0,
                            'saves': 0,
                            'goals_conceded': 0,
                        }
                    all_goalkeepers[player_id]['matches'] += 1
                    saves = gk.get('saves', 0) or 0
                    gc = gk.get('goals_conceded', 0) or 0
                    all_goalkeepers[player_id]['saves'] += saves
                    all_goalkeepers[player_id]['goals_conceded'] += gc
            
            # Process outfield players
            for of in team_entry.get('outfield', []):
                player_id = of.get('player_id', of.get('player_name', ''))
                if player_id:
                    if player_id not in all_outfield:
                        all_outfield[player_id] = {
                            'player_name': of.get('player_name', 'Unknown'),
                            'player_id': player_id,
                            'position': of.get('position', 'N/A'),
                            'matches': 0,
                            'offsides': 0,
                            'interceptions': 0,
                            'clearances': 0,
                            'goals': 0,
                            'assists': 0,
                            'tackles': 0,
                        }
                    all_outfield[player_id]['matches'] += 1
                    
                    # Accumulate stats (handle None values)
                    of_stats = of.get('stats_raw', {})
                    all_outfield[player_id]['offsides'] += (of.get('offsides') or 0)
                    all_outfield[player_id]['interceptions'] += (of.get('interceptions') or 0)
                    all_outfield[player_id]['clearances'] += (of.get('clearances') or 0)
                    all_outfield[player_id]['goals'] += (of_stats.get('totalGoals') or 0)
                    all_outfield[player_id]['assists'] += (of_stats.get('goalAssists') or 0)
    
    # Build result
    result = {
        'team_name': actual_team_name,
        'canonical_name': canonical_name,
        'matches_found': len(matches),
        'goalkeepers': list(all_goalkeepers.values()),
        'outfield': list(all_outfield.values()),
    }
    
    logger.info(f"Successfully fetched data for {actual_team_name}: {len(all_goalkeepers)} GKs, {len(all_outfield)} outfield players")
    return result


def _get_team_suggestions(team_name: str) -> List[str]:
    """
    Get team name suggestions based on partial match.
    
    Args:
        team_name: Team name that wasn't found
        
    Returns:
        List of suggested team names
    """
    suggestions = []
    normalized = normalize_team_name(team_name).lower()
    
    # Load from ratings
    ratings_paths = [
        Path(__file__).parent.parent / "data" / "ratings_wc2026.json",
        Path(__file__).parent.parent.parent / "data" / "ratings_wc2026.json",
    ]
    
    for ratings_path in ratings_paths:
        if ratings_path.exists():
            try:
                with open(ratings_path, 'r', encoding='utf-8') as f:
                    ratings_data = json.load(f)
                teams = list(ratings_data.get('teams', {}).keys())
                
                # Find similar names
                for team in teams:
                    if normalized in team.lower() or team.lower() in normalized:
                        suggestions.append(team)
                    elif len(normalized) > 3 and normalized[:3] == team.lower()[:3]:
                        suggestions.append(team)
                
                if suggestions:
                    break
            except Exception:
                continue
    
    return list(set(suggestions))[:5]


def compute_player_stats(team_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Compute player statistics from team data.
    
    Args:
        team_data: Team data from fetch_team_players
        
    Returns:
        List of player stats dicts
    """
    if not team_data:
        return []
    
    players = []
    
    # Process goalkeepers - already aggregated in fetch_team_players
    for gk in team_data.get('goalkeepers', []):
        players.append({
            'name': gk.get('player_name', 'Unknown'),
            'position': 'GK',
            'matches': gk.get('matches', 1),
            'saves': gk.get('saves', 0),
            'goals_conceded': gk.get('goals_conceded', 0),
        })
    
    # Process outfield players - already aggregated in fetch_team_players
    for of in team_data.get('outfield', []):
        players.append({
            'name': of.get('player_name', 'Unknown'),
            'position': of.get('position', 'N/A'),
            'matches': of.get('matches', 1),
            'offsides': of.get('offsides', 0),
            'interceptions': of.get('interceptions', 0),
            'clearances': of.get('clearances', 0),
            'goals': of.get('goals', 0),
            'assists': of.get('assists', 0),
            'tackles': of.get('tackles', 0),
        })
    
    return players


def fetch_match_summary(event_id: str, league: str) -> dict:
    """
    Fetch match summary from ESPN API.
    
    Args:
        event_id: ESPN event ID
        league: League identifier (e.g., 'fifa.world', 'eng.1')
        
    Returns:
        Raw summary JSON dict or empty dict on error
    """
    try:
        client = EspnClient(sport="soccer", league=league)
        summary = client.get_summary(event_id)
        return summary if summary else {}
    except Exception as e:
        logger.error(f"Error fetching summary for event {event_id}: {e}")
        return {}


def extract_match_context(summary: dict) -> dict:
    """
    Extract match context information from summary.
    
    Args:
        summary: Raw ESPN summary dict
        
    Returns:
        Dict with match context: short_name, date, status, home_team, away_team, scores
    """
    context = {
        "short_name": "",
        "date": "",
        "status": "",
        "home_team": "",
        "away_team": "",
        "home_score": None,
        "away_score": None,
    }
    
    if not summary:
        return context
    
    # Try header first
    header = summary.get("header", {})
    if header:
        context["date"] = header.get("date", "")
        context["status"] = header.get("status", {}).get("type", {}).get("name", "")
        
        # Get teams from header
        competitions = header.get("competitions", [])
        if competitions:
            comp = competitions[0]
            competitors = comp.get("competitors", [])
            for c in competitors:
                home_away = c.get("homeAway", "")
                team_info = c.get("team", {})
                team_name = team_info.get("displayName", "")
                score = c.get("score")
                
                if home_away == "home":
                    context["home_team"] = team_name
                    try:
                        context["home_score"] = int(score) if score is not None else None
                    except (ValueError, TypeError):
                        context["home_score"] = None
                elif home_away == "away":
                    context["away_team"] = team_name
                    try:
                        context["away_score"] = int(score) if score is not None else None
                    except (ValueError, TypeError):
                        context["away_score"] = None
        
        # Build short name
        if context["home_team"] and context["away_team"]:
            context["short_name"] = f"{context['home_team']} vs {context['away_team']}"
    
    return context


def normalize_numeric(value: Any) -> Optional[Union[int, float]]:
    """
    Normalize a value to numeric (int or float).
    
    Args:
        value: Value to normalize (string, int, float, etc.)
        
    Returns:
        int if whole number, float if decimal, None if unparseable
    """
    if value is None:
        return None
    
    if isinstance(value, (int, float)):
        if isinstance(value, float) and value.is_integer():
            return int(value)
        return value
    
    if isinstance(value, str):
        value = value.strip()
        if not value:
            return None
        
        # Remove common suffixes like 'yd', 'm', '%'
        cleaned = value.lower()
        for suffix in ['yd', 'yds', 'm', '%']:
            if cleaned.endswith(suffix):
                cleaned = cleaned[:-len(suffix)].strip()
        
        try:
            if '.' in cleaned:
                result = float(cleaned)
                if result.is_integer():
                    return int(result)
                return result
            else:
                return int(cleaned)
        except ValueError:
            return None
    
    return None


def extract_goalkeeper_metrics(
    player_entry: dict,
    team_meta: dict,
    include_raw: bool = False
) -> Optional[dict]:
    """
    Extract goalkeeper metrics from a player entry.
    
    Args:
        player_entry: Player data dict from roster
        team_meta: Team metadata (team_name, team_abbr)
        include_raw: Whether to include raw stats map
        
    Returns:
        Dict with goalkeeper metrics or None if not a goalkeeper
    """
    if not isinstance(player_entry, dict):
        return None
    
    # Get athlete info
    athlete = player_entry.get("athlete", {})
    if not isinstance(athlete, dict):
        return None
    
    athlete_id = str(athlete.get("id", ""))
    if not athlete_id:
        return None
    
    # Get position
    position_raw = player_entry.get("position", {})
    if isinstance(position_raw, dict):
        position = position_raw.get("abbreviation", "")
        position_name = position_raw.get("name", "").lower()
    else:
        position = str(position_raw) if position_raw else ""
        position_name = position.lower()
    
    # Check if goalkeeper
    is_gk = (
        position.upper() == "G" or
        position.upper() == "GK" or
        "goalkeeper" in position_name or
        "goalie" in position_name
    )
    
    if not is_gk:
        return None
    
    # Build stats map
    stats_list = player_entry.get("stats", [])
    stats_map = build_player_stats_map(stats_list)
    
    # Extract goalkeeper-specific metrics
    player_name = athlete.get("displayName", "") or athlete.get("name", "")
    
    result = {
        "player_name": player_name,
        "player_id": athlete_id,
        "team_name": team_meta.get("team_name", ""),
        "team_abbr": team_meta.get("team_abbr", ""),
        "position": position.upper() if position else "GK",
        "saves": normalize_numeric(stats_map.get("saves")),
        "goals_conceded": normalize_numeric(stats_map.get("goalsConceded")),
    }
    
    if include_raw:
        result["stats_raw"] = stats_map
    
    return result


def extract_outfield_metrics(
    player_entry: dict,
    team_meta: dict,
    include_raw: bool = False
) -> Optional[dict]:
    """
    Extract outfield player defensive metrics from a player entry.
    
    Args:
        player_entry: Player data dict from roster
        team_meta: Team metadata (team_name, team_abbr)
        include_raw: Whether to include raw stats map
        
    Returns:
        Dict with outfield metrics or None if goalkeeper
    """
    if not isinstance(player_entry, dict):
        return None
    
    # Get athlete info
    athlete = player_entry.get("athlete", {})
    if not isinstance(athlete, dict):
        return None
    
    athlete_id = str(athlete.get("id", ""))
    if not athlete_id:
        return None
    
    # Get position
    position_raw = player_entry.get("position", {})
    if isinstance(position_raw, dict):
        position = position_raw.get("abbreviation", "")
        position_name = position_raw.get("name", "").lower()
    else:
        position = str(position_raw) if position_raw else ""
        position_name = position.lower()
    
    # Skip goalkeepers
    is_gk = (
        position.upper() == "G" or
        position.upper() == "GK" or
        "goalkeeper" in position_name or
        "goalie" in position_name
    )
    
    if is_gk:
        return None
    
    # Build stats map
    stats_list = player_entry.get("stats", [])
    stats_map = build_player_stats_map(stats_list)
    
    # Extract outfield defensive metrics
    player_name = athlete.get("displayName", "") or athlete.get("name", "")
    
    result = {
        "player_name": player_name,
        "player_id": athlete_id,
        "team_name": team_meta.get("team_name", ""),
        "team_abbr": team_meta.get("team_abbr", ""),
        "position": position.upper() if position else "",
        "offsides": normalize_numeric(stats_map.get("offsides")),
        "interceptions": normalize_numeric(stats_map.get("interceptions")),
        "clearances": normalize_numeric(stats_map.get("clearances")),
    }
    
    if include_raw:
        result["stats_raw"] = stats_map
    
    return result


def build_player_stats_map(stats_list: list) -> dict:
    """
    Build a stats map from a list of stat objects.
    
    ESPN format: list of dicts with 'name', 'value', 'displayValue'
    
    Args:
        stats_list: List of stat dicts from player entry
        
    Returns:
        Dict mapping stat name -> value
    """
    stats_map = {}
    
    if not isinstance(stats_list, list):
        return stats_map
    
    for stat in stats_list:
        if not isinstance(stat, dict):
            continue
        
        name = stat.get("name", "")
        if not name:
            continue
        
        # Prefer value, fallback to displayValue
        value = stat.get("value")
        if value is None:
            value = stat.get("displayValue")
        
        stats_map[name] = value
    
    return stats_map


def has_nonzero_metrics(player_data: dict, is_goalkeeper: bool) -> bool:
    """
    Check if a player has any non-zero/non-null metrics.
    
    Args:
        player_data: Player metrics dict
        is_goalkeeper: Whether this is a goalkeeper
        
    Returns:
        True if any metric is non-zero/non-null
    """
    if is_goalkeeper:
        metrics_to_check = ["saves", "goals_conceded"]
    else:
        metrics_to_check = ["offsides", "interceptions", "clearances"]
    
    for metric in metrics_to_check:
        value = player_data.get(metric)
        if value is not None and value != 0:
            return True
    
    return False


def extract_boxscore_player_metrics(
    summary: dict,
    event_id: str,
    league: str,
    include_raw: bool = False,
    only_nonzero: bool = False
) -> dict:
    """
    Extract all player defensive metrics from match summary.
    
    Args:
        summary: Raw ESPN summary dict
        event_id: Event ID for reference
        league: League identifier
        include_raw: Include raw stats in output
        only_nonzero: Filter to only players with non-zero metrics
        
    Returns:
        Structured report dict
    """
    report = {
        "event_id": event_id,
        "league": league,
        "match": extract_match_context(summary),
        "teams": [],
    }
    
    if not summary:
        return report
    
    # Get rosters - primary source for player stats
    rosters = summary.get("rosters", [])
    if not isinstance(rosters, list):
        return report
    
    for roster_block in rosters:
        if not isinstance(roster_block, dict):
            continue
        
        # Team metadata
        team_info = roster_block.get("team", {})
        team_name = team_info.get("displayName", "")
        team_abbr = team_info.get("abbreviation", "")
        home_away = roster_block.get("homeAway", "")
        
        team_meta = {
            "team_name": team_name,
            "team_abbr": team_abbr,
            "home_away": home_away,
        }
        
        team_report = {
            "team_name": team_name,
            "team_abbr": team_abbr,
            "goalkeepers": [],
            "outfield": [],
        }
        
        # Get player roster
        player_roster = roster_block.get("roster", [])
        if not isinstance(player_roster, list):
            player_roster = []
        
        for player_entry in player_roster:
            # Try goalkeeper extraction first
            gk_data = extract_goalkeeper_metrics(player_entry, team_meta, include_raw)
            if gk_data:
                if not only_nonzero or has_nonzero_metrics(gk_data, is_goalkeeper=True):
                    team_report["goalkeepers"].append(gk_data)
                continue
            
            # Try outfield extraction
            of_data = extract_outfield_metrics(player_entry, team_meta, include_raw)
            if of_data:
                if not only_nonzero or has_nonzero_metrics(of_data, is_goalkeeper=False):
                    team_report["outfield"].append(of_data)
        
        # Only add team if it has players
        if team_report["goalkeepers"] or team_report["outfield"]:
            report["teams"].append(team_report)
    
    return report


def print_player_defensive_report(report: dict, team_filter: Optional[str] = None) -> None:
    """
    Print formatted player defensive report to console.
    
    Args:
        report: Structured report dict
        team_filter: Optional team name to filter by
    """
    match_info = report.get("match", {})
    
    # Header
    print("=" * 60)
    print("PLAYER DEFENSIVE / GOALKEEPER REPORT")
    print(match_info.get("short_name", "Unknown Match"))
    print(f"Event ID: {report.get('event_id', 'N/A')}")
    print(f"League: {report.get('league', 'N/A')}")
    print(f"Status: {match_info.get('status', 'N/A')}")
    
    home_score = match_info.get("home_score")
    away_score = match_info.get("away_score")
    score_str = f"{home_score if home_score is not None else '-'} - {away_score if away_score is not None else '-'}"
    print(f"Score: {match_info.get('home_team', 'Home')} {score_str} {match_info.get('away_team', 'Away')}")
    print("=" * 60)
    
    teams = report.get("teams", [])
    if not teams:
        print("\nNo player data available.")
        return
    
    for team in teams:
        team_name = team.get("team_name", "Unknown Team")
        
        # Apply team filter
        if team_filter and team_filter.lower() not in team_name.lower():
            continue
        
        print(f"\n{team_name}")
        
        # Goalkeepers
        goalkeepers = team.get("goalkeepers", [])
        if goalkeepers:
            print("  Goalkeepers")
            for gk in goalkeepers:
                pos = gk.get("position", "GK")
                name = gk.get("player_name", "Unknown")
                print(f"    [{pos}] {name}")
                
                saves = gk.get("saves")
                if saves is not None:
                    print(f"      Saves: {saves}")
                
                conceded = gk.get("goals_conceded")
                if conceded is not None:
                    print(f"      Goals conceded: {conceded}")
        
        # Outfield players
        outfield = team.get("outfield", [])
        if outfield:
            print("  Outfield")
            for of in outfield:
                pos = of.get("position", "")
                name = of.get("player_name", "Unknown")
                pos_str = f"[{pos}] " if pos else ""
                print(f"    {pos_str}{name}")
                
                offsides = of.get("offsides")
                if offsides is not None:
                    print(f"      Offsides: {offsides}")
                
                interceptions = of.get("interceptions")
                if interceptions is not None:
                    print(f"      Interceptions: {interceptions}")
                
                clearances = of.get("clearances")
                if clearances is not None:
                    print(f"      Clearances: {clearances}")
        
        if not goalkeepers and not outfield:
            print("  No player data available")
    
    print("\n" + "=" * 60)


def save_report(report: dict, output_path: str) -> None:
    """
    Save report to JSON file.
    
    Args:
        report: Report dict to save
        output_path: Path to output file
    """
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Report saved to {output_path}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Extract player defensive and goalkeeper metrics from ESPN soccer match summary.",
        epilog="""
Examples:
  %(prog)s --event 760500
  %(prog)s --event 760500 --league fifa.world
  %(prog)s --event 760500 --json
  %(prog)s --event 760500 --save output/player_defensive_stats_760500.json
  %(prog)s --event 760500 --team Argentina
  %(prog)s --event 760500 --only-nonzero --pretty

Metrics extracted:
  Goalkeepers: saves, goalsConceded
  Outfield players: offsides, interceptions, clearances

Note: Some metrics may not be available for all leagues/competitions.
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    parser.add_argument(
        "--event",
        required=True,
        help="ESPN event ID (required)",
    )
    parser.add_argument(
        "--league",
        default="fifa.world",
        help="League identifier (default: fifa.world)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output as JSON",
    )
    parser.add_argument(
        "--save",
        metavar="PATH",
        help="Save JSON report to file",
    )
    parser.add_argument(
        "--team",
        metavar="NAME",
        help="Filter by team name (partial match)",
    )
    parser.add_argument(
        "--only-nonzero",
        action="store_true",
        help="Show only players with non-zero metrics",
    )
    parser.add_argument(
        "--include-raw",
        action="store_true",
        help="Include raw stats map in output",
    )
    parser.add_argument(
        "--pretty",
        action="store_true",
        help="Display formatted report in console",
    )
    
    args = parser.parse_args()
    
    # Fetch summary
    summary = fetch_match_summary(args.event, args.league)
    
    if not summary:
        print(f"Error: Could not fetch summary for event {args.event}", file=sys.stderr)
        sys.exit(1)
    
    # Check if boxscore/rosters exist
    rosters = summary.get("rosters", [])
    if not rosters:
        print(f"Warning: No roster data available for event {args.event}", file=sys.stderr)
    
    # Extract metrics
    report = extract_boxscore_player_metrics(
        summary,
        args.event,
        args.league,
        include_raw=args.include_raw,
        only_nonzero=args.only_nonzero,
    )
    
    # Output handling
    output_json = json.dumps(report, indent=2, ensure_ascii=False)
    
    # Save if requested
    if args.save:
        save_report(report, args.save)
    
    # Print based on flags
    if args.json:
        print(output_json)
    elif args.pretty or not args.save:
        print_player_defensive_report(report, team_filter=args.team)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
