#!/usr/bin/env python3
"""
ESPN Player Statistics Module.

Provides functions to extract player statistics from ESPN data sources:
- JSONL derived files (player_match_stats.jsonl, match_events.jsonl)
- ESPN API via summary endpoint

Two main modes:
A. Roster/accumulated stats by team - aggregates player stats across matches
B. Match-by-match player stats - returns per-player per-match statistics

Supports team name normalization for Spanish/English aliases.

New features:
- Minutes played aggregation from player_match_stats.jsonl
- Card timeline extraction from match_events.jsonl
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple
from collections import defaultdict

logger = logging.getLogger(__name__)

# Project root paths - use absolute path based on this file's location
SCRIPT_DIR = Path(__file__).parent  # /workspace/predicciones/scripts
PROJECT_ROOT = SCRIPT_DIR.parent.parent  # /workspace
DERIVED_DATA_PATH = PROJECT_ROOT / "data" / "derived"
OUTPUT_PATH = PROJECT_ROOT / "predicciones" / "output"

# Import team normalization
import sys
sys.path.insert(0, str(PROJECT_ROOT / "predicciones"))

try:
    from predicciones.src.utils.team_normalization import normalize_team_name
except ImportError:
    from src.utils.team_normalization import normalize_team_name


def get_jsonl_player_stats_path() -> Path:
    """Get path to player_match_stats.jsonl file."""
    return DERIVED_DATA_PATH / "player_match_stats.jsonl"


def get_match_events_jsonl_path() -> Path:
    """Get path to match_events.jsonl file."""
    return DERIVED_DATA_PATH / "match_events.jsonl"


def load_player_match_stats_jsonl() -> List[Dict[str, Any]]:
    """
    Load all player match statistics from JSONL file.
    
    Returns:
        List of player stat dicts with fields:
        - event_id, date, player_id, player_name, team, position
        - is_starter, minutes, goals, assists, shots, shots_on_target
        - yellow_cards, red_cards, total_cards
    """
    jsonl_path = get_jsonl_player_stats_path()
    if not jsonl_path.exists():
        logger.warning(f"Player stats JSONL not found: {jsonl_path}")
        return []
    
    players = []
    try:
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        record = json.loads(line)
                        players.append(record)
                    except json.JSONDecodeError as e:
                        logger.debug(f"Error parsing JSONL line: {e}")
    except Exception as e:
        logger.error(f"Error reading JSONL file: {e}")
    
    return players


def load_match_events_jsonl() -> List[Dict[str, Any]]:
    """
    Load all match events from JSONL file.
    
    Returns:
        List of event dicts with fields:
        - event_id, date, competition, home_team, away_team
        - events: list of event objects with:
          - sequence_index, minute, clock_display, period
          - event_type, team_name, player_name, description
          - clock_value (seconds), period_number
    """
    jsonl_path = get_match_events_jsonl_path()
    if not jsonl_path.exists():
        logger.warning(f"Match events JSONL not found: {jsonl_path}")
        return []
    
    events = []
    try:
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        record = json.loads(line)
                        events.append(record)
                    except json.JSONDecodeError as e:
                        logger.debug(f"Error parsing JSONL line: {e}")
    except Exception as e:
        logger.error(f"Error reading JSONL file: {e}")
    
    return events


def normalize_team_for_lookup(team_name: str) -> str:
    """
    Normalize team name for lookup in JSONL data.
    
    The JSONL uses English names like "Argentina", "France", "Mexico".
    This function converts Spanish aliases to English canonical names.
    
    Args:
        team_name: Input team name (may be Spanish or English)
        
    Returns:
        Normalized team name for JSONL lookup
    """
    # First apply standard normalization
    normalized = normalize_team_name(team_name)
    
    # JSONL uses English names, so we need to map Spanish -> English
    spanish_to_english = {
        "Francia": "France",
        "Alemania": "Germany", 
        "España": "Spain",
        "Inglaterra": "England",
        "Marruecos": "Morocco",
        "Suiza": "Switzerland",
        "Países Bajos": "Netherlands",
        "Holanda": "Netherlands",
        "Corea del Sur": "South Korea",
        "Estados Unidos": "United States",
        "USA": "United States",
        "Japón": "Japan",
        "Bélgica": "Belgium",
        "Dinamarca": "Denmark",
        "Turquía": "Turkey",
        "Polonia": "Poland",
        "Ucrania": "Ukraine",
        "México": "Mexico",
        "Canadá": "Canada",
        "Panamá": "Panama",
        "Brasil": "Brazil",
        "Italia": "Italy",
        "Croacia": "Croatia",
        "Perú": "Peru",
        "Nueva Zelanda": "New Zealand",
        "República Democrática del Congo": "DR Congo",
        "Camerún": "Cameroon",
        "Egipto": "Egypt",
        "Túnez": "Tunisia",
        "Argelia": "Algeria",
        "Arabia Saudita": "Saudi Arabia",
        "Irán": "Iran",
    }
    
    # Check if we need to convert Spanish to English
    if normalized in spanish_to_english:
        return spanish_to_english[normalized]
    
    # Also check original input
    if team_name in spanish_to_english:
        return spanish_to_english[team_name]
    
    return normalized


def fetch_team_roster_stats(
    team_name: str,
    competition_slug: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    Fetch accumulated player statistics for a team from JSONL data.
    
    Aggregates stats across all matches for the given team.
    
    Args:
        team_name: Team name (Spanish or English accepted)
        competition_slug: Optional filter by competition (not yet implemented)
        
    Returns:
        List of player stat dicts with accumulated stats:
        - team_name, player_name, player_id, position
        - games_played, minutes_played, goals, assists
        - yellow_cards, red_cards, shots, shots_on_target
    """
    # Normalize team name for JSONL lookup
    normalized_team = normalize_team_for_lookup(team_name)
    logger.info(f"Looking up team: '{team_name}' -> normalized for JSONL: '{normalized_team}'")
    
    # Load all player stats
    all_stats = load_player_match_stats_jsonl()
    if not all_stats:
        logger.warning("No player stats available in JSONL")
        return []
    
    # Filter by team and aggregate
    team_players: Dict[str, Dict[str, Any]] = {}
    
    for record in all_stats:
        record_team = record.get("team", "")
        
        # Match team name (case-insensitive)
        if record_team.lower() != normalized_team.lower():
            continue
        
        player_id = record.get("player_id", record.get("player_name", ""))
        if not player_id:
            continue
        
        # Initialize or update player aggregation
        if player_id not in team_players:
            team_players[player_id] = {
                "team_name": normalized_team,
                "player_name": record.get("player_name", "Unknown"),
                "player_id": player_id,
                "position": record.get("position", ""),
                "games_played": 0,
                "minutes_played": 0,
                "goals": 0,
                "assists": 0,
                "yellow_cards": 0,
                "red_cards": 0,
                "total_cards": 0,
                "shots": 0,
                "shots_on_target": 0,
                "is_starter_count": 0,
            }
        
        player = team_players[player_id]
        player["games_played"] += 1
        
        # Aggregate numeric stats (handle None values)
        minutes = record.get("minutes")
        if minutes is not None:
            player["minutes_played"] += minutes if isinstance(minutes, (int, float)) else 0
        
        goals = record.get("goals")
        if goals is not None:
            player["goals"] += goals if isinstance(goals, (int, float)) else 0
        
        assists = record.get("assists")
        if assists is not None:
            player["assists"] += assists if isinstance(assists, (int, float)) else 0
        
        yellow = record.get("yellow_cards")
        if yellow is not None:
            player["yellow_cards"] += yellow if isinstance(yellow, (int, float)) else 0
        
        red = record.get("red_cards")
        if red is not None:
            player["red_cards"] += red if isinstance(red, (int, float)) else 0
        
        total_cards = record.get("total_cards")
        if total_cards is not None:
            player["total_cards"] += total_cards if isinstance(total_cards, (int, float)) else 0
        
        shots = record.get("shots")
        if shots is not None:
            player["shots"] += shots if isinstance(shots, (int, float)) else 0
        
        shots_on_target = record.get("shots_on_target")
        if shots_on_target is not None:
            player["shots_on_target"] += shots_on_target if isinstance(shots_on_target, (int, float)) else 0
        
        is_starter = record.get("is_starter", False)
        if is_starter:
            player["is_starter_count"] += 1
    
    # Convert to list and sort by games played (descending)
    result = list(team_players.values())
    result.sort(key=lambda x: (-x["games_played"], x["player_name"]))
    
    logger.info(f"Found {len(result)} players for team: {normalized_team}")
    return result


def fetch_team_player_match_stats(
    team_name: str,
    max_matches: int = 10
) -> List[Dict[str, Any]]:
    """
    Fetch player statistics per match for a team.
    
    Returns one row per player per match.
    
    Args:
        team_name: Team name (Spanish or English accepted)
        max_matches: Maximum number of recent matches to include
        
    Returns:
        List of player-match stat dicts with fields:
        - date, competition, match_id (event_id), team_name, opponent
        - home_or_away, player_name, position
        - minutes_played, goals, assists
        - yellow_cards, red_cards
        - shots, shots_on_target (if available)
    """
    # Normalize team name
    normalized_team = normalize_team_for_lookup(team_name)
    logger.info(f"Fetching match stats for team: '{team_name}' -> '{normalized_team}'")
    
    # Load all player stats
    all_stats = load_player_match_stats_jsonl()
    if not all_stats:
        logger.warning("No player stats available in JSONL")
        return []
    
    # First, get unique matches for this team
    team_matches: Dict[str, Dict[str, Any]] = {}
    
    for record in all_stats:
        record_team = record.get("team", "")
        if record_team.lower() != normalized_team.lower():
            continue
        
        event_id = record.get("event_id", "")
        if not event_id:
            continue
        
        if event_id not in team_matches:
            team_matches[event_id] = {
                "event_id": event_id,
                "date": record.get("date", ""),
                "teams_seen": set(),
            }
        team_matches[event_id]["teams_seen"].add(record_team)
    
    # Sort matches by date (most recent first) and limit
    sorted_matches = sorted(
        team_matches.items(),
        key=lambda x: x[1].get("date", ""),
        reverse=True
    )[:max_matches]
    
    selected_event_ids = {event_id for event_id, _ in sorted_matches}
    logger.info(f"Selected {len(selected_event_ids)} matches for team: {normalized_team}")
    
    # Now extract player stats for these matches
    # We need to find opponents - load match events to get full match info
    match_events_path = DERIVED_DATA_PATH / "match_events.jsonl"
    event_opponents: Dict[str, Tuple[str, str, str]] = {}  # event_id -> (home, away, competition)
    
    if match_events_path.exists():
        try:
            with open(match_events_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            event = json.loads(line)
                            eid = event.get("event_id", "")
                            if eid in selected_event_ids:
                                home = event.get("home_team", "")
                                away = event.get("away_team", "")
                                comp = event.get("competition", "")
                                event_opponents[eid] = (home, away, comp)
                        except json.JSONDecodeError:
                            continue
        except Exception as e:
            logger.debug(f"Error reading match events: {e}")
    
    # Build player-match rows with deduplication
    rows: List[Dict[str, Any]] = []
    seen_keys: Set[Tuple[str, str]] = set()  # (event_id, player_id)
    
    for record in all_stats:
        record_team = record.get("team", "")
        if record_team.lower() != normalized_team.lower():
            continue
        
        event_id = record.get("event_id", "")
        if event_id not in selected_event_ids:
            continue
        
        player_id = record.get("player_id", record.get("player_name", ""))
        if not player_id:
            continue
        
        # Deduplication key
        dedup_key = (event_id, str(player_id))
        if dedup_key in seen_keys:
            continue
        seen_keys.add(dedup_key)
        
        # Get opponent info
        opponent = ""
        home_or_away = ""
        competition = ""
        
        if event_id in event_opponents:
            home, away, comp = event_opponents[event_id]
            competition = comp
            if record_team == home:
                opponent = away
                home_or_away = "home"
            elif record_team == away:
                opponent = home
                home_or_away = "away"
        
        # Build row
        row = {
            "date": record.get("date", ""),
            "competition": competition,
            "match_id": event_id,
            "team_name": normalized_team,
            "opponent": opponent,
            "home_or_away": home_or_away,
            "player_name": record.get("player_name", "Unknown"),
            "player_id": player_id,
            "position": record.get("position", ""),
            "minutes_played": record.get("minutes"),
            "goals": record.get("goals"),
            "assists": record.get("assists"),
            "yellow_cards": record.get("yellow_cards"),
            "red_cards": record.get("red_cards"),
            "shots": record.get("shots"),
            "shots_on_target": record.get("shots_on_target"),
            "is_starter": record.get("is_starter", False),
        }
        rows.append(row)
    
    # Sort by date (most recent first), then by player name
    rows.sort(key=lambda x: (x.get("date", "") or ""), reverse=True)
    
    logger.info(f"Generated {len(rows)} player-match rows for team: {normalized_team}")
    return rows


def format_output_table(data: List[Dict[str, Any]], mode: str = "roster") -> str:
    """
    Format player stats as ASCII table.
    
    Args:
        data: List of player stat dicts
        mode: "roster" for accumulated stats, "match" for per-match stats
        
    Returns:
        Formatted table string
    """
    if not data:
        return "No data available."
    
    lines = []
    
    if mode == "roster":
        # Header
        lines.append("-" * 100)
        lines.append(f"{'Player':<30} {'Pos':<6} {'GP':<4} {'Min':<6} {'Gls':<4} {'Ast':<4} {'YC':<3} {'RC':<3} {'Shots':<6}")
        lines.append("-" * 100)
        
        for player in data:
            lines.append(
                f"{player.get('player_name', 'N/A'):<30} "
                f"{player.get('position', 'N/A'):<6} "
                f"{player.get('games_played', 0):<4} "
                f"{player.get('minutes_played', 0):<6} "
                f"{player.get('goals', 0):<4} "
                f"{player.get('assists', 0):<4} "
                f"{player.get('yellow_cards', 0):<3} "
                f"{player.get('red_cards', 0):<3} "
                f"{player.get('shots', 0):<6}"
            )
        
        lines.append("-" * 100)
        lines.append(f"Total players: {len(data)}")
    
    elif mode == "match":
        # Header
        lines.append("-" * 140)
        lines.append(f"{'Date':<12} {'Match ID':<10} {'Opponent':<20} {'H/A':<4} {'Player':<25} {'Pos':<6} {'Min':<5} {'Gls':<4} {'Ast':<4} {'YC':<3} {'RC':<3}")
        lines.append("-" * 140)
        
        for row in data:
            lines.append(
                f"{(row.get('date') or '')[:10]:<12} "
                f"{row.get('match_id', 'N/A'):<10} "
                f"{row.get('opponent', 'N/A'):<20} "
                f"{row.get('home_or_away', 'N/A'):<4} "
                f"{row.get('player_name', 'N/A'):<25} "
                f"{row.get('position', 'N/A'):<6} "
                f"{str(row.get('minutes_played', ''))[:5]:<5} "
                f"{str(row.get('goals', '')):<4} "
                f"{str(row.get('assists', '')):<4} "
                f"{str(row.get('yellow_cards', '')):<3} "
                f"{str(row.get('red_cards', '')):<3}"
            )
        
        lines.append("-" * 140)
        lines.append(f"Total rows: {len(data)}")
    
    return "\n".join(lines)


def format_output_csv(data: List[Dict[str, Any]], mode: str = "roster") -> str:
    """
    Format player stats as CSV.
    
    Args:
        data: List of player stat dicts
        mode: "roster" or "match"
        
    Returns:
        CSV formatted string
    """
    if not data:
        return ""
    
    import csv
    import io
    
    output = io.StringIO()
    
    if mode == "roster":
        fieldnames = [
            "team_name", "player_name", "player_id", "position",
            "games_played", "minutes_played", "goals", "assists",
            "yellow_cards", "red_cards", "total_cards", "shots", "shots_on_target"
        ]
    else:
        fieldnames = [
            "date", "competition", "match_id", "team_name", "opponent", "home_or_away",
            "player_name", "player_id", "position", "minutes_played",
            "goals", "assists", "yellow_cards", "red_cards",
            "shots", "shots_on_target", "is_starter"
        ]
    
    writer = csv.DictWriter(output, fieldnames=fieldnames, extrasaction='ignore')
    writer.writeheader()
    writer.writerows(data)
    
    return output.getvalue()


def format_output_json(data: List[Dict[str, Any]]) -> str:
    """Format player stats as JSON."""
    return json.dumps(data, indent=2, ensure_ascii=False)


def save_to_file(content: str, filepath: Path) -> None:
    """Save content to file."""
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(content)
    logger.info(f"Saved output to: {filepath}")


# =============================================================================
# Main entry point for CLI usage
# =============================================================================

def main():
    """CLI entry point for player stats extraction."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Extract player statistics from ESPN-derived JSONL data"
    )
    parser.add_argument(
        "--team", "-t",
        required=True,
        help="Team name (Spanish or English accepted)"
    )
    parser.add_argument(
        "--mode", "-m",
        choices=["roster", "match"],
        default="roster",
        help="Mode: 'roster' for accumulated stats, 'match' for per-match stats"
    )
    parser.add_argument(
        "--max-matches",
        type=int,
        default=10,
        help="Maximum matches to include (for match mode)"
    )
    parser.add_argument(
        "--format", "-f",
        choices=["table", "csv", "json"],
        default="table",
        help="Output format"
    )
    parser.add_argument(
        "--output", "-o",
        help="Output file path (optional)"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.basicConfig(level=logging.INFO)
    
    # Fetch data based on mode
    if args.mode == "roster":
        data = fetch_team_roster_stats(args.team)
    else:
        data = fetch_team_player_match_stats(args.team, max_matches=args.max_matches)
    
    if not data:
        print(f"No data found for team: {args.team}")
        return 1
    
    # Format output
    if args.format == "table":
        output = format_output_table(data, mode=args.mode)
    elif args.format == "csv":
        output = format_output_csv(data, mode=args.mode)
    else:
        output = format_output_json(data)
    
    # Output
    if args.output:
        save_to_file(output, Path(args.output))
        print(f"Output saved to: {args.output}")
    else:
        print(output)
    
    return 0


# =============================================================================
# Card Timeline Functions - New feature for discipline tracking
# =============================================================================

def extract_card_events_from_match(match_event: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Extract card events (yellow/red) from a match event record.
    
    Args:
        match_event: Match event dict with 'events' array
        
    Returns:
        List of card event dicts with normalized fields
    """
    cards = []
    events_list = match_event.get("events", [])
    
    for event in events_list:
        event_type = event.get("event_type", "").lower()
        description = event.get("description", "").lower()
        
        # Detect card events by type or description
        is_yellow = "yellow" in event_type or "yellow card" in description
        is_red = "red" in event_type or "red card" in description
        
        if not (is_yellow or is_red):
            continue
        
        # Extract player name - try multiple sources
        player_name = event.get("player_name")
        if not player_name:
            # Try to extract from description: "Player Name (Team) is shown..."
            desc = event.get("description", "")
            if "(" in desc and ")" in desc:
                start = desc.find("(") + 1
                end = desc.find(")")
                potential_name = desc[start:end].strip()
                # Only use if it looks like a name (not just team name)
                if potential_name and " " in potential_name:
                    player_name = potential_name
        
        # Also try raw_event.play.participants
        if not player_name:
            raw_event = event.get("raw_event", {})
            play_info = raw_event.get("play", {})
            participants = play_info.get("participants", [])
            if participants:
                athlete = participants[0].get("athlete", {})
                player_name = athlete.get("displayName", "")
        
        if not player_name:
            continue  # Skip if no player identified
        
        # Extract team name from raw_event.play.team
        team_name = ""
        raw_event = event.get("raw_event", {})
        play_info = raw_event.get("play", {})
        team_info = play_info.get("team", {})
        if team_info:
            team_name = team_info.get("displayName", "")
        
        # Fallback to event team_name or match teams
        if not team_name:
            team_name = event.get("team_name", "")
        if not team_name:
            # Infer from description
            desc = event.get("description", "")
            home = match_event.get("home_team", "")
            away = match_event.get("away_team", "")
            if home.lower() in desc.lower():
                team_name = home
            elif away.lower() in desc.lower():
                team_name = away
        
        # Extract clock/timing info
        clock_data = play_info.get("clock", {}) or raw_event.get("time", {})
        
        clock_value = clock_data.get("value", 0) or 0
        clock_display = clock_data.get("displayValue", "") or ""
        period_number = play_info.get("period", {}).get("number", 1) or event.get("period", 1)
        
        # Calculate minute from seconds
        minute_value = int(clock_value // 60) if clock_value else 0
        
        card_type = "red" if is_red else "yellow"
        
        cards.append({
            "event_id": match_event.get("event_id", ""),
            "date": match_event.get("date", ""),
            "competition": match_event.get("competition", ""),
            "home_team": match_event.get("home_team", ""),
            "away_team": match_event.get("away_team", ""),
            "player_id": "",  # Not available in match_events.jsonl
            "player_name": player_name,
            "team_name": team_name,
            "card_type": card_type,
            "minute_display": clock_display or f"{minute_value}'",
            "minute_value": minute_value,
            "clock_seconds": int(clock_value) if clock_value else 0,
            "period_number": period_number,
        })
    
    return cards


def fetch_player_card_timeline(
    team_name: str,
    max_matches: int = 10
) -> List[Dict[str, Any]]:
    """
    Fetch chronological timeline of cards (yellow/red) for a team's players.
    
    Uses match_events.jsonl to extract card events with timing information.
    
    Args:
        team_name: Team name (Spanish or English accepted)
        max_matches: Maximum number of recent matches to include
        
    Returns:
        List of card event dicts with fields:
        - event_id, date, competition
        - player_id, player_name, team_name
        - opponent (derived from home/away)
        - card_type (yellow/red)
        - minute_display (e.g., "45+2'")
        - minute_value (integer minute)
        - clock_seconds (total seconds)
        - period_number (1, 2, 3, 4)
    """
    # Normalize team name
    normalized_team = normalize_team_for_lookup(team_name)
    logger.info(f"Fetching card timeline for team: '{team_name}' -> '{normalized_team}'")
    
    # Load match events
    all_matches = load_match_events_jsonl()
    if not all_matches:
        logger.warning("No match events available in JSONL")
        return []
    
    # Filter matches involving this team and limit
    team_matches = []
    for match in all_matches:
        home = match.get("home_team", "")
        away = match.get("away_team", "")
        
        if home.lower() == normalized_team.lower() or away.lower() == normalized_team.lower():
            team_matches.append(match)
    
    # Sort by date (most recent first) and limit
    team_matches.sort(key=lambda x: x.get("date", ""), reverse=True)
    team_matches = team_matches[:max_matches]
    
    logger.info(f"Found {len(team_matches)} matches for team: {normalized_team}")
    
    # Extract card events from each match
    all_cards = []
    for match in team_matches:
        home = match.get("home_team", "")
        away = match.get("away_team", "")
        
        # Determine opponent
        if home.lower() == normalized_team.lower():
            opponent = away
        else:
            opponent = home
        
        cards = extract_card_events_from_match(match)
        
        # Filter cards for our team only
        for card in cards:
            card_team = card.get("team_name", "")
            if card_team.lower() == normalized_team.lower():
                card["opponent"] = opponent
                all_cards.append(card)
    
    # Sort by clock_seconds (chronological within matches), then by date
    all_cards.sort(key=lambda x: (x.get("date", ""), x.get("clock_seconds", 0)))
    
    logger.info(f"Extracted {len(all_cards)} card events for team: {normalized_team}")
    return all_cards


def format_card_timeline_table(data: List[Dict[str, Any]]) -> str:
    """
    Format card timeline as ASCII table.
    
    Args:
        data: List of card event dicts
        
    Returns:
        Formatted table string
    """
    if not data:
        return "No card events found."
    
    lines = []
    lines.append("-" * 130)
    lines.append(f"{'Date':<12} {'Match':<25} {'Player':<25} {'Card':<6} {'Minute':<10} {'Period':<8}")
    lines.append("-" * 130)
    
    for card in data:
        match_str = f"{card.get('team_name', '')} vs {card.get('opponent', '')}"
        card_symbol = "🟨" if card.get("card_type") == "yellow" else "🟥"
        
        lines.append(
            f"{(card.get('date') or '')[:10]:<12} "
            f"{match_str:<25} "
            f"{card.get('player_name', 'N/A'):<25} "
            f"{card_symbol} {card.get('card_type', 'N/A'):<4} "
            f"{card.get('minute_display', 'N/A'):<10} "
            f"P{card.get('period_number', 1):<7}"
        )
    
    lines.append("-" * 130)
    lines.append(f"Total cards: {len(data)} (Yellow: {sum(1 for c in data if c.get('card_type') == 'yellow')}, Red: {sum(1 for c in data if c.get('card_type') == 'red')})")
    
    return "\n".join(lines)


def format_card_timeline_csv(data: List[Dict[str, Any]]) -> str:
    """
    Format card timeline as CSV.
    
    Args:
        data: List of card event dicts
        
    Returns:
        CSV formatted string
    """
    if not data:
        return ""
    
    import csv
    import io
    
    output = io.StringIO()
    fieldnames = [
        "event_id", "date", "competition", "team_name", "opponent",
        "player_name", "player_id", "card_type",
        "minute_display", "minute_value", "clock_seconds", "period_number"
    ]
    
    writer = csv.DictWriter(output, fieldnames=fieldnames, extrasaction='ignore')
    writer.writeheader()
    writer.writerows(data)
    
    return output.getvalue()


# =============================================================================
# Extended Player Statistics with submodes
# =============================================================================

def fetch_extended_player_stats(
    team_name: str,
    mode: str = "summary",
    max_matches: int = 10
) -> List[Dict[str, Any]]:
    """
    Fetch extended player statistics with various submodes.
    
    Args:
        team_name: Team name (Spanish or English accepted)
        mode: One of:
            - "summary": Accumulated stats (GP, Min, Gls, Ast, YC, RC, Shots)
            - "cards": Historical cards per player
            - "timeline": Chronological card timeline by match
        max_matches: Maximum matches to consider
        
    Returns:
        List of stat dicts based on mode
    """
    if mode == "summary":
        return fetch_team_roster_stats(team_name)
    elif mode == "cards":
        # Get card history grouped by player
        cards = fetch_player_card_timeline(team_name, max_matches=max_matches)
        
        # Group by player
        player_cards: Dict[str, Dict[str, Any]] = {}
        for card in cards:
            player_name = card.get("player_name", "")
            if not player_name:
                continue
            
            if player_name not in player_cards:
                player_cards[player_name] = {
                    "player_name": player_name,
                    "team_name": card.get("team_name", ""),
                    "yellow_cards": 0,
                    "red_cards": 0,
                    "card_details": []
                }
            
            if card.get("card_type") == "yellow":
                player_cards[player_name]["yellow_cards"] += 1
            else:
                player_cards[player_name]["red_cards"] += 1
            
            player_cards[player_name]["card_details"].append({
                "date": card.get("date", ""),
                "opponent": card.get("opponent", ""),
                "card_type": card.get("card_type", ""),
                "minute": card.get("minute_display", "")
            })
        
        result = list(player_cards.values())
        result.sort(key=lambda x: (-(x["yellow_cards"] + x["red_cards"]), x["player_name"]))
        return result
    
    elif mode == "timeline":
        return fetch_player_card_timeline(team_name, max_matches=max_matches)
    
    else:
        logger.warning(f"Unknown mode: {mode}")
        return []


if __name__ == "__main__":
    sys.exit(main())
