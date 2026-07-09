#!/usr/bin/env python3
"""
ESPN Player Statistics Module.

Provides functions to extract player statistics from ESPN data sources:
- JSONL derived files (player_match_stats.jsonl)
- ESPN API via summary endpoint

Two main modes:
A. Roster/accumulated stats by team - aggregates player stats across matches
B. Match-by-match player stats - returns per-player per-match statistics

Supports team name normalization for Spanish/English aliases.
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


if __name__ == "__main__":
    sys.exit(main())
