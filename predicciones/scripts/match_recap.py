#!/usr/bin/env python
"""
Match Recap CLI - Statistical summary of past matches using ESPN API.

This script fetches and displays a statistical recap of a completed match
using the ESPN summary endpoint.

Examples:
    python scripts/match_recap.py --event 760509
    python scripts/match_recap.py --event 760509 --league fifa.world
    python scripts/match_recap.py --event 760509 --json
    python scripts/match_recap.py --event 760509 --save output/argentina_egipto_recap.json
    python scripts/match_recap.py --event 760509 --include-leaders --include-commentary-count
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.espn_client_v2 import EspnClient
from src.domain.exceptions import EspnApiError

logger = logging.getLogger(__name__)


def fetch_match_summary(event_id: str, league: str) -> Dict[str, Any]:
    """
    Fetch match summary from ESPN API.

    Args:
        event_id: The ESPN event ID for the match
        league: League slug (e.g., 'fifa.world')

    Returns:
        Raw JSON response from ESPN summary endpoint

    Raises:
        EspnApiError: If the API request fails
        ValueError: If event_id is invalid
    """
    if not event_id or not event_id.strip():
        raise ValueError("Event ID cannot be empty")

    client = EspnClient(league=league)
    summary = client.get_summary(event_id)

    if not summary:
        raise EspnApiError(f"Empty response from ESPN for event {event_id}")

    # Check for error indicators in response
    if isinstance(summary, dict) and summary.get("error"):
        raise EspnApiError(f"ESPN returned error for event {event_id}: {summary.get('message', 'Unknown error')}")

    return summary


def _extract_team_stats(boxscore: Dict[str, Any], team_type: str) -> Dict[str, Any]:
    """
    Extract statistics for a specific team from boxscore.

    Args:
        boxscore: The boxscore section from summary
        team_type: 'home' or 'away'

    Returns:
        Dictionary with team statistics
    """
    stats = {
        "possession": None,
        "shots": None,
        "shots_on_target": None,
        "xg": None,
        "corners": None,
        "fouls": None,
        "offsides": None,
        "yellow_cards": None,
        "red_cards": None,
        "saves": None,
    }

    teams = boxscore.get("teams", [])
    team_data = None
    for team in teams:
        if team.get("homeAway") == team_type:
            team_data = team
            break

    if not team_data:
        return stats

    statistics = team_data.get("statistics", [])
    if not statistics:
        return stats

    for stat in statistics:
        if not isinstance(stat, dict):
            continue

        name = stat.get("name", "").lower()
        display_value = stat.get("displayValue") or stat.get("value")

        # Map ESPN stat names to our schema
        if "possession" in name or "posesión" in name:
            stats["possession"] = _parse_numeric(display_value)
        elif "shot" in name and "on target" in name:
            stats["shots_on_target"] = _parse_numeric(display_value)
        elif "shot" in name and "total" in name:
            stats["shots"] = _parse_numeric(display_value)
        elif name == "shots" or ("shot" in name and "target" not in name and "total" not in name):
            if stats["shots"] is None:
                stats["shots"] = _parse_numeric(display_value)
        elif "expected goals" in name or name == "xg":
            stats["xg"] = _parse_numeric(display_value)
        elif "corner" in name:
            stats["corners"] = _parse_numeric(display_value)
        elif "foul" in name:
            stats["fouls"] = _parse_numeric(display_value)
        elif "offside" in name:
            stats["offsides"] = _parse_numeric(display_value)
        elif "yellow" in name and "card" in name:
            stats["yellow_cards"] = _parse_numeric(display_value)
        elif "red" in name and "card" in name:
            stats["red_cards"] = _parse_numeric(display_value)
        elif "save" in name:
            stats["saves"] = _parse_numeric(display_value)

    return stats


def _parse_numeric(value: Any) -> Optional[float]:
    """Parse a value to numeric, returning None if not possible."""
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            # Remove % sign if present
            cleaned = value.replace("%", "").strip()
            return float(cleaned)
        except ValueError:
            return None
    return None


def extract_match_recap(
    summary: Dict[str, Any],
    league: str,
    event_id: str,
    include_leaders: bool = False,
    include_commentary_count: bool = False,
) -> Dict[str, Any]:
    """
    Extract structured match recap from ESPN summary.

    Args:
        summary: Raw JSON summary from ESPN
        league: League slug
        event_id: Event ID
        include_leaders: Whether to include leaders data
        include_commentary_count: Whether to include commentary count

    Returns:
        Structured recap dictionary
    """
    recap = {
        "event_id": event_id,
        "league": league,
        "short_name": None,
        "status": None,
        "date": None,
        "venue": None,
        "attendance": None,
        "home_team": None,
        "away_team": None,
        "home_score": None,
        "away_score": None,
        "team_stats": {
            "home": {},
            "away": {},
        },
    }

    # Extract header info
    header = summary.get("header", {})
    recap["date"] = header.get("date") or summary.get("date")
    
    # Status can be in header.status or header.competitions[0].status
    status_type = header.get("status", {}).get("type", {})
    if status_type:
        recap["status"] = status_type.get("name") or status_type.get("state")
    
    # Find competition data - can be in header.competitions or top-level competitions
    competitions = header.get("competitions", [])
    if not competitions:
        competitions = summary.get("competitions", [])
    
    if competitions:
        comp = competitions[0]
        
        # Get status from competition if not already set
        if not recap["status"]:
            comp_status = comp.get("status", {})
            recap["status"] = comp_status.get("type", {}).get("name") or comp_status.get("type", {}).get("state")
        
        # Get date from competition if not already set
        if not recap["date"]:
            recap["date"] = comp.get("date")
        
        venue_info = comp.get("venue", {})
        recap["venue"] = venue_info.get("fullName") or venue_info.get("address", {}).get("city") or "N/A"

        attendance_list = comp.get("attendance")
        if attendance_list:
            # Attendance can be a list or a single value
            if isinstance(attendance_list, list) and len(attendance_list) > 0:
                recap["attendance"] = attendance_list[0]
            else:
                recap["attendance"] = attendance_list

        # Extract teams and scores
        competitors = comp.get("competitors", [])
        home_comp = None
        away_comp = None

        for c in competitors:
            if c.get("homeAway") == "home":
                home_comp = c
            elif c.get("homeAway") == "away":
                away_comp = c

        # Fallback if homeAway not set
        if not home_comp and len(competitors) >= 2:
            home_comp = competitors[0]
            away_comp = competitors[1]

        if home_comp:
            recap["home_team"] = home_comp.get("team", {}).get("displayName") or home_comp.get("team", {}).get("name")
            recap["home_score"] = _parse_numeric(home_comp.get("score"))
            recap["short_name"] = f"{recap['home_team']} vs {recap['away_team']}" if recap.get('away_team') else recap['home_team']

        if away_comp:
            recap["away_team"] = away_comp.get("team", {}).get("displayName") or away_comp.get("team", {}).get("name")
            recap["away_score"] = _parse_numeric(away_comp.get("score"))
            if recap["short_name"] is None:
                recap["short_name"] = f"{recap['home_team'] or 'Home'} vs {recap['away_team']}"

    # Update short_name if we have both teams
    if recap["home_team"] and recap["away_team"]:
        recap["short_name"] = f"{recap['home_team']} vs {recap['away_team']}"

    # Extract boxscore stats
    boxscore = summary.get("boxscore", {})
    recap["team_stats"]["home"] = _extract_team_stats(boxscore, "home")
    recap["team_stats"]["away"] = _extract_team_stats(boxscore, "away")

    # Optional: Include leaders
    if include_leaders:
        leaders_data = summary.get("leaders", [])
        recap["leaders"] = _extract_leaders(leaders_data)

    # Optional: Include commentary count
    if include_commentary_count:
        commentary = summary.get("commentary", [])
        recap["commentary_count"] = len(commentary) if isinstance(commentary, list) else 0

    return recap


def _extract_leaders(leaders_data: List[Any]) -> Dict[str, Any]:
    """
    Extract leaders information from summary.

    Args:
        leaders_data: Raw leaders array from ESPN

    Returns:
        Dictionary with categorized leaders
    """
    leaders = {}

    if not leaders_data or not isinstance(leaders_data, list):
        return leaders

    for leader_group in leaders_data:
        if not isinstance(leader_group, dict):
            continue

        # Check for nested leaders structure (ESPN format)
        nested_leaders = leader_group.get("leaders", [])
        if nested_leaders and isinstance(nested_leaders, list):
            # Process nested format
            for nested_leader in nested_leaders:
                if not isinstance(nested_leader, dict):
                    continue
                
                label = nested_leader.get("displayName") or nested_leader.get("name", "unknown")
                items = nested_leader.get("leaders", [])
                
                if not items or not isinstance(items, list):
                    continue

                top_performers = []
                for item in items[:3]:  # Top 3
                    if not isinstance(item, dict):
                        continue

                    athlete = item.get("athlete", {})
                    team = leader_group.get("team", {})

                    performer = {
                        "name": athlete.get("fullName") or athlete.get("displayName") or athlete.get("lastName"),
                        "team": team.get("abbreviation") or team.get("displayName"),
                        "value": item.get("displayValue") or item.get("value"),
                    }

                    if performer["name"]:
                        top_performers.append(performer)

                if top_performers:
                    leaders[label] = top_performers
        
        # Also check for flat format (label/items)
        elif "label" in leader_group:
            label = leader_group.get("label", "unknown")
            items = leader_group.get("items", [])

            if not items or not isinstance(items, list):
                continue

            top_performers = []
            for item in items[:3]:  # Top 3
                if not isinstance(item, dict):
                    continue

                athlete = item.get("athlete", {})
                team = item.get("team", {})

                performer = {
                    "name": athlete.get("displayName") or athlete.get("shortName"),
                    "team": team.get("abbreviation") or team.get("displayName"),
                    "value": item.get("value") or item.get("displayValue"),
                }

                if performer["name"]:
                    top_performers.append(performer)

            if top_performers:
                leaders[label] = top_performers

    return leaders


def print_match_recap(recap: Dict[str, Any]) -> None:
    """
    Print formatted match recap to console.

    Args:
        recap: Structured recap dictionary
    """
    separator = "=" * 40

    # Header
    print(separator)
    print(f"{recap.get('short_name', 'Unknown Match')}")
    print(f"Event ID: {recap.get('event_id', 'N/A')}")
    print(f"Status: {recap.get('status', 'N/A')}")
    print(f"Date: {recap.get('date', 'N/A')}")
    print(f"Venue: {recap.get('venue', 'N/A')}")

    attendance = recap.get('attendance')
    if attendance:
        print(f"Attendance: {attendance}")

    home_team = recap.get('home_team', 'Home')
    away_team = recap.get('away_team', 'Away')
    home_score = recap.get('home_score', '?')
    away_score = recap.get('away_score', '?')

    print(f"Score: {home_team} {home_score} - {away_score} {away_team}")
    print(separator)

    # Team stats
    team_stats = recap.get("team_stats", {})
    home_stats = team_stats.get("home", {})
    away_stats = team_stats.get("away", {})

    # Stats to display
    stat_labels = {
        "possession": "Possession",
        "shots": "Total shots",
        "shots_on_target": "Shots on target",
        "xg": "xG",
        "corners": "Corners",
        "fouls": "Fouls",
        "offsides": "Offsides",
        "yellow_cards": "Yellow cards",
        "red_cards": "Red cards",
        "saves": "Saves",
    }

    print(f"\n{home_team}")
    for key, label in stat_labels.items():
        value = home_stats.get(key)
        if value is not None:
            # Format possession with %
            if key == "possession":
                print(f"  {label}: {value}%")
            else:
                print(f"  {label}: {value}")

    print(f"\n{away_team}")
    for key, label in stat_labels.items():
        value = away_stats.get(key)
        if value is not None:
            if key == "possession":
                print(f"  {label}: {value}%")
            else:
                print(f"  {label}: {value}")

    # Leaders (optional)
    leaders = recap.get("leaders")
    if leaders:
        print("\nLeaders")
        for category, performers in leaders.items():
            if performers:
                top = performers[0]
                name = top.get("name", "Unknown")
                team_abbr = top.get("team", "")
                value = top.get("value", "")

                # Format value appropriately
                if isinstance(value, float) and value == int(value):
                    value = int(value)

                print(f"  {category}: {name} ({team_abbr}) {value}")

    # Commentary count (optional)
    commentary_count = recap.get("commentary_count")
    if commentary_count is not None:
        print(f"\nCommentary events: {commentary_count}")


def save_recap(recap: Dict[str, Any], output_path: str) -> None:
    """
    Save recap to JSON file.

    Args:
        recap: Structured recap dictionary
        output_path: Path to output file
    """
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(recap, f, indent=2, ensure_ascii=False)

    logger.info(f"Recap saved to {output_path}")


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="View statistical recap of a past match using ESPN API.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --event 760509
      Show recap for event 760509 (Argentina vs Egypt)

  %(prog)s --event 760509 --league fifa.world
      Specify league explicitly (default: fifa.world)

  %(prog)s --event 760509 --json
      Output recap as JSON to stdout

  %(prog)s --event 760509 --save output/recap.json
      Save recap JSON to file

  %(prog)s --event 760509 --json --save output/recap.json
      Both print JSON and save to file

  %(prog)s --event 760509 --include-leaders --include-commentary-count
      Include leaders and commentary count in output
        """
    )

    parser.add_argument(
        "--event",
        type=str,
        required=True,
        help="ESPN event ID for the match"
    )

    parser.add_argument(
        "--league",
        type=str,
        default="fifa.world",
        help="League slug (default: fifa.world)"
    )

    parser.add_argument(
        "--json",
        action="store_true",
        dest="json_output",
        help="Output recap as JSON to stdout"
    )

    parser.add_argument(
        "--save",
        type=str,
        metavar="OUTPUT_PATH",
        help="Save recap JSON to specified file path"
    )

    parser.add_argument(
        "--include-leaders",
        action="store_true",
        help="Include match leaders (goals, assists, etc.)"
    )

    parser.add_argument(
        "--include-commentary-count",
        action="store_true",
        help="Include count of commentary events"
    )

    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )

    args = parser.parse_args()

    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )

    try:
        # Fetch summary from ESPN
        logger.info(f"Fetching summary for event {args.event} from {args.league}")
        summary = fetch_match_summary(args.event, args.league)

        # Extract recap
        recap = extract_match_recap(
            summary=summary,
            league=args.league,
            event_id=args.event,
            include_leaders=args.include_leaders,
            include_commentary_count=args.include_commentary_count,
        )

        # Handle output
        if args.json_output:
            # Print JSON to stdout
            print(json.dumps(recap, indent=2, ensure_ascii=False))
        else:
            # Print formatted recap
            print_match_recap(recap)

        # Save to file if requested
        if args.save:
            save_recap(recap, args.save)

        return 0

    except ValueError as e:
        logger.error(f"Invalid input: {e}")
        return 1
    except EspnApiError as e:
        logger.error(f"ESPN API error: {e}")
        return 2
    except Exception as e:
        logger.exception(f"Unexpected error: {e}")
        return 3


if __name__ == "__main__":
    sys.exit(main())
