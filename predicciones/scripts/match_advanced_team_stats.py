#!/usr/bin/env python
"""
Advanced Team Statistics CLI - Extract advanced performance metrics from ESPN soccer match summary.

This script fetches and displays advanced team statistics from the ESPN summary endpoint,
including passing efficiency, attacks, shooting breakdown, and more.

Examples:
    python scripts/match_advanced_team_stats.py --event 760500
    python scripts/match_advanced_team_stats.py --event 760500 --league fifa.world
    python scripts/match_advanced_team_stats.py --event 760500 --json
    python scripts/match_advanced_team_stats.py --event 760500 --save output/advanced_team_stats_760500.json
    python scripts/match_advanced_team_stats.py --event 760500 --pretty
    python scripts/match_advanced_team_stats.py --event 760500 --json --include-raw
"""
from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from predicciones.src.data.espn_client_v2 import EspnClient
from predicciones.src.domain.exceptions import EspnApiError

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


def extract_match_context(summary: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract basic match context information from ESPN summary.

    Args:
        summary: Raw ESPN summary JSON

    Returns:
        Dictionary with match context (teams, scores, date, status, etc.)
    """
    context = {
        "short_name": None,
        "date": None,
        "status": None,
        "home_team": None,
        "away_team": None,
        "home_score": None,
        "away_score": None,
    }

    # Extract header info
    header = summary.get("header", {})
    context["date"] = header.get("date") or summary.get("date")

    # Status can be in header.status or header.competitions[0].status
    status_type = header.get("status", {}).get("type", {})
    if status_type:
        context["status"] = status_type.get("name") or status_type.get("state")

    # Find competition data
    competitions = header.get("competitions", [])
    if not competitions:
        competitions = summary.get("competitions", [])

    if competitions:
        comp = competitions[0]

        # Get status from competition if not already set
        if not context["status"]:
            comp_status = comp.get("status", {})
            context["status"] = comp_status.get("type", {}).get("name") or comp_status.get("type", {}).get("state")

        # Get date from competition if not already set
        if not context["date"]:
            context["date"] = comp.get("date")

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
            context["home_team"] = home_comp.get("team", {}).get("displayName") or home_comp.get("team", {}).get("name")
            context["home_score"] = _parse_int(home_comp.get("score"))

        if away_comp:
            context["away_team"] = away_comp.get("team", {}).get("displayName") or away_comp.get("team", {}).get("name")
            context["away_score"] = _parse_int(away_comp.get("score"))

    # Update short_name if we have both teams
    if context["home_team"] and context["away_team"]:
        context["short_name"] = f"{context['home_team']} vs {context['away_team']}"

    return context


def build_stats_map(team_stats: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """
    Build a map of statistics by name for easy lookup.

    Args:
        team_stats: List of stat dictionaries from ESPN

    Returns:
        Dictionary mapping stat name (lowercase) to full stat dict
    """
    stats_map = {}
    if not team_stats:
        return stats_map

    for stat in team_stats:
        if not isinstance(stat, dict):
            continue
        name = (stat.get("name") or "").lower()
        if name:
            stats_map[name] = stat

    return stats_map


def parse_fraction_stat(value: Any) -> Tuple[Optional[int], Optional[int]]:
    """
    Parse a fraction string like "412/505" into completed and attempted values.

    Args:
        value: String value like "412/505" or numeric value

    Returns:
        Tuple of (completed, attempted) or (None, None) if parsing fails
    """
    if value is None:
        return None, None

    if isinstance(value, (int, float)):
        return int(value), None

    if not isinstance(value, str):
        return None, None

    value = value.strip()
    if "/" in value:
        parts = value.split("/")
        if len(parts) == 2:
            try:
                completed = int(parts[0].strip())
                attempted = int(parts[1].strip())
                return completed, attempted
            except ValueError:
                pass

    # Try to parse as single number
    try:
        return int(float(value)), None
    except ValueError:
        return None, None


def parse_percentage(value: Any) -> Optional[float]:
    """
    Parse a percentage value from various formats.

    Args:
        value: Value that might be "81.5%", "81.5", 81.5, or decimal 0.815

    Returns:
        Float percentage (0-100 scale) or None if parsing fails
    """
    if value is None:
        return None

    if isinstance(value, (int, float)):
        # If it's a decimal < 1, convert to percentage
        if 0 <= value < 1:
            return round(value * 100, 1)
        return float(value)

    if not isinstance(value, str):
        return None

    value = value.strip()
    if not value:
        return None

    # Remove % sign if present
    cleaned = value.replace("%", "").strip()

    try:
        num = float(cleaned)
        # If it's a decimal < 1, convert to percentage
        if 0 <= num < 1:
            return round(num * 100, 1)
        return num
    except ValueError:
        return None


def parse_float(value: Any) -> Optional[float]:
    """
    Parse a float value from various formats.

    Args:
        value: Value that might be "19.4", "19.4 yd", "19.4 m", or numeric

    Returns:
        Float value or None if parsing fails
    """
    if value is None:
        return None

    if isinstance(value, (int, float)):
        return float(value)

    if not isinstance(value, str):
        return None

    value = value.strip()
    if not value:
        return None

    # Extract numeric part using regex
    match = re.match(r"^-?\d+\.?\d*", value)
    if match:
        try:
            return float(match.group())
        except ValueError:
            pass

    return None


def _parse_int(value: Any) -> Optional[int]:
    """Parse an integer value."""
    if value is None:
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        try:
            return int(value.strip())
        except ValueError:
            try:
                return int(float(value.strip()))
            except ValueError:
                return None
    return None


def extract_advanced_team_metrics(team_block: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract advanced metrics for a single team from its boxscore block.

    Args:
        team_block: Team block from boxscore.teams array

    Returns:
        Dictionary with advanced metrics organized by category
    """
    result = {
        "team_name": None,
        "team_abbr": None,
        "advanced_metrics": {
            "passing": {
                "passes_raw": None,
                "passes_completed": None,
                "passes_attempted": None,
                "pass_percentage_raw": None,
                "pass_percentage": None,
            },
            "attacks": {
                "attacks": None,
                "dangerous_attacks": None,
                "crosses_raw": None,
                "crosses_completed": None,
                "crosses_attempted": None,
                "cross_percentage_raw": None,
                "cross_percentage": None,
            },
            "shooting": {
                "shots_on_target": None,
                "shots_off_target": None,
                "blocked_shots": None,
                "average_shot_distance_raw": None,
                "average_shot_distance": None,
                "hit_woodwork": None,
            },
        },
        "stats_raw": {},
    }

    # Extract team info
    team_info = team_block.get("team", {})
    result["team_name"] = team_info.get("displayName") or team_info.get("name")
    result["team_abbr"] = team_info.get("abbreviation")

    # Build stats map for easy lookup
    statistics = team_block.get("statistics", [])
    stats_map = build_stats_map(statistics)

    # Store raw stats
    for name, stat in stats_map.items():
        result["stats_raw"][name] = stat.get("displayValue") or stat.get("value")

    # ==========================================
    # PASSING EFFICIENCY
    # ==========================================
    passing = result["advanced_metrics"]["passing"]

    # Look for passes in various forms
    # ESPN uses: accuratePasses, totalPasses, passPct
    accurate_passes = stats_map.get("accuratepasses")
    total_passes = stats_map.get("totalpasses")
    pass_pct = stats_map.get("passpct")

    if accurate_passes and total_passes:
        acc_val = accurate_passes.get("displayValue") or accurate_passes.get("value")
        tot_val = total_passes.get("displayValue") or total_passes.get("value")

        # Build raw string like "412/505"
        if acc_val is not None and tot_val is not None:
            passing["passes_raw"] = f"{acc_val}/{tot_val}"
            passing["passes_completed"] = _parse_int(acc_val)
            passing["passes_attempted"] = _parse_int(tot_val)

    # Also check for a combined "passes" field (some leagues use this format)
    if not passing["passes_raw"]:
        passes_stat = stats_map.get("passes")
        if passes_stat:
            passes_val = passes_stat.get("displayValue") or passes_stat.get("value")
            passing["passes_raw"] = passes_val
            completed, attempted = parse_fraction_stat(passes_val)
            if completed is not None:
                passing["passes_completed"] = completed
            if attempted is not None:
                passing["passes_attempted"] = attempted

    # Pass percentage
    if pass_pct:
        pct_val = pass_pct.get("displayValue") or pass_pct.get("value")
        passing["pass_percentage_raw"] = pct_val
        passing["pass_percentage"] = parse_percentage(pct_val)

    # ==========================================
    # ATTACKS AND DANGER
    # ==========================================
    attacks = result["advanced_metrics"]["attacks"]

    # Look for attacks / dangerousAttacks
    attacks_stat = stats_map.get("attacks") or stats_map.get("totalattacks")
    if attacks_stat:
        val = attacks_stat.get("displayValue") or attacks_stat.get("value")
        attacks["attacks"] = _parse_int(val)

    dangerous_attacks_stat = stats_map.get("dangerousattacks") or stats_map.get("dangerousattcks")
    if dangerous_attacks_stat:
        val = dangerous_attacks_stat.get("displayValue") or dangerous_attacks_stat.get("value")
        attacks["dangerous_attacks"] = _parse_int(val)

    # Crosses - ESPN uses: accurateCrosses, totalCrosses, crossPct
    accurate_crosses = stats_map.get("accuratecrosses")
    total_crosses = stats_map.get("totalcrosses")
    cross_pct = stats_map.get("crosspct")

    if accurate_crosses and total_crosses:
        acc_val = accurate_crosses.get("displayValue") or accurate_crosses.get("value")
        tot_val = total_crosses.get("displayValue") or total_crosses.get("value")

        if acc_val is not None and tot_val is not None:
            attacks["crosses_raw"] = f"{acc_val}/{tot_val}"
            attacks["crosses_completed"] = _parse_int(acc_val)
            attacks["crosses_attempted"] = _parse_int(tot_val)

    # Also check for combined "crosses" field
    if not attacks["crosses_raw"]:
        crosses_stat = stats_map.get("crosses")
        if crosses_stat:
            crosses_val = crosses_stat.get("displayValue") or crosses_stat.get("value")
            attacks["crosses_raw"] = crosses_val
            completed, attempted = parse_fraction_stat(crosses_val)
            if completed is not None:
                attacks["crosses_completed"] = completed
            if attempted is not None:
                attacks["crosses_attempted"] = attempted

    # Cross percentage
    if cross_pct:
        pct_val = cross_pct.get("displayValue") or cross_pct.get("value")
        attacks["cross_percentage_raw"] = pct_val
        attacks["cross_percentage"] = parse_percentage(pct_val)

    # ==========================================
    # SHOOTING BREAKDOWN
    # ==========================================
    shooting = result["advanced_metrics"]["shooting"]

    # Shots on target
    shots_on_target_stat = stats_map.get("shotson target") or stats_map.get("shotson_target") or stats_map.get("shotsonTarget")
    if not shots_on_target_stat:
        # Try variations
        for key in stats_map:
            if "shot" in key and "target" in key:
                shots_on_target_stat = stats_map[key]
                break

    if shots_on_target_stat:
        val = shots_on_target_stat.get("displayValue") or shots_on_target_stat.get("value")
        shooting["shots_on_target"] = _parse_int(val)

    # Shots off target - look for various naming conventions
    shots_off_target_stat = None
    for key in stats_map:
        if "shot" in key and ("off" in key or "wide" in key or "miss" in key):
            shots_off_target_stat = stats_map[key]
            break

    if shots_off_target_stat:
        val = shots_off_target_stat.get("displayValue") or shots_off_target_stat.get("value")
        shooting["shots_off_target"] = _parse_int(val)

    # Blocked shots
    blocked_shots_stat = stats_map.get("blockedshots") or stats_map.get("blocked_shots")
    if not blocked_shots_stat:
        for key in stats_map:
            if "blocked" in key and "shot" in key:
                blocked_shots_stat = stats_map[key]
                break

    if blocked_shots_stat:
        val = blocked_shots_stat.get("displayValue") or blocked_shots_stat.get("value")
        shooting["blocked_shots"] = _parse_int(val)

    # Average shot distance
    avg_distance_stat = None
    for key in stats_map:
        if "distance" in key and "shot" in key:
            avg_distance_stat = stats_map[key]
            break
        if "avg" in key and "distance" in key:
            avg_distance_stat = stats_map[key]
            break
        if "averageshotdistance" in key or "average_shot_distance" in key:
            avg_distance_stat = stats_map[key]
            break

    if avg_distance_stat:
        val = avg_distance_stat.get("displayValue") or avg_distance_stat.get("value")
        shooting["average_shot_distance_raw"] = val
        shooting["average_shot_distance"] = parse_float(val)

    # Hit woodwork
    woodwork_stat = None
    for key in stats_map:
        if "woodwork" in key or "post" in key or "bar" in key or "hitwoodwork" in key or "hit_woodwork" in key:
            woodwork_stat = stats_map[key]
            break

    if woodwork_stat:
        val = woodwork_stat.get("displayValue") or woodwork_stat.get("value")
        shooting["hit_woodwork"] = _parse_int(val)

    return result


def extract_advanced_match_team_stats(
    summary: Dict[str, Any],
    event_id: str,
    league: str,
    include_raw: bool = False
) -> Dict[str, Any]:
    """
    Extract advanced team statistics for both teams from ESPN summary.

    Args:
        summary: Raw ESPN summary JSON
        event_id: Event ID
        league: League slug
        include_raw: Whether to include raw stats map

    Returns:
        Structured report dictionary
    """
    # Extract match context
    context = extract_match_context(summary)

    # Initialize report
    report = {
        "event_id": event_id,
        "league": league,
        "match": {
            "short_name": context["short_name"],
            "date": context["date"],
            "status": context["status"],
            "home_team": context["home_team"],
            "away_team": context["away_team"],
            "home_score": context["home_score"],
            "away_score": context["away_score"],
        },
        "teams": [],
    }

    # Get boxscore teams
    boxscore = summary.get("boxscore", {})
    teams_data = boxscore.get("teams", [])

    if not teams_data:
        logger.warning("No team data found in boxscore")
        return report

    # Process each team
    for team_block in teams_data:
        if not isinstance(team_block, dict):
            continue

        team_metrics = extract_advanced_team_metrics(team_block)

        # Remove stats_raw if not requested
        if not include_raw:
            team_metrics.pop("stats_raw", None)

        report["teams"].append(team_metrics)

    return report


def print_advanced_team_stats(report: Dict[str, Any]) -> None:
    """
    Print formatted advanced team statistics report to console.

    Args:
        report: Structured report dictionary
    """
    separator = "=" * 60

    # Header
    print(separator)
    print("ADVANCED TEAM PERFORMANCE REPORT")
    match_info = report.get("match", {})
    print(match_info.get("short_name", "Unknown Match"))
    print(f"Event ID: {report.get('event_id', 'N/A')}")
    print(f"League: {report.get('league', 'N/A')}")
    print(f"Status: {match_info.get('status', 'N/A')}")

    home_team = match_info.get("home_team", "Home")
    away_team = match_info.get("away_team", "Away")
    home_score = match_info.get("home_score", "?")
    away_score = match_info.get("away_score", "?")
    print(f"Score: {home_team} {home_score} - {away_score} {away_team}")
    print(separator)

    # Team stats
    teams = report.get("teams", [])
    for team_data in teams:
        team_name = team_data.get("team_name", "Unknown Team")
        print(f"\n{team_name}")

        metrics = team_data.get("advanced_metrics", {})

        # Passing
        passing = metrics.get("passing", {})
        print("  Passing")
        passes_raw = passing.get("passes_raw")
        if passes_raw:
            print(f"    Completed / Attempted: {passes_raw}")
        else:
            completed = passing.get("passes_completed")
            attempted = passing.get("passes_attempted")
            if completed is not None and attempted is not None:
                print(f"    Completed / Attempted: {completed}/{attempted}")
            elif completed is not None:
                print(f"    Completed: {completed}")
            elif attempted is not None:
                print(f"    Attempted: {attempted}")

        pass_pct = passing.get("pass_percentage")
        if pass_pct is not None:
            print(f"    Accuracy: {pass_pct}%")

        # Attacks
        attacks = metrics.get("attacks", {})
        print("\n  Attacks")
        total_attacks = attacks.get("attacks")
        if total_attacks is not None:
            print(f"    Total attacks: {total_attacks}")

        dangerous_attacks = attacks.get("dangerous_attacks")
        if dangerous_attacks is not None:
            print(f"    Dangerous attacks: {dangerous_attacks}")

        crosses_raw = attacks.get("crosses_raw")
        if crosses_raw:
            print(f"    Crosses: {crosses_raw}")
        else:
            completed = attacks.get("crosses_completed")
            attempted = attacks.get("crosses_attempted")
            if completed is not None and attempted is not None:
                print(f"    Crosses: {completed}/{attempted}")

        cross_pct = attacks.get("cross_percentage")
        if cross_pct is not None:
            print(f"    Cross accuracy: {cross_pct}%")

        # Shooting
        shooting = metrics.get("shooting", {})
        print("\n  Shooting")
        shots_on_target = shooting.get("shots_on_target")
        if shots_on_target is not None:
            print(f"    Shots on target: {shots_on_target}")

        shots_off_target = shooting.get("shots_off_target")
        if shots_off_target is not None:
            print(f"    Shots off target: {shots_off_target}")

        blocked_shots = shooting.get("blocked_shots")
        if blocked_shots is not None:
            print(f"    Blocked shots: {blocked_shots}")

        avg_distance = shooting.get("average_shot_distance")
        if avg_distance is not None:
            print(f"    Avg shot distance: {avg_distance}")

        hit_woodwork = shooting.get("hit_woodwork")
        if hit_woodwork is not None:
            print(f"    Hit woodwork: {hit_woodwork}")

    print("\n" + separator)


def save_report(report: Dict[str, Any], output_path: str) -> None:
    """
    Save report to JSON file.

    Args:
        report: Structured report dictionary
        output_path: Path to output file
    """
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    logger.info(f"Report saved to {output_path}")


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Extract advanced team performance metrics from ESPN soccer match summary.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --event 760500
      Show advanced stats for event 760500 (Argentina vs Cape Verde)

  %(prog)s --event 760500 --league fifa.world
      Specify league explicitly (default: fifa.world)

  %(prog)s --event 760500 --json
      Output report as JSON to stdout

  %(prog)s --event 760500 --save output/advanced_team_stats_760500.json
      Save report JSON to file

  %(prog)s --event 760500 --pretty
      Show formatted report in console (default behavior)

  %(prog)s --event 760500 --json --include-raw
      Include complete raw stats map in JSON output

  %(prog)s --event 760500 --json --save output/report.json --include-raw
      Save JSON with raw stats to file
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
        help="Output report as JSON to stdout"
    )

    parser.add_argument(
        "--save",
        type=str,
        metavar="OUTPUT_PATH",
        help="Save report JSON to specified file path"
    )

    parser.add_argument(
        "--pretty",
        action="store_true",
        help="Show formatted report in console (default if --json not specified)"
    )

    parser.add_argument(
        "--include-raw",
        action="store_true",
        help="Include complete raw stats map in JSON output"
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

        # Extract advanced stats
        report = extract_advanced_match_team_stats(
            summary=summary,
            event_id=args.event,
            league=args.league,
            include_raw=args.include_raw,
        )

        # Handle output
        if args.json_output:
            # Print JSON to stdout
            print(json.dumps(report, indent=2, ensure_ascii=False))
        else:
            # Print formatted report (also if --pretty is specified)
            print_advanced_team_stats(report)

        # Save to file if requested
        if args.save:
            save_report(report, args.save)

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
