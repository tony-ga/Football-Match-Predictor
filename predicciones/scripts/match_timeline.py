#!/usr/bin/env python
"""
Match Timeline CLI - Minute-by-minute match events using ESPN API.

This script fetches and displays a chronological timeline of match events
(goals, cards, corners, fouls, substitutions, offsides, etc.) using ESPN's
hidden API for soccer.

Examples:
    python scripts/match_timeline.py --event 760500
    python scripts/match_timeline.py --event 760500 --league fifa.world
    python scripts/match_timeline.py --event 760500 --source commentary
    python scripts/match_timeline.py --event 760500 --only goal,yellow_card,red_card
    python scripts/match_timeline.py --event 760500 --json
    python scripts/match_timeline.py --event 760500 --save output/argentina_caboverde_timeline.json
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

from src.data.espn_match_events import (
    get_match_event_timeline,
    filter_events_by_type,
    EventSource,
)

logger = logging.getLogger(__name__)


def print_match_summary(timeline: Dict[str, Any]) -> None:
    """Print match summary header."""
    match_info = timeline.get("match", {})
    sources = timeline.get("sources", {})
    
    separator = "=" * 50
    print(separator)
    print(f"{match_info.get('short_name', 'Unknown Match')}")
    print(f"Event ID: {timeline.get('event_id', 'N/A')}")
    print(f"League: {timeline.get('league', 'N/A')}")
    print(f"Status: {match_info.get('status', 'N/A')}")
    print(f"Date: {match_info.get('date', 'N/A')}")
    
    home_team = match_info.get('home_team', 'Home')
    away_team = match_info.get('away_team', 'Away')
    home_score = match_info.get('home_score', '?')
    away_score = match_info.get('away_score', '?')
    
    # Format scores (remove .0 if integer)
    if isinstance(home_score, float) and home_score == int(home_score):
        home_score = int(home_score)
    if isinstance(away_score, float) and away_score == int(away_score):
        away_score = int(away_score)
    
    print(f"Score: {home_team} {home_score} - {away_score} {away_team}")
    print(f"Source: {sources.get('used_source', 'N/A')}")
    print(f"Events loaded: {sources.get('total_events', 0)}")
    print(separator)


def print_events_console(events: List[Dict[str, Any]], limit: Optional[int] = None) -> None:
    """
    Print events in readable console format.
    
    Args:
        events: List of normalized events
        limit: Maximum number of events to display
    """
    if limit:
        events = events[:limit]
    
    for evt in events:
        minute = evt.get("minute", 0)
        clock_display = evt.get("clock_display")
        
        # Format minute with + for added time
        if clock_display:
            minute_str = str(clock_display)
        else:
            minute_str = f"{minute}'"
        
        event_type = evt.get("event_type", "unknown")
        description = evt.get("description", "")
        player_name = evt.get("player_name")
        team_abbr = evt.get("team_abbr")
        
        # Build formatted line
        type_label = f"[{event_type}]"
        
        # Add player/team info if available
        details = ""
        if player_name:
            details = f" ({player_name}"
            if team_abbr:
                details += f" - {team_abbr}"
            details += ")"
        
        line = f"{minute_str:>6} {type_label:<20} {description}{details}"
        print(line)


def save_timeline(timeline: Dict[str, Any], output_path: str) -> None:
    """
    Save timeline to JSON file.
    
    Args:
        timeline: Complete timeline dict
        output_path: Path to output file
    """
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(timeline, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Timeline saved to {output_path}")


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="View minute-by-minute match timeline using ESPN API.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --event 760500
      Show timeline for event 760500 (Argentina vs Cape Verde)

  %(prog)s --event 760500 --league fifa.world
      Specify league explicitly (default: fifa.world)

  %(prog)s --event 760500 --source commentary
      Use commentary as source (default behavior)

  %(prog)s --event 760500 --source keyEvents
      Use keyEvents instead of commentary

  %(prog)s --event 760500 --source auto
      Auto-select best available source

  %(prog)s --event 760500 --only goal,yellow_card,red_card
      Filter to show only goals and cards

  %(prog)s --event 760500 --json
      Output timeline as JSON to stdout

  %(prog)s --event 760500 --save output/timeline.json
      Save timeline JSON to file

  %(prog)s --event 760500 --limit 10
      Limit console output to first 10 events

  %(prog)s --event 760500 --include-summary
      Include match summary header in output
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
        "--source",
        type=str,
        default="commentary",
        choices=["commentary", "keyEvents", "auto", "core"],
        help="Data source preference (default: commentary)"
    )

    parser.add_argument(
        "--only",
        type=str,
        metavar="EVENT_TYPES",
        help="Comma-separated list of event types to filter (e.g., goal,yellow_card,red_card)"
    )

    parser.add_argument(
        "--json",
        action="store_true",
        dest="json_output",
        help="Output timeline as JSON to stdout"
    )

    parser.add_argument(
        "--save",
        type=str,
        metavar="OUTPUT_PATH",
        help="Save timeline JSON to specified file path"
    )

    parser.add_argument(
        "--limit",
        type=int,
        metavar="N",
        help="Limit number of events displayed in console output"
    )

    parser.add_argument(
        "--include-summary",
        action="store_true",
        help="Include match summary header in console output"
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
        # Fetch timeline from ESPN
        logger.info(f"Fetching timeline for event {args.event} from {args.league}")
        timeline = get_match_event_timeline(
            event_id=args.event,
            league=args.league,
            prefer_source=args.source,
        )

        # Apply event type filter if requested
        if args.only:
            event_types = [t.strip().lower() for t in args.only.split(",")]
            original_count = len(timeline["events"])
            timeline["events"] = filter_events_by_type(timeline["events"], event_types)
            filtered_count = len(timeline["events"])
            timeline["sources"]["filtered_from"] = original_count
            timeline["sources"]["filtered_to"] = filtered_count
            logger.info(f"Filtered events: {original_count} -> {filtered_count}")

        # Handle output
        if args.json_output:
            # Print JSON to stdout
            print(json.dumps(timeline, indent=2, ensure_ascii=False))
        else:
            # Console output
            if args.include_summary:
                print_match_summary(timeline)
            
            # Print events
            print_events_console(timeline["events"], limit=args.limit)
            
            # If no summary printed, still show basic info
            if not args.include_summary:
                sources = timeline.get("sources", {})
                print(f"\nSource: {sources.get('used_source', 'N/A')} | Events: {sources.get('total_events', 0)}")
                
                if args.only:
                    print(f"Filtered to: {args.only}")

        # Save to file if requested
        if args.save:
            save_timeline(timeline, args.save)

        return 0

    except ValueError as e:
        logger.error(f"Invalid input: {e}")
        return 1
    except Exception as e:
        logger.exception(f"Unexpected error: {e}")
        return 2


if __name__ == "__main__":
    sys.exit(main())
