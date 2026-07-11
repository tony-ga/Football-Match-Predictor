#!/usr/bin/env python
"""
Match Win Probability CLI - Extract win probabilities from ESPN soccer summaries.

This script fetches and displays match win probabilities using the ESPN summary endpoint.
It extracts pre-match predictor data (homeTeamWinPercentage, awayTeamWinPercentage, tiePercentage)
and optionally the historical win probability flow.

Examples:
    python scripts/match_win_probability.py --event 760500
    python scripts/match_win_probability.py --event 760500 --league fifa.world
    python scripts/match_win_probability.py --event 760500 --json
    python scripts/match_win_probability.py --event 760500 --save output/win_probability_760500.json
    python scripts/match_win_probability.py --event 760500 --include-flow
    python scripts/match_win_probability.py --event 760500 --include-flow --limit 20
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, Optional

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from predicciones.src.data.espn_match_predictor import (
    build_match_probability_report,
    fetch_match_summary,
    print_match_probability_report,
    save_report,
)
from predicciones.src.domain.exceptions import EspnApiError

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Extract match win probabilities from ESPN soccer summaries.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --event 760500
      Fetch and display win probabilities for event 760500
      
  %(prog)s --event 760500 --league fifa.world
      Specify league (default: fifa.world)
      
  %(prog)s --event 760500 --json
      Output as JSON instead of human-readable format
      
  %(prog)s --event 760500 --save output/win_probability_760500.json
      Save report to JSON file
      
  %(prog)s --event 760500 --include-flow
      Include win probability flow (historical/in-play probabilities)
      
  %(prog)s --event 760500 --include-flow --limit 20
      Include flow but limit to first 20 entries in console output
      
Notes:
  - The predictor node contains pre-match win probabilities
  - The winProbability node contains historical flow (if available)
  - Not all matches/competitions have predictor or winProbability data
  - Coverage varies by competition relevance and ESPN data availability
        """,
    )
    
    parser.add_argument(
        "--event",
        type=str,
        required=True,
        help="ESPN event ID for the match (required)",
    )
    
    parser.add_argument(
        "--league",
        type=str,
        default="fifa.world",
        help="League slug (default: fifa.world). Examples: fifa.world, eng.1, esp.1",
    )
    
    parser.add_argument(
        "--json",
        action="store_true",
        dest="json_output",
        help="Output as JSON instead of human-readable format",
    )
    
    parser.add_argument(
        "--save",
        type=str,
        metavar="OUTPUT_PATH",
        help="Save report to JSON file at specified path",
    )
    
    parser.add_argument(
        "--include-flow",
        action="store_true",
        dest="include_flow",
        help="Include win probability flow (historical/in-play probabilities)",
    )
    
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        metavar="N",
        help="Limit number of flow entries displayed in console (only with --include-flow)",
    )
    
    parser.add_argument(
        "--pretty",
        action="store_true",
        help="Pretty-print JSON output (only with --json)",
    )
    
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )
    
    args = parser.parse_args()
    
    # Configure logging
    if args.verbose:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
    else:
        logging.basicConfig(level=logging.WARNING)
    
    try:
        # Fetch summary from ESPN
        logger.info(f"Fetching summary for event {args.event} from league {args.league}")
        summary = fetch_match_summary(args.event, args.league)
        
        if not summary:
            print(f"Error: No data returned from ESPN for event {args.event}", file=sys.stderr)
            print("This may be due to:")
            print("  - Invalid event ID")
            print("  - Competition not supported by ESPN")
            print("  - Temporary API issue")
            sys.exit(1)
        
        # Build report
        report = build_match_probability_report(
            summary=summary,
            event_id=args.event,
            league=args.league,
            include_flow=args.include_flow,
        )
        
        # Handle output
        if args.json_output:
            # JSON output
            if args.pretty:
                print(json.dumps(report, indent=2, ensure_ascii=False))
            else:
                print(json.dumps(report, ensure_ascii=False))
        else:
            # Human-readable output
            print_match_probability_report(
                report=report,
                include_flow=args.include_flow,
                limit=args.limit,
            )
            
            # Show warning if no predictor available
            if not report["predictor"]["available"]:
                print()
                print("Note: ESPN does not provide predictor data for this match.")
                print("This is common for less prominent competitions or older matches.")
        
        # Save to file if requested
        if args.save:
            save_report(report, args.save)
            print(f"\nReport saved to: {args.save}")
        
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except EspnApiError as e:
        print(f"ESPN API Error: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        logger.exception("Unexpected error")
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
