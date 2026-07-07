#!/usr/bin/env python
"""
Match Prediction CLI with ESPN integration.

Provides multiple modes for match prediction:
- upcoming: List and select upcoming matches from ESPN
- match: Predict using ESPN event_id
- teams: Predict using team names
- json: Legacy mode using JSON file input

Usage:
    python scripts/predict_match.py --mode upcoming
    python scripts/predict_match.py --mode match --event-id 401xyz
    python scripts/predict_match.py --mode teams --home "Mexico" --away "England"
    python scripts/predict_match.py --mode json --input data/examples/template_partido.json
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.prompt import Prompt, IntPrompt
from rich.pretty import pprint

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ingestion.json_parser import load_match_json
from src.ingestion.schemas import MatchInput
from src.pipeline.predict import predict_match_pipeline
from src.data.espn_client_v2 import EspnClient
from src.data.espn_normalizers import TeamNormalizer
from src.application.match_selector import MatchSelector
from src.domain.models import UpcomingMatch
from src.domain.exceptions import (
    EspnApiError,
    MatchSelectionError,
    MatchInputBuildError,
    TeamNotFoundError,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)
console = Console()


def print_match_table(matches: List[UpcomingMatch], max_display: int = 20) -> None:
    """Print matches in a formatted table."""
    if not matches:
        console.print("[yellow]No matches found.[/yellow]")
        return
    
    table = Table(title=f"Upcoming Matches ({len(matches)} found)")
    table.add_column("#", style="dim", justify="right")
    table.add_column("Date", style="cyan")
    table.add_column("Competition", style="magenta")
    table.add_column("Stage", style="blue")
    table.add_column("Home", style="green")
    table.add_column("Away", style="red")
    table.add_column("Status", justify="center")
    table.add_column("Venue", style="dim")
    
    for i, match in enumerate(matches[:max_display]):
        date_str = match.date[:16].replace("T", " ") if match.date else "N/A"
        table.add_row(
            str(i),
            date_str,
            match.competition[:25] + "..." if len(match.competition) > 25 else match.competition,
            match.stage or "N/A",
            match.home_team[:20] + "..." if len(match.home_team) > 20 else match.home_team,
            match.away_team[:20] + "..." if len(match.away_team) > 20 else match.away_team,
            match.status.upper(),
            (match.venue or "N/A")[:25] + "..." if match.venue and len(match.venue) > 25 else (match.venue or "N/A"),
        )
    
    console.print(table)
    
    if len(matches) > max_display:
        console.print(f"[dim]... and {len(matches) - max_display} more matches[/dim]")


def format_predictions_output(response: Dict[str, Any], home_team: str, away_team: str) -> Dict[str, Any]:
    """Format predictions for display with percentages and team names."""
    def format_percentages(data, path=""):
        if isinstance(data, dict):
            return {k: format_percentages(v, path=f"{path}.{k}") for k, v in data.items()}
        elif isinstance(data, list):
            return [format_percentages(v, path) for v in data]
        elif isinstance(data, float):
            if "lambda" in path.lower() or "expected_goals" in path.lower() or "total" in path.lower() or "weight" in path.lower():
                return round(data, 3)
            return f"{data * 100:.2f}%"
        return data
    
    def replace_team_names(data, home_name, away_name):
        if isinstance(data, dict):
            new_dict = {}
            for k, v in data.items():
                new_k = str(k).replace('home', home_name).replace('away', away_name)
                new_dict[new_k] = replace_team_names(v, home_name, away_name)
            return new_dict
        elif isinstance(data, list):
            return [replace_team_names(i, home_name, away_name) for i in data]
        return data
    
    # Prune large distributions for cleaner display
    display_response = response.copy()
    display_predictions = display_response.get("predictions", {}).copy()
    if 'home_goals_distribution' in display_predictions:
        del display_predictions['home_goals_distribution']
    if 'away_goals_distribution' in display_predictions:
        del display_predictions['away_goals_distribution']
    display_response["predictions"] = display_predictions
    
    display_response = format_percentages(display_response)
    display_response = replace_team_names(display_response, home_team, away_team)
    
    return display_response


def run_prediction_from_input(
    match_input: MatchInput,
    source_mode: str,
    espn_context: Optional[Dict[str, Any]] = None
) -> None:
    """Run prediction pipeline and display results."""
    home_team = match_input.metadata.home_team
    away_team = match_input.metadata.away_team
    
    console.print(f"\n[bold yellow]Running Prediction Pipeline for {home_team} vs {away_team}...[/bold yellow]")
    
    try:
        response = predict_match_pipeline(
            home_team=home_team,
            away_team=away_team,
            match_date=match_input.metadata.date,
            neutral_venue=match_input.metadata.neutral_venue,
        )
    except Exception as e:
        console.print(f"[bold red]Error running prediction pipeline: {e}[/bold red]")
        logger.exception("Prediction pipeline failed")
        return
    
    # Add ESPN context if available
    if espn_context:
        response["espn_context"] = {
            "event_id": espn_context.get("event_id"),
            "competition": espn_context.get("competition"),
            "stage": espn_context.get("stage"),
            "status": espn_context.get("status"),
            "venue": espn_context.get("venue"),
            "source": espn_context.get("source", "espn"),
        }
    
    console.print("\n[bold green]=== FINAL PREDICTION OUTPUT ===[/bold green]")
    
    # Print match info panel
    match_info = Panel(
        f"[bold]Match:[/bold] {home_team} vs {away_team}\n"
        f"[bold]Competition:[/bold] {match_input.metadata.competition}\n"
        f"[bold]Stage:[/bold] {match_input.metadata.stage.value}\n"
        f"[bold]Date:[/bold] {match_input.metadata.date or 'N/A'}\n"
        f"[bold]Source Mode:[/bold] {source_mode}" +
        (f"\n[bold]Event ID:[/bold] {espn_context.get('event_id')}" if espn_context else ""),
        title="Match Information",
        border_style="blue"
    )
    console.print(match_info)
    
    # Format and print predictions
    display_response = format_predictions_output(response, home_team, away_team)
    
    console.print("\n[bold]Predictions:[/bold]")
    pprint(display_response.get("predictions", {}))
    
    # Print market predictions if available (from pipeline)
    if espn_context and espn_context.get("include_markets"):
        markets = response.get("markets", {})
        if markets:
            console.print("\n[bold]Market Predictions:[/bold]")
            pprint(markets)
    
    # Print team context summary
    team_context = response.get("team_context", {})
    if team_context:
        console.print("\n[bold]Team Context Summary:[/bold]")
        home_ctx = team_context.get("home", {})
        away_ctx = team_context.get("away", {})
        
        console.print(f"  {home_team}:")
        console.print(f"    - Attack λ: {home_ctx.get('lambda_attack', 'N/A')}")
        console.print(f"    - Defense λ: {home_ctx.get('lambda_defense', 'N/A')}")
        console.print(f"    - Form Factor: {home_ctx.get('form_factor', 'N/A')}")
        console.print(f"    - Data Source: {home_ctx.get('data_source', 'N/A')}")
        
        console.print(f"  {away_team}:")
        console.print(f"    - Attack λ: {away_ctx.get('lambda_attack', 'N/A')}")
        console.print(f"    - Defense λ: {away_ctx.get('lambda_defense', 'N/A')}")
        console.print(f"    - Form Factor: {away_ctx.get('form_factor', 'N/A')}")
        console.print(f"    - Data Source: {away_ctx.get('data_source', 'N/A')}")
    
    # Print data freshness
    freshness = response.get("data_freshness", {})
    if freshness:
        console.print("\n[bold]Data Freshness:[/bold]")
        console.print(f"  - Fetched At: {freshness.get('fetched_at', 'N/A')}")
        console.print(f"  - Source: {freshness.get('source', 'N/A')}")
        console.print(f"  - Home Quality: {freshness.get('home_data_quality', 'N/A')}")
        console.print(f"  - Away Quality: {freshness.get('away_data_quality', 'N/A')}")
        
        warnings = freshness.get("warnings", [])
        if warnings:
            console.print(f"  - Warnings: {len(warnings)}")
            for w in warnings[:3]:
                console.print(f"    • {w}")


def mode_upcoming(args) -> None:
    """Handle --mode upcoming."""
    console.print("[bold blue]Fetching upcoming matches from ESPN...[/bold blue]")
    
    try:
        selector = MatchSelector()
        matches = selector.get_upcoming_matches(
            limit=args.limit or 20,
            dates=args.date,
            status_filter="pre"  # Only show upcoming matches by default
        )
    except EspnApiError as e:
        console.print(f"[bold red]ESPN API Error: {e}[/bold red]")
        logger.error(f"ESPN API failed: {e}")
        return
    except MatchSelectionError as e:
        console.print(f"[bold red]Match Selection Error: {e}[/bold red]")
        return
    
    if not matches:
        console.print("[yellow]No upcoming matches found.[/yellow]")
        console.print("Try specifying a different date with --date YYYYMMDD")
        return
    
    # Display matches
    print_match_table(matches, max_display=args.limit or 20)
    
    # Get selection
    if args.auto_pick is not None:
        selected_index = args.auto_pick
    else:
        console.print()
        try:
            selected_index = IntPrompt.ask(
                "[bold]Select match index (or -1 to cancel)[/bold]",
                default=0
            )
        except KeyboardInterrupt:
            console.print("\n[yellow]Cancelled.[/yellow]")
            return
    
    if selected_index < 0:
        console.print("[yellow]Selection cancelled.[/yellow]")
        return
    
    try:
        selected_match = selector.select_by_index(matches, selected_index)
    except MatchSelectionError as e:
        console.print(f"[bold red]Error: {e}[/bold red]")
        return
    
    console.print(f"\n[bold green]Selected: {selected_match.home_team} vs {selected_match.away_team}[/bold green]")
    console.print(f"Event ID: {selected_match.event_id}")
    
    # Build match input and run prediction
    from src.domain.match_input_factory import MatchInputFactory
    
    factory = MatchInputFactory()
    try:
        match_input = factory.build_from_upcoming_match(selected_match)
    except MatchInputBuildError as e:
        console.print(f"[bold red]Error building match input: {e}[/bold red]")
        return
    
    # Get ESPN context
    espn_context = {
        "event_id": selected_match.event_id,
        "competition": selected_match.competition,
        "stage": selected_match.stage,
        "status": selected_match.status,
        "venue": selected_match.venue,
        "source": "espn_scoreboard",
        "include_markets": getattr(args, 'include_markets', False),
    }
    
    run_prediction_from_input(match_input, "upcoming", espn_context)


def mode_match(args) -> None:
    """Handle --mode match with event_id."""
    if not args.event_id:
        console.print("[bold red]Error: --event-id is required for match mode[/bold red]")
        return
    
    console.print(f"[bold blue]Fetching match {args.event_id} from ESPN...[/bold blue]")
    
    try:
        from src.domain.match_input_factory import build_match_input_from_espn_event
        
        match_input = build_match_input_from_espn_event(args.event_id)
    except MatchInputBuildError as e:
        console.print(f"[bold red]Error: {e}[/bold red]")
        logger.error(f"Failed to build match input: {e}")
        return
    
    # Get additional context
    espn_context = {
        "event_id": args.event_id,
        "source": "espn_summary",
        "include_markets": getattr(args, 'include_markets', False),
    }
    
    run_prediction_from_input(match_input, "match", espn_context)


def mode_teams(args) -> None:
    """Handle --mode teams with team names."""
    if not args.home or not args.away:
        console.print("[bold red]Error: --home and --away are required for teams mode[/bold red]")
        return
    
    console.print(f"[bold blue]Building match: {args.home} vs {args.away}[/bold blue]")
    
    try:
        from src.domain.match_input_factory import build_match_input_from_team_names
        
        match_input = build_match_input_from_team_names(
            home_team=args.home,
            away_team=args.away,
            competition=args.competition,
            stage=args.stage,
            match_date=args.date,
        )
    except Exception as e:
        console.print(f"[bold red]Error building match input: {e}[/bold red]")
        logger.error(f"Failed to build match input: {e}")
        return
    
    console.print(f"[green]✓ Teams normalized: {match_input.metadata.home_team} vs {match_input.metadata.away_team}[/green]")
    
    espn_context = {
        "include_markets": getattr(args, 'include_markets', False),
    }
    
    run_prediction_from_input(match_input, "teams", espn_context)


def mode_json(args) -> None:
    """Handle --mode json with file input."""
    if not args.input:
        console.print("[bold red]Error: --input is required for json mode[/bold red]")
        return
    
    input_path = Path(args.input)
    if not input_path.exists():
        console.print(f"[bold red]Error: File not found: {input_path}[/bold red]")
        return
    
    console.print(f"[bold blue]Loading match from {input_path}...[/bold blue]")
    
    try:
        match_input = load_match_json(str(input_path))
    except Exception as e:
        console.print(f"[bold red]Error parsing JSON: {e}[/bold red]")
        logger.error(f"Failed to parse JSON: {e}")
        return
    
    run_prediction_from_input(match_input, "json")


def main():
    parser = argparse.ArgumentParser(
        description="Football Match Predictor with ESPN integration",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # List upcoming matches and select one interactively
  python scripts/predict_match.py --mode upcoming
  
  # Auto-pick first match from upcoming list
  python scripts/predict_match.py --mode upcoming --auto-pick 0
  
  # Predict specific match by ESPN event ID
  python scripts/predict_match.py --mode match --event-id 401234567
  
  # Predict match by team names
  python scripts/predict_match.py --mode teams --home "Mexico" --away "England"
  
  # Legacy JSON mode
  python scripts/predict_match.py --mode json --input data/examples/template_partido.json
  
  # With custom league
  python scripts/predict_match.py --mode upcoming --league eng.1
  
  # With date filter
  python scripts/predict_match.py --mode upcoming --date 20260710
        """
    )
    
    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=["upcoming", "match", "teams", "json"],
        help="Input mode for prediction"
    )
    
    # Mode-specific arguments
    parser.add_argument("--event-id", type=str, help="ESPN event ID (for match mode)")
    parser.add_argument("--home", type=str, help="Home team name (for teams mode)")
    parser.add_argument("--away", type=str, help="Away team name (for teams mode)")
    parser.add_argument("--input", type=str, help="JSON file path (for json mode)")
    
    # Common options
    parser.add_argument("--limit", type=int, default=20, help="Max matches to display (upcoming mode)")
    parser.add_argument("--auto-pick", type=int, help="Auto-select match by index (upcoming mode)")
    parser.add_argument("--date", type=str, help="Date filter YYYYMMDD")
    parser.add_argument("--league", type=str, help="League slug (default: fifa.world)")
    parser.add_argument("--competition", type=str, help="Competition name (teams mode)")
    parser.add_argument("--stage", type=str, help="Match stage (teams mode)")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument(
        "--include-markets",
        action="store_true",
        help="Include additional market predictions (corners, cards, shots, player props)"
    )
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Route to appropriate mode handler
    if args.mode == "upcoming":
        mode_upcoming(args)
    elif args.mode == "match":
        mode_match(args)
    elif args.mode == "teams":
        mode_teams(args)
    elif args.mode == "json":
        mode_json(args)
    else:
        console.print(f"[bold red]Unknown mode: {args.mode}[/bold red]")
        sys.exit(1)


if __name__ == "__main__":
    main()
