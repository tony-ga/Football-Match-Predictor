"""
Football Prediction System - Main CLI Application
==================================================

A unified command-line interface for the football prediction system.
Supports both interactive menu mode and direct CLI commands.

Usage:
    python app.py                      # Interactive menu
    python app.py predict --fixture data/fixtures/test.csv
    python app.py pipeline --date 20260711 --verbose
"""
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.prompt import Prompt, Confirm
from rich.text import Text

from predicciones.src.cli.menu import InteractiveMenu
from predicciones.src.cli.commands import (
    predict_command,
    pipeline_command,
    players_command,
    timelines_command,
    fixtures_command,
    daily_report_command,
    lambda_analysis_command,
    backtest_command,
    recent_command,
)
from predicciones.src.utils.config_loader import config

console = Console()

app = typer.Typer(
    name="football-predictor",
    help="Football Match Prediction System CLI",
    add_completion=True,
)


# =============================================================================
# Interactive Menu Mode (default)
# =============================================================================


@app.callback(invoke_without_command=True)
def main(
    ctx: typer.Context,
    version: bool = typer.Option(
        False, "--version", "-v", help="Show version and exit"
    ),
):
    """
    Football Prediction System - Main Entry Point
    
    Run without arguments to start interactive menu mode.
    Use subcommands for direct CLI access.
    """
    if version:
        console.print("[bold green]Football Predictor v2.0.0[/bold green]")
        console.print(f"Config loaded: {bool(config)}")
        raise typer.Exit()
    
    # If no subcommand was invoked, start interactive menu
    if ctx.invoked_subcommand is None:
        start_interactive_mode()


def start_interactive_mode():
    """Start the interactive menu system."""
    console.print()
    console.print(Panel.fit(
        "[bold blue]⚽ Football Prediction System[/bold blue]\n"
        "[dim]Interactive Menu Mode[/dim]",
        border_style="blue",
    ))
    console.print()
    
    menu = InteractiveMenu(console)
    menu.run()


# =============================================================================
# CLI Commands (Advanced Mode)
# =============================================================================


@app.command("predict")
def predict_cli(
    fixture: str = typer.Option(..., "--fixture", "-f", help="Path to fixture CSV file"),
    output_dir: str = typer.Option("output/predictions", "--output-dir", "-o", help="Output directory"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable verbose output"),
    date: str = typer.Option(None, "--date", "-d", help="Override date (YYYYMMDD)"),
):
    """Generate predictions from a fixture CSV file."""
    predict_command(fixture, output_dir, verbose, date)


@app.command("pipeline")
def pipeline_cli(
    date: str = typer.Option(..., "--date", "-d", help="Date to process (YYYYMMDD)"),
    output_dir: str = typer.Option("output/daily", "--output-dir", "-o", help="Output directory"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable verbose output"),
    skip_validation: bool = typer.Option(False, "--skip-validation", help="Skip sanity checks"),
):
    """Run the complete daily prediction pipeline."""
    pipeline_command(date, output_dir, verbose, skip_validation)


@app.command("players")
def players_cli(
    team: str = typer.Option(..., "--team", "-t", help="Team name"),
    max_matches: int = typer.Option(10, "--max-matches", "-m", help="Max matches to consider"),
    output_format: str = typer.Option("table", "--format", help="Output format: table, csv, json"),
):
    """Get player statistics for a selected team."""
    players_command(team, max_matches, output_format)


@app.command("timelines")
def timelines_cli(
    match_id: str = typer.Option(..., "--match-id", "-m", help="Match ID or fixture"),
    output_dir: str = typer.Option("output/timelines", "--output-dir", "-o", help="Output directory"),
):
    """Get timeline data for past matches."""
    timelines_command(match_id, output_dir)


@app.command("fixtures")
def fixtures_cli(
    date: str = typer.Option(..., "--date", "-d", help="Date (YYYYMMDD)"),
    competition: str = typer.Option(None, "--competition", "-c", help="Filter by competition"),
    output_format: str = typer.Option("table", "--format", help="Output format: table, csv"),
):
    """Get fixtures for a specific date."""
    fixtures_command(date, competition, output_format)


@app.command("daily-report")
def daily_report_cli(
    date: str = typer.Option(..., "--date", "-d", help="Date (YYYYMMDD)"),
    output_dir: str = typer.Option("output/reports", "--output-dir", "-o", help="Output directory"),
    include_analysis: bool = typer.Option(True, "--include-analysis/--no-analysis", help="Include detailed analysis"),
):
    """Generate daily prediction report."""
    daily_report_command(date, output_dir, include_analysis)


@app.command("lambda-analysis")
def lambda_analysis_cli(
    num_matches: int = typer.Option(50, "--num-matches", "-n", help="Number of matches to analyze"),
    output_dir: str = typer.Option("output/lambda_validation", "--output-dir", "-o", help="Output directory"),
    threshold_home: float = typer.Option(3.0, "--threshold-home", help="Home lambda threshold"),
    threshold_away: float = typer.Option(2.5, "--threshold-away", help="Away lambda threshold"),
    threshold_total: float = typer.Option(5.0, "--threshold-total", help="Total lambda threshold"),
):
    """Analyze lambda distribution across matches."""
    lambda_analysis_command(num_matches, output_dir, threshold_home, threshold_away, threshold_total)


@app.command("backtest")
def backtest_cli(
    num_matches: int = typer.Option(200, "--num-matches", "-n", help="Number of matches for backtest"),
    output_dir: str = typer.Option("output/calibration_eval", "--output-dir", "-o", help="Output directory"),
    compare_markov: bool = typer.Option(True, "--compare-markov/--no-compare-markov", help="Compare baseline vs markov-aware"),
):
    """Run backtest/calibration evaluation."""
    backtest_command(num_matches, output_dir, compare_markov)


@app.command("recent")
def recent_cli(
    limit: int = typer.Option(10, "--limit", "-l", help="Number of recent files to show"),
    file_type: str = typer.Option("all", "--type", "-t", help="File type: all, predictions, reports, metrics"),
    show_summary: bool = typer.Option(False, "--summary", "-s", help="Show summary of latest file"),
):
    """View recently generated files and reports."""
    recent_command(limit, file_type, show_summary)


@app.command("config")
def config_cli(
    show_all: bool = typer.Option(False, "--all", "-a", help="Show all configuration"),
    section: str = typer.Option(None, "--section", "-s", help="Show specific section"),
    edit: bool = typer.Option(False, "--edit", "-e", help="Open config file in editor"),
):
    """View or edit system configuration."""
    from predicciones.src.cli.commands import config_command
    config_command(show_all, section, edit)


if __name__ == "__main__":
    app()
