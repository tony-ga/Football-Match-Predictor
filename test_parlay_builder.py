
"""
Test the Same Game Parlay Builder
"""
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from predicciones.src.models.parlay_builder import (
    build_all_same_game_parlays,
    render_same_game_parlay_report,
    check_calibration
)
from predicciones.src.pipeline.predict import predict_match_pipeline
from predicciones.src.cli.commands import list_available_fixtures
from rich.console import Console
import pandas as pd
import logging

# Disable repeated missing calibrator warnings
logging.getLogger("predicciones.src.pipeline.predict").setLevel(logging.ERROR)
logging.getLogger("predicciones.src.models.market_derivation").setLevel(logging.ERROR)
logging.getLogger("predicciones.src.models.lambda_recalibration").setLevel(logging.ERROR)
logging.getLogger("predicciones.src.models.calibration").setLevel(logging.ERROR)

console = Console()


def main():
    console.print("\n[bold green]Testing Same Game Parlay Builder[/bold green]\n")
    
    # Step 1: Load available fixtures
    fixtures = list_available_fixtures()
    if not fixtures:
        console.print("[red]No fixtures available![/red]")
        return
        
    # Use the first match from the first fixture
    fixture_path = fixtures[0]['path']
    df = pd.read_csv(fixture_path)
    if len(df) == 0:
        console.print("[red]No matches in the first fixture![/red]")
        return
        
    home_team = df.iloc[0]['home_team']
    away_team = df.iloc[0]['away_team']
    
    # Step 2: Generate prediction for this match
    console.print(f"[dim]Predicting {home_team} vs {away_team}...[/dim]\n")
    pred = predict_match_pipeline(home_team, away_team, include_markets=False)
    
    # Step 3: Build same game parlays
    calib_status = check_calibration()
    parlays, _ = build_all_same_game_parlays(pred, home_team, away_team, calib_status)
    
    # Step 4: Render report
    render_same_game_parlay_report(parlays, pred, home_team, away_team, calib_status, console)


if __name__ == "__main__":
    main()
