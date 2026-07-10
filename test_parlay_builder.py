
"""
Test the automatic parlay builder
"""
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from predicciones.src.models.parlay_builder import build_all_parlays
from predicciones.src.pipeline.predict import predict_match_pipeline
from rich.console import Console
import pandas as pd

console = Console()

def main():
    # Step 1: Load available fixtures
    from predicciones.src.cli.commands import list_available_fixtures
    fixtures = list_available_fixtures()
    if not fixtures:
        console.print("[red]No fixtures available![/red]")
        return

    # Step 2: Generate predictions for all fixtures
    match_predictions = []
    for fixture in fixtures:
        try:
            fixture_path = fixture['path']
            df = pd.read_csv(fixture_path)
            if len(df) == 0:
                continue

            for _, match in df.iterrows():
                home_team = match['home_team']
                away_team = match['away_team']
                console.print(f"[dim]Predicting {home_team} vs {away_team}...[/dim]")
                pred = predict_match_pipeline(
                    home_team,
                    away_team,
                    include_markets=False
                )
                match_predictions.append(pred)
        except Exception as e:
            console.print(f"[yellow]Skipping fixture {fixture['date']}: {str(e)}[/yellow]")

    if not match_predictions:
        console.print("[red]No valid match predictions available to build parlays![/red]")
        return

    # Step 3: Build and render all parlays
    from predicciones.src.models.parlay_builder import render_parlay, RiskLevel
    parlays = build_all_parlays(match_predictions)
    
    for risk_level in [RiskLevel.LOW, RiskLevel.MEDIUM, RiskLevel.HIGH]:
        render_parlay(parlays[risk_level], risk_level, console)

if __name__ == "__main__":
    main()
