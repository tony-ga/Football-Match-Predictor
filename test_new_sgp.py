
#!/usr/bin/env python3
"""Test the new Same Game Parlay Builder"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from predicciones.src.models.parlay_builder import (
    check_calibration,
    build_all_same_game_parlays,
    render_same_game_parlay_report
)
from predicciones.src.pipeline.predict import predict_match_pipeline
from rich.console import Console

def test_new_sgp():
    console = Console()
    
    console.print("[bold blue]Testing New Same Game Parlay Builder[/bold blue]\n")
    
    # Test match 1 - let's pick Brazil vs Norway first
    home_team = "Brasil"
    away_team = "Norway"
    
    console.print(f"[bold]Generating predictions for {home_team} vs {away_team}...[/bold]")
    pred = predict_match_pipeline(home_team, away_team, include_markets=False)
    
    # Inject mock data for the new market families
    pred["corners"] = {
        "home_avg_for": 6.5,
        "home_avg_against": 4.2,
        "away_avg_for": 5.1,
        "away_avg_against": 5.5,
        "total_std_dev": 2.5
    }
    
    pred["player_stats"] = [
        {"name": "Neymar", "shots_p90": 3.8, "expected_minutes": 90},
        {"name": "Vinicius Jr", "shots_p90": 2.9, "expected_minutes": 85},
        {"name": "Haaland", "shots_p90": 4.1, "expected_minutes": 90}
    ]
    
    # Check calibration and build parlays
    calib_status = check_calibration()
    parlays, _ = build_all_same_game_parlays(pred, home_team, away_team, calib_status)
    
    # Render report
    render_same_game_parlay_report(parlays, pred, home_team, away_team, calib_status, console)

if __name__ == "__main__":
    test_new_sgp()
