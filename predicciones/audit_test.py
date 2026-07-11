
"""
Audit test script to verify pipeline correctness.
- Checks score matrix normalization
- Checks 1X2, BTTS, Over/Under consistency
- Tests end-to-end prediction flow
"""
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
import numpy as np
from predicciones.src.models.dixon_coles import DixonColesModel
from predicciones.src.models.market_derivation import derive_all_markets
from predicciones.src.cli.commands import list_available_fixtures
from predicciones.src.pipeline.predict import predict_match_pipeline
from rich.console import Console

console = Console()


def test_score_matrix_normalization():
    """Test that score matrix sums to 1 (or approx 1) and has valid probabilities."""
    console.print("\n[bold blue]Testing Score Matrix Normalization[/bold blue]")
    
    dc_model = DixonColesModel()
    dc_model.max_goals = 8
    
    test_cases = [
        (1.0, 1.0),
        (1.5, 0.8),
        (0.5, 2.5),
        (2.0, 2.0),
        (0.1, 0.1),
    ]
    
    for lambda_h, lambda_a in test_cases:
        matrix = dc_model.score_matrix(lambda_h, lambda_a)
        total = matrix.sum()
        min_prob = matrix.min()
        max_prob = matrix.max()
        
        console.print(f"  λ_h={lambda_h:.2f}, λ_a={lambda_a:.2f}:")
        console.print(f"    Total sum: {total:.6f} (should be ≈1.0)")
        console.print(f"    Min probability: {min_prob:.6f} (should be ≥0)")
        console.print(f"    Max probability: {max_prob:.6f} (should be ≤1)")
        
        # Check conditions
        assert abs(total - 1.0) < 1e-6, f"Matrix doesn't sum to 1: {total}"
        assert min_prob >= 0, f"Negative probabilities: {min_prob}"
        assert max_prob <= 1, f"Probability >1: {max_prob}"
        
        console.print("  [green]✓ Passed[/green]")


def test_market_derivation_consistency():
    """Test that derived markets are consistent and probabilities sum correctly."""
    console.print("\n[bold blue]Testing Market Derivation Consistency[/bold blue]")
    
    dc_model = DixonColesModel()
    dc_model.max_goals = 8
    lambda_h = 1.5
    lambda_a = 1.0
    matrix = dc_model.score_matrix(lambda_h, lambda_a)
    
    markets = derive_all_markets(matrix, lambda_h, lambda_a)
    
    # Check 1X2
    p1x2 = markets['1x2']
    total_1x2 = p1x2['home'] + p1x2['draw'] + p1x2['away']
    console.print(f"  1X2 sum: {total_1x2:.6f} (should be 1.0)")
    assert abs(total_1x2 - 1.0) < 1e-6
    
    # Check BTTS
    pbtts = markets['btts']
    total_btts = pbtts['yes'] + pbtts['no']
    console.print(f"  BTTS sum: {total_btts:.6f} (should be 1.0)")
    assert abs(total_btts - 1.0) < 1e-6
    
    # Check Over/Under
    ou = markets['over_under']
    for threshold in ['1_5', '2_5', '3_5', '4_5']:
        total_ou = ou[f'over_{threshold}'] + ou[f'under_{threshold}']
        console.print(f"  Over/Under {threshold.replace('_', '.')} sum: {total_ou:.6f} (should be 1.0)")
        assert abs(total_ou - 1.0) < 1e-6
    
    console.print("  [green]✓ All markets consistent[/green]")


def test_end_to_end_prediction():
    """Test end-to-end prediction flow with an existing fixture."""
    console.print("\n[bold blue]Testing End-to-End Prediction[/bold blue]")
    
    fixtures = list_available_fixtures()
    if fixtures:
        console.print(f"  Found {len(fixtures)} available fixtures")
        
        # Use the first fixture
        fixture_path = fixtures[0]['path']
        console.print(f"  Using fixture: {fixture_path}")
        
        # Load fixture
        df_fixture = pd.read_csv(fixture_path)
        if len(df_fixture) > 0:
            match = df_fixture.iloc[0]
            home_team = match['home_team']
            away_team = match['away_team']
            console.print(f"  Predicting: {home_team} vs {away_team}")
            
            # Run prediction (disable alternative markets for speed)
            prediction = predict_match_pipeline(
                home_team, away_team, include_markets=False
            )
            
            console.print(f"\n  [bold green]Prediction Results:[/bold green]")
            console.print(f"    Expected goals: Home={prediction['team_context']['home']['lambda_attack']:.2f}, Away={prediction['team_context']['away']['lambda_attack']:.2f}")
            console.print(f"    1X2: Home={prediction['predictions']['1x2']['home']:.1%}, Draw={prediction['predictions']['1x2']['draw']:.1%}, Away={prediction['predictions']['1x2']['away']:.1%}")
            console.print(f"    BTTS: Yes={prediction['predictions']['btts']['yes']:.1%}, No={prediction['predictions']['btts']['no']:.1%}")
            console.print(f"  [green]✓ End-to-end test passed[/green]")
    else:
        console.print("[yellow]No fixtures found, skipping end-to-end test[/yellow]")


if __name__ == "__main__":
    console.print("[bold green]Starting Football Match Predictor Audit[/bold green]")
    
    test_score_matrix_normalization()
    test_market_derivation_consistency()
    test_end_to_end_prediction()
    
    console.print("\n[bold green]🎉 All audit tests passed![/bold green]")
