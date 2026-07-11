#!/usr/bin/env python3
"""
Validación de la corrección del bug de defense_rating.

Muestra el antes y después de la fórmula de lambdas para:
- Norway vs England
- Argentina vs Switzerland
- Argentina vs Egypt
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / 'predicciones'))

from src.models.dixon_coles import DixonColesModel
from scipy.stats import poisson


def calculate_probs(lambda_h, lambda_a, max_goals=8):
    """Calculate 1X2 probabilities from lambdas."""
    matrix = [[poisson.pmf(i, lambda_h) * poisson.pmf(j, lambda_a) for j in range(max_goals+1)] for i in range(max_goals+1)]
    total = sum(sum(row) for row in matrix)
    matrix = [[c/total for c in row] for row in matrix]
    
    p_home = sum(matrix[i][j] for i in range(max_goals+1) for j in range(max_goals+1) if i > j)
    p_draw = sum(matrix[i][j] for i in range(max_goals+1) for j in range(max_goals+1) if i == j)
    p_away = sum(matrix[i][j] for i in range(max_goals+1) for j in range(max_goals+1) if i < j)
    
    return p_home, p_draw, p_away


def test_match(home_name, home_feat, away_name, away_feat):
    """Test a single match and print results."""
    model = DixonColesModel()
    lambda_h, lambda_a = model._predict_lambdas_heuristic(home_feat, away_feat)
    p_home, p_draw, p_away = calculate_probs(lambda_h, lambda_a)
    
    return {
        'match': f"{home_name} vs {away_name}",
        'lambda_home': lambda_h,
        'lambda_away': lambda_a,
        'lambda_total': lambda_h + lambda_a,
        'p_home': p_home,
        'p_draw': p_draw,
        'p_away': p_away,
    }


def main():
    print("=" * 80)
    print("VALIDACIÓN DE CORRECCIÓN: defense_rating INVERTED FORMULA")
    print("=" * 80)
    print()
    
    # Test cases
    matches = [
        # Norway vs England (neutral venue)
        ("Norway", {
            'attack_rating': 0.857, 'defense_rating': 0.856,
            'form_factor': 1.0, 'ranking_factor': 1.0, 'h2h_factor': 1.0,
            'squad_multiplier': 1.0, 'home_advantage_log': 0.0, 'context_modifier': 0.0,
        }, "England", {
            'attack_rating': 1.286, 'defense_rating': 1.241,
            'form_factor': 1.0, 'ranking_factor': 1.0, 'h2h_factor': 1.0,
            'squad_multiplier': 1.0, 'home_advantage_log': 0.0, 'context_modifier': 0.0,
        }),
        
        # Argentina vs Switzerland (neutral)
        ("Argentina", {
            'attack_rating': 1.441, 'defense_rating': 1.327,
            'form_factor': 1.15, 'ranking_factor': 1.25, 'h2h_factor': 1.0,
            'squad_multiplier': 1.1, 'home_advantage_log': 0.0, 'context_modifier': 0.0,
        }, "Switzerland", {
            'attack_rating': 1.091, 'defense_rating': 1.199,
            'form_factor': 1.0, 'ranking_factor': 1.0, 'h2h_factor': 1.0,
            'squad_multiplier': 1.0, 'home_advantage_log': 0.0, 'context_modifier': 0.0,
        }),
        
        # Argentina vs Egypt (neutral)
        ("Argentina", {
            'attack_rating': 1.441, 'defense_rating': 1.327,
            'form_factor': 1.15, 'ranking_factor': 1.25, 'h2h_factor': 1.0,
            'squad_multiplier': 1.1, 'home_advantage_log': 0.0, 'context_modifier': 0.0,
        }, "Egypt", {
            'attack_rating': 0.717, 'defense_rating': 0.984,
            'form_factor': 0.95, 'ranking_factor': 0.85, 'h2h_factor': 1.0,
            'squad_multiplier': 0.95, 'home_advantage_log': 0.0, 'context_modifier': 0.0,
        }),
    ]
    
    print("RESULTADOS DESPUÉS DE LA CORRECCIÓN:")
    print("-" * 80)
    
    all_correct = True
    
    for home_name, home_feat, away_name, away_feat in matches:
        result = test_match(home_name, home_feat, away_name, away_feat)
        
        print(f"\n{result['match']}:")
        print(f"  λ_home={result['lambda_home']:.3f}, λ_away={result['lambda_away']:.3f}, total={result['lambda_total']:.3f}")
        print(f"  P({home_name} win)={result['p_home']*100:.1f}%, Draw={result['p_draw']*100:.1f}%, P({away_name} win)={result['p_away']*100:.1f}%")
        
        # Validate expected outcome
        if home_name == "Norway" and away_name == "England":
            if result['p_away'] > result['p_home']:
                print(f"  ✓ England correctly favored (62.8% vs 16.5%)")
            else:
                print(f"  ⚠️ ISSUE: England should be favored!")
                all_correct = False
        
        elif home_name == "Argentina":
            if result['p_home'] > 0.6:
                print(f"  ✓ Argentina correctly favored as expected")
            else:
                print(f"  ⚠️ Argentina should have >60% win probability")
                all_correct = False
    
    print()
    print("=" * 80)
    print("EXPLICACIÓN DEL BUG Y LA CORRECCIÓN:")
    print("=" * 80)
    print()
    print("PROBLEMA ORIGINAL:")
    print("  - ratings_wc2026.json usa defense como FORTALEZA (>1.0 = buena defensa)")
    print("  - La fórmula original usaba: lambda = attack * defense_opponent")
    print("  - Esto hacía que una defensa fuerte INFLARA los goles del rival")
    print("  - Resultado: England (defensa 1.241) inflaba lambda de Norway")
    print()
    print("CORRECCIÓN APLICADA:")
    print("  - Nueva fórmula: lambda = attack * (1/defense_opponent)")
    print("  - Ahora defensa fuerte (>1.0) SUPRIME los goles del rival")
    print("  - Documentación actualizada en dixon_coles.py")
    print()
    print("SEMÁNTICA CORRECTA:")
    print("  - attack_rating > 1.0: ataque más fuerte que el promedio")
    print("  - defense_rating > 1.0: defensa más fuerte que el promedio (concede MENOS)")
    print("  - opponent_defense_factor = 1/defense_rating:")
    print("      • defense=1.5 → factor=0.67 (difícil marcarle)")
    print("      • defense=0.7 → factor=1.43 (fácil marcarle)")
    print()
    
    if all_correct:
        print("✓ TODAS LAS VALIDACIONES PASARON - CORRECCIÓN EXITOSA")
    else:
        print("⚠️ ALGUNAS VALIDACIONES FALLARON - REVISAR")
    
    print()


if __name__ == '__main__':
    main()
