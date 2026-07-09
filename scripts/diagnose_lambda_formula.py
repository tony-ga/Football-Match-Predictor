#!/usr/bin/env python3
"""
Diagnostic script to verify lambda formula and defense rating semantics.

Tests the specific issue:
- Norway (fallback ratings) vs England (real ratings)
- Expected: England should be favored or at least not underdog
- Observed bug: Norway win 45%, England win 30% due to inverted defense interpretation
"""
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'predicciones'))

from src.models.dixon_coles import DixonColesModel
from src.features.ratings import LEAGUE_AVG_GOALS


def test_norway_england():
    """Test Norway vs England with current formula."""
    print("=" * 70)
    print("DIAGNOSTIC: Norway vs England Lambda Calculation")
    print("=" * 70)
    
    # Simulate features as they would come from feature pipeline
    norway_features = {
        'nombre': 'Norway',
        'attack_rating': 0.857,  # fallback value
        'defense_rating': 0.856,  # fallback value
        'form_factor': 1.0,
        'ranking_factor': 1.0,
        'h2h_factor': 1.0,
        'squad_multiplier': 1.0,
        'home_advantage_log': 0.0,  # neutral venue
        'context_modifier': 0.0,
    }
    
    england_features = {
        'nombre': 'England',
        'attack_rating': 1.286,  # real rating from JSON
        'defense_rating': 1.241,  # real rating from JSON
        'form_factor': 1.0,
        'ranking_factor': 1.0,
        'h2h_factor': 1.0,
        'squad_multiplier': 1.0,
        'home_advantage_log': 0.0,  # neutral venue
        'context_modifier': 0.0,
    }
    
    model = DixonColesModel()
    
    # Get lambdas using heuristic mode
    lambda_home, lambda_away = model._predict_lambdas_heuristic(
        norway_features, 
        england_features
    )
    
    print(f"\nINPUT RATINGS:")
    print(f"  Norway attack:  {norway_features['attack_rating']:.3f}")
    print(f"  Norway defense: {norway_features['defense_rating']:.3f}")
    print(f"  England attack: {england_features['attack_rating']:.3f}")
    print(f"  England defense: {england_features['defense_rating']:.3f}")
    print(f"  LEAGUE_AVG_GOALS: {LEAGUE_AVG_GOALS}")
    
    print(f"\nFORMULA (current implementation):")
    print(f"  lambda_home (Norway) = attack_NOR * defense_ENG * LEAGUE_AVG_GOALS")
    print(f"                       = {norway_features['attack_rating']:.3f} * {england_features['defense_rating']:.3f} * {LEAGUE_AVG_GOALS}")
    print(f"                       = {norway_features['attack_rating'] * england_features['defense_rating'] * LEAGUE_AVG_GOALS:.3f}")
    print(f"")
    print(f"  lambda_away (England) = attack_ENG * defense_NOR * LEAGUE_AVG_GOALS")
    print(f"                        = {england_features['attack_rating']:.3f} * {norway_features['defense_rating']:.3f} * {LEAGUE_AVG_GOALS}")
    print(f"                        = {england_features['attack_rating'] * norway_features['defense_rating'] * LEAGUE_AVG_GOALS:.3f}")
    
    print(f"\nRESULTING LAMBDAS:")
    print(f"  lambda_home (Norway):  {lambda_home:.4f}")
    print(f"  lambda_away (England): {lambda_away:.4f}")
    print(f"  lambda_total:          {lambda_home + lambda_away:.4f}")
    
    # Calculate probabilities
    from scipy.stats import poisson
    max_goals = 8
    matrix = [[0.0] * (max_goals + 1) for _ in range(max_goals + 1)]
    
    for i in range(max_goals + 1):
        for j in range(max_goals + 1):
            p_home = poisson.pmf(i, lambda_home)
            p_away = poisson.pmf(j, lambda_away)
            matrix[i][j] = p_home * p_away
    
    # Normalize
    total = sum(sum(row) for row in matrix)
    matrix = [[cell / total for cell in row] for row in matrix]
    
    # Calculate 1X2 probabilities
    p_home_win = sum(matrix[i][j] for i in range(max_goals + 1) for j in range(max_goals + 1) if i > j)
    p_draw = sum(matrix[i][j] for i in range(max_goals + 1) for j in range(max_goals + 1) if i == j)
    p_away_win = sum(matrix[i][j] for i in range(max_goals + 1) for j in range(max_goals + 1) if i < j)
    
    print(f"\nPREDICTED PROBABILITIES (1X2):")
    print(f"  Norway win:  {p_home_win * 100:.1f}%")
    print(f"  Draw:        {p_draw * 100:.1f}%")
    print(f"  England win: {p_away_win * 100:.1f}%")
    
    print(f"\n🔍 ANALYSIS:")
    if p_home_win > p_away_win:
        print(f"  ⚠️  BUG CONFIRMED: Norway is favored despite having WORSE ratings!")
        print(f"      Norway attack ({norway_features['attack_rating']:.3f}) < England attack ({england_features['attack_rating']:.3f})")
        print(f"      But England's high defense_rating ({england_features['defense_rating']:.3f})")
        print(f"      INFLATES Norway's lambda instead of suppressing it.")
    else:
        print(f"  ✓ England correctly favored")
    
    return {
        'lambda_home': lambda_home,
        'lambda_away': lambda_away,
        'p_home_win': p_home_win,
        'p_draw': p_draw,
        'p_away_win': p_away_win,
    }


def test_argentina_matches():
    """Test Argentina vs Switzerland and Argentina vs Egypt."""
    print("\n" + "=" * 70)
    print("DIAGNOSTIC: Argentina Matches")
    print("=" * 70)
    
    # Example ratings (adjust based on actual data)
    argentina_features = {
        'nombre': 'Argentina',
        'attack_rating': 1.45,  # strong attack
        'defense_rating': 0.95,  # solid defense (< 1.0 is good)
        'form_factor': 1.15,
        'ranking_factor': 1.25,
        'h2h_factor': 1.0,
        'squad_multiplier': 1.1,
        'home_advantage_log': 0.0,
        'context_modifier': 0.0,
    }
    
    switzerland_features = {
        'nombre': 'Switzerland',
        'attack_rating': 1.05,
        'defense_rating': 1.08,
        'form_factor': 1.0,
        'ranking_factor': 1.0,
        'h2h_factor': 1.0,
        'squad_multiplier': 1.0,
        'home_advantage_log': 0.0,
        'context_modifier': 0.0,
    }
    
    egypt_features = {
        'nombre': 'Egypt',
        'attack_rating': 0.92,
        'defense_rating': 1.15,
        'form_factor': 0.95,
        'ranking_factor': 0.85,
        'h2h_factor': 1.0,
        'squad_multiplier': 0.95,
        'home_advantage_log': 0.0,
        'context_modifier': 0.0,
    }
    
    model = DixonColesModel()
    
    matches = [
        ('Argentina', argentina_features, 'Switzerland', switzerland_features),
        ('Argentina', argentina_features, 'Egypt', egypt_features),
    ]
    
    results = []
    for home_name, home_feat, away_name, away_feat in matches:
        lambda_h, lambda_a = model._predict_lambdas_heuristic(home_feat, away_feat)
        
        from scipy.stats import poisson
        max_goals = 8
        matrix = [[0.0] * (max_goals + 1) for _ in range(max_goals + 1)]
        for i in range(max_goals + 1):
            for j in range(max_goals + 1):
                matrix[i][j] = poisson.pmf(i, lambda_h) * poisson.pmf(j, lambda_a)
        total = sum(sum(row) for row in matrix)
        matrix = [[cell / total for cell in row] for row in matrix]
        
        p_home = sum(matrix[i][j] for i in range(max_goals + 1) for j in range(max_goals + 1) if i > j)
        p_draw = sum(matrix[i][j] for i in range(max_goals + 1) for j in range(max_goals + 1) if i == j)
        p_away = sum(matrix[i][j] for i in range(max_goals + 1) for j in range(max_goals + 1) if i < j)
        
        print(f"\n{home_name} vs {away_name}:")
        print(f"  λ_home={lambda_h:.3f}, λ_away={lambda_a:.3f}, total={lambda_h+lambda_a:.3f}")
        print(f"  P({home_name} win)={p_home*100:.1f}%, Draw={p_draw*100:.1f}%, P({away_name} win)={p_away*100:.1f}%")
        
        results.append({
            'match': f"{home_name} vs {away_name}",
            'lambda_home': lambda_h,
            'lambda_away': lambda_a,
            'p_home': p_home,
            'p_draw': p_draw,
            'p_away': p_away,
        })
    
    return results


def main():
    result_nor_eng = test_norway_england()
    result_arg = test_argentina_matches()
    
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print("\nThe bug is in the SEMANTICS of defense_rating:")
    print("  - Documentation says: 'lower = better defense'")
    print("  - Formula uses: lambda = attack * defense_opponent")
    print("  - This means a HIGH defense_rating INFLATES opponent's lambda")
    print("  - But if defense_rating represents 'goals conceded', high = BAD defense")
    print("")
    print("SOLUTION NEEDED:")
    print("  Option A: Invert defense in formula: lambda = attack * (1/defense)")
    print("  Option B: Rename to 'vulnerability_rating' or 'conceded_multiplier'")
    print("  Option C: Change computation so lower values mean BETTER defense")
    print("")
    print("RECOMMENDED: Option A + clear documentation update")
    

if __name__ == '__main__':
    main()
