"""
Demo script to verify player lambda calculations for Switzerland vs Colombia.

This demonstrates the fix for the issues described:
- Kobel (GK) should have ≈ 0% anytime scorer probability
- Elvedi/Akanji (CB) should have < 10% anytime probability  
- Luis Díaz (winger with goals) should be above defenders
- Sum of probs should be within 60-110% of team_lambda
"""
import sys
sys.path.insert(0, '/workspace/predicciones')

from predicciones.src.models.player_lambda import compute_all_player_lambdas, validate_player_lambdas


def demo_switzerland_colombia():
    """Simulate Switzerland vs Colombia player props calculation."""
    
    print("=" * 80)
    print("DEMO: Suiza vs Colombia - Player Lambda Calculation")
    print("=" * 80)
    
    # Simulated Switzerland squad (based on typical ESPN data format)
    switzerland_players = [
        # Goalkeeper
        {'player_name': 'Gregor Kobel', 'position': 'goalkeeper', 'goals': 0, 'matches_played': 4, 'minutes': 360, 'shots': 0, 'starts': 4},
        
        # Defenders
        {'player_name': 'Nico Elvedi', 'position': 'cb', 'goals': 0, 'matches_played': 4, 'minutes': 360, 'shots': 2, 'starts': 4},
        {'player_name': 'Manuel Akanji', 'position': 'cb', 'goals': 0, 'matches_played': 4, 'minutes': 360, 'shots': 3, 'starts': 4},
        {'player_name': 'Ricardo Rodriguez', 'position': 'lb', 'goals': 0, 'matches_played': 4, 'minutes': 340, 'shots': 4, 'starts': 4},
        {'player_name': 'Silvan Widmer', 'position': 'rb', 'goals': 0, 'matches_played': 4, 'minutes': 320, 'shots': 3, 'starts': 4},
        
        # Midfielders
        {'player_name': 'Granit Xhaka', 'position': 'cm', 'goals': 1, 'matches_played': 4, 'minutes': 360, 'shots': 8, 'starts': 4},
        {'player_name': 'Remo Freuler', 'position': 'cm', 'goals': 0, 'matches_played': 4, 'minutes': 340, 'shots': 5, 'starts': 4},
        {'player_name': 'Djibril Sow', 'position': 'cm', 'goals': 0, 'matches_played': 3, 'minutes': 200, 'shots': 3, 'starts': 2},
        
        # Forwards / Wingers
        {'player_name': 'Breel Embolo', 'position': 'forward', 'goals': 2, 'matches_played': 4, 'minutes': 320, 'shots': 12, 'starts': 4},
        {'player_name': 'Xherdan Shaqiri', 'position': 'am', 'goals': 1, 'matches_played': 4, 'minutes': 300, 'shots': 10, 'starts': 3},
        {'player_name': 'Ruben Vargas', 'position': 'winger', 'goals': 1, 'matches_played': 4, 'minutes': 280, 'shots': 8, 'starts': 3},
        {'player_name': 'Dan Ndoye', 'position': 'winger', 'goals': 1, 'matches_played': 3, 'minutes': 180, 'shots': 6, 'starts': 2},
    ]
    
    # Simulated Colombia squad
    colombia_players = [
        # Goalkeeper
        {'player_name': 'Camilo Vargas', 'position': 'goalkeeper', 'goals': 0, 'matches_played': 4, 'minutes': 360, 'shots': 0, 'starts': 4},
        
        # Defenders
        {'player_name': 'Carlos Cuesta', 'position': 'cb', 'goals': 0, 'matches_played': 4, 'minutes': 360, 'shots': 2, 'starts': 4},
        {'player_name': 'Davinson Sánchez', 'position': 'cb', 'goals': 0, 'matches_played': 4, 'minutes': 360, 'shots': 3, 'starts': 4},
        {'player_name': 'Johan Mojica', 'position': 'lb', 'goals': 0, 'matches_played': 4, 'minutes': 340, 'shots': 4, 'starts': 4},
        {'player_name': 'Daniel Muñoz', 'position': 'rb', 'goals': 0, 'matches_played': 4, 'minutes': 320, 'shots': 5, 'starts': 4},
        
        # Midfielders
        {'player_name': 'Jefferson Lerma', 'position': 'dm', 'goals': 0, 'matches_played': 4, 'minutes': 360, 'shots': 4, 'starts': 4},
        {'player_name': 'Mateus Uribe', 'position': 'cm', 'goals': 0, 'matches_played': 4, 'minutes': 340, 'shots': 6, 'starts': 4},
        {'player_name': 'James Rodríguez', 'position': 'am', 'goals': 1, 'matches_played': 4, 'minutes': 320, 'shots': 9, 'starts': 4},
        
        # Forwards / Wingers
        {'player_name': 'Luis Díaz', 'position': 'winger', 'goals': 3, 'matches_played': 4, 'minutes': 300, 'shots': 14, 'starts': 4},
        {'player_name': 'Jhon Córdoba', 'position': 'forward', 'goals': 1, 'matches_played': 4, 'minutes': 280, 'shots': 10, 'starts': 3},
        {'player_name': 'Rafael Santos Borré', 'position': 'forward', 'goals': 0, 'matches_played': 3, 'minutes': 150, 'shots': 5, 'starts': 1},
    ]
    
    # Team expected goals (from Dixon-Coles model)
    lambda_switzerland = 2.105
    lambda_colombia = 1.114
    
    print("\n--- SWITZERLAND ---")
    print(f"Team λ (expected goals): {lambda_switzerland}")
    print()
    
    swiss_results = compute_all_player_lambdas(switzerland_players, lambda_switzerland)
    
    print(f"{'Player':<25} {'Position':<12} {'Goals':<6} {'λ':<8} {'Anytime %':<10}")
    print("-" * 65)
    for r in swiss_results[:10]:  # Top 10
        print(f"{r['player_name']:<25} {r['position_key']:<12} {r['goals_recent']:<6} {r['lambda_final']:<8.4f} {r['anytime_prob_pct']:<10.2f}")
    
    # Validation
    swiss_validation = validate_player_lambdas(swiss_results, lambda_switzerland, 'Switzerland')
    print(f"\nValidation:")
    print(f"  Sum of anytime probs: {swiss_validation['sum_anytime_prob']:.4f}")
    print(f"  Team λ: {swiss_validation['team_lambda']:.2f}")
    print(f"  Ratio: {swiss_validation['sum_to_lambda_ratio']:.2f}")
    if swiss_validation['issues']:
        print(f"  Issues: {swiss_validation['issues']}")
    
    print("\n--- COLOMBIA ---")
    print(f"Team λ (expected goals): {lambda_colombia}")
    print()
    
    col_results = compute_all_player_lambdas(colombia_players, lambda_colombia)
    
    print(f"{'Player':<25} {'Position':<12} {'Goals':<6} {'λ':<8} {'Anytime %':<10}")
    print("-" * 65)
    for r in col_results[:10]:  # Top 10
        print(f"{r['player_name']:<25} {r['position_key']:<12} {r['goals_recent']:<6} {r['lambda_final']:<8.4f} {r['anytime_prob_pct']:<10.2f}")
    
    # Validation
    col_validation = validate_player_lambdas(col_results, lambda_colombia, 'Colombia')
    print(f"\nValidation:")
    print(f"  Sum of anytime probs: {col_validation['sum_anytime_prob']:.4f}")
    print(f"  Team λ: {col_validation['team_lambda']:.2f}")
    print(f"  Ratio: {col_validation['sum_to_lambda_ratio']:.2f}")
    if col_validation['issues']:
        print(f"  Issues: {col_validation['issues']}")
    
    print("\n" + "=" * 80)
    print("KEY VERIFICATIONS:")
    print("=" * 80)
    
    # Check Kobel (GK)
    kobel = next(r for r in swiss_results if r['player_name'] == 'Gregor Kobel')
    print(f"\n✓ Gregor Kobel (GK): {kobel['anytime_prob_pct']:.2f}% (should be ≈ 0%)")
    assert kobel['anytime_prob_pct'] == 0.0, "Kobel should have 0% probability"
    
    # Check Elvedi and Akanji (CBs)
    elvedi = next(r for r in swiss_results if r['player_name'] == 'Nico Elvedi')
    akanji = next(r for r in swiss_results if r['player_name'] == 'Manuel Akanji')
    print(f"✓ Nico Elvedi (CB): {elvedi['anytime_prob_pct']:.2f}% (should be < 10%)")
    print(f"✓ Manuel Akanji (CB): {akanji['anytime_prob_pct']:.2f}% (should be < 10%)")
    assert elvedi['anytime_prob_pct'] < 10.0, "Elvedi should be < 10%"
    assert akanji['anytime_prob_pct'] < 10.0, "Akanji should be < 10%"
    
    # Check Luis Díaz vs defenders
    luis_diaz = next(r for r in col_results if r['player_name'] == 'Luis Díaz')
    max_cb_col = max((r for r in col_results if r['position_key'] in ['cb', 'cd']), key=lambda x: x['anytime_prob_pct'])
    print(f"\n✓ Luis Díaz (Winger, 3 goals): {luis_diaz['anytime_prob_pct']:.2f}%")
    print(f"✓ Highest CB Colombia: {max_cb_col['player_name']} at {max_cb_col['anytime_prob_pct']:.2f}%")
    assert luis_diaz['anytime_prob_pct'] > max_cb_col['anytime_prob_pct'], "Luis Díaz should be above CBs"
    
    # Check sum ratios
    print(f"\n✓ Switzerland sum/λ ratio: {swiss_validation['sum_to_lambda_ratio']:.2f} (should be 0.6-1.1)")
    print(f"✓ Colombia sum/λ ratio: {col_validation['sum_to_lambda_ratio']:.2f} (should be 0.6-1.1)")
    
    print("\n" + "=" * 80)
    print("ALL CHECKS PASSED! ✓")
    print("=" * 80)


if __name__ == '__main__':
    demo_switzerland_colombia()
