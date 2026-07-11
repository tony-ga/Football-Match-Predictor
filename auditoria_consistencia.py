#!/usr/bin/env python3
"""
Auditoría de consistencia interna para predicciones de France vs Morocco
"""

import csv
import json

# Datos extraídos del CSV para France vs Morocco
expected_goals_home = 2.2163071502801674
expected_goals_away = 1.34841254750773

home_goals_distribution = [
    {'goals': 0, 'probability': 0.10521452142216776},
    {'goals': 1, 'probability': 0.23782284979359986},
    {'goals': 2, 'probability': 0.26878280259911225},
    {'goals': 3, 'probability': 0.2025154298272353},
    {'goals': 4, 'probability': 0.11443951841837065},
    {'goals': '5+', 'probability': 0.07122487793951429}
]

away_goals_distribution = [
    {'goals': 0, 'probability': 0.2590156627407627},
    {'goals': 1, 'probability': 0.350026884561874},
    {'goals': 2, 'probability': 0.2365085157778954},
    {'goals': 3, 'probability': 0.10653710805762448},
    {'goals': 4, 'probability': 0.03599285428248867},
    {'goals': '5+', 'probability': 0.011918974579354844}
]

clean_sheets_home = 0.2590156627407627  # probabilidad de que home no reciba goles
clean_sheets_away = 0.10521452142216776  # probabilidad de que away no reciba goles

team_totals = {
    'home_over_0_5': 0.8947854785778324,
    'away_over_0_5': 0.7409843372592374,
    'home_over_1_5': 0.6569626287842325,
    'away_over_1_5': 0.3909574526973633,
    'home_over_2_5': 0.3881798261851202,
    'away_over_2_5': 0.15444893691946798
}

def calculate_expected_value(distribution, lambda_ref=None):
    """Calcula la esperanza matemática de una distribución de goles truncada.
    
    Para el bucket '5+', usamos el valor esperado condicional E[X | X >= 5]
    de una Poisson con lambda=lambda_ref para mayor precisión.
    Si lambda_ref no se proporciona, usamos 6.0 como aproximación.
    """
    from scipy.stats import poisson
    
    expected = 0.0
    for item in distribution:
        goals = item['goals']
        prob = item['probability']
        if goals == '5+':
            if lambda_ref is not None:
                # Calcular E[X | X >= 5] para Poisson(lambda_ref)
                prob_5_plus = sum(poisson.pmf(k, lambda_ref) for k in range(5, 50))
                if prob_5_plus > 0:
                    e_conditional = sum(k * poisson.pmf(k, lambda_ref) for k in range(5, 50)) / prob_5_plus
                    goals = e_conditional
                else:
                    goals = 6.0
            else:
                goals = 6.0
        expected += goals * prob
    return expected

def verify_clean_sheet(distribution, clean_sheet_prob, team_name):
    """Verifica que clean_sheet coincida con P(0 goles) del oponente"""
    prob_zero = None
    for item in distribution:
        if item['goals'] == 0:
            prob_zero = item['probability']
            break
    
    match = abs(prob_zero - clean_sheet_prob) < 1e-10
    return prob_zero, match

def verify_team_totals(distribution, team_totals, team_name):
    """Verifica que team_totals sean consistentes con la distribución"""
    results = {}
    
    # P(over 0.5) = 1 - P(0)
    prob_0 = next(item['probability'] for item in distribution if item['goals'] == 0)
    expected_over_0_5 = 1 - prob_0
    results['over_0_5'] = {
        'expected': expected_over_0_5,
        'actual': team_totals[f'{team_name}_over_0_5'],
        'match': abs(expected_over_0_5 - team_totals[f'{team_name}_over_0_5']) < 1e-10
    }
    
    # P(over 1.5) = 1 - P(0) - P(1)
    prob_1 = next(item['probability'] for item in distribution if item['goals'] == 1)
    expected_over_1_5 = 1 - prob_0 - prob_1
    results['over_1_5'] = {
        'expected': expected_over_1_5,
        'actual': team_totals[f'{team_name}_over_1_5'],
        'match': abs(expected_over_1_5 - team_totals[f'{team_name}_over_1_5']) < 1e-10
    }
    
    # P(over 2.5) = 1 - P(0) - P(1) - P(2)
    prob_2 = next(item['probability'] for item in distribution if item['goals'] == 2)
    expected_over_2_5 = 1 - prob_0 - prob_1 - prob_2
    results['over_2_5'] = {
        'expected': expected_over_2_5,
        'actual': team_totals[f'{team_name}_over_2_5'],
        'match': abs(expected_over_2_5 - team_totals[f'{team_name}_over_2_5']) < 1e-10
    }
    
    return results

print("=" * 80)
print("AUDITORÍA DE CONSISTENCIA INTERNA: France vs Morocco")
print("=" * 80)

# 1. Calcular esperanzas implícitas
print("\n1. ESPERANZA IMPLÍCITA DE DISTRIBUCIONES")
print("-" * 50)

home_expected = calculate_expected_value(home_goals_distribution, lambda_ref=expected_goals_home)
away_expected = calculate_expected_value(away_goals_distribution, lambda_ref=expected_goals_away)

# Nota: Las distribuciones vienen de una matriz Dixon-Coles (con corrección rho),
# no de Poisson puro, por lo que puede haber pequeñas diferencias (< 0.02)
tolerance = 0.02

print(f"home_goals_distribution -> Esperanza implícita: {home_expected:.4f}")
print(f"  expected_goals.home reportado: {expected_goals_home:.4f}")
print(f"  Diferencia: {abs(home_expected - expected_goals_home):.6f}")
print(f"  ¿Coincide? (tolerancia {tolerance}) {abs(home_expected - expected_goals_home) < tolerance}")

print()

print(f"away_goals_distribution -> Esperanza implícita: {away_expected:.4f}")
print(f"  expected_goals.away reportado: {expected_goals_away:.4f}")
print(f"  Diferencia: {abs(away_expected - expected_goals_away):.6f}")
print(f"  ¿Coincide? (tolerancia {tolerance}) {abs(away_expected - expected_goals_away) < tolerance}")

# 2. Verificar inversión de etiquetas
print("\n2. VERIFICACIÓN DE INVERSIÓN DE ETIQUETAS")
print("-" * 50)

# Si las distribuciones estuvieran invertidas:
home_expected_if_inverted = away_expected
away_expected_if_inverted = home_expected

print(f"Si estuvieran INVERTIDAS:")
print(f"  home_goals_distribution correspondería a expected_goals.away ({expected_goals_away:.4f})")
print(f"    pero su esperanza es {home_expected:.4f} -> ¿Coincide? {abs(home_expected - expected_goals_away) < 0.01}")
print(f"  away_goals_distribution correspondería a expected_goals.home ({expected_goals_home:.4f})")
print(f"    pero su esperanza es {away_expected:.4f} -> ¿Coincide? {abs(away_expected - expected_goals_home) < 0.01}")

inversion_detected = (abs(home_expected - expected_goals_away) < 0.01 and 
                      abs(away_expected - expected_goals_home) < 0.01)
print(f"\n¿Hay inversión detectada? {inversion_detected}")

if not inversion_detected:
    print("✓ Las etiquetas NO están invertidas (coherencia confirmada)")
else:
    print("✗ ALERTA: Las etiquetas ESTÁN invertidas!")

# 3. Verificar clean sheets
print("\n3. VERIFICACIÓN DE CLEAN SHEETS")
print("-" * 50)

# clean_sheets.home = probabilidad de que home no reciba goles = P(away marca 0)
# clean_sheets.away = probabilidad de que away no reciba goles = P(home marca 0)

prob_away_0 = next(item['probability'] for item in away_goals_distribution if item['goals'] == 0)
prob_home_0 = next(item['probability'] for item in home_goals_distribution if item['goals'] == 0)

print(f"clean_sheets.home reportado: {clean_sheets_home:.6f}")
print(f"P(away_goals = 0) de away_goals_distribution: {prob_away_0:.6f}")
print(f"  ¿Coincide? {abs(clean_sheets_home - prob_away_0) < 1e-10}")

print()

print(f"clean_sheets.away reportado: {clean_sheets_away:.6f}")
print(f"P(home_goals = 0) de home_goals_distribution: {prob_home_0:.6f}")
print(f"  ¿Coincide? {abs(clean_sheets_away - prob_home_0) < 1e-10}")

# 4. Verificar team totals
print("\n4. VERIFICACIÓN DE TEAM TOTALS")
print("-" * 50)

print("HOME team_totals:")
home_totals_check = verify_team_totals(home_goals_distribution, team_totals, 'home')
for key, val in home_totals_check.items():
    status = "✓" if val['match'] else "✗"
    print(f"  {status} home_over_{key}: esperado={val['expected']:.6f}, actual={val['actual']:.6f}")

print()

print("AWAY team_totals:")
away_totals_check = verify_team_totals(away_goals_distribution, team_totals, 'away')
for key, val in away_totals_check.items():
    status = "✓" if val['match'] else "✗"
    print(f"  {status} away_over_{key}: esperado={val['expected']:.6f}, actual={val['actual']:.6f}")

# 5. Resumen final
print("\n" + "=" * 80)
print("RESUMEN FINAL DE AUDITORÍA")
print("=" * 80)

all_checks_passed = (
    abs(home_expected - expected_goals_home) < tolerance and
    abs(away_expected - expected_goals_away) < tolerance and
    abs(clean_sheets_home - prob_away_0) < 1e-10 and
    abs(clean_sheets_away - prob_home_0) < 1e-10 and
    all(v['match'] for v in home_totals_check.values()) and
    all(v['match'] for v in away_totals_check.values()) and
    not inversion_detected
)

if all_checks_passed:
    print("✓ TODAS LAS VERIFICACIONES PASARON")
    print("✓ El output es INTERNAMENTE COHERENTE")
    print("✓ No hay columnas invertidas")
    print("✓ expected_goals coincide con esperanzas de distribuciones")
    print("✓ clean_sheets coincide con probabilidades de 0 goles")
    print("✓ team_totals son consistentes con distribuciones")
else:
    print("✗ ALGUNAS VERIFICACIONES FALLARON - REVISAR CÓDIGO")

print("\n" + "=" * 80)
