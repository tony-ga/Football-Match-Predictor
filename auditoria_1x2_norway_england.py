#!/usr/bin/env python3
"""
Auditoría focalizada en probabilidades 1X2 - Norway vs England
====================================================================

Objetivo: Diagnosticar por qué el modelo da ~50% de probabilidad de victoria 
a Norway frente a Inglaterra cuando el mercado y modelos externos la ponen ~25-30%.

Preguntas clave:
1. ¿Es por ratings de equipos?
2. ¿Por localía mal modelada?
3. ¿Por algún bug en lambdas o en la derivación de 1X2?
4. ¿El modelo está sistemáticamente sobreestimando al equipo local?
"""
import sys
from pathlib import Path
import json

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent / 'predicciones'))

from src.models.dixon_coles import DixonColesModel, dc_score_matrix
from src.data.team_ratings_loader import TeamRatingsLoader
from src.models.market_derivation import derive_1x2
import numpy as np


def print_section(title: str):
    print("\n" + "=" * 80)
    print(f" {title}")
    print("=" * 80)


def test_norway_england_full_pipeline():
    """Test completo del pipeline para Norway vs England."""
    print_section("1. DIAGNÓSTICO COMPLETO: Norway vs England (2026-07-11)")
    
    # Cargar ratings
    loader = TeamRatingsLoader()
    
    print("\n📊 RATINGS EN EL SISTEMA:")
    print(f"  Norway: {loader.teams.get('Norway', 'NO ENCONTRADO - usa fallback')}")
    print(f"  Inglaterra: {loader.teams.get('Inglaterra', 'NO ENCONTRADO')}")
    print(f"  England: {loader.teams.get('England', 'NO ENCONTRADO')}")
    print(f"  Default fallback: {loader.default}")
    print(f"  Normalization factors: attack_mean={loader._attack_mean:.4f}, defense_mean={loader._defense_mean:.4f}")
    
    # Construir features
    print("\n🔧 CONSTRUCCIÓN DE FEATURES:")
    home_features, home_info = loader.build_team_features("Norway", is_home=True, verbose=True)
    away_features, away_info = loader.build_team_features("Inglaterra", is_home=False, verbose=True)
    
    print(f"\n  Home features (Norway):")
    print(f"    attack_rating: {home_features['attack_rating']:.4f}")
    print(f"    defense_rating: {home_features['defense_rating']:.4f}")
    print(f"    home_advantage_log: {home_features['home_advantage_log']}")
    print(f"    used_fallback: {home_info['used_default_fallback']}")
    
    print(f"\n  Away features (Inglaterra):")
    print(f"    attack_rating: {away_features['attack_rating']:.4f}")
    print(f"    defense_rating: {away_features['defense_rating']:.4f}")
    print(f"    home_advantage_log: {away_features['home_advantage_log']}")
    print(f"    used_fallback: {away_info['used_default_fallback']}")
    
    # Calcular lambdas
    print("\n🎯 CÁLCULO DE LAMBDAS:")
    model = DixonColesModel()
    lambda_h, lambda_a = model._predict_lambdas_heuristic(home_features, away_features)
    
    print(f"  lambda_home (Norway):  {lambda_h:.4f}")
    print(f"  lambda_away (England): {lambda_a:.4f}")
    print(f"  lambda_total:          {lambda_h + lambda_a:.4f}")
    
    # Mostrar fórmula explícita
    print(f"\n  Fórmula aplicada:")
    print(f"    lambda_home = attack_NOR * (1/defense_ENG) * LEAGUE_AVG_GOALS * exp(home_adv)")
    print(f"                = {home_features['attack_rating']:.4f} * (1/{away_features['defense_rating']:.4f}) * 1.35 * exp({home_features['home_advantage_log']})")
    
    defense_a_inverse = 1.0 / max(away_features['defense_rating'], 0.1)
    expected_lambda_h = home_features['attack_rating'] * defense_a_inverse * 1.35 * np.exp(home_features['home_advantage_log'])
    print(f"                = {expected_lambda_h:.4f}")
    
    print(f"\n    lambda_away = attack_ENG * (1/defense_NOR) * LEAGUE_AVG_GOALS")
    print(f"                = {away_features['attack_rating']:.4f} * (1/{home_features['defense_rating']:.4f}) * 1.35")
    
    defense_h_inverse = 1.0 / max(home_features['defense_rating'], 0.1)
    expected_lambda_a = away_features['attack_rating'] * defense_h_inverse * 1.35
    print(f"                = {expected_lambda_a:.4f}")
    
    # Generar matriz de scores y calcular 1X2
    print("\n📈 MATRIZ DE SCORES Y PROBABILIDADES 1X2:")
    matrix = dc_score_matrix(lambda_h, lambda_a, rho=model.rho, max_goals=8)
    probs_1x2 = derive_1x2(matrix)
    
    print(f"  P(Norway gana):  {probs_1x2['home']*100:.1f}%")
    print(f"  P(Empate):      {probs_1x2['draw']*100:.1f}%")
    print(f"  P(England gana): {probs_1x2['away']*100:.1f}%")
    print(f"  Suma:           {sum(probs_1x2.values()):.6f}")
    
    # Expected goals desde la matriz
    goals = np.arange(9)
    home_marginal = matrix.sum(axis=1)
    away_marginal = matrix.sum(axis=0)
    eg_home = float(np.dot(goals, home_marginal))
    eg_away = float(np.dot(goals, away_marginal))
    
    print(f"\n  Expected Goals (desde matriz):")
    print(f"    Norway:  {eg_home:.2f}")
    print(f"    England: {eg_away:.2f}")
    print(f"    Total:   {eg_home + eg_away:.2f}")
    
    return {
        'lambda_home': lambda_h,
        'lambda_away': lambda_a,
        'p_home': probs_1x2['home'],
        'p_draw': probs_1x2['draw'],
        'p_away': probs_1x2['away'],
        'eg_home': eg_home,
        'eg_away': eg_away,
        'home_features': home_features,
        'away_features': away_features,
        'home_info': home_info,
        'away_info': away_info,
    }


def test_neutral_venue_effect():
    """Testear efecto de neutral_venue en las probabilidades."""
    print_section("2. EFECTO DE LOCALÍA (neutral_venue)")
    
    loader = TeamRatingsLoader()
    model = DixonColesModel()
    
    home_features_ha, _ = loader.build_team_features("Norway", is_home=True, home_advantage_log=0.25)
    away_features_ha, _ = loader.build_team_features("Inglaterra", is_home=False, home_advantage_log=0.0)
    
    home_features_neutral, _ = loader.build_team_features("Norway", is_home=True, home_advantage_log=0.0)
    away_features_neutral, _ = loader.build_team_features("Inglaterra", is_home=False, home_advantage_log=0.0)
    
    # Con localía
    lambda_h_ha, lambda_a_ha = model._predict_lambdas_heuristic(home_features_ha, away_features_ha)
    matrix_ha = dc_score_matrix(lambda_h_ha, lambda_a_ha)
    probs_ha = derive_1x2(matrix_ha)
    
    # Neutral
    lambda_h_neu, lambda_a_neu = model._predict_lambdas_heuristic(home_features_neutral, away_features_neutral)
    matrix_neu = dc_score_matrix(lambda_h_neu, lambda_a_neu)
    probs_neu = derive_1x2(matrix_neu)
    
    print(f"\n  CON LOCALÍA (home_advantage_log=0.25):")
    print(f"    λ_home={lambda_h_ha:.3f}, λ_away={lambda_a_ha:.3f}")
    print(f"    P(Norway)={probs_ha['home']*100:.1f}%, Draw={probs_ha['draw']*100:.1f}%, P(England)={probs_ha['away']*100:.1f}%")
    
    print(f"\n  VENUE NEUTRO (home_advantage_log=0.0):")
    print(f"    λ_home={lambda_h_neu:.3f}, λ_away={lambda_a_neu:.3f}")
    print(f"    P(Norway)={probs_neu['home']*100:.1f}%, Draw={probs_neu['draw']*100:.1f}%, P(England)={probs_neu['away']*100:.1f}%")
    
    print(f"\n  📊 DIFERENCIA:")
    print(f"    ΔP(Norway) = {(probs_ha['home'] - probs_neu['home'])*100:+.1f}%")
    print(f"    ΔP(England) = {(probs_ha['away'] - probs_neu['away'])*100:+.1f}%")
    
    return probs_ha, probs_neu


def test_other_mismatches():
    """Testear otros partidos con favorito claro para identificar patrón sistemático."""
    print_section("3. OTROS PARTIDOS CON FAVORITO CLARO (patrón sistemático)")
    
    loader = TeamRatingsLoader()
    model = DixonColesModel()
    
    # Argentina vs Egypt (favorito claro: Argentina)
    matches = [
        ("Argentina", "Egypt", "Argentina debería ser favorito claro"),
        ("Francia", "Senegal", "Francia debería ser favorito"),
        ("Brasil", "Corea del Sur", "Brasil debería ser favorito"),
        ("Portugal", "Ghana", "Portugal debería ser favorito"),
    ]
    
    results = []
    for home, away, note in matches:
        home_feat, home_info = loader.build_team_features(home, is_home=True)
        away_feat, away_info = loader.build_team_features(away, is_home=False)
        
        lambda_h, lambda_a = model._predict_lambdas_heuristic(home_feat, away_feat)
        matrix = dc_score_matrix(lambda_h, lambda_a)
        probs = derive_1x2(matrix)
        
        print(f"\n  {home} vs {away}:")
        print(f"    Ratings: {home} attack={home_feat['attack_rating']:.2f}, defense={home_feat['defense_rating']:.2f}")
        print(f"             {away} attack={away_feat['attack_rating']:.2f}, defense={away_feat['defense_rating']:.2f}")
        print(f"    λ_home={lambda_h:.3f}, λ_away={lambda_a:.3f}, total={lambda_h+lambda_a:.2f}")
        print(f"    P({home})={probs['home']*100:.1f}%, Draw={probs['draw']*100:.1f}%, P({away})={probs['away']*100:.1f}%")
        print(f"    Nota: {note}")
        
        results.append({
            'match': f"{home} vs {away}",
            'lambda_home': lambda_h,
            'lambda_away': lambda_a,
            'p_home': probs['home'],
            'p_away': probs['away'],
            'favorite_correct': probs['home'] > probs['away'] if 'Argentina' in home or 'Francia' in home or 'Brasil' in home or 'Portugal' in home else True,
        })
    
    return results


def analyze_root_cause(result_nor_eng):
    """Analizar causa raíz del problema."""
    print_section("4. ANÁLISIS DE CAUSA RAÍZ")
    
    print("\n🔍 FACTORES IDENTIFICADOS:\n")
    
    # 1. Ratings de Norway
    home_info = result_nor_eng['home_info']
    away_info = result_nor_eng['away_info']
    
    print(f"  1. FALLBACK RATING PARA NORWAY:")
    print(f"     - Norway NO está en ratings_wc2026.json")
    print(f"     - Usa default: attack=1.1, defense=1.0")
    print(f"     - Después de normalización: attack={result_nor_eng['home_features']['attack_rating']:.3f}, defense={result_nor_eng['home_features']['defense_rating']:.3f}")
    
    print(f"\n  2. INGALATERRA SÍ TIENE RATING:")
    print(f"     - Attack raw: {away_info['away_attack_rating_raw']:.3f}")
    print(f"     - Defense raw: {away_info['away_defense_rating_raw']:.3f}")
    print(f"     - Después de normalización: attack={result_nor_eng['away_features']['attack_rating']:.3f}, defense={result_nor_eng['away_features']['defense_rating']:.3f}")
    
    print(f"\n  3. FÓRMULA DE LAMBDA:")
    print(f"     - lambda_home = attack_home * (1/defense_away) * LEAGUE_AVG_GOALS * exp(home_adv)")
    print(f"     - lambda_away = attack_away * (1/defense_home) * LEAGUE_AVG_GOALS")
    print(f"     ")
    print(f"     ⚠️  PROBLEMA POTENCIAL:")
    print(f"         La defensa alta de Inglaterra ({result_nor_eng['away_features']['defense_rating']:.3f})")
    print(f"         DEBERÍA reducir el lambda de Norway, pero...")
    print(f"         ")
    print(f"         lambda_home = {result_nor_eng['home_features']['attack_rating']:.3f} * (1/{result_nor_eng['away_features']['defense_rating']:.3f}) * 1.35 * exp(0.25)")
    inv_def = 1.0 / result_nor_eng['away_features']['defense_rating']
    print(f"                   = {result_nor_eng['home_features']['attack_rating']:.3f} * {inv_def:.3f} * 1.35 * 1.284")
    print(f"                   = {result_nor_eng['lambda_home']:.3f}")
    
    print(f"\n  4. HOME ADVANTAGE:")
    print(f"     - home_advantage_log = 0.25 (≈ 28.4% boost en lambda)")
    print(f"     - Esto infla artificialmente a Norway como local")
    
    print(f"\n  5. RESULTADO FINAL:")
    print(f"     - λ_Norway = {result_nor_eng['lambda_home']:.3f}")
    print(f"     - λ_England = {result_nor_eng['lambda_away']:.3f}")
    print(f"     - P(Norway gana) = {result_nor_eng['p_home']*100:.1f}%")
    print(f"     - P(England gana) = {result_nor_eng['p_away']*100:.1f}%")
    
    print(f"\n  ✅ CONCLUSIÓN:")
    if result_nor_eng['p_home'] > result_nor_eng['p_away']:
        print(f"     ⚠️  BUG CONFIRMADO: Norway es favorito pese a tener PEOR rating")
        print(f"         Causas combinadas:")
        print(f"         (a) Norway usa fallback rating que la hace parecer competitiva")
        print(f"         (b) Home advantage (0.25) infla demasiado a Norway")
        print(f"         (c) La diferencia real de calidad no se refleja en los lambdas")
    else:
        print(f"     ✓ England es correctamente favorita")
        print(f"         Pero revisar si la probabilidad es razonable vs mercado (~50-70%)")


def propose_fixes(result_nor_eng):
    """Proponer ajustes concretos."""
    print_section("5. PROPUESTA DE AJUSTES")
    
    print("\n🔧 AJUSTES RECOMENDADOS:\n")
    
    print("  OPCIÓN A: Agregar Norway a ratings_wc2026.json con rating realista")
    print("  -----------------------------------------------------------------------")
    print("  Norway FIFA rank ~40-50, ataque decente pero no élite.")
    print("  Rating sugerido: attack=0.95, defense=0.90, fifa_rank=45")
    print()
    
    # Simular con Norway teniendo rating realista
    loader = TeamRatingsLoader()
    model = DixonColesModel()
    
    # Crear features manuales para Norway con rating realista
    norway_realistic = {
        'nombre': 'Norway',
        'attack_rating': 0.95 / loader._attack_mean,  # normalizado
        'defense_rating': 0.90 / loader._defense_mean,
        'form_factor': 1.0,
        'ranking_factor': 1.0,
        'h2h_factor': 1.0,
        'squad_multiplier': 1.0,
        'home_advantage_log': 0.0,  # neutral venue
        'context_modifier': 0.0,
    }
    
    england_feat, _ = loader.build_team_features("Inglaterra", is_home=False)
    
    lambda_h_fix, lambda_a_fix = model._predict_lambdas_heuristic(norway_realistic, england_feat)
    matrix_fix = dc_score_matrix(lambda_h_fix, lambda_a_fix)
    probs_fix = derive_1x2(matrix_fix)
    
    print(f"  SIMULACIÓN CON NORWAY RATING REALISTA (venue neutral):")
    print(f"    λ_Norway = {lambda_h_fix:.3f}, λ_England = {lambda_a_fix:.3f}")
    print(f"    P(Norway) = {probs_fix['home']*100:.1f}%, Draw = {probs_fix['draw']*100:.1f}%, P(England) = {probs_fix['away']*100:.1f}%")
    print()
    
    print("  OPCIÓN B: Reducir home_advantage para partidos neutral_venue")
    print("  -----------------------------------------------------------------------")
    print("  El fixture indica neutral_venue para este partido.")
    print("  Asegurar que home_advantage_log = 0.0 cuando neutral_venue = True")
    print()
    
    print("  OPCIÓN C: Revisar fórmula de lambda para asegurar que defensa alta")
    print("            del oponente REDUZCA el lambda propio correctamente")
    print("  -----------------------------------------------------------------------")
    print("  Fórmula actual: lambda = attack * (1/defense_opponent) * LEAGUE_AVG")
    print("  Esto es CORRECTO - defensa alta reduce lambda.")
    print("  Pero verificar que los valores de defense_rating tengan semántica clara:")
    print("    - defense_rating > 1.0 = MEJOR defensa que promedio (concede MENOS)")
    print("    - defense_rating < 1.0 = PEOR defensa que promedio (concede MÁS)")
    print()
    
    print("  OPCIÓN D: Calibración específica para 1X2")
    print("  -----------------------------------------------------------------------")
    print("  Implementar CalibrationManager para 1X2 usando histórico de resultados.")
    print("  Esto ajustaría las probabilidades crudas del modelo Poisson/DC para")
    print("  alinearlas mejor con frecuencias observadas.")
    print()
    
    print("\n📋 PLAN DE ACCIÓN RECOMENDADO:")
    print("  1. Agregar Norway (y otros equipos faltantes) a ratings_wc2026.json")
    print("  2. Verificar que neutral_venue=true → home_advantage_log=0.0")
    print("  3. Documentar claramente semántica de defense_rating")
    print("  4. Implementar calibración 1X2 basada en histórico")
    print("  5. Validar con backtest que favoritos claros tengan P(win) acorde")


def main():
    print("\n" + "█" * 80)
    print("█ AUDITORÍA DE PROBABILIDADES 1X2 - Norway vs England")
    print("█ Fecha: 2026-07-11 | FIFA World Cup")
    print("█" * 80)
    
    # 1. Diagnóstico completo
    result_nor_eng = test_norway_england_full_pipeline()
    
    # 2. Efecto de localía
    test_neutral_venue_effect()
    
    # 3. Patrón sistemático
    test_other_mismatches()
    
    # 4. Análisis de causa raíz
    analyze_root_cause(result_nor_eng)
    
    # 5. Propuesta de ajustes
    propose_fixes(result_nor_eng)
    
    print("\n" + "█" * 80)
    print("█ FIN DE AUDITORÍA")
    print("█" * 80 + "\n")


if __name__ == '__main__':
    main()
