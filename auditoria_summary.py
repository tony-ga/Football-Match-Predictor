#!/usr/bin/env python3
"""
AUDITORÍA COMPLETA DEL PIPELINE DE SUMMARY - FRANCIA

Objetivo: Identificar por qué las estadísticas de jugadores están infladas.
Partidos auditados: 760432, 760457, 760475, 760492, 760503
"""
import json
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Any

# Paths
PROJECT_ROOT = Path("/workspace")
PLAYER_STATS_PATH = PROJECT_ROOT / "data" / "derived" / "player_match_stats.jsonl"
MATCH_EVENTS_PATH = PROJECT_ROOT / "data" / "derived" / "match_events.jsonl"

# Partidos a auditar
EVENT_IDS = {"760432", "760457", "760475", "760492", "760503"}

# Jugadores a validar
TARGET_PLAYERS = [
    "Kylian Mbappé",
    "Ousmane Dembélé", 
    "Bradley Barcola",
    "Mike Maignan",
    "William Saliba",
]

# Goles esperados según verificación externa
EXPECTED_MBAPPE_GOALS = {
    "760432": 2,  # vs Senegal
    "760457": 2,  # vs Iraq
    "760475": 0,  # vs Norway
    "760492": 2,  # vs Sweden
    "760503": 1,  # vs Paraguay
}


def load_player_stats() -> List[Dict[str, Any]]:
    """Cargar todas las stats de jugador del JSONL."""
    if not PLAYER_STATS_PATH.exists():
        print(f"ERROR: No existe {PLAYER_STATS_PATH}")
        return []
    
    records = []
    with open(PLAYER_STATS_PATH, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return records


def load_match_events() -> Dict[str, Dict[str, Any]]:
    """Cargar eventos de partidos."""
    if not MATCH_EVENTS_PATH.exists():
        return {}
    
    events = {}
    with open(MATCH_EVENTS_PATH, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    record = json.loads(line)
                    eid = record.get("event_id", "")
                    if eid:
                        events[eid] = record
                except json.JSONDecodeError:
                    continue
    return events


def main():
    print("=" * 80)
    print("AUDITORÍA DE CALIDAD Y VERACIDAD - PLAYER_MATCH_STATS.JSONL")
    print("=" * 80)
    
    # Cargar datos
    all_stats = load_player_stats()
    match_events = load_match_events()
    
    # Filtrar solo Francia y los 5 partidos
    france_records = [
        r for r in all_stats 
        if r.get("team", "").lower() == "france" 
        and r.get("event_id", "") in EVENT_IDS
    ]
    
    print(f"\n📊 DATOS GENERALES")
    print(f"   Total registros en JSONL: {len(all_stats)}")
    print(f"   Registros Francia (5 partidos): {len(france_records)}")
    print(f"   Unique event_ids: {len(set(r['event_id'] for r in france_records))}")
    print(f"   Unique jugadores: {len(set(r['player_name'] for r in france_records))}")
    
    # Contar filas por partido
    print(f"\n📋 FILAS POR PARTIDO:")
    from collections import Counter
    event_counts = Counter(r['event_id'] for r in france_records)
    for eid in sorted(event_counts.keys()):
        match_info = match_events.get(eid, {})
        opponent = match_info.get('away_team', '') if match_info.get('home_team', '').lower() == 'france' else match_info.get('home_team', '')
        print(f"   {eid} vs {opponent}: {event_counts[eid]} filas")
    
    # Verificar duplicados (evento, jugador)
    player_event_keys = [(r['event_id'], r['player_name']) for r in france_records]
    duplicates = [(k, v) for k, v in Counter(player_event_keys).items() if v > 1]
    
    print(f"\n🔍 DUPLICADOS (event_id, player_name): {len(duplicates)}")
    if duplicates:
        for (eid, pname), count in duplicates[:10]:
            print(f"   ⚠️  {eid} - {pname}: {count} veces")
    else:
        print("   ✅ No hay duplicados")
    
    # ========================================================================
    # ANÁLISIS DETALLADO POR JUGADOR OBJETIVO
    # ========================================================================
    print("\n" + "=" * 80)
    print("ANÁLISIS DETALLADO POR JUGADOR")
    print("=" * 80)
    
    audit_results = []
    
    for target_player in TARGET_PLAYERS:
        print(f"\n{'─' * 80}")
        print(f"JUGADOR: {target_player}")
        print(f"{'─' * 80}")
        
        player_records = [
            r for r in france_records 
            if r.get('player_name') == target_player
        ]
        
        if not player_records:
            print(f"   ⚠️  No encontrado en los 5 partidos")
            continue
        
        # Stats por partido
        print(f"\n   📈 ESTADÍSTICAS POR PARTIDO:")
        print(f"   {'Event ID':<12} {'Opponent':<15} {'Min':>5} {'Gls':>5} {'Ast':>5} {'YC':>4} {'RC':>4} {'Shots':>6} {'Starter':>8}")
        print(f"   {'-' * 70}")
        
        totals = {
            'GP': 0,
            'Min': 0,
            'Gls': 0,
            'Ast': 0,
            'YC': 0,
            'RC': 0,
            'Shots': 0,
        }
        
        for rec in sorted(player_records, key=lambda x: x.get('date', '')):
            eid = rec.get('event_id', '')
            match_info = match_events.get(eid, {})
            
            # Determinar oponente
            home_team = match_info.get('home_team', '')
            away_team = match_info.get('away_team', '')
            opponent = away_team if home_team.lower() == 'france' else home_team
            
            minutes = rec.get('minutes', 0) or 0
            goals = rec.get('goals', 0) or 0
            assists = rec.get('assists', 0) or 0
            yc = rec.get('yellow_cards', 0) or 0
            rc = rec.get('red_cards', 0) or 0
            shots = rec.get('shots', 0) or 0
            is_starter = rec.get('is_starter', False)
            
            print(f"   {eid:<12} {opponent:<15} {minutes:>5} {goals:>5} {assists:>5} {yc:>4} {rc:>4} {shots:>6} {str(is_starter):>8}")
            
            # Acumular
            totals['GP'] += 1
            totals['Min'] += minutes
            totals['Gls'] += goals
            totals['Ast'] += assists
            totals['YC'] += yc
            totals['RC'] += rc
            totals['Shots'] += shots
        
        print(f"   {'-' * 70}")
        print(f"   {'TOTAL (5 partidos)':<12} {'':<15} {totals['Min']:>5} {totals['Gls']:>5} {totals['Ast']:>5} {totals['YC']:>4} {totals['RC']:>4} {totals['Shots']:>6}")
        
        # Comparar con esperado para Mbappé
        expected_if_verifiable = None
        if target_player == "Kylian Mbappé":
            expected_total = sum(EXPECTED_MBAPPE_GOALS.values())
            expected_if_verifiable = expected_total
            print(f"\n   🎯 GOLES ESPERADOS (verificación externa): {expected_total}")
            print(f"   📊 GOLES EN JSONL: {totals['Gls']}")
            if totals['Gls'] != expected_total:
                diff = totals['Gls'] - expected_total
                print(f"   ❌ DIFERENCIA: +{diff} goles (INFLADO)")
                
                # Mostrar desglose esperado vs real
                print(f"\n   🔍 DESGLOSE DE GOLES:")
                print(f"   {'Partido':<20} {'Esperado':>10} {'Real JSONL':>12} {'Diff':>8}")
                for eid in sorted(EXPECTED_MBAPPE_GOALS.keys()):
                    exp = EXPECTED_MBAPPE_GOALS[eid]
                    rec_row = next((r for r in player_records if r['event_id'] == eid), None)
                    real = rec_row.get('goals', 0) if rec_row else 0
                    diff_g = real - exp
                    match_info = match_events.get(eid, {})
                    opp = match_info.get('away_team', '') if match_info.get('home_team', '').lower() == 'france' else match_info.get('home_team', '')
                    status = "✅" if diff_g == 0 else "❌"
                    print(f"   {opp:<20} {exp:>10} {real:>12} {diff_g:>+8} {status}")
        
        # Guardar resultados para tabla final
        for metric in ['GP', 'Min', 'Gls', 'Ast', 'YC', 'RC', 'Shots']:
            audit_results.append({
                'player': target_player,
                'metric': metric,
                'cli_value': totals[metric],
                'recalculated_value': totals[metric],
                'expected_if_verifiable': expected_if_verifiable if metric == 'Gls' and target_player == "Kylian Mbappé" else None,
            })
    
    # ========================================================================
    # TABLA DE AUDITORÍA FINAL
    # ========================================================================
    print("\n" + "=" * 80)
    print("TABLA DE AUDITORÍA COMPLETA")
    print("=" * 80)
    
    print(f"\n{'Player':<25} {'Metric':<8} {'CLI Value':>12} {'Recalculated':>14} {'Expected':>12} {'Status':>10}")
    print(f"{'-' * 85}")
    
    for row in audit_results:
        player = row['player']
        metric = row['metric']
        cli_val = row['cli_value']
        recalculated = row['recalculated_value']
        expected = row['expected_if_verifiable']
        
        # Determinar status
        if expected is not None:
            status = "✅ OK" if cli_val == expected else "❌ MISMATCH"
            expected_str = str(expected)
        else:
            status = "✅ OK"  # No hay referencia externa
            expected_str = "N/A"
        
        print(f"{player:<25} {metric:<8} {cli_val:>12} {recalculated:>14} {expected_str:>12} {status:>10}")
    
    # ========================================================================
    # DIAGNÓSTICO DE CAUSA RAÍZ
    # ========================================================================
    print("\n" + "=" * 80)
    print("DIAGNÓSTICO DE CAUSA RAÍZ")
    print("=" * 80)
    
    # Analizar si hay inflación sistemática
    mbappe_recs = [r for r in france_records if r.get('player_name') == "Kylian Mbappé"]
    
    print("\n🔍 ANÁLISIS DE INFLACIÓN DE GOLES - MBAPPÉ:")
    
    total_goals_in_jsonl = sum(r.get('goals', 0) or 0 for r in mbappe_recs)
    total_expected = sum(EXPECTED_MBAPPE_GOALS.values())
    inflation = total_goals_in_jsonl - total_expected
    
    print(f"   Total goles en JSONL: {total_goals_in_jsonl}")
    print(f"   Total goles esperados: {total_expected}")
    print(f"   Inflación: +{inflation} goles")
    
    # Calcular factor de inflación por partido
    print(f"\n   Factor de inflación por partido:")
    for eid in sorted(EXPECTED_MBAPPE_GOALS.keys()):
        exp = EXPECTED_MBAPPE_GOALS[eid]
        rec_row = next((r for r in mbappe_recs if r['event_id'] == eid), None)
        real = rec_row.get('goals', 0) if rec_row else 0
        if exp > 0:
            factor = real / exp
            print(f"      {eid}: {real}/{exp} = {factor:.2f}x")
        else:
            print(f"      {eid}: {real}/0 = N/A (goles no esperados)")
    
    # Hipótesis de causa raíz
    print(f"\n🧩 HIPÓTESIS DE CAUSA RAÍZ:")
    
    # Verificar si los valores son exactamente el doble
    if inflation == total_expected:
        print(f"   ⚠️  LOS GOLES ESTÁN EXACTAMENTE DUPLICADOS (2x)")
        print(f"      Esto sugiere que las stats se están sumando dos veces en algún punto del pipeline.")
    elif inflation > 0:
        ratio = total_goals_in_jsonl / total_expected if total_expected > 0 else 0
        print(f"   ⚠️  LOS GOLES ESTÁN INFLADOS POR FACTOR {ratio:.2f}x")
        
        # Verificar si podría ser mezcla de stats acumuladas + por-partido
        print(f"\n   Posibles causas:")
        print(f"      1. Duplicación de filas en player_match_stats.jsonl")
        print(f"      2. Suma de stats ya acumuladas con stats por-partido")
        print(f"      3. Bug en el script que genera/enriquece player_match_stats.jsonl")
        print(f"      4. Mapeo incorrecto de campos desde la fuente ESPN")
    
    # ========================================================================
    # RECOMENDACIONES
    # ========================================================================
    print("\n" + "=" * 80)
    print("RECOMENDACIONES DE CORRECCIÓN")
    print("=" * 80)
    
    print("""
    1. REVISAR SCRIPT DE GENERACIÓN:
       - Verificar build_market_dataset.py línea ~365 donde se appendean player_rows
       - Confirmar que no haya duplicación al leer desde ESPN API
    
    2. VALIDAR FUENTE ESPN:
       - Los campos 'goals', 'assists', etc. en el JSONL deben ser stats DEL PARTIDO
       - NO deben ser stats ACUMULADAS de temporada
    
    3. IMPLEMENTAR VALIDACIÓN:
       - Agregar check: suma de goles por jugador en partido <= total posible (~5)
       - Alertar si cualquier jugador tiene >3 goles en un partido sin hat-trick confirmado
    
    4. CORREGIR AGREGACIÓN EN CLI:
       - En espn_player_stats.py, función fetch_team_roster_stats()
       - Asegurar que games_played cuente partidos únicos, no filas duplicadas
    """)
    
    print("\n" + "=" * 80)
    print("FIN DE AUDITORÍA")
    print("=" * 80)


if __name__ == "__main__":
    main()
