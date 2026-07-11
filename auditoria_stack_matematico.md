# AUDITORÍA GENERAL DEL STACK MATEMÁTICO
## Poisson / Markov / Same Game Parlay Builder

**Fecha:** 2026-01-XX  
**Alcance:** Modelos Poisson/Dixon-Coles, módulos Markov, SGP Builder

---

## 1. POISSON / SCORE MATRIX / MERCADOS

### 1.1 Cálculo de λ_home y λ_away

**Archivos inspeccionados:**
- `/workspace/predicciones/src/models/dixon_coles.py` (753 líneas)
- `/workspace/predicciones/src/models/poisson.py` (119 líneas)
- `/workspace/predicciones/src/features/ratings.py`

**Fuentes de λ:**

El sistema opera en **dos modos**:

#### Modo 1: Dixon-Coles Entrenado (MLE)
```python
# dixon_coles.py líneas 430-467
lambda_h = alpha_h * beta_a * gamma
lambda_a = alpha_a * beta_h

# Donde:
# - alpha_h: parámetro de ataque del equipo local (entrenado por MLE)
# - beta_a: parámetro de defensa del visitante (entrenado por MLE)
# - gamma: ventaja de localía (default 0.25, entrenable)
# - lambda_h *= exp(ctx_h)  # context_modifier desde features
```

**Fórmula completa:**
```
λ_home = α_home × β_away × γ × exp(contexto_home) × ajuste_markov(opcional)
λ_away = α_away × β_home × exp(contexto_away) × ajuste_markov(opcional)
```

#### Modo 2: Heurístico (sin datos históricos)
```python
# dixon_coles.py líneas 469-570
lambda_h = attack_h × (1/defense_a) × LEAGUE_AVG_GOALS × 
           form_h × ranking_h × h2h_h × squad_h × exp(home_adv + ctx_h)

lambda_a = attack_a × (1/defense_h) × LEAGUE_AVG_GOALS ×
           form_a × ranking_a × h2h_a × squad_a × exp(ctx_a)
```

**Supuestos principales:**
1. **Independencia condicional**: Dados λ_home y λ_away, los goles son independientes
2. **Defensa inversa**: `defense_rating > 1.0` significa MEJOR defensa → se INVIerte para calcular λ del oponente
3. **LEAGUE_AVG_GOALS**: Constante global (típicamente ~2.5-2.7)
4. **Context modifiers**: Ajustes aditivos en el espacio logarítmico

**Puntos débiles detectados:**
- ⚠️ No hay validación explícita de que `attack_rating` y `defense_rating` estén en la misma escala
- ⚠️ `context_modifier` puede inflar lambdas sin límites superiores estrictos (se clippea después a max_lambda=4.0)
- ⚠️ No hay calibración de lambdas basada en xG real vs goles observados

---

### 1.2 Construcción de la Matriz de Scores

**Archivo:** `/workspace/predicciones/src/models/dixon_coles.py`

**Rango de goles:** 0–8 (max_goals=8 por defecto, configurable)

**Fórmula Poisson simple** (`poisson.py` líneas 20-65):
```python
P(home=k) = poisson.pmf(k, lambda_home) = (λ^k * e^-λ) / k!
P(away=k) = poisson.pmf(k, lambda_away)
matrix[i,j] = P(home=i) × P(away=j)  # Independencia
```

**Fórmula Dixon-Coles** (`dixon_coles.py` líneas 73-113):
```python
matrix[i,j] = poisson.pmf(i, λ_h) × poisson.pmf(j, λ_a) × τ(i, j, ρ)

donde τ (tau) es la corrección Dixon-Coles:
- τ(0,0) = 1 - λ_h × λ_a × ρ
- τ(1,0) = 1 + λ_a × ρ
- τ(0,1) = 1 + λ_h × ρ
- τ(1,1) = 1 - ρ
- τ(i,j) = 1  para i+j > 2

ρ (rho) = -0.13 por defecto (negativo, captura correlación negativa en scores bajos)
```

**Normalización:**
```python
total = matrix.sum()
if total > 0:
    matrix /= total  # Renormaliza para compensar truncamiento
```

**✅ CORRECTO:** La implementación coincide con la teoría estándar de Dixon-Coles (1997).

**Puntos débiles:**
- ⚠️ `max_goals=8` puede ser insuficiente para partidos con λ_total > 5.0 (pérdida de ~1-2% de probabilidad)
- ⚠️ No hay warning si la masa de probabilidad truncada excede umbral (ej. 0.5%)

---

### 1.3 Derivación de Mercados desde la Matriz

**Archivo:** `/workspace/predicciones/src/models/market_derivation.py` (632 líneas)

#### 1X2 (líneas 33-58)
```python
P(home_win) = Σ matrix[i,j] donde i > j  # np.tril(matrix, k=-1)
P(draw)     = Σ matrix[i,i]              # np.diag(matrix)
P(away_win) = Σ matrix[i,j] donde i < j  # np.triu(matrix, k=1)
```
**✅ CORRECTO** - Coincide con teoría estándar.

#### Double Chance (líneas 61-76)
```python
P(home_or_draw) = P(home) + P(draw)
P(away_or_draw) = P(away) + P(draw)
P(home_or_away) = P(home) + P(away)
```
**✅ CORRECTO** - Suma directa de probabilidades mutuamente exclusivas.

#### BTTS (líneas 79-105)
```python
P(btts_yes) = 1 - P(home=0) - P(away=0) + P(0,0)
            = 1 - matrix[0,:].sum() - matrix[:,0].sum() + matrix[0,0]

# Usa inclusión-exclusión correctamente:
# P(home=0 OR away=0) = P(home=0) + P(away=0) - P(0,0)
# P(btts_yes) = 1 - P(home=0 OR away=0)
```
**✅ CORRECTO** - Implementación matemáticamente precisa.

#### Over/Under (líneas 108-153)
```python
P(over_N.5) = Σ matrix[i,j] donde i+j > N.5
P(under_N.5) = 1 - P(over_N.5)

# Líneas soportadas: 1.5, 2.5, 3.5, 4.5
```
**✅ CORRECTO** - Agregación exacta sobre la matriz.

#### Correct Scores (líneas 202-241)
```python
# Retorna top N scores más probables + categoría 'other'
scores = [(i, j, matrix[i,j]) for i in range(max_goals+1) for j in range(max_goals+1)]
scores.sort(key=lambda x: x['probability'], reverse=True)
other_prob = 1.0 - sum(top_scores)
```
**✅ CORRECTO** - Sin simplificaciones.

#### Clean Sheets (líneas 156-174)
```python
P(CS_home) = P(away=0) = matrix[:,0].sum()
P(CS_away) = P(home=0) = matrix[0,:].sum()
```
**✅ CORRECTO.**

#### Halftime 1X2 (líneas 278-347)
```python
λ_ht_home = λ_home × 0.45  # fracción configurable
λ_ht_away = λ_away × 0.45

# Poisson simple (sin corrección DC)
ht_matrix = outer(poisson(λ_ht_h), poisson(λ_ht_a))

# Ajuste empírico: si λ_total_HT < 1.5, boost draw +4%
```
**⚠️ DISCUTIBLE:**
- No usa corrección Dixon-Coles para HT (asumible pero no documentado)
- Boost de draw empírico (+4%) no está calibrado con datos reales

---

### 1.4 Calibración de Probabilidades

**Archivo:** `/workspace/predicciones/src/models/calibration.py` (192 líneas)

**Métodos disponibles:**
1. **Isotonic Regression** (default)
2. **Platt Scaling** (LogisticRegression)

**Pipeline documentado:**
```
Score Matrix → Market Derivation → Calibration → Sanity
```

**Implementación:**
```python
# calibration.py líneas 161-191
def calibrate_markets(self, raw_markets):
    calibrated = raw_markets.copy()
    
    if '1x2' in self.calibrators:
        calibrated['1x2'] = calibrator.calibrate(raw_markets['1x2'])
    
    if 'btts' in self.calibrators:
        calibrated['btts'] = calibrator.calibrate(raw_markets['btts'])
    
    # Over/Under por línea individual
    for line in ['15', '25', '35']:
        if f'over_under_{line}' in self.calibrators:
            probs = {'over': ..., 'under': ...}
            cal_probs = calibrator.calibrate(probs)
```

**🔴 PROBLEMA CRÍTICO DETECTADO:**

En `parlay_builder.py`:
```python
# Líneas 292-337
for derived in derived_markets:
    prob = derived["probability"]  # ← USA PROBABILIDAD CRUDA
    ...
    candidates.append(PickCandidate(
        model_probability=prob,  # ← NO SE APLICA CALIBRACIÓN
        ...
    ))
```

**No hay llamada a `calibration.calibrate_markets()` en el pipeline del SGP Builder.**

**Verificación en `generate_same_game_candidates`:**
- Las probabilidades vienen directamente de `derive_goal_markets(pred_data)`
- Esas probabilidades son las **crudas** de la matriz Poisson/DC
- `calib_status` solo verifica existencia de archivos .pkl, pero **no aplica calibración**

**Impacto:**
- El SGP Builder usa probabilidades **no calibradas**
- EV y edge se calculan sobre probabilidades crudas
- Esto puede generar sesgos sistemáticos si el modelo está mal calibrado

---

## 2. MARKOV / POSESIONES / SECUENCIAS

### 2.1 Identificación del Módulo

**Archivos inspeccionados:**
- `/workspace/predicciones/src/features/markov_features.py` (480 líneas)
- `/workspace/predicciones/scripts/build_markov_transition_matrix.py`
- `/workspace/predicciones/data/markov/` (tablas precomputadas)

**Estados definidos:**
```python
state_t = {
    "minute_bucket": "0-15"|"16-30"|"31-45+"|"46-60"|"61-75"|"76-90+",
    "score_diff_bucket": "+2_or_more"|"+1"|"0"|"-1"|"-2_or_more",
    "home_red_cards": "0"|"1"|"2",
    "away_red_cards": "0"|"1"|"2",
    "phase": "regular_time",
    "strength_gap_bucket": "unknown",
    "venue_context": "neutral"
}
```

**Matriz de transición P:**
- **No es una matriz Markoviana clásica** (estado → estado)
- En su lugar, son **probabilidades de evento condicionales al estado**:
  - `p_goal_next_window`
  - `p_concede_next_window`
  - `p_corner_next_window_ge1`
  - `p_shot_next_window_ge1`
  - `e_shots_next_window`
  - `e_corners_next_window`

**Cálculo del valor:**
```python
# markov_features.py líneas 203-340
features = get_markov_features(state_t, event_probs_df, baselines)

# Lookup en tablas precomputadas (CSV)
# No hay cálculo online de cadena de Markov
```

---

### 2.2 Conexión con el Modelo de Goles

**Archivo:** `/workspace/predicciones/src/models/dixon_coles.py`

**Integración Markov → λ:**
```python
# dixon_coles.py líneas 605-693
def _apply_markov_adjustment(self, lambda_h, lambda_a, match_state):
    # 1. Build state from match context
    home_state = build_state_from_match_context(minute, score_diff, ...)
    
    # 2. Get Markov features
    home_markov = get_markov_features(home_state, event_probs, baselines)
    
    # 3. Compute ratio vs baseline
    baseline_p_goal = 0.17  # global average
    home_ratio = home_markov['markov_p_goal_next_window'] / baseline_p_goal
    
    # 4. Apply soft adjustment
    weight = self._get_markov_weight(match_state)  # default 0.18
    home_adjustment = 1.0 + weight * (home_ratio - 1.0)
    
    lambda_h_adj = lambda_h * home_adjustment
    return lambda_h_adj, lambda_a_adj
```

**Fórmula de ajuste:**
```
λ_adjusted = λ_base × [1 + w × (ratio - 1)]

donde:
- ratio = p_goal_state / p_goal_baseline
- w = 0.18 (configurable, puede tener schedule por minuto)
```

**✅ CONEXIÓN EXISTE:** Markov **sí influye** en λ_home y λ_away cuando:
1. `use_markov_features = True`
2. `match_state` es proporcionado
3. Las tablas Markov están cargadas

**Modo de operación:**
- **Pre-match:** Markov no se usa (no hay estado)
- **In-play:** Markov ajusta λ según estado del partido

---

### 2.3 Consistencia Interna

**Supuestos Markov:**
1. **Propiedad de Markov:** El futuro depende solo del estado actual (minuto, score_diff, cards)
2. **Estacionariedad:** Las transiciones no cambian durante el partido (asumido en tablas precomputadas)
3. **Independencia de trayectoria:** No importa CÓMO se llegó al estado, solo el estado mismo

**Problemas detectados:**
- ⚠️ **No hay verificación** de que las tablas Markov estén actualizadas con datos recientes
- ⚠️ **Sample sizes bajos:** Algunos estados pueden tener < 20 observaciones (hay smoothing pero es heurístico)
- ⚠️ **Peso fijo 0.18:** No está optimizado; podría sobre/ajustar dependiendo del contexto

**Conexión con SGP:**
- **INDIRECTA:** Markov → λ → matriz → mercados → SGP
- **No hay conexión directa** entre features Markov y construcción de parlays
- El SGP Builder no sabe si las probabilidades incluyen ajuste Markov o no

---

## 3. SAME GAME PARLAY BUILDER (SGP)

### 3.1 Consumo de Probabilidades Poisson

**Archivos inspeccionados:**
- `/workspace/predicciones/src/models/parlay_builder.py` (690 líneas)
- `/workspace/predicciones/src/models/market_derivation.py`
- `/workspace/predicciones/src/models/market_catalog.py`
- `/workspace/predicciones/src/models/ticket_structure.py`

**Flujo de probabilidades:**
```python
# parlay_builder.py líneas 256-340
def generate_same_game_candidates(pred_data, ...):
    derived_markets = derive_goal_markets(pred_data)  # ← Probadilidades CRUDAS
    
    for derived in derived_markets:
        prob = derived["probability"]  # ← SIN CALIBRAR
        
        evaluation = evaluate_core_market_set(...)  # ← Evalúa con prob cruda
        
        candidates.append(PickCandidate(
            model_probability=prob,  # ← Se usa esta prob cruda
            ...
        ))
```

**🔴 INCONSISTENCIA:**
- `pred_data['predictions']` contiene probabilidades **derivadas de la matriz** (crudas)
- No hay paso intermedio de `CalibrationManager.calibrate_markets()`
- `calib_status` solo verifica existencia de archivos, **no aplica calibración**

**Derivación de MarketDefinition:**
```python
# market_derivation.py líneas 427-467
def derive_goal_markets(pred_data):
    markets = pred_data.get('predictions', {})
    
    # 1X2
    candidates.append({"market_key": "1x2_home", "probability": markets['1x2']['home']})
    
    # Totals
    candidates.append({"market_key": "over_2_5", "probability": markets['over_under']['over_2_5']})
    
    # BTTS
    candidates.append({"market_key": "btts_yes", "probability": markets['btts']['yes']})
```

**✅ CORRECTO:** Las probabilidades vienen directamente de la matriz vía `market_derivation`.

---

### 3.2 Manejo de Correlación en Parlays

**Archivo:** `/workspace/predicciones/src/models/ticket_structure.py` (439 líneas)

**Tipos de relaciones:**
```python
class MarketRelationType(Enum):
    CONTRADICTORY = "contradictory"       # Ej: 1x2_home + 1x2_away
    NESTED_REDUNDANT = "nested_redundant" # Ej: over_1_5 + over_2_5
    WEAK_LOW_INFORMATION = "weak_low_information"  # Ej: under_4_5 + over_1_5
    COMPLEMENTARY = "complementary"       # Ej: btts_yes + over_2_5
    NEUTRAL = "neutral"
```

**Reglas de correlación implementadas:**

#### Contradictorias (penalización = 1.0):
```python
# ticket_structure.py líneas 204-231
contradictory_pairs = {
    {"1x2_home", "1x2_away"},
    {"1x2_home", "1x2_draw"},
    {"1x2_away", "1x2_draw"},
    {"btts_yes", "btts_no"},
    # Over/Under conflictantes
}
```

#### Redundancia Anidada (penalización = 0.70-0.85):
```python
# Líneas 233-244
if total_side_a == total_side_b and same_family:
    strength = _nested_totals_strength(side, line_a, line_b)
    # Distancia 1 en la cadena → 0.85
    # Distancia 2 → 0.75
    # Distancia ≥3 → 0.70
```

#### Complementarias (bonus positivo):
```python
# Líneas 290-306
complementary_pairs = {
    {"btts_yes", "over_2_5"}: 0.90,
    {"btts_yes", "over_3_5"}: 0.82,
    {"1x2_home", "btts_no"}: 0.78,
    ...
}
```

**Cálculo de penalizaciones:**
```python
# Líneas 313-325
contradiction_penalty = min(1.0, sum(strength for contradictory))
redundancy_penalty = min(1.0, sum(strength for nested_redundant))
weak_information_penalty = min(1.0, sum(strength for weak_low_info))
```

**✅ BIEN IMPLEMENTADO:** Las reglas capturan intuiciones correctas sobre correlación.

**⚠️ PROBLEMA:**
- No hay **matriz de correlación empírica** basada en datos históricos
- Todas las correlaciones son **heurísticas/reglas manuales**
- Podría haber pares "NEUTRAL" que en realidad están correlacionados positivamente

---

### 3.3 Clasificación de Riesgo

**Archivo:** `/workspace/predicciones/src/models/market_catalog.py` (276 líneas)

**RiskProfile por mercado:**
```python
# market_catalog.py
LOW: over_1_5, under_4_5, double_chance_*, corners_over_6_5
MEDIUM: over_2_5, under_3_5, btts_*, corners_over_7_5
HIGH: 1x2_*, over_3_5, over_4_5, under_1_5, under_2_5
```

**Leg counts por riesgo** (`parlay_builder.py` líneas 493-497):
```python
risk_leg_counts = {
    LOW: [2, 3],      # 2-3 legs
    MEDIUM: [2, 3, 4], # 2-4 legs
    HIGH: [3, 4, 5],   # 3-5 legs
}
```

**Filtros por riesgo:**
```python
# parlay_builder.py línea 508
valid_candidates = [c for c in candidates if mapped_risk in c.risk_fit]
```

**Target de probabilidad combinada** (`parlay_builder.py` líneas 373-378):
```python
risk_level_prob_targets = {
    LOW: 0.4,    # Target: 40% combined prob
    MEDIUM: 0.2, # Target: 20% combined prob
    HIGH: 0.1,   # Target: 10% combined prob
}
prob_score = min(1.0, combined_prob / target)
```

**Filosofía declarada vs implementación:**

| Riesgo | Script declarado | Margen de error | Legs | Target prob |
|--------|------------------|-----------------|------|-------------|
| LOW    | Ancho            | Alto            | 2-3  | 40%         |
| MEDIUM | Equilibrado      | Medio           | 2-4  | 20%         |
| HIGH   | Estrecho/agresivo| Mínimo          | 3-5  | 10%         |

**✅ COHERENTE:** La implementación refleja la filosofía declarada.

**⚠️ PROBLEMA:**
- No hay validación de que la probabilidad combinada **real** esté cerca del target
- El scoring normaliza por target, pero no penaliza desviaciones grandes

---

### 3.4 Conexión EV/Edge ↔ Selección de Picks

**Archivo:** `/workspace/predicciones/src/models/market_evaluation.py` (447 líneas)

**Cálculo de EV y Edge:**
```python
# market_evaluation.py líneas 90-99
def compute_edge(model_prob, reference_prob):
    return model_prob - reference_prob

def compute_ev(model_prob, decimal_odds):
    return (model_prob * decimal_odds) - 1.0
```

**Referencias de probabilidad:**
```python
class ProbabilityReference(Enum):
    RAW_IMPLIED = "raw_implied"       # 1/odds directo
    NO_VIG_FAIR = "no_vig_fair"       # Odds sin vig (2-way o 3-way)
    DERIVED_NO_VIG = "derived_no_vig" # Derivado de 1X2 complementario
    MODEL_ONLY = "model_only"         # Sin odds disponibles
```

**Remoción de vig:**
```python
# Líneas 60-87
def remove_vig_two_way(odds_a, odds_b):
    implied_a = 1/odds_a
    implied_b = 1/odds_b
    total = implied_a + implied_b
    return {"a": implied_a/total, "b": implied_b/total}

def remove_vig_three_way(odds_a, odds_b, odds_c):
    # Similar para 1X2
```

**Uso en SGP Builder:**
```python
# parlay_builder.py líneas 385-398
phase1_selectivity = []
for pick in parlay:
    if pick.market_evaluation is None:
        phase1_selectivity.append(0.45)  # Default
        continue
    
    status_score = {
        ACCEPTED: 1.0,
        MARGINAL: 0.65,
        DISCARDED: 0.0,
    }[evaluation.status]
    
    ev_bonus = clip(evaluation.ev, -0.15, 0.25) * 0.5
    edge_bonus = clip(evaluation.edge, -0.08, 0.12) * 0.8
    
    selectivity = clip(status_score*0.7 + ev_bonus + edge_bonus, 0, 1)
```

**Ponderación en score final** (`parlay_builder.py` líneas 413-426):
```python
final_score = (
    prob_score * 0.18 +
    avg_conf * 0.17 +
    selectivity_score * 0.15 +  # ← Incluye EV/edge aquí
    compatibility_score * 0.10 +
    information_gain * 0.12 +
    family_diversity * 0.10 +
    structure_score * 0.12 +
    specificity * 0.06 +
    complementarity_bonus * 0.05 -
    contradiction_penalty * 0.85 -
    redundancy_penalty * 0.55 -
    weak_info_penalty * 0.30
)
```

**✅ EV/edge SE USA** en la selección, pero con peso moderado (15% vía selectivity).

**⚠️ PROBLEMA DETECTADO:**
- Mercados sin odds (`MODEL_ONLY`) reciben EV=None, edge=None
- Estos mercados **igual pueden ser seleccionados** si pasan otros filtros
- No hay distinción clara entre picks "value-based" (con EV+) vs "model-only"

---

## 4. AUDITORÍA DE CONSISTENCIA GLOBAL

### 4.1 ¿El modelo Poisson está implementado correctamente?

**✅ SÍ**, con matices:

| Componente | Estado | Notas |
|------------|--------|-------|
| Fórmula Poisson | ✅ Correcto | `scipy.stats.poisson.pmf` |
| Independencia home/away | ✅ Correcto | `outer(p_home, p_away)` |
| Corrección Dixon-Coles | ✅ Correcto | τ(i,j,ρ) implementado según paper original |
| 1X2 desde matriz | ✅ Correcto | Suma sobre triángulos |
| BTTS desde matriz | ✅ Correcto | Inclusión-exclusión precisa |
| Over/Under | ✅ Correcto | Suma sobre i+j > threshold |
| Normalización | ✅ Correcto | Compensa truncamiento |

**Supuestos discutibles:**
1. **Independencia conditional:** Asume que dados λ_h y λ_a, los goles son independientes. La corrección DC solo ajusta scores bajos.
2. **max_goals=8:** Puede perder probabilidad en partidos muy abiertos (λ_total > 5)
3. **Halftime:** No usa corrección DC, solo Poisson simple con fracción 0.45

---

### 4.2 ¿Los módulos Markov aportan a las probabilidades de goles?

**✅ SÍ**, pero de forma **indirecta y opcional**:

```
Markov features → ajuste de λ → matriz DC → mercados → SGP
                   ↑
              peso 0.18 (configurable)
```

**Cuando NO aporta:**
- Pre-match (no hay `match_state`)
- Si `use_markov_features = False`
- Si las tablas CSV no están cargadas

**Cuando SÍ aporta:**
- In-play prediction con estado conocido
- Ajusta λ en ±10-20% típico según estado

**Recomendación:** Documentar claramente cuándo Markov está activo y cuánto impacto tiene.

---

### 4.3 ¿Las probabilidades del SGP Builder están calibradas o crudas?

**🔴 NO CALIBRADAS** - Este es el problema más grave detectado.

**Evidencia:**
```python
# parlay_builder.py líneas 292-328
prob = derived["probability"]  # ← Viene de derive_goal_markets()
# derive_goal_markets() lee pred_data['predictions']
# pred_data['predictions'] viene de market_derivation.derive_all_markets()
# derive_all_markets() usa la matriz DC directamente

# NO HAY LLAMADA A:
# calibrated = calibration_manager.calibrate_markets(raw_markets)
```

**Impacto:**
- Si el modelo tiene sesgo sistemático (ej. sobreestima favoritos), el SGP heredará ese sesgo
- EV y edge se calculan sobre probabilidades potencialmente mal calibradas
- La "confianza" del modelo puede estar inflada/artificial

**Solución requerida:**
1. Insertar paso de calibración en el pipeline
2. O alternativamente: marcar claramente que el SGP usa "raw model probabilities"

---

### 4.4 ¿La construcción de parlays respeta la matemática definida?

| Requisito | Estado | Notas |
|-----------|--------|-------|
| Matemática Poisson (totals, BTTS, 1X2) | ✅ Sí | Derivación analítica correcta |
| Correlaciones explícitas | ✅ Sí | Reglas en ticket_structure.py |
| Uso de EV/edge | ⚠️ Parcial | 15% del score, mercados sin odds aceptados |
| Filosofía riesgo/guion | ✅ Sí | Low/Medium/High coherentes |

**Inconsistencias encontradas:**

1. **Probabilidades no calibradas** (ya mencionado)
2. **Mercados sin odds** pueden entrar en parlays sin EV real
3. **Correlaciones heurísticas** sin validación empírica
4. **Joint probability exacta** solo para mercados goal-derived; híbridos usan producto independiente (aproximación)

---

## 5. DIAGRAMA DE FLUJO

```
┌─────────────────────────────────────────────────────────────────────┐
│                         PIPELINE PRINCIPAL                          │
└─────────────────────────────────────────────────────────────────────┘

┌──────────────┐     ┌───────────────┐     ┌──────────────────────┐
│  FEATURES    │     │   LAMBDA      │     │  DIXON-COLES         │
│  (ratings,   │────▶│   PREDICTION  │────▶│  SCORE MATRIX        │
│   form, xG)  │     │  λ_h, λ_a     │     │  matrix[i,j]         │
└──────────────┘     └───────────────┘     └──────────────────────┘
                            │                      │
                            │ (opcional)           │
                     ┌──────▼───────┐              │
                     │   MARKOV     │              │
                     │   ADJUSTMENT │              │
                     │  λ_adj = λ×factor          │
                     └──────────────┘              │
                                                   ▼
                                      ┌────────────────────────┐
                                      │  MARKET DERIVATION     │
                                      │  - 1X2                 │
                                      │  - BTTS                │
                                      │  - Over/Under          │
                                      │  - Correct Scores      │
                                      │  - Halftime            │
                                      └────────────────────────┘
                                                   │
                                                   │ (PROBABILIDADES CRUDAS)
                                                   │ 🔴 SIN CALIBRAR
                                                   ▼
                                      ┌────────────────────────┐
                                      │  SGP BUILDER           │
                                      │  - Market Catalog      │
                                      │  - Evaluation (EV/edge)│
                                      │  - Ticket Structure    │
                                      │  - Risk Profiles       │
                                      └────────────────────────┘
                                                   │
                                                   ▼
                                      ┌────────────────────────┐
                                      │  PARLAYS               │
                                      │  - Low Risk (2-3 legs) │
                                      │  - Medium Risk (2-4)   │
                                      │  - High Risk (3-5)     │
                                      └────────────────────────┘


┌─────────────────────────────────────────────────────────────────────┐
│                    CALIBRATION (EXISTE PERO NO SE USA)              │
│                                                                     │
│  CalibrationManager                                                 │
│  - Isotonic Regression                                              │
│  - Platt Scaling                                                    │
│                                                                     │
│  🔴 NO SE INVOCA EN EL PIPELINE DEL SGP BUILDER                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 6. LISTA PRIORIZADA DE FIXES RECOMENDADOS

### CRÍTICOS (P0)

#### 1. Insertar calibración en el pipeline del SGP Builder
**Archivo:** `parlay_builder.py`  
**Función:** `generate_same_game_candidates()`

```python
# AGREGAR después de derive_goal_markets:
from .calibration import CalibrationManager

calibration_manager = CalibrationManager(config=pred_data.get('config'))
# Intentar cargar calibradores existentes
calibration_manager.load_from_config(base_dir="output/calibrators")

# Aplicar calibración a las predicciones
raw_predictions = pred_data.get('predictions', {})
calibrated_predictions = calibration_manager.calibrate_markets(raw_predictions)

# Usar calibrated_predictions en lugar de raw_predictions
```

**Impacto:** Alto - corrige sesgos sistemáticos en todas las probabilidades del SGP.

---

#### 2. Marcar claramente picks sin EV real
**Archivo:** `parlay_builder.py`  
**Función:** `evaluate_parlay()`

```python
# AGREGAR flag en PickCandidate
@dataclass
class PickCandidate:
    ...
    has_real_ev: bool = False  # ← NUEVO

# En generate_same_game_candidates:
evaluation = evaluation_by_key.get(market_key)
pick = PickCandidate(
    ...
    market_evaluation=evaluation,
    has_real_ev=evaluation.ev is not None and evaluation.ev > 0 if evaluation else False
)

# En evaluate_parlay:
if not any(p.has_real_ev for p in parlay):
    scores["warning"] = -0.1  # Penalizar tickets sin EV real
```

**Impacto:** Medio - mejora transparencia y calidad de picks.

---

### ALTOS (P1)

#### 3. Validar masa de probabilidad truncada en matriz
**Archivo:** `dixon_coles.py`  
**Función:** `dc_score_matrix()`

```python
# AGREGAR después de normalizar:
truncated_mass = 1.0 - matrix.sum()
if truncated_mass > 0.005:  # 0.5%
    logger.warning(
        f"Truncated probability mass = {truncated_mass:.4f}. "
        f"Consider increasing max_goals. λ_total = {lambda_home + lambda_away:.2f}"
    )
```

**Impacto:** Bajo-Medio - previene errores silenciosos en partidos extremos.

---

#### 4. Matriz de correlación empírica
**Archivo:** `ticket_structure.py`  
**Nuevo módulo sugerido:** `correlation_matrix.py`

```python
# Calcular correlaciones históricas entre mercados
def build_empirical_correlation_matrix(historical_matches):
    """
    Para cada par de mercados, calcular correlación de outcomes.
    Ej: btts_yes y over_2_5 deberían tener correlación positiva ~0.4-0.6
    """
    pass

# Usar en classify_market_relation:
empirical_corr = load_empirical_correlations()
if (market_a, market_b) in empirical_corr:
    relation.strength = empirical_corr[(market_a, market_b)]
```

**Impacto:** Medio - reemplaza heurísticas con datos reales.

---

#### 5. Validar que risk_fit sea coherente con probabilidad
**Archivo:** `market_catalog.py`

```python
# AGREGAR test de consistencia:
def validate_risk_profiles():
    catalog = build_market_catalog()
    for key, definition in catalog.items():
        # Verificar que mercados LOW tengan probabilidad típica > 60%
        # Verificar que mercados HIGH tengan probabilidad típica < 40%
        pass
```

**Impacto:** Bajo - asegura coherencia interna.

---

### MEDIOS (P2)

#### 6. Documentar impacto de Markov en λ
**Archivo:** `dixon_coles.py`

```python
# AGREGAR logging:
logger.info(
    f"Markov adjustment applied: λ_home {lambda_h:.3f} → {lambda_h_adj:.3f} "
    f"({(lambda_h_adj/lambda_h - 1)*100:+.1f}%)"
)
```

**Impacto:** Bajo - mejora observabilidad.

---

#### 7. Halftime con Dixon-Coles
**Archivo:** `market_derivation.py`  
**Función:** `derive_halftime()`

```python
# CAMBIAR de Poisson simple a DC:
from .dixon_coles import dc_score_matrix

ht_matrix = dc_score_matrix(lambda_ht_h, lambda_ht_a, rho=-0.13, max_goals=6)
```

**Impacto:** Bajo - mejora consistencia teórica.

---

## 7. RESUMEN EJECUTIVO

### ✅ Lo que funciona bien:
1. **Modelo Poisson/Dixon-Coles:** Implementación correcta y estándar
2. **Derivación de mercados:** Fórmulas analíticas precisas
3. **Correlaciones en SGP:** Reglas intuitivas y bien estructuradas
4. **Risk profiles:** Coherentes con filosofía declarada
5. **Markov integration:** Funciona cuando está activado

### 🔴 Problemas críticos:
1. **Calibración no aplicada:** El SGP usa probabilidades crudas, no calibradas
2. **EV/edge incompleto:** Mercados sin odds pueden entrar sin EV real
3. **Correlaciones heurísticas:** Sin validación empírica

### 📋 Recomendación principal:
**Prioridad inmediata:** Insertar el paso de calibración en el pipeline del SGP Builder antes de la evaluación de mercados. Esto requiere:
1. Verificar que existan calibradores entrenados (`output/calibrators/*.pkl`)
2. Si no existen, entrenarlos con datos históricos
3. Aplicar `calibration_manager.calibrate_markets()` antes de `generate_same_game_candidates()`

---

**Fin de la auditoría.**
