"""
Team feature extraction: converts TeamData (from JSON) into
a flat feature dict ready for rating computation and lambda estimation.
"""
from __future__ import annotations

import logging
from typing import Dict, Any, Optional

import numpy as np

from ..ingestion.schemas import TeamData
from .ratings import (
    compute_attack_rating, compute_defense_rating, compute_form_factor,
    compute_ranking_factor, compute_h2h_factor, compute_context_modifier,
)
from .player_aggregator import aggregate_squad, squad_quality_to_multiplier

logger = logging.getLogger(__name__)


def extract_team_features(
    team: TeamData,
    is_home: bool = False,
    opponent_ranking: float = 50.0,
    config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Extract all features for a team from its TeamData object.

    Returns a comprehensive feature dict with:
    - Raw features from JSON
    - Derived ratings (attack, defense, form, ranking, h2h)
    - Squad quality metrics
    - Context modifier
    - Home advantage flag
    """
    if config is None:
        config = {}

    ctx = team.CONTEXTO
    ext = team.FACTORES_EXTERNOS
    colect = team.FACTORES_COLECTIVOS
    tacticos = team.FACTORES_TACTICOS

    # ------------------------------------------------------------------
    # 1. Historical stats (from CONTEXTO)
    # ------------------------------------------------------------------
    ranking = ctx.ranking_fifa or 50.0
    goals_scored_avg = _safe_rate_with_prior(
        ctx.goles_marcados_ultimos_6,
        ctx.partidos_ultimos_6_meses,
        ranking_fifa=ranking,
        is_goals_scored=True,
    )
    goals_conceded_avg = _safe_rate_with_prior(
        ctx.goles_recibidos_ultimos_6,
        ctx.partidos_ultimos_6_meses,
        ranking_fifa=ranking,
        is_goals_scored=False,
    )
    # Si no hay xG real, pasar None para evitar blend duplicado en compute_attack_rating
    xg_favor = ctx.xg_promedio_favor if (ctx.xg_promedio_favor and ctx.xg_promedio_favor > 0) else None
    xg_contra = ctx.xg_promedio_contra if (ctx.xg_promedio_contra and ctx.xg_promedio_contra > 0) else None

    total_h2h = (ctx.head_to_head_wins + ctx.head_to_head_draws + ctx.head_to_head_losses)
    win_rate = _safe_rate(ctx.victorias_ultimos_6, ctx.partidos_ultimos_6_meses)
    form_ppg = _calc_ppg(
        ctx.victorias_ultimos_6, ctx.empates_ultimos_6,
        ctx.derrotas_ultimos_6, ctx.partidos_ultimos_6_meses
    )

    # ------------------------------------------------------------------
    # 2. Derived ratings
    # ------------------------------------------------------------------
    ranking_factor = compute_ranking_factor(ctx.ranking_fifa or 50.0)

    attack_rating = compute_attack_rating(
        goals_scored_avg=goals_scored_avg,
        xg_avg=xg_favor,
        efficiency=colect.eficiencia_finalizacion,
        creativity=colect.creatividad_ofensiva,
        ranking_factor=ranking_factor,
    )

    defense_rating = compute_defense_rating(
        goals_conceded_avg=goals_conceded_avg,
        xg_against_avg=xg_contra,
        solidity=colect.solidez_defensiva,
        pressing=tacticos.pressing_intensidad,
    )

    form_factor = compute_form_factor(
        win_rate=win_rate,
        form_ppg=form_ppg,
        dias_descanso=ctx.dias_desde_ultimo_partido or 7,
        partidos_acumulados=ctx.partidos_en_15_dias or 1,
    )

    h2h_factor = compute_h2h_factor(
        h2h_wins=ctx.head_to_head_wins or 0,
        h2h_draws=ctx.head_to_head_draws or 0,
        h2h_losses=ctx.head_to_head_losses or 0,
    )

    # ------------------------------------------------------------------
    # 3. Squad quality
    # ------------------------------------------------------------------
    squad_metrics = aggregate_squad(team.JUGADORES)
    squad_multiplier = squad_quality_to_multiplier(
        squad_metrics,
        weight=config.get('feature_weights', {}).get('squad_quality', 0.22)
    )

    # ------------------------------------------------------------------
    # 4. Context modifier (external factors)
    # ------------------------------------------------------------------
    home_advantage_log = 0.0
    if is_home and ext.localía > 0:
        home_advantage_log = config.get('dixon_coles', {}).get('home_advantage', 0.25)
        home_advantage_log *= ext.localía  # scale by localia strength

    # Weather factor
    weather_factor = _compute_weather_factor(
        ext.temperatura_c, ext.humedad_pct, ext.lluvia, ext.viento_kmh
    )

    context_modifier = compute_context_modifier(
        fatigue=ext.fatiga_acumulada,
        motivation=ext.motivacion,
        travel_km=ext.distancia_viaje_km or 0,
        altitude_m=ext.altitud_m or 0,
        importance=ext.importancia_partido,
        pressure=ext.presion_mediatica,
        weather_factor=weather_factor,
        weight_fatigue=config.get('feature_weights', {}).get('fatigue_penalty', 0.05),
        weight_motivation=config.get('feature_weights', {}).get('motivation_boost', 0.04),
        weight_travel=config.get('feature_weights', {}).get('travel_penalty', 0.03),
    )

    # ------------------------------------------------------------------
    # 5. Assemble output
    # ------------------------------------------------------------------
    features = {
        # Identification
        'nombre': team.nombre,
        'is_home': is_home,

        # Raw stats
        'goals_scored_avg': goals_scored_avg,
        'goals_conceded_avg': goals_conceded_avg,
        'xg_favor': xg_favor,
        'xg_contra': xg_contra,
        'win_rate': win_rate,
        'form_ppg': form_ppg,
        'ranking_fifa': ctx.ranking_fifa or 50.0,

        # Derived ratings
        'attack_rating': attack_rating,
        'defense_rating': defense_rating,
        'form_factor': form_factor,
        'ranking_factor': ranking_factor,
        'h2h_factor': h2h_factor,
        'squad_multiplier': squad_multiplier,

        # Context
        'home_advantage_log': home_advantage_log,
        'context_modifier': context_modifier,
        'fatigue': ext.fatiga_acumulada,
        'motivation': ext.motivacion,
        'travel_km': ext.distancia_viaje_km or 0,
        'altitude_m': ext.altitud_m or 0,
        'weather_factor': weather_factor,

        # Tactical
        'pressing': tacticos.pressing_intensidad,
        'bloque': tacticos.bloque_defensivo,
        'transiciones': tacticos.transiciones_rapidas,

        # Collective
        'cohesion': colect.cohesion_grupal,
        'creatividad': colect.creatividad_ofensiva,
        'solidez': colect.solidez_defensiva,
        'disciplina': colect.discipline,

        # Squad
        **{f'squad_{k}': v for k, v in squad_metrics.items()},
    }

    return features


def _safe_rate(numerator, denominator, default: float = 1.2) -> float:
    """Safe division with default fallback."""
    try:
        n = float(numerator or 0)
        d = float(denominator or 0)
        if d == 0:
            return default
        return n / d
    except (TypeError, ValueError):
        return default


def _safe_rate_with_prior(
    numerator,
    denominator,
    ranking_fifa: float = 50.0,
    is_goals_scored: bool = True,
) -> float:
    """
    Safe division that uses FIFA ranking as a Bayesian prior
    when no historical data is available.
    Rank 1 team without data → estimated as 1.8 goals/game scored, 0.7 conceded
    Rank 100 team without data → estimated as 1.0 goals/game scored, 1.4 conceded
    """
    try:
        n = float(numerator or 0)
        d = float(denominator or 0)
        if d > 0:
            return n / d
        # No data: derive prior from ranking
        # normalized: rank 1 → 1.0, rank 100 → 0.0, rank 200 → -1.0
        rank_norm = max(-1.0, (100.0 - ranking_fifa) / 100.0)
        if is_goals_scored:
            # Scored: strong teams score more
            return 1.35 + 0.45 * rank_norm   # rank 1 → 1.80, rank 50 → 1.13, rank 100 → 0.90
        else:
            # Conceded: weak teams concede more
            return 1.35 - 0.45 * rank_norm   # rank 1 → 0.90, rank 50 → 1.13, rank 100 → 1.80
    except (TypeError, ValueError):
        return 1.2


def _calc_ppg(wins, draws, losses, total, default: float = 1.5) -> float:
    """Calculate points per game."""
    try:
        t = int(total or 0)
        if t == 0:
            return default
        pts = int(wins or 0) * 3 + int(draws or 0) * 1
        return pts / t
    except (TypeError, ValueError):
        return default


def _compute_weather_factor(
    temp_c: Optional[float],
    humidity: Optional[float],
    rain: Optional[bool],
    wind_kmh: Optional[float],
) -> float:
    """
    Compute weather impact as a small log-scale modifier.
    Extreme conditions reduce performance slightly.
    Returns a value in [-0.03, 0.0].
    """
    factor = 0.0
    if temp_c is not None:
        if temp_c > 35 or temp_c < 5:
            factor -= 0.02
        elif temp_c > 30:
            factor -= 0.01
    if rain:
        factor -= 0.01
    if wind_kmh is not None and wind_kmh > 40:
        factor -= 0.01
    if humidity is not None and humidity > 85:
        factor -= 0.005
    return float(np.clip(factor, -0.03, 0.0))
