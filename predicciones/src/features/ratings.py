"""
Team rating system: computes offensive, defensive, and global ratings
from historical data and JSON input features.

Ratings are normalized to a comparable scale [0..1] internally,
but exposed as lambda-scale values for the Poisson/DC model.
"""
from __future__ import annotations

import logging
from typing import Dict, Any, Optional

import numpy as np

logger = logging.getLogger(__name__)

# League-average goals per game (used as baseline)
LEAGUE_AVG_GOALS = 1.35  # International football average ~2.5 total, ~1.35 home


def compute_attack_rating(
    goals_scored_avg: float,
    xg_avg: Optional[float] = None,
    efficiency: float = 5.0,  # FACTORES_COLECTIVOS.eficiencia_finalizacion
    creativity: float = 5.0,  # FACTORES_COLECTIVOS.creatividad_ofensiva
    ranking_factor: float = 1.0,
) -> float:
    """
    Compute offensive rating as a multiplier relative to league average.
    Returns: attack multiplier (e.g., 1.5 means 50% above average attack).
    """
    # Base: goals scored relative to league average
    base = goals_scored_avg / LEAGUE_AVG_GOALS

    # Blend with xG if available (pseudo-xG if not directly measured)
    if xg_avg is not None and xg_avg > 0 and xg_avg != goals_scored_avg:
        # Solo blendear si xG es un dato diferente a los goles reales
        # Si xg_avg == goals_scored_avg, es un fallback duplicado — ignorarlo
        xg_factor = xg_avg / LEAGUE_AVG_GOALS
        base = 0.55 * base + 0.45 * xg_factor  # xG pesa más que goles brutos

    # Collective modifiers (normalized to [-0.15, +0.15] range)
    coll_boost = (efficiency - 5.0) / 5.0 * 0.18
    creat_boost = (creativity - 5.0) / 5.0 * 0.10

    # Ranking factor (slight adjustment for elite teams)
    rating = base * ranking_factor * (1 + coll_boost + creat_boost)

    return float(np.clip(rating, 0.1, 4.0))


def compute_defense_rating(
    goals_conceded_avg: float,
    xg_against_avg: Optional[float] = None,
    solidity: float = 5.0,  # FACTORES_COLECTIVOS.solidez_defensiva
    pressing: float = 5.0,  # FACTORES_TACTICOS.pressing_intensidad
) -> float:
    """
    Compute defensive rating as a multiplier on opponent's expected goals.
    Returns: defense suppression factor (lower = better defense).
    Lower means the team concedes less.
    Scale: 0.5 = excellent (concedes half average), 1.5 = poor.
    """
    base = goals_conceded_avg / LEAGUE_AVG_GOALS

    if xg_against_avg is not None and xg_against_avg > 0:
        xg_factor = xg_against_avg / LEAGUE_AVG_GOALS
        base = 0.6 * base + 0.4 * xg_factor

    # Collective modifiers
    solid_reduction = (solidity - 5.0) / 5.0 * 0.18
    press_reduction = (pressing - 5.0) / 5.0 * 0.10

    defense = base * (1 - solid_reduction - press_reduction)

    return float(np.clip(defense, 0.1, 3.0))


def compute_form_factor(
    win_rate: float,
    form_ppg: float,
    dias_descanso: int = 7,
    partidos_acumulados: int = 1,
) -> float:
    """
    Compute recent form factor as a multiplier.
    Returns: form multiplier (e.g., 1.1 = 10% boost from good form).
    """
    # PPG-based form: 3 ppg = perfect, 0 = terrible
    form_normalized = form_ppg / 3.0  # 0..1

    # Win rate complements form
    form_score = 0.7 * form_normalized + 0.3 * win_rate  # 0..1

    # Convert to multiplier: neutral at form_score=0.5
    # Range: [0.72, 1.28] in vez de [0.85, 1.15]
    # Equipo en forma perfecta: 1.28, equipo sin victorias: 0.72
    form_multiplier = 0.72 + 0.56 * form_score

    # Fatigue from accumulated matches
    fatigue_penalty = min(0.05, (partidos_acumulados - 1) * 0.02)

    # Rest bonus/penalty
    if dias_descanso < 4:
        rest_factor = 0.95  # very tired
    elif dias_descanso > 14:
        rest_factor = 0.98  # slightly rusty
    else:
        rest_factor = 1.0

    return float(np.clip(form_multiplier * rest_factor - fatigue_penalty, 0.7, 1.3))


def compute_ranking_factor(ranking_fifa: float, reference_rank: float = 50.0) -> float:
    """
    Convert FIFA ranking to a multiplicative factor.
    Rank 1 = strong boost, Rank 211 = penalty.
    Returns multiplier in [0.65, 1.35].
    """
    # Rank 1 → 1.35, Rank 50 → 1.05, Rank 100 → 0.75, Rank 150+ → 0.65
    normalized = max(0, (reference_rank - ranking_fifa) / reference_rank)
    factor = 0.65 + 0.70 * normalized
    return float(np.clip(factor, 0.65, 1.35))


def compute_h2h_factor(
    h2h_wins: int,
    h2h_draws: int,
    h2h_losses: int,
) -> float:
    """
    Head-to-head factor: slight boost if historically dominant.
    Returns multiplier in [0.95, 1.05].
    """
    total = h2h_wins + h2h_draws + h2h_losses
    if total == 0:
        return 1.0
    win_ratio = (h2h_wins + 0.5 * h2h_draws) / total
    # Nuevo rango: [0.88, 1.12] en vez de [0.95, 1.05]
    # H2H dominante → 1.12, H2H muy adverso → 0.88
    return float(0.88 + 0.24 * win_ratio)


def estimate_lambda(
    attack_rating: float,
    opponent_defense_rating: float,
    form_factor: float = 1.0,
    ranking_factor: float = 1.0,
    h2h_factor: float = 1.0,
    home_advantage: float = 0.0,
    context_modifier: float = 0.0,
    min_lambda: float = 0.05,
    max_lambda: float = 5.0,
) -> float:
    """
    Estimate expected goals (lambda) for a team.

    Formula:
        lambda = attack_rating * opponent_defense_rating *
                 form_factor * ranking_factor * h2h_factor *
                 exp(home_advantage + context_modifier)

    Args:
        attack_rating: Team's offensive multiplier vs league avg.
        opponent_defense_rating: Opponent's defensive factor (lower = harder to score).
        form_factor: Recent form multiplier.
        ranking_factor: FIFA ranking multiplier.
        h2h_factor: Head-to-head history multiplier.
        home_advantage: Log-scale home advantage (e.g., 0.2).
        context_modifier: Sum of contextual log-scale adjustments.
        min_lambda: Minimum expected goals (floor).
        max_lambda: Maximum expected goals (ceiling).

    Returns:
        Expected goals for this team in this match.
    """
    base_lambda = (attack_rating * opponent_defense_rating *
                   LEAGUE_AVG_GOALS * form_factor * ranking_factor * h2h_factor)

    # Apply home advantage and context as log-scale adjustments
    adjusted_lambda = base_lambda * np.exp(home_advantage + context_modifier)

    return float(np.clip(adjusted_lambda, min_lambda, max_lambda))


def compute_context_modifier(
    fatigue: float = 3.0,
    motivation: float = 5.0,
    travel_km: float = 0.0,
    altitude_m: float = 0.0,
    importance: float = 5.0,
    pressure: float = 5.0,
    weather_factor: float = 0.0,
    weight_fatigue: float = 0.05,
    weight_motivation: float = 0.04,
    weight_travel: float = 0.03,
    weight_altitude: float = 0.02,
    weight_importance: float = 0.02,
    weight_pressure: float = 0.01,
) -> float:
    """
    Compute a log-scale context modifier combining external factors.
    Positive values boost lambda, negative values suppress it.

    All factors on 0-10 scale except travel (km) and altitude (m).
    Returns: log-scale modifier (typically in [-0.15, +0.15]).
    """
    # Fatigue: high fatigue reduces performance
    fatigue_mod = -weight_fatigue * (fatigue - 3.0) / 7.0  # neutral at 3

    # Motivation: high motivation boosts performance
    motivation_mod = weight_motivation * (motivation - 5.0) / 5.0  # neutral at 5

    # Travel: long travel reduces performance
    travel_mod = -weight_travel * min(travel_km / 5000.0, 1.0)

    # Altitude: high altitude is a challenge for away teams (handled externally)
    altitude_mod = -weight_altitude * min(altitude_m / 3000.0, 1.0)

    # Importance: high importance games sometimes produce more cautious play
    importance_mod = weight_importance * (importance - 5.0) / 5.0 * 0.5

    # Media pressure (can be positive or negative)
    pressure_mod = -weight_pressure * (pressure - 5.0) / 5.0 * 0.3

    # Weather: negative for extreme conditions
    weather_mod = weather_factor  # already normalized externally

    total = (fatigue_mod + motivation_mod + travel_mod +
             altitude_mod + importance_mod + pressure_mod + weather_mod)

    return float(np.clip(total, -0.20, 0.20))
