"""
Feature pipeline: orchestrates the full feature extraction process
for a complete match (both teams) from a MatchInput object.
"""
from __future__ import annotations

import logging
from typing import Dict, Any, Optional, Tuple

from ..ingestion.schemas import MatchInput
from .team_features import extract_team_features

logger = logging.getLogger(__name__)


def build_match_features(
    match: MatchInput,
    config: Optional[Dict[str, Any]] = None,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Build complete feature dicts for both teams in a match.

    Returns:
        Tuple of (home_features, away_features) dicts.
    """
    if config is None:
        config = {}

    meta = match.metadata
    team1 = match.team1
    team2 = match.team2

    # Determine home/away based on metadata and FACTORES_EXTERNOS
    # If neutral venue, neither team has home advantage
    neutral = meta.neutral_venue

    home_localia = team1.FACTORES_EXTERNOS.localía if not neutral else 0.0
    away_localia = team2.FACTORES_EXTERNOS.localía if not neutral else 0.0

    # team1 is home if their localia is positive or neutral
    team1_is_home = (home_localia >= 0) and not neutral

    home_features = extract_team_features(
        team=team1,
        is_home=team1_is_home,
        opponent_ranking=team2.CONTEXTO.ranking_fifa or 50.0,
        config=config,
    )

    away_features = extract_team_features(
        team=team2,
        is_home=not team1_is_home and not neutral,
        opponent_ranking=team1.CONTEXTO.ranking_fifa or 50.0,
        config=config,
    )

    # Override home advantage if neutral venue
    if neutral:
        home_features['home_advantage_log'] = 0.0
        away_features['home_advantage_log'] = 0.0

    logger.debug(
        f"Features extracted: {team1.nombre} (home={team1_is_home}) vs "
        f"{team2.nombre} (neutral={neutral})"
    )

    return home_features, away_features


def features_to_flat_vector(
    home_features: Dict[str, Any],
    away_features: Dict[str, Any],
) -> Dict[str, float]:
    """
    Convert home/away feature dicts into a single flat vector
    suitable for ML model input (LightGBM, etc.).
    """
    flat = {}
    numeric_keys = [
        'goals_scored_avg', 'goals_conceded_avg', 'xg_favor', 'xg_contra',
        'win_rate', 'form_ppg', 'ranking_fifa', 'attack_rating', 'defense_rating',
        'form_factor', 'ranking_factor', 'h2h_factor', 'squad_multiplier',
        'home_advantage_log', 'context_modifier', 'fatigue', 'motivation',
        'travel_km', 'altitude_m', 'weather_factor', 'pressing', 'bloque',
        'transiciones', 'cohesion', 'creatividad', 'solidez', 'disciplina',
        'squad_quality', 'squad_attack_quality', 'squad_defense_quality',
        'squad_key_player_impact', 'squad_availability_factor',
    ]

    for key in numeric_keys:
        home_val = home_features.get(key, 0.0)
        away_val = away_features.get(key, 0.0)
        if isinstance(home_val, (int, float)):
            flat[f'home_{key}'] = float(home_val)
        if isinstance(away_val, (int, float)):
            flat[f'away_{key}'] = float(away_val)

    # Difference features
    for key in ['attack_rating', 'defense_rating', 'form_factor', 'ranking_factor',
                 'squad_quality', 'xg_favor', 'goals_scored_avg']:
        h = flat.get(f'home_{key}', 0.0)
        a = flat.get(f'away_{key}', 0.0)
        flat[f'diff_{key}'] = h - a
        flat[f'ratio_{key}'] = h / (a + 0.01)

    return flat
