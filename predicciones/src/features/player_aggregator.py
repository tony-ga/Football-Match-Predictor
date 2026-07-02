"""
Player aggregator: converts a list of player objects into team-level
quality metrics used by the rating and lambda estimation modules.
"""
from __future__ import annotations

import logging
from typing import List, Dict, Any, Optional

import numpy as np

logger = logging.getLogger(__name__)

# Position weights for different aspects
POSITION_WEIGHTS = {
    'GK': {'offense': 0.0, 'defense': 0.25, 'overall': 0.10},
    'DEF': {'offense': 0.05, 'defense': 0.40, 'overall': 0.25},
    'MID': {'offense': 0.25, 'defense': 0.20, 'overall': 0.30},
    'FWD': {'offense': 0.70, 'defense': 0.15, 'overall': 0.35},
}


def aggregate_squad(
    players: List[Any],  # List of Jugador objects or dicts
    starters_only: bool = False,
) -> Dict[str, float]:
    """
    Aggregate player-level data into team-level quality metrics.

    Returns a dict with:
    - squad_quality: overall squad quality [0..1]
    - attack_quality: offensive quality [0..1]
    - defense_quality: defensive quality [0..1]
    - key_player_impact: impact of top players [0..1]
    - availability_factor: squad fitness [0..1]
    - xg_contribution: estimated xG contribution from squad
    """
    if not players:
        logger.debug("No players provided, returning neutral squad metrics")
        return _neutral_squad()

    # Convert to dicts if needed
    player_dicts = []
    for p in players:
        if hasattr(p, 'model_dump'):
            player_dicts.append(p.model_dump())
        elif isinstance(p, dict):
            player_dicts.append(p)
        else:
            player_dicts.append({
                'posicion': 'MID', 'rating': 6.0,
                'forma_actual': 6.0, 'titular': True,
                'disponibilidad': 1.0, 'impacto_equipo': 5.0,
                'xg_temporada': 0.0,
            })

    if starters_only:
        player_dicts = [p for p in player_dicts if p.get('titular', True)]

    if not player_dicts:
        return _neutral_squad()

    # Compute weighted averages by position
    attack_scores = []
    defense_scores = []
    overall_scores = []
    impacts = []
    availabilities = []
    xg_contribs = []

    for p in player_dicts:
        pos = str(p.get('posicion', 'MID')).upper()
        if pos not in POSITION_WEIGHTS:
            pos = 'MID'

        weights = POSITION_WEIGHTS[pos]
        rating = float(p.get('rating', 6.0))
        forma = float(p.get('forma_actual', 6.0))
        avail = float(p.get('disponibilidad', 1.0))
        impact = float(p.get('impacto_equipo', 5.0))
        xg = float(p.get('xg_temporada', 0.0))

        # Effective rating: blend of rating and current form
        effective = 0.6 * rating + 0.4 * forma
        effective_normalized = effective / 10.0  # [0..1]

        # Availability penalty
        effective_normalized *= avail

        attack_scores.append(effective_normalized * weights['offense'])
        defense_scores.append(effective_normalized * weights['defense'])
        overall_scores.append(effective_normalized * weights['overall'])
        impacts.append(impact / 10.0)
        availabilities.append(avail)
        xg_contribs.append(xg * weights['offense'])

    n = len(player_dicts)
    # Normalize by position weights sum
    pos_counts = {'GK': 0, 'DEF': 0, 'MID': 0, 'FWD': 0}
    for p in player_dicts:
        pos = str(p.get('posicion', 'MID')).upper()
        if pos in pos_counts:
            pos_counts[pos] += 1

    # Normalizar por el valor esperado de un jugador promedio (rating=6, forma=6):
    # effective = 0.6*6 + 0.4*6 = 6.0  →  normalized = 0.6
    # overall score promedio para MID weight 0.30 → 0.6 * 0.30 = 0.180
    # Para un squad mixto real: expected mean overall_score ≈ 0.140
    _expected_overall = 0.140
    _expected_attack  = 0.070   # FWD heavy: 0.6 * 0.70 * (1/4 players) ≈ 0.105, avg with DEF/MID/GK
    _expected_defense = 0.080

    squad_quality  = float(np.clip(np.mean(overall_scores)  / _expected_overall,  0.0, 1.0)) if overall_scores  else 0.5
    attack_quality = float(np.clip(np.mean(attack_scores)   / _expected_attack,   0.0, 1.0)) if attack_scores   else 0.5
    defense_quality= float(np.clip(np.mean(defense_scores)  / _expected_defense,  0.0, 1.0)) if defense_scores  else 0.5

    # Key player impact: average of top 3 impacts
    top_impacts = sorted(impacts, reverse=True)[:3]
    key_player_impact = np.mean(top_impacts) if top_impacts else 0.5

    availability_factor = np.mean(availabilities)
    xg_contribution = sum(xg_contribs) / max(n, 1)

    result = {
        'squad_quality': float(np.clip(squad_quality, 0.0, 1.0)),
        'attack_quality': float(np.clip(attack_quality, 0.0, 1.0)),
        'defense_quality': float(np.clip(defense_quality, 0.0, 1.0)),
        'key_player_impact': float(np.clip(key_player_impact, 0.0, 1.0)),
        'availability_factor': float(np.clip(availability_factor, 0.0, 1.0)),
        'xg_contribution': float(np.clip(xg_contribution, 0.0, 10.0)),
        'n_players': n,
    }
    logger.debug(f"Squad aggregation: {result}")
    return result


def _neutral_squad() -> Dict[str, float]:
    """Return neutral squad metrics when no player data is available."""
    return {
        'squad_quality': 0.5,
        'attack_quality': 0.5,
        'defense_quality': 0.5,
        'key_player_impact': 0.5,
        'availability_factor': 1.0,
        'xg_contribution': 0.0,
        'n_players': 0,
    }


def squad_quality_to_multiplier(
    squad_metrics: Dict[str, float],
    weight: float = 0.22,   # ampliado de 0.15 → 0.22
) -> float:
    """
    Convert squad quality into a lambda multiplier.
    Returns: multiplier in [1-weight, 1+weight].
    Neutral squad (0.5) -> 1.0 multiplier.
    """
    quality = squad_metrics.get('squad_quality', 0.5)
    # Map [0, 1] -> [1-weight, 1+weight]
    multiplier = 1.0 - weight + 2.0 * weight * quality
    return float(np.clip(multiplier, 1.0 - weight, 1.0 + weight))
