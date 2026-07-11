"""
Player-level lambda calculation with position-based priors and small-sample regularization.

This module implements the explicit function f for computing individual player lambda:

    lambda_i = f(
        position_i,
        goals_recent_i,
        minutes_i,
        matches_i,
        shots_per_90_i,
        team_lambda
    )

Key features:
- Position hierarchy (forwards > wingers > AM > CM > FB > CB > GK)
- Bayesian shrinkage toward position-specific priors for small samples
- Hard caps by position to prevent outliers
- Team lambda normalization to keep sum of player lambdas reasonable
"""
from __future__ import annotations

import logging
import math
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# Position weights for structural prior (hierarchy of goal-scoring roles)
# Higher weight = more offensive role = higher baseline expectation
POSITION_WEIGHTS = {
    # Forwards / strikers
    'f': 1.00,      # forward genérico
    'cf': 1.00,     # centrodelantero / striker
    'striker': 1.00,
    'forward': 1.00,
    'attacker': 1.00,
    'delantero': 1.00,
    'punta': 1.00,
    
    # Attacking midfielders / second strikers
    'am': 0.90,     # attacking midfielder
    'mediapunta': 0.90,
    'ss': 0.90,     # second striker
    
    # Wingers / wide forwards
    'lf': 0.85,     # left forward
    'rf': 0.85,     # right forward
    'lw': 0.85,     # left winger
    'rw': 0.85,     # right winger
    'winger': 0.85,
    'extremo': 0.85,
    'wing-f': 0.80, # wing-forward
    
    # Wide attacking mids
    'am-l': 0.75,   # left attacking mid
    'am-r': 0.75,   # right attacking mid
    'aml': 0.75,
    'amr': 0.75,
    
    # Central midfielders
    'cm': 0.50,     # central midfielder
    'mc': 0.50,
    'midfielder': 0.50,
    'volante': 0.50,
    
    # Defensive midfielders
    'dm': 0.40,     # defensive midfielder
    'cdm': 0.40,
    'pivot': 0.40,
    
    # Wing backs (more offensive than fullbacks)
    'wing-b': 0.40,
    'lwb': 0.40,    # left wing-back
    'rwb': 0.40,    # right wing-back
    
    # Fullbacks
    'lb': 0.35,     # left back
    'rb': 0.35,     # right back
    'fullback': 0.35,
    'lateral': 0.35,
    
    # Centre backs
    'cd': 0.15,     # centre defender
    'cb': 0.15,
    'centreback': 0.15,
    'central': 0.15,
    'defender': 0.15,
    'defensa': 0.15,
    'cd-l': 0.15,   # left centre-back
    'cd-r': 0.15,   # right centre-back
    
    # Goalkeeper
    'g': 0.00,      # goalkeeper
    'gk': 0.00,
    'goalkeeper': 0.00,
    'portero': 0.00,
    'arquero': 0.00,
}

# Prior lambda by position (baseline expectation when data is scarce)
# Represents typical goal rate per match for each position
PRIOR_BY_POSITION = {
    # Forwards
    'f': 0.35, 'cf': 0.35, 'striker': 0.35, 'forward': 0.35, 'attacker': 0.35,
    'delantero': 0.35, 'punta': 0.35,
    
    # Attacking mids
    'am': 0.25, 'mediapunta': 0.25, 'ss': 0.25,
    
    # Wingers
    'lf': 0.22, 'rf': 0.22, 'lw': 0.22, 'rw': 0.22, 'winger': 0.22,
    'extremo': 0.22, 'wing-f': 0.20,
    
    # Wide AMs
    'am-l': 0.20, 'am-r': 0.20, 'aml': 0.20, 'amr': 0.20,
    
    # Central mids
    'cm': 0.12, 'mc': 0.12, 'midfielder': 0.12, 'volante': 0.12,
    
    # Defensive mids
    'dm': 0.08, 'cdm': 0.08, 'pivot': 0.08,
    
    # Wing backs
    'wing-b': 0.06, 'lwb': 0.06, 'rwb': 0.06,
    
    # Fullbacks
    'lb': 0.04, 'rb': 0.04, 'fullback': 0.04, 'lateral': 0.04,
    
    # Centre backs
    'cd': 0.02, 'cb': 0.02, 'centreback': 0.02, 'central': 0.02,
    'defender': 0.02, 'defensa': 0.02, 'cd-l': 0.02, 'cd-r': 0.02,
    
    # Goalkeeper
    'g': 0.001, 'gk': 0.001, 'goalkeeper': 0.001, 'portero': 0.001, 'arquero': 0.001,
}

# Maximum lambda cap by position (prevents outliers from inflating too much)
LAMBDA_MAX_BY_POSITION = {
    # Forwards
    'f': 0.80, 'cf': 0.80, 'striker': 0.80, 'forward': 0.80, 'attacker': 0.80,
    'delantero': 0.80, 'punta': 0.80,
    
    # Attacking mids
    'am': 0.70, 'mediapunta': 0.70, 'ss': 0.70,
    
    # Wingers
    'lf': 0.65, 'rf': 0.65, 'lw': 0.65, 'rw': 0.65, 'winger': 0.65,
    'extremo': 0.65, 'wing-f': 0.60,
    
    # Wide AMs
    'am-l': 0.55, 'am-r': 0.55, 'aml': 0.55, 'amr': 0.55,
    
    # Central mids
    'cm': 0.35, 'mc': 0.35, 'midfielder': 0.35, 'volante': 0.35,
    
    # Defensive mids
    'dm': 0.25, 'cdm': 0.25, 'pivot': 0.25,
    
    # Wing backs
    'wing-b': 0.20, 'lwb': 0.20, 'rwb': 0.20,
    
    # Fullbacks
    'lb': 0.15, 'rb': 0.15, 'fullback': 0.15, 'lateral': 0.15,
    
    # Centre backs
    'cd': 0.10, 'cb': 0.10, 'centreback': 0.10, 'central': 0.10,
    'defender': 0.10, 'defensa': 0.10, 'cd-l': 0.10, 'cd-r': 0.10,
    
    # Goalkeeper
    'g': 0.02, 'gk': 0.02, 'goalkeeper': 0.02, 'portero': 0.02, 'arquero': 0.02,
}

# Hyperparameters for base lambda calculation
ALPHA_GOALS = 0.60   # Weight for goals component
BETA_SHOTS = 0.40    # Weight for shots component

# Small sample thresholds
N_MIN_MATCHES = 6           # Below this, apply shrinkage
SHRINK_FACTOR_BASE = 0.5    # Base shrink factor for very small samples


def map_position(position_str: str) -> str:
    """
    Map a position string to a canonical key for lookups.
    
    Handles variations like 'Forward', 'FWD', 'forward', 'Delantero', etc.
    Returns the first matching key from POSITION_WEIGHTS, or 'cm' as default.
    
    Args:
        position_str: Raw position string from data
        
    Returns:
        Canonical position key
    """
    if not position_str:
        return 'cm'
    
    pos_lower = position_str.lower().strip()
    
    # Direct match
    if pos_lower in POSITION_WEIGHTS:
        return pos_lower
    
    # Try partial matches (e.g., 'left winger' contains 'winger')
    for key in POSITION_WEIGHTS:
        if key in pos_lower or pos_lower in key:
            return key
    
    # Fallback based on common patterns
    if any(x in pos_lower for x in ['forward', 'delan', 'punta', 'striker', 'atac']):
        return 'f'
    elif any(x in pos_lower for x in ['winger', 'extremo', 'ala']):
        return 'lw'
    elif any(x in pos_lower for x in ['medio', 'midfield', 'volante']):
        if any(x in pos_lower for x in ['defens', 'pivot', 'cdm']):
            return 'dm'
        elif any(x in pos_lower for x in ['atac', 'offens']):
            return 'am'
        else:
            return 'cm'
    elif any(x in pos_lower for x in ['defens', 'back', 'central', 'zagu']):
        if any(x in pos_lower for x in ['lateral', 'wing-back', 'fullback']):
            return 'lb'
        else:
            return 'cd'
    elif any(x in pos_lower for x in ['porter', 'keep', 'gol', 'arqu']):
        return 'g'
    
    # Default to central midfielder (conservative)
    return 'cm'


def compute_base_lambda(
    goals_recent: int,
    minutes: float,
    shots_per_90: float,
    alpha: float = ALPHA_GOALS,
    beta: float = BETA_SHOTS,
) -> float:
    """
    Compute base lambda from goals and shots rate.
    
    Formula:
        rate_goals_90 = (goals_recent / max(minutes, 1)) * 90
        base_lambda = alpha * rate_goals_90 + beta * shots_per_90
    
    Args:
        goals_recent: Goals in recent period
        minutes: Minutes played in recent period
        shots_per_90: Shots per 90 minutes
        alpha: Weight for goals component
        beta: Weight for shots component
        
    Returns:
        Base lambda (unadjusted by position)
    """
    rate_goals_90 = (goals_recent / max(minutes, 1.0)) * 90.0
    base_lambda = alpha * rate_goals_90 + beta * shots_per_90
    return base_lambda


def apply_position_weight(base_lambda: float, position_key: str) -> float:
    """
    Apply position weight to base lambda.
    
    Formula:
        lambda_pos = base_lambda * POSITION_WEIGHTS[position]
    
    Args:
        base_lambda: Raw lambda from goals/shots
        position_key: Canonical position key
        
    Returns:
        Position-weighted lambda
    """
    weight = POSITION_WEIGHTS.get(position_key, 0.50)
    return base_lambda * weight


def apply_shrinkage(
    lambda_pos: float,
    position_key: str,
    matches: int,
    n_min: int = N_MIN_MATCHES,
    shrink_base: float = SHRINK_FACTOR_BASE,
) -> float:
    """
    Apply Bayesian shrinkage toward position prior for small samples.
    
    Formula:
        if matches < n_min:
            shrink_factor = matches / n_min  # Linear scaling [0, 1]
            lambda_shrunk = prior + shrink_factor * (lambda_pos - prior)
        else:
            lambda_shrunk = lambda_pos
    
    Args:
        lambda_pos: Position-weighted lambda
        position_key: Canonical position key
        matches: Number of matches played
        n_min: Minimum matches threshold
        shrink_base: Base shrink factor
        
    Returns:
        Shrunk lambda
    """
    if matches >= n_min:
        return lambda_pos
    
    # Linear shrink factor: 0 at 0 matches, 1 at n_min matches
    shrink_factor = matches / n_min
    
    prior = PRIOR_BY_POSITION.get(position_key, 0.10)
    
    # Blend toward prior
    lambda_shrunk = prior + shrink_factor * (lambda_pos - prior)
    
    return lambda_shrunk


def apply_lambda_cap(lambda_shrunk: float, position_key: str) -> float:
    """
    Apply hard cap on lambda by position.
    
    Args:
        lambda_shrunk: Shrunk lambda value
        position_key: Canonical position key
        
    Returns:
        Capped lambda
    """
    lambda_max = LAMBDA_MAX_BY_POSITION.get(position_key, 0.50)
    return min(lambda_shrunk, lambda_max)


def normalize_by_team_lambda(
    player_lambdas: List[Dict[str, Any]],
    team_lambda: float,
    target_ratio_min: float = 0.6,
    target_ratio_max: float = 1.1,
) -> List[Dict[str, Any]]:
    """
    Normalize player lambdas so their sum stays within reasonable range of team_lambda.
    
    Only scales DOWN if sum exceeds target_ratio_max * team_lambda.
    Does NOT scale up poor cases (preserves signal).
    
    Args:
        player_lambdas: List of dicts with 'lambda_final' field
        team_lambda: Team expected goals
        target_ratio_min: Min ratio of sum/team_lambda (not enforced upward)
        target_ratio_max: Max ratio of sum/team_lambda (enforced downward)
        
    Returns:
        List of dicts with updated 'lambda_final' values
    """
    if not player_lambdas:
        return player_lambdas
    
    sum_raw = sum(p.get('lambda_final', 0) for p in player_lambdas)
    
    if sum_raw <= 0:
        return player_lambdas
    
    # Only scale down if sum exceeds max ratio
    max_allowed = target_ratio_max * team_lambda
    
    if sum_raw <= max_allowed:
        # No scaling needed
        return player_lambdas
    
    # Scale factor to bring sum within bounds
    scale_factor = max_allowed / sum_raw
    
    for p in player_lambdas:
        p['lambda_final'] = p.get('lambda_final', 0) * scale_factor
    
    return player_lambdas


def compute_player_lambda(
    position: str,
    goals_recent: int,
    minutes: float,
    matches: int,
    shots_per_90: float,
    team_lambda: float,
    shots_total: float = None,
) -> Dict[str, Any]:
    """
    Compute final player lambda using the full pipeline.
    
    Pipeline:
    1. Map position to canonical key
    2. Compute base lambda from goals/90 and shots/90
    3. Apply position weight
    4. Apply shrinkage for small samples
    5. Apply position cap
    6. (Later) Normalize by team lambda
    
    Special handling:
    - Goalkeepers with 0 goals get lambda ≈ 0
    - Defenders without recent goals capped very low
    
    Args:
        position: Player position string
        goals_recent: Recent goals count
        minutes: Minutes played
        matches: Matches played
        shots_per_90: Shots per 90 minutes
        team_lambda: Team expected goals (for context)
        shots_total: Total shots (optional, for validation)
        
    Returns:
        Dict with all intermediate values and final lambda
    """
    # Step 1: Map position
    position_key = map_position(position)
    
    # Handle edge case: goalkeeper with no goals
    is_goalkeeper = position_key in ['g', 'gk', 'goalkeeper', 'portero', 'arquero']
    if is_goalkeeper and goals_recent == 0:
        return {
            'position_key': position_key,
            'is_goalkeeper': True,
            'base_lambda': 0.0,
            'lambda_pos': 0.0,
            'lambda_shrunk': 0.0,
            'lambda_capped': 0.0,
            'lambda_final': 0.0,
            'prior_used': PRIOR_BY_POSITION.get(position_key, 0.001),
            'cap_used': LAMBDA_MAX_BY_POSITION.get(position_key, 0.02),
        }
    
    # Step 2: Compute base lambda
    base_lambda = compute_base_lambda(goals_recent, minutes, shots_per_90)
    
    # Step 3: Apply position weight
    lambda_pos = apply_position_weight(base_lambda, position_key)
    
    # Step 4: Apply shrinkage for small samples
    lambda_shrunk = apply_shrinkage(lambda_pos, position_key, matches)
    
    # Step 5: Apply cap
    lambda_capped = apply_lambda_cap(lambda_shrunk, position_key)
    
    # Lambda_final will be set after team normalization
    result = {
        'position_key': position_key,
        'is_goalkeeper': is_goalkeeper,
        'base_lambda': round(base_lambda, 4),
        'lambda_pos': round(lambda_pos, 4),
        'lambda_shrunk': round(lambda_shrunk, 4),
        'lambda_capped': round(lambda_capped, 4),
        'lambda_final': round(lambda_capped, 4),  # Will be updated by normalization
        'prior_used': round(PRIOR_BY_POSITION.get(position_key, 0.10), 4),
        'cap_used': round(LAMBDA_MAX_BY_POSITION.get(position_key, 0.50), 4),
        'matches': matches,
        'goals_recent': goals_recent,
        'minutes': minutes,
        'shots_per_90': round(shots_per_90, 2),
    }
    
    return result


def compute_all_player_lambdas(
    players_data: List[Dict[str, Any]],
    team_lambda: float,
) -> List[Dict[str, Any]]:
    """
    Compute lambdas for all players in a team and normalize by team_lambda.
    
    Args:
        players_data: List of dicts with player stats
        team_lambda: Team expected goals
        
    Returns:
        List of dicts with computed lambdas and probabilities
    """
    results = []
    
    for pdata in players_data:
        # Extract fields with defaults
        position = pdata.get('position', '') or ''
        goals = pdata.get('goals', 0) or 0
        matches = pdata.get('matches_played', 1) or 1
        minutes_raw = pdata.get('minutes', 0) or 0
        starts = pdata.get('starts', 0) or 0
        shots = pdata.get('shots', 0) or 0
        
        # Estimate minutes if missing
        if minutes_raw <= 0:
            subs = max(0, matches - starts)
            minutes = starts * 75 + subs * 25
        else:
            minutes = float(minutes_raw)
        
        # Compute shots per 90
        shots_per_90 = (shots / max(matches, 1)) * (90.0 / max(minutes, 1.0)) if minutes > 0 else 0.0
        
        # Compute player lambda
        player_result = compute_player_lambda(
            position=position,
            goals_recent=goals,
            minutes=minutes,
            matches=matches,
            shots_per_90=shots_per_90,
            team_lambda=team_lambda,
            shots_total=shots,
        )
        
        # Add metadata
        player_result['player_name'] = pdata.get('player_name', 'Unknown')
        player_result['team'] = pdata.get('team', '')
        
        results.append(player_result)
    
    # Normalize by team lambda
    results = normalize_by_team_lambda(results, team_lambda)
    
    # Convert to probabilities
    for r in results:
        lambda_final = r.get('lambda_final', 0)
        anytime_prob = 1.0 - math.exp(-lambda_final)
        r['anytime_prob'] = round(anytime_prob, 4)
        r['anytime_prob_pct'] = round(anytime_prob * 100, 2)
    
    # Sort by probability descending
    results.sort(key=lambda x: x['anytime_prob'], reverse=True)
    
    return results


def validate_player_lambdas(
    player_results: List[Dict[str, Any]],
    team_lambda: float,
    team_name: str = '',
) -> Dict[str, Any]:
    """
    Validate computed player lambdas against constraints.
    
    Checks:
    - Sum of anytime probs vs team_lambda
    - Goalkeeper prob ≈ 0
    - Defenders without goals < 10%
    - Forwards with goals in reasonable range
    
    Args:
        player_results: List of computed player results
        team_lambda: Team expected goals
        team_name: Team name for logging
        
    Returns:
        Validation report dict
    """
    validation = {
        'team_name': team_name,
        'team_lambda': team_lambda,
        'num_players': len(player_results),
        'sum_anytime_prob': 0.0,
        'issues': [],
    }
    
    sum_probs = 0.0
    
    for r in player_results:
        prob = r.get('anytime_prob', 0)
        sum_probs += prob
        
        pos = r.get('position_key', '')
        goals = r.get('goals_recent', 0)
        is_gk = r.get('is_goalkeeper', False)
        
        # Check goalkeeper
        if is_gk and prob > 0.01:
            validation['issues'].append(
                f"Goalkeeper {r.get('player_name')} has high prob {prob:.4f}"
            )
        
        # Check defenders without goals
        if pos in ['cd', 'cb', 'defender', 'central', 'defensa'] and goals == 0:
            if prob > 0.10:
                validation['issues'].append(
                    f"Defender {r.get('player_name')} without goals has high prob {prob:.4f}"
                )
    
    validation['sum_anytime_prob'] = round(sum_probs, 4)
    
    # Check sum vs team_lambda
    ratio = sum_probs / team_lambda if team_lambda > 0 else 0
    validation['sum_to_lambda_ratio'] = round(ratio, 4)
    
    if ratio > 1.1:
        validation['issues'].append(
            f"Sum of probs ({sum_probs:.3f}) exceeds 110% of team_lambda ({team_lambda:.2f})"
        )
    elif ratio < 0.6:
        validation['issues'].append(
            f"Sum of probs ({sum_probs:.3f}) below 60% of team_lambda ({team_lambda:.2f})"
        )
    
    return validation
