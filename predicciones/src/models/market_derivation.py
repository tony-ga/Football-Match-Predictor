"""
Market derivation module: derives all betting markets from a score probability matrix.

Pipeline order (per user spec):
    Score Matrix → Market Derivation → Calibration → Sanity (warnings only)

All markets are derived analytically from the matrix — no hardcoded probabilities.

Markets covered:
- 1X2 (home win / draw / away win)
- BTTS (both teams to score)
- Over/Under (1.5, 2.5, 3.5 goals)
- Team totals (home/away over 0.5, 1.5, 2.5)
- Clean sheets (home / away)
- Correct scores (top N most probable)
- Halftime 1X2 (via dedicated halftime submodel)
"""
from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple, Any

import numpy as np

logger = logging.getLogger(__name__)
__all__ = []


# ---------------------------------------------------------------------------
# Core market derivations from score matrix
# ---------------------------------------------------------------------------

def derive_1x2(matrix: np.ndarray) -> Dict[str, float]:
    """
    Derive 1X2 probabilities from score matrix.

    P(home win) = sum of P(i,j) where i > j
    P(draw)     = sum of P(i,j) where i == j
    P(away win) = sum of P(i,j) where i < j

    Returns:
        Dict with 'home', 'draw', 'away' probabilities summing to 1.
    """
    p_home = float(np.sum(np.tril(matrix, k=-1)))  # i > j → home gana (abajo diagonal)
    p_draw = float(np.sum(np.diag(matrix)))
    p_away = float(np.sum(np.triu(matrix, k=1)))   # j > i → away gana (arriba diagonal)

    total = p_home + p_draw + p_away
    if total <= 0:
        raise ValueError("Score matrix sums to zero — check lambda inputs.")
    if total < 0.999:
        logger.warning(f"Matrix sum = {total:.4f} (truncation); renormalizing 1X2")

    return {
        'home': float(p_home / total),
        'draw': float(p_draw / total),
        'away': float(p_away / total),
    }


def derive_double_chance(one_x_two: Dict[str, float]) -> Dict[str, float]:
    """
    Derive Double Chance probabilities from 1X2 results.

    P(Home or Draw) = P(Home) + P(Draw)
    P(Away or Draw) = P(Away) + P(Draw)
    P(Home or Away) = P(Home) + P(Away)

    Returns:
        Dict with 'home_or_draw', 'away_or_draw', 'home_or_away' probabilities.
    """
    return {
        'home_or_draw': float(one_x_two['home'] + one_x_two['draw']),
        'away_or_draw': float(one_x_two['away'] + one_x_two['draw']),
        'home_or_away': float(one_x_two['home'] + one_x_two['away']),
    }


def derive_btts(matrix: np.ndarray) -> Dict[str, float]:
    """
    Derive Both Teams To Score (BTTS) probabilities.

    Formula:
        BTTS_yes = 1 - P(home=0) - P(away=0) + P(0,0)

    This correctly uses inclusion-exclusion to avoid double-counting P(0,0).

    Returns:
        Dict with 'yes' and 'no' probabilities summing to 1.
    """
    # P(home scores 0): sum over all j where i=0
    p_home_0 = float(matrix[0, :].sum())
    # P(away scores 0): sum over all i where j=0
    p_away_0 = float(matrix[:, 0].sum())
    # P(0-0)
    p_00 = float(matrix[0, 0])

    # Inclusion-exclusion: P(home=0 OR away=0) = P(home=0) + P(away=0) - P(0,0)
    p_at_least_one_blank = p_home_0 + p_away_0 - p_00
    p_btts_yes = 1.0 - p_at_least_one_blank

    p_btts_yes = float(np.clip(p_btts_yes, 0.0, 1.0))
    p_btts_no = 1.0 - p_btts_yes

    return {'yes': p_btts_yes, 'no': p_btts_no}


def derive_over_under(
    matrix: np.ndarray,
    threshold: float,
) -> Dict[str, float]:
    """
    Derive over/under probabilities for a given goal threshold.

    P(over N.5) = sum of P(i,j) where i+j > N.5
    P(under N.5) = sum of P(i,j) where i+j < N.5

    Note: threshold is a half-integer (e.g., 2.5), so i+j is never equal to threshold.

    Args:
        matrix: Score probability matrix.
        threshold: Goal line (e.g., 1.5, 2.5, 3.5).

    Returns:
        Dict with 'over' and 'under' probabilities summing to 1.
    """
    max_goals = matrix.shape[0] - 1
    p_over = 0.0

    for i in range(max_goals + 1):
        for j in range(max_goals + 1):
            if i + j > threshold:
                p_over += matrix[i, j]

    p_over = float(np.clip(p_over, 0.0, 1.0))
    p_under = 1.0 - p_over

    return {'over': p_over, 'under': p_under}


def derive_all_over_under(matrix: np.ndarray) -> Dict[str, float]:
    """
    Derive over/under for 1.5, 2.5 and 3.5 goal lines.

    Returns flat dict: over_1_5, under_1_5, over_2_5, under_2_5, over_3_5, under_3_5
    """
    result = {}
    for threshold in [1.5, 2.5, 3.5, 4.5]:
        key = str(threshold).replace('.', '_')
        ou = derive_over_under(matrix, threshold)
        result[f'over_{key}'] = ou['over']
        result[f'under_{key}'] = ou['under']
    return result


def derive_clean_sheets(matrix: np.ndarray) -> Dict[str, float]:
    """
    Derive clean sheet probabilities from the score matrix.

    CS_home = P(away scores 0) = sum_k P(k, 0)   [home keeps clean sheet]
    CS_away = P(home scores 0) = sum_k P(0, k)   [away keeps clean sheet]

    Returns:
        Dict with 'home' (home team keeps CS) and 'away' (away team keeps CS).
    """
    # Home team keeps clean sheet: away scores 0
    cs_home = float(matrix[:, 0].sum())  # sum over all i of P(i, 0)
    # Away team keeps clean sheet: home scores 0
    cs_away = float(matrix[0, :].sum())  # sum over all j of P(0, j)

    return {
        'home': float(np.clip(cs_home, 0.0, 1.0)),
        'away': float(np.clip(cs_away, 0.0, 1.0)),
    }


def derive_team_totals(matrix: np.ndarray) -> Dict[str, float]:
    """
    Derive team total goal probabilities (marginals).

    For each team: P(scores exactly k goals) and P(scores over k.5 goals).

    Returns:
        Dict with over_0_5, over_1_5, over_2_5 for home and away.
    """
    # Home marginal: P(home = i) = sum_j P(i, j)
    home_marginal = matrix.sum(axis=1)  # shape (max_goals+1,)
    # Away marginal: P(away = j) = sum_i P(i, j)
    away_marginal = matrix.sum(axis=0)  # shape (max_goals+1,)

    result = {}
    for threshold in [0.5, 1.5, 2.5]:
        key = str(threshold).replace('.', '_')
        home_over = float(home_marginal[int(np.ceil(threshold)):].sum())
        away_over = float(away_marginal[int(np.ceil(threshold)):].sum())
        result[f'home_over_{key}'] = float(np.clip(home_over, 0.0, 1.0))
        result[f'away_over_{key}'] = float(np.clip(away_over, 0.0, 1.0))

    return result


def derive_correct_scores(
    matrix: np.ndarray,
    top_n: int = 10,
) -> List[Dict[str, Any]]:
    """
    Derive the most probable correct scores.

    Returns:
        List of dicts sorted by probability descending:
        [{'score': '2-1', 'home': 2, 'away': 1, 'probability': 0.145}, ...]

    The last entry is always 'other' containing the residual probability.
    """
    max_goals = matrix.shape[0] - 1
    scores = []

    for i in range(max_goals + 1):
        for j in range(max_goals + 1):
            scores.append({
                'score': f'{i}-{j}',
                'home': i,
                'away': j,
                'probability': float(matrix[i, j]),
            })

    # Sort by probability descending
    scores.sort(key=lambda x: x['probability'], reverse=True)

    top_scores = scores[:top_n]
    covered_prob = sum(s['probability'] for s in top_scores)
    other_prob = float(np.clip(1.0 - covered_prob, 0.0, 1.0))

    top_scores.append({
        'score': 'other',
        'home': -1,
        'away': -1,
        'probability': other_prob,
    })

    return top_scores


def derive_goal_distribution(
    matrix: np.ndarray,
    team: str = 'home',
    max_display: int = 5,
) -> List[Dict[str, Any]]:
    """
    Derive probability distribution over goal totals for a single team.

    Args:
        matrix: Score matrix.
        team: 'home' or 'away'.
        max_display: Number of goal values to show explicitly (rest grouped as 'N+').
    """
    if team == 'home':
        marginal = matrix.sum(axis=1)
    else:
        marginal = matrix.sum(axis=0)

    result = []
    for k in range(min(max_display, len(marginal))):
        result.append({'goals': k, 'probability': float(marginal[k])})

    # Group remaining
    if len(marginal) > max_display:
        rest = float(marginal[max_display:].sum())
        result.append({'goals': f'{max_display}+', 'probability': rest})

    return result


# ---------------------------------------------------------------------------
# Halftime Submodel
# ---------------------------------------------------------------------------

def derive_halftime(
    lambda_home: float,
    lambda_away: float,
    config: Optional[Dict[str, Any]] = None,
    max_goals: int = 6,
) -> Dict[str, float]:
    """
    Estimate halftime 1X2 probabilities using a dedicated submodel.

    Rationale: Simply using lambda/2 is too crude. International data shows
    HT goals are ~43-47% of FT goals, and the distribution differs.
    We use a calibrated fraction from config with its own Poisson matrix.

    Args:
        lambda_home: Full-time expected goals for home.
        lambda_away: Full-time expected goals for away.
        config: Model configuration dict.
        max_goals: Max goals per team in HT (smaller than FT).

    Returns:
        Dict with 'home', 'draw', 'away' halftime probabilities.
    """
    if config is None:
        config = {}

    ht_fraction = config.get('halftime', {}).get('lambda_ht_fraction', 0.45)

    lambda_ht_h = float(np.clip(lambda_home * ht_fraction, 0.05, 4.0))
    lambda_ht_a = float(np.clip(lambda_away * ht_fraction, 0.05, 4.0))

    # Use simple Poisson for HT (DC correction less critical for sub-match periods)
    from scipy.stats import poisson as poisson_dist

    goals = np.arange(max_goals + 1)
    p_h = poisson_dist.pmf(goals, lambda_ht_h)
    p_a = poisson_dist.pmf(goals, lambda_ht_a)
    ht_matrix = np.outer(p_h, p_a)
    ht_matrix /= ht_matrix.sum()

    p_home_win = float(np.sum(np.tril(ht_matrix, k=-1)))
    p_ht_draw = float(np.sum(np.diag(ht_matrix)))
    p_away_win = float(np.sum(np.triu(ht_matrix, k=1)))

    total = p_home_win + p_ht_draw + p_away_win
    if total > 0:
        p_home_win /= total
        p_ht_draw /= total
        p_away_win /= total

    # Calibration adjustment: HT draws are slightly more common than FT model suggests
    # Empirical correction: boost draw by ~3-5% in low-scoring games
    if lambda_ht_h + lambda_ht_a < 1.5:
        draw_boost = 0.04
        excess = draw_boost / 2
        p_ht_draw = min(p_ht_draw + draw_boost, 0.60)
        p_home_win = max(p_home_win - excess, 0.05)
        p_away_win = max(p_away_win - excess, 0.05)
        # Renormalize
        total = p_home_win + p_ht_draw + p_away_win
        p_home_win /= total
        p_ht_draw /= total
        p_away_win /= total

    return {
        'home': float(p_home_win),
        'draw': float(p_ht_draw),
        'away': float(p_away_win),
        'lambda_ht_home': lambda_ht_h,
        'lambda_ht_away': lambda_ht_a,
    }


# ---------------------------------------------------------------------------
# Full Market Bundle
# ---------------------------------------------------------------------------

def derive_all_markets(
    matrix: np.ndarray,
    lambda_home: float,
    lambda_away: float,
    config: Optional[Dict[str, Any]] = None,
    top_correct_scores: int = 10,
) -> Dict[str, Any]:
    """
    Derive all markets from a score matrix in a single call.

    Args:
        matrix: Dixon-Coles score probability matrix.
        lambda_home: Expected goals home (for HT submodel).
        lambda_away: Expected goals away (for HT submodel).
        config: Model config dict.
        top_correct_scores: Number of top correct scores to return.

    Returns:
        Complete markets dict with all derived probabilities.
    """
    markets = {}

    # 1X2
    markets['1x2'] = derive_1x2(matrix)
    
    # Double Chance
    markets['double_chance'] = derive_double_chance(markets['1x2'])

    # BTTS
    markets['btts'] = derive_btts(matrix)

    # Over/Under
    markets['over_under'] = derive_all_over_under(matrix)

    # Clean sheets
    markets['clean_sheets'] = derive_clean_sheets(matrix)

    # Team totals
    markets['team_totals'] = derive_team_totals(matrix)

    # Correct scores
    markets['correct_scores'] = derive_correct_scores(matrix, top_n=top_correct_scores)

    # Halftime
    markets['halftime'] = derive_halftime(lambda_home, lambda_away, config=config)

    # Goal distributions (marginals)
    markets['home_goals_distribution'] = derive_goal_distribution(matrix, 'home')
    markets['away_goals_distribution'] = derive_goal_distribution(matrix, 'away')

    # Expected goals (sanity reference)
    max_g = matrix.shape[0]
    goals = np.arange(max_g)
    home_marginal = matrix.sum(axis=1)
    away_marginal = matrix.sum(axis=0)
    markets['expected_goals'] = {
        'home': float(np.dot(goals, home_marginal[:len(goals)])),
        'away': float(np.dot(goals, away_marginal[:len(goals)])),
        'total': float(np.dot(goals, home_marginal[:len(goals)]) +
                       np.dot(goals, away_marginal[:len(goals)])),
    }

    logger.debug(
        f"Markets derived: 1X2={markets['1x2']}, "
        f"BTTS={markets['btts']}, "
        f"xG=({lambda_home:.2f}, {lambda_away:.2f})"
    )
    return markets

# ---------------------------------------------------------------------------
# SGP Builder Market Extraction & Derivation
# ---------------------------------------------------------------------------

def derive_goal_markets(pred_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Extract probabilities for goal/result/totals/BTTS markets from poisson/lambda data.
    Returns a list of dicts: {"market_key": str, "probability": float}
    """
    markets = pred_data.get('predictions', {})
    if not markets:
        return []

    candidates = []
    
    # 1X2
    if '1x2' in markets:
        candidates.append({"market_key": "1x2_home", "probability": markets['1x2'].get('home', 0.0)})
        candidates.append({"market_key": "1x2_draw", "probability": markets['1x2'].get('draw', 0.0)})
        candidates.append({"market_key": "1x2_away", "probability": markets['1x2'].get('away', 0.0)})
        
    # Double Chance
    if 'double_chance' in markets:
        candidates.append({"market_key": "double_chance_home_or_draw", "probability": markets['double_chance'].get('home_or_draw', 0.0)})
        candidates.append({"market_key": "double_chance_away_or_draw", "probability": markets['double_chance'].get('away_or_draw', 0.0)})
        candidates.append({"market_key": "double_chance_home_or_away", "probability": markets['double_chance'].get('home_or_away', 0.0)})

    # Totals
    if 'over_under' in markets:
        ou = markets['over_under']
        candidates.append({"market_key": "over_1_5", "probability": ou.get('over_1_5', 0.0)})
        candidates.append({"market_key": "over_2_5", "probability": ou.get('over_2_5', 0.0)})
        candidates.append({"market_key": "over_3_5", "probability": ou.get('over_3_5', 0.0)})
        candidates.append({"market_key": "over_4_5", "probability": ou.get('over_4_5', 0.0)})
        candidates.append({"market_key": "under_1_5", "probability": ou.get('under_1_5', 0.0)})
        candidates.append({"market_key": "under_2_5", "probability": ou.get('under_2_5', 0.0)})
        candidates.append({"market_key": "under_3_5", "probability": ou.get('under_3_5', 0.0)})
        candidates.append({"market_key": "under_4_5", "probability": ou.get('under_4_5', 0.0)})

    # BTTS
    if 'btts' in markets:
        candidates.append({"market_key": "btts_yes", "probability": markets['btts'].get('yes', 0.0)})
        candidates.append({"market_key": "btts_no", "probability": markets['btts'].get('no', 0.0)})

    return candidates


def derive_corner_markets(pred_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Derive corner markets based on historical averages and dispersion.
    Expects pred_data['corners'] to have team averages.
    If missing or insufficient, returns empty list (graceful degradation).
    """
    corner_data = pred_data.get('corners')
    if not corner_data:
        return []
        
    try:
        home_for = corner_data['home_avg_for']
        home_against = corner_data['home_avg_against']
        away_for = corner_data['away_avg_for']
        away_against = corner_data['away_avg_against']
    except KeyError:
        return []
        
    expected_home_corners = (home_for + away_against) / 2
    expected_away_corners = (away_for + home_against) / 2
    expected_total = expected_home_corners + expected_away_corners
    
    import scipy.stats as stats
    std_dev = corner_data.get('total_std_dev', expected_total ** 0.5)
    
    candidates = []
    
    def get_over_prob(line: float) -> float:
        return 1.0 - stats.norm.cdf(line, loc=expected_total, scale=std_dev)
        
    def get_under_prob(line: float) -> float:
        return stats.norm.cdf(line, loc=expected_total, scale=std_dev)
        
    candidates.append({"market_key": "corners_over_6_5", "probability": get_over_prob(6.5)})
    candidates.append({"market_key": "corners_over_7_5", "probability": get_over_prob(7.5)})
    candidates.append({"market_key": "corners_over_8_5", "probability": get_over_prob(8.5)})
    candidates.append({"market_key": "corners_over_9_5", "probability": get_over_prob(9.5)})
    
    candidates.append({"market_key": "corners_under_10_5", "probability": get_under_prob(10.5)})
    candidates.append({"market_key": "corners_under_8_5", "probability": get_under_prob(8.5)})
    
    return candidates


def derive_player_shot_markets(pred_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Derive player shot markets based on individual stats.
    Expects pred_data['player_stats'] as a list of players.
    If missing, returns empty list.
    """
    player_data = pred_data.get('player_stats')
    if not player_data:
        return []
        
    import scipy.stats as stats
    candidates = []
    
    for player in player_data:
        name = player.get('name')
        shots_p90 = player.get('shots_p90')
        exp_mins = player.get('expected_minutes', 90)
        
        if not name or not shots_p90:
            continue
            
        expected_shots = shots_p90 * (exp_mins / 90.0)
        std_dev = player.get('shots_std_dev', max(1.0, expected_shots ** 0.5))
        
        safe_name = name.lower().replace(" ", "_")
        
        prob_over_1_5 = 1.0 - stats.norm.cdf(1.5, loc=expected_shots, scale=std_dev)
        if prob_over_1_5 >= 0.2:
            candidates.append({
                "market_key": f"player_shots_over_1_5_{safe_name}",
                "template_key": "player_shots_over_1_5", 
                "name_override": f"{name} Over 1.5 Shots",
                "probability": float(prob_over_1_5)
            })
            
        prob_over_2_5 = 1.0 - stats.norm.cdf(2.5, loc=expected_shots, scale=std_dev)
        if prob_over_2_5 >= 0.2:
            candidates.append({
                "market_key": f"player_shots_over_2_5_{safe_name}",
                "template_key": "player_shots_over_2_5",
                "name_override": f"{name} Over 2.5 Shots",
                "probability": float(prob_over_2_5)
            })
            
        prob_over_3_5 = 1.0 - stats.norm.cdf(3.5, loc=expected_shots, scale=std_dev)
        if prob_over_3_5 >= 0.2:
            candidates.append({
                "market_key": f"player_shots_over_3_5_{safe_name}",
                "template_key": "player_shots_over_3_5",
                "name_override": f"{name} Over 3.5 Shots",
                "probability": float(prob_over_3_5)
            })
            
    return candidates

# ---------------------------------------------------------------------------
# Joint Parlay Probability Utilities (Goal‑derived markets only)
# ---------------------------------------------------------------------------
from typing import List, Tuple
import numpy as np

def _mask_from_condition(matrix: np.ndarray, condition: Tuple[str, any]) -> np.ndarray:
    """Return a boolean mask for a single market condition.

    Supported keys: 1x2_home, 1x2_draw, 1x2_away, double_chance_*, btts_*, over_*/under_*, correct_score_*
    The value part is ignored for boolean markets.
    """
    key, _ = condition
    max_goals = matrix.shape[0] - 1
    i = np.arange(max_goals + 1)[:, None]
    j = np.arange(max_goals + 1)[None, :]
    if key == '1x2_home':
        return i > j
    if key == '1x2_draw':
        return i == j
    if key == '1x2_away':
        return i < j
    if key == 'double_chance_home_or_draw':
        return (i > j) | (i == j)
    if key == 'double_chance_away_or_draw':
        return (i < j) | (i == j)
    if key == 'double_chance_home_or_away':
        return (i > j) | (i < j)
    if key == 'btts_yes':
        return (i > 0) & (j > 0)
    if key == 'btts_no':
        return (i == 0) | (j == 0)
    if key.startswith('over_') or key.startswith('under_'):
        parts = key.split('_')
        threshold = float(parts[1] + '.' + parts[2])
        total = i + j
        return (total > threshold) if key.startswith('over') else (total < threshold)
    raise ValueError(f"Unsupported market condition: {key}")

def compute_joint_parlay_probability(
    matrix: np.ndarray,
    conditions: List[Tuple[str, any]],
    max_goals: int = 8,
) -> float:
    """Exact joint probability for a set of goal‑derived market conditions.

    The function builds a mask for each condition, ANDs them together and
    sums the probabilities of the remaining cells.
    """
    # Truncate matrix if larger than max_goals+1
    if matrix.shape[0] - 1 > max_goals:
        matrix = matrix[: max_goals + 1, : max_goals + 1]
    combined = np.ones_like(matrix, dtype=bool)
    for cond in conditions:
        combined &= _mask_from_condition(matrix, cond)
        if not combined.any():
            return 0.0
    joint = float(matrix[combined].sum())
    return max(0.0, min(1.0, joint))

# Export symbols
__all__.extend([
    'compute_joint_parlay_probability',
])
