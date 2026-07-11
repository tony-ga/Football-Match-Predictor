"""
National Team Rating System using Elo-like methodology.

This module implements an Elo-based rating system for national teams,
designed to replace the static ratings_wc2026.json approach.

Key features:
- Expected result based on rating difference
- Weighting by match importance
- Home advantage adjustment (if applicable)
- Goal difference moderation
- Shrinkage/prior for teams with limited data
- Regularization to prevent extreme gaps between similar-tier teams

Reference methodology:
- FIFA/Coca-Cola World Ranking
- ClubElo / WorldFootballRatings systems
- International football Elo variants
"""
from __future__ import annotations

import logging
import math
from typing import Dict, Any, Optional, Tuple, List
import numpy as np

logger = logging.getLogger(__name__)


# =============================================================================
# CONFIGURATION CONSTANTS
# =============================================================================

# Base K-factor for international matches (similar to FIFA ranking)
BASE_K_FACTOR = 25.0

# Match importance multipliers
IMPORTANCE_MULTIPLIERS = {
    'world_cup_final': 4.0,      # World Cup final
    'world_cup_knockout': 3.5,   # World Cup knockout stages
    'world_cup_group': 3.0,      # World Cup group stage
    'confederation_cup_final': 3.5,
    'confederation_cup_knockout': 3.0,
    'confederation_cup_group': 2.5,
    'qualifier': 2.5,            # World Cup qualifiers
    'friendly': 1.0,             # Friendly matches
    'default': 2.0,              # Default for unknown competitions
}

# Home advantage in Elo points (typically 50-100 points in chess, ~75 in football)
HOME_ADVANTAGE_POINTS = 75.0

# Rating floor and ceiling
MIN_RATING = 800.0
MAX_RATING = 2200.0

# Default rating for new/unrated teams
DEFAULT_RATING = 1200.0

# Prior strength for shrinkage (higher = more confidence in prior)
PRIOR_STRENGTH = 5.0  # Equivalent to ~5 matches of prior belief

# Maximum goal difference bonus multiplier
MAX_GOAL_DIFF_BONUS = 2.0

# Goal difference scaling factor
GOAL_DIFF_SCALE = 0.5

# Regularization: maximum allowed gap between top-30 teams without strong evidence
MAX_TIER_GAP = 150.0  # Points

# Blending weights for final team rating
WEIGHT_ELO = 0.60       # Elo rating weight
WEIGHT_FORM = 0.25      # Recent form weight
WEIGHT_MARKET = 0.15    # Market anchor (optional, can be set to 0)


# =============================================================================
# ELO EXPECTED RESULT FUNCTION
# =============================================================================

def expected_result_from_ratings(
    rating_a: float,
    rating_b: float,
    home_advantage_points: float = 0.0,
) -> float:
    """
    Calculate expected result for team A vs team B using Elo formula.
    
    Formula:
        E_A = 1 / (1 + 10^((rating_b - rating_a - home_adv) / 400))
    
    Args:
        rating_a: Rating of team A
        rating_b: Rating of team B
        home_advantage_points: Bonus points for home team (default 0)
    
    Returns:
        Expected score for team A (0.0 to 1.0)
        - 0.5 = equal teams
        - >0.5 = team A favored
        - <0.5 = team B favored
    """
    rating_diff = rating_a - rating_b + home_advantage_points
    expected = 1.0 / (1.0 + math.pow(10, -rating_diff / 400.0))
    return float(np.clip(expected, 0.01, 0.99))


def goal_difference_multiplier(
    goal_diff: int,
    max_bonus: float = MAX_GOAL_DIFF_BONUS,
    scale: float = GOAL_DIFF_SCALE,
) -> float:
    """
    Calculate goal difference bonus factor for K-factor adjustment.
    
    Formula inspired by World Football Ratings:
        GD_mult = ln(|GD| + 1) * (0.8 + 0.2 / |expected_margin|)
    
    Simplified version:
        GD_mult = scale * ln(|GD| + 1), capped at max_bonus
    
    Args:
        goal_diff: Actual goal difference (positive = win margin)
        max_bonus: Maximum multiplier
        scale: Scaling factor
    
    Returns:
        Multiplier >= 1.0
    """
    if goal_diff == 0:
        return 1.0
    
    abs_gd = abs(goal_diff)
    multiplier = scale * math.log(abs_gd + 1) + 1.0
    return float(min(multiplier, max_bonus))


def calculate_k_factor(
    base_k: float = BASE_K_FACTOR,
    importance: str = 'default',
    goal_diff: int = 0,
    is_upset: bool = False,
    upset_bonus: float = 0.2,
) -> float:
    """
    Calculate dynamic K-factor for rating update.
    
    Args:
        base_k: Base K-factor
        importance: Match importance category
        goal_diff: Goal difference
        is_upset: Whether result was unexpected (lower-rated team won/drew)
        upset_bonus: Extra K-factor multiplier for upsets
    
    Returns:
        Adjusted K-factor
    """
    k = base_k
    
    # Apply importance multiplier
    importance_mult = IMPORTANCE_MULTIPLIERS.get(importance, 1.0)
    k *= importance_mult
    
    # Apply goal difference multiplier
    gd_mult = goal_difference_multiplier(goal_diff)
    k *= gd_mult
    
    # Apply upset bonus
    if is_upset:
        k *= (1.0 + upset_bonus)
    
    return k


def update_elo_rating(
    current_rating: float,
    opponent_rating: float,
    actual_result: float,
    k_factor: float,
    home_advantage_points: float = 0.0,
) -> float:
    """
    Update Elo rating after a match.
    
    Formula:
        new_rating = old_rating + K * (actual_result - expected_result)
    
    Result values:
        1.0 = win
        0.5 = draw
        0.0 = loss
    
    Args:
        current_rating: Current team rating
        opponent_rating: Opponent team rating
        actual_result: Actual match result (1/0.5/0)
        k_factor: Dynamic K-factor
        home_advantage_points: Home advantage bonus (for expected calculation)
    
    Returns:
        Updated rating
    """
    expected = expected_result_from_ratings(
        current_rating, opponent_rating, home_advantage_points
    )
    new_rating = current_rating + k_factor * (actual_result - expected)
    return float(np.clip(new_rating, MIN_RATING, MAX_RATING))


# =============================================================================
# RATING BUILDING FROM HISTORICAL DATA
# =============================================================================

def build_national_team_rating(
    team_name: str,
    historical_results: List[Dict[str, Any]],
    prior_rating: float = DEFAULT_RATING,
    prior_strength: float = PRIOR_STRENGTH,
    include_form: bool = True,
    form_window_months: int = 24,
) -> Dict[str, Any]:
    """
    Build team rating from historical match results.
    
    Implements Bayesian shrinkage toward prior for teams with limited data.
    
    Args:
        team_name: Name of the team
        historical_results: List of match result dicts with keys:
            - date: Match date
            - opponent: Opponent team name
            - opponent_rating: Opponent rating (if known)
            - goals_for: Goals scored by this team
            - goals_against: Goals conceded
            - is_home: Whether team was at home
            - importance: Match importance category
        prior_rating: Starting rating for shrinkage
        prior_strength: Number of "prior matches" for shrinkage
        include_form: Whether to include recent form boost
        form_window_months: Window for form calculation
    
    Returns:
        Dict with:
            - elo_rating: Computed Elo rating
            - form_rating: Recent form rating (if available)
            - final_rating: Blended final rating
            - n_matches: Number of matches used
            - confidence: Confidence level [0, 1]
    """
    if not historical_results:
        # No data: return prior with low confidence
        return {
            'team_name': team_name,
            'elo_rating': prior_rating,
            'form_rating': None,
            'final_rating': prior_rating,
            'n_matches': 0,
            'confidence': 0.1,
            'shrinkage_applied': 1.0,
        }
    
    # Sort by date
    sorted_results = sorted(historical_results, key=lambda x: x.get('date', ''))
    
    # Initialize rating from prior
    rating = prior_rating
    total_k = 0.0
    
    # Process each match sequentially
    for match in sorted_results:
        opponent_rating = match.get('opponent_rating', prior_rating)
        goals_for = match.get('goals_for', 0)
        goals_against = match.get('goals_against', 0)
        is_home = match.get('is_home', False)
        importance = match.get('importance', 'default')
        
        # Determine result
        if goals_for > goals_against:
            actual_result = 1.0
        elif goals_for == goals_against:
            actual_result = 0.5
        else:
            actual_result = 0.0
        
        goal_diff = goals_for - goals_against
        
        # Check for upset
        home_adj = HOME_ADVANTAGE_POINTS if is_home else 0.0
        expected = expected_result_from_ratings(rating, opponent_rating, home_adj)
        is_upset = (actual_result > expected + 0.2)  # Significant upset
        
        # Calculate K-factor
        k = calculate_k_factor(
            base_k=BASE_K_FACTOR,
            importance=importance,
            goal_diff=goal_diff,
            is_upset=is_upset,
        )
        
        # Update rating
        rating = update_elo_rating(
            current_rating=rating,
            opponent_rating=opponent_rating,
            actual_result=actual_result,
            k_factor=k,
            home_advantage_points=home_adj,
        )
        
        total_k += k
    
    # Apply shrinkage toward prior for small samples
    n_matches = len(sorted_results)
    shrink_factor = n_matches / (n_matches + prior_strength)
    shrunk_rating = prior_rating + shrink_factor * (rating - prior_rating)
    
    # Calculate form rating (recent performance)
    form_rating = None
    if include_form and n_matches > 0:
        form_rating = _calculate_recent_form(
            sorted_results, 
            window_months=form_window_months,
            current_rating=shrunk_rating,
        )
    
    # Blend ratings
    if form_rating is not None:
        final_rating = WEIGHT_ELO * shrunk_rating + WEIGHT_FORM * form_rating
    else:
        final_rating = shrunk_rating
    
    # Confidence based on sample size
    confidence = min(1.0, n_matches / 20.0)  # Max confidence at 20+ matches
    
    return {
        'team_name': team_name,
        'elo_rating': round(shrunk_rating, 2),
        'form_rating': round(form_rating, 2) if form_rating else None,
        'final_rating': round(final_rating, 2),
        'n_matches': n_matches,
        'confidence': round(confidence, 3),
        'shrinkage_applied': round(1.0 - shrink_factor, 3),
    }


def _calculate_recent_form(
    results: List[Dict[str, Any]],
    window_months: int = 24,
    current_rating: float = DEFAULT_RATING,
) -> float:
    """
    Calculate recent form rating from last N months of results.
    
    Uses weighted average of results with temporal decay.
    """
    from datetime import datetime, timedelta
    
    if not results:
        return current_rating
    
    # Get latest date
    try:
        latest_date = max(
            datetime.fromisoformat(r.get('date', '2024-01-01').replace('Z', '+00:00'))
            for r in results
        )
    except (ValueError, TypeError):
        latest_date = datetime.now()
    
    cutoff_date = latest_date - timedelta(days=window_months * 30)
    
    # Filter recent results
    recent_results = []
    for r in results:
        try:
            match_date = datetime.fromisoformat(r.get('date', '2024-01-01').replace('Z', '+00:00'))
            if match_date >= cutoff_date:
                recent_results.append(r)
        except (ValueError, TypeError):
            continue
    
    if not recent_results:
        return current_rating
    
    # Calculate form score
    total_weight = 0.0
    weighted_score = 0.0
    
    for r in recent_results:
        # Temporal weight (more recent = higher)
        try:
            match_date = datetime.fromisoformat(r.get('date', '2024-01-01').replace('Z', '+00:00'))
            days_ago = (latest_date - match_date).days
        except:
            days_ago = 365
        
        time_weight = math.exp(-days_ago / 365.0)  # Decay over 1 year
        
        # Result score
        goals_for = r.get('goals_for', 0)
        goals_against = r.get('goals_against', 0)
        
        if goals_for > goals_against:
            result_score = 1.0
        elif goals_for == goals_against:
            result_score = 0.5
        else:
            result_score = 0.0
        
        # Goal difference bonus
        gd_bonus = min(0.2, 0.05 * (goals_for - goals_against))
        result_score += gd_bonus
        
        weighted_score += time_weight * result_score
        total_weight += time_weight
    
    if total_weight == 0:
        return current_rating
    
    avg_score = weighted_score / total_weight
    
    # Convert to rating scale (avg_score of 0.6 ≈ current_rating)
    # Scale: 0.4 → -100pts, 0.5 → 0pts, 0.6 → +100pts, 0.7 → +200pts
    form_adjustment = (avg_score - 0.5) * 500
    form_rating = current_rating + form_adjustment
    
    return float(np.clip(form_rating, MIN_RATING, MAX_RATING))


# =============================================================================
# TEAM STRENGTH TO LAMBDA CONVERSION
# =============================================================================

def team_strength_to_goal_lambdas(
    home_rating: float,
    away_rating: float,
    league_avg_goals: float = 2.5,
    home_advantage_log: float = 0.0,
    context_modifier_home: float = 0.0,
    context_modifier_away: float = 0.0,
    regularization_strength: float = 0.3,
    tier_gap_limit: float = MAX_TIER_GAP,
) -> Tuple[float, float]:
    """
    Convert team strength ratings to expected goals (lambdas).
    
    Uses a ratio-based approach with regularization to prevent extreme gaps.
    
    Args:
        home_rating: Home team rating
        away_rating: Away team rating
        league_avg_goals: Expected total goals in neutral match
        home_advantage_log: Log-scale home advantage
        context_modifier_home: Additional context for home team
        context_modifier_away: Additional context for away team
        regularization_strength: How much to shrink toward league average
        tier_gap_limit: Maximum rating gap to use for top-30 teams
    
    Returns:
        Tuple (lambda_home, lambda_away)
    """
    # Apply tier gap regularization for closely-matched teams
    rating_diff = home_rating - away_rating
    
    # For teams within similar tier, limit the effective gap
    if abs(rating_diff) < tier_gap_limit:
        # Apply soft regularization: shrink extreme differences
        shrink_factor = 1.0 - regularization_strength * (1.0 - abs(rating_diff) / tier_gap_limit)
        rating_diff *= shrink_factor
    
    # Calculate strength ratio
    # Using logistic function to map rating diff to goal ratio
    # At diff=0: ratio=1.0, at diff=100: ratio≈1.5, at diff=200: ratio≈2.0
    strength_ratio = math.pow(10, rating_diff / 400.0)
    
    # Base lambdas from total and ratio
    # lambda_h / lambda_a = strength_ratio
    # lambda_h + lambda_a = league_avg_goals
    lambda_a_base = league_avg_goals / (1.0 + math.sqrt(strength_ratio))
    lambda_h_base = league_avg_goals - lambda_a_base
    
    # Apply home advantage
    lambda_h = lambda_h_base * math.exp(home_advantage_log)
    lambda_a = lambda_a_base
    
    # Apply context modifiers
    lambda_h *= math.exp(context_modifier_home)
    lambda_a *= math.exp(context_modifier_away)
    
    # Clip to reasonable range
    lambda_h = float(np.clip(lambda_h, 0.1, 5.0))
    lambda_a = float(np.clip(lambda_a, 0.1, 5.0))
    
    logger.debug(
        f"Elo→Lambda: rating_diff={rating_diff:.1f}, strength_ratio={strength_ratio:.3f}, "
        f"lambda_h={lambda_h:.3f}, lambda_a={lambda_a:.3f}"
    )
    
    return lambda_h, lambda_a


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def get_initial_rating_from_fifa_rank(fifa_rank: int, max_rank: int = 211) -> float:
    """
    Derive initial Elo rating from FIFA ranking position.
    
    Rank 1 → ~2000 pts, Rank 50 → ~1400 pts, Rank 100 → ~1200 pts
    
    Args:
        fifa_rank: FIFA ranking position (1 = best)
        max_rank: Maximum rank considered
    
    Returns:
        Initial Elo rating
    """
    if fifa_rank <= 0:
        fifa_rank = 50
    
    # Linear interpolation with log adjustment
    # Top 10: 1800-2000, Top 50: 1400-1800, Rest: 1000-1400
    if fifa_rank <= 10:
        rating = 2000 - (fifa_rank - 1) * 20
    elif fifa_rank <= 50:
        rating = 1800 - (fifa_rank - 10) * 10
    else:
        rating = 1400 - min(400, (fifa_rank - 50) * 3)
    
    return float(np.clip(rating, MIN_RATING, MAX_RATING))


def blend_with_market_anchor(
    model_prob_home: float,
    market_prob_home: float,
    model_confidence: float,
    market_weight: float = WEIGHT_MARKET,
) -> float:
    """
    Blend model probability with market anchor.
    
    Args:
        model_prob_home: Model's home win probability
        market_prob_home: Market-implied home win probability
        model_confidence: Model confidence [0, 1]
        market_weight: Weight given to market (0 = ignore market)
    
    Returns:
        Blended probability
    """
    if market_weight <= 0:
        return model_prob_home
    
    # Effective market weight depends on model confidence
    effective_market_weight = market_weight * (1.0 - model_confidence * 0.5)
    
    blended = (
        (1.0 - effective_market_weight) * model_prob_home +
        effective_market_weight * market_prob_home
    )
    
    return float(np.clip(blended, 0.01, 0.99))


def validate_rating_system(
    test_cases: List[Dict[str, Any]],
    tolerance_1x2: float = 0.08,
) -> Dict[str, Any]:
    """
    Validate rating system against test cases.
    
    Args:
        test_cases: List of dicts with:
            - home_team, away_team
            - home_rating, away_rating
            - actual_result (1/X/2 or probabilities)
            - market_probs (optional)
        tolerance_1x2: Acceptable deviation from market
    
    Returns:
        Validation report dict
    """
    issues = []
    passed = 0
    failed = 0
    
    for case in test_cases:
        home_rating = case.get('home_rating', DEFAULT_RATING)
        away_rating = case.get('away_rating', DEFAULT_RATING)
        market_probs = case.get('market_probs', {})
        
        # Calculate expected result
        expected_home = expected_result_from_ratings(home_rating, away_rating)
        
        # Check against market
        if market_probs:
            market_home = market_probs.get('home', 0.5)
            deviation = abs(expected_home - market_home)
            
            if deviation > tolerance_1x2:
                issues.append({
                    'match': f"{case.get('home_team')} vs {case.get('away_team')}",
                    'issue': f'Deviation {deviation:.3f} exceeds tolerance {tolerance_1x2}',
                    'model_home': round(expected_home, 3),
                    'market_home': round(market_home, 3),
                })
                failed += 1
            else:
                passed += 1
        else:
            passed += 1
    
    return {
        'passed': passed,
        'failed': failed,
        'total': passed + failed,
        'issues': issues,
        'pass_rate': passed / (passed + failed) if (passed + failed) > 0 else 0,
    }
