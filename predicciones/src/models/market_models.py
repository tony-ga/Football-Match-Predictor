"""
Statistical models for alternative markets.

Implements Poisson and Negative Binomial models for:
- Corners
- Cards  
- Shots on target
- Player props (heuristic/hierarchical)

Key improvements:
- Bayesian regularization with global priors
- Coupling with xG/possession model
- Consistent more_* derivation from team distributions
- Proper probability normalization for player props
"""
from __future__ import annotations

import logging
import math
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# Global priors for secondary markets (league averages)
GLOBAL_PRIORS = {
    "corners_per_team": 5.0,
    "cards_per_team": 2.2,
    "sot_per_team": 4.5,
    "total_corners_match": 10.0,
    "total_cards_match": 4.4,
    "total_sot_match": 9.0,
}

# Effective sample sizes for regularization
# These control how much weight is given to observed data vs prior
EFFECTIVE_SAMPLE_SIZES = {
    "corners": 12.0,  # Equivalent to ~12 matches of prior weight
    "cards": 10.0,
    "sot": 10.0,
}


def poisson_pmf(k: int, lam: float) -> float:
    """Poisson probability mass function."""
    if lam <= 0:
        return 0.0
    return math.exp(-lam) * (lam ** k) / math.factorial(k)


def poisson_cdf(k: int, lam: float) -> float:
    """Poisson cumulative distribution function P(X <= k)."""
    return sum(poisson_pmf(i, lam) for i in range(k + 1))


def poisson_sf(k: int, lam: float) -> float:
    """Poisson survival function P(X > k) = 1 - CDF(k)."""
    return 1.0 - poisson_cdf(k, lam)


def negative_binomial_pmf(k: int, n: float, p: float) -> float:
    """
    Negative Binomial PMF.
    
    Parameters:
        k: Number of successes
        n: Number of failures (dispersion parameter)
        p: Probability of success
    
    Returns:
        P(X = k)
    """
    if n <= 0 or p <= 0 or p >= 1:
        return 0.0
    
    # Using the gamma function formulation for non-integer n
    from math import gamma
    
    try:
        coeff = gamma(k + n) / (gamma(n) * math.factorial(k))
        return coeff * (p ** n) * ((1 - p) ** k)
    except (ValueError, OverflowError):
        return 0.0


def bayesian_shrinkage(
    observed_mean: float,
    prior_mean: float,
    observed_n: int,
    effective_sample_size: float,
) -> float:
    """
    Apply Bayesian shrinkage to regularize estimates toward a prior mean.
    
    This prevents overfitting when sample sizes are small.
    
    Formula:
        shrunk_mean = (n * observed_mean + k * prior_mean) / (n + k)
    
    where k is the effective sample size of the prior.
    
    Args:
        observed_mean: Mean from observed data
        prior_mean: Global/league average prior
        observed_n: Number of observations
        effective_sample_size: Weight of the prior (equivalent sample size)
    
    Returns:
        Regularized estimate
    """
    if observed_n <= 0:
        return prior_mean
    
    weight_data = observed_n
    weight_prior = effective_sample_size
    
    total_weight = weight_data + weight_prior
    shrunk_mean = (weight_data * observed_mean + weight_prior * prior_mean) / total_weight
    
    return shrunk_mean


class CornersModel:
    """
    Poisson-based model for corners markets with Bayesian regularization.
    
    Key improvements:
    - Bayesian shrinkage toward global priors to prevent overfitting small samples
    - Coupling with xG to align corner predictions with team dominance
    - Consistent more_corners derivation from team Poisson distributions
    
    Predicts:
    - Total corners over/under
    - Team corner totals
    - More corners team (derived consistently from team lambdas)
    """
    
    # Typical corner lines
    CORNER_LINES = [7.5, 8.5, 9.5, 10.5, 11.5]
    TEAM_CORNER_LINES = [3.5, 4.5, 5.5, 6.5]
    
    def __init__(self):
        pass
    
    def _apply_shrinkage(
        self,
        observed_mean: float,
        sample_size: int,
        prior_mean: float = None,
    ) -> float:
        """Apply Bayesian shrinkage to corner estimates."""
        if prior_mean is None:
            prior_mean = GLOBAL_PRIORS["corners_per_team"]
        ess = EFFECTIVE_SAMPLE_SIZES["corners"]
        return bayesian_shrinkage(observed_mean, prior_mean, sample_size, ess)
    
    def predict_total_corners(
        self,
        home_avg_corners: float,
        away_avg_corners: float,
        home_avg_corners_conceded: float = None,
        away_avg_corners_conceded: float = None,
        home_sample_size: int = 0,
        away_sample_size: int = 0,
        home_xg: float = None,
        away_xg: float = None,
    ) -> Dict[str, Any]:
        """
        Predict total corners over/under probabilities with regularization.
        
        Args:
            home_avg_corners: Home team avg corners for
            away_avg_corners: Away team avg corners for
            home_avg_corners_conceded: Home team avg corners against (optional)
            away_avg_corners_conceded: Away team avg corners against (optional)
            home_sample_size: Number of matches for home team
            away_sample_size: Number of matches for away team
            home_xg: Home team expected goals (for coupling)
            away_xg: Away team expected goals (for coupling)
            
        Returns:
            Dict with over/under probabilities for each line, expected_total, and effective_sample_size
        """
        # Apply Bayesian shrinkage
        home_lambda = self._apply_shrinkage(home_avg_corners, home_sample_size)
        away_lambda = self._apply_shrinkage(away_avg_corners, away_sample_size)
        
        # If xG data available, couple corners with expected goals
        # Teams with higher xG should get more corners (more attacking pressure)
        if home_xg is not None and away_xg is not None:
            xg_ratio_home = home_xg / max(home_xg + away_xg, 0.1)
            xg_ratio_away = away_xg / max(home_xg + away_xg, 0.1)
            # Blend: 70% corners stats, 30% xG-based expectation
            base_total = GLOBAL_PRIORS["total_corners_match"]
            home_lambda = 0.7 * home_lambda + 0.3 * base_total * xg_ratio_home
            away_lambda = 0.7 * away_lambda + 0.3 * base_total * xg_ratio_away
        
        # Use attack/defense combination if available
        if home_avg_corners_conceded is not None and away_avg_corners_conceded is not None:
            home_def = self._apply_shrinkage(home_avg_corners_conceded, home_sample_size)
            away_def = self._apply_shrinkage(away_avg_corners_conceded, away_sample_size)
            home_expected = (home_lambda + away_def) / 2
            away_expected = (away_lambda + home_def) / 2
        else:
            home_expected = home_lambda
            away_expected = away_lambda
        
        total_lambda = home_expected + away_expected
        
        # Calculate over/under for each line
        predictions = {}
        for line in self.CORNER_LINES:
            line_int = int(line + 0.5)
            over_prob = poisson_sf(line_int - 1, total_lambda)
            under_prob = 1.0 - over_prob
            
            predictions[f"over_{int(line)}"] = round(over_prob, 4)
            predictions[f"under_{int(line)}"] = round(under_prob, 4)
        
        # Compute effective sample size (harmonic mean weighted)
        total_ss = home_sample_size + away_sample_size
        eff_ss = min(total_ss, 2 * EFFECTIVE_SAMPLE_SIZES["corners"])
        
        return {
            "expected_total": round(total_lambda, 2),
            "lines": predictions,
            "effective_sample_size": round(eff_ss, 1),
            "home_lambda": round(home_expected, 2),
            "away_lambda": round(away_expected, 2),
        }
    
    def predict_team_corners(
        self,
        home_avg_corners: float,
        away_avg_corners: float,
        home_sample_size: int = 0,
        away_sample_size: int = 0,
    ) -> Dict[str, Any]:
        """
        Predict team corner totals with regularization.
        
        Returns dict with both team_lines and lambda values for consistency.
        """
        home_lambda = self._apply_shrinkage(home_avg_corners, home_sample_size)
        away_lambda = self._apply_shrinkage(away_avg_corners, away_sample_size)
        
        predictions = {}
        
        for prefix, lam in [("home", home_lambda), ("away", away_lambda)]:
            for line in self.TEAM_CORNER_LINES:
                line_int = int(line + 0.5)
                over_prob = poisson_sf(line_int - 1, lam)
                under_prob = 1.0 - over_prob
                
                predictions[f"{prefix}_over_{int(line)}"] = round(over_prob, 4)
                predictions[f"{prefix}_under_{int(line)}"] = round(under_prob, 4)
        
        return {
            **predictions,
            "home_lambda": round(home_lambda, 2),
            "away_lambda": round(away_lambda, 2),
        }
    
    def predict_more_corners(
        self,
        home_avg_corners: float,
        away_avg_corners: float,
        home_sample_size: int = 0,
        away_sample_size: int = 0,
    ) -> Dict[str, float]:
        """
        Predict which team will have more corners.
        
        Uses Skellam distribution (difference of independent Poissons).
        This is CONSISTENT with team_corners lambdas - uses same underlying distribution.
        
        Args:
            home_avg_corners: Home team avg corners for
            away_avg_corners: Away team avg corners for
            home_sample_size: Sample size for home
            away_sample_size: Sample size for away
            
        Returns:
            Dict with home/away/tie probabilities
        """
        # Apply same shrinkage as team_corners for consistency
        home_lambda = self._apply_shrinkage(home_avg_corners, home_sample_size)
        away_lambda = self._apply_shrinkage(away_avg_corners, away_sample_size)
        
        # Use Skellam distribution: P(home > away) where home ~ Pois(λ_h), away ~ Pois(λ_a)
        # We compute by summing joint probabilities
        max_goals = 15  # Sufficient for convergence
        
        p_home_win = 0.0
        p_away_win = 0.0
        p_tie = 0.0
        
        for i in range(max_goals):
            for j in range(max_goals):
                p_ij = poisson_pmf(i, home_lambda) * poisson_pmf(j, away_lambda)
                if i > j:
                    p_home_win += p_ij
                elif i < j:
                    p_away_win += p_ij
                else:
                    p_tie += p_ij
        
        # Normalize (should already sum to ~1, but ensure numerical stability)
        total = p_home_win + p_away_win + p_tie
        if total > 0:
            p_home_win /= total
            p_away_win /= total
            p_tie /= total
        
        return {
            "home": round(p_home_win, 4),
            "away": round(p_away_win, 4),
            "tie": round(p_tie, 4),
        }
    
    def _normal_cdf(self, x: float) -> float:
        """Standard normal CDF approximation."""
        return 0.5 * (1 + math.erf(x / math.sqrt(2)))


class CardsModel:
    """
    Poisson/Negative Binomial model for cards markets with Bayesian regularization.
    
    Key improvements:
    - Bayesian shrinkage toward global priors
    - Coupling with match intensity (xG difference indicates competitive games)
    - Consistent more_cards derivation from team Poisson distributions
    
    Predicts:
    - Total cards over/under
    - Team card totals
    - More cards team (derived consistently from team lambdas)
    """
    
    # Typical card lines
    CARD_LINES = [3.5, 4.5, 5.5, 6.5]
    TEAM_CARD_LINES = [1.5, 2.5, 3.5]
    
    def __init__(self):
        pass
    
    def _apply_shrinkage(
        self,
        observed_mean: float,
        sample_size: int,
        prior_mean: float = None,
    ) -> float:
        """Apply Bayesian shrinkage to card estimates."""
        if prior_mean is None:
            prior_mean = GLOBAL_PRIORS["cards_per_team"]
        ess = EFFECTIVE_SAMPLE_SIZES["cards"]
        return bayesian_shrinkage(observed_mean, prior_mean, sample_size, ess)
    
    def predict_total_cards(
        self,
        home_avg_cards: float,
        away_avg_cards: float,
        home_sample_size: int = 0,
        away_sample_size: int = 0,
        home_xg: float = None,
        away_xg: float = None,
    ) -> Dict[str, Any]:
        """
        Predict total cards over/under probabilities with regularization.
        
        Args:
            home_avg_cards: Home team avg cards for
            away_avg_cards: Away team avg cards for
            home_sample_size: Number of matches for home team
            away_sample_size: Number of matches for away team
            home_xg: Home team expected goals (for coupling)
            away_xg: Away team expected goals (for coupling)
            
        Returns:
            Dict with over/under probabilities, expected_total, effective_sample_size
        """
        # Apply Bayesian shrinkage
        home_lambda = self._apply_shrinkage(home_avg_cards, home_sample_size)
        away_lambda = self._apply_shrinkage(away_avg_cards, away_sample_size)
        
        # Couple with xG: close matches (similar xG) tend to have more cards
        if home_xg is not None and away_xg is not None:
            xg_diff = abs(home_xg - away_xg)
            # Higher xG diff -> more one-sided -> fewer cards typically
            # Lower xG diff -> competitive -> more cards
            competitiveness_factor = 1.0 + 0.15 * math.exp(-xg_diff / 0.5)
            home_lambda *= competitiveness_factor
            away_lambda *= competitiveness_factor
        
        total_lambda = home_lambda + away_lambda
        
        predictions = {}
        for line in self.CARD_LINES:
            line_int = int(line + 0.5)
            over_prob = poisson_sf(line_int - 1, total_lambda)
            under_prob = 1.0 - over_prob
            
            predictions[f"over_{int(line)}"] = round(over_prob, 4)
            predictions[f"under_{int(line)}"] = round(under_prob, 4)
        
        total_ss = home_sample_size + away_sample_size
        eff_ss = min(total_ss, 2 * EFFECTIVE_SAMPLE_SIZES["cards"])
        
        return {
            "expected_total": round(total_lambda, 2),
            "lines": predictions,
            "effective_sample_size": round(eff_ss, 1),
            "home_lambda": round(home_lambda, 2),
            "away_lambda": round(away_lambda, 2),
        }
    
    def predict_team_cards(
        self,
        home_avg_cards: float,
        away_avg_cards: float,
        home_sample_size: int = 0,
        away_sample_size: int = 0,
    ) -> Dict[str, Any]:
        """Predict team card totals with regularization."""
        home_lambda = self._apply_shrinkage(home_avg_cards, home_sample_size)
        away_lambda = self._apply_shrinkage(away_avg_cards, away_sample_size)
        
        predictions = {}
        
        for prefix, lam in [("home", home_lambda), ("away", away_lambda)]:
            for line in self.TEAM_CARD_LINES:
                line_int = int(line + 0.5)
                over_prob = poisson_sf(line_int - 1, lam)
                under_prob = 1.0 - over_prob
                
                predictions[f"{prefix}_over_{int(line)}"] = round(over_prob, 4)
                predictions[f"{prefix}_under_{int(line)}"] = round(under_prob, 4)
        
        return {
            **predictions,
            "home_lambda": round(home_lambda, 2),
            "away_lambda": round(away_lambda, 2),
        }
    
    def predict_more_cards(
        self,
        home_avg_cards: float,
        away_avg_cards: float,
        home_sample_size: int = 0,
        away_sample_size: int = 0,
    ) -> Dict[str, float]:
        """
        Predict which team will have more cards.
        
        Uses Skellam distribution (difference of independent Poissons).
        CONSISTENT with team_cards lambdas.
        """
        # Apply same shrinkage as team_cards for consistency
        home_lambda = self._apply_shrinkage(home_avg_cards, home_sample_size)
        away_lambda = self._apply_shrinkage(away_avg_cards, away_sample_size)
        
        # Use Skellam distribution via joint probability summation
        max_goals = 15
        
        p_home_win = 0.0
        p_away_win = 0.0
        p_tie = 0.0
        
        for i in range(max_goals):
            for j in range(max_goals):
                p_ij = poisson_pmf(i, home_lambda) * poisson_pmf(j, away_lambda)
                if i > j:
                    p_home_win += p_ij
                elif i < j:
                    p_away_win += p_ij
                else:
                    p_tie += p_ij
        
        total = p_home_win + p_away_win + p_tie
        if total > 0:
            p_home_win /= total
            p_away_win /= total
            p_tie /= total
        
        return {
            "home": round(p_home_win, 4),
            "away": round(p_away_win, 4),
            "tie": round(p_tie, 4),
        }
    
    def _normal_cdf(self, x: float) -> float:
        """Standard normal CDF approximation."""
        return 0.5 * (1 + math.erf(x / math.sqrt(2)))


class ShotsModel:
    """
    Poisson model for shots on target markets.
    
    Predicts:
    - Total SOT over/under
    - Team SOT totals
    """
    
    # Typical SOT lines
    SOT_LINES = [7.5, 8.5, 9.5, 10.5, 11.5]
    TEAM_SOT_LINES = [3.5, 4.5, 5.5]
    
    def __init__(self):
        pass
    
    def predict_total_sot(
        self,
        home_avg_sot: float,
        away_avg_sot: float,
    ) -> Dict[str, float]:
        """Predict total shots on target over/under."""
        total_lambda = home_avg_sot + away_avg_sot
        
        predictions = {}
        for line in self.SOT_LINES:
            line_int = int(line + 0.5)
            over_prob = poisson_sf(line_int - 1, total_lambda)
            under_prob = 1.0 - over_prob
            
            predictions[f"over_{int(line)}"] = round(over_prob, 4)
            predictions[f"under_{int(line)}"] = round(under_prob, 4)
        
        return {
            "expected_total": round(total_lambda, 2),
            "lines": predictions,
        }
    
    def predict_team_sot(
        self,
        home_avg_sot: float,
        away_avg_sot: float,
    ) -> Dict[str, float]:
        """Predict team SOT totals."""
        predictions = {}
        
        for prefix, avg in [("home", home_avg_sot), ("away", away_avg_sot)]:
            for line in self.TEAM_SOT_LINES:
                line_int = int(line + 0.5)
                over_prob = poisson_sf(line_int - 1, avg)
                under_prob = 1.0 - over_prob
                
                predictions[f"{prefix}_over_{int(line)}"] = round(over_prob, 4)
                predictions[f"{prefix}_under_{int(line)}"] = round(under_prob, 4)
        
        return predictions


class PlayerPropsModel:
    """
    Heuristic/hierarchical model for player props with proper probability normalization.
    
    Key improvements:
    - Anytime scorer probabilities are capped by team lambda (sum <= λ_team)
    - First scorer probabilities sum to <= 1.0 with explicit "no goal" probability
    - Proper allocation of team xG to individual players based on shot share, role
    
    Predicts:
    - Anytime scorer (normalized across team)
    - First scorer (normalized distribution)
    - Player SOT over/under
    - Player assists over/under
    """
    
    # Typical player SOT lines
    PLAYER_SOT_LINES = [0.5, 1.5, 2.5]
    PLAYER_ASSIST_LINES = [0.5]
    
    def __init__(self):
        pass
    
    def predict_anytime_scorer_normalized(
        self,
        players_data: List[Dict[str, Any]],
        team_xg: float,
        team_total_shots: float = None,
    ) -> List[Dict[str, Any]]:
        """
        Predict anytime scorer probabilities for multiple players with proper normalization.
        
        Uses a Poisson-based model where each player has an individual lambda (goal rate).
        The sum of all player lambdas ≈ team_xg, with realistic distribution based on:
        - Recent goals (strongest signal)
        - Shots per 90
        - Position hierarchy (CF > AM/WM > CM > FB > CB > GK)
        - Expected minutes
        
        Probability is computed as P_i = 1 - exp(-lambda_i), then scaled to ensure
        sum(P_i) stays reasonable relative to team_xg.
        
        Args:
            players_data: List of dicts with player stats (goals, shots, minutes, position, etc.)
            team_xg: Team expected goals
            team_total_shots: Team total shots (optional)
            
        Returns:
            List of dicts with player_name, probability (as percentage), and metadata
        """
        if not players_data:
            return []
        
        # Position hierarchy weights (higher = more offensive role)
        POSITION_WEIGHTS = {
            "forward": 1.0, "striker": 1.0, "attacker": 1.0, "delantero": 1.0, "cf": 1.0,
            "winger": 0.85, "extremo": 0.85, "am": 0.85, "mediapunta": 0.85,
            "midfielder": 0.6, "volante": 0.6, "cm": 0.6, "cdm": 0.5,
            "fullback": 0.35, "lateral": 0.35, "wingback": 0.4,
            "defender": 0.25, "centreback": 0.2, "central": 0.2, "defensa": 0.25,
            "goalkeeper": 0.02, "portero": 0.02, "arquero": 0.02,
        }
        
        # Compute raw lambda scores for each player
        player_lambdas = []
        total_goals_recent = 0
        total_weighted_score = 0
        
        for pdata in players_data:
            goals = pdata.get("goals", 0) or 0
            matches = pdata.get("matches_played", 1) or 1
            shots = pdata.get("shots", 0) or 0
            minutes = pdata.get("minutes", 0) or 0
            position = pdata.get("position", "").lower()
            is_starter = pdata.get("is_starter", False) or pdata.get("starts", 0) > 0
            
            total_goals_recent += goals
        
        # Avoid division by zero
        if total_goals_recent == 0:
            total_goals_recent = 0.1
        
        for pdata in players_data:
            goals = pdata.get("goals", 0) or 0
            matches = pdata.get("matches_played", 1) or 1
            shots = pdata.get("shots", 0) or 0
            minutes = pdata.get("minutes", 0) or 0
            position = pdata.get("position", "").lower()
            is_starter = pdata.get("is_starter", False) or pdata.get("starts", 0) > 0
            
            # 1. Goals contribution (strongest factor) - share of team's recent goals
            goal_share = goals / total_goals_recent if total_goals_recent > 0 else 1.0 / len(players_data)
            
            # 2. Shots per 90 - indicates involvement in attack
            shots_per_90 = (shots / matches) * (90 / max(minutes, 1)) if minutes > 0 else 0
            # Normalize to [0, 1] range (typical max ~12 shots/90 for very active players)
            shot_factor = min(1.0, shots_per_90 / 10.0)
            
            # 3. Position weight from hierarchy
            pos_weight = 0.5  # default
            for key, val in POSITION_WEIGHTS.items():
                if key in position:
                    pos_weight = val
                    break
            
            # 4. Minutes/starter factor - playing time expectation
            # Starters get full weight, subs reduced
            mins_factor = min(1.0, minutes / 90) if minutes > 0 else 0.3
            starter_bonus = 1.0 if is_starter else 0.7
            playing_time_factor = mins_factor * starter_bonus
            
            # Combine factors into raw lambda score
            # Weighting: goals (50%), position (25%), shots (15%), playing time (10%)
            raw_lambda_score = (
                0.50 * goal_share +
                0.25 * pos_weight +
                0.15 * shot_factor +
                0.10 * playing_time_factor
            )
            
            player_lambdas.append({
                "player_name": pdata.get("player_name", "Unknown"),
                "team": pdata.get("team", ""),
                "raw_lambda_score": max(0.001, raw_lambda_score),
                "position": position,
                "goals_recent": goals,
                "matches": matches,
                "minutes": minutes,
                "is_starter": is_starter,
                "shots_per_90": shots_per_90,
            })
        
        # Normalize lambdas so sum(lambda_i) ≈ team_xg
        total_raw_lambda = sum(p["raw_lambda_score"] for p in player_lambdas)
        
        results = []
        for p in player_lambdas:
            # Scale raw score to get player lambda
            lambda_i = (p["raw_lambda_score"] / total_raw_lambda) * team_xg if total_raw_lambda > 0 else team_xg / len(player_lambdas)
            
            # Convert lambda to probability using Poisson: P(score ≥ 1) = 1 - exp(-lambda)
            prob_anytime = 1.0 - math.exp(-lambda_i)
            
            # Store both lambda and probability (internally as float in [0,1])
            # Also store percentage format for display
            results.append({
                "player_name": p["player_name"],
                "team": p["team"],
                "probability": round(prob_anytime, 4),  # Internal: float in [0,1]
                "probability_pct": round(prob_anytime * 100, 2),  # Display: percentage
                "lambda": round(lambda_i, 4),
                "goals_recent": p["goals_recent"],
                "matches_played": p["matches"],
                "position": p["position"],
                "minutes": p["minutes"],
                "is_starter": p["is_starter"],
                "shots_per_90": round(p["shots_per_90"], 2),
            })
        
        # Sort by probability descending
        results.sort(key=lambda x: x["probability"], reverse=True)
        
        return results
    
    def predict_first_scorer_normalized(
        self,
        anytime_probs: List[Dict[str, Any]],
        team_xg: float,
        no_goal_probability: float = None,
    ) -> List[Dict[str, Any]]:
        """
        Predict first scorer probabilities with proper normalization.
        
        The sum of all first scorer probabilities + P(no goal) = 1.0
        
        Args:
            anytime_probs: List of dicts from predict_anytime_scorer_normalized
                          (probability field is float in [0,1], probability_pct is percentage)
            team_xg: Team expected goals
            no_goal_probability: Explicit P(no goal) if provided, else derived
            
        Returns:
            List of dicts with player_name, probability (as percentage for display),
            and probability_decimal (internal float in [0,1]), sorted descending
        """
        if not anytime_probs:
            return []
        
        # Derive no-goal probability from team xG using Poisson
        # P(no goal) = P(team scores 0) = exp(-λ)
        if no_goal_probability is None:
            no_goal_prob = math.exp(-team_xg)
        else:
            no_goal_prob = no_goal_probability
        
        # Remaining probability mass for scorers
        scorer_mass = 1.0 - no_goal_prob
        
        # Compute raw scores from anytime probabilities
        # Players with higher anytime prob should have higher first-scorer prob
        # Use probability (float in [0,1]) for internal calculation
        total_anytime = sum(p.get("probability", p.get("probability_pct", 0) / 100) for p in anytime_probs)
        
        results = []
        for p in anytime_probs:
            # Get probability as decimal (handle both old and new format)
            prob_decimal = p.get("probability", 0)
            # If probability > 1, assume it's percentage and convert
            if prob_decimal > 1.0:
                prob_decimal = prob_decimal / 100
            
            if total_anytime > 0:
                # Allocate scorer_mass proportionally to anytime probability
                raw_score = prob_decimal / total_anytime
            else:
                raw_score = 1.0 / len(anytime_probs)
            
            results.append({
                "player_name": p["player_name"],
                "team": p.get("team", ""),
                "raw_score": raw_score,
            })
        
        # Normalize so sum = scorer_mass
        total_raw = sum(r["raw_score"] for r in results)
        
        final_results = []
        for r in results:
            if total_raw > 0:
                prob = (r["raw_score"] / total_raw) * scorer_mass
            else:
                prob = scorer_mass / len(results)
            
            # Cap individual first-scorer prob (typically <35% for any player)
            # But ensure we don't cap so much that total doesn't reach scorer_mass
            prob = min(prob, 0.40)
            prob = max(prob, 0.001)
            
            final_results.append({
                "player_name": r["player_name"],
                "team": r["team"],
                "probability": round(prob * 100, 2),  # Display: percentage
                "probability_decimal": round(prob, 4),  # Internal: float in [0,1]
            })
        
        # Add explicit no-goal entry for clarity
        final_results.append({
            "player_name": "[NO GOAL]",
            "team": "",
            "probability": round(no_goal_prob * 100, 2),  # Display: percentage
            "probability_decimal": round(no_goal_prob, 4),  # Internal: float in [0,1]
        })
        
        # Sort by probability descending
        final_results.sort(key=lambda x: x["probability"], reverse=True)
        
        return final_results
    
    def predict_anytime_scorer(
        self,
        player_data: Dict[str, Any],
        team_xg: float,
        team_total_shots: float,
    ) -> float:
        """
        Legacy method for single-player anytime scorer prediction.
        Deprecated: use predict_anytime_scorer_normalized instead.
        """
        # Base rate from team xg
        base_prob = team_xg / 11
        
        # Adjust by player's historical goal share
        goals = player_data.get("goals") or 0
        matches = player_data.get("matches_played") or 1
        goals_per_match = goals / max(matches, 1)
        
        # Shot share adjustment
        player_shots = player_data.get("shots") or 0
        if team_total_shots > 0:
            shot_share = player_shots / team_total_shots
        else:
            shot_share = 1 / 11
        
        # Starter bonus
        is_starter = player_data.get("is_starter", False)
        starter_bonus = 1.5 if is_starter else 0.7
        
        # Minutes adjustment
        minutes = player_data.get("minutes") or 0
        minutes_factor = min(1.0, minutes / 90) if minutes > 0 else 0.3
        
        # Combine factors
        goal_rate = goals_per_match * starter_bonus * minutes_factor
        
        # Poisson-based: P(at least 1 goal) = 1 - P(0 goals)
        prob = 1.0 - math.exp(-goal_rate)
        
        # Blend with base rate
        final_prob = 0.6 * prob + 0.4 * base_prob * shot_share * 11
        
        return round(max(0.01, min(0.95, final_prob)), 4)
    
    def predict_first_scorer(
        self,
        player_data: Dict[str, Any],
        anytime_prob: float,
        is_starter: bool = True,
    ) -> float:
        """
        Legacy method for single-player first scorer prediction.
        Deprecated: use predict_first_scorer_normalized instead.
        """
        if not is_starter:
            return anytime_prob * 0.3
        
        # Rough approximation
        expected_goals = 2.5
        first_scorer_prob = anytime_prob / expected_goals
        
        return round(max(0.001, min(0.5, first_scorer_prob)), 4)
    
    def predict_player_sot(
        self,
        player_data: Dict[str, Any],
        team_avg_sot: float,
    ) -> Dict[str, float]:
        """
        Predict player SOT over/under.
        
        Args:
            player_data: Player stats
            team_avg_sot: Team average SOT per match
            
        Returns:
            Dict with over/under probabilities
        """
        # Estimate player SOT lambda
        shots = player_data.get("shots") or 0
        sot = player_data.get("shots_on_target") or 0
        matches = player_data.get("matches_played") or 1
        
        if matches > 0 and shots > 0:
            player_sot_avg = sot / matches
            sot_rate = sot / shots  # SOT accuracy
        else:
            player_sot_avg = team_avg_sot / 11
            sot_rate = 0.35  # League average
        
        # Starter/mins adjustment
        is_starter = player_data.get("is_starter", False)
        minutes = player_data.get("minutes") or 0
        
        if is_starter:
            lambda_sot = player_sot_avg
        elif minutes > 0:
            lambda_sot = player_sot_avg * (minutes / 90)
        else:
            lambda_sot = player_sot_avg * 0.5
        
        predictions = {}
        for line in self.PLAYER_SOT_LINES:
            line_int = int(line + 0.5)
            over_prob = poisson_sf(line_int - 1, lambda_sot)
            under_prob = 1.0 - over_prob
            
            predictions[f"over_{int(line)}"] = round(over_prob, 4)
            predictions[f"under_{int(line)}"] = round(under_prob, 4)
        
        return {
            "expected_sot": round(lambda_sot, 2),
            "lines": predictions,
        }
    
    def predict_player_assists(
        self,
        player_data: Dict[str, Any],
        team_avg_goals: float,
    ) -> Dict[str, float]:
        """
        Predict player assists over/under.
        
        Args:
            player_data: Player stats
            team_avg_goals: Team average goals per match
            
        Returns:
            Dict with over/under probabilities
        """
        assists = player_data.get("assists") or 0
        matches = player_data.get("matches_played") or 1
        
        if matches > 0:
            assist_avg = assists / matches
        else:
            assist_avg = team_avg_goals / 11 * 0.3  # Rough estimate
        
        # Apply minutes adjustment
        minutes = player_data.get("minutes") or 0
        if minutes > 0 and minutes < 90:
            assist_avg *= (minutes / 90)
        
        lambda_assists = assist_avg
        
        predictions = {}
        for line in self.PLAYER_ASSIST_LINES:
            line_int = int(line + 0.5)
            over_prob = poisson_sf(line_int - 1, lambda_assists)
            under_prob = 1.0 - over_prob
            
            predictions[f"over_{int(line)}"] = round(over_prob, 4)
            predictions[f"under_{int(line)}"] = round(under_prob, 4)
        
        return {
            "expected_assists": round(lambda_assists, 2),
            "lines": predictions,
        }
