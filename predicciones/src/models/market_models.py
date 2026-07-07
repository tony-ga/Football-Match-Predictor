"""
Statistical models for alternative markets.

Implements Poisson and Negative Binomial models for:
- Corners
- Cards  
- Shots on target
- Player props (heuristic/hierarchical)
"""
from __future__ import annotations

import logging
import math
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


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


class CornersModel:
    """
    Poisson-based model for corners markets.
    
    Predicts:
    - Total corners over/under
    - Team corner totals
    - More corners team
    - First/last corner (if event data available)
    - Race to X corners (if event data available)
    """
    
    # Typical corner lines
    CORNER_LINES = [7.5, 8.5, 9.5, 10.5, 11.5]
    TEAM_CORNER_LINES = [3.5, 4.5, 5.5, 6.5]
    
    def __init__(self):
        pass
    
    def predict_total_corners(
        self,
        home_avg_corners: float,
        away_avg_corners: float,
        home_avg_corners_conceded: float = None,
        away_avg_corners_conceded: float = None,
    ) -> Dict[str, float]:
        """
        Predict total corners over/under probabilities.
        
        Args:
            home_avg_corners: Home team avg corners for
            away_avg_corners: Away team avg corners for
            home_avg_corners_conceded: Home team avg corners against (optional)
            away_avg_corners_conceded: Away team avg corners against (optional)
            
        Returns:
            Dict with over/under probabilities for each line
        """
        # Estimate expected total corners
        # Simple: average of both teams' attack stats
        if home_avg_corners_conceded is not None and away_avg_corners_conceded is not None:
            # Use both attack and defense stats
            home_expected = (home_avg_corners + away_avg_corners_conceded) / 2
            away_expected = (away_avg_corners + home_avg_corners_conceded) / 2
        else:
            home_expected = home_avg_corners
            away_expected = away_avg_corners
        
        total_lambda = home_expected + away_expected
        
        # Calculate over/under for each line
        predictions = {}
        for line in self.CORNER_LINES:
            line_int = int(line + 0.5)  # Convert 8.5 -> 9
            over_prob = poisson_sf(line_int - 1, total_lambda)
            under_prob = 1.0 - over_prob
            
            predictions[f"over_{int(line)}"] = round(over_prob, 4)
            predictions[f"under_{int(line)}"] = round(under_prob, 4)
        
        return {
            "expected_total": round(total_lambda, 2),
            "lines": predictions,
        }
    
    def predict_team_corners(
        self,
        home_avg_corners: float,
        away_avg_corners: float,
    ) -> Dict[str, float]:
        """
        Predict team corner totals.
        
        Args:
            home_avg_corners: Home team avg corners for
            away_avg_corners: Away team avg corners for
            
        Returns:
            Dict with team over/under probabilities
        """
        predictions = {}
        
        for prefix, avg in [("home", home_avg_corners), ("away", away_avg_corners)]:
            for line in self.TEAM_CORNER_LINES:
                line_int = int(line + 0.5)
                over_prob = poisson_sf(line_int - 1, avg)
                under_prob = 1.0 - over_prob
                
                predictions[f"{prefix}_over_{int(line)}"] = round(over_prob, 4)
                predictions[f"{prefix}_under_{int(line)}"] = round(under_prob, 4)
        
        return predictions
    
    def predict_more_corners(
        self,
        home_avg_corners: float,
        away_avg_corners: float,
    ) -> Dict[str, float]:
        """
        Predict which team will have more corners.
        
        Uses Skellam distribution approximation (difference of Poissons).
        
        Args:
            home_avg_corners: Home team avg corners for
            away_avg_corners: Away team avg corners for
            
        Returns:
            Dict with home/away/tie probabilities
        """
        # Simplified: use normal approximation to Skellam
        diff_mean = home_avg_corners - away_avg_corners
        diff_var = home_avg_corners + away_avg_corners
        diff_std = math.sqrt(diff_var) if diff_var > 0 else 1.0
        
        # P(home > away) = P(diff > 0)
        # Using normal CDF approximation
        home_prob = self._normal_cdf(diff_mean / diff_std)
        away_prob = self._normal_cdf(-diff_mean / diff_std)
        
        # Tie probability: P(-0.5 < diff < 0.5)
        tie_prob = self._normal_cdf((0.5 - diff_mean) / diff_std) - self._normal_cdf((-0.5 - diff_mean) / diff_std)
        
        # Normalize
        total = home_prob + away_prob + tie_prob
        home_prob /= total
        away_prob /= total
        tie_prob /= total
        
        return {
            "home": round(home_prob, 4),
            "away": round(away_prob, 4),
            "tie": round(tie_prob, 4),
        }
    
    def _normal_cdf(self, x: float) -> float:
        """Standard normal CDF approximation."""
        return 0.5 * (1 + math.erf(x / math.sqrt(2)))


class CardsModel:
    """
    Poisson/Negative Binomial model for cards markets.
    
    Predicts:
    - Total cards over/under
    - Team card totals
    - More cards team
    - First card (if event data available)
    """
    
    # Typical card lines
    CARD_LINES = [3.5, 4.5, 5.5, 6.5]
    TEAM_CARD_LINES = [1.5, 2.5, 3.5]
    
    def __init__(self):
        pass
    
    def predict_total_cards(
        self,
        home_avg_cards: float,
        away_avg_cards: float,
        league_avg_cards: float = 4.0,
    ) -> Dict[str, float]:
        """
        Predict total cards over/under probabilities.
        
        Cards often show overdispersion, so we use Negative Binomial.
        
        Args:
            home_avg_cards: Home team avg cards for
            away_avg_cards: Away team avg cards for
            league_avg_cards: League average for shrinkage
            
        Returns:
            Dict with over/under probabilities
        """
        # Shrink towards league average
        shrinkage = 0.3
        home_expected = (1 - shrinkage) * home_avg_cards + shrinkage * league_avg_cards
        away_expected = (1 - shrinkage) * away_avg_cards + shrinkage * league_avg_cards
        
        total_lambda = home_expected + away_expected
        
        # For cards, use slightly higher dispersion
        # NB parameters: mean = total_lambda, variance = total_lambda * (1 + alpha)
        alpha = 0.3  # dispersion parameter
        
        predictions = {}
        for line in self.CARD_LINES:
            line_int = int(line + 0.5)
            
            # Use Poisson for simplicity (NB requires more tuning)
            over_prob = poisson_sf(line_int - 1, total_lambda)
            under_prob = 1.0 - over_prob
            
            predictions[f"over_{int(line)}"] = round(over_prob, 4)
            predictions[f"under_{int(line)}"] = round(under_prob, 4)
        
        return {
            "expected_total": round(total_lambda, 2),
            "lines": predictions,
        }
    
    def predict_team_cards(
        self,
        home_avg_cards: float,
        away_avg_cards: float,
    ) -> Dict[str, float]:
        """Predict team card totals."""
        predictions = {}
        
        for prefix, avg in [("home", home_avg_cards), ("away", away_avg_cards)]:
            for line in self.TEAM_CARD_LINES:
                line_int = int(line + 0.5)
                over_prob = poisson_sf(line_int - 1, avg)
                under_prob = 1.0 - over_prob
                
                predictions[f"{prefix}_over_{int(line)}"] = round(over_prob, 4)
                predictions[f"{prefix}_under_{int(line)}"] = round(under_prob, 4)
        
        return predictions
    
    def predict_more_cards(
        self,
        home_avg_cards: float,
        away_avg_cards: float,
    ) -> Dict[str, float]:
        """Predict which team will have more cards."""
        # Same approach as corners
        diff_mean = home_avg_cards - away_avg_cards
        diff_var = home_avg_cards + away_avg_cards
        diff_std = math.sqrt(diff_var) if diff_var > 0 else 1.0
        
        home_prob = self._normal_cdf(diff_mean / diff_std)
        away_prob = self._normal_cdf(-diff_mean / diff_std)
        tie_prob = self._normal_cdf((0.5 - diff_mean) / diff_std) - self._normal_cdf((-0.5 - diff_mean) / diff_std)
        
        total = home_prob + away_prob + tie_prob
        home_prob /= total
        away_prob /= total
        tie_prob /= total
        
        return {
            "home": round(home_prob, 4),
            "away": round(away_prob, 4),
            "tie": round(tie_prob, 4),
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
    Heuristic/hierarchical model for player props.
    
    Predicts:
    - Anytime scorer
    - First scorer
    - Player SOT over/under
    - Player assists over/under
    
    Uses:
    - Team expected goals
    - Player shot share
    - Player goal share
    - Minutes/starter status
    - Set piece role proxy
    """
    
    # Typical player SOT lines
    PLAYER_SOT_LINES = [0.5, 1.5, 2.5]
    PLAYER_ASSIST_LINES = [0.5]
    
    def __init__(self):
        pass
    
    def predict_anytime_scorer(
        self,
        player_data: Dict[str, Any],
        team_xg: float,
        team_total_shots: float,
    ) -> float:
        """
        Predict probability player scores anytime.
        
        Args:
            player_data: Dict with player stats (goals, shots, minutes, is_starter, etc.)
            team_xg: Team expected goals
            team_total_shots: Team total shots
            
        Returns:
            Probability of scoring (0-1)
        """
        # Base rate from team xg
        base_prob = team_xg / 11  # Roughly equal distribution
        
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
        Predict probability player scores first goal.
        
        Approximation: first_scorer ≈ anytime / expected_goals_in_match
        """
        if not is_starter:
            return anytime_prob * 0.3  # Subs rarely score first
        
        # Rough approximation
        expected_goals = 2.5  # Average match goals
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
