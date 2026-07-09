"""
Dixon-Coles model for football match prediction.

Reference: Dixon, M.J. & Coles, S.G. (1997). "Modelling Association Football
Scores and Inefficiencies in the Football Betting Market."
Applied Statistics, 46(2), 265-280.

Key features over simple Poisson:
- Low-score correction (rho parameter): adjusts P(0-0), P(1-0), P(0-1), P(1-1)
  to account for correlation between home and away goals in low-scoring games.
- Temporal decay: more recent matches weighted more heavily in MLE.
- Trainable attack/defense parameters per team.
- Heuristic mode: estimates lambdas from features when no trained params available.
- Markov-aware mode: optionally incorporates Markov state features for in-play
  prediction (requires markov_features module to be loaded).
"""
from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Dict, Optional, Tuple, Any

import numpy as np
from scipy.optimize import minimize
from scipy.special import gammaln
from scipy.stats import poisson

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Optional Markov features integration
# ---------------------------------------------------------------------------
_MARKOV_AVAILABLE = False
try:
    from ..features.markov_features import (
        get_markov_features,
        build_state_from_match_context,
        build_state_for_away_team,
    )
    _MARKOV_AVAILABLE = True
except ImportError:
    logger.debug("Markov features module not available; Markov-aware mode disabled")


# ---------------------------------------------------------------------------
# Dixon-Coles Correction Factor (tau)
# ---------------------------------------------------------------------------

def _dc_tau(home_goals: int, away_goals: int, lambda_h: float, lambda_a: float, rho: float) -> float:
    """
    Dixon-Coles low-score correlation correction.
    Adjusts probabilities for scorelines (0,0), (1,0), (0,1), (1,1).

    tau(0,0) = 1 - lambda_h * lambda_a * rho
    tau(1,0) = 1 + lambda_a * rho
    tau(0,1) = 1 + lambda_h * rho
    tau(1,1) = 1 - rho
    tau(i,j) = 1  for i+j > 2
    """
    if home_goals == 0 and away_goals == 0:
        return 1.0 - lambda_h * lambda_a * rho
    elif home_goals == 1 and away_goals == 0:
        return 1.0 + lambda_a * rho
    elif home_goals == 0 and away_goals == 1:
        return 1.0 + lambda_h * rho
    elif home_goals == 1 and away_goals == 1:
        return 1.0 - rho
    else:
        return 1.0


def dc_score_matrix(
    lambda_home: float,
    lambda_away: float,
    rho: float = -0.13,
    max_goals: int = 8,
) -> np.ndarray:
    """
    Generate a Dixon-Coles corrected score probability matrix.

    Args:
        lambda_home: Expected goals for home team.
        lambda_away: Expected goals for away team.
        rho: Correlation parameter for low scores (typically negative, -0.1 to -0.2).
        max_goals: Maximum goals per team.

    Returns:
        2D numpy array of shape (max_goals+1, max_goals+1).
        matrix[i][j] = P(home=i, away=j) with DC correction.
    """
    lambda_home = float(np.clip(lambda_home, 0.05, 8.0))
    lambda_away = float(np.clip(lambda_away, 0.05, 8.0))

    goals = np.arange(max_goals + 1)
    p_home = poisson.pmf(goals, lambda_home)
    p_away = poisson.pmf(goals, lambda_away)

    matrix = np.zeros((max_goals + 1, max_goals + 1))
    for i in range(max_goals + 1):
        for j in range(max_goals + 1):
            tau = _dc_tau(i, j, lambda_home, lambda_away, rho)
            matrix[i, j] = p_home[i] * p_away[j] * tau

    # Ensure non-negative
    matrix = np.maximum(matrix, 0)

    # Normalize to sum to 1
    total = matrix.sum()
    if total > 0:
        matrix /= total

    return matrix


# ---------------------------------------------------------------------------
# MLE Training of Dixon-Coles Parameters
# ---------------------------------------------------------------------------

def dc_log_likelihood(
    params: np.ndarray,
    teams: list,
    home_teams: np.ndarray,
    away_teams: np.ndarray,
    home_goals: np.ndarray,
    away_goals: np.ndarray,
    weights: Optional[np.ndarray] = None,
    rho: float = -0.13,
) -> float:
    """
    Negative log-likelihood for Dixon-Coles model.
    Minimized by scipy.optimize.minimize.

    params layout:
        params[0:n_teams]        = log attack parameters (alpha_i)
        params[n_teams:2*n_teams] = log defense parameters (beta_i)
        params[2*n_teams]        = log home advantage (gamma)
    """
    n_teams = len(teams)
    team_to_idx = {t: i for i, t in enumerate(teams)}

    log_alpha = params[:n_teams]
    log_beta = params[n_teams:2 * n_teams]
    log_gamma = params[2 * n_teams]

    # Constrain: sum of log_alpha = 0 (identifiability)
    log_alpha = log_alpha - log_alpha.mean()

    alpha = np.exp(log_alpha)
    beta = np.exp(log_beta)
    gamma = np.exp(log_gamma)

    if weights is None:
        weights = np.ones(len(home_teams))

    total_ll = 0.0

    for k in range(len(home_teams)):
        h_idx = team_to_idx.get(home_teams[k])
        a_idx = team_to_idx.get(away_teams[k])

        if h_idx is None or a_idx is None:
            continue

        lambda_h = alpha[h_idx] * beta[a_idx] * gamma
        lambda_a = alpha[a_idx] * beta[h_idx]

        lambda_h = max(lambda_h, 1e-6)
        lambda_a = max(lambda_a, 1e-6)

        hg = int(home_goals[k])
        ag = int(away_goals[k])

        log_p = (
            hg * np.log(lambda_h) - lambda_h - gammaln(hg + 1) +
            ag * np.log(lambda_a) - lambda_a - gammaln(ag + 1) +
            np.log(max(_dc_tau(hg, ag, lambda_h, lambda_a, rho), 1e-10))
        )

        total_ll += weights[k] * log_p

    return -total_ll  # negative because we minimize


def temporal_weights(
    dates,
    reference_date=None,
    xi: float = 0.003,
) -> np.ndarray:
    """
    Compute temporal decay weights for matches.
    More recent matches get higher weight.

    w_k = exp(-xi * days_ago_k)

    Args:
        dates: Array of match dates (datetime-like).
        reference_date: Reference date (default: max date in array).
        xi: Decay rate per day (0 = no decay, 0.003 = typical DC value).
    """
    import pandas as pd
    dates = pd.to_datetime(dates)
    if reference_date is None:
        reference_date = dates.max()
    else:
        reference_date = pd.to_datetime(reference_date)

    days_ago = (reference_date - dates).dt.days.values.astype(float)
    weights = np.exp(-xi * days_ago)
    return weights


# ---------------------------------------------------------------------------
# DixonColes Model Class
# ---------------------------------------------------------------------------

class DixonColesModel:
    """
    Dixon-Coles football match prediction model.

    Three operating modes:
    1. Trained mode: uses MLE-estimated alpha/beta/gamma per team.
    2. Heuristic mode: estimates lambdas from feature dict without trained params.
    3. Markov-aware mode: optionally incorporates Markov state features for in-play
       prediction when use_markov_features=True and match_state is provided.

    The heuristic mode is the default for new teams or when no historical data
    is available. It produces reasonable predictions using the feature engineering
    pipeline.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.dc_config = self.config.get('dixon_coles', {})

        self.rho: float = self.dc_config.get('rho', -0.13)
        self.home_advantage: float = self.dc_config.get('home_advantage', 0.25)
        self.xi: float = self.dc_config.get('time_decay_xi', 0.003)
        self.min_lambda: float = self.dc_config.get('min_lambda', 0.05)
        # Default max_lambda reduced from 5.0 to 4.0 for more realistic football scores
        # Typical high-scoring international matches have lambda_total ~3.0-3.5
        # Individual team lambdas rarely exceed 2.5-3.0 except in extreme mismatches
        self.max_lambda: float = self.dc_config.get('max_lambda', 4.0)
        self.max_goals: int = self.config.get('matrix', {}).get('max_goals', 8)
        
        # Sanity check thresholds for lambda diagnostics (configurable)
        # These trigger warnings but don't clip values
        self.lambda_warning_threshold: float = self.dc_config.get('lambda_warning_threshold', 3.0)
        self.lambda_total_warning_threshold: float = self.dc_config.get('lambda_total_warning_threshold', 5.0)
        
        # Markov features configuration
        self.use_markov_features: bool = self.dc_config.get('use_markov_features', False)
        self.markov_weight: float = self.dc_config.get('markov_weight', 0.18)  # Default 18%
        self.markov_weight_schedule: Optional[Dict[str, float]] = self.dc_config.get(
            'markov_weight_schedule', None
        )  # Optional piecewise schedule by minute phase
        self.markov_event_probs = None
        self.markov_baselines = None
        
        # Match state for in-play prediction (set via set_match_state())
        self.match_state: Optional[Dict[str, Any]] = None

        # Trained parameters (set after fit())
        self.teams_: Optional[list] = None
        self.alpha_: Optional[Dict[str, float]] = None   # attack params
        self.beta_: Optional[Dict[str, float]] = None    # defense params
        self.gamma_: Optional[float] = None              # home advantage
        self.is_fitted_: bool = False

    def fit(
        self,
        df,  # pd.DataFrame with home_team, away_team, home_score, away_score, date
        teams: Optional[list] = None,
    ) -> 'DixonColesModel':
        """
        Fit Dixon-Coles model via MLE on historical match data.

        Args:
            df: Historical matches DataFrame.
            teams: Optional list of team names. If None, inferred from data.
        """
        import pandas as pd

        df = df.dropna(subset=['home_team', 'away_team', 'home_score', 'away_score'])
        df['home_score'] = df['home_score'].astype(int)
        df['away_score'] = df['away_score'].astype(int)

        if teams is None:
            teams = sorted(set(df['home_team'].tolist() + df['away_team'].tolist()))

        self.teams_ = teams
        n_teams = len(teams)

        logger.info(f"Fitting Dixon-Coles on {len(df)} matches, {n_teams} teams")

        # Compute temporal weights
        weights = temporal_weights(df['date'], xi=self.xi) if 'date' in df.columns else None

        # Initial parameters: all zeros
        x0 = np.zeros(2 * n_teams + 1)
        x0[2 * n_teams] = np.log(np.exp(self.home_advantage))  # home advantage prior

        result = minimize(
            dc_log_likelihood,
            x0,
            args=(
                teams,
                df['home_team'].values,
                df['away_team'].values,
                df['home_score'].values,
                df['away_score'].values,
                weights,
                self.rho,
            ),
            method='L-BFGS-B',
            options={'maxiter': 500, 'ftol': 1e-8},
        )

        if not result.success:
            logger.warning(f"Optimization did not fully converge: {result.message}")

        params = result.x
        log_alpha = params[:n_teams] - params[:n_teams].mean()
        log_beta = params[n_teams:2 * n_teams]
        gamma = np.exp(params[2 * n_teams])

        self.alpha_ = {t: float(np.exp(log_alpha[i])) for i, t in enumerate(teams)}
        self.beta_ = {t: float(np.exp(log_beta[i])) for i, t in enumerate(teams)}
        self.gamma_ = float(gamma)
        self.is_fitted_ = True

        logger.info(f"DC fit complete. Home advantage gamma={self.gamma_:.4f}")
        return self

    def predict_lambdas(
        self,
        home_features: Dict[str, Any],
        away_features: Dict[str, Any],
        match_state: Optional[Dict[str, Any]] = None,
    ) -> Tuple[float, float]:
        """
        Estimate lambda_home and lambda_away for a match.

        If model is fitted and teams are known, uses trained DC parameters.
        Otherwise falls back to heuristic estimation from features.
        
        If Markov features are enabled (use_markov_features=True) and match_state
        is provided, incorporates Markov-based intensity adjustments.

        Args:
            home_features: Feature dict for home team (from feature_pipeline).
            away_features: Feature dict for away team (from feature_pipeline).
            match_state: Optional match state dict for in-play prediction.
                Can also be set via set_match_state() method.
                Expected keys: minute, score_diff, home_red_cards, away_red_cards.

        Returns:
            Tuple (lambda_home, lambda_away).
        """
        # Use instance match_state if not provided
        if match_state is None:
            match_state = self.match_state
        
        home_name = home_features.get('nombre', '')
        away_name = away_features.get('nombre', '')

        if (self.is_fitted_ and
                home_name in (self.alpha_ or {}) and
                away_name in (self.alpha_ or {})):
            return self._predict_lambdas_trained(home_name, away_name, home_features, away_features, match_state)
        else:
            return self._predict_lambdas_heuristic(home_features, away_features, match_state)
    
    def set_match_state(
        self,
        minute: int,
        score_diff: int,
        home_red_cards: int = 0,
        away_red_cards: int = 0,
        phase: str = "regular_time",
    ) -> None:
        """
        Set the current match state for in-play prediction.
        
        This state is used by predict_lambdas() when Markov features are enabled.
        
        Args:
            minute: Current match minute (0-90+).
            score_diff: Home score minus away score.
            home_red_cards: Home team red cards.
            away_red_cards: Away team red cards.
            phase: Match phase ("regular_time", etc.).
        """
        self.match_state = {
            'minute': minute,
            'score_diff': score_diff,
            'home_red_cards': home_red_cards,
            'away_red_cards': away_red_cards,
            'phase': phase
        }
        logger.debug(f"Match state set: minute={minute}, score_diff={score_diff}")
    
    def load_markov_tables(
        self,
        event_probs_path: str,
        baselines_path: str,
    ) -> None:
        """
        Load Markov probability tables for Markov-aware mode.
        
        Call this before using Markov features to enable state-based adjustments.
        
        Args:
            event_probs_path: Path to state_event_probabilities.csv
            baselines_path: Path to baseline_probabilities.csv
        """
        if not _MARKOV_AVAILABLE:
            logger.warning("Markov features module not available; cannot load tables")
            return
        
        from ..features.markov_features import load_markov_tables as load_tables
        
        self.markov_event_probs, self.markov_baselines = load_tables(
            event_probs_path, baselines_path
        )
        self.use_markov_features = True
        logger.info(f"Markov tables loaded; Markov-aware mode enabled")

    def _predict_lambdas_trained(
        self,
        home_name: str,
        away_name: str,
        home_features: Dict[str, Any],
        away_features: Dict[str, Any],
        match_state: Optional[Dict[str, Any]] = None,
    ) -> Tuple[float, float]:
        """Use trained DC parameters to estimate lambdas."""
        alpha_h = self.alpha_[home_name]
        beta_a = self.beta_[away_name]
        alpha_a = self.alpha_[away_name]
        beta_h = self.beta_[home_name]
        gamma = self.gamma_

        lambda_h = alpha_h * beta_a * gamma
        lambda_a = alpha_a * beta_h

        # Apply feature-based context modifier on top of DC params
        ctx_h = home_features.get('context_modifier', 0.0)
        ctx_a = away_features.get('context_modifier', 0.0)

        lambda_h *= np.exp(ctx_h)
        lambda_a *= np.exp(ctx_a)
        
        # Apply Markov features adjustment if enabled and state is provided
        if self.use_markov_features and match_state is not None and _MARKOV_AVAILABLE:
            lambda_h, lambda_a = self._apply_markov_adjustment(
                lambda_h, lambda_a, match_state
            )

        lambda_h = float(np.clip(lambda_h, self.min_lambda, self.max_lambda))
        lambda_a = float(np.clip(lambda_a, self.min_lambda, self.max_lambda))

        logger.debug(
            f"DC trained: lambda_h={lambda_h:.4f}, lambda_a={lambda_a:.4f} "
            f"(alpha_h={alpha_h:.4f}, beta_a={beta_a:.4f}, gamma={gamma:.4f})"
        )
        return lambda_h, lambda_a

    def _predict_lambdas_heuristic(
        self,
        home_features: Dict[str, Any],
        away_features: Dict[str, Any],
        match_state: Optional[Dict[str, Any]] = None,
    ) -> Tuple[float, float]:
        """
        Heuristic lambda estimation from feature dicts.

        Formula:
            lambda_home = attack_home * (1/defense_away) * form_home *
                          ranking_home * h2h_home * squad_home *
                          exp(home_adv + ctx_home)

            lambda_away = attack_away * (1/defense_home) * form_away *
                          ranking_away * h2h_away * squad_away *
                          exp(ctx_away)

        SEMANTICS:
            - attack_rating: >1.0 means stronger than average attack
            - defense_rating: >1.0 means STRONGER defense (concedes LESS)
            
        Therefore, opponent's defense_rating is INVERTED:
            - High defense_rating → LOW opponent lambda (hard to score against)
            - Low defense_rating → HIGH opponent lambda (easy to score against)

        This ensures the model can run without any training data.
        """
        from ..features.ratings import LEAGUE_AVG_GOALS

        # Home team lambda
        attack_h = home_features.get('attack_rating', 1.0)
        defense_a = away_features.get('defense_rating', 1.0)
        form_h = home_features.get('form_factor', 1.0)
        ranking_h = home_features.get('ranking_factor', 1.0)
        h2h_h = home_features.get('h2h_factor', 1.0)
        squad_h = home_features.get('squad_multiplier', 1.0)
        home_adv = home_features.get('home_advantage_log', 0.0)
        ctx_h = home_features.get('context_modifier', 0.0)

        # INVERT opponent's defense: high defense_rating = hard to score against
        # So opponent_defense_factor = 1.0 / defense_rating
        # Add small epsilon to avoid division by zero
        defense_a_inverse = 1.0 / max(defense_a, 0.1)

        lambda_h = (
            attack_h * defense_a_inverse * LEAGUE_AVG_GOALS *
            form_h * ranking_h * h2h_h * squad_h *
            np.exp(home_adv + ctx_h)
        )

        # Away team lambda
        attack_a = away_features.get('attack_rating', 1.0)
        defense_h = home_features.get('defense_rating', 1.0)
        form_a = away_features.get('form_factor', 1.0)
        ranking_a = away_features.get('ranking_factor', 1.0)
        h2h_a = away_features.get('h2h_factor', 1.0)
        squad_a = away_features.get('squad_multiplier', 1.0)
        ctx_a = away_features.get('context_modifier', 0.0)

        # INVERT opponent's defense
        defense_h_inverse = 1.0 / max(defense_h, 0.1)

        lambda_a = (
            attack_a * defense_h_inverse * LEAGUE_AVG_GOALS *
            form_a * ranking_a * h2h_a * squad_a *
            np.exp(ctx_a)
        )
        
        # Apply Markov features adjustment if enabled and state is provided
        if self.use_markov_features and match_state is not None and _MARKOV_AVAILABLE:
            lambda_h, lambda_a = self._apply_markov_adjustment(
                lambda_h, lambda_a, match_state
            )

        # Sanity checks for lambda values (log warnings but don't clip)
        lambda_total = lambda_h + lambda_a
        if lambda_h > self.lambda_warning_threshold:
            logger.warning(
                f"High lambda_home detected: {lambda_h:.4f} > {self.lambda_warning_threshold}. "
                f"Match: {home_features.get('nombre', 'unknown')} vs {away_features.get('nombre', 'unknown')}"
            )
        if lambda_a > self.lambda_warning_threshold:
            logger.warning(
                f"High lambda_away detected: {lambda_a:.4f} > {self.lambda_warning_threshold}. "
                f"Match: {home_features.get('nombre', 'unknown')} vs {away_features.get('nombre', 'unknown')}"
            )
        if lambda_total > self.lambda_total_warning_threshold:
            logger.warning(
                f"High lambda_total detected: {lambda_total:.4f} > {self.lambda_total_warning_threshold}. "
                f"Match: {home_features.get('nombre', 'unknown')} vs {away_features.get('nombre', 'unknown')}"
            )

        lambda_h = float(np.clip(lambda_h, self.min_lambda, self.max_lambda))
        lambda_a = float(np.clip(lambda_a, self.min_lambda, self.max_lambda))

        logger.info(
            f"DC heuristic: lambda_h={lambda_h:.4f}, lambda_a={lambda_a:.4f}, "
            f"lambda_total={lambda_h + lambda_a:.4f} "
            f"(attack_h={attack_h:.3f}, def_a={defense_a:.3f}, attack_a={attack_a:.3f}, def_h={defense_h:.3f})"
        )
        return lambda_h, lambda_a
    
    def _get_markov_weight(self, match_state: Dict[str, Any]) -> float:
        """
        Get the Markov adjustment weight for the current match state.
        
        Supports two modes:
        1. Constant weight: uses self.markov_weight (default 0.18)
        2. Piecewise schedule: uses self.markov_weight_schedule based on minute phase
        
        Args:
            match_state: Current match state dict with 'minute' key.
        
        Returns:
            Float weight in [0, 1] range.
        """
        # If a piecewise schedule is defined, use it
        if self.markov_weight_schedule:
            minute = match_state.get('minute', 0)
            
            # Default piecewise schedule (can be overridden via config)
            early_weight = self.markov_weight_schedule.get('early', 0.20)   # 0-30 min
            mid_weight = self.markov_weight_schedule.get('mid', 0.15)      # 31-75 min
            late_weight = self.markov_weight_schedule.get('late', 0.10)    # 76+ min
            
            if minute <= 30:
                return early_weight
            elif minute <= 75:
                return mid_weight
            else:
                return late_weight
        
        # Otherwise use constant weight
        return self.markov_weight
    
    def _apply_markov_adjustment(
        self,
        lambda_h: float,
        lambda_a: float,
        match_state: Dict[str, Any],
    ) -> Tuple[float, float]:
        """
        Apply Markov-based intensity adjustment to lambdas.
        
        Uses the current match state to adjust goal expectations based on
        observed patterns in similar states (minute, score_diff, cards).
        
        Adjustment formula:
            lambda_adj = lambda_base * (1 + weight * (ratio - 1))
        
        where weight is configurable (default 0.18) and can vary by minute phase.
        
        Args:
            lambda_h: Base home lambda.
            lambda_a: Base away lambda.
            match_state: Current match state dict.
        
        Returns:
            Tuple (adjusted_lambda_h, adjusted_lambda_a)
        """
        if self.markov_event_probs is None or self.markov_baselines is None:
            return lambda_h, lambda_a
        
        try:
            from ..features.markov_features import (
                build_state_from_match_context,
                build_state_for_away_team,
                get_markov_features,
            )
            
            # Build state for home team perspective
            home_state = build_state_from_match_context(
                minute=match_state.get('minute', 0),
                score_diff=match_state.get('score_diff', 0),
                home_red_cards=match_state.get('home_red_cards', 0),
                away_red_cards=match_state.get('away_red_cards', 0),
                phase=match_state.get('phase', 'regular_time'),
            )
            
            # Get Markov features for home team
            home_markov = get_markov_features(
                home_state, 
                self.markov_event_probs, 
                self.markov_baselines
            )
            
            # Build state for away team perspective (inverted score_diff)
            away_state = build_state_for_away_team(home_state)
            
            # Get Markov features for away team
            away_markov = get_markov_features(
                away_state,
                self.markov_event_probs,
                self.markov_baselines
            )
            
            # Compute adjustment factors based on goal probability ratios
            # If markov_p_goal > baseline, increase lambda; if lower, decrease
            baseline_p_goal = self.markov_baselines['global'].get('p_goal', 0.17)
            
            home_ratio = home_markov['markov_p_goal_next_window'] / baseline_p_goal
            away_ratio = away_markov['markov_p_goal_next_window'] / baseline_p_goal
            
            # Get weight for current match state (supports constant or piecewise schedule)
            weight = self._get_markov_weight(match_state)
            
            # Apply soft adjustment (dampen extreme ratios)
            home_adjustment = 1.0 + weight * (home_ratio - 1.0)
            away_adjustment = 1.0 + weight * (away_ratio - 1.0)
            
            lambda_h_adj = lambda_h * home_adjustment
            lambda_a_adj = lambda_a * away_adjustment
            
            logger.debug(
                f"Markov adjustment (weight={weight:.2f}): home_ratio={home_ratio:.3f}, away_ratio={away_ratio:.3f} -> "
                f"lambda_h: {lambda_h:.4f} -> {lambda_h_adj:.4f}, "
                f"lambda_a: {lambda_a:.4f} -> {lambda_a_adj:.4f}"
            )
            
            return lambda_h_adj, lambda_a_adj
            
        except Exception as e:
            logger.warning(f"Markov adjustment failed: {e}; using base lambdas")
            return lambda_h, lambda_a

    def score_matrix(
        self,
        lambda_home: float,
        lambda_away: float,
    ) -> np.ndarray:
        """
        Generate Dixon-Coles corrected score probability matrix.

        Args:
            lambda_home: Expected goals for home team.
            lambda_away: Expected goals for away team.

        Returns:
            2D numpy array (max_goals+1) x (max_goals+1).
        """
        return dc_score_matrix(
            lambda_home=lambda_home,
            lambda_away=lambda_away,
            rho=self.rho,
            max_goals=self.max_goals,
        )

    def save(self, path: str | Path) -> None:
        """Persist model to disk."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump({
                'teams': self.teams_,
                'alpha': self.alpha_,
                'beta': self.beta_,
                'gamma': self.gamma_,
                'is_fitted': self.is_fitted_,
                'rho': self.rho,
                'home_advantage': self.home_advantage,
                'xi': self.xi,
                'max_goals': self.max_goals,
            }, f)
        logger.info(f"Model saved to {path}")

    @classmethod
    def load(cls, path: str | Path, config: Optional[Dict[str, Any]] = None) -> 'DixonColesModel':
        """Load model from disk."""
        path = Path(path)
        with open(path, 'rb') as f:
            data = pickle.load(f)

        model = cls(config=config)
        model.teams_ = data['teams']
        model.alpha_ = data['alpha']
        model.beta_ = data['beta']
        model.gamma_ = data['gamma']
        model.is_fitted_ = data['is_fitted']
        model.rho = data.get('rho', -0.13)
        model.home_advantage = data.get('home_advantage', 0.25)
        model.xi = data.get('xi', 0.003)
        model.max_goals = data.get('max_goals', 8)
        logger.info(f"Model loaded from {path}")
        return model
